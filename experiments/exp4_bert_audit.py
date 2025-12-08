# experiments/exp4_bert_audit_disco.py
#
# In-silico ecosystem audit on SST-2 with a DISCO-style dose-splitting design:
# - P_fit: low-dose masking (θ in [0.0, 0.3]) used to learn convex peer weights
# - P_eval: high-dose masking (θ in [0.4, 0.88]) used to evaluate PIER
#
# For each target model, we:
#   1) Collect a batch of (text, low-dose) responses from target and peers
#   2) Solve a single convex projection problem to obtain a global weight vector w_hat
#   3) Use this fixed w_hat to compute PIER on an independent batch of (text, high-dose)

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import torch

# Ensure we can import the local `isqed` package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isqed.real_world import HuggingFaceWrapper, MaskingIntervention
from isqed.geometry import DISCOSolver
from isqed.ecosystem import Ecosystem


def run_bert_experiment():
    print("--- Running Exp 4 (DISCO-style BERT Ecosystem Audit, via Ecosystem) ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =====================================================================
    # 1. Define the peer ecosystem
    # =====================================================================
    peer_ids = [
        "textattack/bert-base-uncased-SST-2",
        "textattack/distilbert-base-uncased-SST-2",  # will also be used as a redundant target
        "textattack/albert-base-v2-SST-2",
        "textattack/xlnet-base-cased-SST-2",
    ]

    peers = []
    print("Loading peers...")
    for pid in peer_ids:
        try:
            model = HuggingFaceWrapper(pid, device)
            peers.append(model)
            print(f"  [OK] Loaded peer: {pid}")
        except Exception as e:
            print(f"  [SKIP] Failed to load {pid}: {e}")

    if not peers:
        print("No peers loaded. Abort.")
        return

    # =====================================================================
    # 2. Define targets
    # =====================================================================
    targets = []

    # Case A: RoBERTa (architectural divergence)
    try:
        roberta = HuggingFaceWrapper("textattack/roberta-base-SST-2", device)
        targets.append(
            {
                "model": roberta,
                "name": "Architectural Divergence (RoBERTa)",
                "type": "Low Redundancy",
            }
        )
    except Exception as e:
        print(f"Warning: RoBERTa failed to load: {e}")

    # Case B: DistilBERT clone (exact peer)
    distil_ref = next((p for p in peers if "distilbert" in p.name.lower()), None)
    if distil_ref:
        targets.append(
            {
                "model": distil_ref,
                "name": "Perfect Redundancy (Clone)",
                "type": "High Redundancy",
                "clone_of_peer_idx": peers.index(distil_ref),
            }
        )
    else:
        print("Warning: No DistilBERT peer found for redundant target.")

    # Case C: DistilBERT variant (different fine-tuning)
    try:
        near_distil = HuggingFaceWrapper(
            "distilbert-base-uncased-finetuned-sst-2-english", device
        )
        targets.append(
            {
                "model": near_distil,
                "name": "Parametric Divergence (Variant)",
                "type": "Uniqueness",
            }
        )
    except Exception as e:
        print(f"Warning: near-redundant DistilBERT failed to load: {e}")

    if not targets:
        print("No targets loaded. Abort.")
        return

    # =====================================================================
    # 3. Data and dose design (P_fit vs P_eval)
    # =====================================================================
    intervention = MaskingIntervention()

    print("Loading SST-2 validation data...")
    try:
        dataset = load_dataset("glue", "sst2", split="validation")
        all_sentences = dataset["sentence"]
        max_samples = 200
        all_sentences = all_sentences[:max_samples]
    except Exception as e:
        print(f"  [WARN] Failed to load SST-2 from HF: {e}")
        all_sentences = [
            "This movie is great.",
            "Terrible acting.",
            "I loved it.",
            "The plot was boring.",
            "Amazing direction and visuals.",
        ] * 40  # 200 sentences

    rng = np.random.RandomState(0)
    all_sentences = np.array(all_sentences)
    rng.shuffle(all_sentences)
    n_total = len(all_sentences)
    n_fit = n_total // 2
    fit_texts = all_sentences[:n_fit].tolist()
    eval_texts = all_sentences[n_fit:].tolist()

    print(f"Total sentences: {n_total}, fit: {len(fit_texts)}, eval: {len(eval_texts)}")

    doses_fit = np.linspace(0.0, 0.4, 10)    # not overlap with eval
    doses_eval = np.linspace(0.4, 0.9, 10)   

    print(f"P_fit doses (low):  {doses_fit}")
    print(f"P_eval doses (high): {doses_eval}")

    # =====================================================================
    # 4. Main loop: DISCO-style audit per target (via Ecosystem)
    # =====================================================================
    all_results = []

    for t_info in targets:
        t_name = t_info["name"]
        t_model = t_info["model"]
        print(f"\n>>> Auditing target: {t_name}")

        # Build an Ecosystem object where the target is audited against all peers
        eco = Ecosystem(target=t_model, peers=peers)

        # -------------------------------------------------
        # 4.1 Fit phase: learn a global convex baseline on P_fit
        # -------------------------------------------------
        print("  [Phase] Fitting convex baseline on low-dose interventions (P_fit)...")

        fit_X = []
        fit_Theta = []
        fit_seeds = []

        for text in fit_texts:
            for theta in doses_fit:
                seed = abs(hash((text, float(theta)))) % (2**32)
                fit_X.append(text)
                fit_Theta.append(float(theta))
                fit_seeds.append(int(seed))

        if not fit_X:
            print("  [WARN] No fit samples for this target. Skipping.")
            continue

        # Query target and peers jointly via Ecosystem
        y_t_fit, Y_p_fit = eco.batched_query(
            X=fit_X,
            Thetas=fit_Theta,
            intervention=intervention,
            seeds=fit_seeds,
        )

        # Optional sanity check for clone target on P_fit
        if "clone_of_peer_idx" in t_info:
            clone_idx = t_info["clone_of_peer_idx"]
            diffs = np.abs(y_t_fit - Y_p_fit[:, clone_idx])
            max_diff = float(np.max(diffs))
            if max_diff > 1e-9:
                print(
                    f"  [ALARM] Clone mismatch on P_fit for {t_name}! "
                    f"Max diff: {max_diff:.3e}"
                )

        # Prepare data for DISCOSolver
        y_t_fit_vec = y_t_fit.reshape(-1, 1)  # (m_fit, 1)
        Y_p_fit_mat = Y_p_fit                 # (m_fit, N_peers)

        try:
            dist_fit, w_hat = DISCOSolver.solve_weights_and_distance(
                y_t_fit_vec,
                Y_p_fit_mat,
            )
        except Exception as e:
            print(f"  [ERROR] DISCOSolver failed during fit phase for {t_name}: {e}")
            continue

        w_hat = np.asarray(w_hat, dtype=float).flatten()
        if w_hat.shape[0] != len(peers):
            print(f"  [WARN] w_hat length {w_hat.shape[0]} != num_peers {len(peers)}")

        print(f"  [Fit] Learned convex weights for {t_name}: {w_hat}")

        # -------------------------------------------------
        # 4.2 Evaluation phase: compute PIER on P_eval using fixed w_hat
        # -------------------------------------------------
        print("  [Phase] Evaluating PIER on high-dose interventions (P_eval)...")

        eval_X = []
        eval_Theta = []
        eval_seeds = []

        for text in eval_texts:
            for theta in doses_eval:
                seed = abs(hash((text, float(theta)))) % (2**32)
                eval_X.append(text)
                eval_Theta.append(float(theta))
                eval_seeds.append(int(seed))

        if not eval_X:
            print("  [WARN] No eval samples for this target. Skipping P_eval.")
            continue

        y_t_eval, Y_p_eval = eco.batched_query(
            X=eval_X,
            Thetas=eval_Theta,
            intervention=intervention,
            seeds=eval_seeds,
        )

        eval_Theta_arr = np.asarray(eval_Theta, dtype=float)
        y_mix_eval = Y_p_eval @ w_hat
        residuals_all = np.abs(y_t_eval - y_mix_eval)

        # Optional sanity check for clone target on P_eval
        if "clone_of_peer_idx" in t_info:
            clone_idx = t_info["clone_of_peer_idx"]
            clone_diffs = np.abs(y_t_eval - Y_p_eval[:, clone_idx])
            max_diff_eval = float(np.max(clone_diffs))
            if max_diff_eval > 1e-9:
                print(
                    f"  [ALARM] Clone mismatch on P_eval for {t_name}! "
                    f"Max diff: {max_diff_eval:.3e}"
                )

        # Aggregate PIER per dose
        for theta in doses_eval:
            mask = np.isclose(eval_Theta_arr, float(theta), atol=1e-8)
            vals = residuals_all[mask]
            if vals.size == 0:
                avg_pier = float("nan")
            else:
                avg_pier = float(np.mean(vals))

            all_results.append(
                {
                    "Dose": float(theta),
                    "PIER": avg_pier,
                    "Model": t_name,
                    "Group": t_info["type"],
                }
            )

    # =====================================================================
    # 5. Save results
    # =====================================================================
    df = pd.DataFrame(all_results)
    output_dir = "results/tables"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "exp4_bert_disco_dosesplit.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved DISCO-style BERT audit results to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    run_bert_experiment()
