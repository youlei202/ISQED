# experiments/exp4_bert_audit_disco.py
#
# In-silico ecosystem audit on SST-2 with a DISCO-style dose-splitting design:
# - P_fit: low-dose masking (θ in [0.0, 0.3]) used to learn convex peer weights
# - P_eval: high-dose masking (θ in [0.4, 0.7]) used to evaluate PIER
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isqed.real_world import HuggingFaceWrapper, MaskingIntervention
from isqed.geometry import DISCOSolver



def run_bert_experiment():
    print("--- Running Exp 4 (DISCO-style BERT Ecosystem Audit) ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================================================================
    # 1. Define the Peer Ecosystem
    # =========================================================================
    peer_ids = [
        "textattack/bert-base-uncased-SST-2",
        "textattack/distilbert-base-uncased-SST-2",  # will also be used as a redundant target
        "textattack/albert-base-v2-SST-2",
        "textattack/xlnet-base-cased-SST-2",
    ]

    peers = []
    print("Loading Peers...")
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

    # =========================================================================
    # 2. Define Targets
    # =========================================================================
    targets = []

    # Case A: RoBERTa (as a divergent architecture)
    try:
        roberta = HuggingFaceWrapper("textattack/roberta-base-SST-2", device)
        targets.append({
            "model": roberta,
            "name": "Architectural Divergence (RoBERTa)",
            "type": "Low Redundancy"
        })
    except Exception as e:
        print(f"Warning: RoBERTa failed to load: {e}")

    # Case B: DistilBERT (redundant clone of peer)
    distil_ref = next((p for p in peers if "distilbert" in p.name), None)
    if distil_ref:
        targets.append({
            "model": distil_ref,
            "name": "Perfect Redundancy (Clone)",
            "type": "High Redundancy",
            "clone_of_peer_idx": peers.index(distil_ref)
        })
    else:
        print("Warning: No DistilBERT peer found for redundant target.")

    # Case C: DistilBERT variant (with different fine-tuning)
    try:
        near_distil = HuggingFaceWrapper(
            "distilbert-base-uncased-finetuned-sst-2-english",  # a different fine-tuned DistilBERT
            device
        )
        targets.append({
            "model": near_distil,
            "name": "Parametric Divergence (Variant)",
            "type": "Uniqueness"
            # No clone_of_peer_idx since not identical to peer
        })
    except Exception as e:
        print(f"Warning: near‑redundant DistilBERT failed to load: {e}")


    if not targets:
        print("No targets loaded. Abort.")
        return

    # =========================================================================
    # 3. Data and Dose Design (P_fit vs P_eval)
    # =========================================================================
    intervention = MaskingIntervention()

    print("Loading SST-2 validation data...")
    try:
        dataset = load_dataset("glue", "sst2", split="validation")
        all_sentences = dataset["sentence"]
        # Use a reasonably sized subset for the demo
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

    # Shuffle and split into fit vs eval texts
    rng = np.random.RandomState(0)
    all_sentences = np.array(all_sentences)
    rng.shuffle(all_sentences)
    n_total = len(all_sentences)
    n_fit = n_total // 2
    fit_texts = all_sentences[:n_fit].tolist()
    eval_texts = all_sentences[n_fit:].tolist()

    print(f"Total sentences: {n_total}, fit: {len(fit_texts)}, eval: {len(eval_texts)}")

    # Dose splitting:
    #   - low-dose (P_fit): used to learn convex peer baseline
    #   - high-dose (P_eval): used to evaluate PIER
    doses_fit = np.linspace(0.0, 0.3, 4)   # e.g. [0.0, 0.1, 0.2, 0.3]
    doses_eval = np.linspace(0.4, 0.88, 5)  # e.g. [0.4, 0.5, 0.6, 0.7]

    print(f"P_fit doses (low):  {doses_fit}")
    print(f"P_eval doses (high): {doses_eval}")

    # =========================================================================
    # 4. Main Loop: DISCO-style audit per target
    # =========================================================================
    all_results = []

    for t_info in targets:
        t_name = t_info["name"]
        t_model = t_info["model"]
        print(f"\n>>> Auditing target: {t_name}")

        # -------------------------------------------------
        # 4.1 Fit phase: learn a global convex baseline on P_fit
        # -------------------------------------------------
        y_t_fit_list = []
        Y_p_fit_list = []

        print("  [Phase] Fitting convex baseline on low-dose interventions (P_fit)...")
        for text in tqdm(fit_texts, desc="    Fit texts", leave=False):
            for theta in doses_fit:
                # Deterministic intervention across models
                seed = abs(hash((text, float(theta)))) % (2**32)
                perturbed_text = intervention.apply(text, theta, seed=seed)

                # Query target
                y_t = t_model._forward(perturbed_text)

                # Query peers
                y_ps = [p._forward(perturbed_text) for p in peers]

                # Optional sanity check for clone target
                if "clone_of_peer_idx" in t_info:
                    p_idx = t_info["clone_of_peer_idx"]
                    diff = float(abs(y_t - y_ps[p_idx]))
                    if diff > 1e-9:
                        print(f"    [ALARM] Clone mismatch on P_fit! Diff: {diff:.3e}")

                y_t_fit_list.append(y_t)
                Y_p_fit_list.append(y_ps)

        y_t_fit_vec = np.array(y_t_fit_list, dtype=float)  # (m_fit,)
        Y_p_fit_mat = np.array(Y_p_fit_list, dtype=float)  # (m_fit, N_peers)
        if y_t_fit_vec.ndim == 1:
            y_t_fit_vec = y_t_fit_vec.reshape(-1, 1)       # (m_fit, 1) if solver expects column

        # Solve for a single convex weight vector w_hat on P_fit
        try:
            dist_fit, w_hat = DISCOSolver.solve_weights_and_distance(
                y_t_fit_vec, Y_p_fit_mat
            )
        except Exception as e:
            print(f"  [ERROR] DISCOSolver failed during fit phase for {t_name}: {e}")
            continue

        w_hat = np.array(w_hat, dtype=float).flatten()
        if w_hat.shape[0] != len(peers):
            print(f"  [WARN] w_hat length {w_hat.shape[0]} != num_peers {len(peers)}")

        print(f"  [Fit] Learned convex weights for {t_name}: {w_hat}")

        # -------------------------------------------------
        # 4.2 Evaluation phase: compute PIER on P_eval using fixed w_hat
        # -------------------------------------------------
        print("  [Phase] Evaluating PIER on high-dose interventions (P_eval)...")
        for theta in tqdm(doses_eval, desc="    Eval doses", leave=False):
            dose_piers = []

            for text in eval_texts:
                seed = abs(hash((text, float(theta)))) % (2**32)
                perturbed_text = intervention.apply(text, theta, seed=seed)

                # Target output
                y_t = t_model._forward(perturbed_text)

                # Peer outputs
                y_ps = np.array([p._forward(perturbed_text) for p in peers], dtype=float)

                # Optional sanity check for clone target on P_eval
                if "clone_of_peer_idx" in t_info:
                    p_idx = t_info["clone_of_peer_idx"]
                    diff = float(abs(y_t - y_ps[p_idx]))
                    if diff > 1e-9:
                        print(f"    [ALARM] Clone mismatch on P_eval! Diff: {diff:.3e}")

                # Convex baseline using fixed w_hat
                y_mix = float(np.dot(w_hat, y_ps))
                pier = float(abs(y_t - y_mix))
                dose_piers.append(pier)

            if dose_piers:
                avg_pier = float(np.mean(dose_piers))
            else:
                avg_pier = float("nan")

            all_results.append(
                {
                    "Dose": float(theta),
                    "PIER": avg_pier,
                    "Model": t_name,
                    "Group": t_info["type"],
                }
            )

    # =========================================================================
    # 5. Save results
    # =========================================================================
    df = pd.DataFrame(all_results)
    output_dir = "results/tables"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "exp4_bert_disco_dosesplit.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved DISCO-style BERT audit results to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    run_bert_experiment()