# experiments/exp5_cross_audit.py
#
# Cross-audit of the full BERT ecosystem with a dual-mode DISCO design:
#   - Mode 1 (Uniqueness): target vs. "rest of ecosystem" peers (size N-1)
#   - Mode 2 (Redundancy): target vs. "full ecosystem" peers (size N)
#
# For each target model, we:
#   1) Learn w_rest on P_fit (low doses) using peers_rest
#   2) Learn w_full on P_fit (low doses) using peers_full
#   3) On P_eval (high doses), compute residuals under both baselines:
#        R_rest(x, θ)  = |Y_t - sum_j w_rest[j] * Y_j|
#        R_full(x, θ)  = |Y_t - sum_j w_full[j] * Y_j|
#   4) Save full residual vectors (no aggregation) for downstream heatmaps.

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


def run_cross_audit_dual_mode():
    print("--- Running Exp 5: Full Ecosystem Dual-Mode Audit (DISCO Standard, via Ecosystem) ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =====================================================================
    # 1. Define ecosystem models
    # =====================================================================
    model_ids = [
        "textattack/bert-base-uncased-SST-2",
        "textattack/distilbert-base-uncased-SST-2",
        "textattack/roberta-base-SST-2",
        "textattack/albert-base-v2-SST-2",
        "textattack/xlnet-base-cased-SST-2",
    ]
    short_names = ["BERT", "DistilBERT", "RoBERTa", "ALBERT", "XLNet"]

    models = []
    print("Loading core ecosystem models...")
    for pid in model_ids:
        try:
            m = HuggingFaceWrapper(pid, device)
            models.append(m)
            print(f"  [OK] Loaded: {pid}")
        except Exception as e:
            print(f"  [FAIL] Skipping {pid}: {e}")

    n = len(models)
    if n < 3:
        print("Less than 3 models loaded. Aborting experiment.")
        return

    short_names = short_names[:n]

    # =====================================================================
    # 2. Data and dose design (P_fit vs P_eval)
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
        ] * 40

    rng = np.random.RandomState(0)
    all_sentences = np.array(all_sentences)
    rng.shuffle(all_sentences)
    n_total = len(all_sentences)
    n_fit = n_total // 2
    fit_texts = all_sentences[:n_fit].tolist()
    eval_texts = all_sentences[n_fit:].tolist()

    print(f"Total sentences: {n_total}, fit: {len(fit_texts)}, eval: {len(eval_texts)}")

    # Dose design: low doses for P_fit, high doses for P_eval
    doses_fit = np.linspace(0.0, 0.3, 4)    # not overlap with eval
    doses_eval = np.linspace(0.4, 0.9, 6)   

    print(f"P_fit doses (low):  {doses_fit}")
    print(f"P_eval doses (high): {doses_eval}")

    # =====================================================================
    # 3. Prepare index for the final residual DataFrame
    # =====================================================================
    # We will store one row per (eval_text, dose_eval) pair.
    n_eval_points = len(eval_texts) * len(doses_eval)
    eval_index = []
    for i in range(len(eval_texts)):
        for j, theta in enumerate(doses_eval):
            eval_index.append(f"Sample {i+1} $\\theta={theta:.2f}$")

    final_residual_data = pd.DataFrame()
    final_residual_data["Eval_Point"] = eval_index

    # =====================================================================
    # 4. Main loop: dual-mode DISCO per target
    # =====================================================================
    for i in range(n):
        target_model = models[i]
        target_name = short_names[i]

        print(f"\n>>> Auditing target: {target_name}")

        # Peer set 1: rest of ecosystem (size N-1)
        peers_rest = [models[j] for j in range(n) if j != i]
        # Peer set 2: full ecosystem (size N)
        peers_full = models

        # Ecosystem views for the two modes
        eco_rest = Ecosystem(target=target_model, peers=peers_rest)
        eco_full = Ecosystem(target=target_model, peers=peers_full)

        # -------------------------------------------------
        # 4.1 FIT phase: learn w_hat_rest and w_hat_full on P_fit
        # -------------------------------------------------
        print("  [Phase] Fitting convex baselines on low-dose interventions (P_fit)...")

        fit_X = []
        fit_Theta = []
        fit_seeds = []

        # Build a shared batch of (text, theta, seed) for both ecosystems
        for text in fit_texts:
            for theta in doses_fit:
                seed = abs(hash((text, float(theta)))) % (2**32)
                fit_X.append(text)
                fit_Theta.append(float(theta))
                fit_seeds.append(int(seed))

        if not fit_X:
            print("  [WARN] No fit samples for this target. Skipping.")
            continue

        # Query target + peers_rest
        y_t_fit_rest, Y_p_fit_rest = eco_rest.batched_query(
            X=fit_X,
            Thetas=fit_Theta,
            intervention=intervention,
            seeds=fit_seeds,
        )
        # Query target + peers_full
        y_t_fit_full, Y_p_fit_full = eco_full.batched_query(
            X=fit_X,
            Thetas=fit_Theta,
            intervention=intervention,
            seeds=fit_seeds,
        )

        # Sanity check: target outputs must coincide across the two ecosystems
        max_target_diff = float(np.max(np.abs(y_t_fit_rest - y_t_fit_full)))
        if max_target_diff > 1e-9:
            print(
                f"  [WARN] Target outputs differ between rest/full ecosystems "
                f"on P_fit for {target_name}: max diff={max_target_diff:.3e}"
            )
        # Use the rest version as the canonical target outputs
        y_t_fit = y_t_fit_rest

        # Prepare inputs for DISCOSolver
        y_t_fit_vec = y_t_fit.reshape(-1, 1)

        # Solve w_hat_rest (Mode 1: Uniqueness)
        Y_p_fit_rest_mat = Y_p_fit_rest
        _, w_hat_rest = DISCOSolver.solve_weights_and_distance(
            y_t_fit_vec,
            Y_p_fit_rest_mat,
        )
        w_hat_rest = np.asarray(w_hat_rest, dtype=float).flatten()

        # Solve w_hat_full (Mode 2: Redundancy)
        Y_p_fit_full_mat = Y_p_fit_full
        _, w_hat_full = DISCOSolver.solve_weights_and_distance(
            y_t_fit_vec,
            Y_p_fit_full_mat,
        )
        w_hat_full = np.asarray(w_hat_full, dtype=float).flatten()

        print(f"  [Fit] Learned w_rest for {target_name}: {w_hat_rest[:2]}...")
        print(f"  [Fit] Learned w_full for {target_name}: {w_hat_full[:2]}...")

        # -------------------------------------------------
        # 4.2 EVAL phase: compute residuals on P_eval
        # -------------------------------------------------
        print("  [Phase] Evaluating residuals on high-dose interventions (P_eval)...")

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

        # Query using the two ecosystems on the same perturbed inputs
        y_t_eval_rest, Y_p_eval_rest = eco_rest.batched_query(
            X=eval_X,
            Thetas=eval_Theta,
            intervention=intervention,
            seeds=eval_seeds,
        )
        y_t_eval_full, Y_p_eval_full = eco_full.batched_query(
            X=eval_X,
            Thetas=eval_Theta,
            intervention=intervention,
            seeds=eval_seeds,
        )

        # Sanity check: target outputs must coincide across the two ecosystems
        max_target_diff_eval = float(np.max(np.abs(y_t_eval_rest - y_t_eval_full)))
        if max_target_diff_eval > 1e-9:
            print(
                f"  [WARN] Target outputs differ between rest/full ecosystems "
                f"on P_eval for {target_name}: max diff={max_target_diff_eval:.3e}"
            )
        y_t_eval = y_t_eval_rest

        # Mode 1: Uniqueness residuals (rest peers only)
        y_mix_rest = Y_p_eval_rest @ w_hat_rest
        r_uniqueness = np.abs(y_t_eval - y_mix_rest)

        # Mode 2: Redundancy residuals (full peers including target)
        y_mix_full = Y_p_eval_full @ w_hat_full
        r_redundancy = np.abs(y_t_eval - y_mix_full)

        # Sanity check on lengths vs pre-built index
        if r_uniqueness.size != n_eval_points or r_redundancy.size != n_eval_points:
            print(
                f"  [WARN] Residual length mismatch for {target_name}: "
                f"got {r_uniqueness.size} vs expected {n_eval_points}"
            )

        final_residual_data[f"{target_name}_Uniqueness_R"] = r_uniqueness
        final_residual_data[f"{target_name}_Redundancy_R"] = r_redundancy

    # =====================================================================
    # 5. Save results
    # =====================================================================
    output_dir = "results/tables"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "exp5_dual_mode_residuals.csv")
    final_residual_data.to_csv(out_path, index=False)
    print(f"\nSaved raw residuals data for {n_eval_points} eval points to: {out_path}")


if __name__ == "__main__":
    run_cross_audit_dual_mode()
