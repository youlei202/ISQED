# experiments/exp9_routing_sanity_moderate.py
#
# Exp 9 (revised): Mid-dose convex-routing sanity check.
#
# Goal:
#   Use a DISCO-style convex routing strategy that matches Exp 6:
#     DOSES_FIT  = np.linspace(0.0, 0.9, 10)
#     DOSES_EVAL = np.linspace(0.0, 0.9, 10)
#
#   For each target model t:
#     1) Split sentences into P_fit and P_eval.
#     2) On P_fit and all DOSES_FIT, learn a global convex weight vector
#        w_hat(t) using DISCOSolver (same strategy as Exp 6).
#     3) On P_eval and all DOSES_EVAL, compute for each dose:
#          - GlobalDisagreement_D (raw MD baseline)
#          - MAE_SinglePeer      (best fixed single peer)
#          - MAE_OracleRouter    (per-sample best peer)
#          - MAE_ConvexRouter    (fixed convex router with w_hat)
#
#   This provides a routing-based sanity check consistent with Exp 6:
#   although weights are learned on independent samples, the behaviour
#   of the convex router should match the PIER-based uniqueness pattern.

import sys
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import torch

# Add project root for `isqed`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isqed.real_world import HuggingFaceWrapper, MaskingIntervention
from isqed.ecosystem import Ecosystem
from isqed.geometry import DISCOSolver

# Add experiments/ for utils
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_dir)
from experiments.utils import make_stable_seed  # or from utils import make_stable_seed

DOSES_FIT = np.linspace(0.0, 0.9, 10)
DOSES_EVAL = np.linspace(0.0, 0.9, 10)


def load_sentences_sst2_like_exp7(
    max_samples: int,
    np_rng: np.random.RandomState,
) -> List[str]:
    print("Loading SST-2 validation data (aligned with Exp 7)...")
    try:
        dataset = load_dataset("glue", "sst2", split="validation")
        all_sentences = list(dataset["sentence"])
    except Exception as e:
        print(f"  [WARN] Failed to load SST-2 from datasets: {e}")
        all_sentences = [
            "This movie is great.",
            "Terrible acting.",
            "I loved it.",
            "The plot was boring.",
            "Amazing direction and visuals.",
        ] * 200

    all_sentences = np.array(all_sentences)
    if max_samples is not None and max_samples < len(all_sentences):
        idx = np_rng.choice(len(all_sentences), size=max_samples, replace=False)
        all_sentences = all_sentences[idx]
    all_sentences = all_sentences.tolist()

    print(f"Using {len(all_sentences)} total sentences.")
    return all_sentences


def split_fit_eval(
    sentences: List[str],
    np_rng: np.random.RandomState,
    fit_frac: float = 0.4,
) -> Tuple[List[str], List[str]]:
    idx = np.arange(len(sentences))
    np_rng.shuffle(idx)
    n_fit = int(len(idx) * fit_frac)
    fit_idx = idx[:n_fit]
    eval_idx = idx[n_fit:]

    fit_texts = [sentences[i] for i in fit_idx]
    eval_texts = [sentences[i] for i in eval_idx]

    print(f"Split into fit={len(fit_texts)}, eval={len(eval_texts)} sentences.")
    return fit_texts, eval_texts


def run_exp9_routing_sanity_middose(
    max_samples: int = 1000,
    output_filename: str = "exp9_routing_sanity_dose_grid_convex.csv",
):
    print("=== Exp 9 (revised): Mid-dose convex-routing sanity check ===")

    np_rng = np.random.RandomState(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load and split sentences
    all_sentences = load_sentences_sst2_like_exp7(max_samples=max_samples, np_rng=np_rng)
    fit_texts, eval_texts = split_fit_eval(all_sentences, np_rng=np_rng, fit_frac=0.4)

    intervention = MaskingIntervention()

    # 2) Load models (same ordering as Exp 7)
    model_ids = [
        "textattack/bert-base-uncased-SST-2",
        "textattack/distilbert-base-uncased-SST-2",
        "textattack/roberta-base-SST-2",
        "textattack/albert-base-v2-SST-2",
        "textattack/xlnet-base-cased-SST-2",
    ]
    short_names = ["BERT", "DistilBERT", "RoBERTa", "ALBERT", "XLNet"]

    models: List[HuggingFaceWrapper] = []
    print("Loading ecosystem models...")
    for mid in model_ids:
        try:
            m = HuggingFaceWrapper(mid, device)
            models.append(m)
            print(f"  [OK] Loaded: {mid}")
        except Exception as e:
            print(f"  [FAIL] Skipping {mid}: {e}")

    n_models = len(models)
    if n_models < 3:
        print("Less than 3 models loaded. Aborting.")
        return

    short_names = short_names[:n_models]

    rows = []

    # 3) For each target, fit convex weights on DOSES_FIT and evaluate on DOSES_EVAL
    for t_idx in range(n_models):
        target_model = models[t_idx]
        target_name = short_names[t_idx]
        peers = [models[j] for j in range(n_models) if j != t_idx]

        print(f"\n>>> Target: {target_name}")

        eco = Ecosystem(target=target_model, peers=peers)

        # -------------------
        # 3.1 Fit phase on P_fit and DOSES_FIT
        # -------------------
        print("  [Phase] Fitting convex router on P_fit (full dose grid)...")

        fit_X = []
        fit_Theta = []
        fit_seeds = []

        for text in fit_texts:
            for theta in DOSES_FIT:
                seed = make_stable_seed(text=text, theta=float(theta))
                fit_X.append(text)
                fit_Theta.append(float(theta))
                fit_seeds.append(int(seed))

        y_t_fit, Y_p_fit = eco.batched_query(
            X=fit_X,
            Thetas=fit_Theta,
            intervention=intervention,
            seeds=fit_seeds,
        )

        y_t_fit_vec = y_t_fit.reshape(-1, 1)
        Y_p_fit_mat = Y_p_fit

        _, w_hat = DISCOSolver.solve_weights_and_distance(
            y_t_fit_vec,
            Y_p_fit_mat,
        )
        w_hat = np.asarray(w_hat, dtype=float).flatten()

        print(f"  [Fit] Learned convex weights w_hat (first 3): {w_hat[:3]}")

        # -------------------
        # 3.2 Eval phase on P_eval and DOSES_EVAL
        # -------------------
        print("  [Phase] Evaluating routing baselines on P_eval (dose grid)...")
        eval_X = []
        eval_Theta = []
        eval_seeds = []

        for text in eval_texts:
            for theta in DOSES_EVAL:
                seed = make_stable_seed(text=text, theta=float(theta))
                eval_X.append(text)
                eval_Theta.append(float(theta))
                eval_seeds.append(int(seed))

        y_t_eval, Y_p_eval = eco.batched_query(
            X=eval_X,
            Thetas=eval_Theta,
            intervention=intervention,
            seeds=eval_seeds,
        )

        eval_Theta_arr = np.asarray(eval_Theta, dtype=float)
        abs_diffs_all = np.abs(Y_p_eval - y_t_eval[:, None])  # (n_eval, n_peers)
        residuals_convex_all = np.abs(y_t_eval - (Y_p_eval @ w_hat))  # (n_eval,)

        # Aggregate by dose
        for theta in DOSES_EVAL:
            theta_val = float(theta)
            mask = np.isclose(eval_Theta_arr, theta_val, atol=1e-8)
            if not np.any(mask):
                continue

            diffs = abs_diffs_all[mask]              # (n_subset, n_peers)
            conv_res = residuals_convex_all[mask]    # (n_subset,)

            global_dis = float(diffs.mean())

            mae_per_peer = diffs.mean(axis=0)        # (n_peers,)
            best_peer_rel_idx = int(np.argmin(mae_per_peer))
            best_peer_name = short_names[
                [j for j in range(n_models) if j != t_idx][best_peer_rel_idx]
            ]
            mae_single = float(mae_per_peer[best_peer_rel_idx])

            per_sample_min = diffs.min(axis=1)       # (n_subset,)
            mae_oracle = float(per_sample_min.mean())

            mae_convex = float(conv_res.mean())

            rows.append(
                {
                    "TargetModel": target_name,
                    "Dose": theta_val,
                    "GlobalDisagreement_D": global_dis,
                    "MAE_SinglePeer": mae_single,
                    "BestPeerName": best_peer_name,
                    "MAE_OracleRouter": mae_oracle,
                    "MAE_ConvexRouter": mae_convex,
                    "NumFitSamples": len(fit_texts),
                    "NumEvalSamples": len(eval_texts),
                }
            )

    df_out = pd.DataFrame(rows)

    out_dir = os.path.join("results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    df_out.to_csv(out_path, index=False)

    print(f"\nSaved convex-routing sanity results to: {out_path}")
    print("Done.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 9 (revised): Mid-dose convex-routing sanity check on dose grid."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of SST-2 validation sentences.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="exp9_routing_sanity_dose_grid_convex.csv",
        help="Output CSV filename under results/tables/.",
    )

    args = parser.parse_args()

    run_exp9_routing_sanity_middose(
        max_samples=args.max_samples,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
