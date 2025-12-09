# experiments/exp9_routing_sanity_dose05.py
#
# Exp 9 (revised): Mid-dose convex-routing sanity check (theta = 0.5).
#
# Goal:
#   At theta ≈ 0.5, PIER from Exp 6 suggests that DistilBERT is relatively
#   easy to approximate via a single global convex combination of peers,
#   whereas ALBERT remains the hardest model to approximate.
#
#   In contrast, simple model differencing (raw disagreement) still ranks
#   ALBERT and DistilBERT as the most "different" models at this dose.
#
#   This experiment uses a DISCO-style convex router to validate PIER:
#     For each target model t:
#       1) Split sentences into P_fit and P_eval.
#       2) On P_fit (at theta = 0.5), learn a *global* convex weight vector
#          w_convex(t) using DISCOSolver.
#       3) On P_eval (dose = 0.5), compute:
#            - GlobalDisagreement_D      (raw MD baseline)
#            - MAE_SinglePeer           (best fixed single peer)
#            - MAE_OracleRouter         (per-sample best peer)
#            - MAE_ConvexRouter         (fixed convex router with w_convex)
#
#   If PIER is correct, at theta ≈ 0.5 we expect:
#       - DistilBERT to have relatively low MAE_ConvexRouter (easy to
#         approximate via a single convex combination),
#       - while ALBERT still has a large MAE_ConvexRouter.
#
#   This provides a complementary case to Exp 8: there, even an oracle
#   router cannot remove ALBERT's low-dose uniqueness; here, a simple
#   global convex router *can* remove much of DistilBERT's mid-dose
#   pseudo-uniqueness, in line with PIER.

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
from experiments.utils import make_stable_seed 


def load_sentences_sst2_like_exp7(
    max_samples: int,
    np_rng: np.random.RandomState,
) -> List[str]:
    """
    Load SST-2 validation sentences and subsample using the same scheme
    as Exp 7:

        - np_rng = RandomState(0)
        - np_rng.choice(...) if max_samples < N

    This ensures that Exp 9 operates on (as much as possible) the same
    sentence pool as Exp 7.
    """
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
    fit_frac: float = 0.5,
) -> Tuple[List[str], List[str]]:
    """
    Deterministically split a list of sentences into P_fit and P_eval.
    """
    idx = np.arange(len(sentences))
    np_rng.shuffle(idx)
    n_fit = int(len(idx) * fit_frac)
    fit_idx = idx[:n_fit]
    eval_idx = idx[n_fit:]

    fit_texts = [sentences[i] for i in fit_idx]
    eval_texts = [sentences[i] for i in eval_idx]

    print(f"Split into fit={len(fit_texts)}, eval={len(eval_texts)} sentences.")
    return fit_texts, eval_texts


def build_query_grid(
    texts: List[str],
    theta: float,
) -> Tuple[List[str], List[float], List[int]]:
    """
    Build lists (X, Thetas, seeds) for a fixed dose theta.
    """
    X = []
    Thetas = []
    seeds = []
    for text in texts:
        seed = make_stable_seed(text=text, theta=theta)
        X.append(text)
        Thetas.append(float(theta))
        seeds.append(int(seed))
    return X, Thetas, seeds


def run_exp9_routing_sanity_middose(
    max_samples: int = 1000,
    dose: float = 0.5,
    output_filename: str = "exp9_routing_sanity_dose05_convex.csv",
):
    print("=== Exp 9 (revised): Mid-dose convex-routing sanity check ===")

    # Seeds aligned with Exp 7
    np_rng = np.random.RandomState(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load and split sentences
    all_sentences = load_sentences_sst2_like_exp7(max_samples=max_samples, np_rng=np_rng)
    fit_texts, eval_texts = split_fit_eval(all_sentences, np_rng=np_rng, fit_frac=0.5)

    theta = float(dose)
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

    # 3) For each target, fit convex weights on P_fit and evaluate on P_eval
    for t_idx in range(n_models):
        target_model = models[t_idx]
        target_name = short_names[t_idx]
        peers = [models[j] for j in range(n_models) if j != t_idx]

        print(f"\n>>> Target: {target_name}")

        eco = Ecosystem(target=target_model, peers=peers)

        # 3.1 Fit phase at theta = dose
        print("  [Phase] Fitting convex router on P_fit (mid dose)...")
        fit_X, fit_Theta, fit_seeds = build_query_grid(fit_texts, theta=theta)

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

        # 3.2 Eval phase at theta = dose
        print("  [Phase] Evaluating routing baselines on P_eval (mid dose)...")
        eval_X, eval_Theta, eval_seeds = build_query_grid(eval_texts, theta=theta)

        y_t_eval, Y_p_eval = eco.batched_query(
            X=eval_X,
            Thetas=eval_Theta,
            intervention=intervention,
            seeds=eval_seeds,
        )

        # Global disagreement (model differencing baseline)
        abs_diffs = np.abs(Y_p_eval - y_t_eval[:, None])  # (n_eval, n_peers)
        global_dis = float(abs_diffs.mean())

        # Best single peer (fixed index across all samples)
        mae_per_peer = abs_diffs.mean(axis=0)  # (n_peers,)
        best_peer_rel_idx = int(np.argmin(mae_per_peer))
        best_peer_name = short_names[[j for j in range(n_models) if j != t_idx][best_peer_rel_idx]]
        mae_single = float(mae_per_peer[best_peer_rel_idx])

        # Oracle router (per-sample best peer)
        per_sample_min = abs_diffs.min(axis=1)  # (n_eval,)
        mae_oracle = float(per_sample_min.mean())

        # Convex router using global w_hat (same as DISCO baseline)
        y_mix_convex = Y_p_eval @ w_hat
        mae_convex = float(np.mean(np.abs(y_t_eval - y_mix_convex)))

        rows.append(
            {
                "TargetModel": target_name,
                "GlobalDisagreement_D": global_dis,
                "MAE_SinglePeer": mae_single,
                "BestPeerName": best_peer_name,
                "MAE_OracleRouter": mae_oracle,
                "MAE_ConvexRouter": mae_convex,
                "Dose": theta,
                "NumFitSamples": len(fit_texts),
                "NumEvalSamples": len(eval_texts),
            }
        )

    df_out = pd.DataFrame(rows)

    out_dir = os.path.join("results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    df_out.to_csv(out_path, index=False)

    print(f"\nSaved mid-dose convex-routing sanity results to: {out_path}")
    print("Done.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 9 (revised): Mid-dose convex-routing sanity check."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of SST-2 validation sentences.",
    )
    parser.add_argument(
        "--dose",
        type=float,
        default=0.5,
        help="Masking dose (theta) for this experiment.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="exp9_routing_sanity_dose05_convex.csv",
        help="Output CSV filename under results/tables/.",
    )

    args = parser.parse_args()

    run_exp9_routing_sanity_middose(
        max_samples=args.max_samples,
        dose=args.dose,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
