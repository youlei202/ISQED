# experiments/exp8_routing_sanity.py

import sys
import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import torch

# Add project root for `isqed`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isqed.real_world import HuggingFaceWrapper, MaskingIntervention

# Add experiments/ for utils
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_dir)
from experiments.utils import make_stable_seed  


def load_sentences_sst2_like_exp7(
    max_samples: int,
    np_rng: np.random.RandomState,
) -> List[str]:
    """
    Load SST-2 validation sentences and subsample using the *same*
    scheme as Exp 7:

        - np_rng = RandomState(0)
        - np_rng.choice(...) if max_samples < N

    This ensures that Exp 8 operates on (as much as possible) the
    same sentence pool as Exp 7.
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

    print(f"Using {len(all_sentences)} sentences at dose = 0.0.")
    return all_sentences


def compute_outputs_at_dose_zero(
    sentences: List[str],
    model_ids: List[str],
    device: str,
) -> (List[HuggingFaceWrapper], np.ndarray):
    """
    For each model and each sentence, compute the scalar output at dose = 0.

    We still route through MaskingIntervention with theta = 0.0 and a
    deterministic seed via make_stable_seed for consistency with Exp 7.
    """
    intervention = MaskingIntervention()
    theta = 0.0

    models: List[HuggingFaceWrapper] = []
    print("Loading models for routing sanity check...")
    for mid in model_ids:
        try:
            m = HuggingFaceWrapper(mid, device)
            models.append(m)
            print(f"  [OK] Loaded: {mid}")
        except Exception as e:
            print(f"  [FAIL] Skipping {mid}: {e}")

    if not models:
        raise RuntimeError("No models loaded. Aborting Exp 8.")

    n_models = len(models)
    n_samples = len(sentences)
    preds = np.zeros((n_models, n_samples), dtype=float)

    print("Querying all models at dose = 0.0 (no masking)...")
    for mi, model in enumerate(models):
        for si, text in enumerate(tqdm(sentences, desc=f"  Model {mi+1}/{n_models}")):
            # Deterministic intervention; consistent with Exp 7
            seed = make_stable_seed(text=text, theta=theta)
            perturbed = intervention.apply(text, theta, seed=seed)
            preds[mi, si] = float(model._forward(perturbed))

    return models, preds


def compute_routing_metrics(
    models: List[HuggingFaceWrapper],
    preds: np.ndarray,
) -> pd.DataFrame:
    """
    Compute, for each target model t:

      - Global disagreement D_t
      - Best single-peer MAE_single(t)
      - Oracle router MAE_oracle(t)
    """
    n_models, n_samples = preds.shape
    short_names = ["BERT", "DistilBERT", "RoBERTa", "ALBERT", "XLNet"]
    short_names = short_names[:n_models]

    rows = []

    for t_idx in range(n_models):
        target_name = short_names[t_idx]
        target_outputs = preds[t_idx]  # (n_samples,)

        peer_indices = [j for j in range(n_models) if j != t_idx]
        peer_outputs = preds[peer_indices]  # (n_peers, n_samples)

        # Absolute differences between target and each peer for each sample
        abs_diffs = np.abs(peer_outputs - target_outputs[None, :])  # (n_peers, n_samples)

        # 1) Global disagreement: mean over peers and samples
        global_dis = float(abs_diffs.mean())

        # 2) Best single peer: choose one peer j for all samples
        mae_per_peer = abs_diffs.mean(axis=1)  # (n_peers,)
        best_peer_idx = int(np.argmin(mae_per_peer))
        best_peer_name = short_names[peer_indices[best_peer_idx]]
        mae_single = float(mae_per_peer[best_peer_idx])

        # 3) Oracle router: choose, for each sample, the closest peer
        per_sample_min = abs_diffs.min(axis=0)  # (n_samples,)
        mae_oracle = float(per_sample_min.mean())

        rows.append(
            {
                "TargetModel": target_name,
                "GlobalDisagreement_D": global_dis,
                "MAE_SinglePeer": mae_single,
                "BestPeerName": best_peer_name,
                "MAE_OracleRouter": mae_oracle,
                "NumSamples": n_samples,
            }
        )

    return pd.DataFrame(rows)

def run_exp8_routing_sanity(
    max_samples: int = 1000,
    output_filename: str = "exp8_routing_sanity_dose0.csv",
):
    """
    Main entry point for Exp 8.

    Steps:
      1) Load and subsample SST-2 sentences using the same RNG scheme as Exp 7.
      2) Query all ecosystem models at dose = 0.
      3) For each target, compute:
           - Global disagreement D_t
           - Best single-peer MAE
           - Oracle router MAE
      4) Save a summary CSV for downstream visualization (Notebook 09).
    """
    print("=== Exp 8: Routing-based sanity check at dose = 0 ===")

    # Fix seeds for reproducibility and alignment with Exp 7
    np_rng = np.random.RandomState(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load sentences (aligned with Exp 7)
    sentences = load_sentences_sst2_like_exp7(max_samples=max_samples, np_rng=np_rng)

    # 2) Define ecosystem models (same ordering as Exp 7)
    model_ids = [
        "textattack/bert-base-uncased-SST-2",
        "textattack/distilbert-base-uncased-SST-2",
        "textattack/roberta-base-SST-2",
        "textattack/albert-base-v2-SST-2",
        "textattack/xlnet-base-cased-SST-2",
    ]

    models, preds = compute_outputs_at_dose_zero(
        sentences=sentences,
        model_ids=model_ids,
        device=device,
    )

    # 3) Compute routing metrics at dose = 0
    df_metrics = compute_routing_metrics(models, preds)
    print("\nRouting metrics at dose = 0:")
    print(df_metrics)

    # 4) Save results
    out_dir = os.path.join("results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    df_metrics.to_csv(out_path, index=False)

    print(f"\nSaved routing sanity check results to: {out_path}")
    print("Done.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 8: Routing-based sanity check at dose = 0."
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
        default="exp8_routing_sanity_dose0.csv",
        help="CSV file name under results/tables/.",
    )

    args = parser.parse_args()

    run_exp8_routing_sanity(
        max_samples=args.max_samples,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()