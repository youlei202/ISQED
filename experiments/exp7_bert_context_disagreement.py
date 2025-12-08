# experiments/exp7_bert_context_disagreement.py
#
# Context-wise disagreement profiles for the BERT ecosystem:
# For each target model, semantic context, and dose, we compute
# an average disagreement score with respect to its peers:
#   D_t(c, theta) = E_{(x,theta) in context c} [ mean_j |Y_t - Y_j| ].
#
# This is a direct, model-agnostic sanity check of the DISCO patterns:
# it does not fit any convex combination and only measures raw prediction
# differences between the target and its peers.

import sys
import os
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import torch

# Ensure we can import the local `isqed` package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isqed.real_world import HuggingFaceWrapper, MaskingIntervention
from isqed.ecosystem import Ecosystem


# ===========================
# 1. Simple context features
# ===========================

NEGATION_MARKERS = {
    "not", "no", "never", "none", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "without"
}


def contains_negation(text: str) -> bool:
    """Return True if the sentence contains a simple negation marker."""
    lower = text.lower()
    if "n't" in lower:
        return True
    for mark in NEGATION_MARKERS:
        if (
            f" {mark} " in lower
            or lower.startswith(mark + " ")
            or lower.endswith(" " + mark)
        ):
            return True
    return False


def bucket_length(n_tokens: int) -> str:
    """Bucket sentence length into short / medium / long."""
    if n_tokens <= 7:
        return "len_short_(<=7)"
    elif n_tokens <= 15:
        return "len_medium_(8-15)"
    else:
        return "len_long_(>15)"


def bucket_sentiment(p_pos: float) -> str:
    """Bucket sentiment strength based on positive-class probability."""
    if p_pos <= 0.2:
        return "sent_strong_negative_(p<=0.2)"
    elif p_pos >= 0.8:
        return "sent_strong_positive_(p>=0.8)"
    else:
        return "sent_ambiguous_(0.2<p<0.8)"


def bucket_negation(flag: bool) -> str:
    """Bucket by presence or absence of negation."""
    return "has_negation" if flag else "no_negation"


def compute_sentence_features(sentences: List[str], ref_model: HuggingFaceWrapper) -> pd.DataFrame:
    """
    Compute basic features for a list of sentences:

      * length (in tokens)
      * negation flag
      * p_pos: positive-class probability from a reference model
      * buckets for length, sentiment, and negation
    """
    records = []
    for text in tqdm(sentences, desc="Computing sentence features"):
        text = str(text)
        length = len(text.strip().split())
        has_neg = contains_negation(text)

        # Reference model probability on the unperturbed sentence
        with torch.no_grad():
            p_pos = float(ref_model._forward(text))

        records.append(
            {
                "sentence": text,
                "length": length,
                "has_negation": has_neg,
                "p_pos": p_pos,
            }
        )

    df = pd.DataFrame.from_records(records)
    df["length_bucket"] = df["length"].apply(bucket_length)
    df["sentiment_bucket"] = df["p_pos"].apply(bucket_sentiment)
    df["negation_bucket"] = df["has_negation"].apply(bucket_negation)
    return df


# ===========================
# 2. Main experiment
# ===========================

def run_context_disagreement_experiment(
    max_samples: int = 1000,
    min_context_size: int = 80,
    output_filename: str = "exp7_bert_context_disagreement.csv",
):
    """
    Compute context-wise disagreement profiles for the BERT ecosystem.

    For each context type (length, sentiment, negation), each context label,
    each target model, and each dose, we compute an average disagreement:

        D_t(c, theta) = mean_{(x,theta), peers j} |Y_t(x,theta) - Y_j(x,theta)|

    and store summary statistics (mean, std, count) in a CSV file.
    """
    print("=== Exp 7: Context-wise disagreement for BERT ecosystem ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------
    # 2.1 Load models
    # -----------------------
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
        print("Less than 3 models loaded. Aborting experiment.")
        return
    short_names = short_names[:n_models]

    # -----------------------
    # 2.2 Data and features
    # -----------------------
    intervention = MaskingIntervention()

    print("Loading SST-2 validation data...")
    try:
        dataset = load_dataset("glue", "sst2", split="validation")
        all_sentences = list(dataset["sentence"])
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
    if max_samples is not None and max_samples < len(all_sentences):
        idx = rng.choice(len(all_sentences), size=max_samples, replace=False)
        all_sentences = all_sentences[idx]
    all_sentences = all_sentences.tolist()

    print(f"Using {len(all_sentences)} total sentences.")

    # Use the first loaded model as reference for p_pos
    ref_model = models[0]
    features_df = compute_sentence_features(all_sentences, ref_model)

    # Doses of interest: a low dose and a high dose to highlight regime change
    doses = [0.40, 0.88]

    print(f"Doses for disagreement analysis: {doses}")

    # -----------------------
    # 2.3 Context-wise loops
    # -----------------------
    results = []

    context_specs = [
        ("length", "length_bucket"),
        ("sentiment", "sentiment_bucket"),
        ("negation", "negation_bucket"),
    ]

    for context_type, col_name in context_specs:
        print(f"\n=== Context Type: {context_type} (column: {col_name}) ===")

        for ctx_label, df_ctx in features_df.groupby(col_name):
            n_ctx = len(df_ctx)
            print(f"\n[Context] {context_type} = {ctx_label}, size = {n_ctx}")
            if n_ctx < min_context_size:
                print(f"  -> Too small (< {min_context_size}), skipping.")
                continue

            # Shuffle and cap the number of sentences for this context
            idx = df_ctx.index.to_numpy()
            rng.shuffle(idx)
            df_ctx_shuffled = features_df.loc[idx]

            max_per_context = min(n_ctx, 200)
            df_ctx_shuffled = df_ctx_shuffled.iloc[:max_per_context]
            sentences_ctx = df_ctx_shuffled["sentence"].tolist()
            print(f"  -> Using {len(sentences_ctx)} sentences in this context.")

            # Pre-build the (X, Theta, seeds) grid shared by all targets
            eval_X = []
            eval_Theta = []
            eval_seeds = []

            for text in sentences_ctx:
                for theta in doses:
                    seed = abs(hash((context_type, ctx_label, text, float(theta)))) % (2**32)
                    eval_X.append(text)
                    eval_Theta.append(float(theta))
                    eval_seeds.append(int(seed))

            if not eval_X:
                print("  -> No evaluation points constructed, skipping.")
                continue

            # For each target, compute disagreement with its peers
            for i in range(n_models):
                target_model = models[i]
                target_name = short_names[i]
                peers = [models[j] for j in range(n_models) if j != i]

                print(f"    >> Target: {target_name}")

                eco = Ecosystem(target=target_model, peers=peers)

                # Query target and peers jointly
                y_t_eval, Y_p_eval = eco.batched_query(
                    X=eval_X,
                    Thetas=eval_Theta,
                    intervention=intervention,
                    seeds=eval_seeds,
                )

                eval_Theta_arr = np.asarray(eval_Theta, dtype=float)

                # Compute per-sample disagreement: mean absolute difference over peers
                # shape: (N_eval,) for y_t_eval, (N_eval, n_peers) for Y_p_eval
                abs_diff = np.abs(Y_p_eval - y_t_eval[:, None])  # broadcast over peers
                mean_abs_diff_per_sample = abs_diff.mean(axis=1)

                # Aggregate by dose
                for theta in doses:
                    mask = np.isclose(eval_Theta_arr, float(theta), atol=1e-8)
                    vals = mean_abs_diff_per_sample[mask]
                    if vals.size == 0:
                        continue

                    mean_dis = float(np.mean(vals))
                    std_dis = float(np.std(vals))
                    num_points = int(vals.size)

                    results.append(
                        {
                            "ContextType": context_type,
                            "ContextLabel": ctx_label,
                            "TargetModel": target_name,
                            "Dose": float(theta),
                            "MeanDisagreement": mean_dis,
                            "StdDisagreement": std_dis,
                            "NumPoints": num_points,
                            "ContextRawSize": int(n_ctx),
                        }
                    )

    if not results:
        print("No results produced (all contexts too small?).")
        return

    results_df = pd.DataFrame(results)
    out_dir = os.path.join("results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved context-wise disagreement summary to: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 7: Context-wise disagreement for BERT ecosystem."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of SST-2 validation sentences.",
    )
    parser.add_argument(
        "--min_context_size",
        type=int,
        default=80,
        help="Minimum number of sentences for a context to be evaluated.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="exp7_bert_context_disagreement.csv",
        help="Output CSV filename under results/tables/.",
    )

    args = parser.parse_args()

    run_context_disagreement_experiment(
        max_samples=args.max_samples,
        min_context_size=args.min_context_size,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
