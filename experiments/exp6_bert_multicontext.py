# experiments/exp6_bert_multicontext.py

import sys
import os
import argparse
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import torch

# Ensure we can import the local `isqed` package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isqed.real_world import HuggingFaceWrapper, MaskingIntervention
from isqed.geometry import DISCOSolver
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
# 2. Multi-context DISCO experiment
# ===========================

def run_multicontext_experiment(
    max_samples: int = 1000,
    min_context_size: int = 80,
    output_filename: str = "exp6_bert_multicontext_pier.csv",
):
    """
    Run a multi-context DISCO uniqueness experiment on the BERT ecosystem.

    Context types:
      * length_bucket:   short / medium / long
      * sentiment_bucket: strong negative / ambiguous / strong positive
      * negation_bucket: has_negation / no_negation

    For each context:
      * take up to `max_per_context` sentences
      * split into P_fit (first half) and P_eval (second half)
      * for each target model:
          - fit w_rest on P_fit using low doses
          - evaluate PIER on P_eval using high doses
      * collect mean PIER per dose and store in a CSV file.
    """
    print("=== Exp 6: BERT Multi-Context DISCO (Uniqueness via Ecosystem) ===")

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
        except Exception as e:
            print(f"  [FAIL] Skipping {mid}: {e}")

    n_models = len(models)
    if n_models < 3:
        print("Less than 3 models loaded. Aborting experiment.")
        return

    # Align short_names with the successfully loaded models
    short_names = short_names[:n_models]

    # -----------------------
    # 2.2 Load SST-2 and build features
    # -----------------------
    intervention = MaskingIntervention()

    print("Loading SST-2 validation data...")
    try:
        dataset = load_dataset("glue", "sst2", split="validation")
        all_sentences = list(dataset["sentence"])
    except Exception as e:
        print(f"  [WARN] Failed to load SST-2 from datasets: {e}")
        # Small fallback set if dataset loading fails
        all_sentences = ["This sentence is a test sentence."] * 1000

    # Subsample to at most `max_samples` sentences
    rng_global = np.random.RandomState(0)
    all_sentences = np.array(all_sentences)
    if max_samples is not None and max_samples < len(all_sentences):
        idx = rng_global.choice(len(all_sentences), size=max_samples, replace=False)
        all_sentences = all_sentences[idx]
    all_sentences = all_sentences.tolist()

    print(f"Using {len(all_sentences)} total sentences for context construction.")

    # Use the first loaded model as reference for p_pos
    ref_model = models[0]
    features_df = compute_sentence_features(all_sentences, ref_model)

    # Dose design: low doses for P_fit, high doses for P_eval
    doses_fit = np.linspace(0.0, 0.3, 4)    # not overlap with eval
    doses_eval = np.linspace(0.4, 0.9, 6)  

    print(f"P_fit doses (low):  {doses_fit}")
    print(f"P_eval doses (high): {doses_eval}")

    # -----------------------
    # 2.3 DISCO per context
    # -----------------------
    results = []

    rng = np.random.RandomState(42)

    # Each context_type uses a specific column in features_df
    context_specs = [
        ("length", "length_bucket"),
        ("sentiment", "sentiment_bucket"),
        ("negation", "negation_bucket"),
    ]

    for context_type, col_name in context_specs:
        print(f"\n=== Context Type: {context_type} (column: {col_name}) ===")

        # Group by context bucket
        for ctx_label, df_ctx in features_df.groupby(col_name):
            n_ctx = len(df_ctx)
            print(f"\n[Context] {context_type} = {ctx_label}, size = {n_ctx}")
            if n_ctx < min_context_size:
                print(f"  -> Too small (< {min_context_size}), skipping.")
                continue

            # Shuffle indices within this context
            idx = df_ctx.index.to_numpy()
            rng.shuffle(idx)
            df_ctx_shuffled = features_df.loc[idx]

            # Cap the number of sentences per context
            max_per_context = min(n_ctx, 200)
            df_ctx_shuffled = df_ctx_shuffled.iloc[:max_per_context]
            sentences_ctx = df_ctx_shuffled["sentence"].tolist()
            print(f"  -> Using {len(sentences_ctx)} sentences in this context.")

            # Split into P_fit and P_eval
            n_fit = len(sentences_ctx) // 2
            fit_texts = sentences_ctx[:n_fit]
            eval_texts = sentences_ctx[n_fit:]
            print(f"  -> Split into fit={len(fit_texts)}, eval={len(eval_texts)}")

            # For each target model, run DISCO uniqueness with peers_rest
            for i in range(n_models):
                target_model = models[i]
                target_name = short_names[i]
                peers_rest = [models[j] for j in range(n_models) if j != i]

                print(f"    >> Target: {target_name}")
                eco = Ecosystem(target=target_model, peers=peers_rest)

                # -------------------
                # FIT: build batch and learn w_hat_rest
                # -------------------
                fit_X = []
                fit_Theta = []
                fit_seeds = []

                for text in fit_texts:
                    for theta in doses_fit:
                        # Seed is deterministic function of context and dose
                        seed = abs(hash((context_type, ctx_label, text, float(theta)))) % (2**32)
                        fit_X.append(text)
                        fit_Theta.append(float(theta))
                        fit_seeds.append(int(seed))

                if len(fit_X) == 0:
                    print("      [WARN] No fit samples, skipping this target/context.")
                    continue

                y_t_fit, Y_p_fit = eco.batched_query(
                    X=fit_X,
                    Thetas=fit_Theta,
                    intervention=intervention,
                    seeds=fit_seeds,
                )

                # Ensure target vector is (N, 1) for DISCOSolver
                y_t_fit_vec = y_t_fit.reshape(-1, 1)
                Y_p_fit_mat = Y_p_fit

                _, w_hat_rest = DISCOSolver.solve_weights_and_distance(
                    y_t_fit_vec,
                    Y_p_fit_mat,
                )
                w_hat_rest = np.asarray(w_hat_rest, dtype=float).flatten()
                num_fit_points = len(fit_X)

                # -------------------
                # EVAL: build batch and compute residuals by dose
                # -------------------
                eval_X = []
                eval_Theta = []
                eval_seeds = []

                for text in eval_texts:
                    for theta in doses_eval:
                        seed = abs(hash((context_type, ctx_label, text, float(theta)))) % (2**32)
                        eval_X.append(text)
                        eval_Theta.append(float(theta))
                        eval_seeds.append(int(seed))

                if len(eval_X) == 0:
                    print("      [WARN] No eval samples, skipping this target/context.")
                    continue

                y_t_eval, Y_p_eval = eco.batched_query(
                    X=eval_X,
                    Thetas=eval_Theta,
                    intervention=intervention,
                    seeds=eval_seeds,
                )

                eval_Theta_arr = np.asarray(eval_Theta, dtype=float)
                # PIER residuals for each evaluation point
                y_mix_eval = Y_p_eval @ w_hat_rest
                residuals_all = np.abs(y_t_eval - y_mix_eval)

                # Aggregate residuals by exact dose (using a tolerance)
                for theta in doses_eval:
                    mask = np.isclose(eval_Theta_arr, float(theta), atol=1e-8)
                    vals = residuals_all[mask]
                    if vals.size == 0:
                        continue

                    mean_pier = float(np.mean(vals))
                    std_pier = float(np.std(vals))
                    num_eval_points = int(vals.size)

                    results.append(
                        {
                            "ContextType": context_type,
                            "ContextLabel": ctx_label,
                            "TargetModel": target_name,
                            "Dose": float(theta),
                            "MeanPIER": mean_pier,
                            "StdPIER": std_pier,
                            "NumEvalPoints": num_eval_points,
                            "NumFitPoints": num_fit_points,
                            "ContextRawSize": int(n_ctx),
                        }
                    )

    # -----------------------
    # 2.4 Save results
    # -----------------------
    if not results:
        print("No results produced (possibly all contexts too small).")
        return

    results_df = pd.DataFrame(results)
    out_dir = os.path.join("results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved multi-context DISCO summary to: {out_path}")


# ===========================
# 3. CLI entry point
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description="Exp 6: Multi-context DISCO for BERT ecosystem on SST-2 (via Ecosystem)."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of SST-2 validation sentences used to construct contexts.",
    )
    parser.add_argument(
        "--min_context_size",
        type=int,
        default=80,
        help="Minimum number of sentences required for a context to be evaluated.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="exp6_bert_multicontext_pier.csv",
        help="CSV file name for saving results under results/tables/.",
    )

    args = parser.parse_args()

    run_multicontext_experiment(
        max_samples=args.max_samples,
        min_context_size=args.min_context_size,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
