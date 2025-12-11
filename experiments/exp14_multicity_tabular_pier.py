# experiments/exp14_multicity_tabular_pier.py
#
# Exp 14: Multi-city tabular ecosystem audit (PIER vs. model removal cost).
#
# UTD19 raw CSV download (utd19_u.csv):
#   ETH Research Collection (open access, CC BY-NC-SA 4.0):
#   https://www.research-collection.ethz.ch/handle/20.500.11850/437802
#   File: utd19_u.csv
#
# In this experiment we assume the file has been downloaded to:
#   /work3/leiyo/utd19_u.csv
#
# Setting:
#   - We have a tabular dataset with a "city" column and a target column y.
#   - For each city c, we train a local regression model M_c on that city's
#     training data.
#   - We also train a single global model M_global on all cities' training data.
#
# Goal:
#   - For each city c, quantify how "unique" its local model is within the
#     ecosystem {M_global} ∪ {M_c' : c' != c}, using PIER.
#   - Then measure how much test performance we lose if we REMOVE M_c and
#     replace it by:
#       (i) the global model only, or
#       (ii) a convex combination of peers (global + other cities).
#
#   If PIER is meaningful, we expect:
#     - Cities with high PIER_c suffer large performance loss when their
#       local model is removed, even if we use an optimal convex router.
#     - Cities with low PIER_c can be safely merged into the global/peer
#       ensemble with little performance degradation.
#
# This makes PIER directly relevant for industrial "model consolidation"
# decisions in large ecosystems (e.g., many city-specific models).

import os
import sys
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Make local package importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

DATA_DIR = os.path.join("/work3/leiyo/utd19")
os.makedirs(DATA_DIR, exist_ok=True)

# Default raw UTD19 CSV location on the cluster
DEFAULT_UTD19_CSV = os.path.join(DATA_DIR, "utd19_u.csv")
# Kept for backwards compatibility (not used in UTD19 mode)
DEFAULT_DATA_CSV = os.path.join(DATA_DIR, "exp14_multicity_synth.csv")

from isqed.ecosystem import Ecosystem
from isqed.geometry import DISCOSolver
from isqed.real_world import TabularIdentityIntervention, TabularModelWrapper

# Import deterministic seed helper
sys.path.append(os.path.join(ROOT_DIR, "experiments"))
from experiments.utils import make_stable_seed


def prepare_utd19_for_exp14(
    df_raw: pd.DataFrame,
    rng: np.random.RandomState,
    city_col: str = "city",
    detid_col: str = "detid",
    day_col: str = "day",
    interval_col: str = "interval",
    speed_col: str = "speed",
    flow_col: str = "flow",
    occ_col: str = "occ",
    error_col: str = "error",
    max_cities: Optional[int] = 40,
) -> pd.DataFrame:
    """
    Transform raw UTD19 measurements (utd19_u.csv) into the format expected by
    Exp14: a single table with columns [city, split, y, numeric features...].

    - Filters to rows without error flag (error_col is NaN).
    - Optionally selects a random subset of cities (up to max_cities).
    - Sorts by (city, detid, day, interval).
    - Builds a next-interval speed regression target: y(t) = speed(t+1).
      (within each (city, detid, day) group).
    - Drops rows with missing values in core numeric columns or in y.
    - Builds time-based train/val/test splits per city (60%/20%/20%).
    """
    df = df_raw.copy()

    required_cols = [city_col, detid_col, day_col, interval_col, speed_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"UTD19 preprocessing expects columns {required_cols}, "
            f"but the following are missing: {missing}"
        )

    # Filter out rows with an error flag (keep only error == NaN)
    if error_col in df.columns:
        before = len(df)
        df = df[df[error_col].isna()].copy()
        print(f"UTD19: filtered error-flagged rows: {before} -> {len(df)}")

    # Choose a subset of cities (random, reproducible)
    all_cities = sorted(df[city_col].dropna().unique().tolist())
    print(f"UTD19: found {len(all_cities)} cities in raw data: {all_cities}")

    if max_cities is not None and max_cities < len(all_cities):
        chosen = rng.choice(all_cities, size=max_cities, replace=False)
        selected_cities = sorted(chosen.tolist())
        print(f"UTD19: randomly selected {len(selected_cities)} cities: {selected_cities}")
    else:
        selected_cities = all_cities
        print("UTD19: using all available cities (no subsampling).")

    df = df[df[city_col].isin(selected_cities)].copy()

    # Sort for consistent temporal ordering
    sort_cols = [city_col, detid_col, day_col, interval_col]
    df = df.sort_values(sort_cols)

    # We will use speed as the regression target (next-interval prediction)
    # and flow/occ/speed/day/interval as numeric features.
    for col in [flow_col, occ_col]:
        if col not in df.columns:
            raise ValueError(
                f"UTD19 preprocessing expects numeric column '{col}' "
                f"but it is missing."
            )

    group_cols = [city_col, detid_col, day_col]
    df["y"] = df.groupby(group_cols)[speed_col].shift(-1)

    # Drop rows with missing measurements or target
    before = len(df)
    df = df.dropna(subset=[speed_col, flow_col, occ_col, "y"]).copy()
    print(f"UTD19: after building target and dropping NaNs: {before} -> {len(df)} rows")

    # Build time-based splits (60% train, 20% val, 20% test) per city
    def assign_split_per_city(g: pd.DataFrame) -> pd.DataFrame:
        n = len(g)
        if n == 0:
            return g
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        # Ensure non-negative and not exceeding n
        n_train = max(0, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
        n_test = n - n_train - n_val

        # In very small groups, this may collapse, but those cities
        # will anyway be filtered out by min_*_samples downstream.
        splits = np.array(["test"] * n, dtype=object)
        if n_train > 0:
            splits[:n_train] = "train"
        if n_val > 0:
            splits[n_train:n_train + n_val] = "val"

        g = g.copy()
        g["split"] = splits
        return g

    df = df.groupby(city_col, group_keys=False).apply(assign_split_per_city)

    print("UTD19: preprocessing complete.")
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    mean_vec: pd.Series,
    std_vec: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) from a city-specific dataframe.

    Features are standardized using global train mean/std for stability.
    """
    X_raw = df[feature_cols]
    X_std = (X_raw - mean_vec) / std_vec.replace(0.0, 1.0)
    X = X_std.values.astype(np.float32)
    y = df[target_col].astype(float).values
    return X, y


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 0,
) -> GradientBoostingRegressor:
    """
    Train a simple GradientBoostingRegressor as the base model.

    You can swap this out for any tabular model (e.g., XGBoost, LightGBM).
    """
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def run_multicity_tabular_experiment(
    data_csv: str,
    city_col: str = "city",
    target_col: str = "y",
    split_col: str = "split",
    train_split_name: str = "train",
    val_split_name: str = "val",
    test_split_name: str = "test",
    max_cities: Optional[int] = 40,
    min_train_samples: int = 500,
    min_val_samples: int = 200,
    min_test_samples: int = 200,
    max_fit_samples: int = 2000,
    max_eval_samples: int = 2000,
    output_filename: str = "exp14_multicity_tabular_summary.csv",
):
    print("=== Exp 14: Multi-city tabular PIER vs model removal cost ===")

    # Fix all relevant seeds for reproducibility
    global_seed = 0
    rng = np.random.RandomState(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    torch.manual_seed(global_seed)

    # ---------------------------------------------------------------
    # 1. Load data (either prepared CSV or raw UTD19)
    # ---------------------------------------------------------------
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"data_csv not found: {data_csv}")

    print(f"Loading CSV from: {data_csv}")
    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} rows from {data_csv}")

    # If the CSV does not contain y/split, assume it's raw UTD19
    # and run the preprocessing step to create them.
    utd19_mode = False
    if (target_col not in df.columns) or (split_col not in df.columns):
        print(
            f"Input CSV is missing '{target_col}' or '{split_col}'. "
            "Assuming raw UTD19 measurements and preprocessing..."
        )
        df = prepare_utd19_for_exp14(
            df_raw=df,
            rng=rng,
            city_col=city_col,
            max_cities=max_cities,
        )
        utd19_mode = True
        # We already subsampled cities inside prepare_utd19_for_exp14
        # so we should not do another max_cities cut later.
        max_cities = None

    # Sanity check for required columns
    for col in [city_col, target_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data.")

    if split_col not in df.columns:
        raise ValueError(
            f"Column '{split_col}' not found. "
            "Please provide a split column with values train/val/test."
        )

    # ---------------------------------------------------------------
    # 2. Determine feature columns and standardization stats
    # ---------------------------------------------------------------
    if utd19_mode:
        # For UTD19 we explicitly pick a stable set of numeric features.
        candidate_cols = ["flow", "occ", "speed", "day", "interval"]
        numeric_cols = [c for c in candidate_cols if c in df.columns]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target and split if they appear in numeric columns
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if split_col in numeric_cols:
        numeric_cols.remove(split_col)

    if not numeric_cols:
        raise ValueError("No numeric feature columns found for tabular modeling.")

    print(f"Using {len(numeric_cols)} numeric feature columns: {numeric_cols}")

    # Drop rows with missing numeric features or target
    before = len(df)
    df = df.dropna(subset=numeric_cols + [target_col])
    print(f"Dropped rows with NaNs in features/target: {before} -> {len(df)}")

    # Select cities (possibly random subset, reproducible)
    all_cities = sorted(df[city_col].unique().tolist())
    original_city_count = len(all_cities)

    if max_cities is not None and max_cities < len(all_cities):
        chosen = rng.choice(all_cities, size=max_cities, replace=False)
        all_cities = sorted(chosen.tolist())
        print(
            f"Randomly selected {len(all_cities)} cities "
            f"out of {original_city_count} total."
        )

    print(f"Using {len(all_cities)} cities: {all_cities}")

    # Global train subset for computing mean/std
    df_train_global = df[df[split_col] == train_split_name]
    if df_train_global.empty:
        raise ValueError("No global train samples found after preprocessing.")

    mean_vec = df_train_global[numeric_cols].mean()
    std_vec = df_train_global[numeric_cols].std().replace(0.0, 1.0)

    # ---------------------------------------------------------------
    # 3. Train per-city models and the global model
    # ---------------------------------------------------------------
    city_models: Dict[str, GradientBoostingRegressor] = {}
    city_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    print("\nTraining per-city models...")
    for city in all_cities:
        df_city = df[df[city_col] == city]

        df_train = df_city[df_city[split_col] == train_split_name]
        df_val = df_city[df_city[split_col] == val_split_name]
        df_test = df_city[df_city[split_col] == test_split_name]

        n_train = len(df_train)
        n_val = len(df_val)
        n_test = len(df_test)

        print(f"  City={city}: train={n_train}, val={n_val}, test={n_test}")

        if (
            n_train < min_train_samples
            or n_val < min_val_samples
            or n_test < min_test_samples
        ):
            print("    [SKIP] Not enough data for this city, skipping.")
            continue

        X_train, y_train = build_feature_matrix(
            df_train, numeric_cols, target_col, mean_vec, std_vec
        )
        X_val, y_val = build_feature_matrix(
            df_val, numeric_cols, target_col, mean_vec, std_vec
        )
        X_test, y_test = build_feature_matrix(
            df_test, numeric_cols, target_col, mean_vec, std_vec
        )

        model_c = train_regressor(X_train, y_train, random_state=global_seed)
        city_models[city] = model_c
        city_data[city] = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    if not city_models:
        print("No eligible cities found after filtering. Abort.")
        return

    # Train global model on all selected cities' training data
    print("\nTraining global model on all selected cities...")
    train_blocks = []
    target_blocks = []
    for city, parts in city_data.items():
        X_train_c, y_train_c = parts["train"]
        train_blocks.append(X_train_c)
        target_blocks.append(y_train_c)

    X_train_global = np.concatenate(train_blocks, axis=0)
    y_train_global = np.concatenate(target_blocks, axis=0)

    global_model = train_regressor(
        X_train_global, y_train_global, random_state=global_seed
    )

    # Wrap models for ecosystem usage
    wrappers: Dict[str, TabularModelWrapper] = {}
    for city, model_c in city_models.items():
        wrappers[city] = TabularModelWrapper(model_c, name=city)
    wrappers["GLOBAL"] = TabularModelWrapper(global_model, name="GLOBAL")

    identity_intervention = TabularIdentityIntervention()

    # ---------------------------------------------------------------
    # 4. Evaluate local/global baselines and compute PIER + router
    # ---------------------------------------------------------------
    rows = []

    for city in sorted(city_models.keys()):
        print(f"\n=== City: {city} ===")

        X_val, y_val = city_data[city]["val"]
        X_test, y_test = city_data[city]["test"]

        # Baseline: local and global MAE on test
        local_model = city_models[city]
        local_pred = local_model.predict(X_test)
        local_mae = float(mean_absolute_error(y_test, local_pred))

        global_pred = global_model.predict(X_test)
        global_mae = float(mean_absolute_error(y_test, global_pred))

        print(f"  Local MAE:  {local_mae:.4f}")
        print(f"  Global MAE: {global_mae:.4f}")

        # -----------------------------------------------------------
        # 4.1 Fit phase: learn convex weights on validation set
        # -----------------------------------------------------------
        target_wrapper = wrappers[city]
        peer_wrappers = [
            wrappers[name]
            for name in wrappers.keys()
            if name != city
        ]

        eco = Ecosystem(target=target_wrapper, peers=peer_wrappers)

        # Build P_fit from val samples (possibly subsampled)
        n_val = X_val.shape[0]
        fit_indices = np.arange(n_val)
        rng.shuffle(fit_indices)
        if n_val > max_fit_samples:
            fit_indices = fit_indices[:max_fit_samples]

        fit_X = [X_val[i] for i in fit_indices]
        fit_Theta = [0.0 for _ in fit_indices]
        fit_seeds = [
            make_stable_seed(
                text=f"{city}|fit|{int(i)}",
                theta=0.0,
                context_type="city",
                ctx_label=city,
            )
            for i in fit_indices
        ]

        y_t_fit, Y_p_fit = eco.batched_query(
            X=fit_X,
            Thetas=fit_Theta,
            intervention=identity_intervention,
            seeds=fit_seeds,
        )

        y_t_fit_vec = y_t_fit.reshape(-1, 1)
        Y_p_fit_mat = Y_p_fit

        _, w_hat = DISCOSolver.solve_weights_and_distance(
            y_t_fit_vec,
            Y_p_fit_mat,
        )
        w_hat = np.asarray(w_hat, dtype=float).flatten()
        print(f"  [Fit] Learned convex weights (first 3): {w_hat[:3]}")

        # -----------------------------------------------------------
        # 4.2 Eval phase: PIER and convex router MAE on test set
        # -----------------------------------------------------------
        n_test = X_test.shape[0]
        eval_indices = np.arange(n_test)
        rng.shuffle(eval_indices)
        if n_test > max_eval_samples:
            eval_indices = eval_indices[:max_eval_samples]

        eval_X = [X_test[i] for i in eval_indices]
        eval_Theta = [0.0 for _ in eval_indices]
        eval_seeds = [
            make_stable_seed(
                text=f"{city}|eval|{int(i)}",
                theta=0.0,
                context_type="city",
                ctx_label=city,
            )
            for i in eval_indices
        ]

        y_t_eval, Y_p_eval = eco.batched_query(
            X=eval_X,
            Thetas=eval_Theta,
            intervention=identity_intervention,
            seeds=eval_seeds,
        )

        # PIER (mean absolute residual between target and convex comb)
        y_mix_eval = Y_p_eval @ w_hat
        residuals = np.abs(y_t_eval - y_mix_eval)
        pier_city = float(np.mean(residuals))

        # Convex router MAE vs ground truth on test
        # For router, we use the same w_hat but evaluate on the *full* test set.
        full_eval_X = [X_test[i] for i in range(n_test)]
        full_eval_Theta = [0.0] * n_test
        full_eval_seeds = [
            make_stable_seed(
                text=f"{city}|test|{int(i)}",
                theta=0.0,
                context_type="city",
                ctx_label=city,
            )
            for i in range(n_test)
        ]

        y_t_test, Y_p_test = eco.batched_query(
            X=full_eval_X,
            Thetas=full_eval_Theta,
            intervention=identity_intervention,
            seeds=full_eval_seeds,
        )

        y_router = Y_p_test @ w_hat
        router_mae = float(mean_absolute_error(y_test, y_router))

        delta_global = float(global_mae - local_mae)
        delta_router = float(router_mae - local_mae)

        print(f"  PIER (city model vs convex peers): {pier_city:.4f}")
        print(f"  Convex router MAE: {router_mae:.4f} (Δ vs local = {delta_router:.4f})")

        rows.append(
            {
                "City": city,
                "Local_MAE": local_mae,
                "Global_MAE": global_mae,
                "Router_MAE": router_mae,
                "Delta_Global": delta_global,
                "Delta_Router": delta_router,
                "PIER": pier_city,
                "NumTrain": int(city_data[city]["train"][0].shape[0]),
                "NumVal": int(city_data[city]["val"][0].shape[0]),
                "NumTest": int(city_data[city]["test"][0].shape[0]),
                "NumFitSamples": int(len(fit_indices)),
                "NumEvalSamples": int(len(eval_indices)),
                "NumPeers": int(len(peer_wrappers)),
            }
        )

    if not rows:
        print("No results to save (no eligible cities?).")
        return

    summary_df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT_DIR, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    summary_df.to_csv(out_path, index=False)

    print(f"\nSaved multi-city tabular summary to: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 14: Multi-city tabular PIER vs model removal cost."
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default=DEFAULT_UTD19_CSV,
        help=(
            "Path to a CSV file. If the file already has columns "
            "[city, split, y, numeric features...], it will be used as-is. "
            "If it is a raw UTD19 measurements file (utd19_u.csv without "
            "'split'/'y'), it will be preprocessed automatically. "
            f"Default: {DEFAULT_UTD19_CSV}"
        ),
    )
    parser.add_argument(
        "--city_col",
        type=str,
        default="city",
        help="Name of the city identifier column.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="y",
        help="Name of the target column (after preprocessing).",
    )
    parser.add_argument(
        "--split_col",
        type=str,
        default="split",
        help="Name of the split column (train/val/test).",
    )
    parser.add_argument(
        "--max_cities",
        type=int,
        default=40,
        help=(
            "Maximum number of cities to use. "
            "For raw UTD19, a random subset of at most this many cities "
            "is selected (with fixed seed for reproducibility). "
            "For already-prepared CSVs, a random subset of this size is "
            "selected from the available cities."
        ),
    )
    parser.add_argument(
        "--min_train_samples",
        type=int,
        default=500,
        help="Minimum number of train samples per city.",
    )
    parser.add_argument(
        "--min_val_samples",
        type=int,
        default=200,
        help="Minimum number of val samples per city.",
    )
    parser.add_argument(
        "--min_test_samples",
        type=int,
        default=200,
        help="Minimum number of test samples per city.",
    )
    parser.add_argument(
        "--max_fit_samples",
        type=int,
        default=2000,
        help="Maximum number of fit samples per city (for P_fit).",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=2000,
        help="Maximum number of eval samples per city (for P_eval).",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="exp14_multicity_tabular_summary.csv",
        help="Output CSV under results/tables/.",
    )

    args = parser.parse_args()

    run_multicity_tabular_experiment(
        data_csv=args.data_csv,
        city_col=args.city_col,
        target_col=args.target_col,
        split_col=args.split_col,
        max_cities=args.max_cities,
        min_train_samples=args.min_train_samples,
        min_val_samples=args.min_val_samples,
        min_test_samples=args.min_test_samples,
        max_fit_samples=args.max_fit_samples,
        max_eval_samples=args.max_eval_samples,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
