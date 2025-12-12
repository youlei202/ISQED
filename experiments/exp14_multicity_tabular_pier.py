# experiments/exp14_multicity_tabular_pier.py
#
# Exp 14: Multi-city tabular ecosystem audit (PIER vs. model removal cost).
#
# UTD19 raw CSV download (utd19_u.csv):
#   ETH Research Collection (open access, CC BY-NC-SA 4.0):
#   https://www.research-collection.ethz.ch/handle/20.500.11850/437802
#   File: utd19_u.csv
#
# Expected local path on the cluster:
#   /work3/leiyo/utd19/utd19_u.csv
#
# This implementation ("Plan-2 vectorized") avoids per-sample ecosystem querying.
# It uses vectorized sklearn.predict, plus chunking for router evaluation.
#
# New additions:
#   - Model caching (dump/load) under DATA_DIR/exp14_models/
#   - Persist_MAE baseline
#   - MeanFlow_Test and Local_MAE_Ratio saved to the summary CSV

import os
import sys
import json
import random
import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# For model caching
import joblib

# Make local package importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

DATA_DIR = "/work3/leiyo/utd19"
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_UTD19_CSV = os.path.join(DATA_DIR, "utd19_u.csv")

from isqed.geometry import DISCOSolver


# ---------------------------------------------------------------------
# 0) Determinism
# ---------------------------------------------------------------------
def set_global_seed(seed: int = 0) -> np.random.RandomState:
    """Fix all relevant RNG seeds for reproducibility."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return rng


# ---------------------------------------------------------------------
# 1) UTD19 preprocessing (robust & minimal)
# ---------------------------------------------------------------------
def prepare_utd19_for_exp14(
    df_raw: pd.DataFrame,
    city_col: str = "city",
    detid_col: str = "detid",
    day_col: str = "day",
    interval_col: str = "interval",
    flow_col: str = "flow",
    occ_col: str = "occ",
    error_col: str = "error",
) -> pd.DataFrame:
    """
    Convert raw UTD19 measurements to a single table usable by Exp14.

    Target:
      y(t) = flow(t+1) within each (city, detid) sequence ordered by (day, interval).

    Features (tabular):
      interval, flow, occ

    Steps:
      - Keep only needed columns.
      - Drop error column (do NOT filter on it).
      - Coerce interval/flow/occ to numeric; replace +/-inf by NaN.
      - Sort by (city, detid, day, interval); day is string "YYYY-MM-DD".
      - Create y via groupby shift(-1).
      - Drop rows with missing values in [interval, flow, occ, y].
      - Create time-based split per city: 60/20/20 (train/val/test).
    """
    required = [city_col, detid_col, day_col, interval_col, flow_col, occ_col]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"UTD19 preprocessing missing required columns: {missing}")

    df = df_raw[required].copy()

    # Drop error column if present in raw (already excluded above, but keep safe)
    if error_col in df.columns:
        df = df.drop(columns=[error_col])

    # Coerce numeric columns
    for c in [interval_col, flow_col, occ_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace inf/-inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Sort to define time order
    df = df.sort_values([city_col, detid_col, day_col, interval_col])

    # Build next-step flow target per (city, detid)
    df["y"] = df.groupby([city_col, detid_col])[flow_col].shift(-1)

    # Drop missing rows in features/target
    before = len(df)
    df = df.dropna(subset=[interval_col, flow_col, occ_col, "y"]).copy()
    print(f"UTD19: after building target and dropping NaNs/inf: {before} -> {len(df)} rows")

    if df.empty:
        raise ValueError("UTD19 preprocessing resulted in an empty dataframe.")

    # Time-based split per city (60/20/20)
    def assign_split(g: pd.DataFrame) -> pd.DataFrame:
        n = len(g)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        n_train = max(0, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))

        splits = np.array(["test"] * n, dtype=object)
        if n_train > 0:
            splits[:n_train] = "train"
        if n_val > 0:
            splits[n_train:n_train + n_val] = "val"

        out = g.copy()
        out["split"] = splits
        return out

    df = df.groupby(city_col, group_keys=False).apply(assign_split)
    print("UTD19: preprocessing complete.")
    return df


# ---------------------------------------------------------------------
# 2) Feature matrix builder (finite-safe)
# ---------------------------------------------------------------------
def build_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "y",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) as float64 arrays, and drop any non-finite rows.
    """
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")

    Xv = X.to_numpy(dtype=float, copy=True)
    yv = y.to_numpy(dtype=float, copy=True)

    mask = np.isfinite(Xv).all(axis=1) & np.isfinite(yv)
    if not mask.all():
        bad = (~mask).sum()
        print(f"      [WARN] Dropping {bad} rows with non-finite X or y in build_xy.")
        Xv = Xv[mask]
        yv = yv[mask]

    return Xv, yv


# ---------------------------------------------------------------------
# 3) Model training + caching
# ---------------------------------------------------------------------
def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_params: Dict,
) -> GradientBoostingRegressor:
    """Train a GradientBoostingRegressor with specified params."""
    model = GradientBoostingRegressor(**model_params, random_state=seed)
    model.fit(X_train, y_train)
    return model


def stable_config_hash(cfg: Dict) -> str:
    """Stable short hash for a JSON-serializable config dict."""
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def get_cache_paths(cache_dir: str, cfg_hash: str, city: Optional[str] = None) -> str:
    """Return model cache path for a city (local) or global model."""
    if city is None:
        return os.path.join(cache_dir, f"global_{cfg_hash}.joblib")
    safe_city = str(city).replace("/", "_")
    return os.path.join(cache_dir, f"local_{safe_city}_{cfg_hash}.joblib")


# ---------------------------------------------------------------------
# 4) Vectorized PIER + Router computation
# ---------------------------------------------------------------------
def predict_matrix(models: List[GradientBoostingRegressor], X: np.ndarray) -> np.ndarray:
    """Return prediction matrix of shape (n, P) using vectorized predict calls."""
    cols = [m.predict(X) for m in models]
    return np.column_stack(cols)


def router_mae_chunked(
    peer_models: List[GradientBoostingRegressor],
    w_hat: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    chunk_size: int = 200_000,
) -> float:
    """
    Compute router MAE on potentially huge test sets without building (n, P) matrix.
    y_pred = sum_k w_k * peer_k.predict(X)
    """
    n = X_test.shape[0]
    abs_sum = 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Xb = X_test[start:end]
        yb = y_test[start:end]

        pred = np.zeros(end - start, dtype=float)
        for wk, mk in zip(w_hat, peer_models):
            if wk == 0.0:
                continue
            pred += wk * mk.predict(Xb)

        abs_sum += np.abs(yb - pred).sum()

    return float(abs_sum / n)


# ---------------------------------------------------------------------
# 5) Main experiment
# ---------------------------------------------------------------------
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
    router_chunk_size: int = 200_000,
    output_filename: str = "exp14_multicity_tabular_summary.csv",
):
    print("=== Exp 14: Multi-city tabular PIER vs model removal cost (Plan-2 vectorized + cache) ===")

    # Fixed seed for reproducibility
    global_seed = 0
    rng = set_global_seed(global_seed)

    # Model hyperparameters (part of cache key)
    model_params = dict(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
    )

    # Cache setup
    cache_dir = os.path.join(DATA_DIR, "exp14_models")
    os.makedirs(cache_dir, exist_ok=True)

    # Define experiment config that determines cached models
    exp_cfg = dict(
        version="exp14_plan2_cache_v1",
        seed=global_seed,
        target_definition="y(t)=flow(t+1) within (city,detid) sorted by (day,interval)",
        feature_cols=["interval", "flow", "occ"],
        split="per-city 60/20/20 time-ordered",
        model="GradientBoostingRegressor",
        model_params=model_params,
    )
    cfg_hash = stable_config_hash(exp_cfg)
    print(f"Cache config hash: {cfg_hash}")
    cfg_path = os.path.join(cache_dir, f"config_{cfg_hash}.json")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as f:
            json.dump(exp_cfg, f, indent=2, sort_keys=True)

    # 1) Load raw CSV
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"data_csv not found: {data_csv}")

    print(f"Loading CSV from: {data_csv}")
    df = pd.read_csv(data_csv, low_memory=False)
    print(f"Loaded {len(df)} rows from {data_csv}")

    # 2) If raw UTD19, preprocess; else assume already prepared
    if (target_col not in df.columns) or (split_col not in df.columns):
        print(f"Input CSV missing '{target_col}' or '{split_col}'. Preprocessing as raw UTD19...")
        df = prepare_utd19_for_exp14(df_raw=df)

    # 3) Feature columns (fixed)
    feature_cols = [c for c in ["interval", "flow", "occ"] if c in df.columns]
    if len(feature_cols) < 3:
        raise ValueError(f"Expected feature columns ['interval','flow','occ'], got {feature_cols}")
    flow_idx = feature_cols.index("flow")

    print(f"Using feature columns: {feature_cols}")

    # 4) Select cities (reproducible)
    all_cities = sorted(df[city_col].dropna().unique().tolist())
    if max_cities is not None and max_cities < len(all_cities):
        chosen = rng.choice(all_cities, size=max_cities, replace=False)
        all_cities = sorted(chosen.tolist())
        print(f"Randomly selected {len(all_cities)} cities.")
    print(f"Using {len(all_cities)} cities: {all_cities}")

    # 5) Train/load per-city models
    city_models: Dict[str, GradientBoostingRegressor] = {}
    city_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    print("\nPreparing per-city datasets and training/loading models...")
    for city in all_cities:
        df_city = df[df[city_col] == city]

        df_train = df_city[df_city[split_col] == train_split_name]
        df_val = df_city[df_city[split_col] == val_split_name]
        df_test = df_city[df_city[split_col] == test_split_name]

        n_train, n_val, n_test = len(df_train), len(df_val), len(df_test)
        print(f"  City={city}: train={n_train}, val={n_val}, test={n_test}")

        if n_train < min_train_samples or n_val < min_val_samples or n_test < min_test_samples:
            print("    [SKIP] Not enough samples for this city.")
            continue

        X_train, y_train = build_xy(df_train, feature_cols, target_col)
        X_val, y_val = build_xy(df_val, feature_cols, target_col)
        X_test, y_test = build_xy(df_test, feature_cols, target_col)

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            print("    [SKIP] After filtering non-finite rows, some split is empty.")
            continue

        # Cache path for local model
        local_path = get_cache_paths(cache_dir, cfg_hash, city=city)

        if os.path.exists(local_path):
            model_c = joblib.load(local_path)
            print(f"    [LOAD] Local model from cache: {local_path}")
        else:
            model_c = train_regressor(X_train, y_train, seed=global_seed, model_params=model_params)
            joblib.dump(model_c, local_path)
            print(f"    [DUMP] Local model saved to: {local_path}")

        city_models[city] = model_c
        city_data[city] = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    if not city_models:
        print("No eligible cities found. Abort.")
        return

    # 6) Train/load global model
    global_path = get_cache_paths(cache_dir, cfg_hash, city=None)
    if os.path.exists(global_path):
        global_model = joblib.load(global_path)
        print(f"\n[LOAD] Global model from cache: {global_path}")
    else:
        print("\nTraining global model on all selected cities...")
        # Train global model on all selected cities' training data (capped at 500k samples)
        X_train_global = np.concatenate([city_data[c]["train"][0] for c in city_models.keys()], axis=0)
        y_train_global = np.concatenate([city_data[c]["train"][1] for c in city_models.keys()], axis=0)

        # Cap global training size for speed/stability
        max_global_train = 500_000
        n_global = X_train_global.shape[0]
        if n_global > max_global_train:
            # Reproducible subsampling
            idx = rng.choice(n_global, size=max_global_train, replace=False)
            X_train_global = X_train_global[idx]
            y_train_global = y_train_global[idx]
            print(f"[Global] Subsampled global train set: {n_global} -> {max_global_train}")
        else:
            print(f"[Global] Using full global train set: {n_global}")

        global_model = train_regressor(
            X_train_global, y_train_global, seed=global_seed, model_params=model_params
        )

        joblib.dump(global_model, global_path)
        print(f"[DUMP] Global model saved to: {global_path}")

    # 7) Evaluate each city
    rows = []
    model_names_sorted = sorted(city_models.keys())

    for city in model_names_sorted:
        print(f"\n=== City: {city} ===")

        X_train, y_train = city_data[city]["train"]
        X_val, y_val = city_data[city]["val"]
        X_test, y_test = city_data[city]["test"]

        local_model = city_models[city]

        # Training-set metrics
        local_train_mae = float(mean_absolute_error(y_train, local_model.predict(X_train)))
        global_train_mae = float(mean_absolute_error(y_train, global_model.predict(X_train)))
        print(f"  Train MAE (local):  {local_train_mae:.4f}")
        print(f"  Train MAE (global): {global_train_mae:.4f}")

        # Test-set baselines
        local_pred = local_model.predict(X_test)
        global_pred = global_model.predict(X_test)

        local_mae = float(mean_absolute_error(y_test, local_pred))
        global_mae = float(mean_absolute_error(y_test, global_pred))
        print(f"  Test  MAE (local):  {local_mae:.4f}")
        print(f"  Test  MAE (global): {global_mae:.4f}")

        # Persistence baseline: y_hat(t+1) = flow(t)
        persist_pred = X_test[:, flow_idx]
        persist_mae = float(mean_absolute_error(y_test, persist_pred))

        # Mean flow and MAE ratio
        mean_flow = float(np.mean(y_test))
        mae_ratio = float(local_mae / mean_flow) if mean_flow > 0 else float("nan")

        # Build peers: GLOBAL + other cities
        peer_names = ["GLOBAL"] + [c for c in model_names_sorted if c != city]
        peer_models = [global_model] + [city_models[c] for c in model_names_sorted if c != city]

        # Fit convex weights on validation subset (using target predictions, not labels)
        n_val = X_val.shape[0]
        fit_idx = np.arange(n_val)
        rng.shuffle(fit_idx)
        if n_val > max_fit_samples:
            fit_idx = fit_idx[:max_fit_samples]

        X_fit = X_val[fit_idx]
        y_t_fit = local_model.predict(X_fit).reshape(-1, 1)  # (n_fit, 1)
        Y_p_fit = predict_matrix(peer_models, X_fit)         # (n_fit, P)

        _, w_hat = DISCOSolver.solve_weights_and_distance(y_t_fit, Y_p_fit)
        w_hat = np.asarray(w_hat, dtype=float).flatten()

        topk = np.argsort(-w_hat)[:5]
        print(f"  [Fit] Top-5 peer weights: {[ (peer_names[i], float(w_hat[i])) for i in topk ]}")

        # PIER on test subset (target vs convex peers)
        n_test = X_test.shape[0]
        eval_idx = np.arange(n_test)
        rng.shuffle(eval_idx)
        if n_test > max_eval_samples:
            eval_idx = eval_idx[:max_eval_samples]

        X_eval = X_test[eval_idx]
        y_t_eval = local_model.predict(X_eval)              # (n_eval,)
        Y_p_eval = predict_matrix(peer_models, X_eval)      # (n_eval, P)
        y_mix_eval = Y_p_eval @ w_hat                       # (n_eval,)
        pier_city = float(np.mean(np.abs(y_t_eval - y_mix_eval)))

        # Router MAE on full test set (chunked weighted sum)
        router_mae = router_mae_chunked(
            peer_models=peer_models,
            w_hat=w_hat,
            X_test=X_test,
            y_test=y_test,
            chunk_size=router_chunk_size,
        )

        delta_global = float(global_mae - local_mae)
        delta_router = float(router_mae - local_mae)

        print(f"  Persist MAE: {persist_mae:.4f}")
        print(f"  MeanFlow(Test): {mean_flow:.4f} | MAE ratio: {mae_ratio:.4f}")
        print(f"  PIER (target vs convex peers): {pier_city:.4f}")
        print(f"  Router MAE (full test, chunked): {router_mae:.4f} (Δ vs local = {delta_router:.4f})")

        rows.append(
            {
                "City": city,
                # extra statistics
                "MeanFlow_Test": mean_flow,
                "Local_MAE_Ratio": mae_ratio,
                "Persist_MAE": persist_mae,
                # train metrics
                "Local_Train_MAE": local_train_mae,
                "Global_Train_MAE": global_train_mae,
                # test metrics
                "Local_MAE": local_mae,
                "Global_MAE": global_mae,
                "Router_MAE": router_mae,
                "Delta_Global": delta_global,
                "Delta_Router": delta_router,
                # PIER
                "PIER": pier_city,
                # sizes
                "NumTrain": int(X_train.shape[0]),
                "NumVal": int(X_val.shape[0]),
                "NumTest": int(X_test.shape[0]),
                "NumFitSamples": int(len(fit_idx)),
                "NumEvalSamples": int(len(eval_idx)),
                "NumPeers": int(len(peer_models)),
                # cache info (useful for reproducibility tracking)
                "CacheHash": cfg_hash,
            }
        )

    if not rows:
        print("No results to save.")
        return

    summary_df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT_DIR, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved summary to: {out_path}")
    print(f"Model cache dir: {cache_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 14 (Plan-2): Vectorized PIER vs model removal cost on UTD19, with model caching."
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default=DEFAULT_UTD19_CSV,
        help=f"Path to UTD19 CSV (default: {DEFAULT_UTD19_CSV}).",
    )
    parser.add_argument(
        "--max_cities",
        type=int,
        default=40,
        help="Max number of cities to use (random subset with fixed seed).",
    )
    parser.add_argument(
        "--max_fit_samples",
        type=int,
        default=2000,
        help="Max validation samples per city for fitting convex weights.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=2000,
        help="Max test samples per city for PIER estimation.",
    )
    parser.add_argument(
        "--router_chunk_size",
        type=int,
        default=200_000,
        help="Chunk size for full-test router MAE computation.",
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
        max_cities=args.max_cities,
        max_fit_samples=args.max_fit_samples,
        max_eval_samples=args.max_eval_samples,
        router_chunk_size=args.router_chunk_size,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
