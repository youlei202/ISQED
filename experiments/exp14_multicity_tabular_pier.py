# experiments/exp14_multicity_tabular_pier.py
#
# Exp 14 (UTD19): Multi-city tabular ecosystem audit (PIER vs. model removal cost).
#
# Key idea (new support line vs NLP/CV experiments):
#   In a large multi-market setting (cities), we often train one model per market.
#   This experiment treats each city model as a member of an ecosystem and asks:
#     - Which city models are truly irreplaceable (high ecosystem substitution cost)?
#     - Which models are redundant (can be replaced by a convex mixture of peers)?
#     - Which models are "unique but unhealthy" (high PIER but worse than ecosystem/router)?
#
# Dataset:
#   UTD19 raw CSV ("utd19_u.csv") with columns:
#     city, detid, day, interval, flow, occ, (optional) error
#
# Forecasting setup:
#   For each (city, detid) time series sorted by (day, interval):
#     y(t) = flow(t + horizon_steps)
#   Features include:
#     - current flow/occ, interval
#     - periodic features (time-of-day and day-of-week)
#     - multi-lag features for flow/occ
#     - simple trend feature (flow_diff1)
#
# Split:
#   Per (city, detid) time-ordered split with a horizon-aware buffer:
#     train: first 60% (excluding last horizon buffer)
#     val:   next 20% (excluding last horizon buffer)
#     test:  last 20% (excluding last horizon buffer)
#
# Models:
#   Default: HistGradientBoostingRegressor (fast sklearn tree-based model).
#   Also supports: GradientBoostingRegressor.
#
# Efficiency:
#   - Vectorized model.predict for PIER fitting/evaluation.
#   - Chunked router evaluation to avoid building huge (n, P) matrices.
#   - Model caching (joblib) under DATA_DIR/exp14_models/.
#
# Outputs:
#   results/tables/exp14_multicity_tabular_summary.csv
#   Per city: local/global/router MAE, PIER, deltas, persistence baseline, and ratios.

import os
import sys
import json
import random
import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

import joblib

# Make local package importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from isqed.geometry import DISCOSolver


# -----------------------------------------------------------------------------
# 0) Defaults and determinism
# -----------------------------------------------------------------------------
DEFAULT_DATA_DIR = os.environ.get("UTD19_DIR", "/work3/leiyo/utd19")
DEFAULT_UTD19_CSV = os.path.join(DEFAULT_DATA_DIR, "utd19_u.csv")


def set_global_seed(seed: int = 0) -> np.random.RandomState:
    """Fix all relevant RNG seeds for reproducibility."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return rng


def stable_config_hash(cfg: Dict) -> str:
    """Stable short hash for a JSON-serializable config dict."""
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def get_cache_path(cache_dir: str, cfg_hash: str, city: Optional[str] = None) -> str:
    """Return model cache path for a city (local) or global model."""
    if city is None:
        return os.path.join(cache_dir, f"global_{cfg_hash}.joblib")
    safe_city = str(city).replace("/", "_")
    return os.path.join(cache_dir, f"local_{safe_city}_{cfg_hash}.joblib")


# -----------------------------------------------------------------------------
# 1) Data preprocessing and feature engineering for UTD19
# -----------------------------------------------------------------------------
def _cap_df_rows(df: pd.DataFrame, max_rows: Optional[int], rng: np.random.RandomState) -> pd.DataFrame:
    """Randomly cap a dataframe to at most max_rows rows (without replacement)."""
    if max_rows is None:
        return df
    if len(df) <= max_rows:
        return df
    idx = rng.choice(len(df), size=max_rows, replace=False)
    return df.iloc[idx].copy()


def _filter_recent_days_per_city(
    df: pd.DataFrame,
    city_col: str,
    day_col: str,
    max_days_per_city: Optional[int],
) -> pd.DataFrame:
    """
    Keep only the most recent `max_days_per_city` days for each city.
    Assumes day_col is datetime64[ns].
    """
    if max_days_per_city is None:
        return df
    if max_days_per_city <= 0:
        raise ValueError("max_days_per_city must be a positive integer or None.")

    max_day = df.groupby(city_col)[day_col].transform("max")
    cutoff = max_day - pd.to_timedelta(max_days_per_city - 1, unit="D")
    return df[df[day_col] >= cutoff].copy()


def _filter_top_detectors_per_city(
    df: pd.DataFrame,
    city_col: str,
    detid_col: str,
    max_detectors_per_city: Optional[int],
    min_points_per_detector: int,
) -> pd.DataFrame:
    """
    Keep at most `max_detectors_per_city` detectors per city,
    selecting detectors with the most observations.
    Also drop detectors with < min_points_per_detector observations.
    """
    if max_detectors_per_city is None and (min_points_per_detector <= 1):
        return df

    counts = (
        df.groupby([city_col, detid_col], sort=False)
        .size()
        .reset_index(name="_n")
    )
    counts = counts[counts["_n"] >= int(min_points_per_detector)].copy()

    if max_detectors_per_city is not None:
        keep_pairs = []
        for city, sub in counts.groupby(city_col, sort=False):
            sub_sorted = sub.sort_values("_n", ascending=False)
            keep_pairs.append(sub_sorted.head(int(max_detectors_per_city))[[city_col, detid_col]])
        keep_pairs_df = pd.concat(keep_pairs, axis=0, ignore_index=True)
    else:
        keep_pairs_df = counts[[city_col, detid_col]].copy()

    df = df.merge(keep_pairs_df, on=[city_col, detid_col], how="inner")
    return df


def prepare_utd19_for_exp14(
    df_raw: pd.DataFrame,
    rng: np.random.RandomState,
    city_col: str = "city",
    detid_col: str = "detid",
    day_col: str = "day",
    interval_col: str = "interval",
    flow_col: str = "flow",
    occ_col: str = "occ",
    error_col: str = "error",
    horizon_steps: int = 12,
    lag_steps: Optional[List[int]] = None,
    filter_error: bool = True,
    max_days_per_city: Optional[int] = None,
    max_detectors_per_city: Optional[int] = None,
    min_points_per_detector: int = 2000,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert raw UTD19 measurements to a single table usable by Exp14.

    Target:
      y(t) = flow(t + horizon_steps) within each (city, detid) sequence ordered
             by (day, interval).

    Features:
      - interval, flow, occ
      - periodic time features:
          tod_sin, tod_cos, dow_sin, dow_cos, is_weekend
      - lag features:
          flow_lag{k}, occ_lag{k} for k in lag_steps
      - trend:
          flow_diff1 = flow - flow_lag1

    Splitting:
      Per (city, detid) time-ordered 60/20/20 split with horizon buffer.

    Returns:
      - processed dataframe with columns [features..., y, split, city, detid, day]
      - feature_cols list
    """
    if lag_steps is None:
        lag_steps = [1, 2, 3, 6, 12]

    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be a positive integer.")
    if any(int(l) <= 0 for l in lag_steps):
        raise ValueError("lag_steps must contain positive integers only.")

    required = [city_col, detid_col, day_col, interval_col, flow_col, occ_col]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"UTD19 preprocessing missing required columns: {missing}")

    use_cols = required + ([error_col] if error_col in df_raw.columns else [])
    df = df_raw[use_cols].copy()

    # Parse day to datetime
    df[day_col] = pd.to_datetime(df[day_col], errors="coerce")
    df = df.dropna(subset=[day_col])

    # Coerce numerics
    for c in [interval_col, flow_col, occ_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace inf with NaN and drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=[interval_col, flow_col, occ_col])

    # Optional error filtering
    if filter_error and (error_col in df.columns):
        df[error_col] = pd.to_numeric(df[error_col], errors="coerce").fillna(0.0)
        df = df[df[error_col] == 0.0].copy()

    # Optional reductions (days and detectors)
    df = _filter_recent_days_per_city(df, city_col=city_col, day_col=day_col, max_days_per_city=max_days_per_city)
    df = _filter_top_detectors_per_city(
        df,
        city_col=city_col,
        detid_col=detid_col,
        max_detectors_per_city=max_detectors_per_city,
        min_points_per_detector=min_points_per_detector,
    )

    # Sort for time order
    df = df.sort_values([city_col, detid_col, day_col, interval_col])

    # Calendar features
    df["dow"] = df[day_col].dt.dayofweek.astype(np.int8)
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    df["dow_sin"] = np.sin(2.0 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["dow"] / 7.0)

    # Time-of-day features from interval:
    # Normalize by per-city max interval + 1 to be robust across cities.
    interval_max = df.groupby(city_col)[interval_col].transform("max")
    interval_period = (interval_max + 1.0).replace(0.0, 1.0)
    tod_frac = (df[interval_col] / interval_period).clip(0.0, 1.0)
    df["tod_sin"] = np.sin(2.0 * np.pi * tod_frac)
    df["tod_cos"] = np.cos(2.0 * np.pi * tod_frac)

    # Group object for detector-level lagging
    g = df.groupby([city_col, detid_col], sort=False)

    # Lag features
    lag_steps_unique = sorted(set(int(l) for l in lag_steps))
    for lag in lag_steps_unique:
        df[f"flow_lag{lag}"] = g[flow_col].shift(lag)
        df[f"occ_lag{lag}"] = g[occ_col].shift(lag)

    # Trend feature
    if 1 in lag_steps_unique:
        df["flow_diff1"] = df[flow_col] - df["flow_lag1"]
    else:
        df["flow_lag1"] = g[flow_col].shift(1)
        df["flow_diff1"] = df[flow_col] - df["flow_lag1"]
        lag_steps_unique = sorted(set(lag_steps_unique + [1]))

    # Horizon target
    df["y"] = g[flow_col].shift(-int(horizon_steps))

    # Horizon-aware per-(city,detid) split (60/20/20)
    df["_t_idx"] = g.cumcount()
    df["_t_len"] = g[flow_col].transform("size")

    train_end = (0.6 * df["_t_len"]).astype(int)
    val_end = (0.8 * df["_t_len"]).astype(int)

    train_mask = df["_t_idx"] < (train_end - horizon_steps)
    val_mask = (df["_t_idx"] >= train_end) & (df["_t_idx"] < (val_end - horizon_steps))
    test_mask = (df["_t_idx"] >= val_end) & (df["_t_idx"] < (df["_t_len"] - horizon_steps))

    df["split"] = None
    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "val"
    df.loc[test_mask, "split"] = "test"

    # Define feature columns
    feature_cols = (
        [interval_col, flow_col, occ_col]
        + ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "flow_diff1"]
        + [f"flow_lag{l}" for l in lag_steps_unique]
        + [f"occ_lag{l}" for l in lag_steps_unique]
    )

    before = len(df)
    df = df.dropna(subset=feature_cols + ["y", "split"]).copy()
    after = len(df)
    print(f"UTD19: after feature/target engineering and split assignment: {before} -> {after} rows")

    df = df.drop(columns=["_t_idx", "_t_len"], errors="ignore")
    return df, feature_cols


# -----------------------------------------------------------------------------
# 2) Feature matrix builder (finite-safe)
# -----------------------------------------------------------------------------
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
        bad = int((~mask).sum())
        print(f"      [WARN] Dropping {bad} rows with non-finite X or y in build_xy.")
        Xv = Xv[mask]
        yv = yv[mask]

    return Xv, yv


# -----------------------------------------------------------------------------
# 3) Model training + caching
# -----------------------------------------------------------------------------
def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_type: str,
    model_params: Dict,
):
    """
    Train a regression model with specified params.
    Supported model_type:
      - "histgb": HistGradientBoostingRegressor
      - "gbrt":   GradientBoostingRegressor
    """
    mt = str(model_type).lower().strip()
    if mt == "histgb":
        model = HistGradientBoostingRegressor(**model_params, random_state=seed)
    elif mt == "gbrt":
        model = GradientBoostingRegressor(**model_params, random_state=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type} (use 'histgb' or 'gbrt')")

    model.fit(X_train, y_train)
    return model


# -----------------------------------------------------------------------------
# 4) Vectorized PIER + Router computation
# -----------------------------------------------------------------------------
def predict_matrix(models: List, X: np.ndarray) -> np.ndarray:
    """Return prediction matrix of shape (n, P) using vectorized predict calls."""
    cols = [m.predict(X) for m in models]
    return np.column_stack(cols)


def router_mae_chunked(
    peer_models: List,
    w_hat: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    chunk_size: int = 200_000,
    weight_tol: float = 1e-10,
) -> float:
    """
    Compute router MAE on potentially huge test sets without building an (n, P) matrix.
    Router prediction: y_pred = sum_k w_k * peer_k.predict(X).

    Only peers with weights > weight_tol are evaluated for efficiency.
    """
    n = int(X_test.shape[0])
    if n == 0:
        return float("nan")

    w_hat = np.asarray(w_hat, dtype=float).flatten()
    keep = np.where(w_hat > float(weight_tol))[0]
    if keep.size == 0:
        # Degenerate case: no positive weights. Return MAE of zeros as a fallback.
        return float(np.mean(np.abs(y_test - 0.0)))

    peer_models_kept = [peer_models[i] for i in keep.tolist()]
    w_kept = w_hat[keep]

    abs_sum = 0.0
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        Xb = X_test[start:end]
        yb = y_test[start:end]

        pred = np.zeros(end - start, dtype=float)
        for wk, mk in zip(w_kept, peer_models_kept):
            pred += wk * mk.predict(Xb)

        abs_sum += np.abs(yb - pred).sum()

    return float(abs_sum / n)


# -----------------------------------------------------------------------------
# 5) Main experiment
# -----------------------------------------------------------------------------
def run_multicity_tabular_experiment(
    data_csv: str,
    data_dir: str,
    city_col: str = "city",
    detid_col: str = "detid",
    day_col: str = "day",
    interval_col: str = "interval",
    flow_col: str = "flow",
    occ_col: str = "occ",
    error_col: str = "error",
    target_col: str = "y",
    split_col: str = "split",
    max_cities: Optional[int] = 40,
    min_train_samples: int = 5000,
    min_val_samples: int = 2000,
    min_test_samples: int = 2000,
    max_train_rows_per_city: Optional[int] = 300_000,
    max_val_rows_per_city: Optional[int] = 200_000,
    max_test_rows_per_city: Optional[int] = 200_000,
    max_fit_samples: int = 4000,
    max_eval_samples: int = 4000,
    router_chunk_size: int = 200_000,
    horizon_steps: int = 12,
    lag_steps: Optional[List[int]] = None,
    filter_error: bool = True,
    max_days_per_city: Optional[int] = 60,
    max_detectors_per_city: Optional[int] = 200,
    min_points_per_detector: int = 5000,
    model_type: str = "histgb",
    output_filename: str = "exp14_multicity_tabular_summary.csv",
):
    print("=== Exp 14: Multi-city tabular PIER vs model removal cost (UTD19, forecasting-aware) ===")

    # Fixed seed for reproducibility
    global_seed = 0
    rng = set_global_seed(global_seed)

    os.makedirs(data_dir, exist_ok=True)
    cache_dir = os.path.join(data_dir, "exp14_models")
    os.makedirs(cache_dir, exist_ok=True)

    if lag_steps is None:
        lag_steps = [1, 2, 3, 6, 12]

    # Model hyperparameters (part of cache key)
    # Tuned for speed and reasonable performance on large tabular time-series features.
    if str(model_type).lower().strip() == "histgb":
        model_params = dict(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=0.0,
        )
    else:
        # GBRT is slower for large datasets; keep it as an optional fallback.
        model_params = dict(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
        )

    # Define experiment config that determines cached models
    exp_cfg = dict(
        version="exp14_utd19_forecasting_v2",
        seed=global_seed,
        data_csv=os.path.abspath(data_csv),
        city_col=city_col,
        detid_col=detid_col,
        day_col=day_col,
        interval_col=interval_col,
        flow_col=flow_col,
        occ_col=occ_col,
        error_col=error_col,
        target_definition=f"y(t)=flow(t+{int(horizon_steps)}) within (city,detid) sorted by (day,interval)",
        split="per-(city,detid) 60/20/20 time-ordered with horizon buffer",
        horizon_steps=int(horizon_steps),
        lag_steps=[int(x) for x in lag_steps],
        filter_error=bool(filter_error),
        max_days_per_city=max_days_per_city,
        max_detectors_per_city=max_detectors_per_city,
        min_points_per_detector=int(min_points_per_detector),
        max_train_rows_per_city=max_train_rows_per_city,
        max_val_rows_per_city=max_val_rows_per_city,
        max_test_rows_per_city=max_test_rows_per_city,
        model_type=str(model_type),
        model_params=model_params,
        max_cities=max_cities,
    )
    cfg_hash = stable_config_hash(exp_cfg)
    print(f"Cache config hash: {cfg_hash}")

    cfg_path = os.path.join(cache_dir, f"config_{cfg_hash}.json")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as f:
            json.dump(exp_cfg, f, indent=2, sort_keys=True)

    # 1) Load raw CSV with minimal columns
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"data_csv not found: {data_csv}")

    usecols = [city_col, detid_col, day_col, interval_col, flow_col, occ_col]
    if error_col is not None:
        usecols.append(error_col)

    print(f"Loading CSV from: {data_csv}")
    df = pd.read_csv(data_csv, low_memory=False, usecols=lambda c: c in set(usecols + [target_col, split_col]))
    print(f"Loaded {len(df)} rows from {data_csv}")

    # 2) Select cities early (before heavy feature engineering)
    if city_col not in df.columns:
        raise ValueError(f"Column '{city_col}' not found in the input CSV.")

    all_cities_raw = sorted(df[city_col].dropna().unique().tolist())
    if max_cities is not None and max_cities < len(all_cities_raw):
        chosen = rng.choice(all_cities_raw, size=int(max_cities), replace=False)
        chosen_cities = sorted(chosen.tolist())
        df = df[df[city_col].isin(chosen_cities)].copy()
        print(f"Randomly selected {len(chosen_cities)} cities (seed={global_seed}).")
    else:
        chosen_cities = all_cities_raw

    print(f"Using {len(chosen_cities)} cities.")

    # 3) If raw UTD19, preprocess; else assume already prepared
    feature_cols: List[str]
    if (target_col not in df.columns) or (split_col not in df.columns):
        print(f"Input CSV missing '{target_col}' or '{split_col}'. Preprocessing as raw UTD19...")
        df, feature_cols = prepare_utd19_for_exp14(
            df_raw=df,
            rng=rng,
            city_col=city_col,
            detid_col=detid_col,
            day_col=day_col,
            interval_col=interval_col,
            flow_col=flow_col,
            occ_col=occ_col,
            error_col=error_col,
            horizon_steps=int(horizon_steps),
            lag_steps=[int(x) for x in lag_steps],
            filter_error=bool(filter_error),
            max_days_per_city=max_days_per_city,
            max_detectors_per_city=max_detectors_per_city,
            min_points_per_detector=int(min_points_per_detector),
        )
    else:
        # Prepared input must have engineered features; we infer feature columns minimally.
        print("Detected prepared CSV (has 'y' and 'split'). Using it directly.")
        # Try to reconstruct feature columns from expected names.
        base = [interval_col, flow_col, occ_col, "tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "flow_diff1"]
        lag_steps_unique = sorted(set(int(x) for x in lag_steps))
        feature_cols = base + [f"flow_lag{l}" for l in lag_steps_unique] + [f"occ_lag{l}" for l in lag_steps_unique]

        missing_feats = [c for c in feature_cols if c not in df.columns]
        if missing_feats:
            raise ValueError(
                "Prepared CSV is missing expected engineered feature columns. "
                f"Missing: {missing_feats[:10]}{'...' if len(missing_feats) > 10 else ''}"
            )

    # 4) Recompute the city list after preprocessing and filtering
    all_cities = sorted(df[city_col].dropna().unique().tolist())
    print(f"Eligible cities after preprocessing: {len(all_cities)}")

    # 5) Train/load per-city models
    city_models: Dict[str, object] = {}
    city_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    print("\nPreparing per-city datasets and training/loading models...")
    for city in all_cities:
        df_city = df[df[city_col] == city]

        df_train = df_city[df_city[split_col] == "train"]
        df_val = df_city[df_city[split_col] == "val"]
        df_test = df_city[df_city[split_col] == "test"]

        # Optional row caps for speed
        df_train = _cap_df_rows(df_train, max_train_rows_per_city, rng)
        df_val = _cap_df_rows(df_val, max_val_rows_per_city, rng)
        df_test = _cap_df_rows(df_test, max_test_rows_per_city, rng)

        n_train, n_val, n_test = len(df_train), len(df_val), len(df_test)
        print(f"  City={city}: train={n_train}, val={n_val}, test={n_test}")

        if n_train < int(min_train_samples) or n_val < int(min_val_samples) or n_test < int(min_test_samples):
            print("    [SKIP] Not enough samples for this city.")
            continue

        X_train, y_train = build_xy(df_train, feature_cols, target_col=target_col)
        X_val, y_val = build_xy(df_val, feature_cols, target_col=target_col)
        X_test, y_test = build_xy(df_test, feature_cols, target_col=target_col)

        if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
            print("    [SKIP] After filtering non-finite rows, some split is empty.")
            continue

        local_path = get_cache_path(cache_dir, cfg_hash, city=city)
        if os.path.exists(local_path):
            model_c = joblib.load(local_path)
            print(f"    [LOAD] Local model from cache: {local_path}")
        else:
            model_c = train_regressor(
                X_train=X_train,
                y_train=y_train,
                seed=global_seed,
                model_type=model_type,
                model_params=model_params,
            )
            joblib.dump(model_c, local_path)
            print(f"    [DUMP] Local model saved to: {local_path}")

        city_models[city] = model_c
        city_data[city] = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    if len(city_models) < 5:
        raise RuntimeError(
            f"Too few eligible cities after filtering/training: {len(city_models)}. "
            "Try lowering min_* thresholds or increasing max_days/max_detectors."
        )

    model_names_sorted = sorted(city_models.keys())
    print(f"\nFinal number of city models: {len(model_names_sorted)}")

    # 6) Train/load global model
    global_path = get_cache_path(cache_dir, cfg_hash, city=None)
    if os.path.exists(global_path):
        global_model = joblib.load(global_path)
        print(f"\n[LOAD] Global model from cache: {global_path}")
    else:
        print("\nTraining global model on all selected cities...")

        X_train_global = np.concatenate([city_data[c]["train"][0] for c in model_names_sorted], axis=0)
        y_train_global = np.concatenate([city_data[c]["train"][1] for c in model_names_sorted], axis=0)

        # Cap global training size for speed/stability
        max_global_train = 800_000
        n_global = int(X_train_global.shape[0])
        if n_global > max_global_train:
            idx = rng.choice(n_global, size=max_global_train, replace=False)
            X_train_global = X_train_global[idx]
            y_train_global = y_train_global[idx]
            print(f"[Global] Subsampled global train set: {n_global} -> {max_global_train}")
        else:
            print(f"[Global] Using full global train set: {n_global}")

        global_model = train_regressor(
            X_train=X_train_global,
            y_train=y_train_global,
            seed=global_seed,
            model_type=model_type,
            model_params=model_params,
        )
        joblib.dump(global_model, global_path)
        print(f"[DUMP] Global model saved to: {global_path}")

    # Locate the index of current-flow feature for persistence baseline
    if flow_col not in feature_cols:
        raise ValueError(f"Flow column '{flow_col}' not found in feature_cols.")
    flow_idx = feature_cols.index(flow_col)

    # 7) Evaluate each city
    rows = []
    for city in model_names_sorted:
        print(f"\n=== City: {city} ===")

        X_train, y_train = city_data[city]["train"]
        X_val, y_val = city_data[city]["val"]
        X_test, y_test = city_data[city]["test"]

        local_model = city_models[city]

        # Train-set diagnostics
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

        # Naive persistence baseline: y_hat(t+h) = flow(t)
        persist_pred = X_test[:, flow_idx]
        persist_mae = float(mean_absolute_error(y_test, persist_pred))

        mean_flow = float(np.mean(y_test))
        local_mae_ratio = float(local_mae / mean_flow) if mean_flow > 0 else float("nan")

        skill_vs_persist = (
            float((persist_mae - local_mae) / persist_mae) if persist_mae > 0 else float("nan")
        )

        # Peer set: GLOBAL + other cities
        peer_names = ["GLOBAL"] + [c for c in model_names_sorted if c != city]
        peer_models = [global_model] + [city_models[c] for c in model_names_sorted if c != city]

        # Fit convex weights on validation subset (target outputs, not labels)
        n_val = int(X_val.shape[0])
        fit_idx = np.arange(n_val)
        rng.shuffle(fit_idx)
        if n_val > int(max_fit_samples):
            fit_idx = fit_idx[: int(max_fit_samples)]

        X_fit = X_val[fit_idx]
        y_t_fit = local_model.predict(X_fit).reshape(-1, 1)  # (n_fit, 1)
        Y_p_fit = predict_matrix(peer_models, X_fit)         # (n_fit, P)

        _, w_hat = DISCOSolver.solve_weights_and_distance(y_t_fit, Y_p_fit)
        w_hat = np.asarray(w_hat, dtype=float).flatten()

        topk = np.argsort(-w_hat)[:5]
        print(f"  [Fit] Top-5 peer weights: {[ (peer_names[i], float(w_hat[i])) for i in topk ]}")

        # PIER on test subset (target vs convex peers)
        n_test = int(X_test.shape[0])
        eval_idx = np.arange(n_test)
        rng.shuffle(eval_idx)
        if n_test > int(max_eval_samples):
            eval_idx = eval_idx[: int(max_eval_samples)]

        X_eval = X_test[eval_idx]
        y_t_eval = local_model.predict(X_eval)          # (n_eval,)
        Y_p_eval = predict_matrix(peer_models, X_eval)  # (n_eval, P)
        y_mix_eval = Y_p_eval @ w_hat
        pier_city = float(np.mean(np.abs(y_t_eval - y_mix_eval)))

        # Router MAE on full test (or capped test) set, chunked
        router_mae = router_mae_chunked(
            peer_models=peer_models,
            w_hat=w_hat,
            X_test=X_test,
            y_test=y_test,
            chunk_size=int(router_chunk_size),
            weight_tol=1e-10,
        )

        delta_global = float(global_mae - local_mae)
        delta_router = float(router_mae - local_mae)

        print(f"  Persist MAE: {persist_mae:.4f} | Skill vs persist: {skill_vs_persist:.4f}")
        print(f"  MeanFlow(Test): {mean_flow:.4f} | Local MAE ratio: {local_mae_ratio:.4f}")
        print(f"  PIER (target vs convex peers): {pier_city:.4f}")
        print(f"  Router MAE (chunked): {router_mae:.4f} (Δ vs local = {delta_router:.4f})")

        rows.append(
            {
                "City": city,
                "HorizonSteps": int(horizon_steps),
                # summary statistics
                "MeanFlow_Test": mean_flow,
                "Local_MAE_Ratio": local_mae_ratio,
                "Persist_MAE": persist_mae,
                "Skill_vs_Persist": skill_vs_persist,
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
                # cache / reproducibility
                "CacheHash": cfg_hash,
                "ModelType": str(model_type),
            }
        )

    if not rows:
        raise RuntimeError("No results to save (unexpected).")

    summary_df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT_DIR, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    summary_df.to_csv(out_path, index=False)

    print(f"\nSaved summary to: {out_path}")
    print(f"Model cache dir: {cache_dir}")
    print("Done.")


def _parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated int list, e.g. '1,2,3,6,12'."""
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 14: UTD19 multi-city forecasting-aware PIER vs replacement cost (with caching)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Directory for UTD19 files and model cache (default: {DEFAULT_DATA_DIR}).",
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
        "--horizon_steps",
        type=int,
        default=12,
        help="Forecast horizon in steps (y(t) = flow(t + horizon_steps)).",
    )
    parser.add_argument(
        "--lag_steps",
        type=str,
        default="1,2,3,6,12",
        help="Comma-separated lag steps for features (default: 1,2,3,6,12).",
    )
    parser.add_argument(
        "--filter_error",
        action="store_true",
        help="If set, drop rows with error != 0 (when 'error' column exists).",
    )
    parser.add_argument(
        "--max_days_per_city",
        type=int,
        default=60,
        help="Keep only the most recent N days per city (None disables).",
    )
    parser.add_argument(
        "--max_detectors_per_city",
        type=int,
        default=200,
        help="Keep only the top-K detectors per city by observation count (None disables).",
    )
    parser.add_argument(
        "--min_points_per_detector",
        type=int,
        default=5000,
        help="Drop detectors with fewer than this many raw observations (before lags/target).",
    )
    parser.add_argument(
        "--min_train_samples",
        type=int,
        default=5000,
        help="Minimum number of train samples per city after feature engineering.",
    )
    parser.add_argument(
        "--min_val_samples",
        type=int,
        default=2000,
        help="Minimum number of val samples per city after feature engineering.",
    )
    parser.add_argument(
        "--min_test_samples",
        type=int,
        default=2000,
        help="Minimum number of test samples per city after feature engineering.",
    )
    parser.add_argument(
        "--max_train_rows_per_city",
        type=int,
        default=300_000,
        help="Cap train rows per city for speed (None disables).",
    )
    parser.add_argument(
        "--max_val_rows_per_city",
        type=int,
        default=200_000,
        help="Cap val rows per city for speed (None disables).",
    )
    parser.add_argument(
        "--max_test_rows_per_city",
        type=int,
        default=200_000,
        help="Cap test rows per city for speed (None disables).",
    )
    parser.add_argument(
        "--max_fit_samples",
        type=int,
        default=4000,
        help="Max validation samples per city for fitting convex weights (P_fit).",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=4000,
        help="Max test samples per city for PIER estimation (P_eval).",
    )
    parser.add_argument(
        "--router_chunk_size",
        type=int,
        default=200_000,
        help="Chunk size for router MAE computation.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="histgb",
        choices=["histgb", "gbrt"],
        help="Regression model family used for local/global models.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="exp14_multicity_tabular_summary.csv",
        help="Output CSV filename under results/tables/.",
    )

    args = parser.parse_args()

    lag_steps = _parse_int_list(args.lag_steps)

    run_multicity_tabular_experiment(
        data_csv=args.data_csv,
        data_dir=args.data_dir,
        max_cities=args.max_cities,
        horizon_steps=args.horizon_steps,
        lag_steps=lag_steps,
        filter_error=bool(args.filter_error),
        max_days_per_city=args.max_days_per_city,
        max_detectors_per_city=args.max_detectors_per_city,
        min_points_per_detector=args.min_points_per_detector,
        min_train_samples=args.min_train_samples,
        min_val_samples=args.min_val_samples,
        min_test_samples=args.min_test_samples,
        max_train_rows_per_city=args.max_train_rows_per_city,
        max_val_rows_per_city=args.max_val_rows_per_city,
        max_test_rows_per_city=args.max_test_rows_per_city,
        max_fit_samples=args.max_fit_samples,
        max_eval_samples=args.max_eval_samples,
        router_chunk_size=args.router_chunk_size,
        model_type=args.model_type,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
