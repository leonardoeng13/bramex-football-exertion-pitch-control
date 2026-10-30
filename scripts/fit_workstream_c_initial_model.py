"""Evaluate fixed, leakage-safe first models across complete held-out matches."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep this small first-pass benchmark deterministic and compatible with restricted runners.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bramex.config import get_project_paths  # noqa: E402
from bramex.features import build_switch_feature_table  # noqa: E402


CANDIDATE_HORIZONS_SECONDS = (30, 60, 300)
FORECAST_STRIDE_SECONDS = 5
RANDOM_SEED = 42


def load_feature_table(compact_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(compact_dir.glob("DFL-MAT-*.npz")):
        match_id = path.stem
        with np.load(path) as cached:
            for half in ("firstHalf", "secondHalf"):
                framerate = int(cached[f"{half}_framerate"][0])
                for horizon in CANDIDATE_HORIZONS_SECONDS:
                    frame = build_switch_feature_table(
                        cached[f"{half}_possession"],
                        cached[f"{half}_ball_status"],
                        cached[f"{half}_ball_xy"],
                        match_id=match_id,
                        half=half,
                        framerate=framerate,
                        horizon_seconds=horizon,
                        stride_seconds=FORECAST_STRIDE_SECONDS,
                    )
                    frame["horizon_seconds"] = horizon
                    frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No compact cache found in {compact_dir}. Run audit_dfl_data.py first."
        )
    return pd.concat(frames, ignore_index=True)


def model_factories():
    return {
        "poisson": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PoissonRegressor(alpha=1.0, max_iter=1000),
        ),
        "hist_gradient_boosting": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=RANDOM_SEED,
            ),
        ),
    }


def evaluate(feature_table: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "match_id",
        "half",
        "frame",
        "gameclock_s",
        "switches_future",
        "horizon_seconds",
    }
    feature_columns = [column for column in feature_table.columns if column not in excluded]
    rows = []
    for horizon, horizon_data in feature_table.groupby("horizon_seconds"):
        for held_out_match in sorted(horizon_data["match_id"].unique()):
            train = horizon_data[horizon_data["match_id"] != held_out_match]
            test = horizon_data[horizon_data["match_id"] == held_out_match]
            y_train = train["switches_future"].to_numpy()
            y_test = test["switches_future"].to_numpy()

            baseline_prediction = np.full(len(test), y_train.mean())
            rows.append(
                {
                    "horizon_seconds": horizon,
                    "held_out_match": held_out_match,
                    "model": "historical_mean",
                    "mae": mean_absolute_error(y_test, baseline_prediction),
                    "n_test_origins": len(test),
                }
            )

            for model_name, factory in model_factories().items():
                model = factory()
                model.fit(train[feature_columns], y_train)
                prediction = np.clip(model.predict(test[feature_columns]), 0, None)
                rows.append(
                    {
                        "horizon_seconds": horizon,
                        "held_out_match": held_out_match,
                        "model": model_name,
                        "mae": mean_absolute_error(y_test, prediction),
                        "n_test_origins": len(test),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    paths = get_project_paths(ROOT)
    feature_table = load_feature_table(paths.cache_dir / "compact")
    scores = evaluate(feature_table)
    summary = (
        scores.groupby(["horizon_seconds", "model"])["mae"]
        .agg(mean_mae="mean", std_across_matches="std")
        .reset_index()
    )
    scores.to_csv(paths.cache_dir / "workstream_c_initial_model_lomo.csv", index=False)
    summary.to_csv(paths.cache_dir / "workstream_c_initial_model_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
