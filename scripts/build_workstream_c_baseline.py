"""Evaluate historical-mean baselines for candidate forecast horizons."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bramex.config import get_project_paths  # noqa: E402
from bramex.targets import build_switch_targets  # noqa: E402


CANDIDATE_HORIZONS_SECONDS = (30, 60, 300)
FORECAST_STRIDE_SECONDS = 5


def load_targets(compact_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(compact_dir.glob("DFL-MAT-*.npz")):
        match_id = path.stem
        with np.load(path) as cached:
            for half in ("firstHalf", "secondHalf"):
                framerate = int(cached[f"{half}_framerate"][0])
                for horizon in CANDIDATE_HORIZONS_SECONDS:
                    target = build_switch_targets(
                        cached[f"{half}_possession"],
                        cached[f"{half}_ball_status"],
                        match_id=match_id,
                        half=half,
                        framerate=framerate,
                        horizon_seconds=horizon,
                        stride_seconds=FORECAST_STRIDE_SECONDS,
                    )
                    target["horizon_seconds"] = horizon
                    frames.append(target)
    if not frames:
        raise FileNotFoundError(
            f"No compact cache found in {compact_dir}. Run audit_dfl_data.py first."
        )
    return pd.concat(frames, ignore_index=True)


def leave_one_match_out_baseline(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, horizon_data in targets.groupby("horizon_seconds"):
        for held_out_match in sorted(horizon_data["match_id"].unique()):
            train = horizon_data[horizon_data["match_id"] != held_out_match]
            test = horizon_data[horizon_data["match_id"] == held_out_match]
            prediction = float(train["switches_future"].mean())
            rows.append(
                {
                    "horizon_seconds": horizon,
                    "held_out_match": held_out_match,
                    "train_mean_prediction": prediction,
                    "test_mean_observed": float(test["switches_future"].mean()),
                    "mae": mean_absolute_error(
                        test["switches_future"],
                        np.full(len(test), prediction),
                    ),
                    "n_test_origins": len(test),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    paths = get_project_paths(ROOT)
    targets = load_targets(paths.cache_dir / "compact")
    distribution = (
        targets.groupby("horizon_seconds")["switches_future"]
        .agg(n="size", mean="mean", std="std", median="median", maximum="max")
        .reset_index()
    )
    zero_share = (
        targets.assign(is_zero=targets["switches_future"].eq(0))
        .groupby("horizon_seconds")["is_zero"]
        .mean()
        .rename("zero_share")
        .reset_index()
    )
    distribution = distribution.merge(zero_share, on="horizon_seconds")
    scores = leave_one_match_out_baseline(targets)

    targets.to_csv(
        paths.cache_dir / "workstream_c_targets_candidates.csv.gz",
        index=False,
        compression="gzip",
    )
    distribution.to_csv(paths.cache_dir / "workstream_c_target_distribution.csv", index=False)
    scores.to_csv(paths.cache_dir / "workstream_c_baseline_lomo.csv", index=False)

    print("Target distributions")
    print(distribution.to_string(index=False))
    print("\nLeave-one-match-out historical-mean baseline")
    print(scores.to_string(index=False))
    print("\nMean MAE by horizon")
    print(scores.groupby("horizon_seconds")["mae"].mean().to_string())


if __name__ == "__main__":
    main()
