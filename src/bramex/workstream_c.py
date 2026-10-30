"""Workstream C built on the team's shared tables.

Targets are counted from the original 25 Hz possession and ball-status flags.
Predictors come from the shared 5 Hz tables and use only the forecast origin or
earlier samples. This keeps fast possession changes while allowing Workstreams
A and B to contribute features through one common pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

# Keep the reproducible benchmark compatible with restricted runners.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, cache_dir
from .data import load_raw_match
from .targets import possession_switch_mask


DEFAULT_HORIZONS_SECONDS = (30, 60, 300)
DEFAULT_STRIDE_SECONDS = 5
RANDOM_SEED = 42
WORKSTREAM_B_FEATURE_COLUMNS = (
    "B_current_pitch_share",
    "B_current_zone_share",
    *(f"B_{name}_{seconds}s" for seconds in (60, 300) for name in (
        "possession_share",
        "pitch_share",
        "zone_share",
        "control_minus_possession",
        "progression_m",
        "seq_start_zone",
    )),
)


def _past_sum(values: np.ndarray, origins: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    starts = np.maximum(0, origins - window + 1)
    return cumulative[origins + 1] - cumulative[starts]


def _possession_duration_frames(possession: np.ndarray, ball_status: np.ndarray) -> np.ndarray:
    """Consecutive live frames held by the current side, including the origin."""
    possession = np.asarray(possession)
    ball_status = np.asarray(ball_status)
    duration = np.zeros(len(possession), dtype=np.int32)
    for i in range(len(possession)):
        if ball_status[i] != 1 or possession[i] not in (1, 2):
            continue
        if i == 0 or ball_status[i - 1] != 1 or possession[i] != possession[i - 1]:
            duration[i] = 1
        else:
            duration[i] = duration[i - 1] + 1
    return duration


def _rolling(values: np.ndarray, window: int, operation: str) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    roller = series.rolling(window=window, min_periods=1)
    if operation == "sum":
        return roller.sum().to_numpy()
    if operation == "mean":
        return roller.mean().to_numpy()
    raise ValueError(f"Unsupported rolling operation: {operation}")


def _event_arrays(events: pd.DataFrame, half: str, n_samples: int) -> dict[str, np.ndarray]:
    arrays = {
        "shots": np.zeros(n_samples, dtype=float),
        "fouls": np.zeros(n_samples, dtype=float),
        "tackles": np.zeros(n_samples, dtype=float),
    }
    half_events = events[events["half"] == half]
    definitions = {"shots": "is_shot", "fouls": "is_foul", "tackles": "is_tackle"}
    for name, column in definitions.items():
        selected = half_events[half_events[column].astype(bool)]
        if selected.empty:
            continue
        counts = selected["sample"].dropna().astype(int).value_counts()
        valid = counts.index[(counts.index >= 0) & (counts.index < n_samples)]
        arrays[name][valid] = counts.loc[valid].to_numpy(dtype=float)
    return arrays


def _workstream_b_origin_features(
    table: pd.DataFrame,
    origins: np.ndarray,
    current_home: np.ndarray,
) -> pd.DataFrame:
    """Translate Monica's Workstream B signals to past-only forecast origins.

    Monica's analysis establishes possession share, pitch control, ball-zone
    control, progression and control at possession starts as the relevant team
    context. Her published integration table is team-by-minute and includes the
    complete current minute. Workstream C predicts every five seconds, so using
    that row directly would expose seconds after most forecast origins. This
    function recomputes the same concepts at 5 Hz using only samples at or
    before each origin.

    Values are expressed from the perspective of the side in possession at the
    forecast origin. TTI control is used when the shared cache contains it;
    otherwise the agreed Voronoi alternative is used.
    """
    origins = np.asarray(origins, dtype=int)
    current_home = np.asarray(current_home, dtype=bool)
    if origins.shape != current_home.shape:
        raise ValueError("origins and current_home must have the same shape")

    control_name = "home_tti" if "home_tti" in table else "home_voronoi"
    zone_name = "home_tti_ball" if "home_tti_ball" in table else "home_voronoi_ball"
    home_control = table[control_name].to_numpy(dtype=float)
    home_zone = table[zone_name].to_numpy(dtype=float)
    possession = table["poss_side"].astype(str).to_numpy()
    inplay = table["inplay"].to_numpy(dtype=bool)
    valid = inplay & np.isin(possession, ("Home", "Away"))

    feature = pd.DataFrame(index=np.arange(len(origins)))
    feature["B_current_pitch_share"] = np.where(
        current_home, home_control[origins], 1 - home_control[origins]
    )
    feature["B_current_zone_share"] = np.where(
        current_home, home_zone[origins], 1 - home_zone[origins]
    )

    ball_x = table["ball_x"].to_numpy(dtype=float)
    home_dir = table["home_dir"].to_numpy(dtype=float)
    same_possession = np.zeros(len(table), dtype=bool)
    same_possession[1:] = (
        valid[1:]
        & valid[:-1]
        & (possession[1:] == possession[:-1])
        & np.isfinite(ball_x[1:])
        & np.isfinite(ball_x[:-1])
    )
    ball_dx = np.zeros(len(table), dtype=float)
    ball_dx[1:] = ball_x[1:] - ball_x[:-1]
    home_progress = np.where(
        same_possession & (possession == "Home"), ball_dx * home_dir, 0.0
    )
    away_progress = np.where(
        same_possession & (possession == "Away"), -ball_dx * home_dir, 0.0
    )

    sequence_start = valid.copy()
    sequence_start[1:] &= (~valid[:-1]) | (possession[1:] != possession[:-1])
    home_start = sequence_start & (possession == "Home") & np.isfinite(home_zone)
    away_start = sequence_start & (possession == "Away") & np.isfinite(home_zone)

    for seconds in (60, 300):
        window = int(round(seconds * CONFIG["SAMPLE_HZ"]))
        live_count = _past_sum(valid.astype(float), origins, window)
        home_possession_count = _past_sum(
            (valid & (possession == "Home")).astype(float), origins, window
        )
        home_possession_share = np.divide(
            home_possession_count,
            live_count,
            out=np.full(len(origins), np.nan),
            where=live_count > 0,
        )

        control_valid = valid & np.isfinite(home_control)
        control_count = _past_sum(control_valid.astype(float), origins, window)
        home_pitch_share = np.divide(
            _past_sum(np.where(control_valid, home_control, 0.0), origins, window),
            control_count,
            out=np.full(len(origins), np.nan),
            where=control_count > 0,
        )
        zone_valid = valid & np.isfinite(home_zone)
        zone_count = _past_sum(zone_valid.astype(float), origins, window)
        home_zone_share = np.divide(
            _past_sum(np.where(zone_valid, home_zone, 0.0), origins, window),
            zone_count,
            out=np.full(len(origins), np.nan),
            where=zone_count > 0,
        )

        poss_share = np.where(
            current_home, home_possession_share, 1 - home_possession_share
        )
        pitch_share = np.where(current_home, home_pitch_share, 1 - home_pitch_share)
        zone_share = np.where(current_home, home_zone_share, 1 - home_zone_share)
        feature[f"B_possession_share_{seconds}s"] = poss_share
        feature[f"B_pitch_share_{seconds}s"] = pitch_share
        feature[f"B_zone_share_{seconds}s"] = zone_share
        feature[f"B_control_minus_possession_{seconds}s"] = pitch_share - poss_share
        feature[f"B_progression_m_{seconds}s"] = np.where(
            current_home,
            _past_sum(home_progress, origins, window),
            _past_sum(away_progress, origins, window),
        )

        home_start_count = _past_sum(home_start.astype(float), origins, window)
        away_start_count = _past_sum(away_start.astype(float), origins, window)
        home_start_zone = np.divide(
            _past_sum(np.where(home_start, home_zone, 0.0), origins, window),
            home_start_count,
            out=np.full(len(origins), np.nan),
            where=home_start_count > 0,
        )
        away_start_zone = np.divide(
            _past_sum(np.where(away_start, 1 - home_zone, 0.0), origins, window),
            away_start_count,
            out=np.full(len(origins), np.nan),
            where=away_start_count > 0,
        )
        feature[f"B_seq_start_zone_{seconds}s"] = np.where(
            current_home, home_start_zone, away_start_zone
        )
    return feature


def _shared_features(
    half_table: pd.DataFrame,
    half_events: pd.DataFrame,
    origins: np.ndarray,
    raw_origins: np.ndarray,
    possession: np.ndarray,
    ball_status: np.ndarray,
    framerate: int,
) -> pd.DataFrame:
    """Build current and past-only features for the requested shared-table samples."""
    table = half_table.sort_values("sample").reset_index(drop=True)
    expected = np.arange(len(table))
    if not np.array_equal(table["sample"].to_numpy(dtype=int), expected):
        raise ValueError("team_samples must contain consecutive samples within each half")

    sample_hz = CONFIG["SAMPLE_HZ"]
    inplay = table["inplay"].to_numpy(dtype=bool)
    feature = pd.DataFrame(index=np.arange(len(origins)))
    feature["possession_home"] = (possession[raw_origins] == 1).astype(int)
    feature["ball_x"] = table.loc[origins, "ball_x"].to_numpy(dtype=float)
    feature["ball_y"] = table.loc[origins, "ball_y"].to_numpy(dtype=float)
    feature["abs_ball_x"] = np.abs(feature["ball_x"])
    feature["abs_ball_y"] = np.abs(feature["ball_y"])
    feature["seconds_into_half"] = raw_origins / framerate
    feature["fraction_into_half"] = raw_origins / max(len(possession) - 1, 1)
    duration = _possession_duration_frames(possession, ball_status)
    feature["possession_duration_s"] = duration[raw_origins] / framerate

    b_features = _workstream_b_origin_features(
        table,
        origins,
        current_home=possession[raw_origins] == 1,
    )
    feature = pd.concat([feature, b_features], axis=1)

    shared_columns = ["home_voronoi", "home_voronoi_ball"]
    for optional in ("home_tti", "home_tti_ball"):
        if optional in table:
            shared_columns.append(optional)
    for column in shared_columns:
        feature[column] = table.loc[origins, column].to_numpy(dtype=float)

    # Load values describe the complete 0.2-second shared-table bin. Shift by one
    # bin so a forecast never sees frames after its nominal origin.
    load_columns = ("home_hsr", "away_hsr", "home_dist", "away_dist")
    for column in load_columns:
        completed = table[column].shift(1).to_numpy(dtype=float)
        feature[f"previous_{column}"] = completed[origins]

    feature["control_balance"] = 2 * feature["home_voronoi"] - 1
    feature["ball_control_balance"] = 2 * feature["home_voronoi_ball"] - 1
    feature["previous_hsr_balance"] = feature["previous_home_hsr"] - feature["previous_away_hsr"]
    feature["previous_distance_balance"] = feature["previous_home_dist"] - feature["previous_away_dist"]

    switches = possession_switch_mask(possession, ball_status).astype(float)
    alive = (ball_status == 1).astype(float)
    for seconds in (30, 60, 300):
        raw_window = int(round(seconds * framerate))
        sample_window = int(round(seconds * sample_hz))
        feature[f"past_switches_{seconds}s"] = _past_sum(switches, raw_origins, raw_window)
        feature[f"past_alive_share_{seconds}s"] = _past_sum(alive, raw_origins, raw_window) / np.minimum(
            raw_window, raw_origins + 1
        )
        for column in ("home_voronoi", "home_voronoi_ball"):
            values = table[column].to_numpy(dtype=float)
            feature[f"past_{column}_mean_{seconds}s"] = _rolling(values, sample_window, "mean")[origins]
        for column in load_columns:
            values = np.where(inplay, table[column].to_numpy(dtype=float), 0.0)
            values = np.r_[0.0, values[:-1]]
            feature[f"past_{column}_sum_{seconds}s"] = _rolling(values, sample_window, "sum")[origins]

    ball_xy = table[["ball_x", "ball_y"]].to_numpy(dtype=float)
    for seconds in (1, 5):
        lag = int(round(seconds * sample_hz))
        prior = origins - lag
        valid = prior >= 0
        displacement = np.full((len(origins), 2), np.nan)
        displacement[valid] = ball_xy[origins[valid]] - ball_xy[prior[valid]]
        feature[f"ball_dx_{seconds}s"] = displacement[:, 0]
        feature[f"ball_dy_{seconds}s"] = displacement[:, 1]
        feature[f"ball_speed_{seconds}s"] = np.linalg.norm(displacement, axis=1) / seconds
        for column in ("home_voronoi", "home_voronoi_ball"):
            values = table[column].to_numpy(dtype=float)
            delta = np.full(len(origins), np.nan)
            delta[valid] = values[origins[valid]] - values[prior[valid]]
            feature[f"{column}_delta_{seconds}s"] = delta

    event_arrays = _event_arrays(half_events, str(table["half"].iloc[0]), len(table))
    for seconds in (30, 60):
        window = int(round(seconds * sample_hz))
        for name, values in event_arrays.items():
            values = np.r_[0.0, values[:-1]]
            feature[f"past_{name}_{seconds}s"] = _past_sum(values, origins, window)
    return feature


def build_forecast_table(
    tables,
    horizons_seconds: Iterable[int] = DEFAULT_HORIZONS_SECONDS,
    stride_seconds: int = DEFAULT_STRIDE_SECONDS,
) -> pd.DataFrame:
    """Create the Workstream C modeling table from shared tables and 25 Hz targets."""
    horizons = tuple(sorted({int(value) for value in horizons_seconds}))
    if not horizons or min(horizons) <= 0:
        raise ValueError("horizons_seconds must contain positive values")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    frames: list[pd.DataFrame] = []
    for match_id in sorted(tables.team_samples["match"].unique()):
        raw = load_raw_match(match_id)
        match_samples = tables.team_samples[tables.team_samples["match"] == match_id]
        match_events = tables.events[tables.events["match"] == match_id]
        for half in sorted(match_samples["half"].unique()):
            half_table = match_samples[match_samples["half"] == half].copy()
            half_events = match_events[match_events["half"] == half].copy()
            possession = np.asarray(raw["poss"][half])
            ball_status = np.asarray(raw["ball_status"][half])
            framerate = int(raw["framerate"][half])
            if framerate % CONFIG["SAMPLE_HZ"]:
                raise ValueError("FRAMERATE must be divisible by SAMPLE_HZ")
            step = framerate // CONFIG["SAMPLE_HZ"]
            raw_stride = int(round(stride_seconds * framerate))
            raw_origins = np.arange(0, len(possession), raw_stride, dtype=int)
            raw_origins = raw_origins[
                (ball_status[raw_origins] == 1) & np.isin(possession[raw_origins], (1, 2))
            ]
            origins = raw_origins // step
            within_shared = origins < len(half_table)
            raw_origins, origins = raw_origins[within_shared], origins[within_shared]
            shared = _shared_features(
                half_table,
                half_events,
                origins,
                raw_origins,
                possession,
                ball_status,
                framerate,
            )
            switches = possession_switch_mask(possession, ball_status)
            cumulative = np.concatenate(([0], np.cumsum(switches, dtype=np.int64)))
            for horizon in horizons:
                horizon_frames = int(round(horizon * framerate))
                valid = raw_origins + horizon_frames < len(possession)
                selected_origins = raw_origins[valid]
                target = cumulative[selected_origins + horizon_frames + 1] - cumulative[selected_origins + 1]
                out = shared.loc[valid].reset_index(drop=True)
                out.insert(0, "match", match_id)
                out.insert(1, "half", half)
                out.insert(2, "sample", origins[valid])
                out.insert(3, "origin_frame", selected_origins)
                out.insert(4, "horizon_seconds", horizon)
                out["switches_future"] = target
                frames.append(out)
    if not frames:
        raise ValueError("No valid Workstream C forecast origins were produced")
    return pd.concat(frames, ignore_index=True)


def feature_columns(forecast_table: pd.DataFrame) -> list[str]:
    excluded = {"match", "half", "sample", "origin_frame", "horizon_seconds", "switches_future"}
    return [column for column in forecast_table.columns if column not in excluded]


def has_workstream_b_features(forecast_table: pd.DataFrame) -> bool:
    """Return whether a cached forecast table has the complete B integration."""
    return set(WORKSTREAM_B_FEATURE_COLUMNS).issubset(forecast_table.columns)


def evaluate_baseline(forecast_table: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the historical training mean on each complete held-out match."""
    rows: list[dict] = []
    for horizon, horizon_data in forecast_table.groupby("horizon_seconds"):
        for held_out in sorted(horizon_data["match"].unique()):
            train = horizon_data[horizon_data["match"] != held_out]
            test = horizon_data[horizon_data["match"] == held_out]
            prediction = np.full(len(test), train["switches_future"].mean())
            rows.append(
                {
                    "horizon_seconds": horizon,
                    "held_out_match": held_out,
                    "model": "historical_mean",
                    "mae": mean_absolute_error(test["switches_future"], prediction),
                    "n_test_origins": len(test),
                }
            )
    return pd.DataFrame(rows)


def evaluate_models(
    forecast_table: pd.DataFrame,
    predictor_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Evaluate fixed baseline, Poisson and boosting models on complete held-out matches."""
    columns = list(
        feature_columns(forecast_table)
        if predictor_columns is None
        else predictor_columns
    )
    factories = {
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
    rows = evaluate_baseline(forecast_table).to_dict("records")
    for horizon, horizon_data in forecast_table.groupby("horizon_seconds"):
        for held_out in sorted(horizon_data["match"].unique()):
            train = horizon_data[horizon_data["match"] != held_out]
            test = horizon_data[horizon_data["match"] == held_out]
            y_train = train["switches_future"].to_numpy(dtype=float)
            y_test = test["switches_future"].to_numpy(dtype=float)
            for model_name, factory in factories.items():
                model = factory()
                model.fit(train[columns], y_train)
                prediction = np.clip(model.predict(test[columns]), 0, None)
                rows.append(
                    {
                        "horizon_seconds": horizon,
                        "held_out_match": held_out,
                        "model": model_name,
                        "mae": mean_absolute_error(y_test, prediction),
                        "n_test_origins": len(test),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_feature_ablation(forecast_table: pd.DataFrame) -> pd.DataFrame:
    """Compare the original C predictors with the leakage-safe B integration."""
    all_columns = feature_columns(forecast_table)
    feature_sets = {
        "C_only": [column for column in all_columns if not column.startswith("B_")],
        "C_plus_B": all_columns,
    }
    frames = []
    for name, columns in feature_sets.items():
        scores = evaluate_models(forecast_table, predictor_columns=columns)
        scores.insert(0, "feature_set", name)
        frames.append(scores)
    return pd.concat(frames, ignore_index=True)


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["horizon_seconds", "model"])["mae"]
        .agg(mean_mae="mean", std_across_matches="std")
        .reset_index()
    )


def compare_to_baseline(
    scores: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 20_000,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Paired model-minus-baseline MAE with a match bootstrap interval.

    Negative values favour the model. Matches, rather than forecast origins,
    are resampled because the outer validation unit is a complete match.
    """
    pivot = scores.pivot(
        index=["horizon_seconds", "held_out_match"], columns="model", values="mae"
    ).reset_index()
    rng = np.random.default_rng(random_seed)
    rows = []
    for horizon, horizon_data in pivot.groupby("horizon_seconds"):
        for model in sorted(set(pivot.columns) - {"horizon_seconds", "held_out_match", "historical_mean"}):
            delta = (horizon_data[model] - horizon_data["historical_mean"]).to_numpy(dtype=float)
            samples = delta[
                rng.integers(0, len(delta), size=(bootstrap_repetitions, len(delta)))
            ].mean(axis=1)
            rows.append(
                {
                    "horizon_seconds": horizon,
                    "model": model,
                    "delta_mae_vs_baseline": delta.mean(),
                    "ci_low": np.quantile(samples, 0.025),
                    "ci_high": np.quantile(samples, 0.975),
                    "matches_beating_baseline": int((delta < 0).sum()),
                    "matches_evaluated": len(delta),
                }
            )
    return pd.DataFrame(rows)


def team_minute_features(forecast_table: pd.DataFrame) -> pd.DataFrame:
    """Create documented C_ features at the team-minute grain for integration."""
    horizon = int(forecast_table["horizon_seconds"].min())
    base = forecast_table[forecast_table["horizon_seconds"] == horizon].copy()
    base["minute"] = (base["seconds_into_half"] // 60).astype(int)
    base.loc[base["half"] == "secondHalf", "minute"] += 45
    grouped = (
        base.groupby(["match", "half", "minute"], as_index=False)
        .agg(
            C_recent_switches_30s=("past_switches_30s", "mean"),
            C_recent_switches_60s=("past_switches_60s", "mean"),
            C_possession_duration_s=("possession_duration_s", "mean"),
            C_ball_speed_5s=("ball_speed_5s", "mean"),
            C_home_control_momentum_5s=("home_voronoi_ball_delta_5s", "mean"),
        )
    )
    rows = []
    for side in ("Home", "Away"):
        frame = grouped.copy()
        frame["side"] = side
        if side == "Away":
            frame["C_home_control_momentum_5s"] *= -1
        frame = frame.rename(columns={"C_home_control_momentum_5s": "C_control_momentum_5s"})
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["match", "half", "minute", "side"])


def save_results(
    forecast_table: pd.DataFrame,
    scores: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    output = Path(output_dir or (cache_dir() / "workstream_c"))
    output.mkdir(parents=True, exist_ok=True)
    forecast_table.to_csv(output / "forecast_table.csv.gz", index=False, compression="gzip")
    scores.to_csv(output / "lomo_scores.csv", index=False)
    summarize_scores(scores).to_csv(output / "model_summary.csv", index=False)
    compare_to_baseline(scores).to_csv(output / "robustness_summary.csv", index=False)
    team_minute_features(forecast_table).to_csv(output / "team_minute_features.csv", index=False)
    return output
