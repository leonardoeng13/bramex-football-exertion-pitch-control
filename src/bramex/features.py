"""Past-only features for the first Workstream C forecasting model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .targets import build_switch_targets, possession_switch_mask


def _past_sum(values: np.ndarray, origins: np.ndarray, window_frames: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    starts = np.maximum(0, origins - window_frames + 1)
    return cumulative[origins + 1] - cumulative[starts]


def build_switch_feature_table(
    possession: np.ndarray,
    ball_status: np.ndarray,
    ball_xy: np.ndarray,
    *,
    match_id: str,
    half: str,
    framerate: int,
    horizon_seconds: float,
    stride_seconds: float = 5.0,
) -> pd.DataFrame:
    """Build a target and feature table using data at or before each origin."""
    possession = np.asarray(possession)
    ball_status = np.asarray(ball_status)
    ball_xy = np.asarray(ball_xy, dtype=float)
    if ball_xy.shape != (len(possession), 2):
        raise ValueError("ball_xy must have shape (n_frames, 2)")

    table = build_switch_targets(
        possession,
        ball_status,
        match_id=match_id,
        half=half,
        framerate=framerate,
        horizon_seconds=horizon_seconds,
        stride_seconds=stride_seconds,
        require_alive_origin=True,
    )
    if table.empty:
        return table

    origins = table["frame"].to_numpy(dtype=int)
    switches = possession_switch_mask(possession, ball_status).astype(float)
    alive = (ball_status == 1).astype(float)

    table["possession_home"] = (possession[origins] == 1).astype(int)
    table["ball_x"] = ball_xy[origins, 0]
    table["ball_y"] = ball_xy[origins, 1]
    table["abs_ball_x"] = np.abs(table["ball_x"])
    table["abs_ball_y"] = np.abs(table["ball_y"])
    table["seconds_into_half"] = origins / framerate
    table["fraction_into_half"] = origins / max(len(possession) - 1, 1)

    for window_seconds in (30, 60, 300):
        window_frames = int(round(window_seconds * framerate))
        table[f"past_switches_{window_seconds}s"] = _past_sum(
            switches, origins, window_frames
        )
        table[f"past_alive_share_{window_seconds}s"] = _past_sum(
            alive, origins, window_frames
        ) / np.minimum(window_frames, origins + 1)

    for lag_seconds in (1, 5):
        lag_frames = int(round(lag_seconds * framerate))
        lag_origins = origins - lag_frames
        valid = lag_origins >= 0
        displacement = np.full((len(origins), 2), np.nan)
        displacement[valid] = ball_xy[origins[valid]] - ball_xy[lag_origins[valid]]
        table[f"ball_dx_{lag_seconds}s"] = displacement[:, 0]
        table[f"ball_dy_{lag_seconds}s"] = displacement[:, 1]
        table[f"ball_speed_{lag_seconds}s"] = (
            np.linalg.norm(displacement, axis=1) / lag_seconds
        )

    return table
