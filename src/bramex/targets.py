"""Leakage-safe target construction for possession-switch forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def possession_switch_mask(
    possession: np.ndarray,
    ball_status: np.ndarray,
    *,
    alive_code: int = 1,
) -> np.ndarray:
    """Mark Home/Away changes between consecutive live-ball frames.

    Changes across a dead-ball interval are excluded because the course criterion
    counts switches only during uninterrupted active play.
    """
    possession = np.asarray(possession)
    ball_status = np.asarray(ball_status)
    if possession.ndim != 1 or ball_status.ndim != 1:
        raise ValueError("possession and ball_status must be one-dimensional")
    if possession.shape != ball_status.shape:
        raise ValueError("possession and ball_status must have equal length")

    switches = np.zeros(possession.shape[0], dtype=bool)
    if switches.size < 2:
        return switches
    switches[1:] = (
        (possession[1:] != possession[:-1])
        & (ball_status[1:] == alive_code)
        & (ball_status[:-1] == alive_code)
    )
    return switches


def build_switch_targets(
    possession: np.ndarray,
    ball_status: np.ndarray,
    *,
    match_id: str,
    half: str,
    framerate: int,
    horizon_seconds: float,
    stride_seconds: float = 1.0,
    require_alive_origin: bool = True,
) -> pd.DataFrame:
    """Count valid switches strictly after each origin and through its horizon.

    By default, forecast origins are limited to frames where the ball is alive.
    The target itself counts only uninterrupted live-ball possession changes.
    """
    if framerate <= 0:
        raise ValueError("framerate must be positive")
    if horizon_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("horizon_seconds and stride_seconds must be positive")

    switches = possession_switch_mask(possession, ball_status)
    horizon_frames = int(round(horizon_seconds * framerate))
    stride_frames = int(round(stride_seconds * framerate))
    if horizon_frames < 1 or stride_frames < 1:
        raise ValueError("horizon and stride must span at least one frame")

    last_origin = switches.size - horizon_frames - 1
    if last_origin < 0:
        return pd.DataFrame(
            columns=["match_id", "half", "frame", "gameclock_s", "switches_future"]
        )

    origins = np.arange(0, last_origin + 1, stride_frames, dtype=int)
    if require_alive_origin:
        origins = origins[np.asarray(ball_status)[origins] == 1]
    cumulative = np.concatenate(([0], np.cumsum(switches, dtype=np.int64)))
    future_counts = cumulative[origins + horizon_frames + 1] - cumulative[origins + 1]

    return pd.DataFrame(
        {
            "match_id": match_id,
            "half": half,
            "frame": origins,
            "gameclock_s": origins / framerate,
            "switches_future": future_counts,
        }
    )
