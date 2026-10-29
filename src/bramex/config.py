"""Part 1: paths, decision-log parameters and the holdout guard."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Every parameter that changes results, with its decision-log entry and status.
# Update the status to "Agreed" when the team agrees at the lab.
_PARAMETERS = [
    # name, value, decision, status, description
    ("FRAMERATE", 25, "-", "Fixed", "DFL tracking frame rate (Hz)"),
    ("SMOOTH_WINDOW", 13, "D1", "Proposed", "Savitzky-Golay window for positions (frames)"),
    ("SMOOTH_ORDER", 2, "D1", "Proposed", "Savitzky-Golay polynomial order"),
    ("MAX_SPEED", 12.0, "D1", "Proposed", "Physical speed cap after smoothing (m/s)"),
    ("MAX_ACC", 8.0, "D1", "Proposed", "Physical acceleration cap after smoothing (m/s2)"),
    ("HSR_SPEED", 5.5, "D3", "Proposed", "High-speed running threshold (m/s)"),
    ("SPRINT_SPEED", 7.0, "D3", "Proposed", "Sprint threshold (m/s)"),
    ("ACC_THRESHOLD", 2.5, "D3", "Proposed", "High acceleration and deceleration threshold (m/s2)"),
    ("SAMPLE_HZ", 5, "D5", "Proposed", "Analysis sampling rate (Hz)"),
    ("GRID_M", 1.0, "D4", "Proposed", "Pitch-control grid resolution (m)"),
    ("BALL_SIGMA_M", 15.0, "D4", "Proposed", "Gaussian weight around the ball for ball-zone control (m)"),
    ("PITCH_CONTROL", ("voronoi",), "D4", "Proposed", "Models computed: voronoi, and tti (time to arrival, slower)"),
    ("TTI_REACTION_S", 0.7, "D4", "Proposed", "Time-to-arrival model: reaction time (s)"),
    ("TTI_VMAX", 5.0, "D4", "Proposed", "Time-to-arrival model: maximum speed (m/s)"),
    ("TTI_SIGMA", 0.45, "D4", "Proposed", "Time-to-arrival model: arrival time uncertainty (s)"),
    ("POSSESSION", "dfl_flag", "D6", "Proposed", "Possession definition"),
    ("VALIDATION", "leave_one_match_out", "D8", "Proposed", "Validation scheme"),
]

# Bump when the table-building code changes, so old caches are not reused
TABLES_VERSION = "1"


class Config(dict):
    """Parameter values by name, with decision-log metadata."""

    def __init__(self):
        super().__init__({name: value for name, value, *_ in _PARAMETERS})
        self.meta = {name: {"decision": d, "status": s, "description": desc} for name, _, d, s, desc in _PARAMETERS}
        self["HOLDOUT_MATCHES"] = []

    def describe(self):
        # Table of parameters for the notebook and the decision log
        rows = [{"parameter": k, "value": self[k], **self.meta[k]} for k in self.meta]
        return pd.DataFrame(rows).set_index("parameter")

    def set_status(self, decision, status="Agreed"):
        # Mark every parameter of one decision (for example "D3") as agreed
        for k, m in self.meta.items():
            if m["decision"] == decision:
                m["status"] = status

    def table_hash(self):
        # Short hash of everything that changes the shared tables (used to name the cache)
        keys = sorted(self.meta)
        payload = {k: self[k] for k in keys} | {"version": TABLES_VERSION}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


CONFIG = Config()

# DFL position codes to position groups used by every workstream
POSITION_GROUP = {
    "TW": "GK", "IVL": "CB", "IVR": "CB", "IVZ": "CB", "LV": "FB", "RV": "FB", "DLM": "FB", "DRM": "FB",
    "DML": "CM", "DMR": "CM", "DMZ": "CM", "HL": "CM", "HR": "CM", "ZO": "AM/W", "OLM": "AM/W", "ORM": "AM/W",
    "LM": "AM/W", "RM": "AM/W", "STZ": "ST", "STL": "ST", "STR": "ST",
}
HALVES = ["firstHalf", "secondHalf"]
PITCH_X, PITCH_Y = 105.0, 68.0


def data_dir():
    # Folder with the DFL XML files (environment variable BRAMEX_DATA_DIR)
    return Path(os.environ.get("BRAMEX_DATA_DIR", "data")).expanduser()


def cache_dir():
    # Cache folder, one subfolder per parameter set so different settings never mix
    root = Path(os.environ.get("BRAMEX_CACHE_DIR", data_dir() / "cache")).expanduser()
    path = root / f"tables_{CONFIG.table_hash()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_environment(data=None, cache=None, quiet=False):
    """Set paths and display options. Works locally, on Google Colab and on other hosted notebooks.

    data:  folder with the DFL XML files (overrides BRAMEX_DATA_DIR)
    cache: folder for cached tables (overrides BRAMEX_CACHE_DIR); share it to avoid recomputing
    """
    if data:
        os.environ["BRAMEX_DATA_DIR"] = str(data)
    if cache:
        os.environ["BRAMEX_CACHE_DIR"] = str(cache)
    # Readable numbers, never scientific notation
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    pd.set_option("display.max_columns", 60)
    np.set_printoptions(suppress=True, precision=4)
    env = "colab" if "google.colab" in sys.modules else "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "local"
    info = {"environment": env, "data_dir": str(data_dir()), "cache_dir": str(cache_dir()),
            "data_found": data_dir().exists(), "python": sys.version.split()[0]}
    if not quiet:
        for k, v in info.items():
            print(f"{k:12s} {v}")
    return info
