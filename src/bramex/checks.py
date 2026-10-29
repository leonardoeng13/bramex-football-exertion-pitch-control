"""Part 4: checks that must pass before any analysis (and on the holdout match)."""
import numpy as np
import pandas as pd

from .config import CONFIG, PITCH_X, PITCH_Y
from .data import load_raw_match


def run_checks(tables, raise_on_fail=True, allow_holdout=False):
    """Return a table of checks; stop the notebook if any fails."""
    rows = []

    def check(name, passed, detail=""):
        rows.append({"check": name, "passed": bool(passed), "detail": detail})

    s, ts, ev = tables.samples, tables.team_samples, tables.events
    for m in sorted(s.match.unique()):
        raw = load_raw_match(m, allow_holdout=allow_holdout)
        check(f"{m}: frame rate is {CONFIG['FRAMERATE']} Hz", all(v == CONFIG["FRAMERATE"] for v in raw["framerate"].values()))
        xl, yl = raw["pitch"]
        check(f"{m}: pitch is 105 x 68 m", (xl[1] - xl[0], yl[1] - yl[0]) == (PITCH_X, PITCH_Y), f"{xl}, {yl}")
        t = ts[(ts.match == m) & ts.inplay]
        share = t.poss_side.isin(["Home", "Away"]).mean()
        check(f"{m}: possession assigned in at least 95% of ball-in-play samples", share >= 0.95, f"{share:.3f}")
        e = ev[ev.match == m]
        n_samples = ts[ts.match == m].groupby("half")["sample"].max() + 1
        raw_sample = (e["gameclock"] * CONFIG["SAMPLE_HZ"]).round()
        inside = ((raw_sample >= 0) & (raw_sample < e["half"].map(n_samples))).mean()
        check(f"{m}: at least 99% of events fall inside the tracked half", inside >= 0.99, f"{inside:.3f}")

    per_team = s.groupby(["match", "half", "sample", "side"]).size()
    check("At most 11 players per team at any sample", per_team.max() <= 11, f"max {per_team.max()}")
    check("Speeds within the physical range", s.speed.between(0, CONFIG["MAX_SPEED"]).all(), f"max {s.speed.max():.2f} m/s")
    vor = s.groupby(["match", "half", "sample"]).voronoi.sum()
    vor = vor[vor > 0]
    check("Voronoi areas add up to the pitch area", np.allclose(vor, PITCH_X * PITCH_Y, atol=1), f"median {vor.median():,.0f} m2")
    check("No holdout match in any table", not s.match.isin(CONFIG["HOLDOUT_MATCHES"]).any() or allow_holdout)
    out = pd.DataFrame(rows)
    if raise_on_fail and not out.passed.all():
        raise AssertionError("data checks failed:\n" + out[~out.passed].to_string(index=False))
    return out
