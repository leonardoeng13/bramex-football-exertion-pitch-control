"""Feature registry, integration (Part 8) and results log (Part 11), shared by all workstreams."""
import numpy as np
import pandas as pd

from .config import CONFIG

KEYS = {"team_minute": ["match", "half", "minute", "side"],
        "player_minute": ["match", "half", "minute", "side", "pID"]}
EVIDENCE = ["Descriptive", "Association", "Robust association", "Predictive (out-of-match)", "Causal (not claimed)"]
FEATURES, RESULTS = {}, []


def register_features(name, df, owner, grain, descriptions):
    """Add a feature set for integration.

    grain: "team_minute" or "player_minute"
    descriptions: {column: one-line description} for every feature column
    Features must use only information up to and including that minute.
    """
    keys = KEYS[grain]
    missing = set(keys) - set(df.columns)
    assert not missing, f"{name}: missing keys {missing}"
    assert not df.duplicated(keys).any(), f"{name}: duplicated keys"
    assert not df.match.isin(CONFIG["HOLDOUT_MATCHES"]).any(), f"{name}: holdout rows present"
    cols = [c for c in df.columns if c not in keys]
    undocumented = set(cols) - set(descriptions)
    assert not undocumented, f"{name}: add a description for {sorted(undocumented)}"
    FEATURES[name] = {"df": df, "owner": owner, "grain": grain, "columns": cols, "descriptions": descriptions}
    print(f"registered {name}: {len(df):,} rows, {len(cols)} features ({owner})")


def feature_registry():
    rows = [{"feature_set": k, "feature": c, "owner": v["owner"], "grain": v["grain"], "description": v["descriptions"][c]}
            for k, v in FEATURES.items() for c in v["columns"]]
    return pd.DataFrame(rows)


def integrate(base, agg="sum"):
    """Merge every registered feature set into the team-minute base table."""
    out = base.copy()
    keys = KEYS["team_minute"]
    for name, f in FEATURES.items():
        df = f["df"]
        if f["grain"] == "player_minute":
            df = df.groupby(keys)[f["columns"]].agg(agg).reset_index()
        clash = set(f["columns"]) & set(out.columns)
        assert not clash, f"{name}: columns already in the table {clash}"
        out = out.merge(df, on=keys, how="left")
    return out


def log_result(workstream, claim, evidence, estimate=np.nan, ci_low=np.nan, ci_high=np.nan, n=np.nan, method=""):
    """One row per claim that may go into the presentation."""
    assert evidence in EVIDENCE, f"evidence must be one of {EVIDENCE}"
    RESULTS.append(dict(workstream=workstream, claim=claim, evidence=evidence, estimate=estimate,
                        ci_low=ci_low, ci_high=ci_high, n=n, method=method))


def results_log():
    return pd.DataFrame(RESULTS)
