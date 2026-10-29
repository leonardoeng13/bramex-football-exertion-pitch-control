"""Statistical helpers shared by the workstreams."""
import numpy as np
import pandas as pd


def within_effect(df, y, x, group, cluster, controls=()):
    """Standardised within-group effect of x on y.

    Demeans y, x and controls inside each group (for example player x match), standardises them,
    fits OLS and clusters standard errors by `cluster` (for example player).
    Returns effect, 95% CI, p-value and n.
    """
    import statsmodels.api as sm

    cols = [y, x, *controls]
    d = df.dropna(subset=cols).copy()
    for c in cols:
        d[c] = d[c] - d.groupby(group)[c].transform("mean")
        sd = d[c].std()
        d[c] = d[c] / sd if sd > 0 else d[c]
    X = sm.add_constant(d[[x, *controls]])
    fit = sm.OLS(d[y], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    lo, hi = fit.conf_int().loc[x]
    return {"effect": fit.params[x], "ci_low": lo, "ci_high": hi, "p": fit.pvalues[x], "n": len(d),
            "clusters": d[cluster].nunique()}


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out
