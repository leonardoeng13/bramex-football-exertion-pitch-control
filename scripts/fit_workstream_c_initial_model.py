"""Fit and evaluate Workstream C models on complete held-out matches."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bramex  # noqa: E402
from bramex.workstream_c import (  # noqa: E402
    build_forecast_table,
    compare_to_baseline,
    evaluate_feature_ablation,
    has_workstream_b_features,
    save_results,
    summarize_scores,
)


def main() -> None:
    output = bramex.cache_dir() / "workstream_c"
    forecast_path = output / "forecast_table.csv.gz"
    if forecast_path.exists():
        forecast = pd.read_csv(forecast_path)
    if not forecast_path.exists() or not has_workstream_b_features(forecast):
        tables = bramex.load_tables()
        bramex.run_checks(tables)
        forecast = build_forecast_table(tables)
    ablation = evaluate_feature_ablation(forecast)
    scores = ablation[ablation["feature_set"] == "C_plus_B"].drop(
        columns="feature_set"
    )
    save_results(forecast, scores, output)
    ablation.to_csv(output / "feature_ablation_lomo.csv", index=False)
    ablation_summary = (
        ablation.groupby(["feature_set", "horizon_seconds", "model"])["mae"]
        .agg(mean_mae="mean", std_across_matches="std")
        .reset_index()
    )
    ablation_summary.to_csv(output / "feature_ablation_summary.csv", index=False)
    robustness = []
    for feature_set, group in ablation.groupby("feature_set"):
        result = compare_to_baseline(group.drop(columns="feature_set"))
        result.insert(0, "feature_set", feature_set)
        robustness.append(result)
    pd.concat(robustness, ignore_index=True).to_csv(
        output / "feature_ablation_robustness.csv", index=False
    )
    print(summarize_scores(scores).to_string(index=False))
    print("\nFeature ablation")
    print(ablation_summary.to_string(index=False))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
