"""Build the shared-table Workstream C forecast table and historical baseline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bramex  # noqa: E402
from bramex.workstream_c import (  # noqa: E402
    build_forecast_table,
    evaluate_baseline,
    team_minute_features,
)


def main() -> None:
    tables = bramex.load_tables()
    bramex.run_checks(tables)
    forecast = build_forecast_table(tables)
    baseline = evaluate_baseline(forecast)
    output = bramex.cache_dir() / "workstream_c"
    output.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output / "forecast_table.csv.gz", index=False, compression="gzip")
    baseline.to_csv(output / "baseline_scores.csv", index=False)
    team_minute_features(forecast).to_csv(output / "team_minute_features.csv", index=False)
    print("Target distribution")
    print(
        forecast.groupby("horizon_seconds")["switches_future"]
        .agg(["count", "mean", "std", "median", "max"])
        .to_string()
    )
    print("\nHistorical-mean leave-one-match-out baseline")
    print(baseline.groupby("horizon_seconds")["mae"].agg(["mean", "std"]).to_string())
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
