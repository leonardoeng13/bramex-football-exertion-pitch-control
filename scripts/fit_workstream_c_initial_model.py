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
    evaluate_models,
    save_results,
    summarize_scores,
)


def main() -> None:
    output = bramex.cache_dir() / "workstream_c"
    forecast_path = output / "forecast_table.csv.gz"
    if forecast_path.exists():
        forecast = pd.read_csv(forecast_path)
    else:
        tables = bramex.load_tables()
        bramex.run_checks(tables)
        forecast = build_forecast_table(tables)
    scores = evaluate_models(forecast)
    save_results(forecast, scores, output)
    print(summarize_scores(scores).to_string(index=False))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
