"""Run the shared checks and add Workstream C possession-switch diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bramex  # noqa: E402
from bramex.targets import possession_switch_mask  # noqa: E402


def main() -> None:
    tables = bramex.load_tables()
    checks = bramex.run_checks(tables)
    rows = []
    for match_id in sorted(tables.team_samples["match"].unique()):
        raw = bramex.load_raw_match(match_id)
        for half in sorted(raw["poss"]):
            possession = np.asarray(raw["poss"][half])
            ball_status = np.asarray(raw["ball_status"][half])
            rows.append(
                {
                    "match": match_id,
                    "half": half,
                    "frames": len(possession),
                    "framerate_hz": raw["framerate"][half],
                    "alive_share": np.mean(ball_status == 1),
                    "possession_switches_active_play": possession_switch_mask(
                        possession, ball_status
                    ).sum(),
                    "possession_codes": ",".join(map(str, sorted(np.unique(possession)))),
                }
            )
    output = bramex.cache_dir() / "workstream_c"
    output.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows)
    checks.to_csv(output / "shared_checks.csv", index=False)
    audit.to_csv(output / "possession_audit.csv", index=False)
    print(checks.to_string(index=False))
    print("\nWorkstream C possession audit")
    print(audit.to_string(index=False))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
