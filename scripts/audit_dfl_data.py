"""Parse every training match and write a compact, reproducible data audit."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from floodlight.io.dfl import read_event_data_xml, read_position_data_xml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bramex.config import get_project_paths  # noqa: E402
from bramex.data import discover_match_files  # noqa: E402
from bramex.targets import possession_switch_mask  # noqa: E402


def audit_match(files, compact_cache_dir: Path) -> list[dict]:
    started = perf_counter()
    xy, possession, ball_status, teamsheets, pitch = read_position_data_xml(
        files.positions, files.matchinformation
    )
    position_seconds = perf_counter() - started

    started = perf_counter()
    events, _, _ = read_event_data_xml(
        files.events,
        files.matchinformation,
        teamsheets["Home"],
        teamsheets["Away"],
    )
    event_seconds = perf_counter() - started

    compact_cache_dir.mkdir(parents=True, exist_ok=True)
    compact_arrays = {}
    for half in sorted(possession):
        compact_arrays[f"{half}_possession"] = np.asarray(possession[half].code)
        compact_arrays[f"{half}_ball_status"] = np.asarray(ball_status[half].code)
        compact_arrays[f"{half}_ball_xy"] = np.asarray(xy[half]["Ball"].xy)
        compact_arrays[f"{half}_framerate"] = np.asarray([possession[half].framerate])
    np.savez_compressed(compact_cache_dir / f"{files.match_id}.npz", **compact_arrays)

    rows = []
    for half in sorted(possession):
        poss = np.asarray(possession[half].code)
        alive = np.asarray(ball_status[half].code)
        switches = possession_switch_mask(poss, alive)
        half_events = pd.concat(
            [events[half]["Home"].events, events[half]["Away"].events],
            ignore_index=True,
        )
        duration_seconds = len(poss) / possession[half].framerate
        rows.append(
            {
                "match_id": files.match_id,
                "half": half,
                "frames": len(poss),
                "framerate_hz": possession[half].framerate,
                "duration_minutes": duration_seconds / 60,
                "alive_share": float(np.mean(alive == 1)),
                "possession_switches_active_play": int(switches.sum()),
                "home_event_count": len(events[half]["Home"].events),
                "away_event_count": len(events[half]["Away"].events),
                "max_event_gameclock_s": float(half_events["gameclock"].max()),
                "events_after_tracking": int((half_events["gameclock"] > duration_seconds).sum()),
                "home_roster_players": len(teamsheets["Home"].teamsheet),
                "away_roster_players": len(teamsheets["Away"].teamsheet),
                "home_tracking_columns": xy[half]["Home"].xy.shape[1],
                "away_tracking_columns": xy[half]["Away"].xy.shape[1],
                "pitch_length_m": pitch.length,
                "pitch_width_m": pitch.width,
                "position_parse_seconds": position_seconds,
                "event_parse_seconds": event_seconds,
            }
        )
    return rows


def main() -> None:
    paths = get_project_paths(ROOT)
    matches = discover_match_files(paths.data_dir)
    compact_cache_dir = paths.cache_dir / "compact"
    rows = []
    for index, files in enumerate(matches.values(), start=1):
        print(f"[{index}/{len(matches)}] Auditing {files.match_id}...", flush=True)
        rows.extend(audit_match(files, compact_cache_dir))

    audit = pd.DataFrame(rows).sort_values(["match_id", "half"]).reset_index(drop=True)
    output = paths.cache_dir / "dfl_audit.csv"
    audit.to_csv(output, index=False)
    print(audit.to_string(index=False))
    print(f"\nSaved audit: {output}")


if __name__ == "__main__":
    main()
