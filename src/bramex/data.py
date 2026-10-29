"""Part 2: find the DFL files of each match, parse them once and cache the raw data."""
import glob
import pickle
import re
from pathlib import Path

from .config import CONFIG, data_dir, cache_dir

KINDS = {"positions": "positions", "matchinformation": "info", "events": "events"}


def discover_matches(folder=None, include_holdout=False):
    """Return {match_id: {"positions": path, "info": path, "events": path}} for complete matches.

    Matches with missing files are reported, not silently dropped.
    """
    folder = Path(folder or data_dir())
    found = {}
    for f in glob.glob(str(folder / "**" / "*.xml"), recursive=True):
        m = re.search(r"DFL-MAT-(\w+)\.xml$", f)
        kind = next((v for k, v in KINDS.items() if k in Path(f).name), None)
        if m and kind:
            found.setdefault(m.group(1), {})[kind] = f
    complete = {k: v for k, v in sorted(found.items()) if set(v) == {"positions", "info", "events"}}
    for k in sorted(set(found) - set(complete)):
        print(f"warning: match {k} is missing {sorted({'positions', 'info', 'events'} - set(found[k]))}")
    if not include_holdout:
        complete = {k: v for k, v in complete.items() if k not in CONFIG["HOLDOUT_MATCHES"]}
    return complete


def check_holdout(match_id, allow_holdout=False):
    # Holdout guard: refuse to touch a holdout match before the freeze
    if match_id in CONFIG["HOLDOUT_MATCHES"] and not allow_holdout:
        raise PermissionError(f"{match_id} is a holdout match and cannot be loaded before the models are frozen")


def load_raw_match(match_id, allow_holdout=False):
    """Parse positions, possession, ball status, team sheets and events of one match (cached)."""
    check_holdout(match_id, allow_holdout)
    path = cache_dir() / f"{match_id}_raw.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    from floodlight.io.dfl import read_event_data_xml, read_position_data_xml

    files = discover_matches(include_holdout=True).get(match_id)
    if files is None:
        raise FileNotFoundError(f"no complete set of DFL files for {match_id} in {data_dir()}")
    xy, poss, ball_status, teamsheets, pitch = read_position_data_xml(files["positions"], files["info"])
    events, _, _ = read_event_data_xml(files["events"], files["info"])
    raw = {
        "xy": {h: {k: v.xy.astype("float32") for k, v in d.items()} for h, d in xy.items()},
        "framerate": {h: d["Home"].framerate for h, d in xy.items()},
        "poss": {h: c.code for h, c in poss.items()},
        "ball_status": {h: c.code for h, c in ball_status.items()},
        "teamsheets": {k: v.teamsheet for k, v in teamsheets.items()},
        "events": {h: {s: e.events for s, e in d.items()} for h, d in events.items()},
        "pitch": (tuple(pitch.xlim), tuple(pitch.ylim)),
    }
    with open(path, "wb") as fh:
        pickle.dump(raw, fh)
    return raw
