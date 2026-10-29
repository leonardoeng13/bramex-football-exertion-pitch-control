"""Part 3: the shared tables every workstream starts from.

samples       player x analysis sample (ball in play): load, pitch control, distance to ball, phase
team_samples  match x analysis sample: possession, ball position, team pitch-control shares
events        one row per DFL event, aligned to the analysis sample
players       player x match: name, team, side, position group
player_minute player x minute x phase aggregates
team_minute   team x minute aggregates (one row per side)
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from .config import CONFIG, HALVES, POSITION_GROUP, PITCH_X, PITCH_Y, cache_dir
from .data import check_holdout, discover_matches, load_raw_match

TABLE_NAMES = ["samples", "team_samples", "events", "players"]
LOAD_COLS = ["dist", "hsr", "sprint", "acc_time", "dec_time", "met_energy"]


# ---------------------------------------------------------------- 3.1 kinematics and load
def kinematics(xy):
    """Positions (T, 2N) to smoothed speed, acceleration (T, N) and velocity (T, 2N).

    Savitzky-Golay on each continuous segment of each player; NaN outside the pitch.
    """
    fr, w, o = CONFIG["FRAMERATE"], CONFIG["SMOOTH_WINDOW"], CONFIG["SMOOTH_ORDER"]
    T, N = xy.shape[0], xy.shape[1] // 2
    speed = np.full((T, N), np.nan, "float32")
    acc = np.full((T, N), np.nan, "float32")
    vel = np.full((T, 2 * N), np.nan, "float32")
    for j in range(N):
        x, y = xy[:, 2 * j], xy[:, 2 * j + 1]
        idx = np.where(~np.isnan(x))[0]
        for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(seg) < w:
                continue
            vx = savgol_filter(x[seg], w, o, deriv=1, delta=1 / fr)
            vy = savgol_filter(y[seg], w, o, deriv=1, delta=1 / fr)
            s = savgol_filter(np.clip(np.hypot(vx, vy), 0, CONFIG["MAX_SPEED"]), w, o)
            s = np.clip(s, 0, CONFIG["MAX_SPEED"])
            speed[seg, j] = s
            acc[seg, j] = np.clip(np.gradient(s, 1 / fr), -CONFIG["MAX_ACC"], CONFIG["MAX_ACC"])
            vel[seg, 2 * j], vel[seg, 2 * j + 1] = vx, vy
    return speed, acc, vel


def metabolic_power(speed, acc):
    # di Prampero / Osgnach energy cost of accelerated running (W/kg)
    es = np.clip(acc / 9.81, -0.5, 0.5)
    ec = (155.4 * es**5 - 30.4 * es**4 - 43.3 * es**3 + 46.3 * es**2 + 19.5 * es + 3.6) * np.sqrt(es**2 + 1)
    return np.clip(ec * speed, 0, 100)


def pitch_grid():
    # Cell centres of the pitch-control grid (metres, origin at the centre spot)
    g = CONFIG["GRID_M"]
    gx = np.arange(-PITCH_X / 2 + g / 2, PITCH_X / 2, g)
    gy = np.arange(-PITCH_Y / 2 + g / 2, PITCH_Y / 2, g)
    GX, GY = np.meshgrid(gx, gy)
    return np.c_[GX.ravel(), GY.ravel()].astype("float32")


# ---------------------------------------------------------------- 3.3 pitch control
def pitch_control_sample(P, V, is_home, ball, cells, use_tti):
    """Pitch control for one sample.

    P (N, 2) positions, V (N, 2) velocities, is_home (N,) bool, ball (2,) position.
    Returns per-player arrays and team shares (home perspective).
    """
    N = len(P)
    area = CONFIG["GRID_M"] ** 2
    out = {k: np.full(N, np.nan, "float32") for k in ("voronoi", "voronoi_ball", "tti", "tti_ball")}
    team = {}
    ok = ~np.isnan(P[:, 0])
    idx = np.where(ok)[0]
    owner = idx[cKDTree(P[idx]).query(cells)[1]]
    has_ball = not np.isnan(ball[0])
    w = np.exp(-((cells - ball) ** 2).sum(1) / (2 * CONFIG["BALL_SIGMA_M"] ** 2)) if has_ball else np.zeros(len(cells))
    out["voronoi"][idx] = np.bincount(owner, minlength=N)[idx] * area
    out["voronoi_ball"][idx] = np.bincount(owner, weights=w, minlength=N)[idx] * area
    hm = is_home[owner]
    team["home_voronoi"] = hm.mean()
    team["home_voronoi_ball"] = (w * hm).sum() / w.sum() if has_ball else np.nan
    if use_tti:
        okv = idx[~np.isnan(V[idx, 0])]
        h = is_home[okv]
        if h.any() and (~h).any():
            beta = np.pi / (np.sqrt(3) * CONFIG["TTI_SIGMA"])
            r = P[okv] + V[okv] * CONFIG["TTI_REACTION_S"]
            tti = CONFIG["TTI_REACTION_S"] + np.sqrt(((cells[None] - r[:, None]) ** 2).sum(-1)) / CONFIG["TTI_VMAX"]
            p_home = 1 / (1 + np.exp(-beta * (tti[~h].min(0) - tti[h].min(0))))
            best_h, best_a = okv[h][tti[h].argmin(0)], okv[~h][tti[~h].argmin(0)]
            share = np.bincount(best_h, weights=p_home, minlength=N) + np.bincount(best_a, weights=1 - p_home, minlength=N)
            share_b = np.bincount(best_h, weights=p_home * w, minlength=N) + np.bincount(best_a, weights=(1 - p_home) * w, minlength=N)
            out["tti"][okv], out["tti_ball"][okv] = share[okv] * area, share_b[okv] * area
            team["home_tti"] = p_home.mean()
            team["home_tti_ball"] = (p_home * w).sum() / w.sum() if has_ball else np.nan
    return out, team


# ---------------------------------------------------------------- build one match
def build_match_tables(match_id, force=False, allow_holdout=False, verbose=True):
    """Build (or read from cache) the four base tables of one match."""
    check_holdout(match_id, allow_holdout)
    paths = {n: cache_dir() / f"{match_id}_{n}.parquet" for n in TABLE_NAMES}
    if not force and all(p.exists() for p in paths.values()):
        return {n: pd.read_parquet(p) for n, p in paths.items()}

    t0 = time.time()
    raw = load_raw_match(match_id, allow_holdout)
    fr, hz = CONFIG["FRAMERATE"], CONFIG["SAMPLE_HZ"]
    step = fr // hz
    cells = pitch_grid()
    use_tti = "tti" in CONFIG["PITCH_CONTROL"]

    # Players in tracking-column order (xID): Home columns first, then Away
    ts = {s: raw["teamsheets"][s].sort_values("xID").reset_index(drop=True) for s in ("Home", "Away")}
    players = pd.concat([ts[s].assign(side=s) for s in ("Home", "Away")], ignore_index=True)
    players["pos_group"] = players["position"].map(POSITION_GROUP)
    players["match"] = match_id
    team_name = {s: ts[s]["team"].iloc[0] for s in ("Home", "Away")}

    samples, team_rows, event_rows = [], [], []
    for half in HALVES:
        XH, XA, XB = raw["xy"][half]["Home"], raw["xy"][half]["Away"], raw["xy"][half]["Ball"]
        NH, NA = XH.shape[1] // 2, XA.shape[1] // 2
        N = NH + NA
        allxy = np.c_[XH, XA]
        speed, acc, vel = kinematics(allxy)
        mp = metabolic_power(speed, acc)
        dist = speed / fr
        nb = allxy.shape[0] // step

        def per_sample(z):
            # Sum 25 Hz quantities into each analysis sample
            return np.nansum(z[: nb * step].reshape(nb, step, -1), axis=1)

        load = {
            "dist": per_sample(dist),
            "hsr": per_sample(np.where(speed >= CONFIG["HSR_SPEED"], dist, 0)),
            "sprint": per_sample(np.where(speed >= CONFIG["SPRINT_SPEED"], dist, 0)),
            "acc_time": per_sample(np.where(acc >= CONFIG["ACC_THRESHOLD"], 1 / fr, 0)),
            "dec_time": per_sample(np.where(acc <= -CONFIG["ACC_THRESHOLD"], 1 / fr, 0)),
            "met_energy": per_sample(mp / fr),
        }
        frames = np.arange(nb) * step
        inplay = raw["ball_status"][half][frames] == 1
        poss = raw["poss"][half][frames]
        ball = XB[frames, :2]
        # Attacking direction: the home team defends the side where its players stand on average
        home_dir = float(-np.sign(np.nanmean(XH[:, 0::2])))
        side_dir = {"Home": home_dir, "Away": -home_dir}
        is_home = np.arange(N) < NH

        pc = {k: np.full((nb, N), np.nan, "float32") for k in ("voronoi", "voronoi_ball", "tti", "tti_ball")}
        dball = np.full((nb, N), np.nan, "float32")
        team_pc = {k: np.full(nb, np.nan, "float32") for k in ("home_voronoi", "home_voronoi_ball", "home_tti", "home_tti_ball")}
        for k in np.where(inplay)[0]:
            P = allxy[frames[k]].reshape(-1, 2)
            ok = ~np.isnan(P[:, 0])
            if (ok & is_home).sum() < 7 or (ok & ~is_home).sum() < 7:
                continue
            out, team = pitch_control_sample(P, vel[frames[k]].reshape(-1, 2), is_home, ball[k], cells, use_tti)
            for c in out:
                pc[c][k] = out[c]
            for c, v in team.items():
                team_pc[c][k] = v
            if not np.isnan(ball[k, 0]):
                dball[k, ok] = np.hypot(*(P[ok] - ball[k]).T)

        t_sec = frames / fr
        minute = (t_sec // 60).astype(int) + (45 if half == "secondHalf" else 0)
        onpitch = ~np.isnan(allxy[frames][:, 0::2])
        pc_cols = ["voronoi", "voronoi_ball"] + (["tti", "tti_ball"] if use_tti else [])
        for j in range(N):
            m = onpitch[:, j] & inplay
            if not m.any():
                continue
            side = "Home" if j < NH else "Away"
            df = pd.DataFrame({
                "match": match_id, "half": half, "sample": np.where(m)[0].astype("int32"),
                "t": t_sec[m].astype("float32"), "minute": minute[m].astype("int16"),
                "pID": players.loc[j, "pID"], "side": side,
                "in_possession": poss[m] == (1 if side == "Home" else 2),
                "speed": (load["dist"][m, j] * hz).astype("float32"),
                **{c: load[c][m, j].astype("float32") for c in LOAD_COLS},
                **{c: pc[c][m, j] for c in pc_cols},
                "dist_ball": dball[m, j],
                "attack_x": (allxy[frames[m], 2 * j] * side_dir[side]).astype("float32"),
            })
            samples.append(df)

        tdf = pd.DataFrame({
            "match": match_id, "half": half, "sample": np.arange(nb, dtype="int32"), "t": t_sec.astype("float32"),
            "minute": minute.astype("int16"), "inplay": inplay,
            "poss_side": np.select([poss == 1, poss == 2], ["Home", "Away"], "None"),
            "ball_x": ball[:, 0], "ball_y": ball[:, 1], "home_dir": home_dir,
            "home_voronoi": team_pc["home_voronoi"], "home_voronoi_ball": team_pc["home_voronoi_ball"],
            "home_hsr": load["hsr"][:, :NH].sum(1).astype("float32"), "away_hsr": load["hsr"][:, NH:].sum(1).astype("float32"),
            "home_dist": load["dist"][:, :NH].sum(1).astype("float32"), "away_dist": load["dist"][:, NH:].sum(1).astype("float32"),
        })
        if use_tti:
            tdf["home_tti"], tdf["home_tti_ball"] = team_pc["home_tti"], team_pc["home_tti_ball"]
        team_rows.append(tdf)

        # 3.4 Events aligned to analysis samples (gameclock is seconds from the start of the half)
        for side in ("Home", "Away"):
            e = raw["events"][half][side].copy()
            e["match"], e["half"], e["side"], e["team"] = match_id, half, side, team_name[side]
            e["sample"] = np.clip((e["gameclock"] * hz).round(), 0, nb - 1).astype("Int64")
            e["minute"] = (e["gameclock"] // 60).astype("Int64") + (45 if half == "secondHalf" else 0)
            e["attack_x"] = e["at_x"] * side_dir[side]
            event_rows.append(e)

    events = pd.concat(event_rows, ignore_index=True)
    events["eID"] = events["eID"].astype(str)
    events["is_shot"] = events["eID"].str.startswith("ShotAtGoal")
    events["is_goal"] = events["eID"].eq("ShotAtGoal_SuccessfulShot")
    events["is_pass"] = events["eID"].str.contains("Pass")
    events["is_cross"] = events["eID"].str.contains("Cross")
    events["is_foul"] = events["eID"].eq("Foul")
    events["is_tackle"] = events["eID"].eq("TacklingGame")
    for c in ("outcome", "qualifier", "tID", "pID"):
        events[c] = events[c].astype(str)

    tables = {"samples": pd.concat(samples, ignore_index=True), "team_samples": pd.concat(team_rows, ignore_index=True),
              "events": events, "players": players.astype({"jID": "Int64", "xID": "Int64"})}
    for n, df in tables.items():
        df.to_parquet(paths[n], index=False)
    if verbose:
        print(f"{match_id}: built in {time.time() - t0:,.0f} s")
    return tables


# ---------------------------------------------------------------- 3.5 aggregations
def make_player_minute(samples):
    """Player x minute x phase: load sums, pitch-control means, number of samples."""
    keys = ["match", "half", "minute", "pID", "player", "team", "side", "pos_group", "in_possession"]
    ctrl = [c for c in ("voronoi", "voronoi_ball", "tti", "tti_ball", "dist_ball", "attack_x") if c in samples]
    agg = {c: "sum" for c in LOAD_COLS} | {c: "mean" for c in ctrl}
    out = samples.groupby(keys, observed=True, dropna=False).agg(agg)
    out["n_samples"] = samples.groupby(keys, observed=True, dropna=False).size()
    return out.reset_index()


def make_team_minute(team_samples, events):
    """Team x minute (one row per side): possession, control, load, ball position, shots."""
    ts = team_samples[team_samples.inplay].copy()
    ts["home_possession"] = (ts.poss_side == "Home").astype(float)
    ts["ball_x_home"] = ts.ball_x * ts.home_dir
    cols = {"home_possession": "mean", "home_voronoi": "mean", "home_voronoi_ball": "mean", "ball_x_home": "mean",
            "home_hsr": "sum", "away_hsr": "sum", "home_dist": "sum", "away_dist": "sum"}
    if "home_tti" in ts:
        cols |= {"home_tti": "mean", "home_tti_ball": "mean"}
    g = ts.groupby(["match", "half", "minute"]).agg(cols)
    g["n_samples"] = ts.groupby(["match", "half", "minute"]).size()
    g = g.reset_index()
    rows = []
    for side in ("Home", "Away"):
        home = side == "Home"
        r = g[["match", "half", "minute", "n_samples"]].copy()
        r["side"] = side
        r["possession"] = g.home_possession if home else 1 - g.home_possession
        r["control"] = g.home_voronoi if home else 1 - g.home_voronoi
        r["control_ball"] = g.home_voronoi_ball if home else 1 - g.home_voronoi_ball
        if "home_tti" in g:
            r["control_tti"] = g.home_tti if home else 1 - g.home_tti
            r["control_tti_ball"] = g.home_tti_ball if home else 1 - g.home_tti_ball
        r["ball_attack_x"] = g.ball_x_home * (1 if home else -1)
        r["hsr"], r["hsr_opp"] = (g.home_hsr, g.away_hsr) if home else (g.away_hsr, g.home_hsr)
        r["dist"], r["dist_opp"] = (g.home_dist, g.away_dist) if home else (g.away_dist, g.home_dist)
        rows.append(r)
    tm = pd.concat(rows, ignore_index=True)
    ev = events.groupby(["match", "half", "minute", "side"])[["is_shot", "is_goal"]].sum()
    ev = ev.rename(columns={"is_shot": "shots", "is_goal": "goals"}).reset_index()
    ev["minute"] = ev["minute"].astype("int64")
    tm["minute"] = tm["minute"].astype("int64")
    tm = tm.merge(ev, on=["match", "half", "minute", "side"], how="left").fillna({"shots": 0, "goals": 0})
    return tm.sort_values(["match", "half", "side", "minute"]).reset_index(drop=True)


@dataclass
class Tables:
    samples: pd.DataFrame
    team_samples: pd.DataFrame
    events: pd.DataFrame
    players: pd.DataFrame
    player_minute: pd.DataFrame
    team_minute: pd.DataFrame

    def summary(self):
        return pd.DataFrame({n: {"rows": len(getattr(self, n)), "columns": getattr(self, n).shape[1]}
                             for n in self.__dataclass_fields__}).T


def load_tables(matches=None, force=False):
    """Build or load every non-holdout match and stack them into the shared tables."""
    matches = list(matches or discover_matches())
    if not matches:
        raise FileNotFoundError("no matches found: set BRAMEX_DATA_DIR to the folder with the DFL XML files")
    parts = {n: [] for n in TABLE_NAMES}
    for i, m in enumerate(matches, 1):
        t = build_match_tables(m, force=force)
        for n in TABLE_NAMES:
            parts[n].append(t[n])
        print(f"[{i}/{len(matches)}] {m} ready")
    d = {n: pd.concat(v, ignore_index=True) for n, v in parts.items()}
    meta = d["players"][["match", "pID", "player", "team", "pos_group"]]
    d["samples"] = d["samples"].merge(meta, on=["match", "pID"], how="left")
    return Tables(**d, player_minute=make_player_minute(d["samples"]),
                  team_minute=make_team_minute(d["team_samples"], d["events"]))
