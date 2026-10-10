#!/usr/bin/env python3
"""
EDGE NBA data converter — cleaned NBA-only version.

Inputs:
  playerlog_<season>.parquet
  gamelog_<season>.parquet
  team_summary_<season>.json   (optional)

Outputs:
  nba_stats_<season>.json

This version:
  - keeps only the 30 NBA teams
  - keeps REGULAR-SEASON games only (NBA game-id prefix 002)
  - keeps only player rows matched to the cleaned NBA gamelog
  - removes exhibition / celebrity / international / playoff noise
  - prevents unmatched playerlog rows from inflating games played
  - builds player season / last 10 / last 5 averages
  - builds team intelligence
  - builds temporary G / F / C position-concession intelligence
  - leaves PG / SG / SF / PF / C for the ESPN depth-chart layer
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NBA_TEAMS = {
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
    "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
    "OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS"
}

METRICS = {
    "min": ["minutes", "min"],
    "pts": ["points", "pts"],
    "reb": ["reboundsTotal", "reb", "rebounds"],
    "ast": ["assists", "ast"],
    "stl": ["steals", "stl"],
    "blk": ["blocks", "blk"],
    "three_pm": ["threePointersMade", "fg3m", "three_pm"],
    "tov": ["turnovers", "tov"],
    "usage": ["usagePercentage", "usage_pct", "usg_pct"],
    "ts_pct": ["trueShootingPercentage", "ts_pct"],
    "plus_minus": ["plusMinusPoints", "plus_minus"],
}

PLAYER_ID_CANDIDATES = ["personId", "person_id", "player_id"]
PLAYER_FIRST_CANDIDATES = ["firstName", "first_name"]
PLAYER_LAST_CANDIDATES = ["familyName", "family_name", "lastName", "last_name"]
PLAYER_NAME_CANDIDATES = ["player", "playerName", "player_name", "name"]
PLAYER_TEAM_ID_CANDIDATES = ["teamId", "team_id"]
PLAYER_TEAM_CODE_CANDIDATES = ["teamTricode", "team_abbreviation", "team"]
PLAYER_POSITION_CANDIDATES = ["position", "pos"]
PLAYER_GAME_ID_CANDIDATES = ["gameId", "game_id"]

GAME_ID_CANDIDATES = ["game_id", "gameId"]
GAME_DATE_CANDIDATES = ["game_date", "gameDate"]
GAME_TEAM_ID_CANDIDATES = ["team_id", "teamId"]
GAME_TEAM_CODE_CANDIDATES = ["team_abbreviation", "teamTricode", "team"]
GAME_MATCHUP_CANDIDATES = ["matchup"]
GAME_WL_CANDIDATES = ["wl", "WL"]
GAME_PACE_CANDIDATES = ["pace", "PACE"]
GAME_OFFRTG_CANDIDATES = ["off_rating", "OFF_RATING", "offensiveRating"]
GAME_DEFRTG_CANDIDATES = ["def_rating", "DEF_RATING", "defensiveRating"]
GAME_NETRTG_CANDIDATES = ["net_rating", "NET_RATING", "netRating"]
GAME_POSS_CANDIDATES = ["poss", "POSS", "possessions"]


def first_existing(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return None


def safe_num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        n = float(v)
    except Exception:
        return None
    return n if math.isfinite(n) else None


def round_or_none(v: Any, digits: int = 2) -> float | None:
    n = safe_num(v)
    return None if n is None else round(n, digits)


def clean_minutes(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None

    if isinstance(v, (int, float, np.integer, np.floating)):
        n = float(v)
        # Invalid / non-participation values occasionally appear in source data.
        return n if n >= 0 else None

    s = str(v).strip()
    if not s:
        return None

    if ":" in s:
        try:
            mm, ss = s.split(":", 1)
            n = float(mm) + float(ss) / 60.0
            return n if n >= 0 else None
        except Exception:
            return None

    try:
        n = float(s)
        return n if n >= 0 else None
    except Exception:
        return None


def player_name(row: pd.Series, df: pd.DataFrame) -> str:
    name_col = first_existing(df, PLAYER_NAME_CANDIDATES, required=False)
    if name_col:
        val = str(row.get(name_col, "") or "").strip()
        if val:
            return val

    fcol = first_existing(df, PLAYER_FIRST_CANDIDATES, required=False)
    lcol = first_existing(df, PLAYER_LAST_CANDIDATES, required=False)

    first = str(row.get(fcol, "") or "").strip() if fcol else ""
    last = str(row.get(lcol, "") or "").strip() if lcol else ""
    return (first + " " + last).strip()


def normalize_position_group(raw: Any) -> str:
    """
    Temporary broad-position layer.

    NBA source rows are commonly G / F / C or blank rather than true
    PG / SG / SF / PF / C. We therefore use:
      G = any guard-labelled row
      F = any forward-labelled row
      C = center-labelled row
      UNKNOWN = no defensible label

    Exact five-position mapping will be added from ESPN depth charts.
    """
    if raw is None:
        return "UNKNOWN"

    s = str(raw).upper().replace(" ", "").strip()
    if not s:
        return "UNKNOWN"

    if s == "C" or "CENTER" in s:
        return "C"

    if s in {"PG", "SG", "G"} or "GUARD" in s:
        return "G"

    if s in {"SF", "PF", "F"} or "FORWARD" in s:
        return "F"

    # Hybrid source labels such as G-F / F-G.
    if "G" in s and "F" in s:
        return "G/F"

    return "UNKNOWN"


def normalize_game_id(v: Any) -> str:
    """
    Normalize NBA game ids to their standard 10-digit form.

    This handles Parquet columns that may store ids numerically and therefore
    drop leading zeroes. Example:
      22500001 -> 0022500001
    """
    if v is None:
        return ""

    s = str(v).strip()

    # Handle numeric-looking strings such as "22500001.0".
    if s.endswith(".0"):
        s = s[:-2]

    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return ""

    return digits.zfill(10)


def is_regular_season_game_id(v: Any) -> bool:
    """
    NBA game-id convention:
      001 = preseason
      002 = regular season
      003 = All-Star
      004 = playoffs
      005 = play-in / other postseason bucket in some feeds

    EDGE player season / L10 / L5 datasets use regular season only so that
    season averages and games played are directly comparable.
    """
    return normalize_game_id(v).startswith("002")


def parse_opponent_and_home(matchup: Any) -> tuple[str | None, bool | None]:
    if matchup is None:
        return None, None

    s = str(matchup).strip().upper()

    if " VS. " in s:
        _, right = s.split(" VS. ", 1)
        return right.strip(), True

    if " VS " in s:
        _, right = s.split(" VS ", 1)
        return right.strip(), True

    if " @ " in s:
        _, right = s.split(" @ ", 1)
        return right.strip(), False

    return None, None


def mean_record(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key in METRICS:
        if key in frame.columns and len(frame):
            value = pd.to_numeric(frame[key], errors="coerce").mean()
            out[key] = round_or_none(
                value,
                3 if key in {"usage", "ts_pct"} else 2
            )
        else:
            out[key] = None

    return out


def rank_desc(values: pd.Series) -> pd.Series:
    # #1 = most conceded.
    return values.rank(
        method="min",
        ascending=False,
        na_option="bottom"
    ).astype("Int64")


def prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()

    game_col = first_existing(g, GAME_ID_CANDIDATES)
    team_id_col = first_existing(g, GAME_TEAM_ID_CANDIDATES, required=False)
    team_col = first_existing(g, GAME_TEAM_CODE_CANDIDATES)
    date_col = first_existing(g, GAME_DATE_CANDIDATES)
    matchup_col = first_existing(g, GAME_MATCHUP_CANDIDATES)
    wl_col = first_existing(g, GAME_WL_CANDIDATES, required=False)

    g["game_id_norm"] = g[game_col].map(normalize_game_id)
    g["team"] = g[team_col].astype(str).str.upper().str.strip()
    g["game_date"] = pd.to_datetime(g[date_col], errors="coerce")
    g["matchup_norm"] = g[matchup_col].astype(str)

    if team_id_col:
        g["team_id_norm"] = g[team_id_col].astype(str)

    parsed = g["matchup_norm"].map(parse_opponent_and_home)
    g["opponent"] = parsed.map(lambda x: x[0])
    g["home"] = parsed.map(lambda x: x[1])

    if wl_col:
        g["wl"] = g[wl_col].astype(str).str.upper()
    else:
        g["wl"] = None

    # Core clean-up:
    # 1) regular season only
    # 2) NBA team vs NBA team only
    # 3) valid dated rows only
    g = g[
        g["game_id_norm"].str.startswith("002")
        & g["team"].isin(NBA_TEAMS)
        & g["opponent"].isin(NBA_TEAMS)
        & g["game_date"].notna()
    ].copy()

    # Remove accidental duplicates while preserving one team row per game.
    dedupe_keys = ["game_id_norm", "team"]
    g = g.sort_values("game_date").drop_duplicates(dedupe_keys, keep="last")

    return g


def prepare_player_games(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    g = prepare_games(games)

    p_game = first_existing(p, PLAYER_GAME_ID_CANDIDATES)
    p_team_id = first_existing(p, PLAYER_TEAM_ID_CANDIDATES, required=False)
    p_team_code = first_existing(p, PLAYER_TEAM_CODE_CANDIDATES, required=False)
    p_pos = first_existing(p, PLAYER_POSITION_CANDIDATES, required=False)
    p_pid = first_existing(p, PLAYER_ID_CANDIDATES, required=False)

    p["game_id_norm"] = p[p_game].map(normalize_game_id)

    if p_team_id:
        p["team_id_norm"] = p[p_team_id].astype(str)

    if p_team_code:
        p["team"] = p[p_team_code].astype(str).str.upper().str.strip()
    else:
        p["team"] = None

    p["player_id"] = p[p_pid].astype(str) if p_pid else None
    p["player"] = p.apply(lambda r: player_name(r, p), axis=1)
    p["position_raw"] = p[p_pos] if p_pos else None
    p["position_group"] = p["position_raw"].map(normalize_position_group)

    for out_name, candidates in METRICS.items():
        col = first_existing(p, candidates, required=False)

        if col:
            if out_name == "min":
                p[out_name] = p[col].map(clean_minutes)
            else:
                p[out_name] = pd.to_numeric(p[col], errors="coerce")
        else:
            p[out_name] = np.nan

    # Restrict source player rows to NBA team codes before joining.
    p = p[p["team"].isin(NBA_TEAMS)].copy()

    join_keys = ["game_id_norm"]

    if p_team_id and "team_id_norm" in g.columns:
        join_keys.append("team_id_norm")
    else:
        join_keys.append("team")

    game_keep = join_keys + [
        "game_date",
        "matchup_norm",
        "opponent",
        "home",
        "wl",
    ]

    merged = p.merge(
        g[game_keep].drop_duplicates(join_keys),
        on=join_keys,
        how="inner"   # IMPORTANT: unmatched playerlog rows are excluded.
    )

    # Only real played rows should count toward averages / games played.
    # DNP rows typically have no valid minutes.
    merged = merged[merged["min"].notna()].copy()

    # Defensive dedupe: one row per player/team/game.
    merged = (
        merged.sort_values(["game_date", "game_id_norm"])
        .drop_duplicates(
            ["player_id", "player", "team", "game_id_norm"],
            keep="last"
        )
    )

    return merged.sort_values(
        ["player_id", "player", "game_date", "game_id_norm"]
    )


def build_players(pg: pd.DataFrame) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for _, grp in pg.groupby(["player_id", "player"], dropna=False):
        grp = grp.sort_values("game_date")
        latest = grp.iloc[-1]

        results.append({
            "player_id": (
                None if pd.isna(latest.get("player_id"))
                else str(latest.get("player_id"))
            ),
            "player": str(latest.get("player") or ""),
            "team": str(latest.get("team") or ""),
            "position_group": str(
                latest.get("position_group") or "UNKNOWN"
            ),
            "position_raw": (
                None if pd.isna(latest.get("position_raw"))
                else str(latest.get("position_raw"))
            ),
            "games_played": int(grp["game_id_norm"].nunique()),
            "season": mean_record(grp),
            "last_10": mean_record(grp.tail(10)),
            "last_5": mean_record(grp.tail(5)),
        })

    results.sort(
        key=lambda x: (
            x["team"],
            -(x["season"].get("min") or 0),
            x["player"]
        )
    )

    return results


def build_position_concessions(pg: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Temporary G/F/C concessions.

    G/F hybrids are excluded rather than guessed. Once ESPN depth charts
    are wired in, this becomes PG/SG/SF/PF/C.
    """
    valid = pg[
        pg["opponent"].isin(NBA_TEAMS)
        & pg["position_group"].isin(["G", "F", "C"])
    ].copy()

    stat_cols = ["pts", "reb", "ast", "stl", "blk", "three_pm"]

    for c in stat_cols:
        valid[c] = pd.to_numeric(valid[c], errors="coerce").fillna(0)

    per_game = (
        valid.groupby(
            ["opponent", "game_id_norm", "game_date", "position_group"],
            dropna=False
        )[stat_cols]
        .sum()
        .reset_index()
        .rename(
            columns={
                "opponent": "team",
                "position_group": "position"
            }
        )
    )

    season = (
        per_game.groupby(["team", "position"])[stat_cols]
        .mean()
        .reset_index()
    )

    game_counts = (
        per_game.groupby(["team", "position"])["game_id_norm"]
        .nunique()
        .rename("games")
        .reset_index()
    )

    season = season.merge(
        game_counts,
        on=["team", "position"],
        how="left"
    )

    period_rows: dict[tuple[str, str], dict[str, Any]] = {}

    for (team, pos), grp in per_game.groupby(["team", "position"]):
        grp = grp.sort_values("game_date")

        period_rows[(team, pos)] = {
            "last_10": {
                c: round_or_none(grp.tail(10)[c].mean(), 2)
                for c in stat_cols
            },
            "last_5": {
                c: round_or_none(grp.tail(5)[c].mean(), 2)
                for c in stat_cols
            },
        }

    for c in stat_cols:
        season[f"{c}_rank"] = (
            season.groupby("position")[c]
            .transform(rank_desc)
        )

    out = []

    order = {"G": 0, "F": 1, "C": 2}

    for _, row in season.iterrows():
        key = (row["team"], row["position"])

        out.append({
            "team": row["team"],
            "position": row["position"],
            "games": int(row["games"]),
            "season": {
                c: round_or_none(row[c], 2)
                for c in stat_cols
            },
            "ranks_most_conceded": {
                c: (
                    int(row[f"{c}_rank"])
                    if not pd.isna(row[f"{c}_rank"])
                    else None
                )
                for c in stat_cols
            },
            "last_10": period_rows[key]["last_10"],
            "last_5": period_rows[key]["last_5"],
        })

    out.sort(
        key=lambda x: (
            x["team"],
            order.get(x["position"], 99)
        )
    )

    return out


def load_team_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_team_intelligence(
    games: pd.DataFrame,
    summary: dict[str, Any] | None
) -> list[dict[str, Any]]:

    clean_games = prepare_games(games)

    if summary and isinstance(summary.get("teams"), dict):
        out = []

        for code, t in summary["teams"].items():
            if code not in NBA_TEAMS:
                continue

            out.append({
                "team": code,
                "team_id": t.get("TEAM_ID"),
                "team_name": t.get("TEAM_NAME"),
                "games_played": t.get("GP"),
                "wins": t.get("W"),
                "losses": t.get("L"),
                "off_rating": round_or_none(t.get("OFF_RATING"), 2),
                "def_rating": round_or_none(t.get("DEF_RATING"), 2),
                "net_rating": round_or_none(t.get("NET_RATING"), 2),
                "pace": round_or_none(t.get("PACE"), 2),
                "possessions": round_or_none(t.get("POSS"), 0),
                "ast_pct": round_or_none(t.get("AST_PCT"), 3),
                "ast_to": round_or_none(t.get("AST_TO"), 2),
                "oreb_pct": round_or_none(t.get("OREB_PCT"), 3),
                "dreb_pct": round_or_none(t.get("DREB_PCT"), 3),
                "reb_pct": round_or_none(t.get("REB_PCT"), 3),
                "tov_pct": round_or_none(t.get("TM_TOV_PCT"), 3),
                "efg_pct": round_or_none(t.get("EFG_PCT"), 3),
                "ts_pct": round_or_none(t.get("TS_PCT"), 3),
            })

        return sorted(out, key=lambda x: x["team"])

    mapping = {
        "off_rating": first_existing(
            clean_games, GAME_OFFRTG_CANDIDATES, required=False
        ),
        "def_rating": first_existing(
            clean_games, GAME_DEFRTG_CANDIDATES, required=False
        ),
        "net_rating": first_existing(
            clean_games, GAME_NETRTG_CANDIDATES, required=False
        ),
        "pace": first_existing(
            clean_games, GAME_PACE_CANDIDATES, required=False
        ),
        "possessions": first_existing(
            clean_games, GAME_POSS_CANDIDATES, required=False
        ),
    }

    out = []

    for team, grp in clean_games.groupby("team"):
        rec = {
            "team": str(team),
            "games_played": int(grp["game_id_norm"].nunique()),
            "wins": int((grp["wl"] == "W").sum()),
            "losses": int((grp["wl"] == "L").sum()),
        }

        for out_key, col in mapping.items():
            rec[out_key] = (
                round_or_none(
                    pd.to_numeric(grp[col], errors="coerce").mean(),
                    2
                )
                if col else None
            )

        out.append(rec)

    return sorted(out, key=lambda x: x["team"])


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    players = payload["players"]
    teams = payload["teams"]
    concessions = payload["position_concessions"]

    team_codes = sorted({t["team"] for t in teams})
    player_teams = sorted({p["team"] for p in players})

    max_gp = max(
        (p["games_played"] for p in players),
        default=0
    )

    positions = sorted({
        x["position"]
        for x in concessions
    })

    return {
        "team_count": len(team_codes),
        "team_codes": team_codes,
        "non_nba_player_teams": [
            t for t in player_teams
            if t not in NBA_TEAMS
        ],
        "player_count": len(players),
        "max_player_games": max_gp,
        "concession_positions": positions,
        "concession_rows": len(concessions),
        "regular_season_only": True,
        "expected_max_player_games": 82,
        "games_over_82": [
            {
                "player": p["player"],
                "team": p["team"],
                "games_played": p["games_played"]
            }
            for p in players
            if p["games_played"] > 82
        ],
    }


def build_season(
    input_dir: Path,
    output_dir: Path,
    season: int
) -> Path:

    pfile = input_dir / f"playerlog_{season}.parquet"
    gfile = input_dir / f"gamelog_{season}.parquet"
    sfile = input_dir / f"team_summary_{season}.json"

    if not pfile.exists():
        raise FileNotFoundError(pfile)

    if not gfile.exists():
        raise FileNotFoundError(gfile)

    players = pd.read_parquet(pfile)
    games = pd.read_parquet(gfile)
    summary = load_team_summary(sfile)

    pg = prepare_player_games(players, games)

    payload = {
        "success": True,
        "dataset": "edge_nba_intelligence",
        "season": season,
        "season_label": f"{season-1}-{str(season)[-2:]}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "llimllib/nba_data",
        "players": build_players(pg),
        "teams": build_team_intelligence(games, summary),
        "position_concessions": build_position_concessions(pg),
        "notes": {
            "scope": "NBA regular season only; NBA teams only; player rows must match the cleaned NBA gamelog.",
            "position_concession_rank": "#1 means most conceded.",
            "position_model": "Temporary G/F/C model. ESPN depth charts will later map active players to PG/SG/SF/PF/C.",
            "first_basket": "Not included. Requires play-by-play data."
        }
    }

    payload["validation"] = validate_payload(payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"nba_stats_{season}.json"

    with outfile.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False
        )

    print(
        f"{season}: "
        f"{payload['validation']['team_count']} teams | "
        f"{payload['validation']['player_count']} players | "
        f"max GP {payload['validation']['max_player_games']} | "
        f"over82 {len(payload['validation']['games_over_82'])} | "
        f"positions {payload['validation']['concession_positions']}"
    )

    return outfile


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=".",
        help="Folder containing NBA source files"
    )

    parser.add_argument(
        "--output",
        default=".",
        help="Output folder"
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2025, 2026]
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    for season in args.seasons:
        out = build_season(input_dir, output_dir, season)
        print(f"Built: {out}")


if __name__ == "__main__":
    main()
