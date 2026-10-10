#!/usr/bin/env python3
"""
EDGE NBA data converter

Inputs expected in one folder:
  playerlog_2025.parquet
  gamelog_2025.parquet
  playerlog_2026.parquet
  gamelog_2026.parquet
  team_summary_2026.json   (optional; team_summary_2025.json also supported)

Outputs:
  nba_stats_2025.json
  nba_stats_2026.json

Builds:
  - player season / last 10 / last 5 averages
  - team intelligence
  - position concessions by opponent
  - league ranks for position concessions

Install:
  pip install pandas pyarrow

Run:
  python edge_nba_converter.py --input . --output .
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


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
    if not math.isfinite(n):
        return None
    return n


def round_or_none(v: Any, digits: int = 2) -> float | None:
    n = safe_num(v)
    return None if n is None else round(n, digits)


def clean_minutes(v: Any) -> float | None:
    """Handles numeric minutes or MM:SS strings."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    if ":" in s:
        try:
            mm, ss = s.split(":", 1)
            return float(mm) + float(ss) / 60.0
        except Exception:
            pass
    try:
        return float(s)
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


def normalize_position(raw: Any) -> str:
    if raw is None:
        return "UNKNOWN"

    s = str(raw).upper().replace(" ", "").strip()

    if not s:
        return "UNKNOWN"

    exact = {"PG", "SG", "SF", "PF", "C"}

    if s in exact:
        return s

    aliases = {
        "POINTGUARD": "PG",
        "SHOOTINGGUARD": "SG",
        "SMALLFORWARD": "SF",
        "POWERFORWARD": "PF",
        "CENTER": "C",
    }

    if s in aliases:
        return aliases[s]

    return "HYBRID"


def parse_opponent_and_home(matchup: Any, team_code: str) -> tuple[str | None, bool | None]:
    if matchup is None:
        return None, None

    s = str(matchup).strip().upper()

    if " VS. " in s:
        left, right = s.split(" VS. ", 1)
        return right.strip(), True

    if " VS " in s:
        left, right = s.split(" VS ", 1)
        return right.strip(), True

    if " @ " in s:
        left, right = s.split(" @ ", 1)
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
    return values.rank(
        method="min",
        ascending=False,
        na_option="bottom"
    ).astype("Int64")


def prepare_player_games(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    g = games.copy()

    p_game = first_existing(p, PLAYER_GAME_ID_CANDIDATES)
    p_team_id = first_existing(p, PLAYER_TEAM_ID_CANDIDATES, required=False)
    p_team_code = first_existing(p, PLAYER_TEAM_CODE_CANDIDATES, required=False)
    p_pos = first_existing(p, PLAYER_POSITION_CANDIDATES, required=False)
    p_pid = first_existing(p, PLAYER_ID_CANDIDATES, required=False)

    g_game = first_existing(g, GAME_ID_CANDIDATES)
    g_team_id = first_existing(g, GAME_TEAM_ID_CANDIDATES, required=False)
    g_team_code = first_existing(g, GAME_TEAM_CODE_CANDIDATES, required=False)
    g_date = first_existing(g, GAME_DATE_CANDIDATES)
    g_matchup = first_existing(g, GAME_MATCHUP_CANDIDATES)
    g_wl = first_existing(g, GAME_WL_CANDIDATES, required=False)

    p["game_id_norm"] = p[p_game].astype(str)

    if p_team_id:
        p["team_id_norm"] = p[p_team_id].astype(str)

    if p_team_code:
        p["team"] = p[p_team_code].astype(str).str.upper().str.strip()
    else:
        p["team"] = None

    p["player_id"] = p[p_pid].astype(str) if p_pid else None
    p["player"] = p.apply(lambda r: player_name(r, p), axis=1)
    p["position_raw"] = p[p_pos] if p_pos else None
    p["position"] = p["position_raw"].map(normalize_position)

    for out_name, candidates in METRICS.items():
        col = first_existing(p, candidates, required=False)

        if col:
            if out_name == "min":
                p[out_name] = p[col].map(clean_minutes)
            else:
                p[out_name] = pd.to_numeric(p[col], errors="coerce")
        else:
            p[out_name] = np.nan

    g["game_id_norm"] = g[g_game].astype(str)

    if g_team_id:
        g["team_id_norm"] = g[g_team_id].astype(str)

    if g_team_code:
        g["team_game"] = g[g_team_code].astype(str).str.upper().str.strip()

    g["game_date"] = pd.to_datetime(g[g_date], errors="coerce")
    g["matchup_norm"] = g[g_matchup].astype(str)

    join_keys = ["game_id_norm"]

    if p_team_id and g_team_id:
        join_keys.append("team_id_norm")

    elif p_team_code and g_team_code:
        p = p.rename(columns={"team": "team_join"})
        g = g.rename(columns={"team_game": "team_join"})
        join_keys.append("team_join")

    keep = join_keys + ["game_date", "matchup_norm"]

    if g_wl:
        keep.append(g_wl)

    merged = p.merge(
        g[keep].drop_duplicates(join_keys),
        on=join_keys,
        how="left"
    )

    if "team_join" in merged.columns:
        merged["team"] = merged["team_join"]

    opp_home = merged.apply(
        lambda r: parse_opponent_and_home(
            r.get("matchup_norm"),
            str(r.get("team", ""))
        ),
        axis=1,
        result_type="expand",
    )

    merged["opponent"] = opp_home[0]
    merged["home"] = opp_home[1]

    if g_wl and g_wl in merged.columns:
        merged["wl"] = merged[g_wl].astype(str)
    else:
        merged["wl"] = None

    merged = merged.sort_values(
        ["player_id", "player", "game_date", "game_id_norm"]
    )

    return merged


def build_players(pg: pd.DataFrame) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for _, grp in pg.groupby(
        ["player_id", "player"],
        dropna=False
    ):
        grp = grp.sort_values("game_date")

        latest = grp.iloc[-1]
        last10 = grp.tail(10)
        last5 = grp.tail(5)

        results.append({
            "player_id": (
                None
                if pd.isna(latest.get("player_id"))
                else str(latest.get("player_id"))
            ),
            "player": str(latest.get("player") or ""),
            "team": str(latest.get("team") or ""),
            "position": str(latest.get("position") or "UNKNOWN"),
            "position_raw": (
                None
                if pd.isna(latest.get("position_raw"))
                else str(latest.get("position_raw"))
            ),
            "games_played": int(len(grp)),
            "season": mean_record(grp),
            "last_10": mean_record(last10),
            "last_5": mean_record(last5),
        })

    results.sort(
        key=lambda x: (
            x["team"] or "",
            -(x["season"].get("min") or 0),
            x["player"]
        )
    )

    return results


def build_position_concessions(pg: pd.DataFrame) -> list[dict[str, Any]]:
    valid = pg[
        pg["opponent"].notna()
        & pg["position"].isin(["PG", "SG", "SF", "PF", "C"])
    ].copy()

    stat_cols = [
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "three_pm",
    ]

    for c in stat_cols:
        valid[c] = pd.to_numeric(
            valid[c],
            errors="coerce"
        ).fillna(0)

    per_game = (
        valid.groupby(
            ["opponent", "game_id_norm", "game_date", "position"],
            dropna=False
        )[stat_cols]
        .sum()
        .reset_index()
        .rename(columns={"opponent": "team"})
    )

    season = (
        per_game.groupby(
            ["team", "position"]
        )[stat_cols]
        .mean()
        .reset_index()
    )

    game_counts = (
        per_game.groupby(
            ["team", "position"]
        )["game_id_norm"]
        .nunique()
        .rename("games")
        .reset_index()
    )

    season = season.merge(
        game_counts,
        on=["team", "position"],
        how="left"
    )

    period_rows = []

    for (team, pos), grp in per_game.groupby(
        ["team", "position"]
    ):
        grp = grp.sort_values("game_date")

        rec = {
            "team": team,
            "position": pos
        }

        for label, n in [
            ("last_10", 10),
            ("last_5", 5)
        ]:
            tail = grp.tail(n)

            rec[label] = {
                c: round_or_none(
                    tail[c].mean(),
                    2
                ) if len(tail) else None
                for c in stat_cols
            }

        period_rows.append(rec)

    periods = {
        (r["team"], r["position"]): r
        for r in period_rows
    }

    for c in stat_cols:
        season[f"{c}_rank"] = (
            season.groupby("position")[c]
            .transform(rank_desc)
        )

    out = []

    for _, row in season.iterrows():
        key = (
            row["team"],
            row["position"]
        )

        season_stats = {
            c: round_or_none(
                row[c],
                2
            )
            for c in stat_cols
        }

        ranks = {
            c: (
                int(row[f"{c}_rank"])
                if not pd.isna(row[f"{c}_rank"])
                else None
            )
            for c in stat_cols
        }

        out.append({
            "team": row["team"],
            "position": row["position"],
            "games": int(row["games"]),
            "season": season_stats,
            "ranks_most_conceded": ranks,
            "last_10": periods[key]["last_10"],
            "last_5": periods[key]["last_5"],
        })

    out.sort(
        key=lambda x: (
            x["team"],
            ["PG", "SG", "SF", "PF", "C"].index(
                x["position"]
            )
        )
    )

    return out


def load_team_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def build_team_intelligence(
    games: pd.DataFrame,
    summary: dict[str, Any] | None
) -> list[dict[str, Any]]:

    if summary and isinstance(
        summary.get("teams"),
        dict
    ):
        out = []

        for code, t in summary["teams"].items():
            out.append({
                "team": code,
                "team_id": t.get("TEAM_ID"),
                "team_name": t.get("TEAM_NAME"),
                "games_played": t.get("GP"),
                "wins": t.get("W"),
                "losses": t.get("L"),
                "off_rating": round_or_none(
                    t.get("OFF_RATING"),
                    2
                ),
                "def_rating": round_or_none(
                    t.get("DEF_RATING"),
                    2
                ),
                "net_rating": round_or_none(
                    t.get("NET_RATING"),
                    2
                ),
                "pace": round_or_none(
                    t.get("PACE"),
                    2
                ),
                "possessions": round_or_none(
                    t.get("POSS"),
                    0
                ),
                "ast_pct": round_or_none(
                    t.get("AST_PCT"),
                    3
                ),
                "ast_to": round_or_none(
                    t.get("AST_TO"),
                    2
                ),
                "oreb_pct": round_or_none(
                    t.get("OREB_PCT"),
                    3
                ),
                "dreb_pct": round_or_none(
                    t.get("DREB_PCT"),
                    3
                ),
                "reb_pct": round_or_none(
                    t.get("REB_PCT"),
                    3
                ),
                "tov_pct": round_or_none(
                    t.get("TM_TOV_PCT"),
                    3
                ),
                "efg_pct": round_or_none(
                    t.get("EFG_PCT"),
                    3
                ),
                "ts_pct": round_or_none(
                    t.get("TS_PCT"),
                    3
                ),
            })

        return sorted(
            out,
            key=lambda x: x["team"]
        )

    g = games.copy()

    team_col = first_existing(
        g,
        GAME_TEAM_CODE_CANDIDATES
    )

    mapping = {
        "off_rating": first_existing(
            g,
            GAME_OFFRTG_CANDIDATES,
            required=False
        ),
        "def_rating": first_existing(
            g,
            GAME_DEFRTG_CANDIDATES,
            required=False
        ),
        "net_rating": first_existing(
            g,
            GAME_NETRTG_CANDIDATES,
            required=False
        ),
        "pace": first_existing(
            g,
            GAME_PACE_CANDIDATES,
            required=False
        ),
        "possessions": first_existing(
            g,
            GAME_POSS_CANDIDATES,
            required=False
        ),
    }

    wl_col = first_existing(
        g,
        GAME_WL_CANDIDATES,
        required=False
    )

    out = []

    for team, grp in g.groupby(team_col):
        rec = {
            "team": str(team),
            "games_played": int(len(grp)),
        }

        if wl_col:
            rec["wins"] = int(
                (
                    grp[wl_col]
                    .astype(str)
                    .str.upper()
                    == "W"
                ).sum()
            )

            rec["losses"] = int(
                (
                    grp[wl_col]
                    .astype(str)
                    .str.upper()
                    == "L"
                ).sum()
            )

        for out_key, col in mapping.items():
            rec[out_key] = (
                round_or_none(
                    pd.to_numeric(
                        grp[col],
                        errors="coerce"
                    ).mean(),
                    2
                )
                if col
                else None
            )

        out.append(rec)

    return sorted(
        out,
        key=lambda x: x["team"]
    )


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

    pg = prepare_player_games(
        players,
        games
    )

    payload = {
        "success": True,
        "dataset": "edge_nba_intelligence",
        "season": season,
        "season_label": f"{season-1}-{str(season)[-2:]}",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": "llimllib/nba_data",
        "players": build_players(pg),
        "teams": build_team_intelligence(
            games,
            summary
        ),
        "position_concessions": build_position_concessions(pg),
        "notes": {
            "position_concession_rank": "#1 means most conceded",
            "hybrid_positions": "Generic/hybrid source positions are excluded from PG/SG/SF/PF/C concession rankings until depth-chart mapping is added.",
            "first_basket": "Not included. Requires play-by-play data."
        }
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    outfile = output_dir / f"nba_stats_{season}.json"

    with outfile.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False
        )

    return outfile


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=".",
        help="Folder containing the NBA source files"
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

    input_dir = Path(
        args.input
    ).resolve()

    output_dir = Path(
        args.output
    ).resolve()

    for season in args.seasons:
        out = build_season(
            input_dir,
            output_dir,
            season
        )

        print(f"Built: {out}")


if __name__ == "__main__":
    main()
