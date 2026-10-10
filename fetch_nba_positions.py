#!/usr/bin/env python3
"""
Fetch ESPN Fantasy Basketball primary positions for the NBA player universe.

Output:
  data/nba_positions_2026.json

For ESPN Fantasy Basketball:
  defaultPositionId 1 = PG
  defaultPositionId 2 = SG
  defaultPositionId 3 = SF
  defaultPositionId 4 = PF
  defaultPositionId 5 = C

We use the primary/default position only for EDGE positional-concession mapping.
That gives every mapped player one exact PG/SG/SF/PF/C bucket and avoids
double-counting multi-position fantasy eligibility.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


NBA_TEAMS = {
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
    "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
    "OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS"
}

ESPN_TEAM_ALIASES = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}

POSITION_BY_DEFAULT_ID = {
    1: "PG",
    2: "SG",
    3: "SF",
    4: "PF",
    5: "C",
}

TEAM_LIST_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/teams?limit=100"
)

FANTASY_PLAYERS_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
    "seasons/{season}/players?scoringPeriodId=0&view=kona_player_info"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


def canonical_team_abbr(value: Any) -> str:
    abbr = str(value or "").upper().strip()
    return ESPN_TEAM_ALIASES.get(abbr, abbr)


def get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> Any:
    merged = dict(HEADERS)
    if headers:
        merged.update(headers)

    r = requests.get(url, headers=merged, timeout=timeout)
    r.raise_for_status()
    return r.json()


def build_team_id_map() -> dict[int, str]:
    payload = get_json(TEAM_LIST_URL)
    mapping: dict[int, str] = {}

    for sport in payload.get("sports") or []:
        for league in sport.get("leagues") or []:
            for item in league.get("teams") or []:
                team = item.get("team") or item
                if not isinstance(team, dict):
                    continue

                team_id = team.get("id")
                abbr = canonical_team_abbr(team.get("abbreviation"))

                if team_id is None or abbr not in NBA_TEAMS:
                    continue

                mapping[int(team_id)] = abbr

    return mapping


def extract_player(item: Any) -> dict[str, Any]:
    # /players normally returns player objects directly, but some ESPN
    # fantasy endpoints wrap them as {"player": {...}}.
    if isinstance(item, dict) and isinstance(item.get("player"), dict):
        return item["player"]

    return item if isinstance(item, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="ESPN fantasy season year. 2026 = 2025-26.",
    )

    parser.add_argument(
        "--output",
        default="data",
        help="Output directory.",
    )

    args = parser.parse_args()

    team_id_map = build_team_id_map()

    if len(set(team_id_map.values())) != 30:
        raise RuntimeError(
            f"Expected 30 NBA teams from ESPN team map, "
            f"got {len(set(team_id_map.values()))}"
        )

    fantasy_filter = json.dumps({
        "players": {
            "limit": 5000,
            "sortPercOwned": {
                "sortPriority": 1,
                "sortAsc": False,
            },
        }
    })

    url = FANTASY_PLAYERS_URL.format(season=args.season)

    payload = get_json(
        url,
        headers={"X-Fantasy-Filter": fantasy_filter},
        timeout=120,
    )

    # ESPN may return either a raw list or an object containing players.
    if isinstance(payload, list):
        raw_players = payload
    elif isinstance(payload, dict):
        raw_players = payload.get("players") or []
    else:
        raw_players = []

    rows: list[dict[str, Any]] = []

    for item in raw_players:
        player = extract_player(item)
        if not player:
            continue

        try:
            team_id = int(player.get("proTeamId"))
        except Exception:
            continue

        team = team_id_map.get(team_id)
        if team not in NBA_TEAMS:
            continue

        try:
            pos_id = int(player.get("defaultPositionId"))
        except Exception:
            continue

        position = POSITION_BY_DEFAULT_ID.get(pos_id)
        if not position:
            continue

        player_id = player.get("id")
        full_name = (
            player.get("fullName")
            or " ".join(
                x for x in [
                    str(player.get("firstName") or "").strip(),
                    str(player.get("lastName") or "").strip(),
                ]
                if x
            ).strip()
        )

        if not full_name:
            continue

        rows.append({
            "espn_player_id": str(player_id) if player_id is not None else None,
            "player": full_name,
            "team": team,
            "position": position,
            "default_position_id": pos_id,
            "active": bool(player.get("active", True)),
            "injured": bool(player.get("injured", False)),
            "injury_status": player.get("injuryStatus"),
        })

    # Defensive dedupe by ESPN player id; fall back to team + name.
    deduped: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        key = (
            row["espn_player_id"] or "",
            f'{row["team"]}|{row["player"]}',
        )
        deduped[key] = row

    rows = sorted(
        deduped.values(),
        key=lambda x: (x["team"], x["position"], x["player"]),
    )

    team_counts = {
        team: sum(1 for r in rows if r["team"] == team)
        for team in sorted(NBA_TEAMS)
    }

    teams_with_players = [
        team for team, count in team_counts.items()
        if count > 0
    ]

    payload_out = {
        "success": len(teams_with_players) == 30 and len(rows) > 300,
        "dataset": "edge_nba_positions",
        "season": args.season,
        "season_label": f"{args.season-1}-{str(args.season)[-2:]}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "ESPN Fantasy Basketball",
        "position_model": "primary defaultPositionId only",
        "positions": ["PG", "SG", "SF", "PF", "C"],
        "players": rows,
        "validation": {
            "expected_teams": 30,
            "teams_with_players": len(teams_with_players),
            "player_rows": len(rows),
            "team_counts": team_counts,
            "missing_teams": sorted(NBA_TEAMS - set(teams_with_players)),
        },
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"nba_positions_{args.season}.json"

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(
            payload_out,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    print(f"Built: {out_file}")
    print(
        f"Validation: {len(teams_with_players)}/30 teams | "
        f"{len(rows)} player rows"
    )

    for team in sorted(team_counts):
        print(f"{team}: {team_counts[team]} players")

    if not payload_out["success"]:
        raise RuntimeError(
            "ESPN fantasy position fetch returned incomplete data."
        )


if __name__ == "__main__":
    main()
