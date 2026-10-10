#!/usr/bin/env python3
"""
Fetch ESPN NBA depth charts from the ESPN CORE API.

Output:
  data/nba_depthcharts.json

Uses:
  sports.core.api.espn.com/v2/sports/basketball/leagues/nba/
  seasons/{season}/teams/{team_id}/depthcharts

This endpoint exposes true PG / SG / SF / PF / C slots.
"""

from __future__ import annotations

import argparse
import json
import re
import time
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

VALID_POSITIONS = {"PG", "SG", "SF", "PF", "C"}

TEAM_LIST_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/teams?limit=100"
)

CORE_DEPTH_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/"
    "leagues/nba/seasons/{season}/teams/{team_id}/depthcharts"
    "?lang=en&region=us"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}

session = requests.Session()
session.headers.update(HEADERS)


def get_json(url: str, retries: int = 3) -> dict[str, Any]:
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(attempt)

    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def canonical_team_abbr(value: Any) -> str:
    abbr = str(value or "").upper().strip()
    return ESPN_TEAM_ALIASES.get(abbr, abbr)


def extract_team_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []

    for sport in payload.get("sports") or []:
        for league in sport.get("leagues") or []:
            for item in league.get("teams") or []:
                team = item.get("team") or item
                if isinstance(team, dict):
                    out.append(team)

    return out


def athlete_id_from_ref(ref: str | None) -> str | None:
    if not ref:
        return None

    m = re.search(r"/athletes/(\d+)", ref)
    return m.group(1) if m else None


def fetch_ref(ref: str | None) -> dict[str, Any]:
    if not ref:
        return {}

    try:
        return get_json(ref)
    except Exception:
        return {}


def normalize_position(position_obj: Any) -> str | None:
    if not isinstance(position_obj, dict):
        return None

    candidates = [
        position_obj.get("abbreviation"),
        position_obj.get("name"),
        position_obj.get("displayName"),
    ]

    aliases = {
        "POINTGUARD": "PG",
        "SHOOTINGGUARD": "SG",
        "SMALLFORWARD": "SF",
        "POWERFORWARD": "PF",
        "CENTER": "C",
    }

    for value in candidates:
        if not value:
            continue

        s = str(value).upper().replace(" ", "").strip()
        s = aliases.get(s, s)

        if s in VALID_POSITIONS:
            return s

    return None


def athlete_details(athlete_obj: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(athlete_obj, dict):
        return {}

    if athlete_obj.get("displayName") or athlete_obj.get("fullName"):
        return athlete_obj

    ref = athlete_obj.get("$ref")
    data = fetch_ref(ref)

    if data:
        return data

    return {
        "id": athlete_id_from_ref(ref),
        "$ref": ref,
    }


def parse_core_depthchart(
    team: dict[str, Any],
    season: int,
) -> list[dict[str, Any]]:

    team_id = str(team["id"])
    team_abbr = canonical_team_abbr(team["abbreviation"])

    url = CORE_DEPTH_URL.format(
        season=season,
        team_id=team_id,
    )

    payload = get_json(url)
    rows = []

    # CORE structure:
    # items[] -> positions[] -> position + athletes[]
    for chart in payload.get("items") or []:
        for pos_obj in chart.get("positions") or []:

            if not isinstance(pos_obj, dict):
                continue

            position_ref_obj = pos_obj.get("position") or {}

            if isinstance(position_ref_obj, dict) and position_ref_obj.get("$ref"):
                position_data = fetch_ref(position_ref_obj.get("$ref"))
            else:
                position_data = position_ref_obj

            position = normalize_position(position_data)

            # Sometimes the abbreviation is available directly on the position row.
            if not position:
                position = normalize_position(pos_obj)

            if not position:
                continue

            for fallback_rank, athlete_entry in enumerate(
                pos_obj.get("athletes") or [],
                start=1,
            ):
                if not isinstance(athlete_entry, dict):
                    continue

                athlete_obj = athlete_entry.get("athlete") or {}
                athlete = athlete_details(athlete_obj)

                ref = (
                    athlete_obj.get("$ref")
                    if isinstance(athlete_obj, dict)
                    else None
                )

                athlete_id = (
                    str(athlete.get("id"))
                    if athlete.get("id") is not None
                    else athlete_id_from_ref(ref)
                )

                name = (
                    athlete.get("displayName")
                    or athlete.get("fullName")
                    or athlete.get("shortName")
                    or ""
                )

                if not athlete_id and not name:
                    continue

                rank = athlete_entry.get("rank")

                try:
                    rank = int(rank)
                except Exception:
                    rank = fallback_rank

                rows.append({
                    "team": team_abbr,
                    "position": position,
                    "depth_rank": rank,
                    "starter": rank == 1,
                    "espn_athlete_id": athlete_id,
                    "player": name,
                    "short_name": athlete.get("shortName"),
                })

    # Deduplicate within position.
    deduped = {}

    for row in rows:
        key = (
            row["team"],
            row["position"],
            row["espn_athlete_id"] or row["player"],
        )

        if key not in deduped or row["depth_rank"] < deduped[key]["depth_rank"]:
            deduped[key] = row

    order = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4}

    return sorted(
        deduped.values(),
        key=lambda x: (
            x["team"],
            order.get(x["position"], 99),
            x["depth_rank"],
            x["player"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="NBA season ending year. 2026 = 2025-26.",
    )

    args = parser.parse_args()
    season = args.season

    team_payload = get_json(TEAM_LIST_URL)
    raw_teams = extract_team_entries(team_payload)

    selected_by_abbr = {}

    for team in raw_teams:
        raw_abbr = team.get("abbreviation")
        abbr = canonical_team_abbr(raw_abbr)

        if abbr in NBA_TEAMS and team.get("id") is not None:
            selected_by_abbr[abbr] = {
                "id": team.get("id"),
                "abbreviation": abbr,
                "espn_abbreviation": raw_abbr,
                "displayName": team.get("displayName"),
            }

    found = set(selected_by_abbr)
    missing = sorted(NBA_TEAMS - found)

    if missing:
        raise RuntimeError(
            f"ESPN team list missing expected NBA teams: {missing}"
        )

    all_rows = []
    team_status = []

    for idx, abbr in enumerate(sorted(NBA_TEAMS), start=1):
        team = selected_by_abbr[abbr]

        try:
            rows = parse_core_depthchart(team, season)
            positions = sorted({x["position"] for x in rows})

            all_rows.extend(rows)

            team_status.append({
                "team": abbr,
                "rows": len(rows),
                "positions": positions,
                "success": len(rows) > 0,
            })

            print(
                f"[{idx:02d}/30] {abbr}: "
                f"{len(rows)} rows | {positions}"
            )

        except Exception as e:
            team_status.append({
                "team": abbr,
                "rows": 0,
                "positions": [],
                "success": False,
                "error": str(e),
            })

            print(f"[{idx:02d}/30] {abbr}: ERROR {e}")

        time.sleep(0.10)

    successful = [x for x in team_status if x["success"]]
    all_five = [
        x for x in successful
        if set(x["positions"]) == VALID_POSITIONS
    ]

    payload = {
        "success": len(successful) == 30 and len(all_rows) > 0,
        "dataset": "edge_nba_depthcharts",
        "season": season,
        "season_label": f"{season-1}-{str(season)[-2:]}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "ESPN CORE API",
        "positions": ["PG", "SG", "SF", "PF", "C"],
        "players": all_rows,
        "validation": {
            "expected_teams": 30,
            "successful_teams": len(successful),
            "player_rows": len(all_rows),
            "teams_with_all_five_positions": len(all_five),
            "failed_teams": [
                x for x in team_status
                if not x["success"]
            ],
            "team_status": team_status,
        },
    }

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "nba_depthcharts.json"

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    print(f"Built: {out_file}")
    print(
        f"Validation: {len(successful)}/30 teams | "
        f"{len(all_five)}/30 with all five positions | "
        f"{len(all_rows)} player rows"
    )

    # Important: do NOT allow a false-positive green Action.
    if len(successful) != 30 or len(all_rows) == 0:
        raise RuntimeError(
            "ESPN depth-chart fetch produced incomplete/empty data."
        )


if __name__ == "__main__":
    main()
