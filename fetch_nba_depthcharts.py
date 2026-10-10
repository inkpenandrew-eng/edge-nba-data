#!/usr/bin/env python3
"""
Fetch current NBA depth charts from ESPN and save a clean JSON file.

Output:
  data/nba_depthcharts.json

The script:
  - fetches the 30 NBA teams from ESPN
  - fetches each team's current depth chart
  - preserves ESPN's exact PG / SG / SF / PF / C slots
  - records depth rank within each position
  - resolves athlete names/ids when ESPN returns reference objects
  - validates that all 30 NBA teams are present
"""

from __future__ import annotations

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

VALID_POSITIONS = {"PG", "SG", "SF", "PF", "C"}

TEAM_LIST_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/teams?limit=100"
)

DEPTH_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/teams/{team_id}/depthcharts"
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
                time.sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def extract_team_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sports = payload.get("sports") or []
    out = []

    for sport in sports:
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


def dereference_athlete(obj: Any) -> dict[str, Any]:
    """
    ESPN depth-chart athlete values may be:
      - a fully embedded athlete object
      - {"$ref": "..."}
      - {"athlete": {"$ref": "..."}}
    """
    if not isinstance(obj, dict):
        return {}

    if obj.get("displayName") or obj.get("fullName"):
        return obj

    ref = obj.get("$ref")
    if ref:
        try:
            return get_json(ref)
        except Exception:
            return {"id": athlete_id_from_ref(ref), "$ref": ref}

    athlete = obj.get("athlete")
    if isinstance(athlete, dict):
        if athlete.get("displayName") or athlete.get("fullName"):
            return athlete

        ref = athlete.get("$ref")
        if ref:
            try:
                return get_json(ref)
            except Exception:
                return {"id": athlete_id_from_ref(ref), "$ref": ref}

    return {}


def normalize_position(pos_key: str, pos_obj: dict[str, Any]) -> str | None:
    candidates = [
        pos_key,
        pos_obj.get("abbreviation"),
        pos_obj.get("name"),
        (pos_obj.get("position") or {}).get("abbreviation")
        if isinstance(pos_obj.get("position"), dict)
        else None,
    ]

    for v in candidates:
        if not v:
            continue

        s = str(v).upper().strip().replace(" ", "")
        aliases = {
            "POINTGUARD": "PG",
            "SHOOTINGGUARD": "SG",
            "SMALLFORWARD": "SF",
            "POWERFORWARD": "PF",
            "CENTER": "C",
        }

        s = aliases.get(s, s)

        if s in VALID_POSITIONS:
            return s

    return None


def parse_athlete_entry(
    team_abbr: str,
    position: str,
    item: dict[str, Any],
    fallback_rank: int,
) -> dict[str, Any] | None:

    athlete_obj = item.get("athlete") if isinstance(item, dict) else None

    if isinstance(athlete_obj, dict):
        athlete = dereference_athlete(athlete_obj)
    else:
        athlete = dereference_athlete(item)

    ref = None

    if isinstance(athlete_obj, dict):
        ref = athlete_obj.get("$ref")

    if not ref and isinstance(item, dict):
        ref = item.get("$ref")

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
        return None

    rank = item.get("rank") if isinstance(item, dict) else None

    try:
        rank = int(rank)
    except Exception:
        rank = fallback_rank

    status_obj = athlete.get("status") or {}
    status = None

    if isinstance(status_obj, dict):
        status = (
            status_obj.get("type", {}).get("name")
            if isinstance(status_obj.get("type"), dict)
            else status_obj.get("name")
        )

    return {
        "team": team_abbr,
        "position": position,
        "depth_rank": rank,
        "starter": rank == 1,
        "espn_athlete_id": athlete_id,
        "player": name,
        "short_name": athlete.get("shortName"),
        "status": status,
    }


def parse_depth_chart(team: dict[str, Any]) -> list[dict[str, Any]]:
    team_id = str(team["id"])
    abbr = str(team["abbreviation"]).upper()

    payload = get_json(DEPTH_URL.format(team_id=team_id))
    rows = []

    # ESPN commonly returns items -> positions.
    items = payload.get("items") or []

    # Some site responses may expose positions directly.
    if not items and payload.get("positions"):
        items = [payload]

    for chart in items:
        positions = chart.get("positions") or {}

        if isinstance(positions, list):
            iterator = [(str(i), p) for i, p in enumerate(positions)]
        elif isinstance(positions, dict):
            iterator = positions.items()
        else:
            continue

        for pos_key, pos_obj in iterator:
            if not isinstance(pos_obj, dict):
                continue

            position = normalize_position(str(pos_key), pos_obj)
            if not position:
                continue

            athletes = pos_obj.get("athletes") or []

            for idx, item in enumerate(athletes, start=1):
                if not isinstance(item, dict):
                    continue

                row = parse_athlete_entry(
                    team_abbr=abbr,
                    position=position,
                    item=item,
                    fallback_rank=idx,
                )

                if row:
                    rows.append(row)

    # Defensive dedupe.
    unique = {}
    for row in rows:
        key = (
            row["team"],
            row["position"],
            row["espn_athlete_id"] or row["player"],
        )

        if key not in unique or row["depth_rank"] < unique[key]["depth_rank"]:
            unique[key] = row

    return sorted(
        unique.values(),
        key=lambda x: (
            x["team"],
            ["PG", "SG", "SF", "PF", "C"].index(x["position"]),
            x["depth_rank"],
            x["player"],
        ),
    )


def main() -> None:
    team_payload = get_json(TEAM_LIST_URL)
    teams = extract_team_entries(team_payload)

    selected = []

    for team in teams:
        abbr = str(team.get("abbreviation") or "").upper()
        if abbr in NBA_TEAMS:
            selected.append({
                "id": team.get("id"),
                "abbreviation": abbr,
                "displayName": team.get("displayName"),
            })

    found = {x["abbreviation"] for x in selected}
    missing = sorted(NBA_TEAMS - found)

    if missing:
        raise RuntimeError(
            f"ESPN team list missing expected NBA teams: {missing}"
        )

    all_rows = []
    team_status = []

    for idx, team in enumerate(
        sorted(selected, key=lambda x: x["abbreviation"]),
        start=1
    ):
        abbr = team["abbreviation"]

        try:
            rows = parse_depth_chart(team)
            all_rows.extend(rows)

            positions = sorted({r["position"] for r in rows})

            team_status.append({
                "team": abbr,
                "rows": len(rows),
                "positions": positions,
                "success": True,
            })

            print(
                f"[{idx:02d}/30] {abbr}: "
                f"{len(rows)} depth rows | {positions}"
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

        time.sleep(0.15)

    successful_teams = {
        x["team"]
        for x in team_status
        if x["success"]
    }

    teams_with_all_five = {
        x["team"]
        for x in team_status
        if set(x["positions"]) == VALID_POSITIONS
    }

    payload = {
        "success": len(successful_teams) == 30,
        "dataset": "edge_nba_depthcharts",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "ESPN",
        "positions": ["PG", "SG", "SF", "PF", "C"],
        "players": all_rows,
        "validation": {
            "expected_teams": 30,
            "successful_teams": len(successful_teams),
            "player_rows": len(all_rows),
            "teams_with_all_five_positions": len(teams_with_all_five),
            "failed_teams": [
                x for x in team_status if not x["success"]
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
        "Validation: "
        f"{len(successful_teams)}/30 teams | "
        f"{len(teams_with_all_five)}/30 with all five positions | "
        f"{len(all_rows)} player rows"
    )

    if len(successful_teams) != 30:
        raise RuntimeError(
            "Depth-chart fetch did not succeed for all 30 NBA teams."
        )


if __name__ == "__main__":
    main()
