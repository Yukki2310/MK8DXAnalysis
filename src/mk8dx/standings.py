"""順位表の計算。

確定済みレースの各行は選手名簿のエントリIDを持ち、チームは「今の」
選手名簿から引く。管理画面でチームを付け替えると、過去のレースにも
即座に反映される（保存し直しは不要）。
"""
from __future__ import annotations

from .roster import Roster
from .tags import normalize_name


def compute(rows: list[dict], roster: Roster) -> list[dict]:
    """db.valid_rows の結果からチーム別の順位表を作る。

    戻り値の各要素: tag, color, total, per_race, last_race_points,
    members(選手名), rank(同点は同順位), diff_from_top。
    チーム未設定の選手と、除外した選手の点は集計に含めない。
    """
    teams: dict[str, dict] = {}
    last_race = max((r["race_number"] for r in rows), default=0)
    for r in rows:
        entry = roster._entry_by_id(r["entry_id"]) if r.get("entry_id") else None
        if entry is None or not entry.team or entry.excluded:
            continue
        key = normalize_name(entry.team)
        team = teams.setdefault(key, {
            "tag": entry.team,
            "color": roster.color_for(entry.team),
            "total": 0,
            "per_race": {},
            "last_race_points": 0,
            "members": [],
        })
        team["total"] += r["score"]
        team["per_race"][r["race_number"]] = (
            team["per_race"].get(r["race_number"], 0) + r["score"])
        if entry.canonical_name not in team["members"]:
            team["members"].append(entry.canonical_name)
    for team in teams.values():
        team["last_race_points"] = team["per_race"].get(last_race, 0)
    result = sorted(teams.values(), key=lambda t: (-t["total"], t["tag"]))
    top = result[0]["total"] if result else 0
    prev_total, prev_rank = None, 0
    for i, team in enumerate(result):
        team["rank"] = prev_rank if team["total"] == prev_total else i + 1
        prev_total, prev_rank = team["total"], team["rank"]
        team["diff_from_top"] = team["total"] - top
    return result
