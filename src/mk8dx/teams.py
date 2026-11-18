"""チーム分け（選手に設定されたチーム名から12人を組にする）。

利用者が管理画面で選手ごとにチーム名を付ける運用を主とし、
名前の先頭文字（タグ）から推定する clustering.assign_teams は、
最初のレースでチーム名を提案するために使う。
"""
from __future__ import annotations

from .clustering import FORMATS, PLAYERS_PER_RACE, Assignment, Team
from .tags import normalize_name


def group_by_labels(labels: list[str | None], fmt: str) -> Assignment:
    """行ごとのチーム名（順位順12個）から割り当てを作り、人数を検証する。"""
    team_size = FORMATS[fmt]
    n_teams = PLAYERS_PER_RACE // team_size
    warnings: list[str] = []
    groups: dict[str, list[int]] = {}
    display: dict[str, str] = {}
    for i, label in enumerate(labels):
        if not label:
            continue
        key = normalize_name(label)
        groups.setdefault(key, []).append(i)
        display.setdefault(key, label)
    missing = sum(1 for label in labels if not label)
    if missing:
        warnings.append(f"チームが未設定の選手が{missing}名います")
    for key, members in groups.items():
        if len(members) != team_size:
            warnings.append(
                f"チーム「{display[key]}」が{len(members)}名です"
                f"（{team_size}名ずつにしてください）"
            )
    if len(groups) != n_teams:
        warnings.append(f"チーム数が{len(groups)}です（{n_teams}チームにしてください）")
    valid = not warnings
    teams = [
        Team(tag=display[key], normalized_tag=key, members=sorted(members))
        for key, members in groups.items()
    ]
    teams.sort(key=lambda t: t.members[0])
    return Assignment(teams=teams, valid=valid, warnings=warnings)
