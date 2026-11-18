"""チームクラスタリング（設計書 4.2）。

対戦形式の人数制約（2v2=各2名/6チーム、3v3=各3名/4チーム、4v4=各4名/3チーム）を
使って、12名のプレイヤー名から最も確からしいチーム分けを推定する。

方針:
- 正規化した名前の共通接頭辞でグループ化する。まず接頭辞1文字で分け、
  人数制約より大きいグループは接頭辞を伸ばして再分割する（再帰）。
- どのグループも人数制約に合わない場合は不整合として警告し、
  先頭1文字でのグループ化をフォールバックとして返す（管理画面で手動修正する前提）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tags import common_prefix, has_forbidden_diacritic, normalize_name

FORMATS: dict[str, int] = {"2v2": 2, "3v3": 3, "4v4": 4}
PLAYERS_PER_RACE = 12


@dataclass
class Team:
    tag: str               # 表示用タグ（元表記の共通接頭辞）
    normalized_tag: str    # 正規化済みタグ
    members: list[int]     # プレイヤーのインデックス（入力順=順位順の0始まり）


@dataclass
class Assignment:
    teams: list[Team]
    valid: bool
    warnings: list[str] = field(default_factory=list)


def _partition(
    indices: list[int], norm_names: list[str], prefix_len: int, team_size: int,
    max_len: int,
) -> list[tuple[str, list[int]]] | None:
    """indices を接頭辞 prefix_len 文字でグループ化し、全グループが
    team_size にそろう分割を返す。そろわなければ None。"""
    if prefix_len > max_len:
        return None
    groups: dict[str, list[int]] = {}
    for i in indices:
        key = norm_names[i][:prefix_len]
        groups.setdefault(key, []).append(i)
    result: list[tuple[str, list[int]]] = []
    for key, members in groups.items():
        if len(members) == team_size:
            result.append((key, members))
        elif len(members) > team_size and len(members) % team_size == 0:
            sub = _partition(members, norm_names, prefix_len + 1, team_size, max_len)
            if sub is None:
                return None
            result.extend(sub)
        else:
            return None
    return result


def assign_teams(names: list[str], fmt: str) -> Assignment:
    """12名の名前（順位順）からチーム割り当てを推定する。"""
    if fmt not in FORMATS:
        raise ValueError(f"未対応の対戦形式です: {fmt}")
    team_size = FORMATS[fmt]
    n_teams = PLAYERS_PER_RACE // team_size
    warnings: list[str] = []

    if len(names) != PLAYERS_PER_RACE:
        warnings.append(f"プレイヤー数が12ではありません: {len(names)}")

    for name in names:
        if has_forbidden_diacritic(name):
            warnings.append(f"分音記号付き文字の使用を検知しました: {name}")

    norm = [normalize_name(n) for n in names]
    max_len = max((len(n) for n in norm), default=0)
    groups = _partition(list(range(len(names))), norm, 1, team_size, max_len)

    if groups is not None and len(groups) == n_teams:
        teams = []
        for key, members in groups:
            raw_members = [names[i] for i in members]
            teams.append(Team(
                tag=common_prefix(raw_members, normalized=False) or key,
                normalized_tag=key,
                members=sorted(members),
            ))
        teams.sort(key=lambda t: t.members[0])
        return Assignment(teams=teams, valid=True, warnings=warnings)

    # フォールバック: 先頭1文字でグループ化し、不整合を警告する
    warnings.append("人数制約に合うチーム分けを自動推定できませんでした")
    fallback: dict[str, list[int]] = {}
    for i, n in enumerate(norm):
        fallback.setdefault(n[:1], []).append(i)
    teams = []
    for key, members in sorted(fallback.items(), key=lambda kv: kv[1][0]):
        if len(members) != team_size:
            warnings.append(
                f"タグ '{key}' の人数が{team_size}ではありません: {len(members)}名"
            )
        raw_members = [names[i] for i in members]
        teams.append(Team(
            tag=common_prefix(raw_members, normalized=False) or key,
            normalized_tag=key,
            members=sorted(members),
        ))
    return Assignment(teams=teams, valid=False, warnings=warnings)
