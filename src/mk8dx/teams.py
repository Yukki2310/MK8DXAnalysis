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


# -- 名前の先頭部分（タグ）の画像照合によるチーム提案 ---------------------------
# 同じチームの選手は同じタグ（名前の先頭の同じ文字列）で始まる。文字認識の
# 読み違い（'Zw³' を '2W3' と読む等）に左右されないよう、名前の文字画像の
# 左端をそろえて、どこまで同じ形が続くか（先頭の一致幅）を測る。
# 大文字小文字・ひらがな/カタカナの違いは画像では一致しないので、文字認識の
# 読みから求めた先頭一致も併用する。
MAX_INK_FRACTION = 0.35  # 文字領域の塗りがこれを超える抽出は失敗とみなす
PREFIX_MAX_PX = 170       # 先頭一致を調べる最大幅
MIN_PREFIX_PX = 14        # これ未満の一致は偶然とみなす（1文字ぶん未満）
PREFIX_WINDOW_PX = 8      # 先頭一致を調べる窓の幅
PREFIX_WINDOW_IOU = 0.75  # 窓内の重なり率がこれ未満なら「形が違う」
TEXT_PREFIX_PX_PER_CHAR = 18  # 文字認識の先頭一致1文字を画像の幅に換算する値


def _leading_strip(image):
    """名前行クロップの文字マスクを、最初の文字の左端から切り出す。
    (そのまま, 少し太らせたもの) を返す。太らせた方は位置ずれの許容に使う。"""
    import cv2
    import numpy as np

    from .identity import extract_name_mask

    mask = extract_name_mask(image)
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0 or mask.mean() > MAX_INK_FRACTION:
        return None  # 文字が無い、または抽出に失敗している（べったり塗られた）行
    strip = mask[:, cols[0]: cols[0] + PREFIX_MAX_PX]
    return strip, cv2.dilate(strip, np.ones((3, 3), np.uint8))


def image_prefix_px(a, b) -> int:
    """2つの先頭マスクで、同じ形が続く幅（px）。MIN_PREFIX_PX 未満は 0。

    左端から幅 PREFIX_WINDOW_PX ずつの窓で比べ、最初に形が食い違った窓の
    手前までを「先頭一致」とする。全体の重なり率で見ると、途中で食い違っても
    後ろの同じ部分（' one' など）に引きずられて一致に見えてしまうため、
    窓ごとに判定する。
    """
    if a is None or b is None:
        return 0
    (ma, da), (mb, db) = a, b
    n = min(ma.shape[1], mb.shape[1])
    matched = 0
    for x in range(0, n - PREFIX_WINDOW_PX + 1, PREFIX_WINDOW_PX):
        sl = slice(x, x + PREFIX_WINDOW_PX)
        union = int((ma[:, sl] | mb[:, sl]).sum())
        if union < 12:           # どちらもほぼ空白（文字間）
            continue
        # 一方の文字が、もう一方を少し太らせたものにどれだけ収まるか（±1pxのずれを許容）
        inter = (int((ma[:, sl] & db[:, sl]).sum()) + int((mb[:, sl] & da[:, sl]).sum())) / 2
        if inter / union < PREFIX_WINDOW_IOU:
            break                # 形が食い違った（空白と文字の食い違いも含む）
        matched = x + PREFIX_WINDOW_PX
    return matched if matched >= MIN_PREFIX_PX else 0


def pair_scores(images: list, names: list[str]) -> list[list[int]]:
    """行どうしの「先頭が同じ」度合い（大きいほど同じタグらしい）。"""
    strips = [_leading_strip(im) if im is not None else None for im in images]
    norm = [normalize_name(n) for n in names]
    n = len(images)
    scores = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            text_len = 0
            while (text_len < min(len(norm[i]), len(norm[j]))
                   and norm[i][text_len] == norm[j][text_len]):
                text_len += 1
            s = max(image_prefix_px(strips[i], strips[j]),
                    text_len * TEXT_PREFIX_PX_PER_CHAR if text_len else 0)
            scores[i][j] = scores[j][i] = s
    return scores


def _best_partition(scores: list[list[int]], size: int) -> list[list[int]] | None:
    """全員が同じタグで結ばれたチーム分けのうち、一致が最も強いものを探す。"""
    n = len(scores)
    best: tuple[int, list[list[int]]] | None = None

    def group_value(members: list[int]) -> int:
        pairs = [scores[a][b] for k, a in enumerate(members) for b in members[k + 1:]]
        return min(pairs) if pairs else 0

    def search(free: list[int], groups: list[list[int]], total: int) -> None:
        nonlocal best
        if not free:
            if best is None or total > best[0]:
                best = (total, [list(g) for g in groups])
            return
        first, rest = free[0], free[1:]
        for combo in _combinations(rest, size - 1):
            members = [first, *combo]
            value = group_value(members)
            if value <= 0:
                continue
            remaining = [i for i in rest if i not in combo]
            groups.append(members)
            search(remaining, groups, total + value)
            groups.pop()

    search(list(range(n)), [], 0)
    return best[1] if best else None


def _combinations(items: list[int], k: int):
    from itertools import combinations

    return combinations(items, k)


def suggest_by_prefix(images: list, names: list[str], fmt: str) -> Assignment | None:
    """名前の先頭（タグ）の画像・文字から、チーム分けを提案する。決められなければ None。"""
    if len(images) != PLAYERS_PER_RACE or any(im is None for im in images):
        return None
    size = FORMATS[fmt]
    groups = _best_partition(pair_scores(images, names), size)
    if groups is None:
        return None
    from .tags import common_prefix

    used: set[str] = set()
    teams: list[Team] = []
    for k, members in enumerate(sorted(groups, key=lambda g: g[0]), start=1):
        raw = [names[i] for i in members]
        label = common_prefix(raw, normalized=False).strip().rstrip("#-_.:・ ")
        if not label:  # 文字認識では共通部分が取れなかった（読み違い）: 最短の名前を仮の呼び名にする
            shortest = min((r for r in raw if r), key=len, default="")
            label = shortest[:3].strip() or f"チーム{k}"
        key = normalize_name(label)
        while key in used:
            label += "'"
            key = normalize_name(label)
        used.add(key)
        teams.append(Team(tag=label, normalized_tag=key, members=sorted(members)))
    return Assignment(teams=teams, valid=True, warnings=[])


def suggest_by_vlm(reader, images: list, fmt: str) -> Assignment | None:
    """視覚言語モデルに名前画像12枚を見せてチーム分けとタグを推定させる（最後の手段）。

    モデルの答えは検証してから使う: 12人が重複なく、決まった人数ずつの組に分かれていること。
    """
    if reader is None or not getattr(reader, "can_suggest_teams", False):
        return None
    if len(images) != PLAYERS_PER_RACE or any(im is None for im in images):
        return None
    size = FORMATS[fmt]
    groups = reader.suggest_teams(images, size)
    if not groups:
        return None
    seen: list[int] = []
    for g in groups:
        if len(g["members"]) != size:
            return None
        seen += g["members"]
    if sorted(seen) != list(range(PLAYERS_PER_RACE)):
        return None
    used: set[str] = set()
    teams: list[Team] = []
    for k, g in enumerate(sorted(groups, key=lambda g: min(g["members"])), start=1):
        label = (g.get("tag") or "").strip().rstrip("#-_.:・ ") or f"チーム{k}"
        while normalize_name(label) in used:
            label += "'"
        used.add(normalize_name(label))
        teams.append(Team(tag=label, normalized_tag=normalize_name(label),
                          members=sorted(g["members"])))
    return Assignment(teams=teams, valid=True, warnings=[])
