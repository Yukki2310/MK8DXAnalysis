"""リザルトフレーム解析パイプライン（設計書 3.3 / 4 / 5.1）。

2フェーズの扱い（実画面で確認済み）:
- フェーズA「ゴール順画面」: 行はゴール順。集計はこちらだけを使う。
  スコア列に '+15' のような加算表示が出る時間帯があり、その後は累計だけが
  残る形になる。
- フェーズB「累計画面」: 行は累計順位順。集計には使わず、自前計算の
  累計との照合（クロスチェック）専用。

フェーズ1〜2の実用化で OCR を数字判定から外した（docs/phase1-notes.md）:
- 同一人物の判定は文字マスク（identity.py）、リザルト画面の検証は順位列の
  見本（rankcheck.py）で行う。
- フェーズの判別は「スコア列に加算表示（明るい筆画）があるか」と
  「行の並びが自前累計の降順か」で行う（runner.py）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .identity import extract_name_mask, trim_mask
from .preprocess import crop_rows
from .roster import Observation, extract_flag_signature
from .tags import normalize_name

PHASE_RACE_RESULT = "race_result"
PHASE_STANDINGS = "standings"
PHASE_UNKNOWN = "unknown"

# 累計の降順判定に必要な最低行数（自前累計が分かる行の数）
MIN_KNOWN_ROWS = 9


@dataclass
class StandingsReading:
    entries: list[dict]
    problems: list[str] = field(default_factory=list)


def looks_like_standings(totals_in_row_order: list[int | None]) -> bool:
    """行順に並べた「自前累計」が降順なら累計画面（フェーズB）とみなす。

    totals_in_row_order は各行の選手の自前累計（不明は None）。
    ゴール順画面の行順が偶然累計の降順になる確率は 12人では極めて低いが、
    1レース目はゴール順＝累計順なので区別できない（照合の結果は同じなので
    実害はない）。
    """
    known = [t for t in totals_in_row_order if t is not None]
    if len(known) < MIN_KNOWN_ROWS:
        return False
    pairs = list(zip(known, known[1:]))
    # 同名同旗の複数人は1エントリで合算されるため、完全な降順までは要求しない
    return sum(1 for a, b in pairs if a >= b) / len(pairs) >= 0.85


def extract_observations(
    frame: np.ndarray, roi: dict, ocr: object = None
) -> list[Observation]:
    """フレームから名簿照合用の観測（文字マスク・国旗署名・名前クロップ）を抽出する。

    名前の文字認識はここでは行わない（name は空）。表示名は新規選手の
    確定時に namer.NameReader が読む。ocr 引数は互換のため残している。
    """
    rows = crop_rows(frame, roi)
    observations: list[Observation] = []
    for rank in range(1, 13):
        regions = rows.get(rank)
        mask = None
        image = None
        if regions and "name" in regions:
            image = regions["name"]
            mask = trim_mask(extract_name_mask(image))
        flag = extract_flag_signature(frame, roi, rank)
        observations.append(
            Observation(name="", flag=flag, mask=mask, image=image))
    return observations


def analyze_standings_frame(
    frame: np.ndarray, roi: dict, ocr: object = None
) -> StandingsReading:
    """累計画面（フェーズB）を読み取る。照合専用で、集計には使わない。

    名前は読まず、各行の文字マスクと国旗を返す（名簿で名寄せする）。
    累計点の数値は使わず、並び順だけで照合する。
    """
    observations = extract_observations(frame, roi)
    entries = [
        {"position": i + 1, "name": "", "total": None,
         "mask": o.mask, "flag": o.flag}
        for i, o in enumerate(observations)
    ]
    problems = [
        f"{e['position']}行目の文字を抽出できませんでした"
        for e in entries if e["mask"] is None
    ]
    return StandingsReading(entries=entries, problems=problems)


def cross_check_standings(
    computed_totals: dict[str, int], standings: StandingsReading
) -> list[str]:
    """自前の累計（名前->累計点）とフェーズBの表示を照合し、不一致を返す。"""
    problems: list[str] = []
    computed = {normalize_name(k): v for k, v in computed_totals.items()}
    shown_order: list[str] = []
    for e in standings.entries:
        if not e["name"]:
            continue  # 名簿と照合できなかった行（並べ替え途中など）は判定に使わない
        key = normalize_name(e["name"])
        shown_order.append(key)
        if key not in computed:
            problems.append(f"累計画面の '{e['name']}' が集計側に存在しません")
        elif e["total"] is not None and computed[key] != e["total"]:
            problems.append(
                f"'{e['name']}' の累計が一致しません: 集計={computed[key]} / 画面={e['total']}"
            )
    expected_order = [
        k for k, _ in sorted(computed.items(), key=lambda kv: -kv[1])
        if k in shown_order
    ]
    if shown_order != expected_order:
        problems.append("累計順位の並びが集計結果と一致しません")
    return problems
