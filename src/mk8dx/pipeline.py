"""リザルトフレーム解析パイプライン（設計書 3.3 / 3.4 / 4 / 5.1）。

2フェーズの扱い（実画面で確認済み）:
- フェーズA「ゴール順画面」: 行はゴール順。スコア列は '+15' のように獲得点が
  '+' 付きで表示される。集計はこちらだけを使う。右端の累計列は加算
  アニメーション中は古い値のことがあり、確定値として使わない。
- フェーズB「累計順位画面」: 行は累計順位順。total列は累計点（'+' なし）。
  集計には使わず、自前計算の累計との照合（クロスチェック）専用。
判別はスコア列の '+' の有無で行う。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import Assignment, assign_teams
from .ocr import OcrEngine, parse_points_text, parse_total_text
from .preprocess import crop_rows, edge_density
from .scoring import points_for_rank, validate_scores
from .tags import normalize_name

# フェーズ判別: スコア列の平均エッジ密度の閾値。
# 実キャプチャ2枚での実測は ゴール順画面=約0.070 / 累計画面=約0.031 だった。
SCORE_INK_THRESHOLD = 0.045

PHASE_RACE_RESULT = "race_result"
PHASE_STANDINGS = "standings"
PHASE_UNKNOWN = "unknown"


@dataclass
class RaceReading:
    entries: list[dict]
    validation_ok: bool
    problems: list[str]
    assignment: Assignment | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class StandingsReading:
    entries: list[dict]
    problems: list[str] = field(default_factory=list)


def classify_phase(frame: np.ndarray, roi: dict, ocr: OcrEngine) -> str:
    """フェーズを判別する。

    1. 上位3行のスコア列OCRで '+' が読めればゴール順画面で確定。
    2. ダメな場合はスコア列の平均エッジ密度で判定する。ゴール順画面は
       全行に数字があるため密度が高く、累計画面はスコア列が空で低い。
    """
    rows = crop_rows(frame, roi)
    for rank in (1, 2, 3):
        regions = rows.get(rank)
        if regions and "score" in regions:
            text, _ = ocr.recognize(regions["score"])
            _, has_plus = parse_points_text(text)
            if has_plus:
                return PHASE_RACE_RESULT
    densities = [
        edge_density(regions["score"])
        for regions in rows.values()
        if "score" in regions
    ]
    if not densities:
        return PHASE_UNKNOWN
    mean_ink = sum(densities) / len(densities)
    return PHASE_RACE_RESULT if mean_ink >= SCORE_INK_THRESHOLD else PHASE_STANDINGS


def analyze_result_frame(
    frame: np.ndarray, roi: dict, ocr: OcrEngine, fmt: str
) -> RaceReading:
    """フェーズA（ゴール順画面）を解析する。"""
    rows = crop_rows(frame, roi)
    entries: list[dict] = []
    scores_by_rank: dict[int, int] = {}
    problems: list[str] = []

    for rank in range(1, 13):
        regions = rows.get(rank)
        if regions is None:
            problems.append(f"{rank}位のROIが未定義です")
            continue
        # 名前は生画像OCRが最も精度が高い（実キャプチャ検証 2026-11-18）
        name, name_conf = ocr.recognize_best(regions["name"])
        if not name:
            problems.append(f"{rank}位の名前を読み取れませんでした")
        # スコアは順位から規定配点で確定する。スコア列の数字はセグメント表示
        # フォントでOCRが安定しないため、読めた場合のクロスチェックにのみ使う。
        score = points_for_rank(rank)
        score_text, _ = ocr.recognize(regions["score"])
        ocr_score, _ = parse_points_text(score_text)
        if ocr_score is not None and ocr_score != score:
            problems.append(
                f"{rank}位のスコアOCR '{score_text}' が規定配点 {score} と一致しません"
            )
        total_text, _ = (
            ocr.recognize(regions["total"]) if "total" in regions else ("", 0.0)
        )
        scores_by_rank[rank] = score
        entries.append(
            {
                "rank": rank,
                "name": name,
                "score": score,
                "score_source": "rank_table",
                "score_ocr_text": score_text,
                "total_shown": parse_total_text(total_text),
                "name_confidence": name_conf,
            }
        )

    validation = None
    if len(scores_by_rank) == 12:
        validation = validate_scores(scores_by_rank)
        problems.extend(validation.problems)

    assignment: Assignment | None = None
    warnings: list[str] = []
    names = [e["name"] for e in entries]
    if len(names) == 12 and all(names):
        assignment = assign_teams(names, fmt)
        warnings.extend(assignment.warnings)
        if not assignment.valid:
            # タグなし名前（CPU名など）では通常起きる。読み取り自体の失敗ではない
            warnings.append("チーム割り当てに不整合があります（要手動確認）")
    else:
        problems.append("プレイヤー名を12名分読み取れませんでした")

    ok = not problems and (validation is None or validation.ok)
    return RaceReading(
        entries=entries,
        validation_ok=ok,
        problems=problems,
        assignment=assignment,
        warnings=warnings,
    )


def analyze_standings_frame(
    frame: np.ndarray, roi: dict, ocr: OcrEngine
) -> StandingsReading:
    """フェーズB（累計順位画面）を読み取る。照合専用で、集計には使わない。"""
    rows = crop_rows(frame, roi)
    entries: list[dict] = []
    problems: list[str] = []
    for position in range(1, 13):
        regions = rows.get(position)
        if regions is None:
            problems.append(f"{position}行目のROIが未定義です")
            continue
        name, _ = ocr.recognize_best(regions["name"])
        total_text, _ = (
            ocr.recognize(regions["total"]) if "total" in regions else ("", 0.0)
        )
        # 累計点はセグメントフォントでOCRが安定しないため best-effort。
        # 読めなくても照合（名前の並び順チェック）は継続できる。
        total = parse_total_text(total_text)
        if not name:
            problems.append(f"{position}行目の名前を読み取れませんでした")
        entries.append({"position": position, "name": name, "total": total})
    return StandingsReading(entries=entries, problems=problems)


def cross_check_standings(
    computed_totals: dict[str, int], standings: StandingsReading
) -> list[str]:
    """自前の累計（名前->累計点）とフェーズBの表示を照合し、不一致を返す。"""
    problems: list[str] = []
    computed = {normalize_name(k): v for k, v in computed_totals.items()}
    shown_order: list[str] = []
    for e in standings.entries:
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
    ]
    if shown_order != expected_order:
        problems.append("累計順位の並びが集計結果と一致しません")
    return problems
