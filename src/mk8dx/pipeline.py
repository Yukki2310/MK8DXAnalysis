"""リザルトフレーム解析パイプライン（設計書 3.3 / 3.4 / 4 / 5.1）。

フレーム -> ROI切り出し -> 前処理 -> OCR -> タグ判定・チーム割り当て -> 検証。
ステップ1（静止画検証）とステップ3（リアルタイム）の両方から使う共通処理。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import Assignment, assign_teams
from .ocr import OcrEngine
from .preprocess import crop_rows, preprocess_for_ocr
from .scoring import points_for_rank, validate_scores


@dataclass
class RaceReading:
    entries: list[dict]  # [{rank, name, score, raw_name_text, score_confidence...}]
    validation_ok: bool
    problems: list[str]
    assignment: Assignment | None
    warnings: list[str] = field(default_factory=list)


def analyze_result_frame(
    frame: np.ndarray, roi: dict, ocr: OcrEngine, fmt: str
) -> RaceReading:
    rows = crop_rows(frame, roi)
    entries: list[dict] = []
    scores_by_rank: dict[int, int] = {}
    problems: list[str] = []

    for rank in range(1, 13):
        regions = rows.get(rank)
        if regions is None:
            problems.append(f"{rank}位のROIが未定義です")
            continue
        name_img = preprocess_for_ocr(regions["name"])
        score_img = preprocess_for_ocr(regions["score"])
        name, name_conf = ocr.recognize(name_img)
        score = ocr.recognize_score(score_img)
        if score is None:
            problems.append(f"{rank}位のスコアを読み取れませんでした")
            # 順位からの規定配点で仮置きし、検証で不一致として検出させる
            score = -1
        elif score != points_for_rank(rank):
            problems.append(
                f"{rank}位のスコア {score} が規定配点 {points_for_rank(rank)} と一致しません"
            )
        scores_by_rank[rank] = score
        entries.append(
            {
                "rank": rank,
                "name": name,
                "score": score,
                "name_confidence": name_conf,
            }
        )

    validation = validate_scores(
        {r: s for r, s in scores_by_rank.items() if s > 0}
    ) if len(scores_by_rank) == 12 and all(s > 0 for s in scores_by_rank.values()) else None
    if validation is not None:
        problems.extend(validation.problems)

    assignment: Assignment | None = None
    warnings: list[str] = []
    names = [e["name"] for e in entries]
    if len(names) == 12 and all(names):
        assignment = assign_teams(names, fmt)
        warnings.extend(assignment.warnings)
        if not assignment.valid:
            problems.append("チーム割り当てに不整合があります（要手動確認）")
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
