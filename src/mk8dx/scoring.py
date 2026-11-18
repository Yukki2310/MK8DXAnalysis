"""配点テーブルとレース結果のバリデーション（設計書 5.1）。

純ロジックのみで構成し、画像処理に依存しない。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 順位 -> 配点（ラウンジ形式）
POINTS: dict[int, int] = {
    1: 15, 2: 12, 3: 10, 4: 9, 5: 8, 6: 7,
    7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
}

# 1レース12名の合計は常に82点
TOTAL_POINTS = sum(POINTS.values())  # 82

VALID_SCORE_SET = frozenset(POINTS.values())


def points_for_rank(rank: int) -> int:
    """順位から規定配点を返す。"""
    return POINTS[rank]


@dataclass
class ValidationResult:
    ok: bool
    total: int
    problems: list[str] = field(default_factory=list)


def validate_scores(scores_by_rank: dict[int, int]) -> ValidationResult:
    """OCRで読み取った順位->スコアを検証する。

    チェック内容（設計書 5.1）:
    - 12名分そろっているか
    - 合計が82点か
    - スコアの集合が規定の12個の配点と完全一致するか（順位との対応は問わない。
      順位とスコアの対応ずれは合計検証では検出できないため、別途目視確認対象）
    """
    problems: list[str] = []
    if set(scores_by_rank.keys()) != set(POINTS.keys()):
        problems.append(
            f"順位の欠け・重複があります: 取得した順位={sorted(scores_by_rank.keys())}"
        )
    total = sum(scores_by_rank.values())
    if total != TOTAL_POINTS:
        problems.append(f"合計点が{TOTAL_POINTS}ではありません: {total}")
    if frozenset(scores_by_rank.values()) != VALID_SCORE_SET:
        problems.append(
            "スコア集合が規定配点と一致しません: "
            f"{sorted(scores_by_rank.values())}"
        )
    return ValidationResult(ok=not problems, total=total, problems=problems)


def expected_scores_by_rank() -> dict[int, int]:
    """順位どおりの規定配点（OCR結果との突き合わせ用）。"""
    return dict(POINTS)
