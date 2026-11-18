"""リザルト画面検知（設計書 3.2）。

- ヘッダー領域のテンプレート照合（テンプレート画像がある場合）
- フレーム間差分による静止判定（アニメーション終了の検出）
テンプレートは実キャプチャから切り出して用意する想定で、無い場合は
静止判定のみで動作する（検知精度はステップ2で実地検証する）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .preprocess import crop


class StillnessJudge:
    """連続フレームの平均絶対差分が閾値以下なら静止とみなす。"""

    def __init__(self, threshold: float = 2.0, required_stable_checks: int = 4):
        self.threshold = threshold
        self.required_stable_checks = required_stable_checks
        self._prev: np.ndarray | None = None
        self._stable_count = 0

    def reset(self) -> None:
        self._prev = None
        self._stable_count = 0

    def update(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180))
        if self._prev is None:
            self._prev = small
            return False
        diff = float(cv2.absdiff(small, self._prev).mean())
        self._prev = small
        if diff <= self.threshold:
            self._stable_count += 1
        else:
            self._stable_count = 0
        return self._stable_count >= self.required_stable_checks


class ResultScreenDetector:
    def __init__(
        self,
        roi: dict,
        template_path: str | Path | None = None,
        match_threshold: float = 0.85,
    ):
        self.roi = roi
        self.match_threshold = match_threshold
        self._template: np.ndarray | None = None
        if template_path and Path(template_path).exists():
            img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self._template = img

    def is_result_screen(self, frame: np.ndarray) -> bool:
        """リザルト画面らしいかを返す。テンプレートが無い場合は常に False
        （ステップ2でテンプレートを用意してから有効化する）。"""
        if self._template is None:
            return False
        header = crop(frame, tuple(self.roi["header_template_region"]))  # type: ignore[arg-type]
        gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
        if gray.shape != self._template.shape:
            gray = cv2.resize(gray, (self._template.shape[1], self._template.shape[0]))
        result = cv2.matchTemplate(gray, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val) >= self.match_threshold
