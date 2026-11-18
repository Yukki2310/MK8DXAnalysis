"""PaddleOCR ラッパー（設計書 3.4）。

- スコア領域: 数字のみに制限し、規定配点の集合 {1..10, 12, 15} に
  含まれる値だけを有効とする（誤読の排除）。
- 名前領域: 日本語・英数字・記号をそのまま受け取る。
PaddleOCR の読み込みは重いため、初回利用時に遅延初期化する。
"""
from __future__ import annotations

import re

import numpy as np

from .scoring import VALID_SCORE_SET

_DIGITS = re.compile(r"\d+")


def parse_total_text(text: str) -> int | None:
    """累計点テキストから整数を取り出す（読めない場合は None）。"""
    m = _DIGITS.search(text)
    if not m:
        return None
    value = int(m.group())
    return value if 0 <= value <= 999 else None


def parse_points_text(text: str) -> tuple[int | None, bool]:
    """スコア列のOCRテキストを解釈する。

    ゴール順画面では '+15' のように '+' 付きで表示されるため、
    (値, '+' があったか) を返す。'+' の有無が2フェーズ判別の鍵になる。
    """
    has_plus = "+" in text or "＋" in text
    m = _DIGITS.search(text)
    if not m:
        return None, has_plus
    value = int(m.group())
    return (value if value in VALID_SCORE_SET else None), has_plus


class OcrEngine:
    def __init__(self, lang: str = "japan"):
        self._lang = lang
        self._ocr = None

    def _ensure(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR  # 重い依存は遅延 import

            self._ocr = PaddleOCR(use_angle_cls=False, lang=self._lang, show_log=False)
        return self._ocr

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """画像から最も確からしいテキストと信頼度を返す。"""
        ocr = self._ensure()
        result = ocr.ocr(image, cls=False)
        if not result or not result[0]:
            return "", 0.0
        # 面積ではなく信頼度最大の断片を採用（ROIは単一テキスト前提）
        best_text, best_conf = "", 0.0
        for line in result[0]:
            text, conf = line[1][0], float(line[1][1])
            if conf > best_conf:
                best_text, best_conf = text, conf
        return best_text.strip(), best_conf

    def recognize_best(self, image: np.ndarray) -> tuple[str, float]:
        """生画像と2倍拡大の両方でOCRし、信頼度の高い方を返す。

        実キャプチャ検証（2026-11-18）で、名前領域は二値化すると読めなくなり、
        生画像（＋拡大）が最も精度が高いことを確認したための方式。
        """
        from .preprocess import upscale

        candidates = [self.recognize(image), self.recognize(upscale(image, 2.0))]
        return max(candidates, key=lambda c: c[1])

    def recognize_score(self, image: np.ndarray) -> int | None:
        """スコア領域専用: 数字だけを拾い、規定配点に含まれる場合のみ返す。"""
        text, _ = self.recognize(image)
        value, _ = parse_points_text(text)
        return value

    def recognize_total(self, image: np.ndarray) -> int | None:
        """累計点領域専用: 任意の正の整数として読み取る（最大12レース×15点=180）。"""
        text, _ = self.recognize(image)
        m = _DIGITS.search(text)
        if not m:
            return None
        value = int(m.group())
        return value if 0 <= value <= 999 else None
