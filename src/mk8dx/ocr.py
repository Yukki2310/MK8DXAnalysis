"""PaddleOCR ラッパー（設計書 3.4）。

- スコア領域: 数字のみに制限し、規定配点の集合 {1..10, 12, 15} に
  含まれる値だけを有効とする（誤読の排除）。
- 名前領域: 日本語モデルと英語モデルのデュアルエンジンで読む。
  実動画検証（2026-11-18, ラウンジTier-X）で判明したこと:
  - japan モデルはラテン文字の名前が苦手（'Xb'->'メロ', 'F'->'E' 等）。
    en モデルは 'racist #1' 'FR Flowx' 等を正確に読む。
  - 逆に日本語名は en モデルでは読めない。名前ごとに候補を選ぶ。
  - PaddleOCR は名前を複数断片に分割して返すことがある。最大信頼度の
    断片だけを採用すると名前の前半/後半が欠けるため、断片は
    x座標順に結合する。
PaddleOCR の読み込みは重いため、初回利用時に遅延初期化する。
"""
from __future__ import annotations

import re

import numpy as np

from .scoring import VALID_SCORE_SET

_DIGITS = re.compile(r"\d+")
_HIRAGANA = re.compile(r"[ぁ-ゖ]")
_KANJI = re.compile(r"[一-龥]")
_KATAKANA = re.compile(r"[ァ-ヶ]")


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
    なお動画圧縮で '+' の筆画が劣化すると japan モデルが 'ト' や '十' と
    誤読する（実動画検証 2026-11-18）。スコア列は数字と '+' しか存在しない
    領域なので、先頭の 'ト'/'十' も '+' とみなす。
    """
    stripped = text.strip()
    has_plus = (
        "+" in text
        or "＋" in text
        or stripped.startswith("ト")
        or stripped.startswith("十")
    )
    m = _DIGITS.search(text)
    if not m:
        return None, has_plus
    value = int(m.group())
    return (value if value in VALID_SCORE_SET else None), has_plus


def choose_name_candidate(
    jp: tuple[str, float], en: tuple[str, float]
) -> tuple[str, float]:
    """日本語モデルと英語モデルの名前候補から1つを選ぶ。

    ルール（実動画検証 2026-11-18）:
    - jp候補にひらがな・漢字が含まれる、またはカタカナが3文字以上
      含まれる場合は日本語（混在）名とみなして jp を優先する。
      例: 'A★ぜんだかいとしき', 'Xb◇ねえあまぁƄ'
    - それ以外（純ラテン名）は信頼度が高い方を選ぶ。jp が 'Xb' を
      'メロ'（カタカナ2文字）と誤読するケースを en が救う。
    最終的な同一人物への名寄せはロスター照合（レース横断の多数決）が担う。
    """
    jp_text, jp_conf = jp
    en_text, en_conf = en
    if not jp_text:
        return en
    if not en_text:
        return jp
    if (
        _HIRAGANA.search(jp_text)
        or _KANJI.search(jp_text)
        or len(_KATAKANA.findall(jp_text)) >= 3
    ):
        return jp
    return jp if jp_conf >= en_conf else en


class OcrEngine:
    def __init__(self, lang: str = "japan"):
        self._lang = lang
        self._ocr = None
        self._ocr_en = None

    def _ensure(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR  # 重い依存は遅延 import

            self._ocr = PaddleOCR(use_angle_cls=False, lang=self._lang, show_log=False)
        return self._ocr

    def _ensure_en(self):
        if self._ocr_en is None:
            from paddleocr import PaddleOCR

            self._ocr_en = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        return self._ocr_en

    @staticmethod
    def _run(ocr, image: np.ndarray) -> tuple[str, float]:
        result = ocr.ocr(image, cls=False)
        if not result or not result[0]:
            return "", 0.0
        # 断片は x 座標順に結合する（最大信頼度の断片だけだと名前が欠ける）
        parts: list[tuple[float, str, float]] = []
        for line in result[0]:
            box, (text, conf) = line[0], line[1]
            x = min(p[0] for p in box)
            parts.append((x, text.strip(), float(conf)))
        parts.sort(key=lambda p: p[0])
        text = "".join(p[1] for p in parts)
        conf = min((p[2] for p in parts), default=0.0)
        return text, conf

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """画像からテキストと信頼度を返す（断片はx順に結合）。"""
        return self._run(self._ensure(), image)

    def recognize_best(self, image: np.ndarray) -> tuple[str, float]:
        """生画像と2倍・3倍拡大でOCRし、信頼度の高いものを返す。

        実キャプチャ検証（2026-11-18）で、名前領域は二値化すると読めなくなり、
        生画像（＋拡大）が最も精度が高いことを確認したための方式。
        """
        from .preprocess import upscale

        candidates = [
            self.recognize(image),
            self.recognize(upscale(image, 2.0)),
            self.recognize(upscale(image, 3.0)),
        ]
        return max(candidates, key=lambda c: c[1])

    def recognize_name(self, image: np.ndarray) -> tuple[str, float]:
        """名前領域専用: 日本語/英語デュアルエンジンで読み、候補を選ぶ。"""
        from .preprocess import upscale

        jp = max(
            (self._run(self._ensure(), v) for v in (image, upscale(image, 2.0))),
            key=lambda c: c[1],
        )
        en = max(
            (self._run(self._ensure_en(), v) for v in (image, upscale(image, 2.0))),
            key=lambda c: c[1],
        )
        return choose_name_candidate(jp, en)

    def recognize_digits_en(self, image: np.ndarray) -> tuple[int | None, float]:
        """数字だけの領域を英語モデル＋拡大で読む（順位列など）。

        japan モデルは '1' を '-' と誤読したり落としたりするが、
        en モデルは拡大で安定して読める（実動画検証 2026-11-18）。
        """
        from .preprocess import upscale

        en = self._ensure_en()
        best: tuple[int | None, float] = (None, 0.0)
        for v in (image, upscale(image, 2.0), upscale(image, 3.0)):
            text, conf = self._run(en, v)
            value = parse_total_text(text)
            if value is not None and conf > best[1]:
                best = (value, conf)
        return best

    def recognize_score(self, image: np.ndarray) -> int | None:
        """スコア領域専用: 数字だけを拾い、規定配点に含まれる場合のみ返す。"""
        text, _ = self.recognize(image)
        value, _ = parse_points_text(text)
        return value

    def recognize_total(self, image: np.ndarray) -> int | None:
        """累計点領域専用: セグメント表示風の数字を英語モデルで読む。

        実動画検証（2026-11-18）: japan モデルの素読みではほぼ読めなかったが、
        en モデル＋拡大で読めるようになる。背景が明るい行は二値化（Otsu）や
        反転が効くため、複数バリアントで読み、信頼度0.9以上の値の多数決、
        なければ最高信頼度の値を採用する。
        """
        import cv2

        from .preprocess import upscale

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
        inv_bgr = cv2.cvtColor(cv2.bitwise_not(otsu), cv2.COLOR_GRAY2BGR)
        variants = [
            upscale(image, 2.0),
            upscale(image, 3.0),
            upscale(otsu_bgr, 2.0),
            upscale(inv_bgr, 2.0),
        ]
        en = self._ensure_en()
        reads: list[tuple[int, float]] = []
        for v in variants:
            text, conf = self._run(en, v)
            value = parse_total_text(text)
            if value is not None:
                reads.append((value, conf))
        if not reads:
            return None
        strong = [(v, c) for v, c in reads if c >= 0.9]
        if strong:
            counts: dict[int, int] = {}
            for v, _ in strong:
                counts[v] = counts.get(v, 0) + 1
            return max(counts.items(), key=lambda kv: kv[1])[0]
        return max(reads, key=lambda rc: rc[1])[0]
