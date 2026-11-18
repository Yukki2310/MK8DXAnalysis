"""選手の表示名の読み取り（新規プレイヤーの名前提案）。

同一人物の判定は identity.py の画像照合が担うので、ここでの文字認識は
「新しく現れた選手に表示名の初期値を付ける」ためだけに使う。読みが
ずれても管理画面で直せば固定される（Roster.rename）。

入力は名前行クロップ（生の画像）。複数のクロップを読み、信頼度つきで
多数決する。
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

NAME_TEXT_WIDTH = 365  # 名前の文字領域の幅（これより右は国旗）


def clean_text(text: str) -> str:
    """OCR 出力の整形: ★の表記ゆれ・日本語間の空白・末尾のゴミを直す。"""
    text = text.replace("*", "★").strip()
    text = re.sub(r"\s*[|｜]\s*$", "", text)         # 末尾の縦棒ゴミ
    text = re.sub(r"(?<=#)\s+", "", text)             # '# 1' -> '#1'
    # 日本語・記号（非ASCII）に隣り合う空白は OCR が入れたものとして除く
    text = re.sub(r"(?<=[^\x00-\x7f])\s+|\s+(?=[^\x00-\x7f])", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


class NameReader:
    """名前行クロップ群から表示名を読む。"""

    def __init__(self, recognizer=None) -> None:
        if recognizer is None:
            from .textrec import NameRecognizer

            recognizer = NameRecognizer()
            recognizer.load()
        self._rec = recognizer

    @property
    def available(self) -> bool:
        return bool(getattr(self._rec, "available", False))

    def read(self, crops: list[np.ndarray]) -> str:
        """複数クロップの読みを、信頼度の合計が最大のものに決めて返す。"""
        if not self.available:
            return ""
        votes: dict[str, float] = defaultdict(float)
        for crop in crops:
            text, score = self._rec.recognize(crop[:, :NAME_TEXT_WIDTH])
            text = clean_text(text)
            if text:
                votes[text] += max(score, 0.01)
        if not votes:
            return ""
        return max(votes.items(), key=lambda kv: kv[1])[0]
