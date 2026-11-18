"""選手の表示名の読み取り（新規プレイヤーの名前提案）。

同一人物の判定は identity.py の画像照合が担うので、ここでの文字認識は
「新しく現れた選手に表示名の初期値を付ける」ためだけに使う。読みが
ずれても管理画面で直せば固定される（Roster.rename）。

入力は名前行クロップ。identity.extract_name_mask で文字だけを取り出し、
白地に黒字へ反転して2倍に拡大すると Windows OCR の認識が安定する
（実測 2026-11-18: 生画像より英字の読みが良い）。日本語エンジンと英語
エンジンの結果から候補を選び、複数クロップがあれば多数決する。
"""
from __future__ import annotations

import re
from collections import Counter

import cv2
import numpy as np

from .identity import extract_name_mask

_CJK = re.compile(r"[ぁ-んァ-ヶ一-龥ー]")
_KANA = re.compile(r"[ぁ-んァ-ヶ]")
_ASCII_WORD = re.compile(r"[0-9A-Za-z]{2,}")


def prepare_for_ocr(name_region: np.ndarray) -> np.ndarray | None:
    """名前クロップを OCR 用画像（白地黒字・2倍）にする。文字が無ければ None。"""
    mask = extract_name_mask(name_region)
    ys, xs = np.where(mask > 0)
    if len(xs) < 40:
        return None
    mask = mask[max(ys.min() - 6, 0): ys.max() + 7,
                max(xs.min() - 10, 0): xs.max() + 11]
    img = cv2.resize(255 - mask * 255, None, fx=2, fy=2,
                     interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def clean_text(text: str) -> str:
    """OCR 出力の整形: ★の表記ゆれ・日本語間の空白・末尾のゴミを直す。"""
    text = text.replace("*", "★").strip()
    text = re.sub(r"\s*[|｜]\s*$", "", text)         # 末尾の縦棒ゴミ
    text = re.sub(r"(?<=#)\s+", "", text)             # '# 1' -> '#1'
    # 日本語・記号（非ASCII）に隣り合う空白は OCR が入れたものとして除く
    text = re.sub(r"(?<=[^\x00-\x7f])\s+|\s+(?=[^\x00-\x7f])", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def choose_candidate(candidates: list[str]) -> str:
    """[日本語エンジン, 英語エンジン] の読みから1つ選ぶ。

    日本語エンジンの読みに仮名が含まれるなら日本語名とみなして採用する
    （漢字だけでは判断しない: 'xı' を '刈' と誤読する例があった）。
    それ以外は、英語エンジンが英数字2文字以上を読めていればそれを採用し、
    無ければ日本語側（漢字だけの名前など）を使う。
    """
    cleaned = [clean_text(c) for c in candidates]
    ja = cleaned[0] if cleaned else ""
    en = cleaned[1] if len(cleaned) > 1 else ""
    if ja and _KANA.search(ja):
        return ja
    if en and _ASCII_WORD.search(en):
        return en
    return ja or en


class NameReader:
    """Windows OCR で名前行クロップ群から表示名を読む。"""

    def __init__(self, ocr=None) -> None:
        if ocr is None:
            from .winocr import WindowsOcr

            ocr = WindowsOcr()
            ocr.load()
        self._ocr = ocr

    @property
    def available(self) -> bool:
        return bool(getattr(self._ocr, "available", False))

    def read(self, crops: list[np.ndarray]) -> str:
        """複数クロップの読みを多数決して返す。読めなければ空文字。"""
        if not self.available:
            return ""
        votes: Counter = Counter()
        for crop in crops:
            prepared = prepare_for_ocr(crop)
            if prepared is None:
                continue
            text = choose_candidate(self._ocr.recognize(prepared))
            if text:
                votes[text] += 1
        if not votes:
            return ""
        return votes.most_common(1)[0][0]
