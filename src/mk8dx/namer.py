"""選手の表示名の読み取り（新規プレイヤーの名前提案）。

同一人物の判定は identity.py の画像照合が担うので、ここでの文字認識は
「新しく現れた選手に表示名の初期値を付ける」ためだけに使う。読みが
ずれても管理画面で直せば固定される（Roster.rename）。

認識エンジン（textrec.py）は精度の高いものを自動で選ぶ:
GPU があれば視覚言語モデル（Qwen3-VL）、無ければ PP-OCRv5。
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

NAME_TEXT_WIDTH = 365  # 名前の文字領域の幅（これより右は国旗）

# キリル文字の見た目が同じラテン文字への置き換え（VLM が 'Xy' を 'Ху' のように
# キリル文字で返すことがあるため。ラウンジの名前はラテン文字と日本語が中心）
_CYRILLIC_TO_LATIN = str.maketrans({
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y", "Ѕ": "S",
    "а": "a", "с": "c", "е": "e", "і": "i", "о": "o", "р": "p", "х": "x",
    "у": "y", "ѕ": "s", "б": "b", "Б": "B",
})
_FULLWIDTH_START, _FULLWIDTH_END = 0xFF01, 0xFF5E   # 全角の ！ ～ （半角 ! ~ との差は 0xFEE0）
_NON_ASCII = "[^\\x00-\\x7f]"


def clean_text(text: str) -> str:
    """認識結果の整形: 全角英数→半角、キリル文字の見間違い、★の表記ゆれ、
    日本語間の空白、末尾のゴミを直す。"""
    text = "".join(
        chr(ord(ch) - 0xFEE0) if _FULLWIDTH_START <= ord(ch) <= _FULLWIDTH_END else ch
        for ch in text
    )
    text = text.translate(_CYRILLIC_TO_LATIN)
    text = text.replace("*", "★").strip()
    text = re.sub(r"\s*[|｜]\s*$", "", text)         # 末尾の縦棒ゴミ
    text = re.sub(r"(?<=#)\s+", "", text)             # '# 1' -> '#1'
    # 日本語・記号（非ASCII）に隣り合う空白は認識が入れたものとして除く
    text = re.sub(rf"(?<={_NON_ASCII})\s+|\s+(?={_NON_ASCII})", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


class NameReader:
    """名前行クロップ群から表示名を読む。認識エンジンは精度の高いものを自動で選ぶ。"""

    def __init__(self, recognizer="auto") -> None:
        if recognizer == "auto":
            from .textrec import create_recognizer

            recognizer = create_recognizer()
        self._rec = recognizer

    @property
    def available(self) -> bool:
        return self._rec is not None and bool(getattr(self._rec, "available", False))

    @property
    def engine_name(self) -> str:
        return getattr(self._rec, "name", "none") if self.available else "none"

    @property
    def max_reads(self) -> int:
        """1選手あたりに読み取りを行う最大回数（精度の高いエンジンは1回で足りる）。"""
        return int(getattr(self._rec, "max_reads", 3))

    @property
    def can_suggest_teams(self) -> bool:
        """画像を見てチーム分け・タグ推定ができるエンジン（視覚言語モデル）か。"""
        return self.available and hasattr(self._rec, "suggest_teams")

    def suggest_teams(self, images: list[np.ndarray], team_size: int) -> list[dict] | None:
        if not self.can_suggest_teams:
            return None
        try:
            return self._rec.suggest_teams(
                [im[:, :NAME_TEXT_WIDTH] for im in images], team_size)
        except Exception:
            return None

    def label_teams(self, groups: list[list[np.ndarray]]) -> list[str]:
        """チームごとの共通タグを画像から読む。できなければ空文字のリスト。"""
        if not self.can_suggest_teams:
            return [""] * len(groups)
        try:
            cropped = [[im[:, :NAME_TEXT_WIDTH] for im in g] for g in groups]
            return [clean_text(t) for t in self._rec.label_teams(cropped)]
        except Exception:
            return [""] * len(groups)

    def read(self, crops: list[np.ndarray]) -> str:
        return self.read_many([crops])[0]

    __call__ = read

    def read_many(self, crop_lists: list[list[np.ndarray]]) -> list[str]:
        """選手ごとのクロップ群をまとめて読む（GPUではまとめた方が速い）。

        複数クロップは信頼度の合計が最大の読みに決める。
        """
        if not self.available:
            return [""] * len(crop_lists)
        per_read = int(getattr(self._rec, "crops_per_read", 3))
        flat: list[np.ndarray] = []
        owners: list[int] = []
        for k, crops in enumerate(crop_lists):
            for crop in crops[-per_read:]:
                flat.append(crop[:, :NAME_TEXT_WIDTH])
                owners.append(k)
        votes: list[dict[str, float]] = [defaultdict(float) for _ in crop_lists]
        for owner, (text, score) in zip(owners, self._rec.recognize_many(flat)):
            text = clean_text(text)
            if text:
                votes[owner][text] += max(score, 0.01)
        return [
            max(v.items(), key=lambda kv: kv[1])[0] if v else "" for v in votes
        ]
