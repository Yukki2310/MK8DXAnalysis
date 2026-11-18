"""順位列テンプレートによるリザルト画面の検証（OCR不要）。

順位の数字（1〜12）は固定位置・固定フォントで描画されるため、順位ごとの
白文字マスクの平均を「見本」にしておき、候補フレームの順位列と照合する。
コース選択画面やプレイヤー紹介画面など、名前列だけが似た画面を棄却できる。

見本は tools/build_rank_templates.py が実フレームから作り、
assets/rank_templates.npz に保存する（リポジトリに含める）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .preprocess import crop_rows

# 順位列の白文字: 全チャンネルが明るい
RANK_WHITE_MIN = 200
# 行の照合が成立する相関のしきい値、画面として成立する最低行数
ROW_MATCH_THRESHOLD = 0.6
MIN_MATCHED_ROWS = 8
# スコア列に「+N」の筆画（明るい画素）がどれだけあれば加算表示中とみなすか
SCORE_INK_MIN_PIXELS = 25


def rank_mask(rank_region: np.ndarray) -> np.ndarray:
    """順位列クロップから白文字マスク(float32, 0/1)を作る。"""
    return (rank_region.min(axis=2) >= RANK_WHITE_MIN).astype(np.float32)


def build_templates(
    frames: list[np.ndarray], roi: dict
) -> dict[int, np.ndarray]:
    """リザルト画面フレーム群から順位1〜12の見本マスクを作る（画素の過半数）。"""
    sums: dict[int, np.ndarray] = {}
    for frame in frames:
        rows = crop_rows(frame, roi)
        for rank in range(1, 13):
            m = rank_mask(rows[rank]["rank"])
            sums[rank] = sums.get(rank, 0) + m
    return {
        rank: (total / len(frames) >= 0.5).astype(np.float32)
        for rank, total in sums.items()
    }


def save_templates(path: str | Path, templates: dict[int, np.ndarray]) -> None:
    np.savez_compressed(path, **{str(k): v.astype(np.uint8) for k, v in templates.items()})


def load_templates(path: str | Path) -> dict[int, np.ndarray]:
    data = np.load(path)
    return {int(k): data[k].astype(np.float32) for k in data.files}


def _row_score(mask: np.ndarray, template: np.ndarray) -> float:
    """行マスクと見本の相関（±3px の平行移動を許容）。見本が空なら 0。"""
    if template.sum() == 0:
        return 0.0
    blurred = cv2.GaussianBlur(mask, (0, 0), 1.0)
    padded = cv2.copyMakeBorder(blurred, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
    tpl = cv2.GaussianBlur(template, (0, 0), 1.0)
    return float(cv2.matchTemplate(padded, tpl, cv2.TM_CCOEFF_NORMED).max())


def matched_rows(
    frame: np.ndarray, roi: dict, templates: dict[int, np.ndarray]
) -> int:
    """順位列の見本と一致した行数（0〜12）。"""
    rows = crop_rows(frame, roi)
    count = 0
    for rank in range(1, 13):
        regions = rows.get(rank)
        tpl = templates.get(rank)
        if not regions or tpl is None:
            continue
        if _row_score(rank_mask(regions["rank"]), tpl) >= ROW_MATCH_THRESHOLD:
            count += 1
    return count


def score_column_ink(frame: np.ndarray, roi: dict) -> int:
    """スコア列の明るい画素の総数。「+N」表示中は大きく、空の列は 0 に近い。"""
    rows = crop_rows(frame, roi)
    total = 0
    for rank in range(1, 13):
        regions = rows.get(rank)
        if regions and "score" in regions:
            # 右端の累計列に近い帯は除外せず、パネル背景より明るい白系だけ数える
            total += int((regions["score"].min(axis=2) >= 170).sum())
    return total


_default: dict[int, np.ndarray] | None = None


def default_templates() -> dict[int, np.ndarray] | None:
    """同梱の見本を読み込む（無ければ None）。"""
    global _default
    if _default is None:
        from .paths import resource_path

        path = resource_path("assets/rank_templates.npz")
        if path.exists():
            _default = load_templates(path)
    return _default


def verify_result_candidate(
    frame: np.ndarray, roi: dict, ocr: object = None
) -> bool:
    """候補フレームが本物のリザルト画面かを順位列の見本で検証する。

    見本が無い環境では名前列の検知結果をそのまま信じる（True）。
    """
    templates = default_templates()
    if templates is None:
        return True
    return matched_rows(frame, roi, templates) >= MIN_MATCHED_ROWS
