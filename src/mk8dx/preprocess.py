"""領域分割と画像前処理（設計書 3.3）。

ROI定義は config/roi_1080p.json から読み込む。座標は実キャプチャで校正する前提で、
すべて設定ファイル側に寄せてある（コードに直書きしない）。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

Rect = tuple[int, int, int, int]  # x, y, width, height


def load_roi_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def crop(frame: np.ndarray, rect: Rect) -> np.ndarray:
    x, y, w, h = rect
    return frame[y : y + h, x : x + w]


def crop_rows(frame: np.ndarray, roi: dict) -> dict[int, dict[str, np.ndarray]]:
    """1〜12位の行ごとに rank / name / score 領域を切り出す。"""
    rows: dict[int, dict[str, np.ndarray]] = {}
    for rank_str, regions in roi["rows"].items():
        if rank_str.startswith("_"):
            continue
        rank = int(rank_str)
        rows[rank] = {
            kind: crop(frame, tuple(regions[kind]))  # type: ignore[arg-type]
            for kind in ("rank", "name", "score")
        }
    return rows


def preprocess_for_ocr(image: np.ndarray, *, upscale: float = 2.0) -> np.ndarray:
    """OCR入力向け前処理: グレースケール -> コントラスト強調 -> 適応的二値化
    -> ノイズ除去 -> 拡大（小さい文字の認識率を上げる）。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if upscale != 1.0:
        cleaned = cv2.resize(
            cleaned, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )
    return cleaned
