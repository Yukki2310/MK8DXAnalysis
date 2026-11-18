#!/usr/bin/env python3
"""画像ベース識別の確認CLI: 2枚のリザルトフレームの行同士を照合する。

使い方: python tools/identity_check.py frameA.png frameB.png
各行について、相手フレームの最も近い行と距離・2位との差を表示する。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mk8dx.identity import (  # noqa: E402
    SAME_THRESHOLD, distance_matrix, extract_name_mask, trim_mask)
from mk8dx.preprocess import crop_rows, load_roi_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def masks(path: str, roi: dict):
    rows = crop_rows(cv2.imread(path), roi)
    return [trim_mask(extract_name_mask(rows[r]["name"])) for r in range(1, 13)]


def main() -> None:
    roi = load_roi_config(ROOT / "config" / "roi_1080p.json")
    a, b = masks(sys.argv[1], roi), masks(sys.argv[2], roi)
    m = distance_matrix(a, b)
    for i in range(12):
        o = np.argsort(m[i])
        verdict = "同一" if m[i, o[0]] < SAME_THRESHOLD else "新規/不在"
        print(f"{i+1:2d}位 -> {o[0]+1:2d}位 距離{m[i,o[0]]:.2f} "
              f"(2位候補{m[i,o[1]]:.2f}) {verdict}")


if __name__ == "__main__":
    main()
