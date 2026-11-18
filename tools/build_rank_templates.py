#!/usr/bin/env python3
"""順位列テンプレート（assets/rank_templates.npz）を実フレームから作る。

fixtures/frames/lounge/（tools/extract_frames.py で作成）の、目視で
リザルト画面と確認済みの区間のフレームを使う。

使い方: python tools/build_rank_templates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from mk8dx.preprocess import load_roi_config  # noqa: E402
from mk8dx.rankcheck import (  # noqa: E402
    MIN_MATCHED_ROWS, build_templates, matched_rows, save_templates)

FRAMES = ROOT / "fixtures" / "frames" / "lounge"
# 目視確認済みのリザルト画面区間（元動画の秒）
GOOD = [(350.0, 352.0), (731.0, 735.5), (895.0, 898.5)]


def main() -> None:
    roi = load_roi_config(ROOT / "config" / "roi_1080p.json")
    manifest = json.loads((FRAMES / "manifest.json").read_text())["frames"]
    frames = []
    for rec in manifest:
        if any(a <= rec["source_time"] <= b for a, b in GOOD):
            frames.append(cv2.imread(str(FRAMES / rec["file"])))
    templates = build_templates(frames, roi)
    out = ROOT / "assets" / "rank_templates.npz"
    out.parent.mkdir(exist_ok=True)
    save_templates(out, templates)
    print(f"{len(frames)} フレームから見本を作成: {out}")

    # 自己検証: 全フレームを走査し、一致行数の分布を表示する
    ok = bad = 0
    for rec in manifest:
        frame = cv2.imread(str(FRAMES / rec["file"]))
        n = matched_rows(frame, roi, templates)
        good = any(a <= rec["source_time"] <= b for a, b in GOOD)
        if good and n >= MIN_MATCHED_ROWS:
            ok += 1
        if not good and n >= MIN_MATCHED_ROWS:
            bad += 1
            print(f"  非リザルト区間で一致: t={rec['source_time']} 行数={n}")
    print(f"リザルト区間で成立: {ok} / 非リザルト区間の誤成立: {bad}")


if __name__ == "__main__":
    main()
