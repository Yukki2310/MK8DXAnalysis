#!/usr/bin/env python3
"""ステップ1検証CLI: リザルト静止画1枚から ROI切り出し -> OCR -> 集計までを通す。

使い方:
  python tools/step1_check.py result.png --format 2v2 [--roi config/roi_1080p.json]

出力はJSON。ROI座標の校正とOCR精度の確認に使う。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from mk8dx.ocr import OcrEngine  # noqa: E402
from mk8dx.pipeline import analyze_result_frame  # noqa: E402
from mk8dx.preprocess import load_roi_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="MK8DX ステップ1 静止画OCR検証")
    parser.add_argument("image", help="リザルト画面のスクリーンショット (1920x1080想定)")
    parser.add_argument("--format", default="2v2", choices=["2v2", "3v3", "4v4"])
    parser.add_argument("--roi", default="config/roi_1080p.json")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"画像を読み込めませんでした: {args.image}", file=sys.stderr)
        return 1
    h, w = frame.shape[:2]
    print(f"画像サイズ: {w}x{h}", file=sys.stderr)

    roi = load_roi_config(args.roi)
    ocr = OcrEngine()
    reading = analyze_result_frame(frame, roi, ocr, args.format)

    output = {
        "entries": reading.entries,
        "validation_ok": reading.validation_ok,
        "problems": reading.problems,
        "warnings": reading.warnings,
        "teams": (
            [
                {"tag": t.tag, "normalized_tag": t.normalized_tag,
                 "members_rank": [i + 1 for i in t.members]}
                for t in reading.assignment.teams
            ]
            if reading.assignment
            else None
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if reading.validation_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
