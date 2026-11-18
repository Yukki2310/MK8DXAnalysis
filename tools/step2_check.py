#!/usr/bin/env python3
"""ステップ2検証CLI: 動画ファイルからリザルト画面区間を検出し、
フェーズ判別・名前OCR・チームクラスタリングまでを通す。

使い方:
  python tools/step2_check.py section.mp4 [--offset 330] [--fps 2] [--out report.json]

- --offset: 動画全体（元動画）におけるこのファイルの開始秒。セクション切り出し
  ファイルのタイムスタンプを元動画の時刻に直して報告するために使う。
- 検出方法: panel領域の行構造（名前列のエッジ密度）でリザルト画面らしい
  区間を拾い、区間ごとに複数候補フレームを完全解析して最良のものを採用する。
- 誤検出対策: コース選択画面も行状テキストで引っかかるため、順位列の数字
  （1/6/12位）が読めることをリザルト画面の条件にする。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mk8dx.clustering import assign_teams  # noqa: E402
from mk8dx.ocr import OcrEngine  # noqa: E402
from mk8dx.pipeline import (  # noqa: E402
    PHASE_RACE_RESULT,
    PHASE_STANDINGS,
    PHASE_UNKNOWN,
    analyze_result_frame,
    analyze_standings_frame,
    classify_phase,
)
from mk8dx.preprocess import crop_rows, edge_density, load_roi_config  # noqa: E402


def frame_metrics(frame: np.ndarray, roi: dict) -> dict:
    """リザルト画面らしさの軽量指標を返す（OCRなし）。"""
    rows = crop_rows(frame, roi)
    name_ink = []
    score_ink = []
    for regions in rows.values():
        if "name" in regions:
            name_ink.append(edge_density(regions["name"]))
        if "score" in regions:
            score_ink.append(edge_density(regions["score"]))
    return {
        "name_ink": float(np.mean(name_ink)) if name_ink else 0.0,
        "score_ink": float(np.mean(score_ink)) if score_ink else 0.0,
        "name_rows_with_ink": sum(1 for v in name_ink if v > 0.02),
    }


def scan_video(
    path: str, roi: dict, sample_fps: float = 2.0,
    name_ink_threshold: float = 0.03,
) -> list[dict]:
    """動画をサンプリングし、リザルト画面らしい区間のリストを返す。

    各区間: {start, end, candidates: [(time, frame), ...]}（秒はファイル内時刻）。
    候補フレームは区間の最初・中間・最後付近から最大4枚まで保持する。
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / sample_fps)))
    segments: list[dict] = []
    current: dict | None = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        t = idx / src_fps
        idx += 1
        if frame.shape[1] != 1920 or frame.shape[0] != 1080:
            frame = cv2.resize(frame, (1920, 1080))
        m = frame_metrics(frame, roi)
        looks = m["name_rows_with_ink"] >= 10 and m["name_ink"] >= name_ink_threshold
        if looks:
            if current is None:
                current = {"start": t, "end": t, "candidates": [(t, frame.copy())]}
            else:
                current["end"] = t
                # 候補は1.5秒以上の間隔で追加し、時間的に分散させて最大5枚
                if t - current["candidates"][-1][0] >= 1.5:
                    current["candidates"].append((t, frame.copy()))
                    if len(current["candidates"]) > 5:
                        current["candidates"].pop(1)
        else:
            if current is not None:
                if current["end"] - current["start"] >= 1.0:
                    segments.append(current)
                current = None
    if current is not None and current["end"] - current["start"] >= 1.0:
        segments.append(current)
    cap.release()
    return segments


def rank_column_ok(frame: np.ndarray, roi: dict, ocr: OcrEngine) -> bool:
    """順位列の数字が読めるかでリザルト画面かを確認する。

    コース選択画面などの誤検出を弾くための検証。フェーズBの並び替え
    アニメーション中は中段の順位数字が一時的に重複表示されるため、
    両端の行（1・2・11・12位）を使い、4行中2行以上の一致で合格とする。
    """
    rows = crop_rows(frame, roi)
    matched = 0
    checked = 0
    for rank in (1, 2, 11, 12):
        regions = rows.get(rank)
        if not regions or "rank" not in regions:
            continue
        checked += 1
        # 順位数字は en モデル＋拡大が安定する（japan は '1'->'-' 等の誤読が多い）
        value, _ = ocr.recognize_digits_en(regions["rank"])
        if value == rank:
            matched += 1
    return checked > 0 and matched >= 2


def analyze_candidate(frame: np.ndarray, roi: dict, ocr: OcrEngine) -> dict:
    """候補フレーム1枚を完全解析し、品質スコア付きの結果を返す。"""
    phase = classify_phase(frame, roi, ocr)
    item: dict = {"phase": phase}
    if phase == PHASE_RACE_RESULT:
        reading = analyze_result_frame(frame, roi, ocr, "2v2")
        item["entries"] = reading.entries
        item["problems"] = reading.problems
        names = [e["name"] for e in reading.entries]
        item["quality"] = sum(1 for n in names if n) + sum(
            e.get("name_confidence", 0.0) for e in reading.entries
        ) / 10.0
        formats: dict = {}
        if len(names) == 12 and all(names):
            for fmt in ("2v2", "3v3", "4v4"):
                a = assign_teams(names, fmt)
                formats[fmt] = {
                    "valid": a.valid,
                    "warnings": a.warnings,
                    "teams": [
                        {"tag": t.tag, "members_rank": [j + 1 for j in t.members]}
                        for t in a.teams
                    ],
                }
        item["clustering"] = formats
    else:
        # standings / unknown（プレーン表示）は名前抽出を共通化する
        standings = analyze_standings_frame(frame, roi, ocr)
        item["entries"] = standings.entries
        item["problems"] = standings.problems
        names = [e["name"] for e in standings.entries]
        item["quality"] = sum(1 for n in names if n)
        if phase == PHASE_STANDINGS and len(names) == 12 and all(names):
            formats = {}
            for fmt in ("2v2", "3v3", "4v4"):
                a = assign_teams(names, fmt)
                formats[fmt] = {
                    "valid": a.valid,
                    "warnings": a.warnings,
                    "teams": [
                        {"tag": t.tag, "members_position": [j + 1 for j in t.members]}
                        for t in a.teams
                    ],
                }
            item["clustering"] = formats
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="MK8DX ステップ2 動画検証")
    parser.add_argument("video", help="リザルト画面を含む動画ファイル (1920x1080想定)")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="元動画における開始秒（セクション切り出し用）")
    parser.add_argument("--fps", type=float, default=2.0, help="サンプリングfps")
    parser.add_argument("--roi", default="config/roi_1080p.json")
    parser.add_argument("--out", default=None, help="結果JSONの出力先")
    parser.add_argument("--frames-dir", default=None,
                        help="採用フレーム画像の保存先ディレクトリ")
    args = parser.parse_args()

    roi = load_roi_config(args.roi)
    ocr = OcrEngine()
    segments = scan_video(args.video, roi, sample_fps=args.fps)
    print(f"検出区間: {len(segments)} 件", file=sys.stderr)

    results = []
    for i, seg in enumerate(segments):
        best: dict | None = None
        best_frame = None
        best_time = None
        rejected = 0
        for t, frame in seg["candidates"]:
            if not rank_column_ok(frame, roi, ocr):
                rejected += 1
                continue
            item = analyze_candidate(frame, roi, ocr)
            if best is None or item.get("quality", 0) > best.get("quality", 0):
                best, best_frame, best_time = item, frame, t
        if best is None:
            print(
                f"  区間{i}: 候補{len(seg['candidates'])}枚すべて順位列検証で棄却"
                "（リザルト画面ではない可能性）",
                file=sys.stderr,
            )
            results.append({
                "segment": i,
                "global_start": args.offset + seg["start"],
                "global_end": args.offset + seg["end"],
                "phase": "rejected",
                "candidates_rejected": rejected,
            })
            continue
        best.update({
            "segment": i,
            "global_start": args.offset + seg["start"],
            "global_end": args.offset + seg["end"],
            "best_global_time": args.offset + (best_time or seg["start"]),
            "candidates_rejected": rejected,
        })
        if args.frames_dir and best_frame is not None:
            out_dir = Path(args.frames_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"seg{i}_{best['phase']}.png"), best_frame)
        results.append(best)
        print(
            f"  区間{i}: 元動画 {best['global_start']:.0f}s-{best['global_end']:.0f}s "
            f"phase={best['phase']} quality={best.get('quality', 0):.1f}",
            file=sys.stderr,
        )

    payload = {"video": args.video, "offset": args.offset, "segments": results}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
