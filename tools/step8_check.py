#!/usr/bin/env python3
"""フェーズ1検証: OCRなし（画像マスクのみ）で名簿の名寄せが成立するか確認する。

fixtures/frames/lounge/（tools/extract_frames.py で作成）のフェーズAフレームを
レースごとに RaceProcessor へ流し、行→エントリID の割り当てと名簿の
エントリ数、1フレームあたりの処理時間を表示する。名前は空文字で渡すため、
識別はすべて文字マスク（と国旗）だけに依存する。

使い方: python tools/step8_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from mk8dx.identity import extract_name_mask, trim_mask  # noqa: E402
from mk8dx.preprocess import crop_rows, load_roi_config  # noqa: E402
from mk8dx.processor import RaceProcessor  # noqa: E402
from mk8dx.roster import Observation, Roster, extract_flag_signature  # noqa: E402

FRAMES = ROOT / "fixtures" / "frames" / "lounge"
# レースごとのフェーズA区間（元動画の秒）。実動画で目視確認した区間
RACES = {1: (350.0, 352.0), 3: (731.0, 733.5), 4: (895.0, 898.0)}


def observations(frame, roi):
    rows = crop_rows(frame, roi)
    out = []
    for rank in range(1, 13):
        mask = trim_mask(extract_name_mask(rows[rank]["name"]))
        out.append(Observation(
            name="", flag=extract_flag_signature(frame, roi, rank), mask=mask))
    return out


def main() -> None:
    roi = load_roi_config(ROOT / "config" / "roi_1080p.json")
    manifest = json.loads((FRAMES / "manifest.json").read_text())["frames"]
    roster = Roster()
    proc = RaceProcessor(roster, "2v2")
    for race, (t0, t1) in RACES.items():
        proc.begin_race(race)
        times = []
        for rec in manifest:
            if t0 <= rec["source_time"] <= t1:
                frame = cv2.imread(str(FRAMES / rec["file"]))
                t = time.perf_counter()
                obs = observations(frame, roi)
                proc.add_frame(obs)
                times.append(time.perf_counter() - t)
        outcome = proc.finalize(force=True)
        res = roster.resolve(proc.consensus())
        ids = res.entry_ids
        dup = {i for i in ids if i is not None and ids.count(i) > 1}
        print(f"レース{race}: フレーム{len(times)}枚 平均{sum(times)/len(times)*1000:.0f}ms/枚 "
              f"名簿{len(roster.entries)}件")
        print("  行→ID:", ids, "重複ID:", sorted(dup) or "なし",
              "未割当:", res.new_candidates or "なし")


if __name__ == "__main__":
    main()
