"""ステップ4検証: RaceProcessor をステップ2の実データで検証する。

使い方:
    python tools/step4_check.py

- パイプライン橋渡し（extract_observations）がフレームから
  12行分の観測（名前＋国旗署名）を抽出できることを確認する。
- 事前登録選手名簿＋closed の RaceProcessor にレース3・4の観測を
  2フレームずつ流し、確定/要確認の振り分けとチーム集計を見る。
- レース1（選手名簿外の選手が多い）を open で流し、未照合として
  要確認に落ちることを確認する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mk8dx.ocr import OcrEngine
from mk8dx.pipeline import extract_observations
from mk8dx.preprocess import load_roi_config
from mk8dx.processor import RaceProcessor
from mk8dx.roster import Roster
from step3_check import CANONICAL_R34, load_segment

ROOT = Path(__file__).resolve().parent.parent


def seeded_processor(**kwargs) -> RaceProcessor:
    roster = Roster()
    for name in CANONICAL_R34:
        roster.seed(name)
    return RaceProcessor(roster, "2v2", **kwargs)


def show(label: str, outcome) -> None:
    print(f"-- {label}: status={outcome.status}")
    for t in outcome.teams:
        print(f"   team {t['tag']!r}: rows {t['members']} = {t['points']}点")
    total = sum(t["points"] for t in outcome.teams)
    if outcome.teams:
        print(f"   合計 {total}点")
    for p in outcome.problems:
        print(f"   problem: {p}")


def main() -> None:
    print("===== パイプライン橋渡し（extract_observations） =====")
    roi = load_roi_config(str(ROOT / "config/roi_1080p.json"))
    frame = cv2.imread(str(ROOT / "data/frames/710/seg0_race_result.png"))
    observations = extract_observations(frame, roi, OcrEngine())
    with_flag = sum(1 for o in observations if o.flag is not None)
    print(f"   12行抽出: 名前 {[o.name for o in observations]}")
    print(f"   国旗署名あり: {with_flag}/12")

    seg = {s: load_segment(s) for s in ["330", "710", "870"]}

    print()
    print("===== 事前登録選手名簿＋closed（2フレーム蓄積） =====")
    proc = seeded_processor(assume_closed=True, min_frames=2, stable_frames=2)
    for section, race, label in [("710", 3, "レース3"), ("870", 4, "レース4")]:
        observations, _entries = seg[section]
        proc.begin_race(race)
        print(f"-- {label}: frame1={proc.add_frame(observations)}", end="")
        print(f" frame2={proc.add_frame(observations)}")
        show(label, proc.finalize())

    print()
    print("===== レース1を open で（選手名簿外が多いケース） =====")
    proc = seeded_processor(assume_closed=False)
    observations, _entries = seg["330"]
    outcome = proc.process_single(observations, race_index=1)
    show("レース1", outcome)
    unresolved = sum(1 for e in outcome.entries if e["resolved_name"] is None)
    print(f"   未照合: {unresolved}/12")


if __name__ == "__main__":
    main()
