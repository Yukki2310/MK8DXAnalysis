"""ステップ7検証: Screen AI の実読みで自動構築が成立するか。

ユーザーの Windows PC で取得したベンチ結果（Screen AI の実読み）を
観測として、ステップ6と同じレース1→3→4の自動構築シナリオを流す。
国旗署名はローカルのフレーム画像から抽出する（OCR対象外のため）。
ステップ6ではレース3・4でチーム割り当てが不成立だった。ここが
Screen AI の安定した読みで成立するかが焦点。

使い方: python tools/step7_check.py [ベンチ結果JSONのパス]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mk8dx.preprocess import load_roi_config  # noqa: E402
from mk8dx.processor import RaceProcessor  # noqa: E402
from mk8dx.roster import Observation, Roster, extract_flag_signature  # noqa: E402
from mk8dx.tags import normalize_name  # noqa: E402

DEFAULT_JSON = Path.home() / "workspace/user/files/screenai_bench_result.json"

RACES = [
    ("race1_a.jpg", "data/frames/330/seg1_race_result.png",
     ["xı c.qq", "RR★", "Rev◆", "ẇf ths", "racist #2",
      "A★ぜんだかいとしき", "FR", "Xb◇ねえあまぁƄ", "Xb",
      "racist #1", "FR", "xı avalon"]),
    ("race3_a.jpg", "data/frames/710/seg0_race_result.png",
     ["xı c.qq", "FR", "racist #1", "Xb", "RR★", "FR Flowx",
      "A★ぜんだかいとしき", "A A", "xı avalon",
      "Xb◇ねえあまぁƄ", "RR★", "racist #2"]),
    ("race4_a.jpg", "data/frames/870/seg2_race_result.png",
     ["A★ぜんだかいとしき", "racist #2", "A A", "FR Flowx",
      "Xb◇ねえあまぁƄ", "racist #1", "xı c.qq", "Xb",
      "xı avalon", "RR★", "FR", "RR★"]),
]


def run_mode(bench: dict, mode: str, roi: dict) -> None:
    print(f"\n########## モード: {mode} ##########")
    processor = RaceProcessor(Roster(), "2v2", assume_closed=False)
    by_sample = {r["sample"]: r["modes"][mode]["reads"] for r in bench}
    for idx, (sample, frame_rel, truth) in enumerate(RACES, start=1):
        frame = cv2.imread(str(ROOT / frame_rel))
        reads = by_sample[sample]
        observations = [
            Observation(name=reads[i],
                        flag=extract_flag_signature(frame, roi, i + 1))
            for i in range(12)
        ]
        outcome = processor.process_single(observations, race_index=idx)
        hits = sum(
            1 for e, t in zip(outcome.entries, truth)
            if e["resolved_name"] is not None
            and normalize_name(e["resolved_name"]) == normalize_name(t)
        )
        print(f"\n===== レース{idx}（{sample}） status={outcome.status} "
              f"厳密一致 {hits}/12 =====")
        for e, t in zip(outcome.entries, truth):
            mark = ("○" if e["resolved_name"] is not None
                    and normalize_name(e["resolved_name"]) == normalize_name(t)
                    else "×")
            print(f"  {mark} {e['rank']:2d}位: 正解={t!r} "
                  f"読み={reads[e['rank']-1]!r} 名寄せ={e['resolved_name']!r}")
        if outcome.teams:
            total = sum(t["points"] for t in outcome.teams)
            print(f"  ★チーム割り当て成立: "
                  f"{[(t['tag'], t['points']) for t in outcome.teams]} "
                  f"合計 {total}")
        else:
            print("  チーム割り当て: 不成立")
        for p in outcome.problems:
            print(f"  problem: {p}")
    roster = processor.roster
    print(f"\n  最終名簿: {len(roster.entries)}エントリ")
    for e in roster.entries:
        print(f"    {e.canonical_name!r} "
              f"({'本' if e.confirmed else '仮'}, 出現{e.races_seen})")


def main() -> None:
    bench_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    roi = load_roi_config(str(ROOT / "config/roi_1080p.json"))
    for mode in ("crop2x", "crop"):
        run_mode(bench, mode, roi)


if __name__ == "__main__":
    main()
