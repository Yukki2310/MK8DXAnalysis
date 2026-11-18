"""ステップ6検証: 登録なし運用（選手名簿の自動構築）を実動画で検証する。

使い方:
    python tools/step6_check.py

ステップ3の自動構築検証（モードB）は「各レース1フレームの読み」が
前提だった。実運用のライブ経路は複数フレームの多数決で名前を
決めるため、同じ動画区間からフェーズAのフレームを複数枚集めて
RaceProcessor（空の名簿・open）に流し、収束の仕方を比べる。

対象はレース1（330区間）→レース3（710区間）→レース4（870区間）。
正解の名前列はステップ2・3で目視確定したもの。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mk8dx.detector import looks_like_result
from mk8dx.ocr import OcrEngine
from mk8dx.pipeline import (
    classify_phase,
    extract_observations,
    verify_result_candidate,
)
from mk8dx.preprocess import load_roi_config
from mk8dx.processor import RaceProcessor
from mk8dx.roster import Roster
from mk8dx.tags import normalize_name

ROOT = Path(__file__).resolve().parent.parent

TRUTH = {
    "330": ["xı c.qq", "RR★", "Rev◆", "ẇf ths", "racist #2",
            "A★ぜんだかいとしき", "FR", "Xb◇ねえあまぁƄ", "Xb",
            "racist #1", "FR", "xı avalon"],
    "710": ["xı c.qq", "FR", "racist #1", "Xb", "RR★", "FR Flowx",
            "A★ぜんだかいとしき", "A A", "xı avalon", "Xb◇ねえあまぁƄ",
            "RR★", "racist #2"],
    "870": ["A★ぜんだかいとしき", "racist #2", "A A", "FR Flowx",
            "Xb◇ねえあまぁƄ", "racist #1", "xı c.qq", "Xb",
            "xı avalon", "RR★", "FR", "RR★"],
}
ORDER = ["330", "710", "870"]
LABELS = {"330": "レース1", "710": "レース3", "870": "レース4"}


def collect_phase_a_frames(section: str, roi: dict, ocr: OcrEngine,
                           max_frames: int = 6) -> list:
    """区間動画からフェーズAの観測を最大 max_frames 件集める。"""
    cap = cv2.VideoCapture(str(ROOT / "data/videos" / f"lounge_{section}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, int(fps * 0.5))
    observations = []
    frame_no = 0
    while len(observations) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_no / fps
        frame_no += 1
        if frame_no % step != 0 or t < 5.0 or t > 35.0:
            continue
        if not looks_like_result(frame, roi):
            continue
        if not verify_result_candidate(frame, roi, ocr):
            continue
        if classify_phase(frame, roi, ocr) != "race_result":
            continue
        observations.append(extract_observations(frame, roi, ocr))
    cap.release()
    return observations


def main() -> None:
    roi = load_roi_config(str(ROOT / "config/roi_1080p.json"))
    ocr = OcrEngine()
    processor = RaceProcessor(Roster(), "2v2", assume_closed=False)
    for seq, section in enumerate(ORDER, start=1):
        frames = collect_phase_a_frames(section, roi, ocr)
        print(f"===== {LABELS[section]}（{section}区間、Aフレーム{len(frames)}枚） =====")
        processor.begin_race(seq)
        state = "accumulating"
        for obs in frames:
            state = processor.add_frame(obs)
            if state == "ready":
                break
        outcome = processor.finalize(force=True)
        resolved = [e["resolved_name"] for e in outcome.entries]
        truth = TRUTH[section]
        hits = sum(
            1 for got, want in zip(resolved, truth)
            if got is not None and normalize_name(got) == normalize_name(want)
        )
        print(f"   status={outcome.status} / 正解 {hits}/12")
        for i, (got, want) in enumerate(zip(resolved, truth), start=1):
            mark = "○" if got is not None and normalize_name(got) == normalize_name(want) else "×"
            print(f"   {mark} {i:2d}位: 正解={want!r} 名寄せ={got!r}")
        if outcome.teams:
            total = sum(t["points"] for t in outcome.teams)
            print(f"   チーム集計あり（合計{total}点）")
        for p in outcome.problems:
            print(f"   problem: {p}")
        entries = processor.roster.entries
        print(f"   名簿: {len(entries)}エントリ"
              f"（本登録 {sum(1 for e in entries if e.confirmed)}）")
        print("   " + ", ".join(
            f"{e.canonical_name}({'本' if e.confirmed else '仮'})" for e in entries
        ))
        print()


if __name__ == "__main__":
    main()
