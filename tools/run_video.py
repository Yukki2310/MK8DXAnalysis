#!/usr/bin/env python3
"""動画ファイルをライブランナーに流して、検知〜集計を通しで確認する。

使い方: python tools/run_video.py clip1.mp4 [clip2.mp4 ...] [--format 2v2] [--interval 0.4]

複数の動画は順に同じセッション・同じ選手名簿へ流す（レースが続く想定）。
カメラの代わりに動画フレームを検知周期で取り出し、時刻は動画時間で仮想化する。
データは一時フォルダのDBに保存され、終了時に順位表と選手名簿を表示する。
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from mk8dx import db, standings  # noqa: E402
from mk8dx.namer import NameReader  # noqa: E402
from mk8dx.preprocess import load_roi_config  # noqa: E402
from mk8dx.processor import RaceProcessor  # noqa: E402
from mk8dx.roster import Roster  # noqa: E402
from mk8dx.runner import LiveRunner  # noqa: E402
from mk8dx.state import MatchState  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--format", default="2v2")
    ap.add_argument("--interval", type=float, default=0.4)
    ap.add_argument("--auto-approve", action="store_true",
                    help="要確認レースを自動承認して先へ進む（検証用）")
    args = ap.parse_args()

    roi = load_roi_config(ROOT / "config" / "roi_1080p.json")
    roster = Roster()
    namer = NameReader()
    print("名前OCR:", namer.engine_name)
    tmp = tempfile.mkdtemp()
    conn = db.connect(Path(tmp) / "run.db")
    session_id = db.create_session(conn, args.format, 12)
    match = MatchState(args.format, 12)
    events: list[tuple[str, dict]] = []
    runner = LiveRunner(
        roi=roi, ocr=None,
        processor=RaceProcessor(
            roster, args.format, namer=namer if namer.available else None),
        match=match, conn=conn, session_id=session_id,
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    clock = 0.0
    for video in args.videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        step = max(1, int(round(fps * args.interval)))
        index = 0
        last_phase = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                phase = runner.step(frame, clock)
                if runner.pending is not None and args.auto_approve:
                    print(f"  (要確認レースを自動承認: {runner.pending.outcome.problems})")
                    runner.approve_pending(clock)
                    phase = runner.match.phase.value
                if phase != last_phase:
                    print(f"[{video.name} t={index / fps:6.1f}s] {phase}")
                    last_phase = phase
                clock += args.interval
            index += 1
        cap.release()
        clock += 2.0  # 動画の切れ目（短い間隔でも次のレースを取りこぼさないことの確認）

    print("\n--- 結果 ---")
    for kind, payload in events:
        if kind == "alert":
            print("警告:", payload.get("message"), payload.get("problems", ""))
        elif kind == "cross_check":
            print("累計照合:", payload.get("problems") or "一致")
    for team in standings.compute(db.valid_rows(conn, session_id), roster):
        print(f"{team['rank']}位 {team['tag']}: {team['total']}点 {team['members']}")
    print("\n選手名簿:")
    for e in roster.entries:
        print(f"  #{e.id} {e.canonical_name!r} チーム={e.team!r} 出現{e.races_seen}回")


if __name__ == "__main__":
    main()
