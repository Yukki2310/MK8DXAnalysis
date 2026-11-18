"""ステップ5検証: ライブランナーを実動画でエンドツーエンドに回す。

使い方:
    python tools/step5_check.py

data/videos/lounge_710.mp4（レース3のフェーズA→Bを含む区間）を
0.5秒間隔でサンプリングして LiveRunner.step に流し、次を確認する。

- リザルト候補の検出→観測蓄積→確定まで自動で進むこと
- レース3のチーム集計が既知の正解（合計82点）と一致すること
- COOLDOWN 中にフェーズBが映り、照合が実行されること
  （この検証ではセッションにレース1・2の結果がないため、累計の
  不一致が検出されるのが正しい挙動。実運用では全レースを
  セッション内で処理するので一致するはず、という前提の確認）

DB は一時ファイルを使う（data/ の実DBは汚さない）。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mk8dx import db
from mk8dx.ocr import OcrEngine
from mk8dx.preprocess import load_roi_config
from mk8dx.processor import RaceProcessor
from mk8dx.roster import Roster
from mk8dx.runner import LiveRunner
from mk8dx.state import MatchState
from step3_check import CANONICAL_R34

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_R3 = {"xı": 19, "f": 19, "ra": 11, "xb": 12, "rr": 10, "a": 11}


def main() -> None:
    events: list[tuple[str, dict]] = []
    roster = Roster()
    for name in CANONICAL_R34:
        roster.seed(name)
    tmp = Path(tempfile.mkdtemp()) / "e2e.db"
    conn = db.connect(tmp)
    session_id = db.create_session(conn, "2v2", 12)
    match = MatchState("2v2", 12, cooldown_sec=20.0)
    runner = LiveRunner(
        roi=load_roi_config(str(ROOT / "config/roi_1080p.json")),
        ocr=OcrEngine(),
        processor=RaceProcessor(roster, "2v2", assume_closed=True),
        match=match,
        conn=conn,
        session_id=session_id,
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    cap = cv2.VideoCapture(str(ROOT / "data/videos/lounge_710.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    sample_every = max(1, int(fps * 0.5))
    frame_no = 0
    last_phase = None
    print("t(s)  phase")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % sample_every == 0:
            t = frame_no / fps
            phase = runner.step(frame, now=t)
            if phase != last_phase:
                print(f"{t:5.1f}  {phase}")
                last_phase = phase
            # 管理者の承認を模擬する（実運用では管理画面から行う）
            if runner.pending is not None:
                print(f"{t:5.1f}  要確認レースを承認（模擬）")
                runner.approve_pending(now=t)
        frame_no += 1
    cap.release()

    print()
    outcome = runner.last_outcome
    assert outcome is not None, "レースが1件も確定しませんでした"
    print(f"確定レース: race {outcome.race_index} status={outcome.status}")
    points = {t["normalized_tag"]: t["points"] for t in outcome.teams}
    print(f"チーム集計: {points}")
    assert points == EXPECTED_R3, f"レース3の集計が不正解です: {points}"
    assert sum(points.values()) == 82

    print()
    print(f"照合結果: {runner.cross_check_problems}")
    assert runner.cross_check_problems is not None, "フェーズB照合が実行されていません"
    assert runner.cross_check_problems, (
        "セッションにレース1・2がないため、累計の不一致が出るはずです"
    )
    print("（レース1・2を含まないセッションなので累計不一致は想定どおり）")
    kinds = [k for k, _ in events]
    print(f"イベント: {kinds}")


if __name__ == "__main__":
    main()
