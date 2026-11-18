"""runner.py（ライブランナー）のテスト。

画像処理関数はフェイクに差し替え、状態遷移・保存・照合・承認の
配線だけを検証する（実画像での検証は tools/step5_check.py）。
"""
import numpy as np
import pytest

from mk8dx import db
from mk8dx.pipeline import StandingsReading
from mk8dx.processor import RaceProcessor
from mk8dx.roster import Observation, Roster
from mk8dx.runner import LiveRunner
from mk8dx.state import MatchState, Phase

# 同名重複のない12名（照合の名前キーを一意にするため RR2 を使用）
ORDER = [
    "xı c.qq", "FR", "racist #1", "Xb", "RR★", "FR Flowx",
    "A★ぜんだかいとしき", "A A", "xı avalon", "Xb◇ねえあまぁƄ",
    "RR2", "racist #2",
]
TOTALS = [15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)


class Script:
    """フェイク関数の脚本（テストから値を切り替える）。"""

    def __init__(self, names: list[str]):
        self.like = False
        self.phase = "race_result"
        self.names = names


def make_runner(tmp_path, script: Script, *, bad_teams: bool = False) -> LiveRunner:
    roster = Roster()
    for i, name in enumerate(ORDER):
        entry = roster.seed(name)
        if bad_teams:
            # 人数が合わないチーム分け（3人・1人・8人）にして、確認が必要な状態にする
            roster.set_team(entry.id, (["A"] * 3 + ["B"] + ["C"] * 8)[i])
    conn = db.connect(tmp_path / "test.db")
    session_id = db.create_session(conn, "2v2", 12)
    match = MatchState("2v2", 12, cooldown_sec=5.0)
    runner = LiveRunner(
        roi={},
        ocr=None,
        processor=RaceProcessor(roster, "2v2", assume_closed=True),
        match=match,
        conn=conn,
        session_id=session_id,
        result_like_fn=lambda frame, roi: script.like,
        verify_fn=lambda frame, roi, ocr: True,
        observations_fn=lambda frame, roi, ocr: [
            Observation(name=n) for n in script.names
        ],
        classify_fn=lambda frame, roi, ocr: script.phase,
        standings_fn=lambda frame, roi, ocr: StandingsReading(
            entries=[
                {"position": i + 1, "name": ORDER[i], "total": TOTALS[i]}
                for i in range(12)
            ]
        ),
    )
    return runner


def test_happy_path_with_cross_check(tmp_path):
    script = Script(ORDER)
    runner = make_runner(tmp_path, script)
    assert runner.step(FRAME, now=0.0) == "WAITING_RACE"
    script.like = True
    assert runner.step(FRAME, now=0.4) == "RESULT_DETECTING"
    # 2フレーム目で割り当てが安定し、確定して COOLDOWN へ
    assert runner.step(FRAME, now=0.8) == "COOLDOWN"
    assert runner.last_outcome is not None and runner.last_outcome.confirmed
    standings = db.standings(runner.conn, runner.session_id)
    assert sum(t["total"] for t in standings) == 82
    # フェーズBが映ったら照合が走り、一致する
    script.phase = "standings"
    runner.step(FRAME, now=1.2)
    assert runner.cross_check_problems == []
    # 画面が消えてクールダウンが過ぎたら次のレース待ちへ
    script.like = False
    # 画面が連続して消えたら解除（1フレームの途切れでは解除しない）
    assert runner.step(FRAME, now=7.0) == "COOLDOWN"
    for k in range(1, 4):
        phase = runner.step(FRAME, now=7.0 + 0.4 * k)
    assert phase == "WAITING_RACE"


def test_review_flow_approve(tmp_path):
    script = Script(ORDER)
    runner = make_runner(tmp_path, script, bad_teams=True)
    script.like = True
    runner.step(FRAME, now=0.4)
    assert runner.step(FRAME, now=0.8) == "AWAITING_REVIEW"
    assert runner.pending is not None
    # 未承認の間は順位表に反映されない
    assert db.standings(runner.conn, runner.session_id) == []
    assert runner.approve_pending(now=1.0)
    assert runner.match.phase is Phase.COOLDOWN
    assert len(db.valid_rows(runner.conn, runner.session_id)) == 12


def test_review_flow_discard(tmp_path):
    script = Script(ORDER)
    runner = make_runner(tmp_path, script, bad_teams=True)
    script.like = True
    runner.step(FRAME, now=0.4)
    runner.step(FRAME, now=0.8)
    assert runner.pending is not None
    race_result_id = runner.pending.race_result_id
    assert runner.discard_pending()
    assert runner.match.phase is Phase.WAITING_RACE
    assert db.race_result_entries(runner.conn, race_result_id) == []
    assert db.standings(runner.conn, runner.session_id) == []


def test_standings_first_is_not_processed_as_race(tmp_path):
    script = Script(ORDER)
    script.phase = "standings"  # Aを見逃してBだけが映ったケース
    runner = make_runner(tmp_path, script)
    script.like = True
    runner.step(FRAME, now=0.4)
    runner.step(FRAME, now=0.8)
    # レースとして保存されず、検出状態から戻る
    assert runner.last_outcome is None
    assert runner.match.phase is Phase.WAITING_RACE
    assert db.standings(runner.conn, runner.session_id) == []


def test_undo_last_race(tmp_path):
    script = Script(ORDER)
    runner = make_runner(tmp_path, script)
    script.like = True
    runner.step(FRAME, now=0.4)
    runner.step(FRAME, now=0.8)
    assert runner.match.current_race == 1
    assert runner.undo_last_race()
    assert runner.match.current_race == 0
    assert runner.match.phase is Phase.WAITING_RACE
    assert db.valid_rows(runner.conn, runner.session_id) == []
    assert not runner.undo_last_race()  # 取り消せるものがもう無い


def test_pending_payload_reports_what_to_fix(tmp_path):
    script = Script(ORDER)
    runner = make_runner(tmp_path, script, bad_teams=True)
    script.like = True
    runner.step(FRAME, now=0.4)
    runner.step(FRAME, now=0.8)
    payload = runner.pending_payload()
    assert payload is not None
    assert all("entry_id" in e for e in payload["entries"])
    assert payload["todo"] and not payload["can_approve"]


def test_shared_connection_can_approve_from_another_thread(tmp_path):
    """ライブ処理のスレッドで保存した結果を、画面操作を受けるスレッドで確定できる。"""
    import threading

    conn = db.connect(tmp_path / "shared.db", shared=True)
    session_id = db.create_session(conn, "2v2", 12)
    result = {}

    def save():
        result["id"] = db.save_race_result(
            conn, session_id, 1,
            [{"rank": 1, "player_name": "p", "team_id": None, "score": 15,
              "entry_id": 1}], is_valid=False)

    t = threading.Thread(target=save)
    t.start()
    t.join()
    db.approve_race_result(conn, result["id"])   # 別スレッドから（以前は例外になった）
    assert len(db.valid_rows(conn, session_id)) == 1

    plain = db.connect(tmp_path / "plain.db")      # 既定は従来どおりスレッド専用
    t = threading.Thread(target=lambda: result.update(err=_try(plain)))
    t.start()
    t.join()
    assert result["err"]


def _try(conn):
    try:
        conn.execute("SELECT 1")
        return False
    except Exception:
        return True


def test_result_visible_follows_the_result_screen(tmp_path):
    script = Script(ORDER)
    runner = make_runner(tmp_path, script)
    events = []
    runner.on_event = lambda kind, payload: events.append(kind)
    assert not runner.result_visible
    script.like = True
    runner.step(FRAME, now=0.4)
    assert runner.result_visible and "visible" in events
    for k in range(1, 3):                      # 1〜2回途切れても表示中のまま
        script.like = False
        runner.step(FRAME, now=0.8 + 0.4 * k)
    assert runner.result_visible
    runner.step(FRAME, now=3.0)                # 続けて見えなくなったら解除
    assert not runner.result_visible
