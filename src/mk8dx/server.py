"""FastAPI + WebSocket サーバー（設計書 6 / 7）。

配信イベント: STATE_UPDATE（全体状態）, ALERT（要手動確認の警告）
管理コマンド: CORRECT_SCORE, RENAME_TEAM, SET_RACE, RESET_MATCH

開発・検証用に POST /api/simulate_result を用意している。カメラやOCRなしで
「名前12名＋スコア」を投入し、集計→配信までを通して確認できる。
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from . import db
from .capture import VirtualCamera, grab_preview, list_cameras
from .clustering import assign_teams
from .ocr import OcrEngine
from .preprocess import load_roi_config
from .processor import RaceProcessor
from .roster import Observation, Roster
from .runner import LiveRunner, save_outcome
from .screenai import create_ocr_engine
from .scoring import validate_scores
from .state import MatchState

BASE_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = BASE_DIR / "web"
DB_PATH = BASE_DIR / "data" / "mk8dx.db"

app = FastAPI(title="MK8DX Auto Scorer")

_conn = None
_match: MatchState | None = None
_session_id: str | None = None
_clients: set[WebSocket] = set()
_lock = asyncio.Lock()

# 選手名簿はセッションをまたいで保持する（選手登録はマッチごとではなく
# グループ単位の情報のため）。プロセッサは形式が変わったら作り直す。
_roster = Roster()


def _persist_roster() -> None:
    """選手名簿をDBへ保存する（呼び出しスレッドごとに接続を作る）。"""
    try:
        conn = db.connect(DB_PATH)
        db.save_roster_data(conn, _roster.to_dict())
        conn.close()
    except Exception:
        pass  # 保存に失敗しても処理自体は止めない


def _load_persisted_roster() -> None:
    """前回までの選手名簿（自動構築の結果を含む）を復元する。"""
    global _roster
    try:
        if DB_PATH.exists():
            conn = db.connect(DB_PATH)
            data = db.load_roster_data(conn)
            conn.close()
            if data:
                _roster = Roster.from_dict(data)
    except Exception:
        pass  # 復元に失敗したら空の名簿から始める


_load_persisted_roster()
_processor: RaceProcessor | None = None
_processor_format: str | None = None

# 要確認レース（simulate_race 由来）。ライブ由来のものは runner.pending が正。
_pending_review: dict | None = None

# ライブランナー（別スレッドで駆動する）
_runner: LiveRunner | None = None
_live_thread: threading.Thread | None = None
_live_stop = threading.Event()
_live_status: dict = {"running": False, "error": None}


def _get_processor(fmt: str) -> RaceProcessor:
    global _processor, _processor_format
    if _processor is None or _processor_format != fmt:
        _processor = RaceProcessor(_roster, fmt, assume_closed=True)
        _processor_format = fmt
    return _processor


def _current_pending() -> dict | None:
    if _runner is not None and _runner.pending is not None:
        payload = _runner.pending_payload()
        if payload is not None:
            return {**payload, "source": "live"}
    if _pending_review is not None:
        return {**_pending_review, "source": "simulate"}
    return None


def _db():
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = db.connect(DB_PATH)
    return _conn


def state_payload() -> dict:
    conn = _db()
    standings = db.standings(conn, _session_id) if _session_id else []
    return {
        "type": "STATE_UPDATE",
        "match": _match.snapshot() if _match else None,
        "standings": standings,
        "pending": _current_pending(),
        "live": _live_status,
    }


async def broadcast(payload: dict) -> None:
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


async def broadcast_state() -> None:
    await broadcast(state_payload())


async def alert(message: str, detail: dict | None = None) -> None:
    await broadcast({"type": "ALERT", "message": message, "detail": detail or {}})


def reset_match(fmt: str, total_races: int = 12) -> None:
    global _match, _session_id
    conn = _db()
    _session_id = db.create_session(conn, fmt, total_races)
    _match = MatchState(fmt, total_races)


class SimulateResult(BaseModel):
    names: list[str]          # 順位順に12名
    scores: list[int]         # 順位順に12個（OCR読み取り値相当）
    race_number: int | None = None


@app.post("/api/session")
async def start_session(body: dict):
    if _live_status["running"]:
        return {"error": "ライブ実行中はセッションを開始できません。先にライブを停止してください"}
    reset_match(body.get("format", "2v2"), int(body.get("total_races", 12)))
    await broadcast_state()
    return {"session_id": _session_id}


@app.post("/api/simulate_result")
async def simulate_result(body: SimulateResult):
    """カメラなし検証用: 読み取り結果を投入して集計・保存・配信する。"""
    global _match
    if _match is None or _session_id is None:
        reset_match("2v2")
    assert _match is not None
    conn = _db()
    race_number = body.race_number or (_match.current_race + 1)
    scores_by_rank = {i + 1: s for i, s in enumerate(body.scores)}
    validation = validate_scores(scores_by_rank)
    assignment = assign_teams(body.names, _match.format)

    team_id_by_index: dict[int, int] = {}
    if assignment:
        for team in assignment.teams:
            team_id = db.get_or_create_team(
                conn, _session_id, team.tag, team.normalized_tag
            )
            for idx in team.members:
                team_id_by_index[idx] = team_id

    entries = [
        {
            "rank": i + 1,
            "player_name": body.names[i],
            "team_id": team_id_by_index.get(i),
            "score": body.scores[i],
            "raw_ocr_text": None,
        }
        for i in range(12)
    ]
    ok = validation.ok and assignment.valid
    db.save_race_result(conn, _session_id, race_number, entries, is_valid=True)
    _match.on_result_candidate()
    _match.on_screen_stable()
    _match.on_processing_done(valid=True)
    if not ok:
        problems = validation.problems + assignment.warnings
        await alert("検証で問題が検出されました", {"problems": problems})
    await broadcast_state()
    return {"ok": ok, "validation": validation.problems, "warnings": assignment.warnings}


def roster_payload() -> dict:
    return {
        "entries": [
            {
                "id": e.id,
                "name": e.canonical_name,
                "confirmed": e.confirmed,
                "capacity": e.capacity,
                "races_seen": e.races_seen,
                "has_flag": bool(e.flag_signatures),
            }
            for e in _roster.entries
        ]
    }


@app.get("/api/roster")
async def get_roster():
    return roster_payload()


@app.post("/api/roster")
async def seed_roster(body: dict):
    """選手名簿へ選手を事前登録する（ステップ3の結論: 事前登録が本命）。"""
    name = (body.get("name") or "").strip()
    if not name:
        return {"error": "name が空です"}
    entry = _roster.seed(name)
    _persist_roster()
    return {"id": entry.id, **roster_payload()}


@app.delete("/api/roster/{entry_id}")
async def delete_roster(entry_id: int):
    removed = _roster.remove(entry_id)
    _persist_roster()
    return {"removed": removed, **roster_payload()}


@app.post("/api/roster/{entry_id}/rename")
async def rename_roster(entry_id: int, body: dict):
    """自動登録された選手の名前を正しい表記に直す。"""
    name = (body.get("name") or "").strip()
    renamed = _roster.rename(entry_id, name)
    if renamed:
        _persist_roster()
    return {"renamed": renamed, **roster_payload()}


class SimulateRace(BaseModel):
    names: list[str]          # 順位順に12名（OCR読み相当の生の名前）
    race_number: int | None = None


@app.post("/api/simulate_race")
async def simulate_race(body: SimulateRace):
    """カメラなし検証用（選手名簿照合あり）: 生の名前12名を投入し、
    名寄せ→チーム判定→集計を通す。確定できないレースは
    AWAITING_REVIEW として警告し、結果は is_valid=False で保存する。"""
    global _match, _pending_review
    if _match is None or _session_id is None:
        reset_match("2v2")
    assert _match is not None
    conn = _db()
    race_number = body.race_number or (_match.current_race + 1)
    processor = _get_processor(_match.format)
    outcome = processor.process_single(
        [Observation(name=n) for n in body.names], race_number
    )
    race_result_id = save_outcome(conn, _session_id, outcome)
    _persist_roster()
    _match.on_result_candidate()
    _match.on_screen_stable()
    _match.on_processing_done(valid=outcome.confirmed)
    if not outcome.confirmed:
        _pending_review = {
            "race_result_id": race_result_id,
            "race_number": race_number,
            "problems": outcome.problems,
            "entries": outcome.entries,
            "teams": outcome.teams,
        }
        await alert(
            "選手名簿との照合で要確認のレースがあります",
            {"problems": outcome.problems, "race_number": race_number},
        )
    await broadcast_state()
    return {
        "status": outcome.status,
        "problems": outcome.problems,
        "warnings": outcome.warnings,
        "teams": outcome.teams,
        "entries": outcome.entries,
    }


@app.post("/api/review/approve")
async def review_approve():
    """要確認レースを承認する（DBを有効化し、状態機械を進める）。"""
    global _pending_review
    if _runner is not None and _runner.pending is not None:
        _runner.approve_pending()
    elif _pending_review is not None and _match is not None:
        db.approve_race_result(_db(), _pending_review["race_result_id"])
        _match.approve_pending()
        _pending_review = None
    else:
        return {"ok": False, "error": "要確認のレースがありません"}
    await broadcast_state()
    return {"ok": True}


@app.post("/api/review/discard")
async def review_discard():
    """要確認レースを破棄する（DBから削除し、再検出に戻る）。"""
    global _pending_review
    if _runner is not None and _runner.pending is not None:
        _runner.discard_pending()
    elif _pending_review is not None and _match is not None:
        db.delete_race_result(_db(), _pending_review["race_result_id"])
        _match.discard_pending()
        _pending_review = None
    else:
        return {"ok": False, "error": "要確認のレースがありません"}
    await broadcast_state()
    return {"ok": True}


def _on_runner_event(loop: asyncio.AbstractEventLoop):
    def callback(kind: str, payload: dict) -> None:
        if kind == "alert":
            asyncio.run_coroutine_threadsafe(
                alert(payload.get("message", ""), payload), loop
            )
        _persist_roster()  # レース確定のたびに名簿の学習結果を保存する
        asyncio.run_coroutine_threadsafe(broadcast_state(), loop)

    return callback


@app.post("/api/live/start")
async def live_start(body: dict):
    """ライブランナーを別スレッドで開始する（OBS仮想カメラ入力）。"""
    global _runner, _live_thread
    if _live_status["running"]:
        return {"ok": False, "error": "すでに実行中です"}
    if _match is None or _session_id is None:
        reset_match(body.get("format", "2v2"), int(body.get("total_races", 12)))
    assert _match is not None and _session_id is not None
    device_index = int(body.get("device_index", 0))
    loop = asyncio.get_running_loop()
    _live_stop.clear()
    _live_status.update({"running": True, "error": None})

    def run() -> None:
        global _runner
        conn = db.connect(DB_PATH)
        try:
            ocr = create_ocr_engine()
            _live_status["ocr_engine"] = (
                "screenai" if ocr.using_screenai else "paddle"
            )
            roi = load_roi_config(str(BASE_DIR / "config" / "roi_1080p.json"))
            # 本登録の名簿がある場合は closed（消去法あり）、無い場合は
            # open で名簿を自動構築する（登録なし運用）
            has_confirmed = any(e.confirmed for e in _roster.entries)
            processor = RaceProcessor(
                _roster, _match.format, assume_closed=has_confirmed  # type: ignore[union-attr]
            )
            runner = LiveRunner(
                roi=roi,
                ocr=ocr,
                processor=processor,
                match=_match,  # type: ignore[arg-type]
                conn=conn,
                session_id=_session_id,  # type: ignore[arg-type]
                on_event=_on_runner_event(loop),
            )
            _runner = runner
            cam = VirtualCamera(device_index)
            cam.open()
            try:
                for frame in cam.frames_at_interval(0.4):
                    if _live_stop.is_set():
                        break
                    runner.step(frame, time.monotonic())
            finally:
                cam.release()
        except Exception as exc:  # カメラが開けない等はステータスに残す
            _live_status["error"] = str(exc)
        finally:
            _live_status["running"] = False
            asyncio.run_coroutine_threadsafe(broadcast_state(), loop)

    _live_thread = threading.Thread(target=run, daemon=True)
    _live_thread.start()
    return {"ok": True}


@app.post("/api/live/stop")
async def live_stop():
    _live_stop.set()
    return {"ok": True}


@app.get("/api/cameras")
async def cameras():
    """利用可能なカメラの一覧（番号と解像度）。

    OpenCV ではデバイス名が取れないため、識別はプレビュー画像で
    行う。ライブ実行中はカメラを掴んでいるため列挙できない。
    """
    if _live_status["running"]:
        return JSONResponse(
            {"error": "ライブ実行中はカメラ一覧を取得できません"},
            status_code=409,
        )
    found = await asyncio.to_thread(list_cameras)
    return {"cameras": found}


@app.get("/api/cameras/{index}/preview")
async def camera_preview(index: int):
    """カメラのプレビュー画像（JPEG）を返す。"""
    if _live_status["running"]:
        return JSONResponse(
            {"error": "ライブ実行中はプレビューを取得できません"},
            status_code=409,
        )
    data = await asyncio.to_thread(grab_preview, index)
    if data is None:
        return JSONResponse(
            {"error": f"カメラ {index} を開けませんでした"}, status_code=404
        )
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/live/status")
async def live_status():
    return _live_status


@app.get("/overlay")
async def overlay():
    return FileResponse(WEB_DIR / "overlay.html")


@app.get("/admin")
async def admin():
    return FileResponse(WEB_DIR / "admin.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    await ws.send_text(json.dumps(state_payload(), ensure_ascii=False))
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            await handle_command(msg)
    except WebSocketDisconnect:
        _clients.discard(ws)


async def handle_command(msg: dict) -> None:
    global _match
    cmd = msg.get("type")
    if cmd == "RESET_MATCH":
        reset_match(msg.get("format", "2v2"), int(msg.get("total_races", 12)))
        await broadcast_state()
    elif cmd == "SET_RACE":
        if _match is not None:
            _match.set_race(int(msg.get("completed", 0)))
            await broadcast_state()
    elif cmd in ("CORRECT_SCORE", "RENAME_TEAM"):
        # DBへの手動修正は管理画面の実装と合わせて拡張する（骨格段階では状態を再配信）
        await alert(f"{cmd} を受け付けました（手動修正の保存は今後実装）", msg)
        await broadcast_state()
