"""FastAPI + WebSocket サーバー。

- 管理画面（/admin）と OBS オーバーレイ（/overlay）の2画面を配信する。
- 状態はすべて WebSocket（/ws）で全クライアントへ送る（STATE_UPDATE）。
- 管理画面からの操作は REST（/api/...）で受ける。

試合の状態（形式・完了レース数）はDBから復元できるので、アプリを
再起動しても直前の試合を続けられる。
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from . import db, standings
from .capture import VirtualCamera, grab_preview, list_cameras
from .clustering import FORMATS
from .namer import NameReader
from .paths import data_dir, resource_path
from .preprocess import load_roi_config
from .processor import RaceProcessor
from .roster import Roster
from .runner import LiveRunner
from .state import MatchState

WEB_DIR = resource_path("web")
ROI_PATH = resource_path("config/roi_1080p.json")
DB_PATH = data_dir() / "mk8dx.db"

app = FastAPI(title="MK8DX Auto Scorer")

_conn = None
_match: MatchState | None = None
_session_id: str | None = None
_clients: set[WebSocket] = set()
# 選手名簿はセッションをまたいで保持する（同じメンバーで何試合もやるため）
_roster = Roster()
# ライブ処理スレッドと管理画面の操作が選手名簿・状態を同時に触らないための鍵
_lock = threading.RLock()

_runner: LiveRunner | None = None
_live_thread: threading.Thread | None = None
_live_stop = threading.Event()
_live_status: dict = {"running": False, "error": None, "name_ocr": None}


# -- 永続化・復元 ------------------------------------------------------
def _persist_roster() -> None:
    try:
        conn = db.connect(DB_PATH)
        db.save_roster_data(conn, _roster.to_dict())
        conn.close()
    except Exception:
        pass  # 保存に失敗しても処理自体は止めない


def _restore() -> None:
    """前回の選手名簿と試合（形式・完了レース数）を復元する。"""
    global _roster, _match, _session_id
    try:
        conn = db.connect(DB_PATH)
        data = db.load_roster_data(conn)
        if data:
            _roster = Roster.from_dict(data)
        latest = db.latest_session(conn)
        conn.close()
        if latest:
            _session_id = latest["id"]
            _match = MatchState(latest["format"], latest["total_races"])
            _match.set_race(latest["completed"])
    except Exception:
        pass  # 復元に失敗したら新しい試合から始める


_restore()


def _db():
    """イベントループ（メインスレッド）専用のDB接続。"""
    global _conn
    if _conn is None:
        _conn = db.connect(DB_PATH)
    return _conn


def _new_match(fmt: str, total_races: int = 12) -> None:
    global _match, _session_id, _runner
    _session_id = db.create_session(_db(), fmt, total_races)
    _match = MatchState(fmt, total_races)
    _runner = None


# -- 状態の組み立て ----------------------------------------------------
def _roster_payload(in_match: set[int]) -> list[dict]:
    return [
        {
            "id": e.id,
            "name": e.canonical_name,
            "team": e.team,
            "team_auto": bool(e.team) and not e.team_manual,
            "races_seen": e.races_seen,
            "has_image": bool(e.thumb or e.masks),
            "image_v": len(e.thumb) if e.thumb else 0,
            "in_match": e.id in in_match,
        }
        for e in _roster.entries
    ]


_last_payload: dict | None = None


def state_payload() -> dict:
    """現在の状態。ライブ処理が名簿を更新中（名前の文字認識で数秒かかることが
    ある）に画面が固まらないよう、鍵が取れなければ直前の状態を返す。"""
    global _last_payload
    if not _lock.acquire(timeout=0.3):
        if _last_payload is not None:
            return _last_payload
        _lock.acquire()
    try:
        _last_payload = _build_state()
        return _last_payload
    finally:
        _lock.release()


def _build_state() -> dict:
    rows = db.valid_rows(_db(), _session_id) if _session_id else []
    pending = None
    if _runner is not None and _runner.session_id == _session_id:
        pending = _runner.pending_payload()
    in_match = {r["entry_id"] for r in rows if r.get("entry_id")}
    if pending:
        in_match |= {
            e["entry_id"] for e in pending["entries"] if e.get("entry_id")
        }
    team_names: list[str] = []
    for e in _roster.entries:
        if e.team and e.team not in team_names:
            team_names.append(e.team)
    return {
        "type": "STATE_UPDATE",
        "match": _match.snapshot() if _match else None,
        "standings": standings.compute(rows, _roster),
        "pending": pending,
        "live": dict(_live_status),
        "roster": _roster_payload(in_match),
        "team_names": team_names,
        "can_undo": bool(rows) and pending is None,
    }


async def broadcast_state() -> None:
    text = json.dumps(state_payload(), ensure_ascii=False)
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


def _ok(**extra) -> dict:
    return {"ok": True, **extra}


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status)


# -- 試合 -------------------------------------------------------------
@app.post("/api/session")
async def start_session(body: dict):
    """新しい試合を始める。選手名簿は残す（reset_roster=true で消す）。"""
    if _live_status["running"]:
        return _error("集計中は新しい試合を始められません。先に「集計をとめる」を押してください")
    fmt = body.get("format", "2v2")
    if fmt not in FORMATS:
        return _error("対戦形式が正しくありません")
    with _lock:
        _new_match(fmt, int(body.get("total_races", 12)))
        if body.get("reset_roster"):
            _roster.entries.clear()
            _roster.team_colors.clear()
            _persist_roster()
    await broadcast_state()
    return _ok()


@app.post("/api/race/undo")
async def undo_race():
    """直前に確定したレースを取り消す。"""
    with _lock:
        if _match is None or _session_id is None:
            return _error("試合が始まっていません")
        if _runner is not None and _runner.session_id == _session_id:
            if not _runner.undo_last_race():
                return _error("取り消せるレースがありません")
        else:
            last = db.last_valid_race_result(_db(), _session_id)
            if last is None:
                return _error("取り消せるレースがありません")
            db.delete_race_result(_db(), last)
            _match.undo_last_race()
    await broadcast_state()
    return _ok()


# -- 要確認レース -------------------------------------------------------
@app.post("/api/review/approve")
async def review_approve():
    with _lock:
        if _runner is None or _runner.pending is None:
            return _error("確認待ちのレースがありません")
        todo = _runner.pending_problems()
        if todo:
            return _error("、".join(todo))
        _runner.approve_pending()
        _persist_roster()
    await broadcast_state()
    return _ok()


@app.post("/api/review/discard")
async def review_discard():
    with _lock:
        if _runner is None or _runner.pending is None:
            return _error("確認待ちのレースがありません")
        _runner.discard_pending()
    await broadcast_state()
    return _ok()


# -- 選手名簿 ----------------------------------------------------------
@app.get("/api/roster/{entry_id}/image")
async def roster_image(entry_id: int):
    """選手の名前の文字画像（誰のことか見分けるための表示用）。"""
    with _lock:
        found = _roster.image_for(entry_id)
    if found is None:
        return Response(status_code=404)
    data, mime = found
    return Response(content=data, media_type=mime,
                    headers={"Cache-Control": "max-age=600"})


@app.post("/api/roster/{entry_id}/rename")
async def rename_player(entry_id: int, body: dict):
    with _lock:
        if not _roster.rename(entry_id, (body.get("name") or "").strip()):
            return _error("名前を変更できませんでした")
        _persist_roster()
    await broadcast_state()
    return _ok()


@app.post("/api/roster/{entry_id}/team")
async def set_player_team(entry_id: int, body: dict):
    with _lock:
        if not _roster.set_team(entry_id, body.get("team")):
            return _error("チームを設定できませんでした")
        _persist_roster()
    await broadcast_state()
    return _ok()


@app.post("/api/roster/{entry_id}/merge")
async def merge_player(entry_id: int, body: dict):
    """この選手を、別の選手と同じ人として統合する（過去のレースも付け替える）。"""
    into = int(body.get("into", -1))
    with _lock:
        if not _roster.merge(entry_id, into):
            return _error("統合できませんでした")
        _db().execute(
            "UPDATE player_scores SET entry_id = ? WHERE entry_id = ?",
            (into, entry_id))
        _db().commit()
        if _runner is not None and entry_id in _runner._totals_by_id:
            _runner._totals_by_id[into] = (
                _runner._totals_by_id.get(into, 0)
                + _runner._totals_by_id.pop(entry_id))
        _persist_roster()
    await broadcast_state()
    return _ok()


@app.delete("/api/roster/{entry_id}")
async def delete_player(entry_id: int):
    with _lock:
        if not _roster.remove(entry_id):
            return _error("削除できませんでした")
        _persist_roster()
    await broadcast_state()
    return _ok()


# -- 集計（ライブ） ----------------------------------------------------
def _on_runner_event(loop: asyncio.AbstractEventLoop):
    def callback(kind: str, payload: dict) -> None:
        _persist_roster()  # レース確定のたびに名簿の学習結果を保存する
        asyncio.run_coroutine_threadsafe(broadcast_state(), loop)

    return callback


def _friendly_error(exc: Exception, device_index: int) -> str:
    if isinstance(exc, RuntimeError) and "開けません" in str(exc):
        return (
            f"カメラ（番号 {device_index}）を開けませんでした。"
            "OBSで「仮想カメラ開始」を押してから、映像を選び直してください。"
        )
    return f"予期しないエラーが起きました: {exc}"


@app.post("/api/live/start")
async def live_start(body: dict):
    global _runner, _live_thread
    if _live_status["running"]:
        return _error("すでに集計中です")
    with _lock:
        if _match is None or _session_id is None:
            _new_match(body.get("format", "2v2"))
    assert _match is not None and _session_id is not None
    device_index = int(body.get("device_index", 0))
    loop = asyncio.get_running_loop()
    _live_stop.clear()
    _live_status.update({"running": True, "error": None})
    session_id, match = _session_id, _match

    def run() -> None:
        global _runner
        conn = db.connect(DB_PATH)
        try:
            namer = NameReader()
            _live_status["name_ocr"] = "rapidocr" if namer.available else "none"
            roi = load_roi_config(str(ROI_PATH))
            # 確定した選手がいれば「全員そろっている」前提で消去法も使い、
            # いなければ名簿を自動構築する
            has_confirmed = any(e.confirmed for e in _roster.entries)
            processor = RaceProcessor(
                _roster, match.format, assume_closed=has_confirmed,
                namer=namer.read if namer.available else None,
            )
            runner = LiveRunner(
                roi=roi, ocr=None, processor=processor, match=match,
                conn=conn, session_id=session_id,
                on_event=_on_runner_event(loop),
                screenshot_dir=data_dir() / "screenshots",
            )
            _runner = runner
            cam = VirtualCamera(device_index)
            cam.open()
            asyncio.run_coroutine_threadsafe(broadcast_state(), loop)
            try:
                for frame in cam.frames_at_interval(0.4):
                    if _live_stop.is_set():
                        break
                    with _lock:
                        runner.step(frame, time.monotonic())
            finally:
                cam.release()
        except Exception as exc:
            _live_status["error"] = _friendly_error(exc, device_index)
        finally:
            _live_status["running"] = False
            conn.close()
            asyncio.run_coroutine_threadsafe(broadcast_state(), loop)

    _live_thread = threading.Thread(target=run, daemon=True)
    _live_thread.start()
    await broadcast_state()
    return _ok()


@app.post("/api/live/stop")
async def live_stop():
    _live_stop.set()
    return _ok()


@app.get("/api/cameras")
async def cameras():
    """利用できる映像（カメラ）の一覧。集計中はカメラを使用中のため取得できない。"""
    if _live_status["running"]:
        return JSONResponse(
            {"error": "集計中は映像の一覧を更新できません"}, status_code=409)
    return {"cameras": await asyncio.to_thread(list_cameras)}


@app.get("/api/cameras/{index}/preview")
async def camera_preview(index: int):
    if _live_status["running"]:
        return Response(status_code=409)
    data = await asyncio.to_thread(grab_preview, index)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


# -- 画面・WebSocket -----------------------------------------------------
@app.get("/")
async def index():
    return RedirectResponse("/admin")


@app.get("/overlay")
async def overlay():
    return FileResponse(WEB_DIR / "overlay.html")


@app.get("/admin")
async def admin():
    return FileResponse(WEB_DIR / "admin.html")


@app.get("/api/state")
async def get_state():
    return state_payload()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    await ws.send_text(json.dumps(state_payload(), ensure_ascii=False))
    try:
        while True:
            await ws.receive_text()  # クライアントからの送信は使わない（切断検知のみ）
    except WebSocketDisconnect:
        _clients.discard(ws)
