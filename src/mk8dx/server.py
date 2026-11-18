"""FastAPI + WebSocket サーバー（設計書 6 / 7）。

配信イベント: STATE_UPDATE（全体状態）, ALERT（要手動確認の警告）
管理コマンド: CORRECT_SCORE, RENAME_TEAM, SET_RACE, RESET_MATCH

開発・検証用に POST /api/simulate_result を用意している。カメラやOCRなしで
「名前12名＋スコア」を投入し、集計→配信までを通して確認できる。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import db
from .clustering import assign_teams
from .processor import RaceProcessor
from .roster import Observation, Roster
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

# ロスターはセッションをまたいで保持する（選手登録はマッチごとではなく
# グループ単位の情報のため）。プロセッサは形式が変わったら作り直す。
_roster = Roster()
_processor: RaceProcessor | None = None
_processor_format: str | None = None


def _get_processor(fmt: str) -> RaceProcessor:
    global _processor, _processor_format
    if _processor is None or _processor_format != fmt:
        _processor = RaceProcessor(_roster, fmt, assume_closed=True)
        _processor_format = fmt
    return _processor


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
    """ロスターへ選手を事前登録する（ステップ3の結論: 事前登録が本命）。"""
    name = (body.get("name") or "").strip()
    if not name:
        return {"error": "name が空です"}
    entry = _roster.seed(name)
    return {"id": entry.id, **roster_payload()}


@app.delete("/api/roster/{entry_id}")
async def delete_roster(entry_id: int):
    removed = _roster.remove(entry_id)
    return {"removed": removed, **roster_payload()}


class SimulateRace(BaseModel):
    names: list[str]          # 順位順に12名（OCR読み相当の生の名前）
    race_number: int | None = None


@app.post("/api/simulate_race")
async def simulate_race(body: SimulateRace):
    """カメラなし検証用（ロスター照合あり）: 生の名前12名を投入し、
    名寄せ→チーム判定→集計を通す。確定できないレースは
    AWAITING_REVIEW として警告し、結果は is_valid=False で保存する。"""
    global _match
    if _match is None or _session_id is None:
        reset_match("2v2")
    assert _match is not None
    conn = _db()
    race_number = body.race_number or (_match.current_race + 1)
    processor = _get_processor(_match.format)
    outcome = processor.process_single(
        [Observation(name=n) for n in body.names], race_number
    )

    team_id_by_index: dict[int, int] = {}
    if outcome.assignment is not None:
        for team in outcome.assignment.teams:
            team_id = db.get_or_create_team(
                conn, _session_id, team.tag, team.normalized_tag
            )
            for idx in team.members:
                team_id_by_index[idx] = team_id
    entries = [
        {
            "rank": e["rank"],
            "player_name": e["resolved_name"] or e["raw_name"],
            "team_id": team_id_by_index.get(e["rank"] - 1),
            "score": e["score"],
            "raw_ocr_text": e["raw_name"],
        }
        for e in outcome.entries
    ]
    db.save_race_result(
        conn, _session_id, race_number, entries, is_valid=outcome.confirmed
    )
    _match.on_result_candidate()
    _match.on_screen_stable()
    _match.on_processing_done(valid=outcome.confirmed)
    if not outcome.confirmed:
        await alert(
            "ロスター照合で要確認のレースがあります",
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
