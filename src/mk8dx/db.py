"""SQLite データ層（設計書 5.3 のスキーマ）。"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    format TEXT NOT NULL,
    total_races INTEGER DEFAULT 12,
    current_race INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    tag_name TEXT NOT NULL,
    normalized_tag TEXT NOT NULL,
    color_code TEXT
);
CREATE TABLE IF NOT EXISTS race_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    race_number INTEGER NOT NULL,
    is_valid BOOLEAN DEFAULT TRUE,
    screenshot_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS player_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_result_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    team_id INTEGER,
    score INTEGER NOT NULL,
    raw_ocr_text TEXT
);
"""

# チームカラー（オーバーレイのカラーバー用、最大6チーム）
TEAM_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def create_session(conn: sqlite3.Connection, fmt: str, total_races: int = 12) -> str:
    session_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO sessions (id, format, total_races) VALUES (?, ?, ?)",
        (session_id, fmt, total_races),
    )
    conn.commit()
    return session_id


def get_or_create_team(
    conn: sqlite3.Connection, session_id: str, tag: str, normalized_tag: str
) -> int:
    row = conn.execute(
        "SELECT id FROM teams WHERE session_id = ? AND normalized_tag = ?",
        (session_id, normalized_tag),
    ).fetchone()
    if row:
        return int(row["id"])
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM teams WHERE session_id = ?", (session_id,)
    ).fetchone()["c"]
    color = TEAM_COLORS[count % len(TEAM_COLORS)]
    cur = conn.execute(
        "INSERT INTO teams (session_id, tag_name, normalized_tag, color_code)"
        " VALUES (?, ?, ?, ?)",
        (session_id, tag, normalized_tag, color),
    )
    conn.commit()
    return int(cur.lastrowid)


def save_race_result(
    conn: sqlite3.Connection,
    session_id: str,
    race_number: int,
    entries: list[dict],
    is_valid: bool = True,
    screenshot_path: str | None = None,
) -> int:
    """entries: [{rank, player_name, team_id, score, raw_ocr_text}, ...]"""
    cur = conn.execute(
        "INSERT INTO race_results (session_id, race_number, is_valid, screenshot_path)"
        " VALUES (?, ?, ?, ?)",
        (session_id, race_number, int(is_valid), screenshot_path),
    )
    race_result_id = int(cur.lastrowid)
    conn.executemany(
        "INSERT INTO player_scores"
        " (race_result_id, rank, player_name, team_id, score, raw_ocr_text)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                race_result_id,
                e["rank"],
                e["player_name"],
                e.get("team_id"),
                e["score"],
                e.get("raw_ocr_text"),
            )
            for e in entries
        ],
    )
    conn.execute(
        "UPDATE sessions SET current_race = ? WHERE id = ?",
        (race_number, session_id),
    )
    conn.commit()
    return race_result_id


def approve_race_result(conn: sqlite3.Connection, race_result_id: int) -> None:
    """要確認（is_valid=0）で保存したレースを承認して有効化する。"""
    conn.execute(
        "UPDATE race_results SET is_valid = 1 WHERE id = ?", (race_result_id,)
    )
    conn.commit()


def delete_race_result(conn: sqlite3.Connection, race_result_id: int) -> None:
    """レース結果とその選手スコアを削除する（要確認レースの破棄用）。"""
    conn.execute(
        "DELETE FROM player_scores WHERE race_result_id = ?", (race_result_id,)
    )
    conn.execute("DELETE FROM race_results WHERE id = ?", (race_result_id,))
    conn.commit()


def race_result_entries(conn: sqlite3.Connection, race_result_id: int) -> list[dict]:
    """保存済みレースの選手行を順位順に返す（管理画面の確認用）。"""
    rows = conn.execute(
        "SELECT rank, player_name, team_id, score, raw_ocr_text"
        " FROM player_scores WHERE race_result_id = ? ORDER BY rank",
        (race_result_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def standings(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """チーム別 standings: 合計点・レース別獲得点・1位との差を付けて返す。"""
    rows = conn.execute(
        """
        SELECT t.id AS team_id, t.tag_name, t.color_code,
               rr.race_number AS race_number, SUM(ps.score) AS race_points
        FROM player_scores ps
        JOIN race_results rr ON rr.id = ps.race_result_id
        JOIN teams t ON t.id = ps.team_id
        WHERE rr.session_id = ? AND rr.is_valid = 1 AND ps.team_id IS NOT NULL
        GROUP BY t.id, rr.race_number
        ORDER BY rr.race_number
        """,
        (session_id,),
    ).fetchall()
    teams: dict[int, dict] = {}
    for r in rows:
        team = teams.setdefault(
            r["team_id"],
            {
                "team_id": r["team_id"],
                "tag": r["tag_name"],
                "color": r["color_code"],
                "total": 0,
                "per_race": {},
                "last_race_points": 0,
            },
        )
        team["per_race"][r["race_number"]] = r["race_points"]
        team["total"] += r["race_points"]
        team["last_race_points"] = r["race_points"]
    result = sorted(teams.values(), key=lambda t: -t["total"])
    if result:
        top = result[0]["total"]
        for i, team in enumerate(result):
            team["rank"] = i + 1
            team["diff_from_top"] = team["total"] - top
    return result
