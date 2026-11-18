"""レース進行の状態管理（設計書 5.2）。

状態遷移:
  WAITING_RACE --(リザルト候補検出)--> RESULT_DETECTING
  RESULT_DETECTING --(1〜2秒静止を確認)--> PROCESSING
  RESULT_DETECTING --(画面が消えた)--> WAITING_RACE
  PROCESSING --(検証OK または 手動承認)--> COOLDOWN（レース番号を歩進）
  PROCESSING --(検証NG・未承認)--> RESULT_DETECTING には戻さず管理画面の判断待ち
  COOLDOWN --(リザルト画面が消えた かつ 最低60秒経過)--> WAITING_RACE
"""
from __future__ import annotations

import enum
import time


class Phase(enum.Enum):
    WAITING_RACE = "WAITING_RACE"
    RESULT_DETECTING = "RESULT_DETECTING"
    PROCESSING = "PROCESSING"
    AWAITING_REVIEW = "AWAITING_REVIEW"  # 検証NGで管理画面の判断待ち
    COOLDOWN = "COOLDOWN"
    FINISHED = "FINISHED"


class MatchState:
    def __init__(self, fmt: str, total_races: int = 12, cooldown_sec: float = 60.0):
        self.format = fmt
        self.total_races = total_races
        self.cooldown_sec = cooldown_sec
        self.current_race = 0  # 完了したレース数
        self.phase = Phase.WAITING_RACE
        self._cooldown_started_at: float | None = None

    # --- イベント ---
    def on_result_candidate(self) -> None:
        if self.phase is Phase.WAITING_RACE:
            self.phase = Phase.RESULT_DETECTING

    def on_screen_gone(self) -> None:
        if self.phase is Phase.RESULT_DETECTING:
            self.phase = Phase.WAITING_RACE

    def on_screen_stable(self) -> None:
        if self.phase is Phase.RESULT_DETECTING:
            self.phase = Phase.PROCESSING

    def on_processing_done(self, valid: bool, now: float | None = None) -> None:
        if self.phase is not Phase.PROCESSING:
            return
        if valid:
            self._complete_race(now)
        else:
            self.phase = Phase.AWAITING_REVIEW

    def approve_pending(self, now: float | None = None) -> None:
        """検証NGのレースを管理画面で手動承認したとき。"""
        if self.phase is Phase.AWAITING_REVIEW:
            self._complete_race(now)

    def discard_pending(self) -> None:
        """検証NGのレースを破棄して再検出に戻すとき。"""
        if self.phase is Phase.AWAITING_REVIEW:
            self.phase = Phase.WAITING_RACE

    # --- 周期処理 ---
    def tick(self, screen_gone: bool, now: float | None = None) -> None:
        """COOLDOWN の解除判定。メインループから定期的に呼ぶ。"""
        if self.phase is not Phase.COOLDOWN:
            return
        now = time.monotonic() if now is None else now
        elapsed = now - (self._cooldown_started_at or now)
        if screen_gone and elapsed >= self.cooldown_sec:
            self.phase = (
                Phase.FINISHED if self.current_race >= self.total_races
                else Phase.WAITING_RACE
            )

    # --- 手動操作 ---
    def set_race(self, completed: int) -> None:
        """完了レース数を手動で変更する（管理画面の SET_RACE）。"""
        self.current_race = max(0, min(completed, self.total_races))

    def undo_last_race(self) -> None:
        """直前のレースの取り消し: 完了数を1戻し、次のレース待ちにする。"""
        self.current_race = max(0, self.current_race - 1)
        self.phase = Phase.WAITING_RACE
        self._cooldown_started_at = None

    def reset(self) -> None:
        self.current_race = 0
        self.phase = Phase.WAITING_RACE
        self._cooldown_started_at = None

    # --- 内部 ---
    def _complete_race(self, now: float | None) -> None:
        self.current_race += 1
        self._cooldown_started_at = time.monotonic() if now is None else now
        self.phase = Phase.COOLDOWN

    def snapshot(self) -> dict:
        return {
            "format": self.format,
            "total_races": self.total_races,
            "current_race": self.current_race,
            "next_race": min(self.current_race + 1, self.total_races),
            "phase": self.phase.value,
        }
