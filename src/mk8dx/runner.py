"""ライブランナー（ステップ5で実装、docs/step5-notes.md 参照）。

キャプチャフレームを検知周期で受け取り、状態機械（state.py）と
RaceProcessor をつないで「検出→確定→保存→照合」を回す中核ループ。
フレーム源（カメラ/動画ファイル）からは独立しており、step() に
フレームと時刻を渡すだけで駆動できる（テスト・エンドツーエンド
検証は動画ファイルで時刻を仮想化する）。

流れ:
- WAITING_RACE: 名前列インクでリザルト候補を検出したら
  RESULT_DETECTING に入り、観測の蓄積を始める。
- RESULT_DETECTING: 観測を processor に供給し、割り当てが安定
  （ready）したら確定する。確定直前に最終フレームをフェーズ判別し、
  累計画面（フェーズB）だった場合はレースとして処理せず、
  直近レースの照合に回す（Aを見逃したケースの安全弁）。
  候補が2サンプル連続で消えたら WAITING_RACE に戻る。
- 確定できないレースは is_valid=False で保存して AWAITING_REVIEW
  に倒し、管理画面の承認/破棄を待つ（approve_pending /
  discard_pending が DB と状態機械をまとめて進める）。
- COOLDOWN: 画面が消え、最低時間が過ぎたら次のレースへ。
  その間にフェーズBが映ったら、自前累計との照合を1回だけ行い、
  不一致があれば警告イベントを出す。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import db
from .detector import looks_like_result
from .pipeline import (
    PHASE_STANDINGS,
    StandingsReading,
    analyze_standings_frame,
    classify_phase,
    extract_observations,
    verify_result_candidate,
)
from .processor import RaceOutcome, RaceProcessor
from .roster import Observation
from .state import MatchState, Phase


def save_outcome(
    conn, session_id: str, outcome: RaceOutcome
) -> int:
    """RaceOutcome を DB に保存し、race_result_id を返す。

    confirmed のときだけ is_valid=True（順位表に反映される）。
    """
    team_id_by_index: dict[int, int] = {}
    if outcome.assignment is not None:
        for team in outcome.assignment.teams:
            team_id = db.get_or_create_team(
                conn, session_id, team.tag, team.normalized_tag
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
    return db.save_race_result(
        conn, session_id, outcome.race_index, entries,
        is_valid=outcome.confirmed,
    )


@dataclass
class PendingReview:
    race_result_id: int
    outcome: RaceOutcome


@dataclass
class LiveRunner:
    roi: dict
    ocr: object
    processor: RaceProcessor
    match: MatchState
    conn: object
    session_id: str
    on_event: Callable[[str, dict], None] | None = None
    # 差し替え可能にしてテストではフェイクを使う
    classify_fn: Callable = classify_phase
    observations_fn: Callable = extract_observations
    standings_fn: Callable = analyze_standings_frame
    result_like_fn: Callable = looks_like_result
    verify_fn: Callable = verify_result_candidate

    pending: PendingReview | None = None
    last_outcome: RaceOutcome | None = None
    cross_check_problems: list[str] | None = None
    _gone_count: int = 0
    _computed_totals: dict[str, int] = field(default_factory=dict)
    _last_completed_race: int | None = None
    _cross_checked_race: int | None = None
    _last_classify_at: float = -10.0
    _last_verify_fail_at: float = -10.0

    # -- メインループ ------------------------------------------------
    def step(self, frame: np.ndarray, now: float) -> str:
        """フレーム1サンプル分を処理し、現在のフェーズ名を返す。"""
        phase = self.match.phase
        if phase is Phase.WAITING_RACE:
            # 名前列インクだけではレース前の紹介画面も候補になるため、
            # 順位列の数字で検証してから検出状態に入る（検証失敗は
            # 1秒間抑制して無駄なOCRを避ける）
            if (
                self.result_like_fn(frame, self.roi)
                and now - self._last_verify_fail_at >= 1.0
            ):
                if self.verify_fn(frame, self.roi, self.ocr):
                    self.match.on_result_candidate()
                    self.processor.begin_race(self.match.current_race + 1)
                    self._gone_count = 0
                    self._feed(frame)
                else:
                    self._last_verify_fail_at = now
        elif phase is Phase.RESULT_DETECTING:
            if self.result_like_fn(frame, self.roi):
                self._gone_count = 0
                if self._feed(frame) == "ready":
                    self._finalize(frame, now)
            else:
                self._gone_count += 1
                if self._gone_count >= 2:
                    self.match.on_screen_gone()
        elif phase is Phase.COOLDOWN:
            self._cooldown_step(frame, now)
        # AWAITING_REVIEW / PROCESSING / FINISHED では何もしない
        self.match.tick(
            screen_gone=not self.result_like_fn(frame, self.roi), now=now
        )
        return self.match.phase.value

    def _feed(self, frame: np.ndarray) -> str:
        observations = self.observations_fn(frame, self.roi, self.ocr)
        return self.processor.add_frame(observations)

    def _finalize(self, frame: np.ndarray, now: float) -> None:
        # 安全弁: 最終フレームが累計画面ならレースとして処理しない
        phase = self.classify_fn(frame, self.roi, self.ocr)
        if phase == PHASE_STANDINGS:
            self.match.on_screen_gone()
            self._try_cross_check(frame, now)
            return
        self.match.on_screen_stable()
        outcome = self.processor.finalize()
        race_result_id = save_outcome(self.conn, self.session_id, outcome)
        self.last_outcome = outcome
        self.match.on_processing_done(valid=outcome.confirmed, now=now)
        if outcome.confirmed:
            self._register_completed(outcome)
        else:
            self.pending = PendingReview(race_result_id, outcome)
            self._emit("alert", {
                "message": "選手名簿との照合で要確認のレースがあります",
                "problems": outcome.problems,
                "race_number": outcome.race_index,
            })
        self._emit("state", {})

    def _cooldown_step(self, frame: np.ndarray, now: float) -> None:
        if (
            self._last_completed_race is not None
            and self._cross_checked_race != self._last_completed_race
            and now - self._last_classify_at >= 1.0
            and self.result_like_fn(frame, self.roi)
        ):
            self._last_classify_at = now
            self._try_cross_check(frame, now)

    def _try_cross_check(self, frame: np.ndarray, now: float) -> None:
        if self._last_completed_race is None:
            return
        if self._cross_checked_race == self._last_completed_race:
            return
        if self.classify_fn(frame, self.roi, self.ocr) != PHASE_STANDINGS:
            return
        reading = self.standings_fn(frame, self.roi, self.ocr)
        resolved_reading = self._resolve_standings(reading)
        from .pipeline import cross_check_standings

        problems = cross_check_standings(self._computed_totals, resolved_reading)
        self.cross_check_problems = problems
        self._cross_checked_race = self._last_completed_race
        if problems:
            self._emit("alert", {
                "message": "累計画面との照合で不一致があります",
                "problems": problems,
                "race_number": self._last_completed_race,
            })
        self._emit("cross_check", {"problems": problems})

    def _resolve_standings(self, reading: StandingsReading) -> StandingsReading:
        """照合用にフェーズBの名前を選手名簿で名寄せする（学習なし）。"""
        observations = [Observation(name=e["name"]) for e in reading.entries]
        resolution = self.processor.roster.resolve(observations)
        names = self.processor.roster.resolved_names(resolution)
        entries = []
        for e, resolved in zip(reading.entries, names):
            entries.append({**e, "name": resolved or e["name"]})
        return StandingsReading(entries=entries, problems=reading.problems)

    def _register_completed(self, outcome: RaceOutcome) -> None:
        for e in outcome.entries:
            name = e["resolved_name"]
            if name:
                self._computed_totals[name] = (
                    self._computed_totals.get(name, 0) + e["score"]
                )
        self._last_completed_race = outcome.race_index

    # -- 承認フロー --------------------------------------------------
    def approve_pending(self, now: float | None = None) -> bool:
        """要確認レースを承認する（DB有効化＋累計反映＋状態遷移）。"""
        if self.pending is None:
            return False
        db.approve_race_result(self.conn, self.pending.race_result_id)
        self._register_completed(self.pending.outcome)
        self.pending = None
        self.match.approve_pending(now)
        self._emit("state", {})
        return True

    def discard_pending(self) -> bool:
        """要確認レースを破棄する（DBから削除して再検出に戻る）。"""
        if self.pending is None:
            return False
        db.delete_race_result(self.conn, self.pending.race_result_id)
        self.pending = None
        self.match.discard_pending()
        self._emit("state", {})
        return True

    def pending_payload(self) -> dict | None:
        if self.pending is None:
            return None
        outcome = self.pending.outcome
        return {
            "race_result_id": self.pending.race_result_id,
            "race_number": outcome.race_index,
            "problems": outcome.problems,
            "entries": outcome.entries,
            "teams": outcome.teams,
        }

    def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event is not None:
            self.on_event(kind, payload)
