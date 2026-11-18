"""レース処理の統合（ステップ4で実装、docs/step4-notes.md 参照）。

フェーズA（ゴール順画面）が表示されている間、複数フレームの観測を
蓄積し、安定したところで選手名簿照合→チーム判定→集計までを行う。

- フレームごとに名前の読みは揺れるため、順位ごとに多数決で
  コンセンサス観測を作る。
- 安定判定は「試し照合の割り当て（エントリID列）が
  stable_frames 回連続で同じ」こと。割り当てが安定するまでは
  確定しない。試し照合では学習しない（学習は確定時に1回だけ）。
- 確定時の振り分け: 全行が名寄せでき、低信頼の埋め合わせがなく、
  クラスタリングが成立すれば confirmed。1つでも欠ければ
  awaiting_review（暫定結果つき）として管理画面の判断に回す。
  ステップ3の検証では埋め合わせは全件正しかったが、消去法は
  推測である以上、自動確定の根拠にはしない。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .clustering import Assignment, assign_teams
from .roster import Observation, Resolution, Roster
from .scoring import points_for_rank

STATUS_CONFIRMED = "confirmed"
STATUS_AWAITING_REVIEW = "awaiting_review"


@dataclass
class RaceOutcome:
    """1レース分の処理結果。"""

    status: str                    # confirmed / awaiting_review
    race_index: int
    entries: list[dict]            # {rank, raw_name, resolved_name, score, low_confidence}
    teams: list[dict]              # {tag, normalized_tag, members(順位), points}
    problems: list[str] = field(default_factory=list)   # 確定を妨げる問題
    warnings: list[str] = field(default_factory=list)   # 非ブロッキングの注意
    assignment: Assignment | None = None

    @property
    def confirmed(self) -> bool:
        return self.status == STATUS_CONFIRMED


class RaceProcessor:
    """1レース分の観測蓄積と確定処理を担う。セッション中は使い回す。"""

    def __init__(
        self,
        roster: Roster,
        fmt: str,
        *,
        assume_closed: bool = False,
        min_frames: int = 2,
        stable_frames: int = 2,
    ) -> None:
        self.roster = roster
        self.format = fmt
        self.assume_closed = assume_closed
        self.min_frames = min_frames
        self.stable_frames = stable_frames
        self._race_index: int | None = None
        self._votes: list[Counter] = []       # 順位ごとの名前投票
        self._last_seen: list[dict[str, int]] = []  # 順位ごとの最終観測フレーム
        self._flags: list = []                # 順位ごとの国旗署名（最初の有効値）
        self._masks: list = []                # 順位ごとの文字マスク（最新の有効値）
        self._frames_seen = 0
        self._last_signature: tuple | None = None
        self._stable_count = 0

    # -- 蓄積 ------------------------------------------------------
    def begin_race(self, race_index: int) -> None:
        self._race_index = race_index
        self._votes = [Counter() for _ in range(12)]
        self._last_seen = [dict() for _ in range(12)]
        self._flags = [None] * 12
        self._masks = [None] * 12
        self._frames_seen = 0
        self._last_signature = None
        self._stable_count = 0

    def add_frame(self, observations: list[Observation]) -> str:
        """観測1フレーム分を追加し、'accumulating' か 'ready' を返す。"""
        if self._race_index is None:
            raise RuntimeError("begin_race を先に呼んでください")
        if len(observations) != 12:
            raise ValueError(f"観測は12行分必要です: {len(observations)}")
        for i, obs in enumerate(observations):
            self._votes[i][obs.name] += 1
            self._last_seen[i][obs.name] = self._frames_seen
            if self._flags[i] is None and obs.flag is not None:
                self._flags[i] = obs.flag
            if obs.mask is not None:
                self._masks[i] = obs.mask
        self._frames_seen += 1
        signature = self._trial_signature()
        if signature is not None and signature == self._last_signature:
            self._stable_count += 1
        else:
            self._stable_count = 1
            self._last_signature = signature
        return "ready" if self.ready else "accumulating"

    @property
    def ready(self) -> bool:
        return (
            self._frames_seen >= self.min_frames
            and self._stable_count >= self.stable_frames
            and self._last_signature is not None
        )

    def consensus(self) -> list[Observation]:
        """順位ごとの多数決でコンセンサス観測を作る。

        同数の場合は最後に観測された表記を優先する（アニメーション
        完了後のフレームほど読みが正しいため）。
        """
        out: list[Observation] = []
        for i in range(12):
            if self._votes[i]:
                name = max(
                    self._votes[i].items(),
                    key=lambda kv: (kv[1], self._last_seen[i][kv[0]]),
                )[0]
            else:
                name = ""
            out.append(Observation(name=name, flag=self._flags[i],
                                   mask=self._masks[i]))
        return out

    def _trial_signature(self) -> tuple | None:
        """試し照合（学習なし）の割り当て列。安定判定に使う。"""
        if self._frames_seen == 0:
            return None
        resolution = self.roster.resolve(
            self.consensus(), assume_closed=self.assume_closed
        )
        return tuple(resolution.entry_ids)

    # -- 確定 ------------------------------------------------------
    def finalize(self, *, force: bool = False) -> RaceOutcome:
        """蓄積した観測を確定処理する（選手名簿学習はここで1回だけ）。"""
        if self._race_index is None:
            raise RuntimeError("begin_race を先に呼んでください")
        if not self.ready and not force:
            raise RuntimeError("観測が安定していません（force=True で強制確定）")
        consensus = self.consensus()
        ids_before = {e.id for e in self.roster.entries}
        resolution = self.roster.resolve_race(
            consensus, self._race_index, assume_closed=self.assume_closed
        )
        new_ids = {e.id for e in self.roster.entries} - ids_before
        if new_ids:
            # 学習で新規エントリができた行は、作られた名簿に対して
            # もう一度照合し直すと名前が引ける（登録なし運用の初回
            # レースでも集計できるようにするため）。学習は済んでいる
            # ので、ここでの照合は学習なしの純粋な解決だけを行う。
            resolution = self.roster.resolve(
                consensus, assume_closed=self.assume_closed
            )
        outcome = self._build_outcome(consensus, resolution)
        if new_ids:
            outcome.problems.append(
                f"選手名簿に新規の選手が{len(new_ids)}名登録されました"
                "（自動登録の内容を確認してください）"
            )
            outcome.status = "awaiting_review"
        return outcome

    def process_single(self, observations: list[Observation],
                       race_index: int) -> RaceOutcome:
        """1フレーム分だけで確定する（検証・シミュレーション用）。"""
        self.begin_race(race_index)
        self.add_frame(observations)
        return self.finalize(force=True)

    def _build_outcome(
        self, consensus: list[Observation], resolution: Resolution
    ) -> RaceOutcome:
        problems: list[str] = []
        warnings: list[str] = []
        resolved = self.roster.resolved_names(resolution)
        entries: list[dict] = []
        for i, obs in enumerate(consensus):
            rank = i + 1
            if resolved[i] is None:
                problems.append(
                    f"{rank}位の名前を選手名簿と照合できませんでした: {obs.name!r}"
                )
            if resolution.low_confidence[i]:
                problems.append(
                    f"{rank}位は消去法での割り当てのため要確認です: {obs.name!r}"
                )
            entries.append({
                "rank": rank,
                "raw_name": obs.name,
                "resolved_name": resolved[i],
                "score": points_for_rank(rank),
                "low_confidence": resolution.low_confidence[i],
            })

        teams: list[dict] = []
        assignment: Assignment | None = None
        if all(n is not None for n in resolved):
            assignment = assign_teams(list(resolved), self.format)  # type: ignore[arg-type]
            warnings.extend(assignment.warnings)
            if assignment.valid:
                for team in assignment.teams:
                    members = [m + 1 for m in team.members]
                    teams.append({
                        "tag": team.tag,
                        "normalized_tag": team.normalized_tag,
                        "members": members,
                        "points": sum(points_for_rank(r) for r in members),
                    })
            else:
                problems.append("チーム割り当てが人数制約を満たしませんでした")
        status = STATUS_CONFIRMED if not problems else STATUS_AWAITING_REVIEW
        return RaceOutcome(
            status=status,
            race_index=self._race_index or 0,
            entries=entries,
            teams=teams,
            problems=problems,
            warnings=warnings,
            assignment=assignment,
        )
