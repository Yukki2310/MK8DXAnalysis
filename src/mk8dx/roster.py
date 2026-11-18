"""ロスター照合（ステップ3で実装、docs/step3-notes.md 参照）。

リザルト画面のOCR読み（名前＋国旗）を、正規名簿（ロスター）の
エントリへ名寄せする。単一フレームの生OCRは記号欠落や文字化けが
多く、そのままではクラスタリング（接頭辞の完全一致前提）が成立
しないため、この層で吸収する。

設計:
- エントリは (名前, 国旗) の組で識別する。同名プレイヤーは国旗で
  区別し、国旗まで同じ場合は1エントリに「容量（最大同時出現数）」
  を持たせて複数行への割り当てを許す（同名＝同タグ＝同チームなので
  集計に影響しない）。
- 名寄せは行ごとの最近傍ではなく、12行×ロスタースロットの一括
  割り当て（ハンガリアン法）で総コスト最小にする。個別に決めると
  'RR'（★欠落）が 'FR' に吸着するような局所的な誤りが起きるが、
  全体最適では FR 枠を本物の FR が取るため RR は RR★ 枠に回る。
- コストは重み付き編集距離。ステップ2で観測した系統的誤り
  （記号の欠落は安い、既知の混同ペアは安い、タグ先頭の相違は高い）
  を織り込む。国旗が一致/不一致ならボーナス/ペナルティを加える。
- 割り当て結果はロスター学習に使う: 観測表記の多数決で正規名を
  更新し、国旗署名を蓄積し、容量を観測最大に合わせる。未割り当ての
  行は新規エントリ候補（仮登録）になり、複数レースで再見したら
  本登録になる。低信頼の埋め合わせ割り当ては学習に使わない。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .tags import normalize_name

# --- 国旗署名 ---
# 署名同士の平均絶対差。実測（2026-11-18）: 同一国旗 0.084 /
# 異なる国旗 0.31 以上。この間のグレーゾーンでは補正しない。
FLAG_SAME_THRESHOLD = 0.15
FLAG_DIFF_THRESHOLD = 0.25
FLAG_BONUS = -1.2
FLAG_PENALTY = 1.8

# --- 名前コスト ---
# '*' は en モデルが '★' を読んだ形。コスト計算では '★' と同一視する
SYMBOL_CHARS = set("★◆◇☆*")
SYMBOL_EDIT_COST = 0.3
CONFUSION_COST = 0.35
SYMBOL_SUB_COST = 0.6
BASE_EDIT_COST = 1.0
FIRST_CHAR_PENALTY = 1.2
FRAGMENT_PENALTY_PER_CHAR = 0.8
# ステップ2の実データで観測した混同ペア（正規化後の文字同士）
_CONFUSION_PAIRS = {
    frozenset(pair)
    for pair in [
        ("ı", "i"),
        ("i", "l"),
        ("1", "7"),
        ("0", "o"),
        ("★", "大"),
        ("★", "か"),
        ("ー", "-"),
    ]
}

# --- 割り当て ---
UNASSIGNED_COST = 4.0     # このコストのダミー枠に落ちたら未割り当て
PROVISIONAL_PENALTY = 0.4  # 仮登録エントリへの割り当ては少し割高にする
NEW_ENTRY_MIN_LEN = 2      # これより短い観測名は新規登録しない（ゴミ読み対策）


def extract_flag_signature(frame: np.ndarray, roi: dict, rank: int) -> np.ndarray | None:
    """行の国旗画像から署名（同一判定専用の特徴量）を作る。

    国旗は名前行右端の帯（roi の "flag"）の中にある。帯の左端は
    背景だけが写るので、その中央色との差分が大きい連結成分を
    旗の矩形として切り出し、16x11 に縮小して署名にする。
    国旗の「国名」を当てるのではなく、同一/相異なるだけを判定する。
    ハイライト行（黄色背景）でも背景中央色を使うため影響を受けない。
    """
    import cv2

    regions = roi.get("rows", {}).get(str(rank), {})
    if "flag" not in regions:
        return None
    x, y, w, h = regions["flag"]
    strip = frame[y : y + h, x : x + w]
    if strip.size == 0:
        return None
    strip_i = strip.astype(np.int16)
    bg = np.median(strip_i[:, :40].reshape(-1, 3), axis=0)
    dist = np.abs(strip_i - bg).sum(axis=2)
    mask = (dist > 90).astype(np.uint8)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if cx >= 35 and area >= 300:
            if best is None or area > best[4]:
                best = (cx, cy, cw, ch, area)
    if best is None:
        return None
    cx, cy, cw, ch, _ = best
    crop = strip[cy : cy + ch, cx : cx + cw]
    sig = cv2.resize(crop, (16, 11)).astype(np.float32).flatten() / 255.0
    return sig


def flag_distance(a: np.ndarray, b: np.ndarray) -> float:
    """国旗署名同士の距離（平均絶対差、0..1）。"""
    return float(np.abs(a - b).mean())


def _flag_adjustment(obs_flag: np.ndarray | None, entry: "RosterEntry") -> float:
    """観測国旗とエントリ国旗の一致度によるコスト補正。

    ボーナスは署名が1つでもあれば適用する。一方ペナルティは
    本登録エントリにだけ適用する: 仮登録エントリの国旗は単一の
    ノイズ混じり抽出から学んだだけのことがあり、それを根拠に
    名前が一致する行を弾くと照合全体が崩れる（ステップ3検証で
    実測した失敗パターン）。
    """
    if obs_flag is None or not entry.flag_signatures:
        return 0.0
    d = min(flag_distance(obs_flag, s) for s in entry.flag_signatures)
    if d < FLAG_SAME_THRESHOLD:
        return FLAG_BONUS
    if d > FLAG_DIFF_THRESHOLD and entry.confirmed:
        return FLAG_PENALTY
    return 0.0


def _compact(name: str) -> str:
    """コスト計算用に正規化し、空白を除く（OCRは空白をよく落とす）。

    en モデルが '★' を '*' と読むため、コスト計算上は同一視する。
    """
    return normalize_name(name).replace(" ", "").replace("　", "").replace("*", "★")


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if frozenset((a, b)) in _CONFUSION_PAIRS:
        return CONFUSION_COST
    if a in SYMBOL_CHARS or b in SYMBOL_CHARS:
        return SYMBOL_SUB_COST
    return BASE_EDIT_COST


def _indel_cost(ch: str) -> float:
    return SYMBOL_EDIT_COST if ch in SYMBOL_CHARS else BASE_EDIT_COST


def name_cost(observed: str, canonical: str) -> float:
    """観測名と正規名の重み付き編集距離（小さいほど同一人物らしい）。

    - 記号（★◆◇☆）の欠落・混入は安い（OCRが最も落としやすい）
    - 既知の混同ペア（ı/i、1/7、★/大 など）は安い
    - 先頭文字が違う場合はペナルティ（タグの先頭はチーム判定の要）
    """
    a, b = _compact(observed), _compact(canonical)
    if not a or not b:
        return UNASSIGNED_COST + 1.0
    # 重み付きレーベンシュタイン
    prev = [0.0]
    for ch_b in b:
        prev.append(prev[-1] + _indel_cost(ch_b))
    for ch_a in a:
        cur = [prev[0] + _indel_cost(ch_a)]
        for j, ch_b in enumerate(b, start=1):
            cur.append(min(
                prev[j] + _indel_cost(ch_a),          # 削除（観測の余分な文字）
                cur[j - 1] + _indel_cost(ch_b),       # 挿入（正規名の欠けた文字）
                prev[j - 1] + _sub_cost(ch_a, ch_b),  # 置換
            ))
        prev = cur
    cost = prev[-1]
    if a[0] != b[0] and frozenset((a[0], b[0])) not in _CONFUSION_PAIRS:
        cost += FIRST_CHAR_PENALTY
    # 断片ペナルティ: 観測が正規名の半分未満の長さしかない場合、
    # 編集距離は絶対値が小さくなりすぎ、短いゴミ読み（'-' 等）が
    # 短い正規名を奪ってしまう（ステップ4検証で実測）。不足分を足す。
    if len(a) * 2 < len(b):
        cost += (len(b) - len(a)) * FRAGMENT_PENALTY_PER_CHAR
    return cost


def hungarian(cost: list[list[float]]) -> list[int]:
    """コスト行列（行数 <= 列数）に対する最小コスト割り当て。

    戻り値は行ごとの割り当て列インデックス。Jonker-Volgenant 法の
    簡易実装（O(n^2 m)）。12行×数十列程度の規模を想定する。
    """
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0 or m == 0 or n > m:
        raise ValueError("hungarian: 行数 <= 列数 の行列が必要です")
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)    # 列 j に割り当てられた行（1始まり、0=なし）
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


@dataclass
class Observation:
    """リザルト画面の1行分の観測。"""

    name: str
    flag: np.ndarray | None = None


@dataclass
class RosterEntry:
    """ロスターの1エントリ（名前×国旗で識別される選手）。"""

    id: int
    variants: Counter = field(default_factory=Counter)  # 観測表記 -> 回数
    flag_signatures: list[np.ndarray] = field(default_factory=list)
    capacity: int = 1          # 最大同時出現数（同名・同国旗の複数人対策）
    confirmed: bool = False    # 本登録か（仮登録は割り当てで少し割高）
    races_seen: int = 0
    last_seen_race: int = -1
    pinned_name: str | None = None  # 事前登録された正規名（多数決で変えない）

    @property
    def canonical_name(self) -> str:
        """現在の正規表記。

        事前登録された名前は固定する（OCRは '★' を系統的に落とすため、
        多数決では正規表記が読み落とし側に流れてしまう）。自動構築の
        エントリは多数決で決める: 最多の観測表記を採り、同数なら
        記号を含む表記（読み落としが少ない側）、次に長い表記を優先。
        """
        if self.pinned_name:
            return self.pinned_name
        if not self.variants:
            return ""
        return max(
            self.variants.items(),
            key=lambda kv: (
                kv[1],
                any(c in SYMBOL_CHARS for c in kv[0]),
                len(kv[0]),
            ),
        )[0]


@dataclass
class Resolution:
    """1レース分の照合結果。"""

    entry_ids: list[int | None]        # 行ごと（None=未割り当て）
    costs: list[float]                 # 行ごとの割り当てコスト
    low_confidence: list[bool]         # 埋め合わせで割り当てた行か
    new_candidates: list[int] = field(default_factory=list)  # 未割り当ての行番号


class Roster:
    """セッションを通じて維持する正規名簿。"""

    def __init__(self) -> None:
        self.entries: list[RosterEntry] = []
        self._next_id = 1

    # -- 登録 ------------------------------------------------------
    def seed(self, name: str, flag: np.ndarray | None = None,
             capacity: int = 1) -> RosterEntry:
        """事前登録（本登録扱い）。管理画面からの手動登録もここを使う。"""
        entry = RosterEntry(id=self._next_id, confirmed=True)
        self._next_id += 1
        entry.variants[name] = 2  # 初期の正規表記がすぐ覆らない重み
        entry.pinned_name = name
        entry.capacity = capacity
        if flag is not None:
            entry.flag_signatures.append(flag)
        self.entries.append(entry)
        return entry

    def _new_provisional(self, obs: Observation) -> RosterEntry:
        entry = RosterEntry(id=self._next_id, confirmed=False)
        self._next_id += 1
        entry.variants[obs.name] = 1
        if obs.flag is not None:
            entry.flag_signatures.append(obs.flag)
        self.entries.append(entry)
        return entry

    def remove(self, entry_id: int) -> bool:
        """エントリを削除する（管理画面用）。削除できたら True。"""
        for i, e in enumerate(self.entries):
            if e.id == entry_id:
                del self.entries[i]
                return True
        return False

    def _entry_by_id(self, entry_id: int) -> RosterEntry | None:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    # -- 照合 ------------------------------------------------------
    def _slots(self) -> list[RosterEntry]:
        """容量ぶんに展開した割り当て枠。"""
        slots: list[RosterEntry] = []
        for e in self.entries:
            slots.extend([e] * max(1, e.capacity))
        return slots

    def resolve(self, observations: list[Observation], *,
                assume_closed: bool = False) -> Resolution:
        """観測12行をロスターへ一括割り当てする。

        assume_closed=True のときは「参加者はロスター全員」という
        前提で、未割り当ての行と空き枠を埋め合わせる（低信頼として
        記録し、学習には使わない）。事前登録済みロスターでの運用向け。
        """
        n = len(observations)
        slots = self._slots()
        n_dummy = n  # 未割り当て受けのダミー枠
        n_cols = len(slots) + n_dummy
        matrix: list[list[float]] = []
        for obs in observations:
            row: list[float] = []
            for e in slots:
                c = name_cost(obs.name, e.canonical_name)
                c += _flag_adjustment(obs.flag, e)
                if not e.confirmed:
                    c += PROVISIONAL_PENALTY
                row.append(c)
            row.extend([UNASSIGNED_COST] * n_dummy)
            matrix.append(row)
        cols = hungarian(matrix) if n and n_cols >= n else []
        entry_ids: list[int | None] = [None] * n
        costs = [UNASSIGNED_COST] * n
        low_conf = [False] * n
        used_slots: set[int] = set()
        for r, c in enumerate(cols):
            if 0 <= c < len(slots):
                entry_ids[r] = slots[c].id
                costs[r] = matrix[r][c]
                used_slots.add(c)
        if assume_closed:
            unresolved = [r for r in range(n) if entry_ids[r] is None]
            free = [c for c in range(len(slots)) if c not in used_slots]
            if unresolved and free:
                sub = [[matrix[r][c] for c in free] for r in unresolved]
                for r, sc in zip(unresolved, hungarian(sub)):
                    c = free[sc]
                    entry_ids[r] = slots[c].id
                    costs[r] = matrix[r][c]
                    low_conf[r] = True
        new_candidates = [r for r in range(n) if entry_ids[r] is None]
        return Resolution(
            entry_ids=entry_ids, costs=costs,
            low_confidence=low_conf, new_candidates=new_candidates,
        )

    # -- 学習 ------------------------------------------------------
    def learn(self, observations: list[Observation], resolution: Resolution,
              race_index: int) -> None:
        """割り当て結果をロスターに反映する（多数決・容量・国旗）。"""
        per_entry_rows: Counter = Counter()
        for r, entry_id in enumerate(resolution.entry_ids):
            if entry_id is None or resolution.low_confidence[r]:
                continue  # 埋め合わせは推測なので学習しない
            entry = self._entry_by_id(entry_id)
            if entry is None:
                continue
            obs = observations[r]
            entry.variants[obs.name] += 1
            per_entry_rows[entry_id] += 1
            if obs.flag is not None:
                if not entry.flag_signatures:
                    entry.flag_signatures.append(obs.flag)
                else:
                    d = min(flag_distance(obs.flag, s)
                            for s in entry.flag_signatures)
                    # 明らかに違う国旗は取り込まない（誤割り当ての毒入れ防止）
                    if d <= FLAG_DIFF_THRESHOLD and len(entry.flag_signatures) < 8:
                        entry.flag_signatures.append(obs.flag)
            if entry.last_seen_race != race_index:
                entry.races_seen += 1
                entry.last_seen_race = race_index
            if not entry.confirmed and entry.races_seen >= 2:
                entry.confirmed = True
        for entry_id, count in per_entry_rows.items():
            entry = self._entry_by_id(entry_id)
            if entry is not None and count > entry.capacity:
                entry.capacity = count
        # 未割り当て行は新規プレイヤー候補として仮登録する
        for r in resolution.new_candidates:
            obs = observations[r]
            if len(_compact(obs.name)) >= NEW_ENTRY_MIN_LEN:
                entry = self._new_provisional(obs)
                entry.races_seen = 1
                entry.last_seen_race = race_index

    def resolve_race(self, observations: list[Observation], race_index: int, *,
                     assume_closed: bool = False) -> Resolution:
        """照合して学習までを一括で行う通常経路。"""
        resolution = self.resolve(observations, assume_closed=assume_closed)
        self.learn(observations, resolution, race_index)
        return resolution

    def resolved_names(self, resolution: Resolution) -> list[str | None]:
        """照合結果を正規名のリストに変換する（未割り当ては None）。"""
        out: list[str | None] = []
        for entry_id in resolution.entry_ids:
            entry = self._entry_by_id(entry_id) if entry_id is not None else None
            out.append(entry.canonical_name if entry else None)
        return out
