"""選手名簿照合（ステップ3で実装、docs/step3-notes.md 参照）。

リザルト画面のOCR読み（名前＋国旗）を、正規名簿（選手名簿）の
エントリへ名寄せする。単一フレームの生OCRは記号欠落や文字化けが
多く、そのままではクラスタリング（接頭辞の完全一致前提）が成立
しないため、この層で吸収する。

設計:
- エントリは (名前, 国旗) の組で識別する。同名プレイヤーは国旗で
  区別し、国旗まで同じ場合は1エントリに「容量（最大同時出現数）」
  を持たせて複数行への割り当てを許す（同名＝同タグ＝同チームなので
  集計に影響しない）。
- 名寄せは行ごとの最近傍ではなく、12行×選手名簿スロットの一括
  割り当て（ハンガリアン法）で総コスト最小にする。個別に決めると
  'AA'（★欠落）が 'BA' に吸着するような局所的な誤りが起きるが、
  全体最適では BA 枠を本物の BA が取るため AA は AA★ 枠に回る。
- コストは重み付き編集距離。ステップ2で観測した系統的誤り
  （記号の欠落は安い、既知の混同ペアは安い、タグ先頭の相違は高い）
  を織り込む。国旗が一致/不一致ならボーナス/ペナルティを加える。
- 割り当て結果は選手名簿学習に使う: 観測表記の多数決で正規名を
  更新し、国旗署名を蓄積し、容量を観測最大に合わせる。未割り当ての
  行は新規エントリ候補（仮登録）になり、複数レースで再見したら
  本登録になる。低信頼の埋め合わせ割り当ては学習に使わない。
"""
from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .identity import (
    SAME_THRESHOLD as IMAGE_SAME,
    mask_from_dict,
    mask_to_dict,
    prepare_mask,
    prepared_distance,
)
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
SHORT_NAME_SURCHARGE = 1.0
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

# チームの表示色（オーバーレイのカラーバー）
TEAM_PALETTE = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]

# --- 画像ベース識別（identity.py、フェーズ1）---
# 文字マスク距離をコストへ換算する倍率。距離0.4以上でUNASSIGNED_COST(4.0)
# に達し「新規プレイヤー」になる。OCR名のコストは同点時の微調整に使うだけ。
IMAGE_COST_SCALE = 10.0
IMAGE_NAME_TIEBREAK = 0.03
IMAGE_NAME_TIEBREAK_CAP = 5.0
MAX_ELIMINATED = 2          # 消去法で決めてよい行数の上限（それ以上は新しい選手とみなす）
MASK_EXEMPLARS = 4          # エントリごとに保持するマスク見本の最大数
MASK_NEW_DIVERSITY = 0.04   # これ未満の距離の見本は追加しない（重複回避）
MASK_POISON_LIMIT = 0.3     # これを超えて違うマスクは取り込まない

# --- 割り当て ---
UNASSIGNED_COST = 4.0     # このコストのダミー枠に落ちたら未割り当て
PROVISIONAL_PENALTY = 0.4  # 仮登録エントリへの割り当ては少し割高にする
NEW_ENTRY_MIN_LEN = 2      # これより短い観測名は新規登録しない（ゴミ読み対策）
MERGE_MAX_COST = 0.75      # これ以下で似たエントリは統合候補（同時出現した組は除く）
PRUNE_AFTER_RACES = 3      # 仮登録がこのレース数現れなければ除去する

_NAME_LIKE = None  # 遅延コンパイルする正規表現（_is_name_like 参照）


def _is_name_like(name: str) -> bool:
    """新規登録に値する読みか（記号だけのゴミ読みを弾く）。

    英数字・かな・漢字のいずれかを含み、空白を除いて2文字以上ある
    ことを要求する。'-' や '!!' のような読みは選手名簿に残さない。
    """
    global _NAME_LIKE
    if _NAME_LIKE is None:
        import re

        _NAME_LIKE = re.compile(r"[0-9A-Za-zぁ-んァ-ヶ一-龥]")
    compact = _compact(name)
    return len(compact) >= NEW_ENTRY_MIN_LEN and bool(_NAME_LIKE.search(compact))


THUMB_WIDTH = 365  # 名前の文字領域の幅（これより右は国旗）


def encode_thumb(image: np.ndarray | None) -> bytes | None:
    """名前行クロップをJPEGに圧縮する（名簿に保存する小さな見本画像）。"""
    if image is None:
        return None
    import cv2

    ok, buf = cv2.imencode(".jpg", image[:, :THUMB_WIDTH],
                           [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None


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

    ボーナス・ペナルティとも本登録エントリにだけ適用する:
    仮登録エントリの国旗は単一のノイズ混じり抽出から学んだだけの
    ことがあり、文字化けした名前の仮登録が国旗ボーナスで別人の行を
    奪う事故が起きる（ステップ3検証の「三日」エントリが典型）。
    仮登録の国旗は、本登録になった時点から効き始める。
    """
    if obs_flag is None or not entry.flag_signatures or not entry.confirmed:
        return 0.0
    d = min(flag_distance(obs_flag, s) for s in entry.flag_signatures)
    if d < FLAG_SAME_THRESHOLD:
        return FLAG_BONUS
    if d > FLAG_DIFF_THRESHOLD:
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
    # 短い名前の弱い一致への加算: 2文字同士の名前は置換2回でも
    # コストが低くなり、無関係な短い正規名に吸着する（ステップ6
    # 検証で 'AA' が '三日' に吸着した実測）。完全一致や記号欠落
    # レベルの強い一致には加算しない。
    if min(len(a), len(b)) <= 2 and cost >= 1.5:
        cost += SHORT_NAME_SURCHARGE
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
    # 名前の文字マスク（identity.trim_mask の出力）。あれば画像照合を使う
    mask: np.ndarray | None = None
    # 名前行クロップ（BGR）。新規選手の表示名を読むときだけ使う（保存しない）
    image: np.ndarray | None = None


@dataclass
class RosterEntry:
    """選手名簿の1エントリ（名前×国旗で識別される選手）。"""

    id: int
    variants: Counter = field(default_factory=Counter)  # 観測表記 -> 回数
    flag_signatures: list[np.ndarray] = field(default_factory=list)
    capacity: int = 1          # 最大同時出現数（同名・同国旗の複数人対策）
    confirmed: bool = False    # 本登録か（仮登録は割り当てで少し割高）
    races_seen: int = 0
    last_seen_race: int = -1
    pinned_name: str | None = None  # 事前登録された正規名（多数決で変えない）
    co_seen: set[int] = field(default_factory=set)  # 同じレースに同時出現した相手のID
    masks: list[np.ndarray] = field(default_factory=list)  # 文字マスク見本
    name_reads: int = 0        # 表示名を文字認識で読んだ回数（上限あり）
    thumb: bytes | None = None  # 名前行の実画像（JPEG）。管理画面で誰か見分ける用
    team: str | None = None    # チーム名（表示・集計用）。None は未設定
    team_manual: bool = False  # 人が設定したか（False は自動推定）
    excluded: bool = False     # 集計から除く選手（不要な参加者）
    _prepared: list[np.ndarray] = field(default_factory=list, repr=False)

    def add_mask(self, mask: np.ndarray) -> bool:
        """見本を追加する。重複・毒入れ・上限超えは追加しない。"""
        if len(self.masks) >= MASK_EXEMPLARS:
            return False
        prepared = prepare_mask(mask)
        if self._prepared:
            d = min(prepared_distance(prepared, p) for p in self._prepared)
            if d < MASK_NEW_DIVERSITY or d > MASK_POISON_LIMIT:
                return False
        self.masks.append(mask)
        self._prepared.append(prepared)
        return True

    def image_distance(self, prepared_obs: np.ndarray | None) -> float | None:
        """観測マスク(prepare済み)と見本の最小距離。比較不能なら None。"""
        if prepared_obs is None or not self._prepared:
            return None
        return min(prepared_distance(prepared_obs, p) for p in self._prepared)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "variants": dict(self.variants),
            "flag_signatures": [
                [round(float(x), 4) for x in sig] for sig in self.flag_signatures
            ],
            "capacity": self.capacity,
            "confirmed": self.confirmed,
            "races_seen": self.races_seen,
            "last_seen_race": self.last_seen_race,
            "pinned_name": self.pinned_name,
            "co_seen": sorted(self.co_seen),
            "masks": [mask_to_dict(m) for m in self.masks],
            "name_reads": self.name_reads,
            "team": self.team,
            "team_manual": self.team_manual,
            "excluded": self.excluded,
            "thumb": base64.b64encode(self.thumb).decode("ascii") if self.thumb else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RosterEntry":
        entry = cls(id=int(data["id"]))
        entry.variants = Counter(data.get("variants", {}))
        entry.flag_signatures = [
            np.array(sig, dtype=np.float32)
            for sig in data.get("flag_signatures", [])
        ]
        entry.capacity = int(data.get("capacity", 1))
        entry.confirmed = bool(data.get("confirmed", False))
        entry.races_seen = int(data.get("races_seen", 0))
        entry.last_seen_race = int(data.get("last_seen_race", -1))
        entry.pinned_name = data.get("pinned_name")
        entry.co_seen = {int(x) for x in data.get("co_seen", [])}
        entry.name_reads = int(data.get("name_reads", 0))
        entry.thumb = base64.b64decode(data["thumb"]) if data.get("thumb") else None
        entry.team = data.get("team")
        entry.team_manual = bool(data.get("team_manual", False))
        entry.excluded = bool(data.get("excluded", False))
        entry.masks = [mask_from_dict(m) for m in data.get("masks", [])]
        entry._prepared = [prepare_mask(m) for m in entry.masks]
        return entry

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
        names = {k: v for k, v in self.variants.items() if k}
        if not names:
            return f"選手{self.id}" if self.masks else ""
        return max(
            names.items(),
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
        # チーム名 -> 色（オーバーレイのカラーバー）。チームを作った順に割り当てる
        self.team_colors: dict[str, str] = {}

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
        if obs.name:
            entry.variants[obs.name] = 1
        if obs.flag is not None:
            entry.flag_signatures.append(obs.flag)
        if obs.mask is not None:
            entry.add_mask(obs.mask)
        entry.thumb = encode_thumb(obs.image)
        self.entries.append(entry)
        return entry

    def remove(self, entry_id: int) -> bool:
        """エントリを削除する（管理画面用）。削除できたら True。"""
        for i, e in enumerate(self.entries):
            if e.id == entry_id:
                del self.entries[i]
                return True
        return False

    def rename(self, entry_id: int, name: str) -> bool:
        """エントリの正規名を人が直す（自動構築後の手直し用）。

        人が確認した名前なので固定名（pinned）にし、本登録扱いに
        する。国旗や出現履歴は引き継がれるため、削除して登録し直す
        より安全に名簿を整えられる。
        """
        entry = self._entry_by_id(entry_id)
        if entry is None or not name.strip():
            return False
        entry.pinned_name = name.strip()
        entry.variants[entry.pinned_name] += 2
        entry.confirmed = True
        return True

    # -- チーム・統合・画像 -------------------------------------------
    def set_team(self, entry_id: int, team: str | None, *, manual: bool = True) -> bool:
        """選手のチームを設定する（空文字・None で未設定に戻す）。"""
        entry = self._entry_by_id(entry_id)
        if entry is None:
            return False
        team = (team or "").strip() or None
        entry.team = team
        entry.team_manual = manual and team is not None
        if team:
            self.color_for(team)
        return True

    def set_excluded(self, entry_id: int, excluded: bool) -> bool:
        """選手を集計から除く／戻す。点は消さず、順位表の計算で除くだけ。"""
        entry = self._entry_by_id(entry_id)
        if entry is None:
            return False
        entry.excluded = bool(excluded)
        return True

    def color_for(self, team: str) -> str:
        """チーム名の色を返す（初めてのチームには未使用の色を割り当てる）。"""
        key = normalize_name(team)
        for label, color in self.team_colors.items():
            if normalize_name(label) == key:
                return color
        used = set(self.team_colors.values())
        free = [c for c in TEAM_PALETTE if c not in used]
        color = free[0] if free else TEAM_PALETTE[len(self.team_colors) % len(TEAM_PALETTE)]
        self.team_colors[team] = color
        return color

    def merge(self, from_id: int, into_id: int) -> bool:
        """from_id の選手を into_id と同一人物として統合する（誤った分裂の修正用）。"""
        a, b = self._entry_by_id(from_id), self._entry_by_id(into_id)
        if a is None or b is None or a is b:
            return False
        if b.team is None and a.team is not None:
            b.team, b.team_manual = a.team, a.team_manual
        b.excluded = b.excluded and a.excluded
        if a.pinned_name and not b.pinned_name:
            b.pinned_name = a.pinned_name
        self._absorb(b, a)
        return True

    def image_for(self, entry_id: int) -> tuple[bytes, str] | None:
        """選手の名前の画像と MIME。実画像があればそれ、無ければ文字マスクの画像。"""
        entry = self._entry_by_id(entry_id)
        if entry is None:
            return None
        if entry.thumb:
            return entry.thumb, "image/jpeg"
        png = self.mask_png(entry_id)
        return (png, "image/png") if png else None

    def mask_png(self, entry_id: int) -> bytes | None:
        """選手の名前の文字マスクの画像（実画像が無いときの代替PNG）。"""
        import cv2

        entry = self._entry_by_id(entry_id)
        if entry is None or not entry.masks:
            return None
        mask = entry.masks[0]
        img = np.full((mask.shape[0] + 12, mask.shape[1] + 20), 40, np.uint8)
        img[6:6 + mask.shape[0], 10:10 + mask.shape[1]][mask > 0.5] = 255
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes() if ok else None

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
                assume_closed: bool = False,
                eliminate_among: set[int] | None = None) -> Resolution:
        """観測12行を選手名簿へ一括割り当てする。

        assume_closed=True のときは「参加者は選手名簿全員」という
        前提で、未割り当ての行と空き枠を埋め合わせる（低信頼として
        記録し、学習には使わない）。事前登録済み選手名簿での運用向け。

        eliminate_among を渡すと、その選手の中だけを消去法の候補にする（未割り当ての
        行が MAX_ELIMINATED 行以内のときだけ）。前のレースに出ていた選手が1〜2人
        見つからないときに、その人だと決めるために使う。例えば配信画面の下に出る
        「レースを自動終了しました」の表示が12位の名前に重なって読めない場合。
        """
        n = len(observations)
        slots = self._slots()
        n_dummy = n  # 未割り当て受けのダミー枠
        n_cols = len(slots) + n_dummy
        matrix: list[list[float]] = []
        for obs in observations:
            row: list[float] = []
            prepared_obs = prepare_mask(obs.mask)
            img_dist: dict[int, float | None] = {}
            for e in slots:
                if e.id not in img_dist:
                    img_dist[e.id] = e.image_distance(prepared_obs)
                d = img_dist[e.id]
                if d is not None:
                    # 画像で比較できる: 文字マスク距離が主、OCR名は同点時のみ
                    nc = (
                        min(name_cost(obs.name, e.canonical_name),
                            IMAGE_NAME_TIEBREAK_CAP)
                        if obs.name else 0.0
                    )
                    c = d * IMAGE_COST_SCALE + IMAGE_NAME_TIEBREAK * nc
                    # 同じ文字（同名の別人）同士は国旗で区別する
                    if d < IMAGE_SAME:
                        c += _flag_adjustment(obs.flag, e)
                else:
                    c = name_cost(obs.name, e.canonical_name)
                    # 名前が完全一致する場合は国旗で覆さない（ステップ7
                    # 検証: 抽出ノイズで国旗ペナルティがかかり、同チームの
                    # 'name #1' が '#2' に化けた）。国旗が要るのは同名の
                    # 別人同士の区別だが、その場合は名前コストが両方とも
                    # 0で、どの枠に入っても表示名は同じになる。
                    if c > 0:
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
        if assume_closed or eliminate_among is not None:
            unresolved = [r for r in range(n) if entry_ids[r] is None]
            free = [c for c in range(len(slots)) if c not in used_slots]
            if eliminate_among is not None:
                free = [c for c in free if slots[c].id in eliminate_among]
                if len(unresolved) > MAX_ELIMINATED:
                    unresolved = []   # 多すぎるときは推測しない（新しい選手が来た可能性）
            if unresolved and free and len(unresolved) <= len(free):
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
        """割り当て結果を選手名簿に反映する（多数決・容量・国旗）。"""
        per_entry_rows: Counter = Counter()
        flags_by_entry: dict[int, list[np.ndarray]] = {}
        for r, entry_id in enumerate(resolution.entry_ids):
            if entry_id is None or resolution.low_confidence[r]:
                continue  # 埋め合わせは推測なので学習しない
            entry = self._entry_by_id(entry_id)
            if entry is None:
                continue
            obs = observations[r]
            if obs.name:
                entry.variants[obs.name] += 1
            if obs.mask is not None:
                entry.add_mask(obs.mask)
            # 実画像は最初の数レースの安定したフレームで更新する（読み取りが良くなる）
            if obs.image is not None and (entry.thumb is None or entry.races_seen < 2):
                entry.thumb = encode_thumb(obs.image) or entry.thumb
            per_entry_rows[entry_id] += 1
            if obs.flag is not None:
                flags_by_entry.setdefault(entry_id, []).append(obs.flag)
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
        # 容量の国旗ヒント: 同じ名前で明らかに違う国旗が観測されたら、
        # 同名の別人がいる証拠として容量を広げる（ステップ7検証:
        # 片方の AA★ が空読みになったレースがあると同時出現だけでは
        # 容量が育たず、次のレースでもう片方が幻エントリに流れた）。
        # 遠い国旗そのものは毒入れ防止のため署名には取り込まない。
        for entry_id, flags in flags_by_entry.items():
            entry = self._entry_by_id(entry_id)
            if entry is None:
                continue
            reps: list[np.ndarray] = list(entry.flag_signatures)
            n_clusters = 1 if reps else 0
            for f in flags:
                if all(flag_distance(f, c) > FLAG_DIFF_THRESHOLD
                       for c in reps):
                    reps.append(f)
                    n_clusters += 1
            hint = min(n_clusters, 4)
            if hint > entry.capacity:
                entry.capacity = hint
        # 未割り当て行は新規プレイヤー候補として仮登録する
        new_ids: list[int] = []
        for r in resolution.new_candidates:
            obs = observations[r]
            if obs.mask is not None or _is_name_like(obs.name):
                entry = self._new_provisional(obs)
                entry.races_seen = 1
                entry.last_seen_race = race_index
                new_ids.append(entry.id)
        # 同時出現の記録: このレースで直接割り当てられた組と新規組は
        # 「別々の行にいた」ので別人同士。以後の統合判定で使う
        present = {eid for eid in per_entry_rows} | set(new_ids)
        for eid in present:
            entry = self._entry_by_id(eid)
            if entry is not None:
                entry.co_seen |= present - {eid}
        self._merge_duplicates()
        self._prune_stale(race_index)

    def _merge_duplicates(self) -> None:
        """同一人物らしい重複エントリを統合する（自動構築の掃除）。

        条件: 正規名のコストが MERGE_MAX_COST 以下で、国旗が矛盾せず、
        かつ同じレースに同時出現したことがない組だけを統合する。
        同時出現した組は別人確定なので、コストが低くても統合しない
        （例: 'name #1' と 'name #2' は混同ペアでコストが低いが
        常に同時出現するため統合されない）。
        """
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(self.entries):
                for b in self.entries[i + 1:]:
                    if b.id in a.co_seen or a.id in b.co_seen:
                        continue
                    if a._prepared and b._prepared:
                        # 画像で比較できる組は画像だけで同一判定する
                        d = min(prepared_distance(pa, pb)
                                for pa in a._prepared for pb in b._prepared)
                        if d > IMAGE_SAME:
                            continue
                    elif name_cost(a.canonical_name, b.canonical_name) > MERGE_MAX_COST:
                        continue
                    if a.flag_signatures and b.flag_signatures:
                        d = min(
                            flag_distance(sa, sb)
                            for sa in a.flag_signatures
                            for sb in b.flag_signatures
                        )
                        if d > FLAG_DIFF_THRESHOLD:
                            continue  # 国旗が違う＝同名の別人
                    keep, drop = self._stronger(a, b)
                    self._absorb(keep, drop)
                    changed = True
                    break
                if changed:
                    break

    @staticmethod
    def _stronger(a: RosterEntry, b: RosterEntry) -> tuple[RosterEntry, RosterEntry]:
        key = lambda e: (e.races_seen, e.confirmed, -e.id)  # noqa: E731
        return (a, b) if key(a) >= key(b) else (b, a)

    def _absorb(self, keep: RosterEntry, drop: RosterEntry) -> None:
        keep.variants.update(drop.variants)
        for sig in drop.flag_signatures:
            if len(keep.flag_signatures) >= 8:
                break
            if all(flag_distance(sig, s) > 0.02 for s in keep.flag_signatures):
                keep.flag_signatures.append(sig)
        for m in drop.masks:
            keep.add_mask(m)
        keep.capacity = max(keep.capacity, drop.capacity)
        keep.races_seen += drop.races_seen
        keep.last_seen_race = max(keep.last_seen_race, drop.last_seen_race)
        keep.confirmed = keep.confirmed or drop.confirmed
        if keep.pinned_name is None:
            keep.pinned_name = drop.pinned_name
        keep.co_seen = (keep.co_seen | drop.co_seen) - {keep.id, drop.id}
        for e in self.entries:
            if drop.id in e.co_seen:
                e.co_seen.discard(drop.id)
                if e.id != keep.id:
                    e.co_seen.add(keep.id)
        self.entries.remove(drop)

    def _prune_stale(self, race_index: int) -> None:
        """長く現れない仮登録を除去する（1回きりの文字化け対策）。

        本登録と事前登録（pinned）は除去しない。離脱した選手が
        戻ってきた場合は、新規の仮登録として作り直される。
        """
        self.entries = [
            e for e in self.entries
            if e.confirmed
            or e.pinned_name is not None
            or e.team_manual
            or e.races_seen > 1
            or race_index - e.last_seen_race < PRUNE_AFTER_RACES
        ]

    # -- 永続化 ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "next_id": self._next_id,
            "entries": [e.to_dict() for e in self.entries],
            "team_colors": dict(self.team_colors),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Roster":
        roster = cls()
        roster.entries = [RosterEntry.from_dict(d) for d in data.get("entries", [])]
        roster._next_id = int(data.get("next_id", len(roster.entries) + 1))
        roster.team_colors = dict(data.get("team_colors", {}))
        return roster

    def resolve_race(self, observations: list[Observation], race_index: int, *,
                     assume_closed: bool = False,
                     eliminate_among: set[int] | None = None) -> Resolution:
        """照合して学習までを一括で行う通常経路。"""
        resolution = self.resolve(observations, assume_closed=assume_closed,
                                  eliminate_among=eliminate_among)
        self.learn(observations, resolution, race_index)
        return resolution

    def resolved_names(self, resolution: Resolution) -> list[str | None]:
        """照合結果を正規名のリストに変換する（未割り当ては None）。"""
        out: list[str | None] = []
        for entry_id in resolution.entry_ids:
            entry = self._entry_by_id(entry_id) if entry_id is not None else None
            out.append(entry.canonical_name if entry else None)
        return out
