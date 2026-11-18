"""roster.py（ロスター照合）のテスト。"""
import itertools
import random

import numpy as np

from mk8dx.roster import (
    Observation,
    Roster,
    hungarian,
    name_cost,
)


def test_hungarian_matches_brute_force():
    rng = random.Random(42)
    for _ in range(30):
        n, m = 4, 6
        cost = [[rng.uniform(0, 5) for _ in range(m)] for _ in range(n)]
        assignment = hungarian(cost)
        got = sum(cost[r][c] for r, c in enumerate(assignment))
        best = min(
            sum(cost[r][c] for r, c in enumerate(cols))
            for cols in itertools.permutations(range(m), n)
        )
        assert abs(got - best) < 1e-9


def test_name_cost_symbol_drop_is_cheap():
    # '★' の欠落は安く、別タグへの置換は高い（ステップ2の実測パターン）
    assert name_cost("RR", "RR★") < name_cost("RR", "FR")
    assert name_cost("RR", "RR★") < 0.5


def test_name_cost_known_confusions():
    assert name_cost("XI c.qq", "xı c.qq") < 1.0
    assert name_cost("racist #7", "racist #1") < 1.0
    # 先頭文字が違うとペナルティが乗る（タグ先頭はチーム判定の要）
    assert name_cost("AR", "RR") > name_cost("RR", "RR")
    # 同タグの別選手より、他タグの短い名前の方が近く見えることがある。
    # それでも実運用では本人のエントリが完全一致（コスト0）で勝つため、
    # 一括割り当てと組み合わせれば問題にならない
    assert name_cost("FR", "FR") == 0.0


def _flag(value: float) -> np.ndarray:
    return np.full(16 * 11 * 3, value, dtype=np.float32)


def test_global_assignment_avoids_local_greedy_error():
    # 行ごとに最近傍で決めると 'RR' が 'FR' と競合しうるケースでも、
    # 一括割り当てでは全体最適になる
    roster = Roster()
    roster.seed("RR★")
    roster.seed("FR")
    res = roster.resolve([Observation("RR"), Observation("FR")])
    names = roster.resolved_names(res)
    assert names == ["RR★", "FR"]


def test_duplicate_names_disambiguated_by_flag():
    roster = Roster()
    jp = roster.seed("RR★", flag=_flag(0.0))
    be = roster.seed("RR★", flag=_flag(1.0))
    # 名前が化けた行でも国旗が合えば正しい側に入る
    res = roster.resolve([
        Observation("日日-", flag=_flag(0.0)),
        Observation("RR", flag=_flag(1.0)),
    ])
    assert res.entry_ids == [jp.id, be.id]


def test_capacity_allows_same_entry_twice():
    roster = Roster()
    entry = roster.seed("RR★", capacity=2)
    roster.seed("FR")
    res = roster.resolve([
        Observation("RR★"), Observation("RR"), Observation("FR"),
    ])
    assert res.entry_ids == [entry.id, entry.id, roster.entries[1].id]


def test_provisional_entry_created_and_confirmed():
    roster = Roster()
    res = roster.resolve_race([Observation("new player")], race_index=1)
    assert res.new_candidates == [0]
    assert len(roster.entries) == 1
    assert not roster.entries[0].confirmed
    # 次のレースでも現れれば本登録になる
    roster.resolve_race([Observation("new player")], race_index=2)
    assert roster.entries[0].confirmed


def test_canonical_name_follows_majority():
    # 自動構築エントリの正規表記は観測の多数決で決まる
    roster = Roster()
    roster.resolve_race([Observation("A大ぜんだかいとしき")], race_index=1)
    entry = roster.entries[0]
    assert entry.canonical_name == "A大ぜんだかいとしき"
    for race in (2, 3):
        roster.resolve_race([Observation("A★ぜんだかいとしき")], race_index=race)
    assert entry.canonical_name == "A★ぜんだかいとしき"


def test_fragment_read_does_not_steal_short_canonical():
    # 1文字のゴミ読み '-' は 'RR★' への素の距離が小さいが、
    # 断片ペナルティで未割り当てコスト（4.0）より高くなる
    assert name_cost("-", "RR★") > 4.0
    # 3文字の化け読みは断片扱いにならない
    assert name_cost("日日-", "RR★") < 4.0


def test_ascii_star_treated_as_symbol():
    # en モデルは '★' を '*' と読む。コスト計算では同一視する
    assert name_cost("RR", "RR*") < 0.5
    assert name_cost("RR*", "RR★") == 0.0


def test_provisional_flag_mismatch_does_not_penalize():
    # 仮登録エントリの国旗（単一抽出由来）は、名前一致を弾く根拠にしない
    roster = Roster()
    res = roster.resolve_race(
        [Observation("some player", flag=_flag(0.0))], race_index=1
    )
    assert len(roster.entries) == 1
    res = roster.resolve([Observation("some player", flag=_flag(1.0))])
    assert res.entry_ids[0] == roster.entries[0].id


def test_seeded_canonical_name_is_pinned():
    # OCRは '★' を系統的に落とすため、事前登録名は多数決で流れない
    roster = Roster()
    entry = roster.seed("RR★")
    for race in (1, 2, 3):
        roster.learn(
            [Observation("RR")],
            roster.resolve([Observation("RR")]),
            race_index=race,
        )
    assert entry.canonical_name == "RR★"


def test_closed_fill_marks_low_confidence_and_skips_learning():
    roster = Roster()
    for name in ["alpha", "bravo", "charlie"]:
        roster.seed(name)
    res = roster.resolve(
        [Observation("alpha"), Observation("bravo"), Observation("!!!")],
        assume_closed=True,
    )
    assert res.entry_ids[2] is not None
    assert res.low_confidence[2]
    roster.learn(
        [Observation("alpha"), Observation("bravo"), Observation("!!!")],
        res, race_index=1,
    )
    charlie = roster.entries[2]
    assert "!!!" not in charlie.variants
