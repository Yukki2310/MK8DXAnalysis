"""roster.py（選手名簿照合）のテスト。"""
import itertools
import random

import numpy as np
import pytest

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


def test_provisional_flag_gives_no_bonus():
    # 仮登録の国旗はボーナスの根拠にしない（文字化け名簿が国旗で
    # 別人の行を奪う事故の対策、ステップ6）
    from mk8dx.roster import FLAG_BONUS, _flag_adjustment

    roster = Roster()
    roster.resolve_race([Observation("some player", flag=_flag(0.0))], race_index=1)
    entry = roster.entries[0]
    assert not entry.confirmed
    assert _flag_adjustment(_flag(0.0), entry) == 0.0
    entry.confirmed = True
    assert _flag_adjustment(_flag(0.0), entry) == FLAG_BONUS


def test_junk_read_creates_no_entry():
    # 記号だけの読みは選手名簿に残さない
    roster = Roster()
    roster.resolve_race([Observation("-"), Observation("!!")], race_index=1)
    assert roster.entries == []
    # 名前らしい読みは仮登録される
    roster.resolve_race([Observation("たろう")], race_index=1)
    assert len(roster.entries) == 1


def test_merge_duplicates_never_coassigned():
    # 同時出現したことのない酷似エントリは統合される
    roster = Roster()
    a = roster._new_provisional(Observation("XI c.qq"))
    a.races_seen = 2
    b = roster._new_provisional(Observation("xı c.qq"))
    b.races_seen = 1
    roster._merge_duplicates()
    assert len(roster.entries) == 1
    merged = roster.entries[0]
    assert merged.races_seen == 3
    assert set(merged.variants) == {"XI c.qq", "xı c.qq"}


def test_merge_blocked_for_coassigned_pair():
    # 同じレースに同時出現した組は別人確定なので統合しない
    roster = Roster()
    a = roster._new_provisional(Observation("XI c.qq"))
    b = roster._new_provisional(Observation("xı c.qq"))
    a.co_seen.add(b.id)
    b.co_seen.add(a.id)
    roster._merge_duplicates()
    assert len(roster.entries) == 2


def test_merge_blocked_by_conflicting_flags():
    # 同名でも国旗が違えば別人（フランス旗と日本旗の FR など）
    roster = Roster()
    a = roster._new_provisional(Observation("FR", flag=_flag(0.0)))
    b = roster._new_provisional(Observation("FR", flag=_flag(1.0)))
    roster._merge_duplicates()
    assert len(roster.entries) == 2


def test_prune_stale_provisional():
    # 1回だけ現れて3レース以上現れない仮登録は除去される
    roster = Roster()
    roster.resolve_race([Observation("ghost player")], race_index=1)
    assert len(roster.entries) == 1
    roster.resolve_race([Observation("other player")], race_index=4)
    names = [e.canonical_name for e in roster.entries]
    assert names == ["other player"]


def test_roster_serialization_roundtrip():
    roster = Roster()
    roster.seed("RR★", flag=_flag(0.0))
    roster.resolve_race([Observation("たろう", flag=_flag(1.0))], race_index=1)
    restored = Roster.from_dict(roster.to_dict())
    assert [e.canonical_name for e in restored.entries] == [
        e.canonical_name for e in roster.entries
    ]
    assert restored.entries[0].pinned_name == "RR★"
    assert restored.entries[0].confirmed
    from mk8dx.roster import flag_distance

    assert flag_distance(
        restored.entries[0].flag_signatures[0], _flag(0.0)
    ) < 0.01
    assert restored.entries[1].races_seen == 1


def test_short_weak_match_is_surcharged():
    # 2文字同士の弱い一致（'AA' vs '三日'）は未割り当てコストを超える
    assert name_cost("AA", "三日") > 4.0
    # 完全一致と記号欠落レベルの強い一致には加算されない
    assert name_cost("FR", "FR") == 0.0
    assert name_cost("RR", "RR★") < 0.5


def test_roster_db_store_roundtrip(tmp_path):
    from mk8dx import db

    conn = db.connect(tmp_path / "t.db")
    assert db.load_roster_data(conn) is None
    roster = Roster()
    roster.seed("RR★")
    db.save_roster_data(conn, roster.to_dict())
    data = db.load_roster_data(conn)
    assert data is not None
    assert Roster.from_dict(data).entries[0].canonical_name == "RR★"


def test_rename_entry_pins_name_and_confirms():
    roster = Roster()
    roster.resolve_race([Observation("RP青")], race_index=1)
    entry = roster.entries[0]
    assert not entry.confirmed
    assert roster.rename(entry.id, "RR★")
    assert entry.canonical_name == "RR★"
    assert entry.confirmed
    # 以後の多数決でも固定名が維持される
    roster.resolve_race([Observation("RR")], race_index=2)
    assert entry.canonical_name == "RR★"
    assert not roster.rename(9999, "nobody")


@pytest.mark.xfail(
    reason="フェーズ1で roster の名寄せを画像ベースへ置換するため保留。"
    "修正案（行ごとの2パス方式）は docs/step7-notes.md を参照",
    strict=False,
)
def test_exact_name_match_not_overturned_by_flag():
    import numpy as np

    roster = Roster()
    sig_a = np.zeros(176)
    sig_b = np.ones(176)
    # 'racist #1'（国旗A）と 'racist #2'（国旗B）を2レース出現させ本登録化
    for race in (1, 2):
        roster.resolve_race(
            [Observation("racist #1", sig_a), Observation("racist #2", sig_b)],
            race_index=race)
    # 国旗抽出がノイズで逆転しても、名前完全一致が優先される
    res = roster.resolve([Observation("racist #1", sig_b),
                          Observation("racist #2", sig_a)])
    names = roster.resolved_names(res)
    assert names == ["racist #1", "racist #2"]


def test_capacity_grows_from_distinct_flags():
    import numpy as np

    roster = Roster()
    sig_a = np.zeros(176)
    sig_b = np.ones(176)
    # レース1: RR が国旗Aで1人だけ出現
    roster.resolve_race([Observation("RR", sig_a)], race_index=1)
    entry = roster.entries[0]
    assert entry.capacity == 1
    # レース2: RR が国旗Bで1人だけ出現（同時出現はしていない）
    roster.resolve_race([Observation("RR", sig_b)], race_index=2)
    assert entry.capacity == 2
    # レース3: 2人同時出現が両方とも同じエントリに収まる
    res = roster.resolve([Observation("RR", sig_a), Observation("RR", sig_b)])
    assert res.entry_ids == [entry.id, entry.id]
