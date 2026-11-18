"""累計クロスチェックとフェーズ判別のテスト（画像不要の部分）。"""
from mk8dx.pipeline import (
    StandingsReading,
    cross_check_standings,
    looks_like_standings,
)


def test_cross_check_ok():
    computed = {"あきら": 27, "たろう": 20}
    standings = StandingsReading(entries=[
        {"position": 1, "name": "あきら", "total": 27},
        {"position": 2, "name": "たろう", "total": 20},
    ])
    assert cross_check_standings(computed, standings) == []


def test_cross_check_mismatch_detected():
    computed = {"あきら": 27, "たろう": 20}
    standings = StandingsReading(entries=[
        {"position": 1, "name": "あきら", "total": 25},  # 累計が違う
        {"position": 2, "name": "たろう", "total": 20},
    ])
    assert cross_check_standings(computed, standings)


def test_cross_check_order_detected():
    computed = {"あきら": 27, "たろう": 20}
    standings = StandingsReading(entries=[
        {"position": 1, "name": "たろう", "total": 20},  # 並びが逆
        {"position": 2, "name": "あきら", "total": 27},
    ])
    problems = cross_check_standings(computed, standings)
    assert any("並び" in p for p in problems)


def test_cross_check_total_not_read_still_checks_order():
    computed = {"あきら": 27, "たろう": 20}
    standings = StandingsReading(entries=[
        {"position": 1, "name": "あきら", "total": None},
        {"position": 2, "name": "たろう", "total": None},
    ])
    assert cross_check_standings(computed, standings) == []


def test_standings_detected_when_totals_descend():
    assert looks_like_standings([31, 29, 24, 23, 22, 21, 21, 18, 17, 16, 13, 11])


def test_race_order_is_not_standings():
    # ゴール順画面: 行順の自前累計は一般に降順にならない
    assert not looks_like_standings([36, 25, 31, 27, 19, 30, 37, 27, 21, 27, 18, 30])


def test_too_few_known_rows_is_not_standings():
    assert not looks_like_standings([31, 29, None, None, None, None, None, 18, None, None, None, None])
