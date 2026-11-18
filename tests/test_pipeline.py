"""OCRテキスト解釈と累計クロスチェックのテスト（画像・OCRエンジン不要の部分）。"""
from mk8dx.ocr import parse_points_text
from mk8dx.pipeline import StandingsReading, cross_check_standings


def test_parse_plus_points():
    assert parse_points_text("+15") == (15, True)
    assert parse_points_text("＋9") == (9, True)


def test_parse_plain_total_is_not_race_points():
    # 累計画面の '15' は '+' がない。値としては読めるがフェーズ判別で区別する
    value, has_plus = parse_points_text("15")
    assert value == 15
    assert has_plus is False


def test_parse_invalid_score():
    assert parse_points_text("+11") == (None, True)  # 規定配点にない
    assert parse_points_text("") == (None, False)
    assert parse_points_text("abc") == (None, False)


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
    problems = cross_check_standings(computed, standings)
    assert problems


def test_cross_check_order_detected():
    computed = {"あきら": 27, "たろう": 20}
    standings = StandingsReading(entries=[
        {"position": 1, "name": "たろう", "total": 20},  # 並びが逆
        {"position": 2, "name": "あきら", "total": 27},
    ])
    problems = cross_check_standings(computed, standings)
    assert any("並び" in p for p in problems)
