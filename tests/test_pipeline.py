"""OCRテキスト解釈と累計クロスチェックのテスト（画像・OCRエンジン不要の部分）。"""
from mk8dx.ocr import choose_name_candidate, parse_points_text
from mk8dx.pipeline import (
    PHASE_STANDINGS,
    PHASE_UNKNOWN,
    StandingsReading,
    classify_plain_phase,
    cross_check_standings,
)


def test_parse_plus_points():
    assert parse_points_text("+15") == (15, True)
    assert parse_points_text("＋9") == (9, True)


def test_parse_plus_lookalikes():
    # 動画圧縮で '+' が 'ト'/'十' に化けるケース（スコア列前提）
    assert parse_points_text("ト15") == (15, True)
    assert parse_points_text("十目") == (None, True)


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


def test_plain_phase_points_table_descending_is_standings():
    # 1レース目直後のフェーズB: 累計=配点表がそのまま降順に並ぶ
    totals = [15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert classify_plain_phase(totals) == PHASE_STANDINGS


def test_plain_phase_partial_points_descending_is_standings():
    # セグメント数字が一部読めなくても、配点の部分集合が降順ならBとみなす
    totals = [15, 12, 10, 9, None, None, None, None, 4, 3, None, None]
    assert classify_plain_phase(totals) == PHASE_STANDINGS


def test_plain_phase_sorted_cumulative_is_standings():
    totals = [46, 41, 39, 37, 36, 36, 34, 32, 31, 30, 27, 21]
    assert classify_plain_phase(totals) == PHASE_STANDINGS


def test_plain_phase_unsorted_is_unknown():
    # フェーズBの並び替えアニメーション途中 / フェーズA後半は決められない
    totals = [31, 16, 23, 22, 24, 18, 21, 21, None, None, 29, None]
    assert classify_plain_phase(totals) == PHASE_UNKNOWN


def test_plain_phase_too_few_readable_is_unknown():
    totals = [15, None, None, None, None, None, None, None, None, None, None, None]
    assert classify_plain_phase(totals) == PHASE_UNKNOWN


def test_choose_name_candidate_japanese():
    jp = ("A★ぜんだかいとしき", 0.9)
    en = ("AXENpLba", 0.95)
    assert choose_name_candidate(jp, en) == jp


def test_choose_name_candidate_mixed_kana():
    jp = ("Xb◇ねえあまぁƄ", 0.8)
    en = ("Xbb", 0.99)
    assert choose_name_candidate(jp, en) == jp


def test_choose_name_candidate_latin_prefers_confidence():
    # jp が 'Xb' をカタカナ2文字 'メロ' と誤読したケースは en を選ぶ
    jp = ("メロ", 0.7)
    en = ("Xb", 0.99)
    assert choose_name_candidate(jp, en) == en


def test_choose_name_candidate_empty_side():
    assert choose_name_candidate(("", 0.0), ("FR", 0.9)) == ("FR", 0.9)
    assert choose_name_candidate(("FR", 0.9), ("", 0.0)) == ("FR", 0.9)
