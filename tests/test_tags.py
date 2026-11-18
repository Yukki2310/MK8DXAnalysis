from mk8dx.tags import (
    common_prefix,
    first_char_key,
    has_forbidden_diacritic,
    normalize_name,
)


def test_case_insensitive():
    assert normalize_name("ABC") == normalize_name("abc")


def test_hiragana_katakana_unified():
    assert normalize_name("あいう") == normalize_name("アイウ")


def test_dakuten_distinguished():
    # 濁音は清音と別文字として区別される（ラウンジルール4）
    assert normalize_name("た") != normalize_name("だ")
    assert normalize_name("は") != normalize_name("ば") != normalize_name("ぱ")


def test_decomposed_dakuten_composed():
    # 結合文字（か + 結合濁点）も「が」1文字にそろう
    assert normalize_name("が") == normalize_name("が")


def test_halfwidth_kana_unified():
    assert normalize_name("ｱｲｳ") == normalize_name("アイウ")


def test_first_char_key():
    assert first_char_key("あbc") == "ア"
    assert first_char_key("Xyz") == "x"


def test_forbidden_diacritic_detected():
    assert has_forbidden_diacritic("Äbc")
    assert has_forbidden_diacritic("café")
    # 日本語の濁点は対象外
    assert not has_forbidden_diacritic("がんばる")
    assert not has_forbidden_diacritic("abc")


def test_common_prefix():
    assert common_prefix(["あきら", "あきこ"]) == "アキ"
    assert common_prefix(["abc", "abd"]) == "ab"
    assert common_prefix(["abc", "xyz"]) == ""
