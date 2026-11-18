from mk8dx.clustering import assign_teams


def names_2v2():
    # 6チーム x 2名（順位順にシャッフルされた並び）
    return [
        "あきら", "Xたろう", "あかり", "べーた1", "Xじろう", "べーた2",
        "みく", "Zed1", "みき", "Zed2", "Qoo1", "Qoo2",
    ]


def test_2v2_basic():
    result = assign_teams(names_2v2(), "2v2")
    assert result.valid, result.warnings
    assert len(result.teams) == 6
    assert all(len(t.members) == 2 for t in result.teams)


def test_3v3_basic():
    names = [
        "あ1", "い1", "う1", "え1",
        "あ2", "い2", "う2", "え2",
        "あ3", "い3", "う3", "え3",
    ]
    result = assign_teams(names, "3v3")
    assert result.valid, result.warnings
    assert len(result.teams) == 4
    assert all(len(t.members) == 3 for t in result.teams)


def test_shared_first_letter_split_by_second():
    # 先頭文字が同じ2チーム（あき…/あお…）は2文字目で分かれる
    names = [
        "あき1", "あお1", "か1", "き1", "く1", "け1",
        "あき2", "あお2", "か2", "き2", "く2", "け2",
    ]
    result = assign_teams(names, "2v2")
    assert result.valid, result.warnings
    assert len(result.teams) == 6


def test_dakuten_teams_not_merged():
    # 「た」始まりと「だ」始まりは別チーム
    names = [
        "た1", "だ1", "か1", "き1", "く1", "け1",
        "た2", "だ2", "か2", "き2", "く2", "け2",
    ]
    result = assign_teams(names, "2v2")
    assert result.valid, result.warnings
    tags = {t.normalized_tag for t in result.teams}
    assert "タ" in tags and "ダ" in tags


def test_inconsistent_warns_and_falls_back():
    # 3名同じ先頭文字（2v2ではあり得ない）-> 不整合として警告
    names = [
        "あ1", "あ2", "あ3", "か1", "き1", "く1",
        "け1", "こ1", "か2", "き2", "く2", "け2",
    ]
    result = assign_teams(names, "2v2")
    assert not result.valid
    assert result.warnings
