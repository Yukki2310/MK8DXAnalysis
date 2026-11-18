"""チーム設定・順位表（動的集計）・統合・取り消しのテスト。"""
import numpy as np
import pytest

from mk8dx import db, standings
from mk8dx.processor import RaceProcessor
from mk8dx.roster import Observation, Roster
from mk8dx.teams import group_by_labels

POINTS = [15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


def _rows(entry_ids_by_race):
    rows = []
    for race, ids in enumerate(entry_ids_by_race, start=1):
        for rank, eid in enumerate(ids, start=1):
            rows.append({"race_number": race, "rank": rank, "score": POINTS[rank - 1],
                         "entry_id": eid, "player_name": ""})
    return rows


def _roster_with_teams(teams):
    roster = Roster()
    for i, team in enumerate(teams, start=1):
        e = roster.seed(f"選手{i}")
        roster.set_team(e.id, team)
    return roster


TEAMS_2V2 = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F"]


def test_group_by_labels_valid_and_invalid():
    ok = group_by_labels(TEAMS_2V2, "2v2")
    assert ok.valid and len(ok.teams) == 6
    bad = group_by_labels(["A"] * 3 + ["B"] + TEAMS_2V2[4:], "2v2")
    assert not bad.valid
    assert any("名" in w for w in bad.warnings)
    missing = group_by_labels([None] + TEAMS_2V2[1:], "2v2")
    assert not missing.valid
    assert any("未設定" in w for w in missing.warnings)


def test_group_by_labels_ignores_case_and_kana():
    labels = ["ab", "AB", "かな", "カナ"] + ["x", "x", "y", "y", "z", "z", "w", "w"]
    result = group_by_labels(labels, "2v2")
    assert result.valid


def test_standings_total_is_82_per_race_and_ranks():
    roster = _roster_with_teams(TEAMS_2V2)
    rows = _rows([list(range(1, 13)), list(reversed(range(1, 13)))])
    table = standings.compute(rows, roster)
    assert sum(t["total"] for t in table) == 82 * 2
    assert [t["rank"] for t in table][0] == 1
    assert table[0]["diff_from_top"] == 0
    assert all(t["diff_from_top"] <= 0 for t in table)


def test_standings_follow_team_changes_retroactively():
    roster = _roster_with_teams(TEAMS_2V2)
    rows = _rows([list(range(1, 13))])
    before = {t["tag"]: t["total"] for t in standings.compute(rows, roster)}
    assert before["A"] == 15 + 12
    # 選手2を B に移すと、過去のレースの点も B へ移る
    roster.set_team(2, "B")
    after = {t["tag"]: t["total"] for t in standings.compute(rows, roster)}
    assert after["A"] == 15 and after["B"] == 10 + 9 + 12


def test_standings_ties_share_rank_and_skip_unassigned():
    roster = _roster_with_teams(TEAMS_2V2)
    roster.set_team(12, None)  # チーム未設定の選手は集計に含めない
    table = standings.compute(_rows([list(range(1, 13))]), roster)
    assert sum(t["total"] for t in table) == 82 - 1
    tied = standings.compute(
        [{"race_number": 1, "rank": 1, "score": 5, "entry_id": 1, "player_name": ""},
         {"race_number": 1, "rank": 2, "score": 5, "entry_id": 3, "player_name": ""}],
        roster)
    assert [t["rank"] for t in tied] == [1, 1]


def test_set_team_colors_are_stable_and_distinct():
    roster = _roster_with_teams(TEAMS_2V2)
    colors = [roster.color_for(t) for t in ["A", "B", "C", "D", "E", "F"]]
    assert len(set(colors)) == 6
    assert roster.color_for("a") == colors[0]  # 大小文字は同じチーム
    restored = Roster.from_dict(roster.to_dict())
    assert restored.color_for("A") == colors[0]


def test_team_persists_through_roundtrip():
    roster = _roster_with_teams(TEAMS_2V2)
    restored = Roster.from_dict(roster.to_dict())
    assert [e.team for e in restored.entries] == TEAMS_2V2
    assert all(e.team_manual for e in restored.entries)


def test_merge_combines_entries_and_team():
    roster = Roster()
    a, b = roster.seed("x"), roster.seed("y")
    roster.set_team(a.id, "T")
    assert roster.merge(a.id, b.id)
    assert [e.id for e in roster.entries] == [b.id]
    assert roster.entries[0].team == "T"
    assert not roster.merge(999, b.id)


def test_mask_png_for_entry_with_mask():
    roster = Roster()
    e = roster.seed("x")
    assert roster.mask_png(e.id) is None
    e.add_mask(np.ones((20, 60), np.float32))
    png = roster.mask_png(e.id)
    assert png and png[:4] == b"\x89PNG"


def test_processor_uses_manual_labels_over_names():
    """名前の先頭文字がバラバラでも、チームが設定済みなら確定する。"""
    roster = Roster()
    names = [f"{chr(97 + i)}name" for i in range(12)]  # 先頭文字がすべて違う
    for i, n in enumerate(names):
        e = roster.seed(n)
        roster.set_team(e.id, TEAMS_2V2[i])
    proc = RaceProcessor(roster, "2v2", assume_closed=True)
    outcome = proc.process_single([Observation(name=n) for n in names], 1)
    assert outcome.confirmed
    assert sum(t["points"] for t in outcome.teams) == 82
    assert proc.team_problems([e["entry_id"] for e in outcome.entries]) == []
    # チームを壊すと、確定できない理由が分かる
    roster.set_team(1, None)
    problems = proc.team_problems([e["entry_id"] for e in outcome.entries])
    assert problems and any("未設定" in p for p in problems)


def test_processor_auto_labels_from_tags_for_new_teams():
    roster = Roster()
    names = ["xa", "xb", "ya", "yb", "za", "zb", "wa", "wb", "va", "vb", "ua", "ub"]
    for n in names:
        roster.seed(n)
    proc = RaceProcessor(roster, "2v2", assume_closed=True)
    outcome = proc.process_single([Observation(name=n) for n in names], 1)
    assert outcome.confirmed
    assert all(e.team and not e.team_manual for e in roster.entries)


def test_db_migration_and_valid_rows(tmp_path):
    path = tmp_path / "old.db"
    import sqlite3

    # entry_id 列のない旧スキーマ
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE player_scores (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " race_result_id INTEGER NOT NULL, rank INTEGER NOT NULL,"
        " player_name TEXT NOT NULL, team_id INTEGER, score INTEGER NOT NULL,"
        " raw_ocr_text TEXT);")
    conn.commit()
    conn.close()
    conn = db.connect(path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(player_scores)")}
    assert "entry_id" in cols
    sid = db.create_session(conn, "2v2")
    rid = db.save_race_result(conn, sid, 1, [
        {"rank": r, "player_name": "p", "team_id": None, "score": POINTS[r - 1],
         "entry_id": r} for r in range(1, 13)], is_valid=True)
    pending = db.save_race_result(conn, sid, 2, [
        {"rank": 1, "player_name": "p", "team_id": None, "score": 15, "entry_id": 1}],
        is_valid=False)
    rows = db.valid_rows(conn, sid)
    assert len(rows) == 12 and {r["race_result_id"] for r in rows} == {rid}
    assert db.last_valid_race_result(conn, sid) == rid
    latest = db.latest_session(conn)
    assert latest["id"] == sid and latest["completed"] == 1
    db.delete_race_result(conn, pending)


def _name_image(text):
    import cv2

    img = np.full((72, 525, 3), 40, np.uint8)
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 7)
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
    return img


def test_suggest_teams_from_leading_glyphs_even_when_ocr_text_is_wrong():
    from mk8dx.teams import suggest_by_prefix

    tags = ["AB", "CD", "EF", "GH", "IJ", "KL"]
    truth = []
    for t in tags:
        truth += [f"{t} one", f"{t}two"]
    # 並びを入れ替え（ゴール順は毎回違う）
    order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    images = [_name_image(truth[i]) for i in order]
    garbled = ["?"] * 12   # 文字認識が全く読めなかった
    result = suggest_by_prefix(images, garbled, "2v2")
    assert result is not None and result.valid
    found = {frozenset(order[m] for m in t.members) for t in result.teams}
    assert found == {frozenset((2 * k, 2 * k + 1)) for k in range(6)}


def test_suggest_teams_returns_none_when_prefixes_do_not_pair_up():
    from mk8dx.teams import suggest_by_prefix

    words = ["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ", "KK", "LL"]
    images = [_name_image(w) for w in words]
    assert suggest_by_prefix(images, [""] * 12, "2v2") is None


def test_suggest_teams_uses_ocr_prefix_for_case_differences():
    from mk8dx.teams import suggest_by_prefix

    # 大文字小文字の違いは画像では一致しないが、ラウンジの規則では同じタグ
    names = ["ab one", "AB two", "cd x", "CD y", "ef x", "EF y",
             "gh x", "GH y", "ij x", "IJ y", "kl x", "KL y"]
    images = [_name_image(n) for n in names]
    result = suggest_by_prefix(images, names, "2v2")
    assert result is not None
    assert {tuple(t.members) for t in result.teams} == {(2 * k, 2 * k + 1) for k in range(6)}
