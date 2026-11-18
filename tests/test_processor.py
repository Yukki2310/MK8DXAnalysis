"""processor.py（レース処理の統合）のテスト。

名前リストはステップ2の実データ（ラウンジ動画のOCR読みと正規名）を
そのまま使う。
"""
from mk8dx.processor import RaceProcessor
from mk8dx.roster import Observation, Roster

CANONICAL = [
    "xı c.qq", "xı avalon", "FR", "FR Flowx", "racist #1", "racist #2",
    "Xb", "Xb◇ねえあまぁƄ", "RR★", "RR★",
    "A★ぜんだかいとしき", "A A",
]
# レース3の着順に並べた正規名
R3_ORDER = [
    "xı c.qq", "FR", "racist #1", "Xb", "RR★", "FR Flowx",
    "A★ぜんだかいとしき", "A A", "xı avalon", "Xb◇ねえあまぁƄ",
    "RR★", "racist #2",
]
# レース3のOCR生読み（ステップ2の実測）
R3_READS = [
    "XI c.qq", "FR", "racist #1", "Xb", "RR", "FRFlowx",
    "Aかぜんだかいとしき", "AA", "XIavalon", "今ねえあまあ!", "日日-", "-",
]
# レース1のOCR生読み（選手名簿外の選手が多い）
R1_READS = [
    "xi c.qq", "RR*", "Rev", "ths", "racist #2", "A大ぜんだかいとしき",
    "FR", "メ念れえあまあ", "Xb", "racist #7", "三日", "xavalon",
]
EXPECTED_R3_POINTS = {"xı": 19, "f": 19, "ra": 11, "xb": 12, "rr": 10, "a": 11}


def _seeded_roster() -> Roster:
    roster = Roster()
    for name in CANONICAL:
        roster.seed(name)
    return roster


def _obs(names: list[str]) -> list[Observation]:
    return [Observation(name=n) for n in names]


def test_clean_race_confirmed():
    proc = RaceProcessor(_seeded_roster(), "2v2", assume_closed=True)
    outcome = proc.process_single(_obs(R3_ORDER), race_index=3)
    assert outcome.confirmed
    assert outcome.problems == []
    points = {t["normalized_tag"]: t["points"] for t in outcome.teams}
    assert points == EXPECTED_R3_POINTS
    assert sum(points.values()) == 82


def test_unreadable_rows_are_decided_by_elimination_without_review():
    proc = RaceProcessor(_seeded_roster(), "2v2", assume_closed=True)
    outcome = proc.process_single(_obs(R3_READS), race_index=3)
    # 化け読みの行は名簿の空きから決める。確認は求めず、お知らせだけにする
    assert outcome.confirmed
    assert any("前のレースの選手" in w for w in outcome.warnings)
    points = {t["normalized_tag"]: t["points"] for t in outcome.teams}
    assert points == EXPECTED_R3_POINTS


def test_auto_elimination_only_among_previous_race_players():
    """前のレースの選手が1人見つからないときだけ、その人だと決める。"""
    roster = _seeded_roster()
    proc = RaceProcessor(roster, "2v2", auto_eliminate=True)
    first = proc.process_single(_obs(R3_ORDER), race_index=1)
    assert first.confirmed
    garbled = list(R3_ORDER)
    garbled[11] = "レースを自動終了しました"       # 12位の名前に重なった表示
    second = proc.process_single(_obs(garbled), race_index=2)
    assert second.confirmed
    assert second.entries[11]["resolved_name"] == R3_ORDER[11]
    assert second.entries[11]["low_confidence"]
    # 3人以上が読めないときは推測しない（新しい選手が来た可能性）
    many = list(R3_ORDER)
    many[9], many[10], many[11] = "ぜぜぜぜぜぜぜ", "ヴヴヴヴヴヴヴ", "ゐゐゐゐゐゐゐ"
    third = proc.process_single(_obs(many), race_index=3)
    assert not any(e["low_confidence"] for e in third.entries)


def test_stability_and_majority_across_frames():
    proc = RaceProcessor(
        _seeded_roster(), "2v2", assume_closed=True,
        min_frames=2, stable_frames=2,
    )
    proc.begin_race(3)
    assert proc.add_frame(_obs(R3_READS)) == "accumulating"
    # 安定判定は「試し照合の割り当てが連続で同じ」こと。生読みが
    # 揺れても割り当てが安定していれば ready になる（この実データでは
    # ノイズのある1フレーム目から割り当ては正しく安定した）
    assert proc.add_frame(_obs(R3_ORDER)) == "ready"
    outcome = proc.finalize()
    assert outcome.confirmed
    resolved = [e["resolved_name"] for e in outcome.entries]
    assert resolved == R3_ORDER


def test_consensus_majority_prefers_frequent_and_recent():
    proc = RaceProcessor(
        _seeded_roster(), "2v2", assume_closed=True,
        min_frames=99, stable_frames=99,
    )
    proc.begin_race(3)
    proc.add_frame(_obs(R3_READS))
    proc.add_frame(_obs(R3_ORDER))
    # 1対1の同数では最後に観測された表記（クリーンな読み）が勝つ
    assert [o.name for o in proc.consensus()] == R3_ORDER
    proc.add_frame(_obs(R3_READS))
    # 2対1になれば多数派（ノイズ側でも）が勝つ
    assert [o.name for o in proc.consensus()] == R3_READS


def test_open_mode_unresolved_goes_to_review():
    proc = RaceProcessor(_seeded_roster(), "2v2", assume_closed=False)
    outcome = proc.process_single(_obs(R1_READS), race_index=1)
    assert not outcome.confirmed
    # 名簿にない読みは新規登録され、再照合で名前は引けるようになるが、
    # チーム割り当てが成立しないため集計はされず要確認に回る
    assert outcome.teams == []
    assert any("初めて見る選手" in p for p in outcome.warnings)
