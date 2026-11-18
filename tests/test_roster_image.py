"""roster.py の画像ベース照合（文字マスク）のテスト。合成画像を使う。"""
import cv2
import numpy as np

from mk8dx.identity import extract_name_mask, trim_mask
from mk8dx.roster import Observation, Roster, RosterEntry


def _mask(text: str, bg: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.clip(
        np.full((72, 525, 3), bg, int) + rng.integers(-15, 15, (72, 525, 3)),
        0, 255,
    ).astype(np.uint8)
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                (255, 255, 255), 2)
    return trim_mask(extract_name_mask(img))


TEXTS = ["xi c.qq", "FR", "racist 1", "Xb", "RR*", "avalon"]


def _obs(texts, names=None, bg=40, seed=0):
    return [
        Observation(name=(names[i] if names else ""), mask=_mask(t, bg, seed + i))
        for i, t in enumerate(texts)
    ]


def test_same_identity_survives_garbled_ocr_names():
    roster = Roster()
    roster.resolve_race(_obs(TEXTS, names=TEXTS), race_index=1)
    ids1 = [e.id for e in roster.entries]
    # 2レース目: OCR名は全部ゴミ読み、順位も入れ替わり、背景も違う
    shuffled = list(reversed(TEXTS))
    res = roster.resolve(_obs(shuffled, names=["#?"] * 6, bg=75, seed=50))
    expected = [ids1[TEXTS.index(t)] for t in shuffled]
    assert res.entry_ids == expected
    assert not res.new_candidates


def test_new_player_gets_new_entry_even_with_empty_name():
    roster = Roster()
    roster.resolve_race(_obs(TEXTS[:5], names=TEXTS[:5]), race_index=1)
    n_before = len(roster.entries)
    res = roster.resolve(_obs(TEXTS[:5] + ["avalon"], names=[""] * 6))
    assert res.new_candidates == [5]
    roster.learn(_obs(TEXTS[:5] + ["avalon"], names=[""] * 6), res, 2)
    assert len(roster.entries) == n_before + 1
    # 名前が読めなかった新規は 選手N の仮名になる
    assert roster.entries[-1].canonical_name.startswith("選手")


def test_provisional_duplicates_merge_by_image():
    roster = Roster()
    # 別々のレースで同じ人が別の読みで登録された状況
    e1 = roster._new_provisional(Observation("RR*", mask=_mask("RR*")))
    e1.races_seen = e1.last_seen_race = 1
    e2 = roster._new_provisional(
        Observation("日日計", mask=_mask("RR*", bg=70, seed=9)))
    e2.races_seen = e2.last_seen_race = 2
    roster._merge_duplicates()
    assert len(roster.entries) == 1


def test_masks_survive_persistence():
    roster = Roster()
    roster.resolve_race(_obs(TEXTS, names=TEXTS), race_index=1)
    restored = Roster.from_dict(roster.to_dict())
    res = restored.resolve(_obs(TEXTS, names=[""] * 6, bg=60, seed=20))
    assert res.entry_ids == [e.id for e in restored.entries]
    assert isinstance(restored.entries[0], RosterEntry)


def test_without_masks_falls_back_to_name_matching():
    roster = Roster()
    roster.seed("FR")
    res = roster.resolve([Observation("FR")])
    assert res.entry_ids == [roster.entries[0].id]
