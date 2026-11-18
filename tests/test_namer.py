"""namer.py（表示名の整形・候補選択）と processor の名付けのテスト。"""
import cv2
import numpy as np

from mk8dx.namer import NameReader, choose_candidate, clean_text, prepare_for_ocr
from mk8dx.processor import RaceProcessor
from mk8dx.roster import Observation, Roster


def test_clean_text_fixes_common_ocr_artifacts():
    assert clean_text("RR*") == "RR★"
    assert clean_text("racist # 2 |") == "racist #2"
    assert clean_text("A ☆ せ ん だ か い") == "A☆せんだかい"
    assert clean_text("FR   Flowx") == "FR Flowx"


def test_choose_candidate_prefers_japanese_with_kana():
    assert choose_candidate(["A ☆ せ ん", "A"]) == "A☆せん"


def test_choose_candidate_latin_uses_english_engine():
    # 日本語エンジンが 'xı' を漢字に誤読しても、仮名が無ければ英語側を採る
    assert choose_candidate(["刈 c.", "Xl c.qq"]) == "Xl c.qq"


def test_choose_candidate_kanji_only_falls_back_to_japanese():
    assert choose_candidate(["三日", ""]) == "三日"


def test_prepare_for_ocr_none_for_blank():
    assert prepare_for_ocr(np.full((72, 525, 3), 30, np.uint8)) is None


def test_prepare_for_ocr_returns_white_background_image():
    img = np.full((72, 525, 3), 40, np.uint8)
    cv2.putText(img, "FR", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(img, "FR", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    out = prepare_for_ocr(img)
    assert out is not None
    assert out[0, 0].tolist() == [255, 255, 255]  # 余白は白


class FakeOcr:
    available = True

    def __init__(self, texts):
        self.texts = list(texts)

    def recognize(self, image):
        return [self.texts.pop(0), ""]


def test_name_reader_majority_vote():
    img = np.full((72, 525, 3), 40, np.uint8)
    cv2.putText(img, "FR", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(img, "FR", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    reader = NameReader(FakeOcr(["FRx", "FR", "FR"]))
    assert reader.read([img, img, img]) == "FR"


def test_name_reader_unavailable_returns_empty():
    class Off:
        available = False

    assert NameReader(Off()).read([np.zeros((72, 525, 3), np.uint8)]) == ""


def _text_image(text):
    img = np.full((72, 525, 3), 40, np.uint8)
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    return img


def test_processor_names_new_players_with_namer():
    from mk8dx.identity import extract_name_mask, trim_mask

    texts = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliet", "kilo", "lima"]
    imgs = [_text_image(t) for t in texts]
    obs = [Observation("", mask=trim_mask(extract_name_mask(i)), image=i)
           for i in imgs]
    calls = []

    def namer(crops):
        calls.append(len(crops))
        return f"name{len(calls)}"

    roster = Roster()
    proc = RaceProcessor(roster, "2v2", namer=namer)
    proc.begin_race(1)
    proc.add_frame(obs)
    outcome = proc.finalize(force=True)
    assert len(calls) == 12
    assert all(e["resolved_name"].startswith("name") for e in outcome.entries)
    # 2レース目: 読み取り回数の上限まで再度読み、以後は読まない
    for race in (2, 3, 4):
        proc.begin_race(race)
        proc.add_frame(obs)
        proc.finalize(force=True)
    assert len(calls) == 12 * 3
