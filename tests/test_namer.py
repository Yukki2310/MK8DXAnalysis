"""namer.py（表示名の整形・多数決）と processor の名付け、roster の見本画像のテスト。"""
import cv2
import numpy as np

from mk8dx.identity import extract_name_mask, trim_mask
from mk8dx.namer import NameReader, clean_text
from mk8dx.processor import RaceProcessor
from mk8dx.roster import Observation, Roster, encode_thumb


def test_clean_text_fixes_common_ocr_artifacts():
    assert clean_text("RR*") == "RR★"
    assert clean_text("racist # 2 |") == "racist #2"
    assert clean_text("A ☆ せ ん だ か い") == "A☆せんだかい"
    assert clean_text("FR   Flowx") == "FR Flowx"


class FakeRecognizer:
    available = True

    def __init__(self, reads):
        self.reads = list(reads)

    def recognize_many(self, images):
        return [self.reads.pop(0) for _ in images]


def _img(text="FR"):
    img = np.full((72, 525, 3), 40, np.uint8)
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    return img


def test_name_reader_votes_by_confidence():
    reader = NameReader(FakeRecognizer([("FRx", 0.4), ("FR", 0.9), ("FR", 0.8)]))
    assert reader.read([_img(), _img(), _img()]) == "FR"
    # 信頼度の高い1票が、低い2票に勝つことがある
    reader = NameReader(FakeRecognizer([("A", 0.2), ("B", 0.9), ("A", 0.3)]))
    assert reader.read([_img(), _img(), _img()]) == "B"


def test_name_reader_unavailable_or_empty_returns_empty():
    class Off:
        available = False

    assert NameReader(Off()).read([_img()]) == ""
    assert NameReader(FakeRecognizer([("", 0.0)])).read([_img()]) == ""


def test_processor_names_new_players_with_namer():
    texts = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliet", "kilo", "lima"]
    imgs = [_img(t) for t in texts]
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
    # 2レース目以降: 読み取り回数の上限まで再度読み、以後は読まない
    for race in (2, 3, 4):
        proc.begin_race(race)
        proc.add_frame(obs)
        proc.finalize(force=True)
    assert len(calls) == 12 * 3


def test_roster_keeps_real_thumbnail_and_persists_it():
    img = _img("FR")
    roster = Roster()
    roster.resolve_race(
        [Observation("", mask=trim_mask(extract_name_mask(img)), image=img)], 1)
    entry = roster.entries[0]
    assert entry.thumb and entry.thumb[:2] == b"\xff\xd8"  # JPEG
    data, mime = roster.image_for(entry.id)
    assert mime == "image/jpeg" and data == entry.thumb
    restored = Roster.from_dict(roster.to_dict())
    assert restored.entries[0].thumb == entry.thumb


def test_image_falls_back_to_mask_png_without_thumb():
    roster = Roster()
    e = roster.seed("x")
    assert roster.image_for(e.id) is None
    e.add_mask(np.ones((20, 60), np.float32))
    data, mime = roster.image_for(e.id)
    assert mime == "image/png"


def test_encode_thumb_none():
    assert encode_thumb(None) is None


def test_clean_text_normalizes_fullwidth_and_cyrillic_lookalikes():
    assert clean_text("Xbねえあまあｂ") == "Xbねえあまあb"      # 全角英字 → 半角
    assert clean_text("Хб") == "Xb"                      # キリル文字の Х б
    assert clean_text("ＦＲ ＩＲＯＮＭＡＮ") == "FR IRONMAN"


def test_read_many_batches_and_respects_crops_per_read():
    class Rec:
        available = True
        crops_per_read = 1
        max_reads = 1
        name = "fake"

        def __init__(self):
            self.calls = []

        def recognize_many(self, images):
            self.calls.append(len(images))
            return [("N%d" % i, 1.0) for i in range(len(images))]

    rec = Rec()
    reader = NameReader(rec)
    out = reader.read_many([[_img(), _img(), _img()], [_img()], [_img(), _img()]])
    assert out == ["N0", "N1", "N2"]
    assert rec.calls == [3]            # 選手ごとに1枚だけ、全員まとめて1回で読む
    assert reader.max_reads == 1 and reader.engine_name == "fake"
