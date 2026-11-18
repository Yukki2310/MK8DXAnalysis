"""Screen AI エンジンとハイブリッドのテスト（locro はモック）。

実際の locro / Screen AI コンポーネントはテスト環境に無いため、
sys.modules に偽の locro を差し込んで検証する。
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from mk8dx.screenai import HybridOcrEngine, ScreenAiEngine, create_ocr_engine


def _page(words_with_conf):
    words = [SimpleNamespace(text=t, confidence=c) for t, c in words_with_conf]
    line = SimpleNamespace(text=" ".join(t for t, _ in words_with_conf),
                           words=words, bounding_box=None)
    return SimpleNamespace(blocks=[SimpleNamespace(lines=[line])])


class FakeScreenAI:
    """locro.ScreenAI の代わり。受け取った画像サイズを記録する。"""

    instances = []

    def __init__(self):
        self.seen_sizes = []
        FakeScreenAI.instances.append(self)

    def ocr_pil_image(self, img):
        self.seen_sizes.append(img.size)
        return _page([("A★", 0.9), ("ぜんだかいとしき", 0.8)])


@pytest.fixture
def fake_locro(monkeypatch):
    FakeScreenAI.instances = []
    module = types.ModuleType("locro")
    module.ScreenAI = FakeScreenAI
    monkeypatch.setitem(sys.modules, "locro", module)
    return module


def test_screenai_recognize_name_upscales_and_joins(fake_locro):
    engine = ScreenAiEngine()
    assert engine.load()
    image = np.zeros((72, 525, 3), dtype=np.uint8)
    text, conf = engine.recognize_name(image)
    assert text == "A★ ぜんだかいとしき"
    assert conf == pytest.approx(0.85)
    # 2倍拡大されてから渡っている
    assert FakeScreenAI.instances[0].seen_sizes == [(1050, 144)]


def test_screenai_unavailable_without_load():
    engine = ScreenAiEngine()
    assert not engine.available
    assert engine.recognize_name(np.zeros((10, 10, 3), np.uint8)) == ("", 0.0)


class FakePaddle:
    def recognize_name(self, image):
        return "PADDLE", 0.7

    def recognize_total(self, image):
        return 42


def test_hybrid_prefers_screenai(fake_locro):
    screenai = ScreenAiEngine()
    assert screenai.load()
    hybrid = HybridOcrEngine(FakePaddle(), screenai)  # type: ignore[arg-type]
    assert hybrid.using_screenai
    text, _ = hybrid.recognize_name(np.zeros((72, 525, 3), np.uint8))
    assert text == "A★ ぜんだかいとしき"


def test_hybrid_falls_back_on_empty(fake_locro, monkeypatch):
    class EmptyAI(FakeScreenAI):
        def ocr_pil_image(self, img):
            return _page([])

    fake_locro.ScreenAI = EmptyAI
    screenai = ScreenAiEngine()
    assert screenai.load()
    hybrid = HybridOcrEngine(FakePaddle(), screenai)  # type: ignore[arg-type]
    assert hybrid.recognize_name(np.zeros((10, 10, 3), np.uint8)) == ("PADDLE", 0.7)


def test_hybrid_falls_back_on_error(fake_locro):
    class BoomAI(FakeScreenAI):
        def ocr_pil_image(self, img):
            raise RuntimeError("boom")

    fake_locro.ScreenAI = BoomAI
    screenai = ScreenAiEngine()
    assert screenai.load()
    hybrid = HybridOcrEngine(FakePaddle(), screenai)  # type: ignore[arg-type]
    assert hybrid.recognize_name(np.zeros((10, 10, 3), np.uint8)) == ("PADDLE", 0.7)


def test_hybrid_delegates_digits_to_paddle(fake_locro):
    screenai = ScreenAiEngine()
    assert screenai.load()
    hybrid = HybridOcrEngine(FakePaddle(), screenai)  # type: ignore[arg-type]
    assert hybrid.recognize_total(np.zeros((10, 10, 3), np.uint8)) == 42


def test_create_engine_without_locro(monkeypatch):
    monkeypatch.setitem(sys.modules, "locro", None)
    hybrid = create_ocr_engine()
    assert not hybrid.using_screenai
