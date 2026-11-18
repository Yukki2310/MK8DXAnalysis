"""Chrome Screen AI（locro）エンジンとハイブリッド構成（ステップ7）。

背景（実測 2026-11-18、ユーザーの Windows PC でのベンチマーク）:
PaddleOCR は同じ名前をフレームごとに違う形に文字化けさせるため、
選手名簿が分裂・誤吸着し、自動構築でのチーム割り当てが崩れた。
Screen AI は厳密一致こそ同程度だが、読みがレースをまたいで安定して
おり（RR★→常に「RR」、A★ぜんだかいとしき は完全一致）、12名分を
約0.44秒で読む。名寄せには厳密さより安定性が効くため、名前読みの
主エンジンとして採用し、PaddleOCR をフォールバックに回す。

構成:
- ScreenAiEngine: locro の薄いラッパー。名前は2倍拡大で読む
  （ベンチで crop2x が最良だったため）。
- HybridOcrEngine: recognize_name だけ Screen AI を優先し、
  空読み・例外時は PaddleOCR にフォールバックする。数字系
  （順位・スコア・累計）は従来どおり PaddleOCR が担う。
- locro や Screen AI コンポーネントが無い環境では、ハイブリッドは
  PaddleOCR のみで動作する（create_ocr_engine が吸収する）。

locro の導入は任意（GitHub から pip install -e .）。
Screen AI コンポーネントは Chrome が自動取得する。
"""
from __future__ import annotations

import numpy as np

from .ocr import OcrEngine


def _to_pil(image: np.ndarray):
    """OpenCV の BGR 配列を PIL Image（RGB）に変換する。"""
    from PIL import Image

    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    rgb = image[:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")


def _page_text(page) -> tuple[str, float]:
    """locro の OcrPage から (テキスト, 信頼度) を作る。

    テキストは行→単語の順に空白で結合する。信頼度は単語信頼度の
    平均（単語が無い行だけの場合は 0.0）。
    """
    texts: list[str] = []
    confs: list[float] = []
    for block in page.blocks:
        for line in block.lines:
            words = getattr(line, "words", None) or []
            if words:
                texts.append(" ".join(w.text for w in words))
                confs.extend(float(w.confidence) for w in words)
            elif getattr(line, "text", ""):
                texts.append(line.text)
    text = " ".join(t.strip() for t in texts if t.strip())
    conf = sum(confs) / len(confs) if confs else 0.0
    return text, conf


class ScreenAiEngine:
    """locro（Chrome Screen AI）のラッパー。load() 成功時のみ使える。"""

    def __init__(self):
        self._ai = None

    def load(self) -> bool:
        """locro とコンポーネントを読み込む。失敗したら False。"""
        if self._ai is not None:
            return True
        try:
            from locro import ScreenAI

            self._ai = ScreenAI()
            return True
        except Exception:
            self._ai = None
            return False

    @property
    def available(self) -> bool:
        return self._ai is not None

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        if self._ai is None:
            return "", 0.0
        page = self._ai.ocr_pil_image(_to_pil(image))
        return _page_text(page)

    def recognize_name(self, image: np.ndarray) -> tuple[str, float]:
        """名前領域専用: 2倍拡大して読む（ベンチ最良の方式）。"""
        if self._ai is None:
            return "", 0.0
        pil = _to_pil(image)
        pil = pil.resize((pil.width * 2, pil.height * 2),
                         _lanczos())
        page = self._ai.ocr_pil_image(pil)
        return _page_text(page)


def _lanczos():
    from PIL import Image

    return Image.LANCZOS


class HybridOcrEngine:
    """Screen AI（名前のみ）＋ PaddleOCR のハイブリッド。

    pipeline が要求する OcrEngine と同じメソッド群を提供する。
    """

    def __init__(self, paddle: OcrEngine,
                 screenai: ScreenAiEngine | None):
        self._paddle = paddle
        self._screenai = screenai

    @property
    def using_screenai(self) -> bool:
        return self._screenai is not None and self._screenai.available

    def recognize_name(self, image: np.ndarray) -> tuple[str, float]:
        if self.using_screenai:
            try:
                text, conf = self._screenai.recognize_name(image)  # type: ignore[union-attr]
                if text.strip():
                    return text, conf
            except Exception:
                pass  # フォールバックへ
        return self._paddle.recognize_name(image)

    # 数字系と汎用読みは PaddleOCR が担う
    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        return self._paddle.recognize(image)

    def recognize_best(self, image: np.ndarray) -> tuple[str, float]:
        return self._paddle.recognize_best(image)

    def recognize_digits_en(self, image: np.ndarray):
        return self._paddle.recognize_digits_en(image)

    def recognize_score(self, image: np.ndarray):
        return self._paddle.recognize_score(image)

    def recognize_total(self, image: np.ndarray):
        return self._paddle.recognize_total(image)


def create_ocr_engine() -> HybridOcrEngine:
    """ライブ用エンジンを作る。Screen AI が使えなければ Paddle のみ。"""
    screenai = ScreenAiEngine()
    ok = screenai.load()
    return HybridOcrEngine(OcrEngine(), screenai if ok else None)
