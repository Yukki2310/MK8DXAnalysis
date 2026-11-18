"""名前の文字認識エンジン（PP-OCRv5 認識モデルを ONNX Runtime で実行）。

行画像1枚を読むだけなので、文字検出は使わず認識モデルだけを使う
（RapidOCR の rec 専用モード）。実測（2026-11-18、ラウンジ動画の名前24行）:
- PP-OCRv5 mobile（ch、日本語のかな・英数字を含む）: 24行中22行が完全一致、約15〜25ms/行
- Windows 標準OCR: 16/24、PP-OCRv4 japan: 9/24
- 入力は文字マスクより生の行画像（2倍拡大）の方が良かった
読めないのは ◇ ♭ などモデルの語彙にない記号だけだった。

モデルは assets/models/ に同梱する（tools/fetch_models.py で取得）。
無い場合は RapidOCR の自動ダウンロードに任せる（開発時のみ）。
"""
from __future__ import annotations

import cv2
import numpy as np

from .paths import resource_path

MODEL_FILE = "ch_PP-OCRv5_rec_mobile.onnx"


class NameRecognizer:
    def __init__(self) -> None:
        self._engine = None

    def load(self) -> bool:
        """エンジンを読み込む。依存やモデルが無ければ False。"""
        if self._engine is not None:
            return True
        try:
            from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

            params = {
                "Global.use_det": False,
                "Global.use_cls": False,
                "Global.log_level": "error",
                "Rec.lang_type": LangRec.CH,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
            model = resource_path(f"assets/models/{MODEL_FILE}")
            if model.exists():
                params["Rec.model_path"] = str(model)
            self._engine = RapidOCR(params=params)
        except Exception:
            self._engine = None
        return self._engine is not None

    @property
    def available(self) -> bool:
        return self._engine is not None

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """名前行の画像（BGR）を読み、(テキスト, 信頼度) を返す。"""
        if self._engine is None:
            return "", 0.0
        big = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        try:
            result = self._engine(big, use_det=False, use_cls=False, use_rec=True)
        except Exception:
            return "", 0.0
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if not texts:
            return "", 0.0
        return "".join(texts), float(min(scores)) if scores else 0.0
