"""Windows 標準 OCR（Windows.Media.Ocr）のラッパー。

モデル配布が不要で、インストーラに載せるものが増えない。日本語と英語の
認識器が入っていれば使える（Windows 10/11 の言語機能）。名前は
「表示名の初期提案」にしか使わないため（同一人物の判定は identity.py の
画像照合が担う）、読みが多少ずれても致命的ではない。

使えない環境（winrt が無い、言語パックが無い）では load() が False を返し、
呼び出し側は「選手N」の仮名にフォールバックする。
"""
from __future__ import annotations

import asyncio

import cv2
import numpy as np


class WindowsOcr:
    def __init__(self, languages: tuple[str, ...] = ("ja", "en-US")) -> None:
        self._languages = languages
        self._engines: list = []

    def load(self) -> bool:
        if self._engines:
            return True
        try:
            from winrt.windows.globalization import Language
            from winrt.windows.media.ocr import OcrEngine

            for tag in self._languages:
                lang = Language(tag)
                if OcrEngine.is_language_supported(lang):
                    engine = OcrEngine.try_create_from_language(lang)
                    if engine is not None:
                        self._engines.append(engine)
        except Exception:
            self._engines = []
        return bool(self._engines)

    @property
    def available(self) -> bool:
        return bool(self._engines)

    async def _recognize_async(self, engine, image: np.ndarray) -> str:
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.storage.streams import (
            DataWriter,
            InMemoryRandomAccessStream,
        )

        ok, buf = cv2.imencode(".png", image)
        if not ok:
            return ""
        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        writer.write_bytes(buf.tobytes())
        await writer.store_async()
        await writer.flush_async()
        stream.seek(0)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await engine.recognize_async(bitmap)
        return result.text or ""

    def recognize(self, image: np.ndarray) -> list[str]:
        """全言語エンジンで読んだ候補テキストを返す（失敗は空文字）。"""
        out: list[str] = []
        for engine in self._engines:
            try:
                out.append(
                    asyncio.run(self._recognize_async(engine, image)).strip()
                )
            except Exception:
                out.append("")
        return out
