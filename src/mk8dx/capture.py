"""OBS仮想カメラ入力（設計書 3.1）。

OpenCV VideoCapture で仮想カメラデバイスを開く。検知周期（300〜500ms）と
OCR実行タイミングを分離するため、このモジュールはフレーム取得だけを担う。
"""
from __future__ import annotations

import time
from collections.abc import Iterator

import cv2
import numpy as np


class VirtualCamera:
    def __init__(self, device_index: int = 0, width: int = 1920, height: int = 1080):
        self.device_index = device_index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"カメラデバイス {self.device_index} を開けませんでした。"
                "OBS仮想カメラが開始しているか確認してください。"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise RuntimeError("カメラが開かれていません")
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames_at_interval(self, interval_sec: float = 0.4) -> Iterator[np.ndarray]:
        """検知周期ごとにフレームを yield する（設計書 3.1 の画面検知周期）。"""
        while True:
            started = time.monotonic()
            frame = self.read()
            if frame is not None:
                yield frame
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, interval_sec - elapsed))
