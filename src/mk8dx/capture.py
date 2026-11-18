"""OBS仮想カメラ入力（設計書 3.1）。

OpenCV VideoCapture で仮想カメラデバイスを開く。検知周期（300〜500ms）と
OCR実行タイミングを分離するため、このモジュールはフレーム取得だけを担う。
"""
from __future__ import annotations

import time
from collections.abc import Iterator

import cv2
import numpy as np


def list_cameras(max_index: int = 8) -> list[dict]:
    """利用可能なカメラを列挙する（番号・解像度）。

    OpenCV にはデバイス名の取得手段がないため、識別は解像度と
    プレビュー画像で行う（管理画面でプレビューを確認する想定）。
    開けない番号は一覧に含めない。
    """
    found: list[dict] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            found.append({"index": index, "width": int(w), "height": int(h)})
        finally:
            cap.release()
    return found


def grab_preview(index: int) -> bytes | None:
    """カメラのプレビュー画像（JPEGバイト列）を1枚取得する。"""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None
        frame = None
        for _ in range(5):  # 露出が落ち着くまで数フレーム読み捨てる
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None
    finally:
        cap.release()


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
