"""OBS仮想カメラ入力（設計書 3.1）。

OpenCV VideoCapture で仮想カメラデバイスを開く。検知周期（300〜500ms）と
OCR実行タイミングを分離するため、このモジュールはフレーム取得だけを担う。
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterator

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")  # 番号の範囲外などの警告を出さない

import cv2
import numpy as np

try:  # 環境変数より先に cv2 が読み込まれていた場合のため、直接も設定する
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

# Windows ではカメラを DirectShow で開く（OBS仮想カメラもこちらで見える）。
# 既定の順序だと FFmpeg の番号解釈が先に試され、範囲外の警告が大量に出る。
_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
# 連続してこの数だけ開けない番号が続いたら、それ以降は探さない
_MAX_CONSECUTIVE_MISSES = 2


def list_cameras(max_index: int = 8) -> list[dict]:
    """利用可能なカメラを列挙する（番号・解像度）。

    OpenCV にはデバイス名の取得手段がないため、識別は解像度と
    プレビュー画像で行う（管理画面でプレビューを確認する想定）。
    開けない番号は一覧に含めない。
    """
    found: list[dict] = []
    misses = 0
    for index in range(max_index):
        cap = cv2.VideoCapture(index, _BACKEND)
        try:
            if not cap.isOpened():
                misses += 1
                if misses >= _MAX_CONSECUTIVE_MISSES:
                    break
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                misses += 1
                continue
            misses = 0
            h, w = frame.shape[:2]
            found.append({"index": index, "width": int(w), "height": int(h)})
        finally:
            cap.release()
    return found


def grab_preview(index: int) -> bytes | None:
    """カメラのプレビュー画像（JPEGバイト列）を1枚取得する。"""
    cap = cv2.VideoCapture(index, _BACKEND)
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
        cap = cv2.VideoCapture(self.device_index, _BACKEND)
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
