"""OBS仮想カメラ・キャプチャーボードの入力。

他のアプリ（OBS など）がすでに使っているデバイスを開こうとすると、ドライバによっては
開けなかったり、呼び出しが返ってこなかったりする。アプリ全体が固まらないよう:

- デバイスの一覧は名前だけを先に取り（デバイスを開かない）、状態の確認（開いて1枚読む）は
  デバイスごとに別プロセスで行って、時間内に終わらなければ「使用中」とみなす。
- 集計用の映像の取得は専用スレッドで行い、止まったら検知してエラーにする。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")  # 番号の範囲外などの警告を出さない

import cv2
import numpy as np

try:  # 環境変数より先に cv2 が読み込まれていた場合のため、直接も設定する
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

# Windows ではカメラを DirectShow で開く（OBS仮想カメラもこちらで見える）。
_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
PROBE_TIMEOUT = 5.0         # 1台の状態確認にかけてよい時間（秒）
OPEN_TIMEOUT = 8.0          # 集計開始時にデバイスを開くのを待つ時間
STALL_TIMEOUT = 10.0        # 映像がこの時間届かなければ、止まったとみなす
_list_lock = threading.Lock()


# -- 別プロセスで動くワーカー（python -m mk8dx --camera-worker ...）------------
def _worker_command(*args: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "mk8dx", *args]


def _run_worker(args: list[str], timeout: float) -> bool:
    """ワーカーを実行する。時間内に終われば True、超えたら強制終了して False。"""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.run(_worker_command(*args), timeout=timeout, creationflags=flags,
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except Exception:       # TimeoutExpired（強制終了済み）を含む
        return False


def camera_worker(kind: str, index: int, out: str, preview: str | None) -> None:
    """ワーカー本体。結果は out の JSON に書く（標準出力はGUI版では使えないため）。"""
    result: dict = {}
    try:
        if kind == "names":
            from pygrabber.dshow_graph import FilterGraph

            result = {"names": FilterGraph().get_input_devices()}
        else:
            cap = cv2.VideoCapture(index, _BACKEND)
            frame = None
            if cap.isOpened():
                started = time.monotonic()
                while time.monotonic() - started < 3.0:
                    ok, f = cap.read()
                    if ok and f is not None:
                        frame = f
                        # プレビューは露出が落ち着くまで少し読み続ける
                        if preview is None or time.monotonic() - started > 0.6:
                            break
            cap.release()
            if frame is None:
                result = {"ok": False}
            else:
                h, w = frame.shape[:2]
                result = {"ok": True, "width": int(w), "height": int(h)}
                if preview:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        Path(preview).write_bytes(buf.tobytes())
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    Path(out).write_text(json.dumps(result), encoding="utf-8")


def _worker_result(kind: str, index: int = 0, preview: str | None = None,
                   timeout: float = PROBE_TIMEOUT) -> dict | None:
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "result.json")
        args = ["--camera-worker", kind, "--index", str(index), "--out", out]
        if preview:
            args += ["--preview", preview]
        if not _run_worker(args, timeout):
            return None
        try:
            return json.loads(Path(out).read_text(encoding="utf-8"))
        except Exception:
            return None


# -- 一覧・プレビュー ----------------------------------------------------------
def list_cameras() -> list[dict]:
    """利用できる映像の一覧: [{index, name, state('ok'|'busy'), width, height}]。

    名前はデバイスを開かずに取る。状態は1台ずつ別プロセスで確認し、開けない・
    時間内に終わらないものは 'busy'（他のアプリが使用中）にする。
    """
    with _list_lock:
        names_result = _worker_result("names", timeout=8.0)
        names: list[str] = (names_result or {}).get("names", [])
        indices = list(range(len(names))) if names else list(range(6))

        def probe(i: int) -> dict:
            res = _worker_result("probe", i)
            item = {"index": i, "name": names[i] if i < len(names) else f"映像 {i + 1}"}
            if res and res.get("ok"):
                return {**item, "state": "ok", "width": res["width"], "height": res["height"]}
            return {**item, "state": "busy", "width": 0, "height": 0}

        with ThreadPoolExecutor(max_workers=max(1, len(indices))) as pool:
            found = list(pool.map(probe, indices))
        # 名前が取れなかった環境では、開けなかった番号は一覧に出さない
        return found if names else [c for c in found if c["state"] == "ok"]


def find_index_by_name(name: str) -> int | None:
    """前回選んだ映像の名前から、いまの番号を探す（デバイスは開かない）。"""
    res = _worker_result("names", timeout=8.0)
    names = (res or {}).get("names", [])
    return names.index(name) if name in names else None


def grab_preview(index: int) -> bytes | None:
    """カメラのプレビュー画像（JPEGバイト列）を1枚取得する。使用中なら None。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "preview.jpg")
        res = _worker_result("probe", index, preview=path, timeout=PROBE_TIMEOUT + 2)
        if res and res.get("ok") and Path(path).exists():
            return Path(path).read_bytes()
    return None


# -- 集計用の映像 --------------------------------------------------------------
class VirtualCamera:
    """デバイスを開き、専用スレッドで最新のフレームを取り続ける。"""

    def __init__(self, device_index: int = 0, width: int = 1920, height: int = 1080):
        self.device_index = device_index
        self.width = width
        self.height = height
        self._opened = threading.Event()
        self._open_failed = False
        self._latest: np.ndarray | None = None
        self._latest_at = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def open(self, timeout: float = OPEN_TIMEOUT) -> None:
        """デバイスを開く。他のアプリ（OBSなど）が使用中で応答しないときは、待たずにエラーにする。

        DirectShow のデバイスは、開いたスレッドの中で使い続け、閉じるのも同じスレッドで行う
        必要がある（別のスレッドから触るとアクセス違反でアプリが落ちる）。そのため、
        開く・設定する・読み続ける・閉じる をすべて専用スレッドの中で行い、
        ここでは「開けたか」を時間を区切って待つだけにする。
        """
        self._opened = threading.Event()
        thread = threading.Thread(target=self._device_loop, daemon=True)
        thread.start()
        if not self._opened.wait(timeout) or self._open_failed:
            self._stop.set()        # 待ちきれなかった後で開けた場合は、スレッド側がすぐ閉じる
            raise RuntimeError(
                f"カメラデバイス {self.device_index} を開けませんでした。"
                "他のアプリ（OBSなど）が使用中の可能性があります。")
        self._latest_at = time.monotonic()

    def _device_loop(self) -> None:
        cap = None
        try:
            cap = cv2.VideoCapture(self.device_index, _BACKEND)
            if not cap.isOpened():
                self._open_failed = True
                return
            if self._stop.is_set():     # 待ちきれずに諦められた後だった
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._latest_at = time.monotonic()
            self._opened.set()
            while not self._stop.is_set():
                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self._latest, self._latest_at = frame, time.monotonic()
                        self._seq += 1
                else:
                    time.sleep(0.05)
        except Exception:
            self._open_failed = True
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._opened.set()

    def release(self) -> None:
        """止める。デバイスは専用スレッドが自分で閉じる（応答しなくても、ここは待たない）。"""
        self._stop.set()

    def frames_at_interval(self, interval_sec: float = 0.4,
                           stop: threading.Event | None = None) -> Iterator[np.ndarray]:
        """検知周期ごとに最新のフレームを yield する。映像が止まったら例外にする。"""
        last_seq = -1
        while not self._stop.is_set() and not (stop and stop.is_set()):
            started = time.monotonic()
            with self._lock:
                frame, seq, at = self._latest, self._seq, self._latest_at
            if time.monotonic() - at > STALL_TIMEOUT:
                raise RuntimeError(
                    "映像が途切れました。OBSの仮想カメラ・キャプチャーボードを確認してください。")
            if frame is not None and seq != last_seq:
                last_seq = seq
                yield frame
            time.sleep(max(0.0, interval_sec - (time.monotonic() - started)))
