"""capture.py（応答しないデバイスでも固まらない仕組み）のテスト。実際のカメラは使わない。"""
import sys
import threading
import time

import pytest

from mk8dx import capture


def test_worker_that_hangs_is_killed_after_timeout(monkeypatch):
    monkeypatch.setattr(
        capture, "_worker_command",
        lambda *a: [sys.executable, "-c", "import time; time.sleep(60)"])
    started = time.monotonic()
    assert capture._run_worker(["x"], timeout=1.0) is False
    assert time.monotonic() - started < 10


def test_list_cameras_marks_unresponsive_devices_busy(monkeypatch):
    def fake(kind, index=0, preview=None, timeout=0):
        if kind == "names":
            return {"names": ["キャプチャーボード", "仮想カメラ"]}
        return {"ok": True, "width": 1920, "height": 1080} if index == 1 else None  # 0番は応答なし

    monkeypatch.setattr(capture, "_worker_result", fake)
    found = capture.list_cameras()
    assert [c["name"] for c in found] == ["キャプチャーボード", "仮想カメラ"]
    assert [c["state"] for c in found] == ["busy", "ok"]
    assert found[1]["width"] == 1920


def test_list_cameras_without_names_lists_only_working_devices(monkeypatch):
    def fake(kind, index=0, preview=None, timeout=0):
        if kind == "names":
            return None
        return {"ok": True, "width": 640, "height": 480} if index == 2 else {"ok": False}

    monkeypatch.setattr(capture, "_worker_result", fake)
    found = capture.list_cameras()
    assert [c["index"] for c in found] == [2]


def test_open_gives_up_when_the_device_does_not_answer(monkeypatch):
    class Hanging:
        def __init__(self, *a):
            time.sleep(5)

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(capture.cv2, "VideoCapture", Hanging)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="開けませんでした"):
        capture.VirtualCamera(0).open(timeout=0.5)
    assert time.monotonic() - started < 3


def test_device_is_opened_used_and_closed_in_one_thread(monkeypatch):
    """DirectShow のデバイスは、開いたスレッドとは別のスレッドで触るとアプリが落ちる。"""
    import numpy as np

    threads: dict[str, set] = {}

    def note(what):
        threads.setdefault(what, set()).add(threading.get_ident())

    class Fake:
        def __init__(self, *a):
            note("open")

        def isOpened(self):
            return True

        def set(self, *a):
            note("set")

        def read(self):
            note("read")
            time.sleep(0.01)
            return True, np.zeros((2, 2, 3), np.uint8)

        def release(self):
            note("release")

    monkeypatch.setattr(capture.cv2, "VideoCapture", Fake)
    cam = capture.VirtualCamera(0)
    cam.open(timeout=2)
    stop = threading.Event()
    assert next(cam.frames_at_interval(0.02, stop=stop)) is not None
    cam.release()
    time.sleep(0.2)
    assert set(threads) == {"open", "set", "read", "release"}
    used = set().union(*threads.values())
    assert len(used) == 1 and threading.get_ident() not in used


def test_failed_open_reports_error(monkeypatch):
    class Closed:
        def __init__(self, *a):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(capture.cv2, "VideoCapture", Closed)
    with pytest.raises(RuntimeError, match="開けませんでした"):
        capture.VirtualCamera(0).open(timeout=2)


def test_frames_stop_and_raise_when_the_stream_stalls(monkeypatch):
    monkeypatch.setattr(capture, "STALL_TIMEOUT", 0.2)
    cam = capture.VirtualCamera(0)
    cam._latest_at = time.monotonic() - 1.0       # 1秒前から1枚も届いていない
    with pytest.raises(RuntimeError, match="途切れ"):
        next(cam.frames_at_interval(0.05))


def test_frames_generator_stops_on_stop_event():
    import numpy as np

    cam = capture.VirtualCamera(0)
    cam._latest, cam._latest_at, cam._seq = np.zeros((2, 2, 3), np.uint8), time.monotonic(), 1
    stop = threading.Event()
    gen = cam.frames_at_interval(0.05, stop=stop)
    assert next(gen) is cam._latest
    stop.set()
    assert list(gen) == []
