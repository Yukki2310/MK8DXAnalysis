"""アプリの起動エントリ（python -m mk8dx / インストーラ版の MK8DXAnalysis.exe）。

ローカルサーバーを起動し、管理画面を既定のブラウザで開く。
OBS のブラウザソースには http://localhost:<port>/overlay を指定する。
"""
from __future__ import annotations

import argparse
import socket
import threading
import webbrowser


def _free_port(preferred: int) -> int:
    """希望ポートが空いていればそれを、使用中なら空きポートを返す。"""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def selfcheck() -> int:
    """配布版の健全性確認。問題があれば 1 を返す。"""
    problems: list[str] = []
    try:
        import cv2
        import numpy  # noqa: F401
    except Exception as exc:
        problems.append(f"OpenCV/NumPy: {exc}")
        cv2 = None
    from .paths import resource_path
    from .rankcheck import default_templates
    from .textrec import models_dir

    for rel in ("web/admin.html", "web/overlay.html", "config/roi_1080p.json"):
        if not resource_path(rel).exists():
            problems.append(f"同梱ファイルがありません: {rel}")
    if default_templates() is None:
        problems.append("順位列テンプレート(assets/rank_templates.npz)がありません")
    try:
        from .server import app  # noqa: F401
    except Exception as exc:
        problems.append(f"サーバーの読み込み: {exc}")
    from .namer import NameReader

    reader = NameReader()
    print("名前の文字認識エンジン:", reader.engine_name)
    for reason in getattr(reader._rec, "skipped", []):
        print("  使えなかったエンジン -", reason)
    ocr_ok = False
    if reader.available and cv2 is not None:
        import numpy as np

        probe = np.full((72, 365, 3), 40, np.uint8)
        cv2.putText(probe, "Racer", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                    (255, 255, 255), 2)
        ocr_ok = bool(reader.read([probe]))
    print("名前の文字認識:", "利用可" if ocr_ok else "利用不可（仮名で動作）")
    if not ocr_ok:
        problems.append("文字認識モデルを実行できません")
    if not (models_dir() / "ch_PP-OCRv5_rec_mobile.onnx").exists():
        problems.append("文字認識モデル(assets/models)がありません")
    if reader.engine_name == "PP-OCRv5" and (
            models_dir() / "Qwen3-VL-4B-Instruct").exists():
        print("注意: 視覚言語モデルを同梱していますが、この環境では使えません"
              "（NVIDIA GPU が無い、または空きメモリ不足）。軽量版で動作します。")
    for p in problems:
        print("NG:", p)
    print("selfcheck:", "NG" if problems else "OK")
    return 1 if problems else 0


def _already_running(port: int) -> bool:
    """同じポートで、このアプリがすでに動いているか。"""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(prog="mk8dx")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true",
                        help="管理画面を自動で開かない")
    parser.add_argument("--tray", action="store_true",
                        help="コンソールの代わりにタスクトレイから操作する（インストーラ版の既定）")
    parser.add_argument("--selfcheck", action="store_true",
                        help="同梱リソースと依存の読み込みを確認して終了する")
    args = parser.parse_args()
    if args.selfcheck:
        raise SystemExit(selfcheck())

    from .paths import data_dir, is_frozen

    use_tray = args.tray or is_frozen()
    if use_tray and _already_running(args.port):
        # すでに起動している: 二重に起動せず、管理画面だけ開く
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.port}/admin")
        return

    import uvicorn

    from .server import _live_stop, app

    port = _free_port(args.port)
    admin_url = f"http://localhost:{port}/admin"
    print("MK8DX 自動集計を起動します")
    print(f"  管理画面        : {admin_url}")
    print(f"  OBSオーバーレイ : http://localhost:{port}/overlay")
    print(f"  データ保存先    : {data_dir()}")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(admin_url)).start()

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    if not use_tray:
        print("終了するにはこのウィンドウを閉じてください。")
        server.run()
        return

    # コンソールを出さない運用: サーバーは裏のスレッドで動かし、トレイアイコンから終了する
    from .tray import run_tray

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    def quit_app() -> None:
        _live_stop.set()              # 集計中ならカメラを解放させる
        server.should_exit = True

    try:
        run_tray(lambda: webbrowser.open(admin_url), quit_app)
    finally:
        quit_app()
        thread.join(timeout=8)


if __name__ == "__main__":
    main()
