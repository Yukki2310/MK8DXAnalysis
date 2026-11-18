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

    print("名前OCR(Windows):", "利用可" if NameReader().available else "利用不可（仮名で動作）")
    for p in problems:
        print("NG:", p)
    print("selfcheck:", "NG" if problems else "OK")
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="mk8dx")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true",
                        help="管理画面を自動で開かない")
    parser.add_argument("--selfcheck", action="store_true",
                        help="同梱リソースと依存の読み込みを確認して終了する")
    args = parser.parse_args()
    if args.selfcheck:
        raise SystemExit(selfcheck())

    import uvicorn

    from .paths import data_dir
    from .server import app

    port = _free_port(args.port)
    print("MK8DX 自動集計を起動します")
    print(f"  管理画面        : http://localhost:{port}/admin")
    print(f"  OBSオーバーレイ : http://localhost:{port}/overlay")
    print(f"  データ保存先    : {data_dir()}")
    print("終了するにはこのウィンドウを閉じてください。")
    if not args.no_browser:
        threading.Timer(
            1.5, lambda: webbrowser.open(f"http://localhost:{port}/admin")
        ).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
