"""タスクトレイのアイコン（インストーラ版はコンソールを出さず、これで操作・終了する）。"""
from __future__ import annotations

from collections.abc import Callable


def make_icon_image(size: int = 64):
    """表彰台（金・銀・銅）を描いたアイコン画像（PIL）。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (27, 35, 51, 255))
    d = ImageDraw.Draw(img)
    u = size / 16
    bars = [  # (左, 高さ, 色): 銀 金 銅 の並び
        (2, 7, (207, 214, 226, 255)),
        (6, 10, (246, 199, 68, 255)),
        (10, 5, (217, 134, 61, 255)),
    ]
    for left, height, color in bars:
        d.rectangle([left * u, (14 - height) * u, (left + 4) * u - 1, 14 * u], fill=color)
    return img


def run_tray(open_admin: Callable[[], None], quit_app: Callable[[], None]) -> None:
    """トレイアイコンを表示し、終了が選ばれるまでブロックする。"""
    import pystray

    def on_quit(icon, _item) -> None:
        quit_app()
        icon.stop()

    icon = pystray.Icon(
        "mk8dx", make_icon_image(), "MK8DX 自動集計",
        menu=pystray.Menu(
            pystray.MenuItem("管理画面を開く", lambda _i, _m: open_admin(), default=True),
            pystray.MenuItem("終了", on_quit),
        ),
    )
    icon.run()
