"""PyInstaller 用ランチャー（パッケージ内の相対importを使うため絶対importで起動する）。

インストーラ版はコンソールを持たない（windowed）。print や警告で落ちないよう、
標準出力・標準エラーをデータフォルダの app.log に向ける。
"""
import sys

if getattr(sys, "frozen", False):
    from mk8dx.paths import data_dir

    _log = data_dir() / "app.log"
    try:
        if _log.exists() and _log.stat().st_size > 5_000_000:
            _log.unlink()             # 肥大化したら作り直す
    except OSError:
        pass
    sys.stdout = sys.stderr = open(_log, "a", encoding="utf-8", buffering=1)

from mk8dx.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
