"""パス解決（開発時とインストーラ版（PyInstaller）の両対応）。

- resource_path: 読み取り専用の同梱リソース（web/, config/, assets/）
- data_dir: 書き込むデータ（DB・スクリーンショット）。開発時はリポジトリの
  data/、配布版は %APPDATA%\\MK8DXAnalysis。環境変数 MK8DX_DATA_DIR で上書き可。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def base_dir() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def resource_path(relative: str) -> Path:
    return base_dir() / relative


def data_dir() -> Path:
    override = os.environ.get("MK8DX_DATA_DIR")
    if override:
        path = Path(override)
    elif is_frozen():
        path = Path(os.environ.get("APPDATA", Path.home())) / "MK8DXAnalysis"
    else:
        path = base_dir() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path
