#!/usr/bin/env python3
"""文字認識モデルを assets/models/ に用意する（配布版のビルド前に1回実行）。

RapidOCR が自動ダウンロードしたモデルをコピーする。初回はネットワークが必要。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mk8dx.textrec import MODEL_FILE  # noqa: E402


def main() -> None:
    dest_dir = ROOT / "assets" / "models"
    dest = dest_dir / MODEL_FILE
    if dest.exists():
        print(f"取得済み: {dest}")
        return
    import rapidocr
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    RapidOCR(params={
        "Global.use_det": False, "Global.use_cls": False,
        "Rec.lang_type": LangRec.CH, "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
    })
    src = Path(rapidocr.__file__).parent / "models" / MODEL_FILE
    if not src.exists():
        sys.exit(f"モデルが見つかりません: {src}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"コピーしました: {dest} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
