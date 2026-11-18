#!/usr/bin/env python3
"""文字認識モデルを assets/models/ に用意する（配布版のビルド前に1回実行）。

- PP-OCRv5 認識モデル（ONNX、16MB）: RapidOCR が自動ダウンロードしたものをコピーする。
- Qwen3-VL-4B / 2B（視覚言語モデル、約8.9GB / 4.3GB）: Hugging Face から取得する。
  `--no-vlm` で省略できる（その場合は PP-OCRv5 だけの軽量版になる）。
初回はネットワークが必要。取得済みのものはスキップする。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mk8dx.textrec import RAPID_MODEL_FILE, VLM_TIERS  # noqa: E402

MODELS = ROOT / "assets" / "models"


def fetch_rapid() -> None:
    dest = MODELS / RAPID_MODEL_FILE
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
    src = Path(rapidocr.__file__).parent / "models" / RAPID_MODEL_FILE
    if not src.exists():
        sys.exit(f"モデルが見つかりません: {src}")
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"コピーしました: {dest} ({dest.stat().st_size // 1024} KB)")


def fetch_vlm() -> None:
    from huggingface_hub import snapshot_download

    for folder, hf_id, _ in VLM_TIERS:
        dest = MODELS / folder
        if (dest / "config.json").exists() and any(dest.glob("*.safetensors")):
            print(f"取得済み: {dest}")
            continue
        print(f"取得中: {hf_id}（数GB、時間がかかります）")
        snapshot_download(
            hf_id, local_dir=str(dest),
            ignore_patterns=["*.md", ".gitattributes", "*.msgpack", "*.h5", "*.onnx"])
        print(f"保存しました: {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-vlm", action="store_true", help="視覚言語モデルを取得しない")
    args = ap.parse_args()
    fetch_rapid()
    if not args.no_vlm:
        fetch_vlm()


if __name__ == "__main__":
    main()
