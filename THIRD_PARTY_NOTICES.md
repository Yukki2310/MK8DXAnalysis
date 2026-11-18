# 第三者ソフトウェア・モデルの表記

MK8DX 自動集計（MIT License、`LICENSE` 参照）の配布物には、次のソフトウェアとモデルが含まれます。
各ライセンスの全文は、それぞれの配布元で確認できます。

## モデル

| 名前 | 用途 | ライセンス | 配布元 |
|---|---|---|---|
| Qwen3-VL-4B-Instruct / Qwen3-VL-2B-Instruct | 名前の文字認識・チーム推定 | Apache License 2.0 | https://huggingface.co/Qwen |
| PP-OCRv5 認識モデル（ONNX 変換版） | 名前の文字認識（GPU なしの代替） | Apache License 2.0 | https://github.com/PaddlePaddle/PaddleOCR / https://github.com/RapidAI/RapidOCR |

## ライブラリ

| 名前 | ライセンス |
|---|---|
| PyTorch / torchvision | BSD-3-Clause ほか（`torch` の配布物に記載） |
| Transformers / Accelerate / huggingface_hub / safetensors / tokenizers | Apache License 2.0 |
| RapidOCR | Apache License 2.0 |
| ONNX Runtime | MIT License |
| OpenCV（opencv-python-headless） | Apache License 2.0 |
| NumPy | BSD-3-Clause ほか |
| FastAPI / pydantic | MIT License |
| Starlette / Uvicorn / websockets | BSD-3-Clause |
| Pillow | MIT-CMU（HPND） |
| **pystray** | **LGPL-3.0**（https://github.com/moses-palmer/pystray ）。タスクトレイのアイコンに使用。改変せずそのまま同梱しています。ソースは上記で入手できます。 |
| PyInstaller | GPL-2.0-or-later（実行ファイル作成用の例外条項により、作成物にはGPLは及びません） |

このほか、Python 本体とその標準ライブラリ（PSF License）、各ライブラリが依存するパッケージが含まれます。
