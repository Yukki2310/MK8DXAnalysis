# PyInstaller spec（onedir）。ビルド: installer/build.ps1
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = SPECPATH + "/.."

import importlib.util

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("rapidocr")
    + ["websockets", "mk8dx.server"]
)
extra_datas = []
extra_binaries = []
import os

# 開発中の素早い再ビルド用: MK8DX_SPEC_NO_MODELS=1 でモデルを同梱せず、
# 実行時に MK8DX_MODELS_DIR で場所を指す
if os.environ.get("MK8DX_SPEC_NO_MODELS"):
    asset_datas = [(ROOT + "/assets/rank_templates.npz", "assets")]
else:
    asset_datas = [(ROOT + "/assets", "assets")]
# 視覚言語モデル（Qwen3-VL）を同梱するビルドのときだけ、torch / transformers を含める
have_vlm = (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None)
if have_vlm:
    needed_models = {"auto", "qwen3_vl", "qwen2_vl", "qwen3", "qwen2"}
    hidden += collect_submodules(
        "transformers",
        filter=lambda n: not n.startswith("transformers.models.")
        or n.split(".")[2] in needed_models)
    hidden += (collect_submodules("accelerate") + collect_submodules("huggingface_hub")
               + ["tokenizers", "safetensors", "regex"])
    extra_datas += collect_data_files("transformers")
    # torchvision は C 拡張（演算子 nms など）を自分で読み込むため、拡張ごと丸ごと含める
    tv_datas, tv_binaries, tv_hidden = collect_all("torchvision")
    extra_datas += tv_datas
    extra_binaries += tv_binaries
    hidden += tv_hidden

a = Analysis(
    [SPECPATH + "/launcher.py"],
    pathex=[ROOT + "/src"],
    binaries=extra_binaries,
    datas=[
        (ROOT + "/web", "web"),
        (ROOT + "/config", "config"),
    ] + asset_datas + collect_data_files("rapidocr", excludes=["models/*"]) + extra_datas,
    hiddenimports=hidden,
    excludes=["paddle", "paddleocr", "paddlex", "tensorflow",
              "matplotlib", "pandas", "scipy", "tkinter", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="MK8DXAnalysis",
    console=True,   # ログ表示とウィンドウを閉じて終了できるように、あえてコンソール付き
)
coll = COLLECT(exe, a.binaries, a.datas, name="MK8DXAnalysis")
