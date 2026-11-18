# PyInstaller spec（onedir）。ビルド: installer/build.ps1
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = SPECPATH + "/.."

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("rapidocr")
    + ["websockets", "mk8dx.server"]
)

a = Analysis(
    [SPECPATH + "/launcher.py"],
    pathex=[ROOT + "/src"],
    datas=[
        (ROOT + "/web", "web"),
        (ROOT + "/config", "config"),
        (ROOT + "/assets", "assets"),
    ] + collect_data_files("rapidocr", excludes=["models/*"]),
    hiddenimports=hidden,
    excludes=["paddle", "paddleocr", "paddlex", "torch", "tensorflow",
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
