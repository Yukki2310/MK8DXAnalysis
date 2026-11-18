# 配布物のビルド手順（Windows / PowerShell）。
#   powershell -ExecutionPolicy Bypass -File installer\build.ps1            # 精度優先（視覚言語モデル同梱）
#   powershell -ExecutionPolicy Bypass -File installer\build.ps1 -LightOnly # 軽量版（PP-OCRv5 のみ）
#
# 1. 配布専用の venv（.venv-build）に実行時依存＋PyInstaller だけを入れる
#    （開発用 venv の paddle 等を巻き込まないため）
# 2. 文字認識モデルを assets\models に用意する（初回のみネットワーク、視覚言語モデルは約13GB）
# 3. PyInstaller(onedir) で dist\MK8DXAnalysis を作る
# 4. Inno Setup(ISCC.exe) があればインストーラ、無ければ zip を作る
param([switch]$LightOnly)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv-build")) {
    py -3.12 -m venv .venv-build
}
$py = ".venv-build\Scripts\python.exe"
& $py -m pip install -q -r requirements.txt pyinstaller
if (-not $LightOnly) {
    # GPU 版の torch は専用の配布元から入れる（CUDA 12.6）
    & $py -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126
    & $py -m pip install -q -r requirements-vlm.txt
}
& $py -m pip install -q -e . --no-deps

$fetch = Join-Path $root "tools\fetch_models.py"
if ($LightOnly) { & $py $fetch --no-vlm } else { & $py $fetch }

& $py -m PyInstaller installer\mk8dx.spec --noconfirm --distpath dist --workpath build

$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if (-not $iscc) {
    $candidate = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (Test-Path $candidate) { $iscc = Get-Item $candidate }
}
if ($iscc) {
    & $iscc.Source installer\mk8dx.iss
    Write-Host "インストーラ: dist\installer\MK8DXAnalysis-Setup.exe（容量が大きい場合は複数ファイルに分割されます）"
} else {
    $zip = "dist\MK8DXAnalysis-portable.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    # 4GB を超えるファイルも扱える Windows 標準の tar（bsdtar）で zip を作る
    tar.exe -a -c -f $zip -C dist MK8DXAnalysis
    Write-Host "Inno Setup が見つからないため zip を作成しました: $zip"
    Write-Host "インストーラが必要なら Inno Setup 6 を入れて再実行してください。"
}
