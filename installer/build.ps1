# 配布物のビルド手順（Windows / PowerShell）。
#   powershell -ExecutionPolicy Bypass -File installer\build.ps1
# 1. 配布専用の venv（.venv-build）に実行時依存＋PyInstaller だけを入れる
#    （開発用 venv の paddle 等を巻き込まないため）
# 2. PyInstaller(onedir) で dist\MK8DXAnalysis を作る
# 3. Inno Setup(ISCC.exe) があればインストーラ、無ければ zip を作る
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv-build")) {
    py -3.12 -m venv .venv-build
}
$py = ".venv-build\Scripts\python.exe"
& $py -m pip install -q -r requirements.txt pyinstaller
& $py -m pip install -q -e . --no-deps

& $py -m PyInstaller installer\mk8dx.spec --noconfirm --distpath dist --workpath build

$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if (-not $iscc) {
    $candidate = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (Test-Path $candidate) { $iscc = Get-Item $candidate }
}
if ($iscc) {
    & $iscc.Source installer\mk8dx.iss
    Write-Host "インストーラ: dist\installer\MK8DXAnalysis-Setup.exe"
} else {
    $zip = "dist\MK8DXAnalysis-portable.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    Compress-Archive -Path dist\MK8DXAnalysis -DestinationPath $zip
    Write-Host "Inno Setup が見つからないため zip を作成しました: $zip"
    Write-Host "インストーラが必要なら Inno Setup 6 を入れて再実行してください。"
}
