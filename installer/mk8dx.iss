; Inno Setup スクリプト。installer/build.ps1 が PyInstaller の出力(dist\MK8DXAnalysis)を作った後に
; ISCC.exe installer\mk8dx.iss でコンパイルする。出力: dist\installer\MK8DXAnalysis-Setup.exe
#define AppName "MK8DX 自動集計"
#define AppVersion "0.2.0"
#define Models "_internal\assets\models"

[Setup]
AppId={{6E1C0F7A-2B3D-4C58-9A41-8D2F3C7B1E90}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\MK8DXAnalysis
DefaultGroupName={#AppName}
OutputDir=..\dist\installer
OutputBaseFilename=MK8DXAnalysis-Setup
SetupIconFile=..\build\app.ico
; 大きなインストーラは複数ファイルに分割する。モデル（数GB）は圧縮してもほとんど縮まず、
; 展開に時間がかかるだけなので、下の [Files] で無圧縮にしている（インストールが速い）
Compression=lzma2/fast
SolidCompression=no
LZMAUseSeparateProcess=yes
DiskSpanning=yes
; GitHub Releases の1ファイル上限(2GiB)に余裕を持たせる
DiskSliceSize=2000000000
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\MK8DXAnalysis.exe

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Types]
Name: "standard"; Description: "標準"
Name: "custom"; Description: "カスタム"; Flags: iscustom

[Components]
Name: "main"; Description: "本体（高精度な文字認識 Qwen3-VL-4B を含む）"; Types: standard custom; Flags: fixed
Name: "small"; Description: "小型の文字認識モデル（GPUメモリが少ないとき用・約4GB）"; Types: custom

[Files]
; 本体（モデルを除く）
Source: "..\dist\MK8DXAnalysis\*"; DestDir: "{app}"; Excludes: "{#Models}\*"; Flags: recursesubdirs ignoreversion; Components: main
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion; Components: main
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion; Components: main
; 文字認識モデル（無圧縮）
Source: "..\dist\MK8DXAnalysis\{#Models}\ch_PP-OCRv5_rec_mobile.onnx"; DestDir: "{app}\{#Models}"; Flags: ignoreversion nocompression; Components: main
Source: "..\dist\MK8DXAnalysis\{#Models}\Qwen3-VL-4B-Instruct\*"; DestDir: "{app}\{#Models}\Qwen3-VL-4B-Instruct"; Flags: recursesubdirs ignoreversion nocompression; Components: main
Source: "..\dist\MK8DXAnalysis\{#Models}\Qwen3-VL-2B-Instruct\*"; DestDir: "{app}\{#Models}\Qwen3-VL-2B-Instruct"; Flags: recursesubdirs ignoreversion nocompression; Components: small

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\MK8DXAnalysis.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\MK8DXAnalysis.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "デスクトップにショートカットを作る"

[Run]
Filename: "{app}\MK8DXAnalysis.exe"; Description: "起動する"; Flags: nowait postinstall skipifsilent
