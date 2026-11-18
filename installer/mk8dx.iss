; Inno Setup スクリプト。installer/build.ps1 が PyInstaller の出力(dist\MK8DXAnalysis)を作った後に
; ISCC.exe installer\mk8dx.iss でコンパイルする。出力: dist\installer\MK8DXAnalysis-Setup.exe
#define AppName "MK8DX 自動集計"
#define AppVersion "0.2.0"

[Setup]
AppId={{6E1C0F7A-2B3D-4C58-9A41-8D2F3C7B1E90}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\MK8DXAnalysis
DefaultGroupName={#AppName}
OutputDir=..\dist\installer
OutputBaseFilename=MK8DXAnalysis-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\MK8DXAnalysis.exe

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Files]
Source: "..\dist\MK8DXAnalysis\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\MK8DXAnalysis.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\MK8DXAnalysis.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "デスクトップにショートカットを作る"

[Run]
Filename: "{app}\MK8DXAnalysis.exe"; Description: "起動する"; Flags: nowait postinstall skipifsilent
