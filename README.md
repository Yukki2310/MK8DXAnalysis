# MK8DX 自動集計

マリオカート8 デラックスのリザルト画面を OBS の仮想カメラから自動で読み取り、チームごとの得点を集計して、
OBS のオーバーレイと管理画面に表示する Windows アプリです。

> 任天堂および「マリオカート」シリーズとは無関係の非公式ツールです。ゲームのデータは含みません。

## 使い方

1. [Releases](../../releases) から `MK8DXAnalysis-Setup.exe` と `MK8DXAnalysis-Setup-*.bin` を**すべて同じフォルダ**にダウンロードして、`Setup.exe` を実行します。
2. 起動すると管理画面が開きます。OBS の仮想カメラを開始して、映像ソースと形式（2v2 / 3v3 / 4v4）を選び、「▶ 開始」を押します。
3. OBS のブラウザソース（1920×1080）に、管理画面の「OBS」タブのアドレスを貼ります。普段は小さな順位表、リザルト画面の間だけ大きな順位表が出ます。

終了はタスクトレイのアイコンから行います。

## 動作環境

- Windows 10 / 11（64bit）、OBS Studio の仮想カメラ（1920×1080 推奨）
- NVIDIA GPU（空きメモリ約10.5GB以上）があれば、高精度な文字認識（Qwen3-VL）を使います。無い場合は CPU で動く軽量なモデルに切り替わります。

## 開発

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m pytest tests
.venv\Scripts\python -m mk8dx            # サーバーを起動
```

配布物のビルド（PyInstaller と Inno Setup 6/7）: `powershell -ExecutionPolicy Bypass -File installer\build.ps1`
（`-LightOnly` で軽量版）。方式の詳細は [docs/](docs/) を参照してください。

## ライセンス

MIT License（[LICENSE](LICENSE)）。同梱するモデル・ライブラリのライセンスは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
