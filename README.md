# MK8DXAnalysis

マリオカート8 デラックスのレース映像（OBS仮想カメラ）からリザルト画面を自動検知し、
チーム得点を自動集計してOBSオーバーレイ（ブラウザソース）へリアルタイム反映するシステム。

設計書: `docs/design.md`（初期の設計。実装は下記の「方式」に変更済み）

## 方式（v0.2）

汎用OCR（PaddleOCR・Chrome Screen AI）では、ゲーム独自フォントと半透明パネルのため
名前の読みがフレームごとに揺れ、同一人物を別人として記録してしまった。そこで
**名前を文字列として読んで名寄せする方式をやめた**。

- **同一人物の判定は画像で行う**（`identity.py`）: 名前行から文字だけを取り出した
  二値マスク同士の類似度で照合する。背景（コース映像）に依存せず、順位が入れ替わっても追える。
- **順位は固定フォントなのでテンプレート照合**（`rankcheck.py`）: コース選択画面などの誤検知を棄却する。
- **ゴール順画面／累計画面の判別**: スコア列の加算表示（`+N`）の有無と、
  行の並びが自前累計の降順かどうかで判定（`runner.py`）。
- **文字認識は新規選手の名前提案だけ**（`namer.py`）: Windows 標準OCR（日本語＋英語）で読み、
  管理画面で直せば固定される。OCRが使えない環境では「選手N」の仮名で動く。
- 得点は順位から配点表で確定する（OCRしない）。82点検証あり。

## 構成

```
config/roi_1080p.json      リザルト画面の関心領域(ROI)定義（行ピッチ78px）
assets/rank_templates.npz  順位列テンプレート（tools/build_rank_templates.py で生成）
src/mk8dx/
  __main__.py              起動エントリ（サーバー起動＋管理画面を開く、--selfcheck）
  paths.py                 開発時／配布版のパス解決（データは %APPDATA%\MK8DXAnalysis）
  identity.py              文字マスク抽出と距離計算（画像ベース識別）
  rankcheck.py             順位列テンプレートによるリザルト画面検証
  roster.py                選手名簿（画像距離＋国旗で一括割り当て、多数決、永続化）
  namer.py / winocr.py     新規選手の名前提案（Windows標準OCR）
  processor.py             複数フレームの観測蓄積→確定→チーム判定
  runner.py                ライブランナー（検知→確定→保存→累計照合、承認フロー）
  pipeline.py              フレームから観測（マスク・国旗）を抽出、累計クロスチェック
  scoring.py / tags.py / clustering.py   配点・タグ正規化・チーム判定（純ロジック）
  state.py / db.py         状態遷移とSQLite
  capture.py               OBS仮想カメラ入力
  server.py                FastAPI + WebSocket
web/overlay.html           OBSブラウザソース用オーバーレイ（背景透過）
web/admin.html             管理画面（モニター・選手名簿・承認・カメラ選択）
installer/                 PyInstaller spec / Inno Setup / ビルドスクリプト
tools/                     extract_frames, build_rank_templates, identity_check, step8_check, run_video
tests/                     単体テスト（画像・OCR不要）
```

## 使い方（配布版）

インストーラ（または zip）で導入し、`MK8DXAnalysis.exe` を起動する。
管理画面が開く。OBS のブラウザソースには表示されたオーバーレイURL
（既定 `http://localhost:8000/overlay`）を指定する。

1. 管理画面で対戦形式を選んでセッション開始。
2. OBS の仮想カメラを開始し、「カメラソース」でプレビューを見て仮想カメラを選ぶ。
3. 「ライブ開始」。レースが自動で検知・集計される。
4. 最初のレースで選手が自動登録される。**選手名簿で名前を正しい表記（チームタグが先頭に来る形）に
   直す**とチーム判定が成立する。名前が読めなかった選手は「選手N」と表示されるので、同じく直す。
5. 要確認のレースは管理画面で承認／破棄する。

## 開発

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest tests
.venv\Scripts\python -m mk8dx                      # サーバー起動
.venv\Scripts\python tools\run_video.py --auto-approve clip1.mp4 clip2.mp4   # 動画で通し確認
```

配布物のビルド（PyInstaller onedir ＋ Inno Setup。Inno Setup が無ければ zip）:

```powershell
powershell -ExecutionPolicy Bypass -File installer\build.ps1
dist\MK8DXAnalysis\MK8DXAnalysis.exe --selfcheck
```

評価用フレームの用意: `tools/extract_frames.py`（yt-dlp / ffmpeg が必要）。動画とフレームは
`.gitignore` 対象。ROI を変えたら `tools/build_rank_templates.py` で順位テンプレートを作り直す。

## 実装状況と既知の制約

- [x] 画像ベース識別、順位テンプレート検証、加算表示／並び順によるフェーズ判別
- [x] Windows標準OCRによる名前提案、Paddle/Screen AI 依存の除去
- [x] PyInstaller 配布版（selfcheck とサーバー応答を確認）、Inno Setup スクリプト
- [ ] 実機（OBS仮想カメラ＋Switch実況）での通し確認。検証は YouTube 動画（圧縮映像）のみ
- 1080p 前提のROI。別解像度は 1920x1080 へ拡大縮小して入力する想定（未対応）
- 'racist #1' と '#2' のような酷似名は、画像距離の差が小さく最も弱い
- 日本語版ゲーム画面のみ対象。同名同旗の複数人は1エントリとして扱う
- 配布版は未署名のため、初回起動時に SmartScreen の警告が出ることがある
- `docs/step*-notes.md` は旧方式（OCR）時代の検証記録。経緯として残している
