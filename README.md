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
- **文字認識は新規選手の名前提案だけ**（`namer.py` / `textrec.py`）: 精度優先で、使える中で最良のエンジンを自動で選ぶ。
  NVIDIA GPU（空き約10.5GB以上）なら視覚言語モデル **Qwen3-VL-4B**、足りなければ 2B、GPUが無ければ
  PP-OCRv5 の認識モデル（ONNX、CPUで動く）。名前220枚の実測で、Qwen3-VL-4B は厳密一致 196枚、
  PP-OCRv5 は 158枚、EasyOCR は 40枚（詳細は `docs/ocr-selection.md`）。記号（★ ◇）も読める。
  どれも使えなければ「選手N」の仮名で動く。管理画面には、読み取った文字ではなく**ゲーム画面の名前の
  実画像**を出すので、読み違いがあっても見分けられ、画面で直せば固定される。
- **チームの自動提案**（`teams.py`）: 名前の左端から「同じ形が続く幅」を画像で測り、同じタグで始まる選手を
  人数制約（2/3/4人）に合う組に分ける。文字認識が読み違えても動き、大文字小文字の違いは文字の読みで補う。
  決められなければ名前の先頭文字からの推定にフォールバックする。それでもだめなら、視覚言語モデルに
  名前の画像12枚を見せてチーム分けとタグを推定させる（答えは人数などを検証してから使う。実測で2レースとも
  正解、約12秒）。チームのタグは、すでに付いているものがあればそれにそろえるので、名前の読みが
  レースごとに少し違っても表記が揺れない（手動で付けたタグを優先）。それでもだめなら画面で設定する。
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
  namer.py / textrec.py    新規選手の名前提案（Qwen3-VL → PP-OCRv5 の順に使えるものを選ぶ）
  processor.py             複数フレームの観測蓄積→確定→チーム判定
  runner.py                ライブランナー（検知→確定→保存→累計照合、承認フロー）
  pipeline.py              フレームから観測（マスク・国旗）を抽出、累計クロスチェック
  scoring.py / tags.py / clustering.py   配点・タグ正規化・チーム判定（純ロジック）
  state.py / db.py         状態遷移とSQLite
  capture.py               OBS仮想カメラ入力
  teams.py / standings.py  チーム分けの検証、順位表の計算（チームは今の選手名簿から引く）
  server.py                FastAPI + WebSocket（管理画面は /admin、OBS用は /overlay）
web/overlay.html           OBSブラウザソース用オーバーレイ（普段は小さく、リザルト画面の間だけ大きく展開）
web/admin.html             管理画面（1画面のダッシュボード: 順位・直前の結果・選手・OBS設定）
installer/                 PyInstaller spec / Inno Setup / ビルドスクリプト
tools/                     extract_frames, build_rank_templates, fetch_models, identity_check, step8_check, run_video
tests/                     単体テスト（画像・OCR不要）
```

## 使い方（配布版）

`MK8DXAnalysis.exe` を起動すると管理画面（ダッシュボード）がブラウザで開く。コンソールは出ず、
タスクトレイのアイコンから「管理画面を開く」「終了」ができる（二重に起動しても、開き直すだけ）。ログは `%APPDATA%\MK8DXAnalysispp.log`。

1. 右上の「映像」で OBS の仮想カメラを選び、形式（2v2 / 3v3 / 4v4）を選んで「▶ 開始」。
2. レースが終わるたびに自動で集計される。左に順位、右に直前のレースの結果（画面の名前・選手・チーム・点）が出る。
   名前やチームは結果の表でそのまま直せて、順位にもすぐ反映される。
   チームを決められなかったときだけ「要確認」になるので、同じチームの選手に同じチーム名を入れて「確定」。
3. 「OBS」タブのアドレスを OBS のブラウザソース（1920×1080）に貼る。普段は小さな順位表が右上に出て、
   ゲームのリザルト画面が出ている間だけ、画面の左側に大きな順位表が展開する。位置はタブで変えられる。

間違えたとき: 「⋯」メニューの「直前のレースを取り消す」、選手タブでチーム名・統合・削除。
アプリを閉じても、次に開くと直前の試合がそのまま続けられる。

## 開発

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pip install -e . --no-deps   # python -m mk8dx に必要（pytest だけなら不要）
.venv\Scripts\python -m pytest tests
.venv\Scripts\python -m mk8dx                      # サーバー起動
.venv\Scripts\python tools\run_video.py --auto-approve clip1.mp4 clip2.mp4   # 動画で通し確認
```

配布物のビルド（PyInstaller onedir ＋ Inno Setup。Inno Setup が無ければ zip。文字認識モデルは初回に自動取得。
視覚言語モデルを含む完全版は約17GBになる。`-LightOnly` を付けると PP-OCRv5 のみの軽量版）。
Inno Setup（6 または 7）があればインストーラ（`dist\installer`、複数ファイルに分割）まで作る。
モデルは無圧縮で格納してインストールを速くしており、小型モデル（Qwen3-VL-2B）は任意のコンポーネント:

```powershell
powershell -ExecutionPolicy Bypass -File installer\build.ps1
dist\MK8DXAnalysis\MK8DXAnalysis.exe --selfcheck
```

評価用フレームの用意: `tools/extract_frames.py`（yt-dlp / ffmpeg が必要）。動画とフレームは
`.gitignore` 対象。ROI を変えたら `tools/build_rank_templates.py` で順位テンプレートを作り直す。

## 実装状況と既知の制約

- [x] 画像ベース識別、順位テンプレート検証、加算表示／並び順によるフェーズ判別
- [x] 精度優先の文字認識（Qwen3-VL → PP-OCRv5）による名前提案、Paddle/Screen AI/Windows OCR 依存の除去
- [x] PyInstaller 配布版（selfcheck とサーバー応答を確認）、Inno Setup スクリプト
- [ ] 実機（OBS仮想カメラ＋Switch実況）での通し確認。検証は YouTube 動画（圧縮映像）のみ
- 1080p 前提のROI。別解像度は 1920x1080 へ拡大縮小して入力する想定（未対応）
- 'racist #1' と '#2' のような酷似名は、画像距離の差が小さく最も弱い
- 日本語版ゲーム画面のみ対象。同名同旗の複数人は1エントリとして扱う
- 配布版は未署名のため、初回起動時に SmartScreen の警告が出ることがある
- `docs/step*-notes.md` は旧方式（OCR）時代の検証記録。経緯として残している
