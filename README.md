# MK8DXAnalysis

マリオカート8 デラックスのレース映像（OBS仮想カメラ）からリザルト画面をOCRで読み取り、
チーム得点を自動集計してOBSオーバーレイ（ブラウザソース）へリアルタイム反映するシステム。

設計書: `docs/design.md`（リポジトリ外で共有された詳細設計書の要点を実装に反映）

## 構成

```
config/roi_1080p.json      リザルト画面の関心領域(ROI)定義 ※実キャプチャで要校正
src/mk8dx/
  scoring.py               配点テーブルと82点バリデーション（純ロジック）
  tags.py                  タグ正規化（大小同一視・ひらがな/カタカナ同一視・濁音区別・分音記号検知）
  clustering.py            人数制約を使ったチーム自動判定（純ロジック）
  state.py                 レース進行の状態遷移（WAITING_RACE→DETECTING→PROCESSING→COOLDOWN）
  db.py                    SQLiteスキーマと集計クエリ
  preprocess.py            ROI切り出しと画像前処理（OpenCV）
  ocr.py                   PaddleOCRラッパー（スコアは数字のみに制限）
  detector.py              リザルト画面検知（テンプレート照合＋静止判定）
  capture.py               OBS仮想カメラ入力（OpenCV VideoCapture）
  server.py                FastAPI + WebSocket（STATE_UPDATE / ALERT 配信、管理画面コマンド受付）
web/overlay.html           OBSブラウザソース用オーバーレイ（背景透過）
web/admin.html             管理画面（モニター・修正・進行コントロール）
tools/step1_check.py       ステップ1検証: 静止画1枚 → ROI/OCR → 集計までを通すCLI
tests/                     純ロジックの単体テスト（画像なしで実行可能）
```

## 実装状況

- [x] 純ロジック（配点・検証、タグ正規化、チーム判定、状態遷移、DB）＋単体テスト
- [x] 画像処理・OCR・検知・キャプチャ・サーバーの骨格
- [x] ステップ1: 実キャプチャ2枚でROI校正とOCR検証を実施（詳細: `docs/step1-notes.md`）
  - 2フェーズ（ゴール順／累計順位）の判別OK。名前OCRは12名中9〜11名が完全一致
  - スコア列の数字はセグメントフォントでOCRが安定しないため、スコアは順位から
    規定配点表で確定する方式に変更（OCRはクロスチェック専用）
- [ ] ステップ2: 動画での検知精度・重複防止の実地検証（要: レース終了前後の動画）
  - あわせてロスター方式（レースをまたいだ多数決での名前の名寄せ）を実装する
- [ ] ステップ3: OBS仮想カメラ実機接続とオーバーレイ最終調整

## セットアップ

```bash
pip install -r requirements.txt
python -m pytest tests/        # 純ロジックのテスト（重い依存なしで実行可）
python tools/step1_check.py result.png --format 2v2   # 静止画でのOCR検証
uvicorn mk8dx.server:app --app-dir src --port 8000    # サーバー起動
```

OBSのブラウザソースには `http://localhost:8000/overlay` 、
管理画面は `http://localhost:8000/admin` を開く。
