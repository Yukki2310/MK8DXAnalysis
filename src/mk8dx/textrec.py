"""名前の文字認識エンジン（精度優先の構成）。

実測（2026-11-18、ラウンジ動画から自動ラベル付けした名前画像220枚、
`docs/ocr-selection.md`）:

| エンジン | 緩い一致 | 厳密一致 | 1枚あたり |
|---|---|---|---|
| Qwen3-VL-4B（GPU） | 211/220 | 196/220 | 約0.1秒（12枚をまとめて1.3秒） |
| Qwen3-VL-2B（GPU） | 216/220 | 186/220 | 約0.1秒 |
| Qwen2.5-VL-3B | 200/220 | 149/220 | |
| PP-OCRv5 mobile（ONNX） | 191/220 | 158/220 | 約10ms |
| manga-ocr | 56/220 | 48/220 | |
| EasyOCR | 41/220 | 40/220 | |

視覚言語モデル（VLM）は ◇ ★ ♭ などの記号や、背景のにじんだ文字も読める。
配布版は、インストール容量を減らすため Qwen3-VL-2B だけを同梱する（4B は厳密一致が
わずかに高いが、容量が約4GB増える）。GPU（NVIDIA、空きメモリ約5.5GB以上）があれば
2B、無ければ PP-OCRv5 の認識モデルで読む。どれも使えないときは仮名（選手N）になり、
画面で名前を付ける。

モデルは assets/models/ に同梱する（tools/fetch_models.py で取得）。無い場合は
開発時のみ Hugging Face / RapidOCR の自動ダウンロードに任せる。
"""
from __future__ import annotations

import json
import os
import re

import cv2
import numpy as np

from .paths import is_frozen, resource_path

RAPID_MODEL_FILE = "ch_PP-OCRv5_rec_mobile.onnx"
MODEL_FILE = RAPID_MODEL_FILE  # 互換用の別名

VLM_PROMPT = (
    "この画像は対戦ゲームのプレイヤー名です。画像に書かれている文字を、記号（★ ☆ ◇ # など）や"
    "大文字小文字も含めて、一字一句そのまま書き起こしてください。"
    "説明や引用符は不要で、文字だけを出力してください。"
)
TEAM_RULES = (
    "ルール: 同じチームの選手は、名前の先頭が同じ文字列（タグ）で始まります。"
    "英字の大文字と小文字は同じとみなし、ひらがなとカタカナも同じとみなします。"
    "濁点・半濁音は区別します。文字認識の結果ではなく、画像を見て判断してください。"
)
# (モデルのフォルダ名, Hugging Face のID, 必要な空きGPUメモリ[GB])
VLM_TIERS = [
    ("Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", 5.5),
]
VLM_BATCH = 12


def models_dir():
    """モデルの置き場所。環境変数 MK8DX_MODELS_DIR で変えられる（既定は同梱の assets/models）。"""
    from pathlib import Path

    override = os.environ.get("MK8DX_MODELS_DIR")
    return Path(override) if override else resource_path("assets/models")


class RapidRecognizer:
    """PP-OCRv5 認識モデル（ONNX Runtime）。GPU が無い環境でも動く軽量版。"""

    name = "PP-OCRv5"
    skipped: list = []
    max_reads = 3          # 1選手あたりの読み取りレース数（複数回の多数決で安定させる）
    crops_per_read = 3     # 1回に読むクロップ数

    def __init__(self) -> None:
        self._engine = None

    def load(self) -> bool:
        if self._engine is not None:
            return True
        try:
            from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

            params = {
                "Global.use_det": False,
                "Global.use_cls": False,
                "Global.log_level": "error",
                "Rec.lang_type": LangRec.CH,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
            model = models_dir() / RAPID_MODEL_FILE
            if model.exists():
                params["Rec.model_path"] = str(model)
            self._engine = RapidOCR(params=params)
        except Exception:
            self._engine = None
        return self._engine is not None

    @property
    def available(self) -> bool:
        return self._engine is not None

    def recognize_many(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        """名前行の画像（BGR）を1枚ずつ読み、(テキスト, 信頼度) のリストを返す。"""
        out: list[tuple[str, float]] = []
        for image in images:
            if self._engine is None:
                out.append(("", 0.0))
                continue
            big = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            try:
                result = self._engine(big, use_det=False, use_cls=False, use_rec=True)
            except Exception:
                out.append(("", 0.0))
                continue
            texts = getattr(result, "txts", None)
            scores = getattr(result, "scores", None)
            if not texts:
                out.append(("", 0.0))
            else:
                out.append(("".join(texts), float(min(scores)) if scores else 0.0))
        return out


class VlmRecognizer:
    """Qwen3-VL（視覚言語モデル）で名前を書き起こす。NVIDIA GPU が必要。"""

    max_reads = 1          # 精度が高いので、選手ごとに最初の1回だけ読む
    crops_per_read = 1

    def __init__(self, folder: str, hf_id: str, min_free_gb: float) -> None:
        self.name = folder
        self._folder, self._hf_id, self._min_free_gb = folder, hf_id, min_free_gb
        self._model = None
        self._proc = None
        self.error = ""     # 使えなかった理由（診断用）

    def load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch

            if not torch.cuda.is_available():
                self.error = "CUDA が使えません（NVIDIA GPU・ドライバを確認）"
                return False
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            if free_gb < self._min_free_gb:
                self.error = f"GPUの空きメモリが足りません（{free_gb:.1f}GB < {self._min_free_gb}GB）"
                return False
            from transformers import AutoModelForImageTextToText, AutoProcessor

            local = models_dir() / self._folder
            if not local.exists() and is_frozen():
                # 配布版では勝手にダウンロードしない（同梱していない小型モデルなど）
                self.error = "モデルが同梱されていません"
                return False
            source = str(local) if local.exists() else self._hf_id
            offline = local.exists()
            kwargs = {"local_files_only": True} if offline else {}
            self._proc = AutoProcessor.from_pretrained(source, **kwargs)
            self._proc.tokenizer.padding_side = "left"
            self._model = AutoModelForImageTextToText.from_pretrained(
                source, dtype=torch.bfloat16, device_map="cuda", **kwargs)
            self._model.eval()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            if os.environ.get("MK8DX_DEBUG"):
                import traceback

                traceback.print_exc()
                cause = exc.__cause__ or exc.__context__
                if cause is not None:
                    traceback.print_exception(cause)
            self._model = None
            self._proc = None
        return self._model is not None

    @property
    def available(self) -> bool:
        return self._model is not None

    def recognize_many(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for i in range(0, len(images), VLM_BATCH):
            out.extend((t, 1.0) for t in self._run(images[i:i + VLM_BATCH]))
        return out

    def _run(self, images: list[np.ndarray]) -> list[str]:
        contents = [
            [{"type": "image", "image": _pil(img)}, {"type": "text", "text": VLM_PROMPT}]
            for img in images
        ]
        return self._generate(contents, max_new_tokens=40)

    def _generate(self, contents: list[list[dict]], max_new_tokens: int) -> list[str]:
        """メッセージ内容（画像・テキスト）ごとに、まとめて1回で生成する。失敗は空文字。"""
        if self._model is None or not contents:
            return [""] * len(contents)
        import torch

        messages = [[{"role": "user", "content": c}] for c in contents]
        try:
            inputs = self._proc.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
            ).to(self._model.device)
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            texts = self._proc.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)
            return [t.strip() for t in texts]
        except Exception:
            return [""] * len(contents)

    # -- チーム分けの推定（名前画像12枚を見て、同じタグの選手を組にする）----
    def suggest_teams(self, images: list[np.ndarray], team_size: int) -> list[dict] | None:
        """12枚の名前画像から [{"tag": 文字列, "members": [0始まりの行番号...]}] を返す。

        読めなかった・形式が違う場合は None。結果の妥当性（人数など）は呼び出し側で検証する。
        """
        if len(images) != 12:
            return None
        content: list[dict] = []
        for i, img in enumerate(images):
            content += [{"type": "text", "text": f"{i + 1}番:"},
                        {"type": "image", "image": _pil(img)}]
        content.append({"type": "text", "text": (
            f"上の12枚は、対戦ゲーム（{team_size}人1組で{12 // team_size}チーム）のプレイヤー名の画像です。"
            + TEAM_RULES +
            "どの選手が同じチームかを推定し、次のJSONだけを出力してください。"
            '{"teams":[{"tag":"共通のタグ","members":[番号,番号]}, ...]}')})
        raw = self._generate([content], max_new_tokens=500)[0]
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None
        try:
            teams = json.loads(match.group(0))["teams"]
            return [{"tag": str(t.get("tag", "")),
                     "members": [int(x) - 1 for x in t["members"]]} for t in teams]
        except Exception:
            return None

    def label_teams(self, groups: list[list[np.ndarray]]) -> list[str]:
        """チームごとの名前画像から、共通するタグ（チーム名）を書き起こす。"""
        contents = []
        for images in groups:
            content = [{"type": "image", "image": _pil(img)} for img in images]
            content.append({"type": "text", "text": (
                "これらは同じチームの選手名です。" + TEAM_RULES +
                "名前の先頭に共通するタグ（チーム名）だけを、記号も含めてそのまま書いてください。"
                "説明は不要です。")})
            contents.append(content)
        return self._generate(contents, max_new_tokens=30)


def _pil(img: np.ndarray):
    from PIL import Image

    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))


def create_recognizer(prefer: str | None = None):
    """使える中で最も精度の高い認識エンジンを返す（どれも使えなければ None）。

    環境変数 MK8DX_OCR で指定もできる: 'vlm'（VLMのみ）/ 'rapid'（軽量版のみ）。
    """
    choice = (prefer or os.environ.get("MK8DX_OCR", "")).lower()
    candidates: list = []
    if choice != "rapid":
        candidates += [VlmRecognizer(*tier) for tier in VLM_TIERS]
    if choice != "vlm":
        candidates.append(RapidRecognizer())
    errors: list[str] = []
    for rec in candidates:
        if rec.load():
            rec.skipped = errors     # 先に試して使えなかったエンジンの理由
            return rec
        if getattr(rec, "error", ""):
            errors.append(f"{rec.name}: {rec.error}")
    return None


# 旧名（互換）
NameRecognizer = RapidRecognizer
