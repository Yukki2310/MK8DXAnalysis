"""リザルトフレーム解析パイプライン（設計書 3.3 / 3.4 / 4 / 5.1）。

2フェーズの扱い（実画面で確認済み）:
- フェーズA「ゴール順画面」: 行はゴール順。スコア列は '+15' のように獲得点が
  '+' 付きで表示される。集計はこちらだけを使う。右端の累計列は加算
  アニメーション中は古い値のことがあり、確定値として使わない。
- フェーズB「累計順位画面」: 行は累計順位順。total列は累計点（'+' なし）。
  集計には使わず、自前計算の累計との照合（クロスチェック）専用。
判別はスコア列の '+' の有無で行う。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import Assignment, assign_teams
from .identity import extract_name_mask, trim_mask
from .ocr import OcrEngine, parse_points_text
from .preprocess import crop_rows, edge_density
from .roster import Observation, extract_flag_signature
from .scoring import POINTS, points_for_rank, validate_scores
from .tags import normalize_name

# フェーズ判別: スコア列の平均エッジ密度の閾値。
# 実キャプチャ2枚での実測は ゴール順画面=約0.070 / 累計画面=約0.031 だった。
SCORE_INK_THRESHOLD = 0.045

PHASE_RACE_RESULT = "race_result"
PHASE_STANDINGS = "standings"
PHASE_UNKNOWN = "unknown"


@dataclass
class RaceReading:
    entries: list[dict]
    validation_ok: bool
    problems: list[str]
    assignment: Assignment | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class StandingsReading:
    entries: list[dict]
    problems: list[str] = field(default_factory=list)


def classify_plain_phase(totals: list[int | None]) -> str:
    """スコア列が空の「プレーン表示」フレームを累計列の値で判別する。

    実動画検証（2026-11-18）で、フェーズAは加算アニメーション完了後に
    '+' が消え、累計列だけが残る表示になることを確認した。フェーズBも
    同じ見た目になるため、値のパターンで切り分ける:
    - 読めた累計がすべて配点表の値で、行順に完全な降順なら standings。
      これは「1レース目終了直後のフェーズB」の署名（累計＝配点で、
      累計順＝降順）。1レース目のフェーズA後半も原理的には同じ見た目に
      なるが、集計トリガーは '+' のあるフェーズA前半であり、照合用途では
      どちらでも（名前, 累計）の組は正しいため、standings に寄せる。
    - 行順の累計がほぼ降順（8割以上の隣接ペアが非増加）なら、
      累計順に並んだフェーズB → standings
    - それ以外（フェーズA後半の非ソートな累計、フェーズBの並び替え
      アニメーション途中など）はこの1枚では決められない → unknown
    ライブ運用では集計トリガーは '+' のあるフェーズA前半なので、
    unknown でも集計は止まらない。照合は前後の文脈で判断する。
    """
    readable = [t for t in totals if t is not None]
    if len(readable) >= 5:
        pairs = list(zip(readable, readable[1:]))
        descending = all(a > b for a, b in pairs)
        if descending and set(readable) <= set(POINTS.values()):
            return PHASE_STANDINGS
        if len(readable) >= 8 and pairs and (
            sum(1 for a, b in pairs if a >= b) / len(pairs) >= 0.8
        ):
            return PHASE_STANDINGS
    return PHASE_UNKNOWN


def classify_phase(frame: np.ndarray, roi: dict, ocr: OcrEngine) -> str:
    """フェーズを判別する。

    1. スコア列OCRで '+' と規定配点が読めればゴール順画面（フェーズA確定）。
       '+' だけが複数行で読める場合も、加算アニメーション途中の
       フェーズAとみなす（'+' は累計画面には現れない）。
    2. スコア列の平均エッジ密度が高く、かつ配点らしい値が複数行で
       読める場合もフェーズA。密度だけでは判定しない: パネルは半透明で、
       明るい背景が透けると累計画面でも密度が上がるため（実動画で
       誤判定を確認済み 2026-11-18）。
    3. スコア列が空（プレーン表示）の場合は累計列の値パターンで判別する
       （classify_plain_phase）。決められない場合は unknown を返す。
    """
    rows = crop_rows(frame, roi)
    plus_rows = 0
    valid_rows = 0
    for rank in range(1, 13):
        regions = rows.get(rank)
        if not regions or "score" not in regions:
            continue
        text, _ = ocr.recognize(regions["score"])
        value, has_plus = parse_points_text(text)
        if has_plus:
            plus_rows += 1
        if value is not None:
            valid_rows += 1
    if valid_rows >= 1 and plus_rows >= 1:
        return PHASE_RACE_RESULT
    if plus_rows >= 2:
        return PHASE_RACE_RESULT
    densities = [
        edge_density(regions["score"])
        for regions in rows.values()
        if "score" in regions
    ]
    if densities:
        mean_ink = sum(densities) / len(densities)
        if mean_ink >= SCORE_INK_THRESHOLD and valid_rows >= 3:
            return PHASE_RACE_RESULT
    totals: list[int | None] = []
    for rank in range(1, 13):
        regions = rows.get(rank)
        if regions and "total" in regions:
            totals.append(ocr.recognize_total(regions["total"]))
        else:
            totals.append(None)
    return classify_plain_phase(totals)


def verify_result_candidate(
    frame: np.ndarray, roi: dict, ocr: OcrEngine
) -> bool:
    """候補フレームが本物のリザルト画面か、順位列で検証する。

    レース前のプレイヤー紹介画面なども名前列インクが高く候補に
    なるため、順位列（1,2,11,12位）の数字OCRで2つ以上一致したら
    本物とみなす（ステップ2でコース選択画面の棄却に実績のある方法）。
    """
    rows = crop_rows(frame, roi)
    hits = 0
    for rank in (1, 2, 11, 12):
        value, _conf = ocr.recognize_digits_en(rows[rank]["rank"])
        if value is not None and int(value) == rank:
            hits += 1
    return hits >= 2


def extract_observations(
    frame: np.ndarray, roi: dict, ocr: OcrEngine
) -> list[Observation]:
    """フェーズAフレームから選手名簿照合用の観測（名前＋国旗署名）を抽出する。

    RaceProcessor.add_frame に渡すための橋渡し。名前は
    analyze_result_frame と同じ読み方（日英デュアルエンジン）をし、
    国旗は roster.extract_flag_signature で署名化する。
    """
    rows = crop_rows(frame, roi)
    observations: list[Observation] = []
    for rank in range(1, 13):
        regions = rows.get(rank)
        name = ""
        if regions and "name" in regions:
            name, _ = ocr.recognize_name(regions["name"])
        flag = extract_flag_signature(frame, roi, rank)
        mask = None
        if regions and "name" in regions:
            mask = trim_mask(extract_name_mask(regions["name"]))
        observations.append(Observation(name=name, flag=flag, mask=mask))
    return observations


def analyze_result_frame(
    frame: np.ndarray, roi: dict, ocr: OcrEngine, fmt: str
) -> RaceReading:
    """フェーズA（ゴール順画面）を解析する。"""
    rows = crop_rows(frame, roi)
    entries: list[dict] = []
    scores_by_rank: dict[int, int] = {}
    problems: list[str] = []

    for rank in range(1, 13):
        regions = rows.get(rank)
        if regions is None:
            problems.append(f"{rank}位のROIが未定義です")
            continue
        # 名前は生画像OCRが最も精度が高い（実キャプチャ検証 2026-11-18）
        name, name_conf = ocr.recognize_name(regions["name"])
        if not name:
            problems.append(f"{rank}位の名前を読み取れませんでした")
        # スコアは順位から規定配点で確定する。スコア列の数字はセグメント表示
        # フォントでOCRが安定しないため、読めた場合のクロスチェックにのみ使う。
        score = points_for_rank(rank)
        score_text, _ = ocr.recognize(regions["score"])
        ocr_score, _ = parse_points_text(score_text)
        if ocr_score is not None and ocr_score != score:
            problems.append(
                f"{rank}位のスコアOCR '{score_text}' が規定配点 {score} と一致しません"
            )
        total_shown = (
            ocr.recognize_total(regions["total"]) if "total" in regions else None
        )
        scores_by_rank[rank] = score
        entries.append(
            {
                "rank": rank,
                "name": name,
                "score": score,
                "score_source": "rank_table",
                "score_ocr_text": score_text,
                "total_shown": total_shown,
                "name_confidence": name_conf,
            }
        )

    validation = None
    if len(scores_by_rank) == 12:
        validation = validate_scores(scores_by_rank)
        problems.extend(validation.problems)

    assignment: Assignment | None = None
    warnings: list[str] = []
    names = [e["name"] for e in entries]
    if len(names) == 12 and all(names):
        assignment = assign_teams(names, fmt)
        warnings.extend(assignment.warnings)
        if not assignment.valid:
            # タグなし名前（CPU名など）では通常起きる。読み取り自体の失敗ではない
            warnings.append("チーム割り当てに不整合があります（要手動確認）")
    else:
        problems.append("プレイヤー名を12名分読み取れませんでした")

    ok = not problems and (validation is None or validation.ok)
    return RaceReading(
        entries=entries,
        validation_ok=ok,
        problems=problems,
        assignment=assignment,
        warnings=warnings,
    )


def analyze_standings_frame(
    frame: np.ndarray, roi: dict, ocr: OcrEngine
) -> StandingsReading:
    """フェーズB（累計順位画面）を読み取る。照合専用で、集計には使わない。"""
    rows = crop_rows(frame, roi)
    entries: list[dict] = []
    problems: list[str] = []
    for position in range(1, 13):
        regions = rows.get(position)
        if regions is None:
            problems.append(f"{position}行目のROIが未定義です")
            continue
        name, _ = ocr.recognize_name(regions["name"])
        # 累計点はセグメントフォントのため専用リーダー（en+拡大+二値化の
        # 多数決）で best-effort に読む。読めなくても名前の照合は継続できる。
        total = (
            ocr.recognize_total(regions["total"]) if "total" in regions else None
        )
        if not name:
            problems.append(f"{position}行目の名前を読み取れませんでした")
        entries.append({"position": position, "name": name, "total": total})
    return StandingsReading(entries=entries, problems=problems)


def cross_check_standings(
    computed_totals: dict[str, int], standings: StandingsReading
) -> list[str]:
    """自前の累計（名前->累計点）とフェーズBの表示を照合し、不一致を返す。"""
    problems: list[str] = []
    computed = {normalize_name(k): v for k, v in computed_totals.items()}
    shown_order: list[str] = []
    for e in standings.entries:
        key = normalize_name(e["name"])
        shown_order.append(key)
        if key not in computed:
            problems.append(f"累計画面の '{e['name']}' が集計側に存在しません")
        elif e["total"] is not None and computed[key] != e["total"]:
            problems.append(
                f"'{e['name']}' の累計が一致しません: 集計={computed[key]} / 画面={e['total']}"
            )
    expected_order = [
        k for k, _ in sorted(computed.items(), key=lambda kv: -kv[1])
    ]
    if shown_order != expected_order:
        problems.append("累計順位の並びが集計結果と一致しません")
    return problems
