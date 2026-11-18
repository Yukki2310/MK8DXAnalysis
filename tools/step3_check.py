"""ステップ3検証: 選手名簿照合をステップ2の実データで検証する。

使い方:
    python tools/step3_check.py

ステップ2のレポート（data/reports/step2_*.json の受理区間の名前）と
保存フレーム（data/frames/<section>/*.png から抽出する国旗署名）を
入力に、2つの運用モードを検証する。

- モードA（事前登録＋closed）: レース3・4の正規選手名簿を seed し、
  assume_closed=True で照合する。2v2 クラスタリングの成立と、
  レース4フェーズBの（名前, 累計）一致率を見る。
- モードB（自動構築）: 選手名簿空からレース1→3→4の順に流し、
  仮登録・本登録・名寄せの挙動とクラスタリング成立可否を見る。
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mk8dx.clustering import assign_teams
from mk8dx.preprocess import load_roi_config
from mk8dx.roster import Observation, Roster, extract_flag_signature

ROOT = Path(__file__).resolve().parent.parent

# 目視で確定したレース3・4の正規選手名簿
CANONICAL_R34 = [
    "xı c.qq", "xı avalon", "FR", "FR Flowx", "racist #1", "racist #2",
    "Xb", "Xb◇ねえあまぁƄ", "RR★", "RR★",
    "A★ぜんだかいとしき", "A A",
]
# レース4フェーズBの（名前 -> 累計）目視確定値。RR★は2名分を集合で持つ
TRUTH_TOTALS_R4B = {
    "xı c.qq": 46, "A A": 41, "FR Flowx": 39, "A★ぜんだかいとしき": 37,
    "xı avalon": 36, "Xb": 34, "racist #1": 32, "racist #2": 30,
    "Xb◇ねえあまぁƄ": 27, "FR": 21,
}
TRUTH_RR_TOTALS = {36, 31}


def load_segment(section: str) -> tuple[list[Observation], list[dict]]:
    """ステップ2レポートの受理区間から観測（名前＋国旗署名）を作る。"""
    report = json.loads(
        (ROOT / f"data/reports/step2_{section}.json").read_text(encoding="utf-8")
    )
    roi = load_roi_config(str(ROOT / "config/roi_1080p.json"))
    frames = sorted(glob.glob(str(ROOT / f"data/frames/{section}/*.png")))
    frame = cv2.imread(frames[0]) if frames else None
    for seg in report["segments"]:
        if seg["phase"] == "rejected":
            continue
        entries = sorted(
            seg["entries"], key=lambda e: e.get("rank") or e.get("position")
        )
        observations = []
        for i, e in enumerate(entries, start=1):
            flag = (
                extract_flag_signature(frame, roi, i)
                if frame is not None else None
            )
            observations.append(Observation(name=e["name"], flag=flag))
        return observations, entries
    raise RuntimeError(f"{section}: 受理区間がありません")


def show(label: str, roster: Roster, resolution, entries: list[dict]) -> None:
    names = roster.resolved_names(resolution)
    print(f"-- {label}")
    for i, (obs_name, resolved) in enumerate(
        zip([e["name"] for e in entries], names), start=1
    ):
        mark = " (低信頼)" if resolution.low_confidence[i - 1] else ""
        print(f"   {i:>2}. {obs_name!r:28} -> {resolved!r}{mark}")
    filled = [n for n in names if n]
    if len(filled) == 12:
        a = assign_teams(filled, "2v2")
        print(f"   2v2 clustering: valid={a.valid}")
        if a.valid:
            for t in a.teams:
                print(f"     team {t.tag!r}: rows {[m + 1 for m in t.members]}")
    else:
        print(f"   resolved {len(filled)}/12 (clustering は行わない)")


def main() -> None:
    seg = {s: load_segment(s) for s in ["330", "710", "870", "1070"]}

    print("===== モードA: 事前登録選手名簿（closed） =====")
    roster = Roster()
    # RR★ は同名2名: 710フレームの国旗（5行目=ベルギー、11行目=日本）で seed
    obs710, _ = seg["710"]
    flag_be = obs710[4].flag
    flag_jp = obs710[10].flag
    for name in CANONICAL_R34:
        if name == "RR★":
            continue
        roster.seed(name)
    roster.seed("RR★", flag=flag_jp)
    roster.seed("RR★", flag=flag_be)

    for section, race, label in [
        ("710", 3, "レース3 フェーズA"),
        ("870", 4, "レース4 フェーズA"),
        ("1070", 4, "レース4 フェーズB"),
    ]:
        observations, entries = seg[section]
        res = roster.resolve_race(observations, race, assume_closed=True)
        show(label, roster, res, entries)
        if section == "1070":
            names = roster.resolved_names(res)
            ok = 0
            rr_totals = set()
            for name, e in zip(names, entries):
                total = e.get("total")
                if name == "RR★":
                    rr_totals.add(total)
                elif name in TRUTH_TOTALS_R4B and TRUTH_TOTALS_R4B[name] == total:
                    ok += 1
            rr_ok = rr_totals == TRUTH_RR_TOTALS
            print(f"   (名前,累計) 一致: {ok}/10 + RR★2名 {'一致' if rr_ok else f'不一致 {rr_totals}'}")

    print()
    print("===== モードB: 選手名簿自動構築（open） =====")
    roster = Roster()
    for section, race, label in [
        ("330", 1, "レース1 フェーズA"),
        ("710", 3, "レース3 フェーズA"),
        ("870", 4, "レース4 フェーズA"),
        ("1070", 4, "レース4 フェーズB"),
    ]:
        observations, entries = seg[section]
        res = roster.resolve_race(observations, race)
        show(label, roster, res, entries)
    confirmed = sum(1 for e in roster.entries if e.confirmed)
    print(f"   選手名簿: 全{len(roster.entries)}エントリ（本登録 {confirmed}）")


if __name__ == "__main__":
    main()
