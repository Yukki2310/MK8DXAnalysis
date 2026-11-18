#!/usr/bin/env python3
"""評価用フレーム抽出（実用化計画フェーズ0）。

動画（ローカルファイル、または yt-dlp 経由の YouTube URL）の指定区間から
1080p フレームを一定間隔で PNG 保存する。保存先は既定で fixtures/frames/
（.gitignore 対象。動画・フレームはリポジトリに入れない）。

使い方:
  python tools/extract_frames.py --url https://www.youtube.com/watch?v=<動画ID> \\
      --section 5:40-6:00 --section 12:00-12:20 --fps 2
  python tools/extract_frames.py --video local.mp4 --section 0:10-0:30

必要な外部コマンド: ffmpeg（常に）、yt-dlp（--url のとき: pip install yt-dlp）。
出力: fixtures/frames/<tag>/<開始秒>_<連番>.png と manifest.json
（各フレームの元動画時刻。ラベル付けやベンチの入力になる）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_time(text: str) -> float:
    """'5:40' / '1:02:03' / '340' を秒に変換する。"""
    parts = [float(p) for p in text.split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def parse_section(text: str) -> tuple[float, float]:
    start, _, end = text.partition("-")
    if not end:
        raise ValueError(f"区間は 開始-終了 の形式で指定してください: {text}")
    s, e = parse_time(start), parse_time(end)
    if e <= s:
        raise ValueError(f"終了が開始以前です: {text}")
    return s, e


def require(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        sys.exit(f"{cmd} が見つかりません。インストールして PATH に通してください。")
    return path


def download_sections(url: str, sections: list[tuple[float, float]],
                      workdir: Path) -> list[Path]:
    """yt-dlp で区間だけ 1080p 以下の mp4 として取得する。"""
    ytdlp = require("yt-dlp")
    files: list[Path] = []
    for s, e in sections:
        out = workdir / f"sec_{int(s)}_{int(e)}.mp4"
        subprocess.run(
            [
                ytdlp, "-f", "bv*[height<=1080]+ba/b[height<=1080]",
                "--download-sections", f"*{s}-{e}", "--force-keyframes-at-cuts",
                "--merge-output-format", "mp4", "-o", str(out), url,
            ],
            check=True,
        )
        files.append(out)
    return files


def extract(video: Path, offset: float, duration: float | None, fps: float,
            outdir: Path) -> list[dict]:
    """ffmpeg でフレームを PNG 出力し、manifest 用の記録を返す。"""
    ffmpeg = require("ffmpeg")
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{int(offset):06d}"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(video)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", f"fps={fps},scale=1920:1080",
            str(outdir / f"{prefix}_%04d.png")]
    subprocess.run(cmd, check=True)
    records = []
    for i, p in enumerate(sorted(outdir.glob(f"{prefix}_*.png"))):
        records.append({"file": p.name, "source_time": round(offset + i / fps, 3)})
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube などの URL（yt-dlp 使用）")
    src.add_argument("--video", type=Path, help="ローカル動画ファイル")
    ap.add_argument("--section", action="append", required=True,
                    help="抽出区間 例: 5:40-6:00（複数指定可）")
    ap.add_argument("--fps", type=float, default=2.0, help="抽出fps（既定2）")
    ap.add_argument("--tag", default="default", help="出力サブフォルダ名")
    ap.add_argument("--out", type=Path, default=ROOT / "fixtures" / "frames")
    args = ap.parse_args()

    sections = [parse_section(s) for s in args.section]
    outdir = args.out / args.tag
    records: list[dict] = []

    if args.url:
        with tempfile.TemporaryDirectory() as tmp:
            files = download_sections(args.url, sections, Path(tmp))
            for (s, e), f in zip(sections, files):
                records += extract(f, s, None, args.fps, outdir)
    else:
        for s, e in sections:
            # ローカル動画は -ss で頭出ししたいので一旦切り出す
            ffmpeg = require("ffmpeg")
            with tempfile.TemporaryDirectory() as tmp:
                clip = Path(tmp) / "clip.mp4"
                subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-ss", str(s), "-t",
                     str(e - s), "-i", str(args.video), "-c", "copy", str(clip)],
                    check=True)
                records += extract(clip, s, e - s, args.fps, outdir)

    manifest = outdir / "manifest.json"
    manifest.write_text(
        json.dumps({"frames": records}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"{len(records)} フレームを {outdir} に保存しました")


if __name__ == "__main__":
    main()
