"""rankcheck.py（順位列テンプレート）と paths.py のテスト。"""
import cv2
import numpy as np

from mk8dx import paths
from mk8dx.rankcheck import (
    MIN_MATCHED_ROWS,
    build_templates,
    load_templates,
    matched_rows,
    save_templates,
    score_column_ink,
)

ROI = {"rows": {
    str(r): {
        "rank": [0, 80 * (r - 1), 115, 72],
        "score": [120, 80 * (r - 1), 150, 72],
    }
    for r in range(1, 13)
}}


def _frame(draw_ranks=True, plus=False):
    img = np.full((960, 300, 3), 40, np.uint8)
    for r in range(1, 13):
        if draw_ranks:
            cv2.putText(img, str(r), (10, 80 * (r - 1) + 55),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3)
        if plus and r == 1:
            cv2.putText(img, "+15", (130, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                        (255, 255, 255), 2)
    return img


def test_templates_match_result_like_frame_only():
    templates = build_templates([_frame(), _frame()], ROI)
    assert matched_rows(_frame(), ROI, templates) == 12
    assert matched_rows(_frame(draw_ranks=False), ROI, templates) < MIN_MATCHED_ROWS


def test_templates_roundtrip(tmp_path):
    templates = build_templates([_frame()], ROI)
    path = tmp_path / "t.npz"
    save_templates(path, templates)
    loaded = load_templates(path)
    assert set(loaded) == set(templates)
    assert matched_rows(_frame(), ROI, loaded) == 12


def test_score_column_ink_detects_plus():
    assert score_column_ink(_frame(plus=True), ROI) > score_column_ink(_frame(), ROI)
    assert score_column_ink(_frame(), ROI) == 0


def test_data_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MK8DX_DATA_DIR", str(tmp_path / "d"))
    assert paths.data_dir() == tmp_path / "d"
    assert (tmp_path / "d").is_dir()


def test_resource_path_points_to_repo_in_dev():
    assert paths.resource_path("web/admin.html").exists()
