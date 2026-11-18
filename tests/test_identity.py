"""identity.py（画像ベース識別）のテスト。実画像は使わず合成画像で検証する。"""
import cv2
import numpy as np

from mk8dx.identity import (
    SAME_THRESHOLD,
    distance_matrix,
    extract_name_mask,
    mask_distance,
    trim_mask,
)

H, W = 72, 525


def _row(text: str, *, highlight=False, bg=40, y=50, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    if highlight:
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (0, 215, 255)  # 黄色(BGR)
        color, outline = (70, 70, 70), None
    else:
        img = np.full((H, W, 3), bg, np.uint8)
        if noise:
            img = np.clip(
                img.astype(int) + rng.integers(-noise, noise, img.shape), 0, 255
            ).astype(np.uint8)
        color, outline = (255, 255, 255), (0, 0, 0)
    if outline:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_DUPLEX, 1.2, outline, 5)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
    return img


def _mask(img):
    return trim_mask(extract_name_mask(img))


def test_same_text_different_background_is_close():
    a = _mask(_row("racist 1", bg=30, noise=15, seed=1))
    b = _mask(_row("racist 1", bg=80, noise=25, seed=2))
    assert mask_distance(a, b) < SAME_THRESHOLD


def test_different_text_is_far():
    a = _mask(_row("racist 1"))
    b = _mask(_row("Xb"))
    assert mask_distance(a, b) > 0.4


def test_vertical_offset_is_absorbed():
    a = _mask(_row("FR", y=35))
    b = _mask(_row("FR", y=60))
    assert mask_distance(a, b) < SAME_THRESHOLD


def test_highlight_row_matches_itself_across_frames():
    a = _mask(_row("RR", highlight=True))
    b = _mask(_row("RR", highlight=True, y=45))
    assert mask_distance(a, b) < SAME_THRESHOLD
    assert mask_distance(a, _mask(_row("Xb", highlight=True))) > 0.3


def test_empty_row_has_no_mask():
    blank = np.full((H, W, 3), 30, np.uint8)
    assert trim_mask(extract_name_mask(blank)) is None
    assert mask_distance(None, _mask(_row("FR"))) > 5


def test_distance_matrix_picks_true_partner():
    names = ["xi c.qq", "FR", "racist 1", "Xb"]
    left = [_mask(_row(n, bg=30, noise=10, seed=i)) for i, n in enumerate(names)]
    right = [_mask(_row(n, bg=70, noise=20, seed=10 + i)) for i, n in
             reversed(list(enumerate(names)))]
    m = distance_matrix(left, right)
    assert list(m.argmin(axis=1)) == [3, 2, 1, 0]


def test_edge_touching_fragments_do_not_widen_mask():
    clean = _row("xi avalon")
    dirty = clean.copy()
    dirty[H - 1, 100:140] = 255   # 区切り線の欠片（下端に接する）
    dirty[0, 200:230] = 255       # 上端
    assert mask_distance(_mask(clean), _mask(dirty)) < SAME_THRESHOLD
