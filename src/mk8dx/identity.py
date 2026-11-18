"""画像ベースのプレイヤー識別（実用化計画フェーズ1、docs/phase1-notes.md 参照）。

名前を OCR で文字列として読む代わりに、名前行の「文字だけを取り出した
二値マスク」同士の類似度で同一人物かを判定する。同じプレイヤーの名前は
毎レース同じフォント・同じ大きさで描画されるため、背景（半透明パネル越しの
コース映像）を除けば画像はほぼ一致する。

実フレーム（ラウンジ動画、2026-11-18）での実測:
- 同一人物の距離は 0.01〜0.2、別人は 0.6 以上（'ranger #1' と '#2' のような
  酷似ペアでも 0.2 以上）。しきい値は SAME_THRESHOLD を参照。
- 通常行は白文字（暗い輪郭）。自分の行（黄色ハイライト）は暗い灰色の文字が
  黄色地に乗る。極性が逆なので別の抽出を使う。
- 行ごとに縦位置がずれる（累計画面の並べ替え途中など）ため、文字の
  外接矩形で切り出してから比較する。
"""
from __future__ import annotations

import cv2
import numpy as np

# 名前の文字領域に使う幅。これより右は国旗なので除外する。
NAME_TEXT_WIDTH = 365
# 同一人物とみなす距離の上限。実測の同一人物 ≤0.2 / 別人 ≥0.2(酷似)〜0.6。
SAME_THRESHOLD = 0.12
# 行の文字として必要な、局所的な明るさのピーク（これ未満は文字なしの行）
MIN_TEXT_PEAK = 45
# マスクが小さすぎる（文字が抽出できていない）行は比較不能とする
MIN_MASK_PIXELS = 40
NO_MATCH = 9.0

_KERNEL15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def is_highlight_row(row: np.ndarray) -> bool:
    """自分の行（黄色ハイライト）か。G・R が明るく B が暗い行。"""
    return float(row[:, :, 1:].mean()) > 110


def _remove_noise(mask: np.ndarray) -> np.ndarray:
    """文字ではない成分を除く。

    - 行の上下端に接する成分（区切り線・隣の行の欠片）と極小成分
    - 背景の明るい細い横線（幅が広く高さが小さい成分）
    - 文字の帯（字形らしい大きさの成分が占める縦範囲）から外れた成分
    これらが外接矩形を広げると、同じ名前でも距離が大きくなる
    （実フレームで同じ名前が 0.85 になった原因）。
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h = mask.shape[0]
    keep = np.zeros(n, dtype=bool)
    glyph_like = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < 6 or y == 0 or y + hh >= h:
            continue
        if w >= 80 and hh <= 4:
            continue
        keep[i] = True
        glyph_like[i] = hh >= 8 and area >= 20
    if glyph_like.any():
        top = min(stats[i][1] for i in range(1, n) if glyph_like[i])
        bottom = max(stats[i][1] + stats[i][3] for i in range(1, n) if glyph_like[i])
        for i in range(1, n):
            if keep[i] and not glyph_like[i]:
                cy = stats[i][1] + stats[i][3] / 2
                if cy < top - 6 or cy > bottom + 6:
                    keep[i] = False
    return keep[labels].astype(np.uint8)


def extract_name_mask(name_region: np.ndarray) -> np.ndarray:
    """名前領域(BGR)から文字の二値マスク(uint8, 0/1)を取り出す。

    文字の色は行によって逆になる: 通常は白文字、自分の行（黄色のバー）は
    暗い灰色の文字。背景が明るい（黄土色・白っぽいコース）と通常の行でも
    平均輝度が高くなるため、輝度だけで極性を決めると誤る。そこで白文字用と
    暗文字用の両方で抽出し、「面積が大きく成分が少ない」（=文字らしい）方を
    採用する。
    """
    c = name_region[:, :NAME_TEXT_WIDTH]
    candidates = [_remove_noise(_white_text_mask(c)), _remove_noise(_dark_text_mask(c))]
    # 同程度の品質なら、黄色の行は暗文字・それ以外は白文字という事前知識を使う
    prior = 1 if is_highlight_row(c) else 0

    def quality(mask: np.ndarray) -> float:
        n, _ = cv2.connectedComponents(mask, connectivity=8)
        return float(mask.sum()) / n  # n は背景を含むので 1 + 成分数

    scores = [quality(m) for m in candidates]
    scores[prior] *= 1.15
    return candidates[int(np.argmax(scores))]


def _dark_text_mask(c: np.ndarray) -> np.ndarray:
    """暗い文字 on 明るい地（黄色のバー）: 局所的に背景より暗い画素。"""
    g = c.max(axis=2)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, _KERNEL15)
    return (cv2.subtract(bg, g) >= 45).astype(np.uint8)


def _white_text_mask(c: np.ndarray) -> np.ndarray:
    # 白文字: 全チャンネルが明るい（min チャンネル）かつ局所的に明るい。
    # 文字の明るさは映像ごとに違う（圧縮やぼけで灰白色になる）ため、固定の
    # しきい値ではなく「その行の文字の明るさ」に合わせたヒステリシス二値化にする。
    g = c.min(axis=2)
    bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, _KERNEL15)
    th = cv2.subtract(g, bg)
    peak = float(np.percentile(th, 99.7))
    if peak < MIN_TEXT_PEAK:
        return np.zeros_like(g, dtype=np.uint8)
    low = (th >= max(0.30 * peak, 22)) & (g >= 100)
    high = th >= 0.55 * peak
    n, labels = cv2.connectedComponents(low.astype(np.uint8), connectivity=8)
    seeds = np.unique(labels[high & low])
    keep = np.zeros(n, dtype=bool)
    keep[seeds[seeds > 0]] = True
    return keep[labels].astype(np.uint8)


def trim_mask(mask: np.ndarray) -> np.ndarray | None:
    """文字の外接矩形で切り出す（縦横ずれの吸収）。文字が無ければ None。"""
    ys, xs = np.where(mask > 0)
    if len(xs) < MIN_MASK_PIXELS:
        return None
    return mask[
        max(ys.min() - 2, 0): ys.max() + 3,
        max(xs.min() - 2, 0): xs.max() + 3,
    ].astype(np.float32)


def prepare_mask(mask: np.ndarray | None) -> np.ndarray | None:
    """比較用にぼかす（小さなずれの許容）。繰り返し比較する側で1回だけ行う。"""
    if mask is None:
        return None
    return cv2.GaussianBlur(mask, (0, 0), 1.2)


def prepared_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """prepare_mask 済みマスク同士の距離（小さいほど同一）。0〜約2。

    ±6px の平行移動で最大の正規化相互相関を取る。大きさ（幅・高さ）の
    違いはペナルティとして加える。
    """
    if a is None or b is None:
        return NO_MATCH
    if a.shape[1] < b.shape[1]:
        a, b = b, a
    padded = cv2.copyMakeBorder(a, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=0)
    if b.shape[0] > padded.shape[0] or b.shape[1] > padded.shape[1]:
        return NO_MATCH
    corr = float(cv2.matchTemplate(padded, b, cv2.TM_CCOEFF_NORMED).max())
    penalty = abs(a.shape[1] - b.shape[1]) / max(a.shape[1], b.shape[1]) + abs(
        a.shape[0] - b.shape[0]
    ) / max(a.shape[0], b.shape[0])
    return 1.0 - corr + penalty


def mask_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """トリム済みマスク同士の距離（ぼかしも内部で行う）。"""
    return prepared_distance(prepare_mask(a), prepare_mask(b))


def mask_to_dict(mask: np.ndarray) -> dict:
    """マスクを JSON 化できる形（ビットパック＋base64）に変換する。"""
    import base64

    bits = np.packbits(mask > 0.5)
    return {"shape": list(mask.shape),
            "bits": base64.b64encode(bits.tobytes()).decode("ascii")}


def mask_from_dict(data: dict) -> np.ndarray:
    import base64

    h, w = data["shape"]
    bits = np.frombuffer(base64.b64decode(data["bits"]), dtype=np.uint8)
    return np.unpackbits(bits)[: h * w].reshape(h, w).astype(np.float32)


def distance_matrix(
    left: list[np.ndarray | None], right: list[np.ndarray | None]
) -> np.ndarray:
    """トリム済みマスク列同士の距離行列 (len(left) × len(right))。"""
    return np.array([[mask_distance(a, b) for b in right] for a in left])
