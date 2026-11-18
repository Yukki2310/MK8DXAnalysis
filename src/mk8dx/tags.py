"""プレイヤー名・タグの正規化（設計書 4.1 ラウンジルール準拠）。

ルール:
1. 同一チームは必ず同じ文字から始まるタグを使う（先頭文字の一致が前提）
2. 英字の大文字・小文字は同一視する
3. ひらがな・カタカナは同一視する（すべてカタカナに統一）
4. 濁音・半濁音は厳格に区別する（「タ」「ダ」「パ」は別文字。
   Unicode正規化で濁点を分離・除去しないこと）
5. 分音記号付きラテン文字（ä, é 等）への代用は禁止ルールとして検知する
"""
from __future__ import annotations

import unicodedata

# ひらがな(U+3041–U+3096) -> カタカナ は +0x60
_HIRA_START = 0x3041
_HIRA_END = 0x3096


def _hiragana_to_katakana(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if _HIRA_START <= code <= _HIRA_END:
            out.append(chr(code + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def normalize_name(name: str) -> str:
    """比較用に名前を正規化する。

    手順:
    - NFKC: 半角カナ・全角英数などを統一（濁点付きカナは合成文字に戻る）
    - NFC: 結合濁点（か+゙）を合成文字（が）にそろえる。
      これにより濁音は1文字として保持され、清音と区別される
    - ひらがな -> カタカナ統一
    - 英字を小文字化
    前後の空白は除去する。
    """
    s = unicodedata.normalize("NFKC", name)
    s = unicodedata.normalize("NFC", s)
    s = _hiragana_to_katakana(s)
    s = s.lower()
    return s.strip()


def first_char_key(name: str) -> str:
    """正規化後の先頭1文字（チーム判定の最小キー）。"""
    n = normalize_name(name)
    return n[0] if n else ""


def has_forbidden_diacritic(raw_name: str) -> bool:
    """分音記号付きラテン文字（代用禁止）の使用を検知する。

    ラテン文字が合成済み分音記号付き文字（ä, é, ñ 等）の場合に True。
    日本語の濁点（か+゙ 等）はラテン文字ではないため対象外。
    """
    for ch in raw_name:
        if "LATIN" not in unicodedata.name(ch, ""):
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        # 基底文字 + 結合記号 に分解される = 分音記号付き
        if len(decomposed) > 1 and any(
            unicodedata.combining(c) for c in decomposed[1:]
        ):
            return True
    return False


def common_prefix(names: list[str], *, normalized: bool = True) -> str:
    """名前リストの共通接頭辞を返す（タグ候補の抽出用）。"""
    if not names:
        return ""
    items = [normalize_name(n) for n in names] if normalized else list(names)
    prefix = items[0]
    for item in items[1:]:
        i = 0
        while i < len(prefix) and i < len(item) and prefix[i] == item[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix
