import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

try:
    from lxml import etree as ET  # type: ignore[import]
except ImportError:  # pragma: no cover
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]

try:
    import mwparserfromhell as mwp  # type: ignore[import]
except ImportError:  # pragma: no cover
    mwp = None

try:
    from bs4 import BeautifulSoup  # type: ignore[import]
except ImportError:  # pragma: no cover
    BeautifulSoup = None

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATASET_PATH = DATA_DIR / "wiki_pairs.txt"
SEED = 42
random.seed(SEED)

VI_DIACRITIC_CHARS = "àáảãạâầấẩẫậăằắẳẵặđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơớờởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
VI_ALPHABET = "aăâbcdđeêghiklmnoôơpqrstuưvxyAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXY"
MIN_ARTICLE_WORDS = 150
FALLBACK_ARTICLE_CHARS = 1000
MAX_ARTICLES = 100000


def remove_diacritics(text: str) -> str:
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def _strip_templates(text: str) -> str:
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\{\|[\s\S]*?\|\}", " ", text)
    return text


LANGUAGE_CODE_RE = re.compile(r"^[a-z]{2,3}$", re.IGNORECASE)
PIPE_SEGMENT_RE = re.compile(r"\|[^|\n]*\|")


def is_mostly_vietnamese(text: str, threshold: float = 0.5) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    viet_chars = sum(1 for c in letters if c in VI_ALPHABET or c in VI_DIACRITIC_CHARS)
    return (viet_chars / len(letters)) >= threshold


def _token_has_non_latin_chars(token: str) -> bool:
    stripped = token.strip("-–—:;.,!?()[]{}<>|\"\'")
    if not stripped:
        return False
    for ch in stripped:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if "LATIN" not in name:
                return True
    return False


def strip_non_latin_sequences(text: str) -> str:
    tokens = text.split()
    cleaned: List[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if _token_has_non_latin_chars(token):
            idx += 1
            continue
        token_core = token.rstrip(":|-\u2013\u2014")
        if LANGUAGE_CODE_RE.fullmatch(token_core.lower()) and idx + 1 < len(tokens):
            next_token = tokens[idx + 1]
            if _token_has_non_latin_chars(next_token):
                idx += 2
                continue
        cleaned.append(token)
        idx += 1
    return " ".join(cleaned)


def remove_pipe_segments(text: str) -> str:
    """Remove wiki pipe segments such as |thumb|right|250px| entirely."""
    previous = None
    while previous != text:
        previous = text
        text = PIPE_SEGMENT_RE.sub(" ", text)
    return text


def remove_ascii_token_lists(text: str, min_tokens: int = 6, ascii_threshold: float = 0.8) -> str:
    lines: List[str] = []
    for line in text.splitlines():
        tokens = re.split(r"[\s,]+", line.strip())
        tokens = [tok for tok in tokens if tok]
        if len(tokens) < min_tokens:
            lines.append(line)
            continue
        ascii_tokens = sum(1 for tok in tokens if re.fullmatch(r"[a-z0-9_:\-\.]{1,30}", tok.lower()))
        viet_chars = sum(1 for ch in line if ch in VI_ALPHABET or ch in VI_DIACRITIC_CHARS)
        if ascii_tokens >= int(ascii_threshold * len(tokens)) and viet_chars < 3:
            continue
        lines.append(line)
    return "\n".join(lines)


def filter_non_vietnamese_paragraphs(text: str, threshold: float = 0.45, min_len: int = 30) -> str:
    kept: List[str] = []
    for paragraph in re.split(r"\n{1,}", text):
        para = paragraph.strip()
        if not para:
            continue
        if len(para) < min_len:
            if is_mostly_vietnamese(para, threshold=0.6):
                kept.append(para)
            continue
        if is_mostly_vietnamese(para, threshold=threshold) or any(c in VI_DIACRITIC_CHARS for c in para):
            kept.append(para)
    return "\n\n".join(kept)


def wikitext_to_plain(text: str) -> str:
    if not text:
        return ""
    original = text
    if mwp is not None:
        try:
            parsed = mwp.parse(text)
            text = parsed.strip_code()
        except Exception:  # pragma: no cover - best effort fallback
            text = original
    text = _strip_templates(text)
    text = re.sub(r"\[\[(?:File|Image|Tập tin|Ảnh|Media):[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"(?ims)^={2,}\s*(?:tham khảo|liên kết ngoài|xem thêm|chú thích|nguồn|tài liệu tham khảo|thể loại)\s*={2,}.*?(?=^={2,}|\Z)", " ", text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"<!--[\s\S]*?-->", " ", text)
    text = re.sub(r"<ref[^>]*>[\s\S]*?</ref>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^[ \t]*[\*#].*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"http[s]?://\S+", " ", text)
    if BeautifulSoup is not None:
        text = BeautifulSoup(text, "lxml").get_text(" ")
    text = remove_pipe_segments(text)
    text = remove_ascii_token_lists(text)
    text = filter_non_vietnamese_paragraphs(text)
    text = strip_non_latin_sequences(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?;:])\s+")


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) >= 5]


def iter_wiki_text(xml_path: Path, max_articles: Optional[int] = None) -> Iterator[str]:
    # Some dumps use different namespaces or no namespace; be flexible and
    # accept any element whose tag ends with 'text'. Use iterparse without a
    # tag filter and check the element tag at runtime.
    context = ET.iterparse(str(xml_path), events=("end",))
    count = 0
    for _, elem in context:
        tag = getattr(elem, "tag", "")
        # handle namespaced tags like '{...}text' or plain 'text'
        if isinstance(tag, str) and (tag.endswith('}text') or tag == 'text'):
            if elem.text:
                yield elem.text
                count += 1
                if max_articles and count >= max_articles:
                    break
        # free memory
        elem.clear()


def generate_pairs(
    xml_path: Path,
    max_articles: int = 5000,
    min_sentence_len: int = 20,
    max_sentence_len: int = 300,
    min_article_words: int = MIN_ARTICLE_WORDS,
    fallback_article_chars: int = FALLBACK_ARTICLE_CHARS,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    articles = 0
    sentences_seen = 0
    pairs_found = 0
    for raw_text in iter_wiki_text(xml_path, max_articles=max_articles):
        if raw_text.lstrip().lower().startswith("#redirect"):
            continue
        articles += 1
        plain = wikitext_to_plain(raw_text)
        word_count = len(plain.split())
        if word_count < min_article_words and len(plain) < fallback_article_chars:
            continue
        for sentence in split_sentences(plain):
            sentences_seen += 1
            if len(sentence) < min_sentence_len:
                continue
            if len(sentence) > max_sentence_len:
                continue
            trailing = sentence.rstrip(")]\"'»›««")
            if not trailing or trailing[-1] not in ".?!,":
                continue
            stripped = remove_diacritics(sentence)
            if stripped == sentence:
                continue
            src_tokens = stripped.split()
            tgt_tokens = sentence.split()
            if len(src_tokens) != len(tgt_tokens):
                continue
            pairs.append((stripped, sentence))
            pairs_found += 1
        # occasional progress print for long dumps
        if articles % 500 == 0:
            print(f"Processed {articles} articles, {sentences_seen} sentences, {pairs_found} pairs found so far")
    random.shuffle(pairs)
    print(f"Finished parsing: {articles} articles, {sentences_seen} sentences, {pairs_found} total pairs")
    return pairs


def save_pairs(pairs: Iterable[Tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for stripped, label in pairs:
            f.write(f"{stripped}\t{label}\n")


def main(xml_path: Optional[str] = None, output_path: Optional[str] = None, max_articles: int = 20000) -> None:
    xml_file = Path(xml_path or "viwiki-latest-pages-articles-multistream.xml")
    if not xml_file.is_file():
        raise FileNotFoundError(f"XML dump not found: {xml_file}")
    out_path = Path(output_path) if output_path else DATASET_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = generate_pairs(xml_file, max_articles=MAX_ARTICLES)
    if not pairs:
        raise RuntimeError("No sentence pairs produced from the dump")
    save_pairs(pairs, out_path)
    metadata = {
        "total_pairs": len(pairs),
        "source_xml": str(xml_file),
        "output_file": str(out_path),
    }
    with (out_path.parent / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()
