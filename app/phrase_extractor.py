from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from .utils import ascii_text, normalize_text

NOISE_PATTERNS = [
    r"^\s*$",
    r"^\+?\d+(?:[\.,]\d+)?$",
    r"^(?:rei|ei)\s*\d+$",
    r"^(?:d\s*=\s*)?\d+(?:[\.,]\d+)?\s*(?:mm|cm|m)$",
    r"^\d+(?:[\.,]\d+)?\s*[x×]\s*\d+(?:[\.,]\d+)?(?:\s*(?:mm|cm|m))?$",
    r"^(?:gr\.?\s*)?\d+(?:[\.,]\d+)?\s*mm$",
]

TAIL_PATTERNS = [
    r"\bwg\s+specyfikacj\w*\b.*$",
    r"\bwg\s+projekt\w*\b.*$",
    r"\bwg\s+zestawieni\w*\b.*$",
    r"\bwg\s+list\w*\b.*$",
    r"\bacc\.?\s*to\s*specification\b.*$",
    r"\bref\.?\s*to\s*specification\b.*$",
    r"\bref\.?\s*to\s*struct\.?\s*eng\.?\s*design\b.*$",
    r"\bacc\.?\s*to\s*door\s*schedule\b.*$",
    r"\brefer\s*to\s*finishes\s*schedule\b.*$",
]

STOP_WORDS = {
    "ze", "z", "na", "do", "wg", "i", "oraz", "w", "o", "od", "dla",
    "and", "to", "of", "on", "the", "with", "acc", "ref", "specification", "schedule",
}

GENERIC_SINGLE_WORDS = {"door", "steel", "board", "wall", "strop"}


@dataclass
class PhraseCandidate:
    phrase: str
    normalized_phrase: str
    occurrence_count: int
    source_line: str | None = None


def is_noise(text: str) -> bool:
    text_n = normalize_text(text)
    if not text_n:
        return True
    text_ascii = ascii_text(text_n)
    if len(text_ascii) < 4:
        return True
    return any(re.fullmatch(p, text_ascii, flags=re.IGNORECASE) for p in NOISE_PATTERNS)


def strip_context_tail(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"[,;:]+", " ", cleaned)
    cleaned = re.sub(r"\b(?:d\s*=\s*\d+(?:[\.,]\d+)?\s*mm|gr\.?\s*\d+(?:[\.,]\d+)?\s*mm|\d+(?:[\.,]\d+)?\s*mm\s*thick|\d+(?:[\.,]\d+)?\s*[x×]\s*\d+(?:[\.,]\d+)?\s*mm)\b", " ", cleaned, flags=re.IGNORECASE)
    for patt in TAIL_PATTERNS:
        cleaned = re.sub(patt, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_.")
    return cleaned


def tokenize_phrase(text: str) -> list[str]:
    text = strip_context_tail(text)
    tokens = re.findall(r"[\wążźćńółęą-]+", text.lower())
    return [t for t in tokens if t]


def _valid_ngram(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if tokens[0] in STOP_WORDS or tokens[-1] in STOP_WORDS:
        return False
    if all(t in STOP_WORDS for t in tokens):
        return False
    joined = " ".join(tokens)
    joined_ascii = ascii_text(joined)
    if len(joined_ascii) < 4:
        return False
    if re.fullmatch(r"[\d\s\./x×+-]+", joined_ascii):
        return False
    return True


def generate_ngrams(words: list[str], min_n: int = 2, max_n: int = 4) -> list[str]:
    out: list[str] = []
    max_n = min(max_n, len(words))
    for n in range(min_n, max_n + 1):
        for i in range(0, len(words) - n + 1):
            gram_tokens = words[i : i + n]
            if _valid_ngram(gram_tokens):
                out.append(" ".join(gram_tokens))
    return out


def extract_core_phrases(raw_text: str) -> list[PhraseCandidate]:
    counter: Counter[str] = Counter()
    source_line_by_phrase: dict[str, str] = {}
    for line in re.split(r"[\n\r]+", str(raw_text)):
        core = strip_context_tail(line)
        if is_noise(core):
            continue
        words = tokenize_phrase(core)
        if len(words) < 2:
            continue
        full_phrase = " ".join(words)
        if 2 <= len(words) <= 6 and _valid_ngram(words):
            counter[full_phrase] += 1
            source_line_by_phrase.setdefault(full_phrase, normalize_text(line))
        for gram in generate_ngrams(words, 2, 4):
            counter[gram] += 1
            source_line_by_phrase.setdefault(gram, normalize_text(line))

    result: list[PhraseCandidate] = []
    for phrase, occ in counter.items():
        if len(phrase.split()) == 1 and phrase in GENERIC_SINGLE_WORDS:
            continue
        result.append(
            PhraseCandidate(
                phrase=phrase,
                normalized_phrase=ascii_text(normalize_text(phrase)),
                occurrence_count=occ,
                source_line=source_line_by_phrase.get(phrase),
            )
        )
    return sorted(result, key=lambda c: (-c.occurrence_count, c.phrase))
