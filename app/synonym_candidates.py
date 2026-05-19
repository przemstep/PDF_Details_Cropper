from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from shutil import copy2
from typing import Callable

import pandas as pd

from .phrase_extractor import extract_core_phrases
from .utils import ascii_text, normalize_text

try:
    from rapidfuzz import fuzz
except Exception:  # noqa: BLE001
    fuzz = None

GENERIC_WORDS = {"door", "steel", "board", "wall", "strop"}


@dataclass
class _Agg:
    phrase: str
    normalized_phrase: str
    occurrence_count: int = 0
    source_ids: set[str] = field(default_factory=set)
    example_texts: list[str] = field(default_factory=list)


def _emit(cb: Callable[[str, str, str], None] | None, level: str, message: str) -> None:
    if cb:
        cb("classification", level, message)


def _find_col(df: pd.DataFrame, expected: str) -> str | None:
    e = expected.lower().strip()
    for col in df.columns:
        if col.lower().strip() == e:
            return col
    return None


def _similarity(a: str, b: str) -> float:
    if fuzz is not None:
        return float(fuzz.token_set_ratio(a, b))
    return SequenceMatcher(None, a, b).ratio() * 100.0


def _load_reference_terms(dictionary_path: Path) -> list[tuple[str, str]]:
    sheets = pd.read_excel(dictionary_path, sheet_name=None)
    elements = sheets.get("Elements", pd.DataFrame())
    synonyms = sheets.get("Synonyms", pd.DataFrame())
    terms: list[tuple[str, str]] = []
    code_col = _find_col(elements, "ElementCode") or _find_col(elements, "Code")
    if code_col:
        for _, row in elements.iterrows():
            code = str(row.get(code_col, "")).strip()
            for col_name in ("TitlePL", "TitleEN", "NamePL", "NameEN"):
                col = _find_col(elements, col_name)
                if col:
                    v = normalize_text(str(row.get(col, ""))).strip()
                    if code and v and v.lower() != "nan":
                        terms.append((code, v))

    if not synonyms.empty:
        s_code = _find_col(synonyms, "ElementCode") or "ElementCode"
        s_phrase = _find_col(synonyms, "Phrase") or "Phrase"
        for _, row in synonyms.iterrows():
            code = str(row.get(s_code, "")).strip()
            phrase = normalize_text(str(row.get(s_phrase, ""))).strip()
            if code and phrase and phrase.lower() != "nan":
                terms.append((code, phrase))
    return terms


def _suggest_code(candidate_phrase: str, refs: list[tuple[str, str]]) -> tuple[str, str, float]:
    c_norm = ascii_text(normalize_text(candidate_phrase))
    if len(c_norm.split()) == 1 and c_norm in GENERIC_WORDS:
        return "", "", 0.0

    for code, term in refs:
        t_norm = ascii_text(normalize_text(term))
        if not t_norm:
            continue
        if c_norm in t_norm or t_norm in c_norm:
            return code, "substring", 1.0

    if not refs:
        return "", "", 0.0

    best_code = ""
    best_score = 0.0
    for code, term in refs:
        score = _similarity(c_norm, ascii_text(normalize_text(term)))
        if score > best_score:
            best_score = score
            best_code = code
    if best_score >= 90.0:
        return best_code, "fuzzy", round(best_score / 100.0, 3)
    return "", "", 0.0


def generate_synonym_candidates(analyzed_matches_path: Path, dictionary_path: Path, output_dir: Path, status_callback: Callable[[str, str, str], None] | None = None) -> Path:
    df = pd.read_excel(analyzed_matches_path)
    crop_col = _find_col(df, "crop_id") or _find_col(df, "CropID")
    text_col = _find_col(df, "raw_text")
    if not text_col:
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and col != crop_col:
                text_col = col
                _emit(status_callback, "WARN", f"raw_text missing: using fallback text column '{col}'")
                break
    if not text_col:
        raise ValueError("Nie znaleziono kolumny tekstowej")

    refs = _load_reference_terms(dictionary_path)
    agg: dict[str, _Agg] = {}
    for _, row in df.iterrows():
        crop_id = str(row.get(crop_col, "")).strip() if crop_col else ""
        raw = str(row.get(text_col, ""))
        for cand in extract_core_phrases(raw):
            key = cand.normalized_phrase
            if key not in agg:
                agg[key] = _Agg(phrase=cand.phrase, normalized_phrase=key)
            cur = agg[key]
            cur.occurrence_count += cand.occurrence_count
            if crop_id:
                cur.source_ids.add(crop_id)
            if cand.source_line and cand.source_line not in cur.example_texts and len(cur.example_texts) < 3:
                cur.example_texts.append(cand.source_line[:160])

    rows = []
    for key, item in agg.items():
        code, by, conf = _suggest_code(item.phrase, refs)
        rows.append({
            "Phrase": item.phrase,
            "NormalizedPhrase": item.normalized_phrase,
            "SourceCount": len(item.source_ids),
            "OccurrenceCount": item.occurrence_count,
            "ExampleCropIDs": ";".join(sorted(item.source_ids)[:5]),
            "ExampleTexts": " | ".join(item.example_texts[:3]),
            "SuggestedElementCode": code,
            "SuggestedBy": by,
            "Confidence": conf,
            "Decision": "pending",
            "Notes": "",
        })

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["SourceCount", "OccurrenceCount", "Phrase"], ascending=[False, False, True])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "synonym_candidates.xlsx"
    out_df.to_excel(out_path, sheet_name="Candidates", index=False)
    _emit(status_callback, "INFO", f"Generated {len(out_df)} synonym candidates: {out_path}")
    return out_path


def import_accepted_synonyms(candidates_path: Path, dictionary_path: Path, confirm_overwrite: Callable[[str], bool] | None = None, status_callback: Callable[[str, str, str], None] | None = None) -> int:
    if confirm_overwrite is None:
        _emit(status_callback, "WARN", "confirm_overwrite not provided - skipping dictionary update")
        return 0
    msg = "Zaakceptowane kandydaty zostaną dopisane do zakładki Synonyms w master_dictionary.xlsx. Plik zostanie nadpisany. Czy kontynuować?"
    if not confirm_overwrite(msg):
        _emit(status_callback, "INFO", "Import canceled by user")
        return 0

    cdf = pd.read_excel(candidates_path, sheet_name="Candidates")
    decision = cdf.get("Decision", pd.Series(dtype=str)).astype(str).str.lower().str.strip()
    accepted = cdf[decision.eq("accept")].copy()
    accepted = accepted[
        accepted.get("SuggestedElementCode", pd.Series(dtype=str)).astype(str).str.strip().ne("")
        & accepted.get("Phrase", pd.Series(dtype=str)).astype(str).str.strip().ne("")
    ]
    if accepted.empty:
        return 0

    sheets = pd.read_excel(dictionary_path, sheet_name=None)
    syn = sheets.get("Synonyms", pd.DataFrame(columns=["ElementCode", "Phrase", "Weight", "MatchType", "Active"]))
    for col, default in (("Weight", 1.0), ("MatchType", "exact"), ("Active", True)):
        if col not in syn.columns:
            syn[col] = default

    add = pd.DataFrame({
        "ElementCode": accepted["SuggestedElementCode"].astype(str).str.strip(),
        "Phrase": accepted["Phrase"].astype(str).str.strip(),
        "Weight": 1.0,
        "MatchType": "exact",
        "Active": True,
    })
    before = len(syn)
    merged = pd.concat([syn, add], ignore_index=True).drop_duplicates(subset=["ElementCode", "Phrase"], keep="first")
    added_count = len(merged) - before

    backup = dictionary_path.with_name(f"{dictionary_path.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{dictionary_path.suffix}")
    copy2(dictionary_path, backup)
    _emit(status_callback, "INFO", f"Created backup: {backup}")

    with pd.ExcelWriter(dictionary_path, engine="openpyxl") as writer:
        for name, sheet in sheets.items():
            if name != "Synonyms":
                sheet.to_excel(writer, sheet_name=name, index=False)
        merged.to_excel(writer, sheet_name="Synonyms", index=False)

    _emit(status_callback, "INFO", f"Imported {added_count} accepted synonyms")
    return added_count


class SynonymCandidateService:
    def generate_candidates(self, analyzed_matches_path: Path, dictionary_path: Path, output_dir: Path, status_callback: Callable[[str, str, str], None] | None = None) -> Path:
        return generate_synonym_candidates(analyzed_matches_path, dictionary_path, output_dir, status_callback)

    def import_accepted_candidates(self, candidates_path: Path, dictionary_path: Path, confirm_overwrite: Callable[[str], bool] | None = None, status_callback: Callable[[str, str, str], None] | None = None) -> int:
        return import_accepted_synonyms(candidates_path, dictionary_path, confirm_overwrite, status_callback)
