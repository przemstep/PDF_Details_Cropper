from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable
import re

import pandas as pd

from .utils import ascii_text, normalize_text

try:
    from rapidfuzz import fuzz
except Exception:  # noqa: BLE001
    fuzz = None


DEFAULT_NOISE_PATTERNS = [
    r"^\s*$",
    r"^\+?\d+(?:[\.,]\d+)?$",
    r"^(?:rei|ei|ewi?)\s*\d+$",
    r"^(?:d\s*=\s*)?\d+(?:[\.,]\d+)?\s*[x×]\s*\d+(?:[\.,]\d+)?(?:\s*mm)?$",
    r"^(?:gr\.?\s*)?\d+(?:[\.,]\d+)?\s*mm$",
    r"^\w?$",
]

DEFAULT_STOP_PHRASES = [
    "wg specyfikacji",
    "acc. to specification",
    "ref. to specification",
    "wg projektu",
    "ref. to struct. eng. design",
    "finish refer to finishes schedule",
]


@dataclass
class _CandidateAgg:
    phrase: str
    normalized_phrase: str
    source_ids: set[str]
    occurrence_count: int
    example_texts: list[str]


class SynonymCandidateService:
    COLUMNS = [
        "Phrase",
        "NormalizedPhrase",
        "SourceCount",
        "OccurrenceCount",
        "ExampleCropIDs",
        "ExampleTexts",
        "SuggestedElementCode",
        "SuggestedBy",
        "Confidence",
        "Decision",
        "Notes",
    ]

    def __init__(self, fuzzy_threshold: int = 80) -> None:
        self.fuzzy_threshold = fuzzy_threshold

    @staticmethod
    def _find_col(df: pd.DataFrame, expected: str) -> str | None:
        exp = expected.lower().strip()
        for col in df.columns:
            if col.lower().strip() == exp:
                return col
        return None

    def _emit(self, status_callback: Callable[[str, str, str], None] | None, level: str, msg: str) -> None:
        if status_callback:
            status_callback("classification", level, msg)

    def _best_text_column(self, df: pd.DataFrame, status_callback: Callable[[str, str, str], None] | None) -> str:
        raw = self._find_col(df, "raw_text")
        if raw:
            return raw
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and col.lower().strip() not in {"crop_id", "cropid"}:
                self._emit(status_callback, "WARN", f"raw_text missing: using fallback text column '{col}'")
                return col
        raise ValueError("Nie znaleziono kolumny tekstowej (raw_text).")

    def _clean_line(self, line: str) -> str:
        line = normalize_text(line)
        line = re.sub(r"\s+", " ", line).strip(" -_:;,.\t")
        line = re.sub(r"\b(?:d\s*=\s*\d+(?:[\.,]\d+)?\s*mm|gr\.?\s*\d+(?:[\.,]\d+)?\s*mm)\b", "", line, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", line).strip()

    def _is_noise(self, line: str, patterns: list[re.Pattern], stop_phrases: list[str]) -> bool:
        n = normalize_text(line).lower()
        n_ascii = ascii_text(n)
        if len(n_ascii) < 3:
            return True
        if all(ch in "|/\\_~.-+*=()[]{}" for ch in n_ascii):
            return True
        for p in patterns:
            if p.search(n_ascii):
                return True
        for phrase in stop_phrases:
            if phrase in n or phrase in n_ascii:
                return True
        return False

    def _similarity(self, a: str, b: str) -> float:
        if fuzz is not None:
            return float(fuzz.token_set_ratio(a, b))
        return SequenceMatcher(None, a, b).ratio() * 100.0

    def _load_element_terms(self, dictionary_path: Path) -> list[tuple[str, str]]:
        elements = pd.read_excel(dictionary_path, sheet_name="Elements")
        synonyms = pd.read_excel(dictionary_path, sheet_name="Synonyms") if dictionary_path.exists() else pd.DataFrame()
        code_col = self._find_col(elements, "ElementCode") or self._find_col(elements, "Code")
        if not code_col:
            return []
        terms: list[tuple[str, str]] = []
        for row in elements.itertuples(index=False):
            code = str(getattr(row, code_col, "")).strip()
            for c in ("TitlePL", "NamePL", "TitleEN", "NameEN"):
                col = self._find_col(elements, c)
                if col:
                    phrase = normalize_text(str(getattr(row, col, ""))).strip()
                    if code and phrase and phrase.lower() != "nan":
                        terms.append((code, phrase))

        if not synonyms.empty:
            syn_code = self._find_col(synonyms, "ElementCode") or "ElementCode"
            syn_phrase = self._find_col(synonyms, "Phrase") or "Phrase"
            for row in synonyms.itertuples(index=False):
                code = str(getattr(row, syn_code, "")).strip()
                phrase = normalize_text(str(getattr(row, syn_phrase, ""))).strip()
                if code and phrase and phrase.lower() != "nan":
                    terms.append((code, phrase))
        return terms

    def generate_candidates(
        self,
        analyzed_matches_path: Path,
        dictionary_path: Path,
        output_dir: Path,
        status_callback: Callable[[str, str, str], None] | None = None,
        noise_patterns: list[str] | None = None,
        stop_phrases: list[str] | None = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_excel(analyzed_matches_path)
        text_col = self._best_text_column(df, status_callback)
        crop_col = self._find_col(df, "crop_id") or self._find_col(df, "CropID") or "crop_id"

        compiled = [re.compile(p, re.IGNORECASE) for p in (noise_patterns or DEFAULT_NOISE_PATTERNS)]
        stops = [normalize_text(s).lower() for s in (stop_phrases or DEFAULT_STOP_PHRASES)]
        aggregations: dict[str, _CandidateAgg] = {}

        for row in df.itertuples(index=False):
            crop_id = str(getattr(row, crop_col, ""))
            raw = str(getattr(row, text_col, ""))
            for src_line in re.split(r"[\n\r;]+", raw):
                cleaned = self._clean_line(src_line)
                if not cleaned or self._is_noise(cleaned, compiled, stops):
                    continue
                normalized = normalize_text(cleaned).lower()
                key = ascii_text(normalized)
                if key not in aggregations:
                    aggregations[key] = _CandidateAgg(cleaned, normalized, set(), 0, [])
                agg = aggregations[key]
                agg.occurrence_count += 1
                if crop_id:
                    agg.source_ids.add(crop_id)
                if cleaned not in agg.example_texts and len(agg.example_texts) < 3:
                    agg.example_texts.append(cleaned[:120])

        terms = self._load_element_terms(dictionary_path)
        rows = []
        for key, agg in sorted(aggregations.items(), key=lambda kv: (-kv[1].occurrence_count, kv[1].phrase)):
            suggested_code = ""
            suggested_by = ""
            confidence = 0.0

            for code, term in terms:
                t_norm = ascii_text(normalize_text(term).lower())
                if t_norm and t_norm in key:
                    suggested_code = code
                    suggested_by = "contains"
                    confidence = 0.95
                    break

            if not suggested_code and terms:
                best = max(terms, key=lambda ct: self._similarity(key, ascii_text(normalize_text(ct[1]).lower())))
                score = self._similarity(key, ascii_text(normalize_text(best[1]).lower()))
                if score >= self.fuzzy_threshold:
                    suggested_code = best[0]
                    suggested_by = "fuzzy"
                    confidence = round(score / 100.0, 3)

            rows.append(
                {
                    "Phrase": agg.phrase,
                    "NormalizedPhrase": agg.normalized_phrase,
                    "SourceCount": len(agg.source_ids),
                    "OccurrenceCount": agg.occurrence_count,
                    "ExampleCropIDs": "; ".join(list(sorted(agg.source_ids))[:5]),
                    "ExampleTexts": " | ".join(agg.example_texts),
                    "SuggestedElementCode": suggested_code,
                    "SuggestedBy": suggested_by,
                    "Confidence": confidence,
                    "Decision": "pending",
                    "Notes": "",
                }
            )

        out_path = output_dir / "synonym_candidates.xlsx"
        pd.DataFrame(rows, columns=self.COLUMNS).to_excel(out_path, sheet_name="Candidates", index=False)
        self._emit(status_callback, "INFO", f"Generated {len(rows)} synonym candidates: {out_path}")
        return out_path

    def import_accepted_candidates(
        self,
        candidates_path: Path,
        dictionary_path: Path,
        confirm_overwrite: Callable[[str], bool],
        status_callback: Callable[[str, str, str], None] | None = None,
    ) -> int:
        msg = "Zaakceptowane kandydaty zostaną dopisane do zakładki Synonyms w master_dictionary.xlsx. Plik zostanie nadpisany. Czy kontynuować?"
        if not confirm_overwrite(msg):
            self._emit(status_callback, "INFO", "Import canceled by user")
            return 0

        candidates = pd.read_excel(candidates_path, sheet_name="Candidates")
        accepted = candidates[
            candidates.get("Decision", "").astype(str).str.lower().str.strip().eq("accept")
            & candidates.get("SuggestedElementCode", "").astype(str).str.strip().ne("")
            & candidates.get("Phrase", "").astype(str).str.strip().ne("")
        ].copy()

        if accepted.empty:
            self._emit(status_callback, "INFO", "No accepted candidates to import")
            return 0

        sheets = pd.read_excel(dictionary_path, sheet_name=None)
        existing_syn = sheets.get("Synonyms", pd.DataFrame(columns=["ElementCode", "Phrase", "Weight", "MatchType", "Active"]))

        weights = pd.to_numeric(accepted["Weight"], errors="coerce").fillna(1.0) if "Weight" in accepted.columns else pd.Series([1.0] * len(accepted))
        match_types = accepted["MatchType"].astype(str).replace("nan", "exact") if "MatchType" in accepted.columns else pd.Series(["exact"] * len(accepted))
        import_rows = pd.DataFrame(
            {
                "ElementCode": accepted["SuggestedElementCode"].astype(str).str.strip(),
                "Phrase": accepted["Phrase"].astype(str).str.strip(),
                "Weight": weights,
                "MatchType": match_types,
                "Active": True,
            }
        )
        merged = pd.concat([existing_syn, import_rows], ignore_index=True)
        merged = merged.drop_duplicates(subset=["ElementCode", "Phrase"]).reset_index(drop=True)

        backup = dictionary_path.with_name(f"{dictionary_path.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{dictionary_path.suffix}")
        dictionary_path.replace(backup)

        with pd.ExcelWriter(dictionary_path, engine="openpyxl") as writer:
            for name, sheet in sheets.items():
                if name == "Synonyms":
                    continue
                sheet.to_excel(writer, sheet_name=name, index=False)
            merged.to_excel(writer, sheet_name="Synonyms", index=False)

        self._emit(status_callback, "INFO", f"Imported {len(import_rows)} accepted candidates. Backup: {backup}")
        return len(import_rows)
