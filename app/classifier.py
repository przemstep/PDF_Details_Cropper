from __future__ import annotations

from dataclasses import asdict

from collections import defaultdict
from pathlib import Path
from typing import Callable
import re

import pandas as pd

from .models import ClassificationResult
from .utils import ascii_text, normalize_text, save_json


class DetailClassifier:
    def __init__(self, min_threshold: float = 0.5) -> None:
        self.min_threshold = min_threshold

    @staticmethod
    def _find_col(df: pd.DataFrame, expected: str) -> str | None:
        expected_n = expected.lower().strip()
        for col in df.columns:
            if col.lower().strip() == expected_n:
                return col
        return None

    def build_synonyms_from_elements(self, elements: pd.DataFrame) -> pd.DataFrame:
        if elements.empty:
            return pd.DataFrame(columns=["ElementCode", "Phrase", "Weight", "MatchType", "Active"])

        code_col = self._find_col(elements, "ElementCode") or self._find_col(elements, "Code")
        if not code_col:
            return pd.DataFrame(columns=["ElementCode", "Phrase", "Weight", "MatchType", "Active"])

        pl_col = (
            self._find_col(elements, "NamePL")
            or self._find_col(elements, "PolishName")
            or self._find_col(elements, "Name_PL")
            or self._find_col(elements, "nazwa polska")
        )
        en_col = (
            self._find_col(elements, "NameEN")
            or self._find_col(elements, "EnglishName")
            or self._find_col(elements, "Name_EN")
            or self._find_col(elements, "nazwa angielska")
        )
        category_col = (
            self._find_col(elements, "Category")
            or self._find_col(elements, "Family")
            or self._find_col(elements, "Group")
        )

        rows: list[dict[str, object]] = []

        def add_row(code: str, phrase_val: object, weight: float) -> None:
            phrase = normalize_text(str(phrase_val)) if phrase_val is not None else ""
            if not code or not phrase or phrase.lower() == "nan":
                return
            rows.append({"ElementCode": code, "Phrase": phrase, "Weight": weight, "MatchType": "exact", "Active": True})

        for element in elements.itertuples(index=False):
            code = str(getattr(element, code_col, "")).strip()
            add_row(code, getattr(element, pl_col, "") if pl_col else "", 1.0)
            add_row(code, getattr(element, en_col, "") if en_col else "", 1.0)
            add_row(code, getattr(element, category_col, "") if category_col else "", 0.35)

        if not rows:
            return pd.DataFrame(columns=["ElementCode", "Phrase", "Weight", "MatchType", "Active"])

        synonyms = pd.DataFrame(rows)
        return synonyms.drop_duplicates(subset=["ElementCode", "Phrase"]).reset_index(drop=True)

    def classify(self, analyzed_matches_path: Path, dictionary_path: Path, output_dir: Path, status_callback: Callable[[str, str, str], None] | None = None) -> list[ClassificationResult]:
        def emit(level: str, msg: str) -> None:
            if status_callback:
                status_callback("classification", level, msg)

        output_dir.mkdir(parents=True, exist_ok=True)
        report_rows: list[dict] = []
        results: list[ClassificationResult] = []

        ex = pd.read_excel(analyzed_matches_path)
        if not dictionary_path.exists():
            raise FileNotFoundError(f"Brak master_dictionary: {dictionary_path}")

        try:
            elements = pd.read_excel(dictionary_path, sheet_name="Elements")
            synonyms = pd.read_excel(dictionary_path, sheet_name="Synonyms")
            negative = pd.read_excel(dictionary_path, sheet_name="NegativeSynonyms")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Nieczytelny master_dictionary: {exc}") from exc

        if synonyms.empty:
            emit("WARN", "Synonyms empty: using fallback generated from Elements. For better classification generate synonym candidates from extracted text.")
            synonyms = self.build_synonyms_from_elements(elements)

        elem_code_col = self._find_col(elements, "ElementCode") or self._find_col(elements, "Code")
        elem_meta = {getattr(r, elem_code_col): r for r in elements.itertuples(index=False)} if elem_code_col else {}

        syn_active_col = self._find_col(synonyms, "Active") or "Active"
        syn_phrase_col = self._find_col(synonyms, "Phrase") or "Phrase"
        syn_code_col = self._find_col(synonyms, "ElementCode") or "ElementCode"
        syn_weight_col = self._find_col(synonyms, "Weight") or "Weight"
        syn_match_type_col = self._find_col(synonyms, "MatchType") or "MatchType"

        neg_phrase_col = self._find_col(negative, "Phrase") or "Phrase"
        neg_code_col = self._find_col(negative, "ElementCode") or "ElementCode"
        neg_penalty_col = self._find_col(negative, "Penalty") or "Penalty"
        priority_attr = self._find_col(elements, "Priority") or "Priority"

        for idx, row in enumerate(ex.itertuples(index=False), start=1):
            emit("INFO", f"processing={idx}/{len(ex)}")
            raw = str(getattr(row, "raw_text", ""))
            txt = normalize_text(raw)
            txt_ascii = ascii_text(txt).lower()
            scores = defaultdict(float)
            matched: dict[str, list[str]] = defaultdict(list)
            best_match: dict[str, tuple[str, float]] = {}
            err = ""
            try:
                for syn in synonyms.itertuples(index=False):
                    if not bool(getattr(syn, syn_active_col, True)):
                        continue
                    phrase = normalize_text(str(getattr(syn, syn_phrase_col, "")))
                    phrase = ascii_text(phrase).lower()
                    if phrase and phrase in txt_ascii:
                        code = str(getattr(syn, syn_code_col))
                        weight = float(getattr(syn, syn_weight_col, 1.0))
                        match_type = str(getattr(syn, syn_match_type_col, "")).lower()
                        boost = weight * (1.3 if " " in phrase else 1.0)
                        if match_type == "fuzzy":
                            boost *= 0.7
                        scores[code] += boost
                        matched[code].append(phrase)
                        prev = best_match.get(code)
                        if prev is None or boost > prev[1]:
                            best_match[code] = (phrase, boost)

                for neg in negative.itertuples(index=False):
                    phrase = ascii_text(normalize_text(str(getattr(neg, neg_phrase_col, "")))).lower()
                    if phrase and phrase in txt_ascii:
                        code = str(getattr(neg, neg_code_col))
                        penalty = float(getattr(neg, neg_penalty_col, 0.0))
                        scores[code] -= penalty

                weighted = {}
                priority_by_code: dict[str, float] = {}
                for code, s in scores.items():
                    priority = getattr(elem_meta.get(code), priority_attr, 100) if elem_meta.get(code) is not None else 100
                    priority_v = float(priority)
                    priority_by_code[code] = priority_v
                    weighted[code] = s * (priority_v / 100.0)

                high = sorted([(c, s) for c, s in weighted.items() if priority_by_code.get(c, 100) >= 40 and s >= self.min_threshold], key=lambda kv: kv[1], reverse=True)[:2]
                low = sorted([(c, s) for c, s in weighted.items() if priority_by_code.get(c, 100) < 40 and s >= self.min_threshold], key=lambda kv: kv[1], reverse=True)[:1]
                top = high + low
                total = sum(max(v, 0.0) for v in weighted.values()) or 1.0
                confidence = (top[0][1] / total) if top else 0.0
                proposed = "-".join([c for c, _ in top]) if top else "UNMATCHED"

                keywords = sorted(set(k for c, _ in top for k in matched.get(c, [])))
                search_tags = [k for k in keywords if len(k) > 2 and not re.fullmatch(r"\d+", k)]
                search_text = f"{proposed} {' '.join(search_tags)} {txt[:120]}".strip()
                status = "matched" if top else "unmatched"
                matched_key = top[0][0] if top else ""
                match_score = float(top[0][1]) if top else 0.0
            except Exception as exc:  # noqa: BLE001
                status = "error"
                err = str(exc)
                proposed = "ERROR"
                confidence = 0.0
                keywords = []
                search_tags = []
                search_text = ""
                matched_key = ""
                match_score = 0.0

            crop_id = str(getattr(row, "crop_id", getattr(row, "CropID", "")))
            results.append(ClassificationResult(crop_id=crop_id, proposed_code=proposed, detected_elements=[c for c, _ in top] if status != "error" else [], scores={}, confidence=confidence, keywords=keywords, technical_phrases=keywords, search_tags=search_tags, sharepoint_tags="; ".join(search_tags), search_text=search_text))
            report_rows.append({"source_file": str(analyzed_matches_path), "extracted_text_file": str(getattr(row, "crop_id", "")), "normalized_text": txt, "matched_dictionary_key": matched_key, "classification_code": proposed, "match_score": match_score, "confidence": confidence, "status": status, "error_message": err})

        pd.DataFrame([asdict(r) for r in results]).to_excel(output_dir / "classification_results.xlsx", index=False)
        match_rows = []
        for r in report_rows:
            match_rows.append({
                "crop_id": r["extracted_text_file"],
                "classification_code": r["classification_code"],
                "matched_dictionary_key": r["matched_dictionary_key"],
                "match_score": r["match_score"],
                "normalized_text": r["normalized_text"],
            })
        pd.DataFrame(match_rows).to_excel(output_dir / "text_element_matches.xlsx", index=False)
        save_json(output_dir / "classification_results.json", [asdict(r) for r in results])
        pd.DataFrame(report_rows).to_excel(output_dir / "classification_report.xlsx", index=False)
        save_json(output_dir / "classification_report.json", report_rows)
        return results
