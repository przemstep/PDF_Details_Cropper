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

    def classify(self, extraction_path: Path, dictionary_path: Path, output_dir: Path, status_callback: Callable[[str, str, str], None] | None = None) -> list[ClassificationResult]:
        def emit(level: str, msg: str) -> None:
            if status_callback:
                status_callback("classification", level, msg)

        output_dir.mkdir(parents=True, exist_ok=True)
        report_rows: list[dict] = []
        results: list[ClassificationResult] = []

        ex = pd.read_excel(extraction_path)
        if not dictionary_path.exists():
            raise FileNotFoundError(f"Brak master_dictionary: {dictionary_path}")

        try:
            elements = pd.read_excel(dictionary_path, sheet_name="Elements")
            synonyms = pd.read_excel(dictionary_path, sheet_name="Synonyms")
            negative = pd.read_excel(dictionary_path, sheet_name="NegativeSynonyms")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Nieczytelny master_dictionary: {exc}") from exc

        if synonyms.empty:
            raise RuntimeError("master_dictionary (Synonyms) jest pusty")

        elem_code_col = self._find_col(elements, "ElementCode") or "ElementCode"
        elem_meta = {getattr(r, elem_code_col): r for r in elements.itertuples(index=False)}

        for idx, row in enumerate(ex.itertuples(index=False), start=1):
            emit("INFO", f"processing={idx}/{len(ex)}")
            raw = str(getattr(row, "raw_text", ""))
            txt = normalize_text(raw)
            txt_ascii = ascii_text(txt).lower()
            scores = defaultdict(float)
            matched: dict[str, list[str]] = defaultdict(list)
            err = ""
            try:
                for syn in synonyms.itertuples(index=False):
                    if not bool(getattr(syn, self._find_col(synonyms, "Active") or "Active", True)):
                        continue
                    phrase = normalize_text(str(getattr(syn, self._find_col(synonyms, "Phrase") or "Phrase", "")))
                    phrase = ascii_text(phrase).lower()
                    if phrase and phrase in txt_ascii:
                        code = str(getattr(syn, self._find_col(synonyms, "ElementCode") or "ElementCode"))
                        weight = float(getattr(syn, self._find_col(synonyms, "Weight") or "Weight", 1.0))
                        match_type = str(getattr(syn, self._find_col(synonyms, "MatchType") or "MatchType", "")).lower()
                        boost = weight * (1.3 if " " in phrase else 1.0)
                        if match_type == "fuzzy":
                            boost *= 0.7
                        scores[code] += boost
                        matched[code].append(phrase)

                for neg in negative.itertuples(index=False):
                    phrase = ascii_text(normalize_text(str(getattr(neg, self._find_col(negative, "Phrase") or "Phrase", "")))).lower()
                    if phrase and phrase in txt_ascii:
                        code = str(getattr(neg, self._find_col(negative, "ElementCode") or "ElementCode"))
                        penalty = float(getattr(neg, self._find_col(negative, "Penalty") or "Penalty", 0.0))
                        scores[code] -= penalty

                weighted = {}
                for code, s in scores.items():
                    priority_attr = self._find_col(elements, "Priority") or "Priority"
                    priority = getattr(elem_meta.get(code), priority_attr, 100) if elem_meta.get(code) is not None else 100
                    weighted[code] = s * (float(priority) / 100.0)

                top = sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)[:3]
                top = [(c, s) for c, s in top if s >= self.min_threshold]
                total = sum(max(v, 0.0) for v in weighted.values()) or 1.0
                confidence = (top[0][1] / total) if top else 0.0
                proposed = ".".join([c for c, _ in top]) if top else "UNMATCHED"

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
            report_rows.append({"source_file": str(extraction_path), "extracted_text_file": str(getattr(row, "crop_id", "")), "normalized_text": txt, "matched_dictionary_key": matched_key, "classification_code": proposed, "match_score": match_score, "confidence": confidence, "status": status, "error_message": err})

        pd.DataFrame([asdict(r) for r in results]).to_excel(output_dir / "classification_results.xlsx", index=False)
        save_json(output_dir / "classification_results.json", [asdict(r) for r in results])
        pd.DataFrame(report_rows).to_excel(output_dir / "classification_report.xlsx", index=False)
        save_json(output_dir / "classification_report.json", report_rows)
        return results
