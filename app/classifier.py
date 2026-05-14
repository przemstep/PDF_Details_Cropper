from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import pandas as pd

from .models import ClassificationResult
from .utils import ascii_text, normalize_text, save_json


class DetailClassifier:
    def __init__(self, min_threshold: float = 0.5) -> None:
        self.min_threshold = min_threshold

    def classify(self, extraction_path: Path, dictionary_path: Path, output_dir: Path) -> list[ClassificationResult]:
        ex = pd.read_excel(extraction_path)
        elements = pd.read_excel(dictionary_path, sheet_name="Elements")
        synonyms = pd.read_excel(dictionary_path, sheet_name="Synonyms")
        negative = pd.read_excel(dictionary_path, sheet_name="NegativeSynonyms")

        elem_meta = {r.ElementCode: r for r in elements.itertuples(index=False)}
        results: list[ClassificationResult] = []

        for row in ex.itertuples(index=False):
            raw = str(getattr(row, "raw_text", ""))
            txt = normalize_text(raw)
            txt_ascii = ascii_text(txt)
            scores = defaultdict(float)
            matched: dict[str, list[str]] = defaultdict(list)

            for syn in synonyms.itertuples(index=False):
                if not getattr(syn, "Active", True):
                    continue
                phrase = normalize_text(str(syn.Phrase))
                if phrase and phrase in txt_ascii:
                    boost = float(syn.Weight) * (1.3 if " " in phrase else 1.0)
                    if str(syn.MatchType).lower() == "fuzzy":
                        boost *= 0.7
                    scores[syn.ElementCode] += boost
                    matched[syn.ElementCode].append(phrase)

            for neg in negative.itertuples(index=False):
                phrase = normalize_text(str(neg.Phrase))
                if phrase and phrase in txt_ascii:
                    scores[neg.ElementCode] -= float(neg.Penalty)

            weighted = {}
            for code, s in scores.items():
                priority = getattr(elem_meta.get(code), "Priority", 100) or 100
                weighted[code] = s * (float(priority) / 100.0)

            top = sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top = [(c, s) for c, s in top if s >= self.min_threshold]
            total = sum(max(v, 0.0) for v in weighted.values()) or 1.0
            confidence = (top[0][1] / total) if top else 0.0
            proposed = ".".join([c for c, _ in top]) if top else "UNCLASSIFIED"

            keywords = sorted(set(k for c, _ in top for k in matched.get(c, [])))
            search_tags = [k for k in keywords if len(k) > 2 and not re.fullmatch(r"\d+", k)]
            search_text = f"{proposed} {' '.join(search_tags)} {txt[:120]}".strip()
            results.append(
                ClassificationResult(
                    crop_id=str(getattr(row, "crop_id", getattr(row, "CropID", ""))),
                    proposed_code=proposed,
                    detected_elements=[c for c, _ in top],
                    scores={c: float(s) for c, s in top},
                    confidence=confidence,
                    keywords=keywords,
                    technical_phrases=keywords,
                    search_tags=search_tags,
                    sharepoint_tags="; ".join(search_tags),
                    search_text=search_text,
                )
            )

        out_df = pd.DataFrame([r.__dict__ for r in results])
        output_dir.mkdir(parents=True, exist_ok=True)
        out_df.to_excel(output_dir / "classification_results.xlsx", index=False)
        save_json(output_dir / "classification_results.json", [r.__dict__ for r in results])
        return results
