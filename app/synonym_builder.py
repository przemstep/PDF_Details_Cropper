from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd

from .utils import normalize_text

DEFAULT_STOPWORDS = {
    "i", "oraz", "the", "and", "for", "with", "bez", "lub", "ref", "typ", "mm", "cm"
}


class SynonymBuilder:
    def __init__(self, dict_path: Path) -> None:
        self.dict_path = dict_path

    def analyze_extractions(self, extraction_xlsx: Path) -> pd.DataFrame:
        df = pd.read_excel(extraction_xlsx)
        texts = [normalize_text(str(v)) for v in df.get("raw_text", pd.Series(dtype=str)).fillna("")]
        freq: Counter[str] = Counter()
        for txt in texts:
            tokens = [t for t in re.findall(r"[\w\-ąćęłńóśźż]+", txt) if len(t) > 2 and not t.isdigit()]
            tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS]
            for n in (1, 2, 3):
                for i in range(0, max(0, len(tokens) - n + 1)):
                    freq[" ".join(tokens[i : i + n])] += 1
        out = pd.DataFrame({"Phrase": list(freq.keys()), "Count": list(freq.values())}).sort_values("Count", ascending=False)
        return out

    def ensure_dictionary_template(self) -> None:
        self.dict_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(self.dict_path, engine="openpyxl") as writer:
            pd.DataFrame(columns=["ElementCode", "TitlePL", "TitleEN", "Category", "Priority"]).to_excel(writer, sheet_name="Elements", index=False)
            pd.DataFrame(columns=["ElementCode", "Language", "Phrase", "Weight", "MatchType", "Active"]).to_excel(writer, sheet_name="Synonyms", index=False)
            pd.DataFrame(columns=["ElementCode", "Phrase", "Penalty"]).to_excel(writer, sheet_name="NegativeSynonyms", index=False)
            pd.DataFrame(columns=["Phrase"]).to_excel(writer, sheet_name="StopWords", index=False)
            pd.DataFrame(columns=["Phrase", "Count", "ExampleCropIDs"]).to_excel(writer, sheet_name="FrequencyAnalysis", index=False)
