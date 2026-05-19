from pathlib import Path

import pandas as pd

from app.synonym_candidates import generate_synonym_candidates, import_accepted_synonyms


def _make_dict(path: Path) -> None:
    with pd.ExcelWriter(path) as w:
        pd.DataFrame([
            {"ElementCode": "E1", "TitlePL": "Barierka", "TitleEN": "Guardrail", "NamePL": "Barierka", "NameEN": "Guardrail"},
            {"ElementCode": "E2", "TitlePL": "Drzwi pożarowe", "TitleEN": "Fire door", "NamePL": "Drzwi pożarowe", "NameEN": "Fire door"},
        ]).to_excel(w, sheet_name="Elements", index=False)
        pd.DataFrame([{"ElementCode": "E1", "Phrase": "stainless steel top rail", "Weight": 1.0, "MatchType": "exact", "Active": True}]).to_excel(w, sheet_name="Synonyms", index=False)
        pd.DataFrame(columns=["ElementCode", "Phrase", "Penalty"]).to_excel(w, sheet_name="NegativeSynonyms", index=False)


def test_aggregation_source_count(tmp_path: Path):
    src = tmp_path / "ex.xlsx"
    pd.DataFrame([
        {"crop_id": "c1", "raw_text": "STAINLESS STEEL TOP RAIL"},
        {"crop_id": "c2", "raw_text": "STAINLESS STEEL TOP RAIL"},
    ]).to_excel(src, index=False)
    d = tmp_path / "dict.xlsx"
    _make_dict(d)
    out = generate_synonym_candidates(src, d, tmp_path)
    df = pd.read_excel(out)
    row = df[df["NormalizedPhrase"] == "stainless steel top rail"].iloc[0]
    assert row["SourceCount"] == 2


def test_import_pending_not_imported(tmp_path: Path):
    d = tmp_path / "master_dictionary.xlsx"
    _make_dict(d)
    c = tmp_path / "synonym_candidates.xlsx"
    pd.DataFrame([
        {"Phrase": "internal fire door", "SuggestedElementCode": "E2", "Decision": "pending"}
    ]).to_excel(c, sheet_name="Candidates", index=False)
    imported = import_accepted_synonyms(c, d, confirm_overwrite=lambda _: True)
    assert imported == 0


def test_import_accept_and_deduplicate_and_backup(tmp_path: Path):
    d = tmp_path / "master_dictionary.xlsx"
    _make_dict(d)
    c = tmp_path / "synonym_candidates.xlsx"
    pd.DataFrame([
        {"Phrase": "internal fire door", "SuggestedElementCode": "E2", "Decision": "accept"},
        {"Phrase": "internal fire door", "SuggestedElementCode": "E2", "Decision": "accept"},
    ]).to_excel(c, sheet_name="Candidates", index=False)
    imported = import_accepted_synonyms(c, d, confirm_overwrite=lambda _: True)
    assert imported == 1
    syn = pd.read_excel(d, sheet_name="Synonyms")
    assert len(syn[(syn["ElementCode"] == "E2") & (syn["Phrase"] == "internal fire door")]) == 1
    assert list(tmp_path.glob("master_dictionary.backup_*.xlsx"))
