from pathlib import Path

import pandas as pd

from app.synonym_candidates import SynonymCandidateService


def _make_dict(path: Path) -> None:
    with pd.ExcelWriter(path) as w:
        pd.DataFrame([
            {"ElementCode": "E1", "TitlePL": "Barierka", "TitleEN": "Guardrail", "NamePL": "Barierka", "NameEN": "Guardrail"},
            {"ElementCode": "E2", "TitlePL": "Drzwi pożarowe", "TitleEN": "Fire door", "NamePL": "Drzwi pożarowe", "NameEN": "Fire door"},
        ]).to_excel(w, sheet_name="Elements", index=False)
        pd.DataFrame([{"ElementCode": "E1", "Phrase": "stainless steel top rail", "Weight": 1.0, "MatchType": "exact", "Active": True}]).to_excel(w, sheet_name="Synonyms", index=False)
        pd.DataFrame(columns=["ElementCode", "Phrase", "Penalty"]).to_excel(w, sheet_name="NegativeSynonyms", index=False)


def test_noise_only_has_no_candidates(tmp_path: Path):
    src = tmp_path / "ex.xlsx"
    pd.DataFrame([{"crop_id": "c1", "raw_text": "+9.56\n150\nREI120\nD=40 mm\n150x5 mm"}]).to_excel(src, index=False)
    d = tmp_path / "dict.xlsx"
    _make_dict(d)
    out = SynonymCandidateService().generate_candidates(src, d, tmp_path)
    df = pd.read_excel(out)
    assert df.empty


def test_en_and_pl_candidates_and_aggregation(tmp_path: Path):
    src = tmp_path / "ex.xlsx"
    pd.DataFrame([
        {"crop_id": "c1", "raw_text": "STAINLESS STEEL TOP RAIL\nSŁUPEK BALUSTRADY ZE STALI NIERDZEWNEJ"},
        {"crop_id": "c2", "raw_text": "STAINLESS STEEL TOP RAIL"},
    ]).to_excel(src, index=False)
    d = tmp_path / "dict.xlsx"
    _make_dict(d)
    out = SynonymCandidateService().generate_candidates(src, d, tmp_path)
    df = pd.read_excel(out)
    assert (df["Phrase"].str.lower() == "stainless steel top rail").any()
    assert (df["Phrase"].str.lower() == "słupek balustrady ze stali nierdzewnej").any()
    rail = df[df["Phrase"].str.lower() == "stainless steel top rail"].iloc[0]
    assert rail["SourceCount"] == 2
    assert rail["OccurrenceCount"] == 2


def test_import_accept_and_deduplicate_and_backup(tmp_path: Path):
    dictionary = tmp_path / "master_dictionary.xlsx"
    _make_dict(dictionary)
    candidates = tmp_path / "synonym_candidates.xlsx"
    pd.DataFrame([
        {"Phrase": "internal fire door", "SuggestedElementCode": "E2", "Decision": "accept"},
        {"Phrase": "internal fire door", "SuggestedElementCode": "E2", "Decision": "accept"},
        {"Phrase": "ignored phrase", "SuggestedElementCode": "E2", "Decision": "pending"},
    ]).to_excel(candidates, sheet_name="Candidates", index=False)

    imported = SynonymCandidateService().import_accepted_candidates(candidates, dictionary, confirm_overwrite=lambda _: True)
    assert imported == 2

    syn = pd.read_excel(dictionary, sheet_name="Synonyms")
    accepted = syn[(syn["ElementCode"] == "E2") & (syn["Phrase"] == "internal fire door")]
    assert len(accepted) == 1

    backups = list(tmp_path.glob("master_dictionary.backup_*.xlsx"))
    assert backups
