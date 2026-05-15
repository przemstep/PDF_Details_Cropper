from pathlib import Path
import pandas as pd
import fitz

from app.classifier import DetailClassifier
from app.pdf_extractor import PDFExtractor


def make_dict(path: Path):
    with pd.ExcelWriter(path) as w:
        pd.DataFrame([{"ElementCode":"E1","Priority":100}]).to_excel(w, sheet_name="Elements", index=False)
        pd.DataFrame([{"ElementCode":"E1","Phrase":"pompa","Weight":1.0,"MatchType":"exact","Active":True}]).to_excel(w, sheet_name="Synonyms", index=False)
        pd.DataFrame(columns=["ElementCode","Phrase","Penalty"]).to_excel(w, sheet_name="NegativeSynonyms", index=False)


def test_classification_always_writes_report(tmp_path: Path):
    ex = pd.DataFrame([{"crop_id":"c1","raw_text":"brak"},{"crop_id":"c2","raw_text":"pompa"}])
    ex_path = tmp_path / "extraction_data.xlsx"
    ex.to_excel(ex_path, index=False)
    dpath = tmp_path / "master_dictionary.xlsx"
    make_dict(dpath)
    out = tmp_path / "out"
    DetailClassifier().classify(ex_path, dpath, out)
    rep = pd.read_excel(out / "classification_report.xlsx")
    assert len(rep) == 2
    assert set(rep["status"]) == {"matched", "unmatched"}


def test_sanitize_clip_rules(tmp_path: Path):
    page = fitz.Rect(0, 0, 100, 100)
    assert PDFExtractor.sanitize_clip((1, 1, 2, 2), page) is not None
    assert PDFExtractor.sanitize_clip((float("nan"), 0, 1, 1), page) is None
    assert PDFExtractor.sanitize_clip((5, 5, 5, 10), page) is None


def test_page_number_naming_and_record_single_bbox(tmp_path: Path):
    pdf_path = tmp_path / "a.pdf"
    doc = fitz.open()
    p = doc.new_page(width=200, height=200)
    a = p.add_rect_annot(fitz.Rect(10, 10, 100, 100))
    a.set_colors(stroke=(0, 85/255, 0))
    a.update()
    doc.save(pdf_path)
    doc.close()

    out = tmp_path / "out"
    recs = PDFExtractor(out).extract(pdf_path, "proj")
    assert recs
    r = recs[0]
    assert "page_001" in r.base_name
    assert Path(r.crop_pdf).stem == Path(r.preview_png).stem == r.base_name
    assert r.page_index == 0 and r.page_number == 1
