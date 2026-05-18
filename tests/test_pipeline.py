from pathlib import Path
import pandas as pd
import fitz
import pytest

from app.classifier import DetailClassifier
from app.pdf_extractor import PDFExtractor
from app.settings import AppSettings


@pytest.fixture
def minimal_settings() -> AppSettings:
    return AppSettings()


@pytest.fixture
def extractor_factory(tmp_path: Path, minimal_settings: AppSettings):
    def _create(intermediate_dir: Path, **kwargs) -> PDFExtractor:
        return PDFExtractor(intermediate_dir=intermediate_dir, root_dir=tmp_path, settings=minimal_settings, **kwargs)

    return _create


def make_dict(path: Path):
    with pd.ExcelWriter(path) as w:
        pd.DataFrame([{"ElementCode":"E1","Priority":100}]).to_excel(w, sheet_name="Elements", index=False)
        pd.DataFrame([{"ElementCode":"E1","Phrase":"pompa","Weight":1.0,"MatchType":"exact","Active":True}]).to_excel(w, sheet_name="Synonyms", index=False)
        pd.DataFrame(columns=["ElementCode","Phrase","Penalty"]).to_excel(w, sheet_name="NegativeSynonyms", index=False)


def test_classification_always_writes_report(tmp_path: Path):
    ex = pd.DataFrame([{"crop_id": "c1", "raw_text": "brak"}, {"crop_id": "c2", "raw_text": "pompa"}])
    ex_path = tmp_path / "extraction_data.xlsx"
    ex.to_excel(ex_path, index=False)
    dpath = tmp_path / "master_dictionary.xlsx"
    make_dict(dpath)
    out = tmp_path / "out"
    DetailClassifier().classify(ex_path, dpath, out)
    rep = pd.read_excel(out / "classification_report.xlsx")
    assert len(rep) == 2
    assert set(rep["status"]) == {"matched", "unmatched"}


def test_build_synonyms_from_elements_when_synonyms_empty(tmp_path: Path):
    extraction_path = tmp_path / "extraction_data.xlsx"
    pd.DataFrame([{"crop_id": "c1", "raw_text": "detail steel roof"}]).to_excel(extraction_path, index=False)

    dictionary_path = tmp_path / "master_dictionary.xlsx"
    with pd.ExcelWriter(dictionary_path) as writer:
        pd.DataFrame(
            [{"ElementCode": "RF.STL", "NamePL": "Dach stalowy", "NameEN": "Steel Roof", "Category": "Dach", "Priority": 90}]
        ).to_excel(writer, sheet_name="Elements", index=False)
        pd.DataFrame(columns=["ElementCode", "Phrase", "Weight", "MatchType", "Active"]).to_excel(writer, sheet_name="Synonyms", index=False)
        pd.DataFrame(columns=["ElementCode", "Phrase", "Penalty"]).to_excel(writer, sheet_name="NegativeSynonyms", index=False)

    output_dir = tmp_path / "out"
    results = DetailClassifier().classify(extraction_path, dictionary_path, output_dir)

    assert (output_dir / "classification_report.xlsx").exists()
    assert results[0].proposed_code == "RF.STL"

    report = pd.read_excel(output_dir / "classification_report.xlsx")
    assert report.loc[0, "status"] == "matched"
    assert report.loc[0, "classification_code"] == "RF.STL"


def test_sanitize_clip_rules(tmp_path: Path):
    page = fitz.Rect(0, 0, 100, 100)
    assert PDFExtractor.sanitize_clip((1, 1, 2, 2), page) is not None
    assert PDFExtractor.sanitize_clip((float("nan"), 0, 1, 1), page) is None
    assert PDFExtractor.sanitize_clip((5, 5, 5, 10), page) is None


def test_resolve_annotation_crop_rect_padding_and_page_clamp(tmp_path: Path, extractor_factory):
    pdf_path = tmp_path / "pad.pdf"
    doc = fitz.open()
    p = doc.new_page(width=200, height=200)
    a = p.add_rect_annot(fitz.Rect(5, 5, 30, 30))
    a.set_colors(stroke=(0, 85 / 255, 0))
    a.update()
    doc.save(pdf_path)
    doc.close()

    with fitz.open(pdf_path) as doc2:
        page = doc2[0]
        annot = page.first_annot
        ex = extractor_factory(tmp_path, crop_padding_pt=10)
        clip, annot_rect, _ = ex.resolve_annotation_crop_rect(annot, page)
        assert clip is not None and annot_rect is not None
        assert clip.x0 == 0 and clip.y0 == 0
        assert clip.x1 >= annot_rect.x1 and clip.y1 >= annot_rect.y1
        assert clip in page.rect


def test_page_number_naming_and_record_single_bbox(tmp_path: Path, extractor_factory):
    pdf_path = tmp_path / "a.pdf"
    doc = fitz.open()
    p = doc.new_page(width=200, height=200)
    a = p.add_rect_annot(fitz.Rect(10, 10, 100, 100))
    a.set_colors(stroke=(0, 85 / 255, 0))
    a.update()
    doc.save(pdf_path)
    doc.close()

    out = tmp_path / "out"
    recs = extractor_factory(out).extract(pdf_path, "proj")
    assert recs
    r = recs[0]
    assert "page_001" in r.base_name
    assert Path(r.crop_pdf).stem == Path(r.preview_png).stem == r.base_name
    assert r.page_index == 0 and r.page_number == 1
    assert r.annotation_rect_pdf_coords is not None
    assert r.bbox_pdf_coords[0] <= r.annotation_rect_pdf_coords[0]


def test_invalid_empty_non_finite_bbox_skipped(tmp_path: Path):
    page = fitz.Rect(0, 0, 100, 100)
    assert PDFExtractor.sanitize_clip((1, 1, 1, 3), page) is None
    assert PDFExtractor.sanitize_clip((1, 1, float("inf"), 3), page) is None


def test_png_without_annots(tmp_path: Path, extractor_factory):
    pdf_path = tmp_path / "annot.pdf"
    doc = fitz.open()
    p = doc.new_page(width=200, height=200)
    p.draw_rect(fitz.Rect(30, 30, 170, 170), color=(0, 0, 0), width=2)
    a = p.add_rect_annot(fitz.Rect(20, 20, 180, 180))
    a.set_colors(stroke=(0, 85 / 255, 0))
    a.update()
    doc.save(pdf_path)
    doc.close()

    out = tmp_path / "out"
    rec = extractor_factory(out).extract(pdf_path, "proj")[0]
    pix = fitz.Pixmap(rec.preview_png)
    assert pix.alpha == 0
