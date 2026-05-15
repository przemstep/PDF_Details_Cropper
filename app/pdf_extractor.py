from __future__ import annotations

import logging
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable

import fitz
import pandas as pd

from .models import CropRecord
from .utils import save_json, safe_filename

LOGGER = logging.getLogger(__name__)

COLOR_RULES: dict[str, tuple[int, int, int]] = {
    "detail": (0, 85, 0),
    "titleblock": (0, 0, 139),
    "ignore": (255, 255, 0),
}

OCR_ENABLED = True
OCR_LANGUAGE = "eng+pol"
OCR_MIN_TEXT_LENGTH = 10
OCR_DPI = 300


class OCRUnavailableError(RuntimeError):
    pass


def crop_record_to_dict(record: object) -> dict:
    if is_dataclass(record):
        return asdict(record)
    if hasattr(record, "model_dump"):
        return record.model_dump()
    if hasattr(record, "_asdict"):
        return record._asdict()
    return {}


class PDFExtractor:
    def __init__(self, intermediate_dir: Path, color_tolerance: int = 3, author_filter: str | None = None) -> None:
        self.intermediate_dir = intermediate_dir
        self.color_tolerance = color_tolerance
        self.author_filter = author_filter

    @staticmethod
    def _to_rgb255(color: tuple[float, float, float] | None) -> tuple[int, int, int] | None:
        if color is None or len(color) < 3:
            return None
        return tuple(max(0, min(255, int(round(channel * 255)))) for channel in color[:3])

    def _resolve_area_type(self, stroke: tuple[int, int, int] | None, fill: tuple[int, int, int] | None) -> str:
        colors = [c for c in (stroke, fill) if c is not None]
        for color in colors:
            for name, target in COLOR_RULES.items():
                if all(abs(color[i] - target[i]) <= self.color_tolerance for i in range(3)):
                    return name
        return "unknown"

    @staticmethod
    def sanitize_clip(rect: fitz.Rect | tuple[float, float, float, float], page_rect: fitz.Rect) -> fitz.Rect | None:
        r = fitz.Rect(rect)
        vals = [r.x0, r.y0, r.x1, r.y1]
        if not all(math.isfinite(v) for v in vals):
            return None
        x0, x1 = sorted([r.x0, r.x1])
        y0, y1 = sorted([r.y0, r.y1])
        r = fitz.Rect(x0, y0, x1, y1)
        if r.is_empty or r.width <= 0 or r.height <= 0:
            return None
        if (r & page_rect).is_empty:
            return None
        clipped = r & page_rect
        if clipped.is_empty or clipped.width <= 0 or clipped.height <= 0:
            return None
        return clipped

    def ensure_output_dirs(self) -> dict[str, Path]:
        dirs = {
            "cropped_details": self.intermediate_dir / "cropped_details",
            "cropped_preview": self.intermediate_dir / "cropped_preview",
            "text": self.intermediate_dir / "text",
            "text_meta": self.intermediate_dir / "text_meta",
            "logs": self.intermediate_dir / "logs",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def _run_ocr(self, png_path: Path) -> str:
        try:
            import pytesseract
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            raise OCRUnavailableError(str(exc)) from exc
        return pytesseract.image_to_string(Image.open(png_path), lang=OCR_LANGUAGE)

    def _extract_text_hybrid(self, page: fitz.Page, clip: fitz.Rect, png_path: Path) -> tuple[str, dict, bool]:
        pymupdf_text = (page.get_text("text", clip=clip) or "").strip()
        ocr_text = ""
        errors: list[str] = []
        text_source = "pymupdf"
        used_ocr = False

        if OCR_ENABLED and len(pymupdf_text) < OCR_MIN_TEXT_LENGTH:
            try:
                ocr_text = (self._run_ocr(png_path) or "").strip()
                used_ocr = True
            except Exception as exc:  # noqa: BLE001
                errors.append(f"ocr_failed:{exc}")

        combined = pymupdf_text
        if ocr_text and pymupdf_text:
            combined = f"{pymupdf_text}\n\n{ocr_text}".strip()
            text_source = "pymupdf+ocr"
        elif ocr_text:
            combined = ocr_text
            text_source = "ocr"
        elif not pymupdf_text:
            text_source = "empty"

        meta = {
            "text_source": text_source,
            "pymupdf_text_length": len(pymupdf_text),
            "ocr_text_length": len(ocr_text),
            "ocr_engine": "pytesseract",
            "errors": errors,
        }
        return combined, meta, used_ocr

    def extract(self, source_pdf: Path, project_name: str, status_callback: Callable[[str, str, str], None] | None = None) -> list[CropRecord]:
        dirs = self.ensure_output_dirs()

        def emit(level: str, message: str) -> None:
            if status_callback:
                status_callback("extract", level, message)

        records: list[CropRecord] = []
        report_rows: list[dict] = []
        with fitz.open(source_pdf) as doc:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_number = page_index + 1
                emit("INFO", f"strona={page_number}/{len(doc)}")
                annot = page.first_annot
                annotation_index = 0
                while annot:
                    annotation_index += 1
                    clip = self.sanitize_clip(annot.rect, page.rect)
                    if clip is None:
                        report_rows.append({"page_number": page_number, "annotation_index": annotation_index, "status": "skipped", "error": "invalid_clip"})
                        annot = annot.next
                        continue

                    annot_type = annot.type
                    annot_type_name = annot_type[1] if annot_type else ""
                    if not (annot_type and (annot_type[0] == fitz.PDF_ANNOT_SQUARE or annot_type_name == "Square")):
                        annot = annot.next
                        continue

                    stroke_rgb = self._to_rgb255(annot.colors.get("stroke") if annot.colors else None)
                    fill_rgb = self._to_rgb255(annot.colors.get("fill") if annot.colors else None)
                    area_type = self._resolve_area_type(stroke_rgb, fill_rgb)
                    if area_type != "detail":
                        annot = annot.next
                        continue

                    page_str = f"page_{page_number:03d}"
                    detail_str = f"detail_{annotation_index:03d}"
                    base_name = safe_filename(f"{project_name}_{page_str}_{detail_str}")
                    crop_pdf_path = dirs["cropped_details"] / f"{base_name}.pdf"
                    crop_png_path = dirs["cropped_preview"] / f"{base_name}.png"
                    txt_path = dirs["text"] / f"{base_name}.txt"
                    txt_meta_path = dirs["text_meta"] / f"{base_name}.json"

                    crop_doc = fitz.open()
                    crop_page = crop_doc.new_page(width=clip.width, height=clip.height)
                    crop_page.show_pdf_page(crop_page.rect, doc, page_index, clip=clip)
                    crop_doc.save(crop_pdf_path)
                    crop_doc.close()

                    pix = page.get_pixmap(clip=clip, dpi=OCR_DPI)
                    pix.save(crop_png_path)
                    text, text_meta, ocr_used = self._extract_text_hybrid(page, clip, crop_png_path)
                    txt_path.write_text(text, encoding="utf-8")
                    save_json(txt_meta_path, text_meta)

                    emit("INFO", f"strona={page_number} bbox={annotation_index} output={base_name}")
                    LOGGER.info(
                        "crop base=%s page_number=%s bbox=%s pdf=%s png=%s txt=%s",
                        base_name,
                        page_number,
                        (clip.x0, clip.y0, clip.x1, clip.y1),
                        crop_pdf_path,
                        crop_png_path,
                        txt_path,
                    )

                    record = CropRecord(
                        crop_id=base_name,
                        base_name=base_name,
                        source_pdf=source_pdf.name,
                        page_index=page_index,
                        page_number=page_number,
                        annotation_id=str(annot.xref),
                        bbox_pdf_coords=(clip.x0, clip.y0, clip.x1, clip.y1),
                        raw_text=text,
                        text_length=len(text),
                        ocr_used=ocr_used,
                        crop_pdf=str(crop_pdf_path),
                        preview_png=str(crop_png_path),
                        annotation_index=annotation_index,
                        area_type=area_type,
                        status="exported",
                    )
                    records.append(record)
                    report_rows.append({"base_name": base_name, "page_number": page_number, "bbox": record.bbox_pdf_coords, "status": "exported"})
                    annot = annot.next

        pd.DataFrame([crop_record_to_dict(r) for r in records]).to_excel(self.intermediate_dir / "extraction_data.xlsx", index=False)
        save_json(self.intermediate_dir / "extraction_data.json", [crop_record_to_dict(r) for r in records])
        pd.DataFrame(report_rows).to_csv(self.intermediate_dir / "extraction_report.csv", index=False)
        save_json(self.intermediate_dir / "extraction_report.json", report_rows)
        emit("INFO", f"done processed={len(records)}")
        return records
