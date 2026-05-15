from __future__ import annotations

import logging
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable

import fitz
import pandas as pd

from .models import CropRecord
from .utils import save_json

LOGGER = logging.getLogger(__name__)

COLOR_RULES: dict[str, tuple[int, int, int]] = {
    "detail": (0, 85, 0),
    "titleblock": (0, 0, 139),
    "ignore": (255, 255, 0),
}

MIN_CROP_WIDTH_PT = 50
MIN_CROP_HEIGHT_PT = 30
MIN_CROP_AREA_PT2 = 2000
MAX_CROP_WIDTH_PAGE_RATIO = 0.95
MAX_CROP_HEIGHT_PAGE_RATIO = 0.95
MAX_CROP_AREA_PAGE_RATIO = 0.90


def crop_record_to_dict(record: object) -> dict:
    if is_dataclass(record):
        return asdict(record)
    if hasattr(record, "model_dump"):
        return record.model_dump()
    if hasattr(record, "_asdict"):
        return record._asdict()
    if hasattr(record, "to_dict"):
        return record.to_dict()
    return {
        "crop_id": getattr(record, "crop_id", ""),
        "source_pdf": getattr(record, "source_pdf", ""),
        "page": getattr(record, "page", 0),
        "annotation_id": getattr(record, "annotation_id", ""),
        "bbox": getattr(record, "bbox", None),
        "raw_text": getattr(record, "raw_text", ""),
        "text_length": getattr(record, "text_length", 0),
        "ocr_used": getattr(record, "ocr_used", False),
        "crop_pdf": getattr(record, "crop_pdf", ""),
        "preview_png": getattr(record, "preview_png", ""),
        "area_type": getattr(record, "area_type", "unknown"),
    }


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
        if not colors:
            return "unknown"
        for color in colors:
            for name, target in COLOR_RULES.items():
                if all(abs(color[i] - target[i]) <= self.color_tolerance for i in range(3)):
                    return name
        return "unknown"

    def _validate_bbox(self, clip: fitz.Rect, page_rect: fitz.Rect) -> str | None:
        width, height = clip.width, clip.height
        area = width * height
        if width < MIN_CROP_WIDTH_PT:
            return f"width<{MIN_CROP_WIDTH_PT}"
        if height < MIN_CROP_HEIGHT_PT:
            return f"height<{MIN_CROP_HEIGHT_PT}"
        if area < MIN_CROP_AREA_PT2:
            return f"area<{MIN_CROP_AREA_PT2}"
        if width / page_rect.width > MAX_CROP_WIDTH_PAGE_RATIO:
            return "too_wide"
        if height / page_rect.height > MAX_CROP_HEIGHT_PAGE_RATIO:
            return "too_tall"
        if area / (page_rect.width * page_rect.height) > MAX_CROP_AREA_PAGE_RATIO:
            return "too_large_area"
        return None

    @staticmethod
    def sanitize_clip(rect: fitz.Rect | tuple[float, float, float, float], page_rect: fitz.Rect, min_size: float = 1.0) -> fitz.Rect | None:
        r = fitz.Rect(rect)
        values = [r.x0, r.y0, r.x1, r.y1]
        if not all(math.isfinite(v) for v in values):
            return None

        x0, x1 = sorted([r.x0, r.x1])
        y0, y1 = sorted([r.y0, r.y1])
        r = fitz.Rect(x0, y0, x1, y1)
        r = r & page_rect

        if r.is_empty or r.width < min_size or r.height < min_size:
            return None
        return r

    def ensure_output_dirs(self) -> dict[str, Path]:
        dirs = {
            "cropped_details": self.intermediate_dir / "cropped_details",
            "cropped_preview": self.intermediate_dir / "cropped_preview",
            "text": self.intermediate_dir / "text",
            "logs": self.intermediate_dir / "logs",
            "debug": self.intermediate_dir / "debug",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def extract(
        self,
        source_pdf: Path,
        project_name: str,
        status_callback: Callable[[str, str], None] | None = None,
    ) -> list[CropRecord]:
        dirs = self.ensure_output_dirs()

        def emit(level: str, message: str) -> None:
            if status_callback:
                status_callback(level, message)

        emit("INFO", "Wczytywanie PDF")

        records: list[CropRecord] = []
        report_rows: list[dict] = []
        processed_bbox = 0
        skipped_bbox = 0
        errors = 0
        detected_bbox = 0
        with fitz.open(source_pdf) as doc:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                emit("INFO", f"Analizowana strona: {page_idx + 1} / {len(doc)}")
                annot = page.first_annot
                item_idx = 0
                while annot:
                    rect = annot.rect
                    if rect:
                        detected_bbox += 1
                        annot_type = annot.type
                        annot_type_name = annot_type[1] if annot_type else ""
                        stroke_rgb = self._to_rgb255(annot.colors.get("stroke") if annot.colors else None)
                        fill_rgb = self._to_rgb255(annot.colors.get("fill") if annot.colors else None)
                        area_type = self._resolve_area_type(stroke_rgb, fill_rgb)

                        item_idx += 1
                        info = annot.info or {}
                        name = info.get("name")
                        subject = info.get("subject", "")
                        author = info.get("title", "")
                        crop_id = name or f"{project_name}_S{page_idx + 1}_D{item_idx}"
                        width = rect.width
                        height = rect.height
                        area = width * height

                        emit("INFO", f"Analizowany bbox: {item_idx} (strona {page_idx + 1})")
                        emit("INFO", f"Strona {page_idx + 1}, bbox {item_idx}: type={annot_type_name}, stroke={stroke_rgb}, fill={fill_rgb}, class={area_type}")
                        LOGGER.info("PAGE RECT: %s", page.rect)
                        LOGGER.info("RAW BBOX: %s", rect)

                        clip = self.sanitize_clip(rect, page.rect)
                        LOGGER.info("SANITIZED CLIP: %s", clip)
                        if clip is None:
                            skipped_bbox += 1
                            errors += 1
                            report_rows.append({
                                "source_pdf": source_pdf.name, "page_number": page_idx + 1, "annotation_index": item_idx,
                                "type": annot_type_name, "name": name, "subject": subject, "author": author,
                                "stroke_rgb": stroke_rgb, "fill_rgb": fill_rgb, "classification": area_type,
                                "bbox": (rect.x0, rect.y0, rect.x1, rect.y1), "width": width, "height": height, "area": area,
                                "status": "skipped", "errors": "invalid_clip", "output_pdf": "", "output_png": "",
                            })
                            LOGGER.error(
                                "Strona %s, bbox %s: pominięto błędny bbox: %s",
                                page_idx + 1,
                                item_idx,
                                rect,
                            )
                            emit("ERROR", f"Błąd: Strona {page_idx + 1}, bbox {item_idx} — pominięto błędny bbox")
                            annot = annot.next
                            continue

                        if not (annot_type and (annot_type[0] == fitz.PDF_ANNOT_SQUARE or annot_type_name == "Square")):
                            skipped_bbox += 1
                            report_rows.append({"source_pdf": source_pdf.name, "page_number": page_idx + 1, "annotation_index": item_idx,
                                "type": annot_type_name, "name": name, "subject": subject, "author": author,
                                "stroke_rgb": stroke_rgb, "fill_rgb": fill_rgb, "classification": area_type,
                                "bbox": (clip.x0, clip.y0, clip.x1, clip.y1), "width": clip.width, "height": clip.height, "area": clip.width*clip.height,
                                "status": "skipped", "errors": "not_square", "output_pdf": "", "output_png": ""})
                            annot = annot.next
                            continue

                        if self.author_filter and author != self.author_filter:
                            skipped_bbox += 1
                            report_rows.append({"source_pdf": source_pdf.name, "page_number": page_idx + 1, "annotation_index": item_idx,
                                "type": annot_type_name, "name": name, "subject": subject, "author": author,
                                "stroke_rgb": stroke_rgb, "fill_rgb": fill_rgb, "classification": area_type,
                                "bbox": (clip.x0, clip.y0, clip.x1, clip.y1), "width": clip.width, "height": clip.height, "area": clip.width*clip.height,
                                "status": "skipped", "errors": "author_mismatch", "output_pdf": "", "output_png": ""})
                            annot = annot.next
                            continue

                        size_error = self._validate_bbox(clip, page.rect)
                        if size_error:
                            skipped_bbox += 1
                            report_rows.append({"source_pdf": source_pdf.name, "page_number": page_idx + 1, "annotation_index": item_idx,
                                "type": annot_type_name, "name": name, "subject": subject, "author": author,
                                "stroke_rgb": stroke_rgb, "fill_rgb": fill_rgb, "classification": area_type,
                                "bbox": (clip.x0, clip.y0, clip.x1, clip.y1), "width": clip.width, "height": clip.height, "area": clip.width*clip.height,
                                "status": "skipped", "errors": size_error, "output_pdf": "", "output_png": ""})
                            annot = annot.next
                            continue

                        if area_type != "detail":
                            skipped_bbox += 1
                            report_rows.append({"source_pdf": source_pdf.name, "page_number": page_idx + 1, "annotation_index": item_idx,
                                "type": annot_type_name, "name": name, "subject": subject, "author": author,
                                "stroke_rgb": stroke_rgb, "fill_rgb": fill_rgb, "classification": area_type,
                                "bbox": (clip.x0, clip.y0, clip.x1, clip.y1), "width": clip.width, "height": clip.height, "area": clip.width*clip.height,
                                "status": "skipped", "errors": f"classification_{area_type}", "output_pdf": "", "output_png": ""})
                            annot = annot.next
                            continue

                        crop_pdf_path = dirs["cropped_details"] / f"{crop_id}.pdf"
                        crop_png_path = dirs["cropped_preview"] / f"{crop_id}.png"
                        txt_path = dirs["text"] / f"{crop_id}.txt"

                        crop_doc = fitz.open()
                        crop_page = crop_doc.new_page(width=clip.width, height=clip.height)
                        crop_page.show_pdf_page(crop_page.rect, doc, page_idx, clip=clip)
                        crop_doc.save(crop_pdf_path)
                        crop_doc.close()

                        emit("INFO", "Renderowanie fragmentu")
                        pix = page.get_pixmap(clip=clip, dpi=150)
                        pix.save(crop_png_path)
                        text = page.get_text("text", clip=clip) or ""
                        txt_path.write_text(text, encoding="utf-8")
                        processed_bbox += 1

                        records.append(
                            CropRecord(
                                crop_id=crop_id,
                                source_pdf=source_pdf.name,
                                page=page_idx + 1,
                                annotation_id=str(annot.xref),
                                bbox=(clip.x0, clip.y0, clip.x1, clip.y1),
                                raw_text=text,
                                text_length=len(text),
                                ocr_used=False,
                                crop_pdf=str(crop_pdf_path),
                                preview_png=str(crop_png_path),
                                area_type=area_type,
                                annotation_index=item_idx,
                                annotation_type=annot_type_name,
                                annotation_name=name or "",
                                annotation_subject=subject,
                                author=author,
                                stroke_rgb=stroke_rgb,
                                fill_rgb=fill_rgb,
                                border_width=(annot.border or {}).get("width") if annot.border else None,
                                border_style=str((annot.border or {}).get("style")) if annot.border else "",
                                opacity_value=annot.opacity,
                                width=clip.width,
                                height=clip.height,
                                area=clip.width * clip.height,
                                status="exported",
                            )
                        )
                        report_rows.append({
                            "source_pdf": source_pdf.name, "page_number": page_idx + 1, "annotation_index": item_idx,
                            "type": annot_type_name, "name": name, "subject": subject, "author": author,
                            "stroke_rgb": stroke_rgb, "fill_rgb": fill_rgb, "classification": area_type,
                            "bbox": (clip.x0, clip.y0, clip.x1, clip.y1), "width": clip.width, "height": clip.height, "area": clip.width * clip.height,
                            "status": "exported", "errors": "", "output_pdf": str(crop_pdf_path), "output_png": str(crop_png_path),
                        })
                    annot = annot.next

        emit("INFO", "Eksport danych")
        df = pd.DataFrame([crop_record_to_dict(r) for r in records])
        df.to_excel(self.intermediate_dir / "extraction_data.xlsx", index=False)
        save_json(self.intermediate_dir / "extraction_data.json", [crop_record_to_dict(r) for r in records])
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(self.intermediate_dir / "extraction_report.csv", index=False)
        save_json(self.intermediate_dir / "extraction_report.json", report_rows)
        emit("INFO", "Zakończono")
        emit(
            "INFO",
            (
                f"Zakończono analizę. Strony: {len({r.page for r in records})}; "
                f"BBoxes wykryte: {detected_bbox}; BBoxes przetworzone: {processed_bbox}; "
                f"BBoxes pominięte: {skipped_bbox}; Błędy: {errors}; "
                f"Folder wynikowy: {self.intermediate_dir}; Log: {self.intermediate_dir / 'logs/pdf_analyzer.log'}"
            ),
        )
        LOGGER.info("Extraction finished: %d records", len(records))
        return records
