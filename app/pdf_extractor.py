from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import fitz
import pandas as pd

from .models import CropRecord
from .utils import save_json

LOGGER = logging.getLogger(__name__)

COLOR_RULES: dict[str, tuple[int, int, int]] = {
    "detail": (0, 85, 0),
    "drawing_table": (0, 0, 139),
    "ignore": (255, 255, 0),
}


class PDFExtractor:
    def __init__(self, intermediate_dir: Path, color_tolerance: int = 20) -> None:
        self.intermediate_dir = intermediate_dir
        self.color_tolerance = color_tolerance

    @staticmethod
    def _to_rgb255(color: tuple[float, float, float] | None) -> tuple[int, int, int] | None:
        if color is None or len(color) < 3:
            return None
        return tuple(max(0, min(255, int(round(channel * 255)))) for channel in color[:3])

    def _resolve_area_type(self, color: tuple[int, int, int] | None) -> str:
        if color is None:
            return "detail"

        best_name = "detail"
        best_distance = float("inf")
        for name, target in COLOR_RULES.items():
            distance = sum(abs(color[i] - target[i]) for i in range(3))
            if distance < best_distance:
                best_name = name
                best_distance = distance

        return best_name if best_distance <= self.color_tolerance * 3 else "detail"

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
                        fill_rgb = self._to_rgb255(annot.colors.get("fill") if annot.colors else None)
                        area_type = self._resolve_area_type(fill_rgb)

                        if area_type == "ignore":
                            annot = annot.next
                            continue

                        item_idx += 1
                        name = annot.info.get("name") if annot.info else None
                        crop_id = name or f"{project_name}_S{page_idx + 1}_D{item_idx}"
                        text = page.get_text("text", clip=rect) or ""

                        emit("INFO", f"Analizowany bbox: {item_idx} (strona {page_idx + 1})")
                        emit("INFO", f"Strona {page_idx + 1}, bbox {item_idx}: analiza")
                        LOGGER.info("PAGE RECT: %s", page.rect)
                        LOGGER.info("RAW BBOX: %s", rect)

                        clip = self.sanitize_clip(rect, page.rect)
                        LOGGER.info("SANITIZED CLIP: %s", clip)
                        if clip is None:
                            skipped_bbox += 1
                            errors += 1
                            LOGGER.error(
                                "Strona %s, bbox %s: pominięto błędny bbox: %s",
                                page_idx + 1,
                                item_idx,
                                rect,
                            )
                            emit("ERROR", f"Błąd: Strona {page_idx + 1}, bbox {item_idx} — pominięto błędny bbox")
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
                            )
                        )
                    annot = annot.next

        emit("INFO", "Eksport danych")
        df = pd.DataFrame([r.__dict__ for r in records])
        df.to_excel(self.intermediate_dir / "extraction_data.xlsx", index=False)
        save_json(self.intermediate_dir / "extraction_data.json", [r.__dict__ for r in records])
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
