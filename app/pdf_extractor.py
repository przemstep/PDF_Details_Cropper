from __future__ import annotations

import logging
from pathlib import Path

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

    def extract(self, source_pdf: Path, project_name: str) -> list[CropRecord]:
        crop_pdf_dir = self.intermediate_dir / "crops_pdf"
        crop_png_dir = self.intermediate_dir / "crops_png"
        text_dir = self.intermediate_dir / "text"
        for d in (crop_pdf_dir, crop_png_dir, text_dir):
            d.mkdir(parents=True, exist_ok=True)

        records: list[CropRecord] = []
        with fitz.open(source_pdf) as doc:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                annot = page.first_annot
                item_idx = 0
                while annot:
                    rect = annot.rect
                    if rect:
                        fill_rgb = self._to_rgb255(annot.colors.get("fill") if annot.colors else None)
                        area_type = self._resolve_area_type(fill_rgb)

                        if area_type == "ignore":
                            annot = annot.next
                            continue

                        item_idx += 1
                        name = annot.info.get("name") if annot.info else None
                        crop_id = name or f"{project_name}_S{page_idx + 1}_D{item_idx}"
                        text = page.get_text("text", clip=rect) or ""

                        crop_pdf_path = crop_pdf_dir / f"{crop_id}.pdf"
                        crop_png_path = crop_png_dir / f"{crop_id}.png"
                        txt_path = text_dir / f"{crop_id}.txt"

                        crop_doc = fitz.open()
                        crop_page = crop_doc.new_page(width=rect.width, height=rect.height)
                        crop_page.show_pdf_page(crop_page.rect, doc, page_idx, clip=rect)
                        crop_doc.save(crop_pdf_path)
                        crop_doc.close()

                        pix = page.get_pixmap(clip=rect, dpi=150)
                        pix.save(crop_png_path)
                        txt_path.write_text(text, encoding="utf-8")

                        records.append(
                            CropRecord(
                                crop_id=crop_id,
                                source_pdf=source_pdf.name,
                                page=page_idx + 1,
                                annotation_id=str(annot.xref),
                                bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                                raw_text=text,
                                text_length=len(text),
                                ocr_used=False,
                                crop_pdf=str(crop_pdf_path),
                                preview_png=str(crop_png_path),
                                area_type=area_type,
                            )
                        )
                    annot = annot.next

        df = pd.DataFrame([r.__dict__ for r in records])
        df.to_excel(self.intermediate_dir / "extraction_data.xlsx", index=False)
        save_json(self.intermediate_dir / "extraction_data.json", [r.__dict__ for r in records])
        LOGGER.info("Extraction finished: %d records", len(records))
        return records
