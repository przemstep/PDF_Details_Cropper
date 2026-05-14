from __future__ import annotations

import logging
from pathlib import Path

import fitz
import pandas as pd

from .models import CropRecord
from .utils import save_json

LOGGER = logging.getLogger(__name__)


class PDFExtractor:
    def __init__(self, intermediate_dir: Path) -> None:
        self.intermediate_dir = intermediate_dir

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
                            )
                        )
                    annot = annot.next

        df = pd.DataFrame([r.__dict__ for r in records])
        df.to_excel(self.intermediate_dir / "extraction_data.xlsx", index=False)
        save_json(self.intermediate_dir / "extraction_data.json", [r.__dict__ for r in records])
        LOGGER.info("Extraction finished: %d records", len(records))
        return records
