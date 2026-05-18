from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .settings import AppSettings, resolve_tesseract_path

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OCRRuntimeStatus:
    state: str
    message: str
    tesseract_path: str
    tessdata_path: str
    languages: str


class OCRRuntime:
    def __init__(self, root_dir: Path, settings: AppSettings) -> None:
        self.root_dir = root_dir
        self.settings = settings

    def initialize(self) -> OCRRuntimeStatus:
        langs = self.settings.ocr_languages or "eng+pol"
        path, source = resolve_tesseract_path(self.root_dir, self.settings)
        LOGGER.info("ocr.detected_tesseract_path path=%s source=%s", path, source)
        if not path.exists():
            return OCRRuntimeStatus("missing", "OCR missing", str(path), "", langs)
        if not path.is_file():
            return OCRRuntimeStatus("invalid", "Invalid Tesseract path", str(path), "", langs)
        try:
            import pytesseract
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("ocr.pytesseract_import_error error=%s", exc)
            return OCRRuntimeStatus("missing", "OCR missing", str(path), "", langs)

        pytesseract.pytesseract.tesseract_cmd = str(path)
        tessdata_path = path.parent / "tessdata"
        os.environ["TESSDATA_PREFIX"] = str(tessdata_path)
        LOGGER.info("ocr.detected_tessdata path=%s", tessdata_path)

        if not tessdata_path.exists():
            return OCRRuntimeStatus("missing_tessdata", "Missing tessdata", str(path), str(tessdata_path), langs)

        missing_langs = []
        for lang in [x.strip() for x in langs.split("+") if x.strip()]:
            trained = tessdata_path / f"{lang}.traineddata"
            if not trained.exists():
                missing_langs.append(lang)

        if missing_langs:
            return OCRRuntimeStatus(
                "missing_lang",
                f"Missing language data: {'/'.join(missing_langs)}",
                str(path),
                str(tessdata_path),
                langs,
            )

        return OCRRuntimeStatus("available", "OCR initialized", str(path), str(tessdata_path), langs)
