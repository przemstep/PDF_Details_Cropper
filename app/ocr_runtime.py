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
    diagnostics: dict[str, str] | None = None


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
            LOGGER.error("Install dependency: pip install pytesseract Pillow")
            return OCRRuntimeStatus("missing_python_dependency", "OCR Python dependency missing: pytesseract", str(path), "", langs)

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

    def diagnostic_test(self) -> OCRRuntimeStatus:
        langs = self.settings.ocr_languages or "eng+pol"
        path, source = resolve_tesseract_path(self.root_dir, self.settings)
        LOGGER.info("ocr.detected_tesseract_path path=%s source=%s", path, source)

        diagnostics: dict[str, str] = {}

        if path.exists() and path.is_file():
            diagnostics["tesseract_exe"] = "ok"
        else:
            diagnostics["tesseract_exe"] = "missing"

        tessdata_path = path.parent / "tessdata"
        lang_checks: list[str] = []
        for lang in [x.strip() for x in langs.split("+") if x.strip()]:
            trained = tessdata_path / f"{lang}.traineddata"
            if trained.exists():
                lang_checks.append(f"{lang}:ok")
            else:
                lang_checks.append(f"{lang}:missing")
        diagnostics["tessdata"] = ", ".join(lang_checks)

        try:
            import pytesseract  # noqa: F401

            diagnostics["pytesseract_import"] = "ok"
        except Exception as exc:  # noqa: BLE001
            diagnostics["pytesseract_import"] = f"missing ({exc})"
            LOGGER.error("ocr.pytesseract_import_error error=%s", exc)
            LOGGER.error("Install dependency: pip install pytesseract Pillow")

        if diagnostics["tesseract_exe"] != "ok":
            state = "missing"
            message = "OCR missing"
        elif "missing" in diagnostics["tessdata"]:
            state = "missing_lang"
            message = "Missing language data"
        elif diagnostics["pytesseract_import"] != "ok":
            state = "missing_python_dependency"
            message = "OCR Python dependency missing: pytesseract"
        else:
            state = "available"
            message = "OCR initialized"

        return OCRRuntimeStatus(state, message, str(path), str(tessdata_path), langs, diagnostics)
