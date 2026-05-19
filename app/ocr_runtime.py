from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .settings import AppSettings, resolve_tesseract_path

LOGGER = logging.getLogger(__name__)
OCR_TIMEOUT_SECONDS = 30


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
        if not path.exists() or not path.is_file():
            return OCRRuntimeStatus("missing_tesseract", "missing tesseract.exe", str(path), "", langs)

        tessdata_path = path.parent / "tessdata"
        if not tessdata_path.exists():
            return OCRRuntimeStatus("missing_tessdata", "missing tessdata", str(path), str(tessdata_path), langs)

        missing_langs = []
        for lang in [x.strip() for x in langs.split("+") if x.strip()]:
            trained = tessdata_path / f"{lang}.traineddata"
            if not trained.exists():
                missing_langs.append(lang)

        if missing_langs:
            return OCRRuntimeStatus("missing_tessdata", f"missing language data: {'/'.join(missing_langs)}", str(path), str(tessdata_path), langs)

        return OCRRuntimeStatus("available", "OCR ready", str(path), str(tessdata_path), langs)

    def diagnostic_test(self) -> OCRRuntimeStatus:
        status = self.initialize()
        diagnostics = {
            "tesseract_exe": "ok" if status.state != "missing_tesseract" else "missing",
            "tessdata": "ok" if status.state == "available" else "missing",
        }
        return OCRRuntimeStatus(status.state, status.message, status.tesseract_path, status.tessdata_path, status.languages, diagnostics)


def run_tesseract_ocr(image_path: Path, tesseract_exe: Path, lang: str = "eng+pol", timeout_seconds: int = OCR_TIMEOUT_SECONDS, tessdata_path: Path | None = None) -> tuple[str, str | None]:
    if not tesseract_exe.exists():
        return "", f"missing executable: {tesseract_exe}"

    env = os.environ.copy()
    if tessdata_path is not None and tessdata_path.exists():
        env["TESSDATA_PREFIX"] = str(tessdata_path)

    cmd = [str(tesseract_exe), str(image_path), "stdout", "-l", lang]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, env=env, check=False)
    except subprocess.TimeoutExpired:
        return "", f"OCR execution failed: timeout after {timeout_seconds}s"
    except Exception as exc:  # noqa: BLE001
        return "", f"OCR execution failed: {exc}"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return "", f"OCR execution failed: {stderr or f'code={result.returncode}'}"

    return (result.stdout or "").strip(), None
