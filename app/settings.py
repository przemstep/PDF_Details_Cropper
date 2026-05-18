from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PORTABLE_TESSERACT_EXE = Path("tesseract") / "tesseract.exe"
DEFAULT_OCR_LANGUAGES = "eng+pol"


@dataclass(slots=True)
class AppSettings:
    tesseract_exe_path: str = ""
    ocr_languages: str = DEFAULT_OCR_LANGUAGES


def settings_path(root_dir: Path) -> Path:
    return root_dir / "config" / "settings.json"


def load_settings(root_dir: Path) -> AppSettings:
    path = settings_path(root_dir)
    if not path.exists():
        return AppSettings()
    data = json.loads(path.read_text(encoding="utf-8"))
    return AppSettings(
        tesseract_exe_path=(data.get("tesseract_exe_path") or "").strip(),
        ocr_languages=(data.get("ocr_languages") or DEFAULT_OCR_LANGUAGES).strip(),
    )


def save_settings(root_dir: Path, settings: AppSettings) -> None:
    path = settings_path(root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_tesseract_path(root_dir: Path, settings: AppSettings) -> tuple[Path, str]:
    configured = (settings.tesseract_exe_path or "").strip()
    if configured:
        return Path(configured), "settings"
    env_path = os.getenv("PDF_ANALYZER_TESSERACT_EXE", "").strip()
    if env_path:
        return Path(env_path), "env"
    default_path = Path(DEFAULT_TESSERACT_EXE)
    if default_path.exists():
        return default_path, "default"
    portable_path = (root_dir / PORTABLE_TESSERACT_EXE).resolve()
    return portable_path, "portable"
