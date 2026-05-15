from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path


def ensure_log_dir(output_folder: Path | str) -> Path:
    output_folder = Path(output_folder)
    log_dir = output_folder / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def ascii_text(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def safe_filename(name: str, max_len: int = 120) -> str:
    name = ascii_text(name)
    name = re.sub(r"[#%*:<>?/\\|\"]", "", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:max_len]


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
