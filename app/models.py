from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AnnotationBox:
    page_number: int
    annotation_id: str
    annotation_name: str
    annotation_subject: str
    annotation_contents: str
    bbox: tuple[float, float, float, float]
    fill_color: tuple[float, float, float] | None = None
    opacity: float | None = None


@dataclass(slots=True)
class CropRecord:
    crop_id: str
    base_name: str
    source_pdf: str
    page_index: int
    page_number: int
    annotation_id: str
    bbox_pdf_coords: tuple[float, float, float, float]
    raw_text: str
    text_length: int
    ocr_used: bool
    crop_pdf: str
    preview_png: str
    area_type: str = "detail"
    annotation_index: int = 0
    annotation_type: str = ""
    annotation_name: str = ""
    annotation_subject: str = ""
    author: str = ""
    stroke_rgb: tuple[int, int, int] | None = None
    fill_rgb: tuple[int, int, int] | None = None
    border_width: float | None = None
    border_style: str = ""
    opacity_value: float | None = None
    width: float = 0.0
    height: float = 0.0
    area: float = 0.0
    status: str = "accepted"
    errors: str = ""


@dataclass(slots=True)
class ElementDefinition:
    element_code: str
    title_pl: str
    title_en: str
    category: str
    priority: int = 100


@dataclass(slots=True)
class SynonymDefinition:
    element_code: str
    language: str
    phrase: str
    weight: float
    match_type: str
    active: bool = True


@dataclass(slots=True)
class ClassificationResult:
    crop_id: str
    proposed_code: str
    detected_elements: list[str]
    scores: dict[str, float]
    confidence: float
    keywords: list[str] = field(default_factory=list)
    keywords_pl: list[str] = field(default_factory=list)
    keywords_en: list[str] = field(default_factory=list)
    technical_phrases: list[str] = field(default_factory=list)
    search_tags: list[str] = field(default_factory=list)
    sharepoint_tags: str = ""
    search_text: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
