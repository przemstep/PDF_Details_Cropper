# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1
# 26.05.2026, working code, one summary PNG, annotated PDF


from __future__ import annotations

import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class DetectorConfig:
    dpi: int = 150

    # PDF text
    min_text_width_px: int = 6
    min_text_height_px: int = 4
    text_zone_margin_px: int = 2

    # ink / components
    min_component_pixels: int = 20
    min_geometry_pixels: int = 40
    max_component_area_ratio: float = 0.90
    text_ink_overlap_ratio: float = 0.80

    # long/large ink inside text zones should remain geometry
    geometry_override_min_width_px: int = 150
    geometry_override_min_height_px: int = 80
    geometry_override_min_pixels_multiplier: int = 5

    # thresholding
    adaptive_threshold_block_size: int = 31
    adaptive_threshold_c: int = 12

    # clustering
    geometry_touch_px: int = 6
    text_geometry_touch_px: int = 12

    # side/top/bottom text attachment
    side_text_search_px: int = 360
    side_text_vertical_slack_px: int = 120
    top_bottom_text_search_px: int = 80
    top_bottom_text_horizontal_slack_px: int = 60
    attach_inside_text: bool = True
    attach_inside_text_inset_px: int = 2

    # final bbox
    output_margin_px: int = 35
    min_detail_width_px: int = 120
    min_detail_height_px: int = 120
    min_detail_area_px: int = 15_000
    max_detail_area_ratio: float = 0.70

    # absorb overlapping core boxes
    absorb_contain_margin_px: int = 8
    absorb_overlap_ratio_threshold: float = 0.25

    # dedupe final candidates
    dedupe_smaller_overlap_threshold: float = 0.80

    # debug / output
    save_debug: bool = True
    export_crops: bool = False

    # PDF annotations
    write_pdf_annotations: bool = False
    annotation_opacity: float = 0.5

    #OCR
    use_ocr_fallback: bool = True
    ocr_lang: str = "eng+pol"
    ocr_min_pdf_text_count: int = 3
    tesseract_cmd: Optional[str] = None


# ============================================================
# 2. DATA
# ============================================================

@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    def area(self) -> int:
        return self.width() * self.height()

    def intersects(self, other: "Box") -> bool:
        return not (
            self.x1 <= other.x0
            or self.x0 >= other.x1
            or self.y1 <= other.y0
            or self.y0 >= other.y1
        )

    def contains(self, other: "Box", margin: int = 0) -> bool:
        return (
            self.x0 <= other.x0 + margin
            and self.y0 <= other.y0 + margin
            and self.x1 >= other.x1 - margin
            and self.y1 >= other.y1 - margin
        )

    def union(self, other: "Box") -> "Box":
        return Box(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def expand(self, margin: int, image_w: int, image_h: int) -> "Box":
        return Box(
            max(0, self.x0 - margin),
            max(0, self.y0 - margin),
            min(image_w, self.x1 + margin),
            min(image_h, self.y1 + margin),
        )


@dataclass
class TextObject:
    bbox: Box
    text: str


@dataclass
class InkComponent:
    bbox: Box
    kind: str  # geometry / text_ink / noise
    pixel_area: int
    text_overlap_ratio: float = 0.0
    fill_ratio: float = 0.0


@dataclass
class DetailCandidate:
    core_box: Box
    final_box: Box
    texts: List[TextObject]


# ============================================================
# 3. HELPERS
# ============================================================

def validate_config(cfg: DetectorConfig) -> DetectorConfig:
    if cfg.adaptive_threshold_block_size < 3:
        cfg.adaptive_threshold_block_size = 3

    if cfg.adaptive_threshold_block_size % 2 == 0:
        cfg.adaptive_threshold_block_size += 1

    cfg.annotation_opacity = max(0.0, min(1.0, cfg.annotation_opacity))

    return cfg


def boxes_touch_or_intersect(a: Box, b: Box, tolerance_px: int) -> bool:
    return not (
        a.x1 + tolerance_px <= b.x0
        or a.x0 - tolerance_px >= b.x1
        or a.y1 + tolerance_px <= b.y0
        or a.y0 - tolerance_px >= b.y1
    )


def contains_inside(a: Box, b: Box, inset: int = 0) -> bool:
    return (
        a.x0 + inset <= b.x0
        and a.y0 + inset <= b.y0
        and a.x1 - inset >= b.x1
        and a.y1 - inset >= b.y1
    )


def union_boxes(boxes: List[Box]) -> Optional[Box]:
    if not boxes:
        return None

    out = boxes[0]
    for b in boxes[1:]:
        out = out.union(b)

    return out


def intersection_area(a: Box, b: Box) -> int:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)

    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


# ============================================================
# 4. PDF RENDER + TEXT
# ============================================================

def render_page_to_image(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def extract_text_objects(
    page: fitz.Page,
    scale: float,
    cfg: DetectorConfig,
) -> List[TextObject]:
    result: List[TextObject] = []
    data = page.get_text("dict")

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            line_box: Optional[Box] = None
            parts: List[str] = []

            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                if not txt:
                    continue

                x0, y0, x1, y1 = span["bbox"]
                b = Box(
                    int(x0 * scale),
                    int(y0 * scale),
                    int(x1 * scale),
                    int(y1 * scale),
                )

                if b.width() < cfg.min_text_width_px or b.height() < cfg.min_text_height_px:
                    continue

                line_box = b if line_box is None else line_box.union(b)
                parts.append(txt)

            if line_box is not None:
                result.append(TextObject(bbox=line_box, text=" ".join(parts)))

    return result


def extract_text_objects_ocr(
    image_bgr: np.ndarray,
    cfg: DetectorConfig,
) -> List[TextObject]:
    """
    OCR fallback dla stron, gdzie PDF text layer jest pusty.
    Zwraca TextObject w pikselach obrazu, więc nie wymaga scale.
    """

    if cfg.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray,
        lang=cfg.ocr_lang,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )

    result: List[TextObject] = []

    n = len(data["text"])

    for i in range(n):
        text = str(data["text"][i]).strip()

        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except ValueError:
            conf = -1

        if conf < 30:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        if w < cfg.min_text_width_px or h < cfg.min_text_height_px:
            continue

        result.append(
            TextObject(
                bbox=Box(x, y, x + w, y + h),
                text=text,
            )
        )

    return result

def extract_text_objects_with_fallback(
    page: fitz.Page,
    image_bgr: np.ndarray,
    scale: float,
    cfg: DetectorConfig,
) -> List[TextObject]:
    """
    Najpierw próbuje pobrać tekst z PDF.
    Jeśli tekstu jest za mało, używa OCR.
    """

    text_objects = extract_text_objects(page, scale, cfg)

    if not cfg.use_ocr_fallback:
        return text_objects

    if len(text_objects) >= cfg.ocr_min_pdf_text_count:
        return text_objects

    print("[INFO] PDF text layer pusty / ubogi — uruchamiam OCR fallback")

    try:
        ocr_objects = extract_text_objects_ocr(image_bgr, cfg)

        if ocr_objects:
            return ocr_objects

    except Exception as exc:
        print(f"[WARNING] OCR fallback failed: {exc}")

    return text_objects

# ============================================================
# 5. MASKS
# ============================================================

def make_ink_mask(image_bgr: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=cfg.adaptive_threshold_block_size,
        C=cfg.adaptive_threshold_c,
    )


def remove_border_margin(mask: np.ndarray, margin_px: int = 8) -> np.ndarray:
    out = mask.copy()

    out[:margin_px, :] = 0
    out[-margin_px:, :] = 0
    out[:, :margin_px] = 0
    out[:, -margin_px:] = 0

    return out


def remove_border_connected_components(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    keep = np.zeros(num_labels, dtype=np.uint8)

    for label_id in range(1, num_labels):
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]

        touches_border = (
            x <= 1
            or y <= 1
            or x + bw >= w - 2
            or y + bh >= h - 2
        )

        if not touches_border:
            keep[label_id] = 255

    return keep[labels].astype(np.uint8)


def build_text_zone_mask(
    image_shape: Tuple[int, int, int],
    text_objects: List[TextObject],
    margin_px: int,
) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for txt in text_objects:
        b = txt.bbox.expand(margin_px, w, h)
        mask[b.y0:b.y1, b.x0:b.x1] = 255

    return mask


# ============================================================
# 6. COMPONENTS
# ============================================================

def classify_component(
    box: Box,
    pixel_area: int,
    text_overlap_ratio: float,
    cfg: DetectorConfig,
) -> str:
    if pixel_area < cfg.min_component_pixels:
        return "noise"

    if text_overlap_ratio >= cfg.text_ink_overlap_ratio:
        looks_like_large_geometry = (
            box.width() >= cfg.geometry_override_min_width_px
            or box.height() >= cfg.geometry_override_min_height_px
            or pixel_area >= cfg.min_geometry_pixels * cfg.geometry_override_min_pixels_multiplier
        )

        if looks_like_large_geometry:
            return "geometry"

        return "text_ink"

    if pixel_area >= cfg.min_geometry_pixels:
        return "geometry"

    return "noise"


def _debug_write(
    cfg: DetectorConfig,
    debug_dir: Optional[Path],
    name: str,
    img: np.ndarray,
) -> None:
    if cfg.save_debug and debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / name), img)


def extract_ink_components(
    ink_mask: np.ndarray,
    text_zone_mask: np.ndarray,
    image_shape: Tuple[int, int, int],
    cfg: DetectorConfig,
    debug_dir: Optional[Path] = None,
    image_bgr: Optional[np.ndarray] = None,
) -> List[InkComponent]:
    h, w = ink_mask.shape[:2]
    page_area = h * w

    mask = remove_border_connected_components(ink_mask)
    # _debug_write(cfg, debug_dir, "03a_mask_no_border_components.png", mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    thick = cv2.dilate(mask, kernel, iterations=1)
    # _debug_write(cfg, debug_dir, "03b_thick_ink.png", thick)

    closed = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, kernel, iterations=1)
    # _debug_write(cfg, debug_dir, "03c_closed_ink.png", closed)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    components: List[InkComponent] = []
    debug_img = image_bgr.copy() if (cfg.save_debug and image_bgr is not None) else None

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        pixel_area = int(stats[label_id, cv2.CC_STAT_AREA])

        box = Box(x, y, x + bw, y + bh)

        if box.area() > page_area * cfg.max_component_area_ratio:
            continue

        component_mask = labels == label_id
        text_overlap_pixels = int(np.sum(text_zone_mask[component_mask] > 0))
        text_overlap_ratio = text_overlap_pixels / max(1, pixel_area)

        kind = classify_component(box, pixel_area, text_overlap_ratio, cfg)

        fill_ratio = pixel_area / max(1, box.area())

        components.append(
            InkComponent(
                bbox=box,
                kind=kind,
                pixel_area=pixel_area,
                text_overlap_ratio=text_overlap_ratio,
                fill_ratio=fill_ratio,
            )
        )

        if debug_img is not None:
            color = {
                "geometry": (0, 180, 0),
                "text_ink": (255, 0, 0),
                "noise": (120, 120, 120),
            }.get(kind, (255, 255, 255))

            cv2.rectangle(debug_img, (box.x0, box.y0), (box.x1, box.y1), color, 1)

    # if debug_img is not None:
    #     _debug_write(cfg, debug_dir, "03d_classified_components.png", debug_img)

    return components


def classify_geometry_role(
    comp: InkComponent,
    cfg: DetectorConfig,
) -> str:
    """
    Klasyfikuje komponent geometrii jako:
    - core  -> potencjalny rdzeń detalu
    - small -> tekst jako grafika / opis / wymiar / symbol / drobny fragment

    Nie używa samego percentyla. Bierze pod uwagę:
    - pole bbox,
    - proporcje bbox,
    - fill ratio,
    - overlap ze strefą tekstu PDF.
    """
    box = comp.bbox
    area = box.area()
    w = box.width()
    h = box.height()

    if w <= 0 or h <= 0:
        return "small"

    aspect = max(w, h) / max(1, min(w, h))
    fill = comp.fill_ratio
    text_overlap = comp.text_overlap_ratio

    # 1. Jeśli komponent pokrywa się z tekstem PDF, traktujemy go jako mały.
    # To chroni przed uznaniem opisów za rdzenie detali.
    if text_overlap > 0.30:
        return "small"

    # 2. Bardzo długie i rzadkie bboxy to zwykle opis, wymiar albo odnośnik.
    if aspect > 6 and fill < 0.25:
        return "small"

    # 3. Mały, ale gęsty element może być realną geometrią,
    # ale nie powinien sam tworzyć dużego detalu.
    if area < cfg.min_detail_area_px * 0.50 and fill > 0.35:
        return "small"

    # 4. Duży, sensownie wypełniony obszar to rdzeń detalu.
    if area >= cfg.min_detail_area_px and fill > 0.12:
        return "core"

    # 5. Większy, ale rzadki obszar może być kreskowaniem / przekrojem.
    if area >= cfg.min_detail_area_px * 0.75 and fill > 0.06 and aspect <= 6:
        return "core"

    return "small"

# ============================================================
# 7. GEOMETRY CLUSTERING
# ============================================================

def cluster_components_by_touching(
    components: List[InkComponent],
    tolerance_px: int,
) -> List[List[InkComponent]]:
    """
    Stabilna wersja O(n²).

    Celowo rezygnujemy tutaj ze spatial grid, bo poprzednia wersja indeksowała
    komponent tylko po x0/y0 i mogła nie porównywać dużych, sąsiadujących bboxów.
    Przy typowej liczbie komponentów dla strony technicznej ta wersja jest
    bezpieczniejsza diagnostycznie.
    """
    n = len(components)

    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if boxes_touch_or_intersect(components[i].bbox, components[j].bbox, tolerance_px):
                union(i, j)

    groups: dict[int, List[InkComponent]] = defaultdict(list)

    for i, comp in enumerate(components):
        groups[find(i)].append(comp)

    return list(groups.values())


def absorb_inner_and_overlapping_boxes(
    boxes: List[Box],
    contain_margin_px: int = 8,
    overlap_ratio_threshold: float = 0.25,
) -> List[Box]:
    n = len(boxes)

    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            inter = intersection_area(boxes[i], boxes[j])
            smaller_area = min(boxes[i].area(), boxes[j].area())
            overlap_ratio = inter / max(1, smaller_area)

            should_merge = (
                boxes[i].contains(boxes[j], margin=contain_margin_px)
                or boxes[j].contains(boxes[i], margin=contain_margin_px)
                or overlap_ratio >= overlap_ratio_threshold
            )

            if should_merge:
                union(i, j)

    groups: dict[int, List[Box]] = defaultdict(list)

    for i, box in enumerate(boxes):
        groups[find(i)].append(box)

    merged: List[Box] = []

    for group in groups.values():
        u = union_boxes(group)
        if u is not None:
            merged.append(u)

    return merged


# ============================================================
# 8. TEXT ATTACHMENT
# ============================================================

def old_should_attach_text_to_geometry(
    text_box: Box,
    geometry_box: Box,
    image_shape: Tuple[int, int, int],
    cfg: DetectorConfig,
) -> bool:
    h, w = image_shape[:2]

    if cfg.attach_inside_text and contains_inside(
        geometry_box,
        text_box,
        inset=cfg.attach_inside_text_inset_px,
    ):
        return True

    if boxes_touch_or_intersect(
        geometry_box,
        text_box,
        tolerance_px=cfg.text_geometry_touch_px,
    ):
        return True

    left_zone = Box(
        max(0, geometry_box.x0 - cfg.side_text_search_px),
        max(0, geometry_box.y0 - cfg.side_text_vertical_slack_px),
        geometry_box.x0,
        min(h, geometry_box.y1 + cfg.side_text_vertical_slack_px),
    )

    right_zone = Box(
        geometry_box.x1,
        max(0, geometry_box.y0 - cfg.side_text_vertical_slack_px),
        min(w, geometry_box.x1 + cfg.side_text_search_px),
        min(h, geometry_box.y1 + cfg.side_text_vertical_slack_px),
    )

    top_zone = Box(
        max(0, geometry_box.x0 - cfg.top_bottom_text_horizontal_slack_px),
        max(0, geometry_box.y0 - cfg.top_bottom_text_search_px),
        min(w, geometry_box.x1 + cfg.top_bottom_text_horizontal_slack_px),
        geometry_box.y0,
    )

    bottom_zone = Box(
        max(0, geometry_box.x0 - cfg.top_bottom_text_horizontal_slack_px),
        geometry_box.y1,
        min(w, geometry_box.x1 + cfg.top_bottom_text_horizontal_slack_px),
        min(h, geometry_box.y1 + cfg.top_bottom_text_search_px),
    )

    return (
        text_box.intersects(left_zone)
        or text_box.intersects(right_zone)
        or text_box.intersects(top_zone)
        or text_box.intersects(bottom_zone)
    )

def should_attach_text_to_geometry(
    text_box: Box,
    geometry_box: Box,
    all_text_boxes: List[Box],
    cfg: DetectorConfig,
) -> bool:
    """
    Tekst dołącz tylko jeśli:
    1. jest w środku geometrii
    2. dotyka geometrii
    3. jest najbliższym tekstem bocznym
    """

    # tekst wewnątrz geometrii
    if cfg.attach_inside_text and geometry_box.contains(
        text_box,
        margin=-cfg.attach_inside_text_inset_px,
    ):
        return True

    # tekst styczny
    if boxes_touch_or_intersect(
        geometry_box,
        text_box,
        tolerance_px=cfg.text_geometry_touch_px,
    ):
        return True

    # nie dołączaj tylko dlatego, że jest "w strefie"
    return False

def attach_closest_side_texts(
    core_box: Box,
    texts: List[TextObject],
    image_shape: Tuple[int, int, int],
    max_side_distance_px: int = 80,
    max_vertical_slack_px: int = 120,
    max_texts_per_side: int = 8,
) -> List[TextObject]:
    """
    Dołącza najbliższe teksty boczne po lewej i prawej stronie core_box.

    Ważne:
    - nie bierze wszystkich tekstów ze strefy bocznej,
    - bierze tylko najbliższe,
    - osobno dla lewej i prawej strony.
    """

    left: List[tuple[int, TextObject]] = []
    right: List[tuple[int, TextObject]] = []

    core_y0 = core_box.y0 - max_vertical_slack_px
    core_y1 = core_box.y1 + max_vertical_slack_px

    for txt in texts:
        tb = txt.bbox
        text_cy = (tb.y0 + tb.y1) / 2

        if text_cy < core_y0 or text_cy > core_y1:
            continue

        # tekst po lewej stronie geometrii
        if tb.x1 <= core_box.x0:
            dx = core_box.x0 - tb.x1
            if dx <= max_side_distance_px:
                left.append((dx, txt))

        # tekst po prawej stronie geometrii
        elif tb.x0 >= core_box.x1:
            dx = tb.x0 - core_box.x1
            if dx <= max_side_distance_px:
                right.append((dx, txt))

    left.sort(key=lambda x: x[0])
    right.sort(key=lambda x: x[0])

    selected = [txt for _, txt in left[:max_texts_per_side]]
    selected.extend(txt for _, txt in right[:max_texts_per_side])

    return selected

# ============================================================
# 9. CANDIDATES
# ============================================================

def old_build_detail_candidates(
    text_objects: List[TextObject],
    components: List[InkComponent],
    image_shape: Tuple[int, int, int],
    cfg: DetectorConfig,
) -> List[DetailCandidate]:
    """
    Buduje kandydatów bbox detali na podstawie 04a_geometry_only.

    Nowa logika:
    - nie zakładamy, że tekst zawsze jest dostępny z PDF,
    - tekst jako grafika jest traktowany jak mały bbox geometryczny,
    - duże bboxy są rdzeniami detali,
    - małe bboxy są dołączane do najbliższych / pokrywających się rdzeni.
    """
    h, w = image_shape[:2]
    page_area = h * w

    # 1. Bierzemy wyłącznie geometry z 04a.
    geometry_boxes = [
        c.bbox for c in components
        if c.kind == "geometry"
        and c.bbox.area() < page_area * cfg.max_detail_area_ratio
    ]

    if not geometry_boxes:
        return []

    # 2. Dzielimy geometrię według relatywnej powierzchni:
    #    duże = rdzenie detali, małe = opisy/wymiary/symbole jako grafika.
    large_boxes, small_boxes = split_geometry_boxes_by_relative_area(
        geometry_boxes,
        large_percentile=75.0,
        min_large_area_px=max(cfg.min_detail_area_px, 10_000),
    )

    # 3. Duże rdzenie mogą się jeszcze lekko scalać, jeśli realnie się dotykają.
    large_components = [
        InkComponent(bbox=b, kind="geometry", pixel_area=b.area())
        for b in large_boxes
    ]

    large_clusters = cluster_components_by_touching(
        large_components,
        tolerance_px=cfg.geometry_touch_px,
    )

    core_boxes: List[Box] = []

    for cluster in large_clusters:
        core = union_boxes([c.bbox for c in cluster])

        if core is None:
            continue

        if core.area() < cfg.min_detail_area_px * 0.50:
            continue

        if core.area() > page_area * cfg.max_detail_area_ratio:
            continue

        core_boxes.append(core)

    # 4. Rdzenie wchłaniają małe bboxy: tekst jako grafika, opisy, wymiary.
    expanded_core_boxes = absorb_related_small_boxes_into_cores(
        core_boxes=core_boxes,
        small_boxes=small_boxes,
        image_shape=image_shape,
        cfg=cfg,
    )

    candidates: List[DetailCandidate] = []

    for core in expanded_core_boxes:
        # 5. Jeśli tekst PDF istnieje, nadal go używamy jako dodatkowe źródło.
        assigned_texts = [
            txt for txt in text_objects
            if should_attach_text_to_geometry(txt.bbox, core, image_shape, cfg)
        ]

        final = union_boxes([core] + [t.bbox for t in assigned_texts])

        if final is None:
            continue

        final = final.expand(cfg.output_margin_px, w, h)

        if final.width() < cfg.min_detail_width_px:
            continue

        if final.height() < cfg.min_detail_height_px:
            continue

        if final.area() < cfg.min_detail_area_px:
            continue

        if final.area() > page_area * cfg.max_detail_area_ratio:
            continue

        candidates.append(
            DetailCandidate(
                core_box=core,
                final_box=final,
                texts=assigned_texts,
            )
        )

    return dedupe_candidates(candidates)

def build_detail_candidates(
    text_objects: List[TextObject],
    components: List[InkComponent],
    image_shape: Tuple[int, int, int],
    cfg: DetectorConfig,
) -> List[DetailCandidate]:
    """
    Buduje kandydatów bbox detali bez percentyla.

    Logika:
    - komponenty geometryczne dzielimy na core/small po cechach,
    - core = rdzeń detalu,
    - small = opisy jako grafika, symbole, wymiary, drobne elementy,
    - small są dołączane do core tylko względem pierwotnego core,
      żeby bbox nie rósł lawinowo,
    - tekst PDF/OCR jest dołączany ostrożnie:
      * tekst wewnątrz / styczny,
      * najbliższe teksty boczne z limitem liczby.
    """
    h, w = image_shape[:2]
    page_area = h * w

    geometry_components = [
        c for c in components
        if c.kind == "geometry"
        and c.bbox.area() < page_area * cfg.max_detail_area_ratio
    ]

    if not geometry_components:
        return []

    core_components: List[InkComponent] = []
    small_boxes: List[Box] = []

    for comp in geometry_components:
        role = classify_geometry_role(comp, cfg)

        if role == "core":
            core_components.append(comp)
        else:
            small_boxes.append(comp.bbox)

    if not core_components:
        sorted_components = sorted(
            geometry_components,
            key=lambda c: c.bbox.area(),
            reverse=True,
        )

        fallback_count = max(1, min(5, len(sorted_components) // 4))

        core_components = sorted_components[:fallback_count]
        small_boxes = [c.bbox for c in sorted_components[fallback_count:]]

    core_clusters = cluster_components_by_touching(
        core_components,
        tolerance_px=cfg.geometry_touch_px,
    )

    core_boxes: List[Box] = []

    for cluster in core_clusters:
        core = union_boxes([c.bbox for c in cluster])

        if core is None:
            continue

        if core.area() < cfg.min_detail_area_px * 0.50:
            continue

        if core.area() > page_area * cfg.max_detail_area_ratio:
            continue

        core_boxes.append(core)

    # Delikatne scalenie rdzeni zagnieżdżonych / mocno nakładających się.
    # Nie zastępuje to absorpcji small_boxes.
    core_boxes = absorb_inner_and_overlapping_boxes(
        core_boxes,
        contain_margin_px=cfg.absorb_contain_margin_px,
        overlap_ratio_threshold=cfg.absorb_overlap_ratio_threshold,
    )

    expanded_core_boxes: List[Box] = []

    for core in core_boxes:
        related_boxes: List[Box] = [core]

        for small in small_boxes:
            # Kluczowa zmiana:
            # relację sprawdzamy względem pierwotnego core,
            # nie względem rosnącego merged bboxa.
            if is_related_small_box_to_core(
                small=small,
                core=core,
                image_shape=image_shape,
                contain_margin_px=cfg.absorb_contain_margin_px,
                overlap_ratio_threshold=cfg.absorb_overlap_ratio_threshold,
                touch_px=cfg.text_geometry_touch_px,
                side_search_px=cfg.side_text_search_px,
                vertical_slack_px=cfg.side_text_vertical_slack_px,
            ):
                related_boxes.append(small)

        merged = union_boxes(related_boxes)

        if merged is None:
            continue

        if merged.area() > page_area * cfg.max_detail_area_ratio:
            continue

        expanded_core_boxes.append(merged)

    candidates: List[DetailCandidate] = []
    all_text_boxes = [txt.bbox for txt in text_objects]

    for core in expanded_core_boxes:
        touching_texts = [
            txt for txt in text_objects
            if should_attach_text_to_geometry(
                text_box=txt.bbox,
                geometry_box=core,
                all_text_boxes=all_text_boxes,
                cfg=cfg,
            )
        ]

        side_texts = attach_closest_side_texts(
            core_box=core,
            texts=text_objects,
            image_shape=image_shape,
            max_side_distance_px=min(cfg.side_text_search_px, 45),
            max_vertical_slack_px=min(cfg.side_text_vertical_slack_px, 25),
            max_texts_per_side=4,
        )

        assigned_texts = unique_text_objects(touching_texts + side_texts)

        final = union_boxes([core] + [t.bbox for t in assigned_texts])

        if final is None:
            continue

        final = final.expand(cfg.output_margin_px, w, h)

        if final.width() < cfg.min_detail_width_px:
            continue

        if final.height() < cfg.min_detail_height_px:
            continue

        if final.area() < cfg.min_detail_area_px:
            continue

        if final.area() > page_area * cfg.max_detail_area_ratio:
            continue

        candidates.append(
            DetailCandidate(
                core_box=core,
                final_box=final,
                texts=assigned_texts,
            )
        )

    return dedupe_candidates(
        candidates,
        smaller_overlap_threshold=cfg.dedupe_smaller_overlap_threshold,
    )


def split_geometry_boxes_by_relative_area(
    boxes: List[Box],
    large_percentile: float = 75.0,
    min_large_area_px: int = 15_000,
) -> tuple[List[Box], List[Box]]:
    """
    Dzieli zielone bboxy z 04a na:
    - duże rdzenie detali,
    - mniejsze elementy pomocnicze: opisy jako grafika, wymiary, symbole, małe fragmenty.

    Działa również wtedy, gdy tekst nie jest dostępny z PDF i występuje jako grafika.
    """
    if not boxes:
        return [], []

    areas = np.array([b.area() for b in boxes], dtype=np.float64)
    threshold = max(float(np.percentile(areas, large_percentile)), float(min_large_area_px))

    large = [b for b in boxes if b.area() >= threshold]
    small = [b for b in boxes if b.area() < threshold]

    return large, small

def is_related_small_box_to_core(
    small: Box,
    core: Box,
    image_shape: Tuple[int, int, int],
    contain_margin_px: int = 8,
    overlap_ratio_threshold: float = 0.25,
    touch_px: int = 12,
    side_search_px: int = 120,
    vertical_slack_px: int = 60,
) -> bool:
    """
    Sprawdza, czy mały bbox należy do dużego rdzenia detalu.

    Warunki:
    - jest wewnątrz core,
    - pokrywa się z core,
    - dotyka core po lekkim offsetcie,
    - jest blisko z lewej/prawej strony core.
    """
    h, w = image_shape[:2]

    if core.contains(small, margin=contain_margin_px):
        return True

    inter = intersection_area(core, small)
    overlap_ratio = inter / max(1, small.area())

    if overlap_ratio >= overlap_ratio_threshold:
        return True

    if boxes_touch_or_intersect(core, small, tolerance_px=touch_px):
        return True

    small_cy = (small.y0 + small.y1) / 2

    if small_cy < core.y0 - vertical_slack_px:
        return False

    if small_cy > core.y1 + vertical_slack_px:
        return False

    # Mały box po lewej stronie core
    if small.x1 <= core.x0:
        dx = core.x0 - small.x1
        return dx <= side_search_px

    # Mały box po prawej stronie core
    if small.x0 >= core.x1:
        dx = small.x0 - core.x1
        return dx <= side_search_px

    return False


def absorb_related_small_boxes_into_cores(
    core_boxes: List[Box],
    small_boxes: List[Box],
    image_shape: Tuple[int, int, int],
    cfg: DetectorConfig,
) -> List[Box]:

    result: List[Box] = []

    for core in core_boxes:
        related: List[Box] = [core]

        for small in small_boxes:
            if is_related_small_box_to_core(
                small=small,
                core=core,  # ważne: oryginalny core, nie merged
                image_shape=image_shape,
                contain_margin_px=cfg.absorb_contain_margin_px,
                overlap_ratio_threshold=cfg.absorb_overlap_ratio_threshold,
                touch_px=cfg.text_geometry_touch_px,
                side_search_px=cfg.side_text_search_px,
                vertical_slack_px=cfg.side_text_vertical_slack_px,
            ):
                related.append(small)

        merged = union_boxes(related)
        if merged is not None:
            result.append(merged)

    return result
    

def unique_text_objects(texts: List[TextObject]) -> List[TextObject]:
    seen = set()
    result: List[TextObject] = []

    for txt in texts:
        key = (
            txt.bbox.x0,
            txt.bbox.y0,
            txt.bbox.x1,
            txt.bbox.y1,
            txt.text,
        )

        if key not in seen:
            seen.add(key)
            result.append(txt)

    return result



def dedupe_candidates(
    candidates: List[DetailCandidate],
    smaller_overlap_threshold: float = 0.80,
) -> List[DetailCandidate]:
    """
    Usuwa kandydaty zagnieżdżone / prawie pokrywające się.

    Nie używamy samego IoU, bo mały box w dużym może mieć niski IoU,
    mimo że jest w praktyce duplikatem.
    """
    candidates = sorted(
        candidates,
        key=lambda c: c.final_box.area(),
        reverse=True,
    )

    result: List[DetailCandidate] = []

    for cand in candidates:
        keep = True

        for kept in result:
            inter = intersection_area(cand.final_box, kept.final_box)
            smaller = min(cand.final_box.area(), kept.final_box.area())
            overlap_ratio = inter / max(1, smaller)

            if overlap_ratio >= smaller_overlap_threshold:
                keep = False
                break

        if keep:
            result.append(cand)

    return result


# ============================================================
# 10. PDF ANNOTATIONS
# ============================================================

def add_candidate_annotations_to_pdf(
    input_pdf_path: Path,
    output_pdf_path: Path,
    candidates_by_page: dict[int, List[DetailCandidate]],
    dpi: int,
    color_rgb: tuple[float, float, float] = (0.0, 85 / 255, 0.0),
    opacity: float = 0.5,
) -> None:
    scale = dpi / 72
    doc = fitz.open(input_pdf_path)

    for page_number, candidates in candidates_by_page.items():
        page_index = page_number - 1

        if page_index < 0 or page_index >= len(doc):
            print(f"[WARNING] Skipping invalid PDF page: {page_number}")
            continue

        page = doc[page_index]

        for idx, cand in enumerate(candidates, start=1):
            b = cand.final_box
            rect = fitz.Rect(
                b.x0 / scale,
                b.y0 / scale,
                b.x1 / scale,
                b.y1 / scale,
            )

            if rect.is_empty or rect.is_infinite:
                print(f"[WARNING] Skipping invalid annotation rect on page {page_number}, candidate {idx}")
                continue
            

            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=color_rgb, fill=color_rgb)
            annot.set_opacity(opacity)
            annot.set_border(width=1.5)
            annot.set_info(
                content=f"Detail candidate {idx}",
                title="PDF Detail Detector",
            )
            annot.update()

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_pdf_path, garbage=4, deflate=True)
    doc.close()

    print(f"[INFO] Saved annotated PDF: {output_pdf_path}")


# ============================================================
# 11. DEBUG / OUTPUT
# ============================================================

def draw_page_summary_debug(
    image: np.ndarray,
    text_objects: List[TextObject],
    components: List[InkComponent],
    candidates: List[DetailCandidate],
    path: Path,
) -> None:
    """
    Jeden PNG na stronę:
    - tekst PDF: niebieski
    - geometria: zielony
    - finalni kandydaci: czerwony
    """
    out = image.copy()

    # Tekst — niebieski
    for txt in text_objects:
        b = txt.bbox
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), (255, 0, 0), 1)

    # Geometria — zielony
    for comp in components:
        if comp.kind != "geometry":
            continue
        b = comp.bbox
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), (0, 180, 0), 1)

    # Final candidates — czerwony
    for idx, cand in enumerate(candidates, start=1):
        b = cand.final_box
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), (0, 0, 255), 3)
        cv2.putText(
            out,
            f"D{idx}",
            (b.x0 + 8, b.y0 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)
    

def draw_texts(
    image: np.ndarray,
    texts: List[TextObject],
    path: Path,
) -> None:
    out = image.copy()

    for txt in texts:
        b = txt.bbox
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), (255, 0, 0), 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def draw_components(
    image: np.ndarray,
    components: List[InkComponent],
    path: Path,
    kinds: Optional[set[str]] = None,
) -> None:
    out = image.copy()

    color_map = {
        "geometry": (0, 180, 0),
        "text_ink": (255, 0, 0),
        "noise": (120, 120, 120),
    }

    for comp in components:
        if kinds is not None and comp.kind not in kinds:
            continue

        b = comp.bbox
        cv2.rectangle(
            out,
            (b.x0, b.y0),
            (b.x1, b.y1),
            color_map.get(comp.kind, (255, 255, 255)),
            1,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def draw_candidates(
    image: np.ndarray,
    candidates: List[DetailCandidate],
    path: Path,
) -> None:
    out = image.copy()

    for idx, cand in enumerate(candidates, start=1):
        c = cand.core_box
        f = cand.final_box

        cv2.rectangle(out, (c.x0, c.y0), (c.x1, c.y1), (0, 180, 0), 2)

        for txt in cand.texts:
            b = txt.bbox
            cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), (255, 0, 0), 1)

        cv2.rectangle(out, (f.x0, f.y0), (f.x1, f.y1), (0, 0, 255), 3)

        cv2.putText(
            out,
            f"D{idx}",
            (f.x0 + 8, f.y0 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def export_crops(
    image_bgr: np.ndarray,
    candidates: List[DetailCandidate],
    output_dir: Path,
    page_number: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    h, w = image_bgr.shape[:2]

    for idx, cand in enumerate(candidates, start=1):
        b = cand.final_box

        x0 = max(0, min(w, b.x0))
        y0 = max(0, min(h, b.y0))
        x1 = max(0, min(w, b.x1))
        y1 = max(0, min(h, b.y1))

        if x1 <= x0 or y1 <= y0:
            print(f"[WARNING] Empty crop skipped: page={page_number}, detail={idx}")
            continue

        crop = image_bgr[y0:y1, x0:x1]

        if crop.size == 0:
            print(f"[WARNING] Empty crop array skipped: page={page_number}, detail={idx}")
            continue

        cv2.imwrite(
            str(output_dir / f"page_{page_number:03d}_detail_{idx:03d}.png"),
            crop,
        )


# ============================================================
# 12. PAGE PIPELINE
# ============================================================

def detect_details_on_page(
    page: fitz.Page,
    page_number: int,
    output_dir: Path,
    cfg: DetectorConfig,
) -> List[DetailCandidate]:
    scale = cfg.dpi / 72

    debug_dir: Optional[Path] = None

    if cfg.save_debug:
        debug_dir = output_dir / "debug" / f"page_{page_number:03d}"
        debug_dir_solo = output_dir
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"[page {page_number}] render")
    image = render_page_to_image(page, cfg.dpi)
    # _debug_write(cfg, debug_dir, "00_render.png", image)

    print(f"[page {page_number}] PDF text")
    # text_objects = extract_text_objects(page, scale, cfg)
    text_objects = extract_text_objects_with_fallback(
        page=page,
        image_bgr=image,
        scale=scale,
        cfg=cfg,
    )

    # if cfg.save_debug and debug_dir:
        # draw_texts(image, text_objects, debug_dir / "01_pdf_text_boxes.png")

    print(f"[page {page_number}] ink mask")
    ink_mask_raw = make_ink_mask(image, cfg)
    # _debug_write(cfg, debug_dir, "02_ink_mask_raw.png", ink_mask_raw)

    ink_mask = remove_border_margin(ink_mask_raw, margin_px=8)
    # _debug_write(cfg, debug_dir, "02b_ink_mask_no_margin.png", ink_mask)

    print(f"[page {page_number}] text zone mask")
    text_zone_mask = build_text_zone_mask(
        image.shape,
        text_objects,
        cfg.text_zone_margin_px,
    )
    # _debug_write(cfg, debug_dir, "03_text_zone_mask.png", text_zone_mask)

    print(f"[page {page_number}] components")
    components = extract_ink_components(
        ink_mask=ink_mask,
        text_zone_mask=text_zone_mask,
        image_shape=image.shape,
        cfg=cfg,
        debug_dir=debug_dir,
        image_bgr=image,
    )

    # if cfg.save_debug and debug_dir:
    #     draw_components(
    #         image,
    #         components,
    #         debug_dir / "04_all_classified_components.png",
    #     )
    #     draw_components(
    #         image,
    #         components,
    #         debug_dir / "04a_geometry_only.png",
    #         kinds={"geometry"},
    #     )

    print(f"[page {page_number}] candidates")
    candidates = build_detail_candidates(
        text_objects,
        components,
        image.shape,
        cfg,
    )

    candidates = dedupe_candidates(
        candidates,
        smaller_overlap_threshold=cfg.dedupe_smaller_overlap_threshold,
    )

    if cfg.save_debug:
        draw_page_summary_debug(
            image=image,
            text_objects=text_objects,
            components=components,
            candidates=candidates,
            path=debug_dir_solo / f"page_{page_number:03d}_summary.png",
        )

    # if cfg.save_debug and debug_dir:
    #     draw_candidates(
    #         image,
    #         candidates,
    #         debug_dir / "05_final_detail_boxes.png",
    #     )

    if cfg.export_crops:
        export_crops(
            image,
            candidates,
            output_dir / "crops",
            page_number,
        )

    print(f"[page {page_number}] done — {len(candidates)} candidate(s)")

    return candidates


# ============================================================
# 13. PDF PIPELINE
# ============================================================

def parse_pages_arg(
    pages: Optional[str | Iterable[int]],
    total_pages: int,
) -> List[int]:
    if pages is None:
        return list(range(1, total_pages + 1))

    if isinstance(pages, str):
        page_numbers = [
            int(x.strip())
            for x in pages.split(",")
            if x.strip()
        ]
    else:
        page_numbers = list(pages)

    valid: List[int] = []

    for p in page_numbers:
        if 1 <= p <= total_pages:
            valid.append(p)
        else:
            print(f"[WARNING] Skipping invalid page number: {p}")

    return valid


def detect_details_in_pdf(
    pdf_path: Path,
    output_dir: Path,
    cfg: Optional[DetectorConfig] = None,
    pages: Optional[str | Iterable[int]] = None,
) -> dict[int, List[DetailCandidate]]:
    cfg = validate_config(cfg or DetectorConfig())

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    page_numbers = parse_pages_arg(pages, total_pages)

    if not page_numbers:
        print("[WARNING] No valid pages to analyse.")
        doc.close()
        return {}

    print(f"[INFO] Pages to analyse: {page_numbers}")

    results: dict[int, List[DetailCandidate]] = {}

    for page_number in page_numbers:
        print(f"[INFO] Page {page_number}/{total_pages}")

        try:
            results[page_number] = detect_details_on_page(
                page=doc[page_number - 1],
                page_number=page_number,
                output_dir=output_dir,
                cfg=cfg,
            )

        except Exception as exc:
            print(f"[ERROR] Page {page_number}: {exc}")
            traceback.print_exc()
            results[page_number] = []

    doc.close()

    if cfg.write_pdf_annotations:
        annotated_path = output_dir / "annotated.pdf"

        add_candidate_annotations_to_pdf(
            input_pdf_path=pdf_path,
            output_pdf_path=annotated_path,
            candidates_by_page=results,
            dpi=cfg.dpi,
            color_rgb=(0.0, 85 / 255, 0.0),
            opacity=cfg.annotation_opacity,
        )

    return results


# ============================================================
# 14. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    pdf_path = r"C:\Users\przemyslaw.stepien\OneDrive - PM Group\800. PYTHON\PDF Detail Crop\data\Zbiorczy katalog Detali 12-2021_clean.pdf"
    output_dir = Path("PDFCropperOutput_geometry_relative")

    config = DetectorConfig(
        dpi=150,

        # min_component_pixels=20,
        # min_geometry_pixels=40,
        # max_component_area_ratio=0.90,
        # text_ink_overlap_ratio=0.80,

        # geometry_override_min_width_px=150,
        # geometry_override_min_height_px=80,
        # geometry_override_min_pixels_multiplier=5,

        # adaptive_threshold_block_size=31,
        # adaptive_threshold_c=12,

        # geometry_touch_px=4,
        # text_geometry_touch_px=8,

        # side_text_search_px=160,
        # side_text_vertical_slack_px=60,
        # top_bottom_text_search_px=80,
        # top_bottom_text_horizontal_slack_px=60,
        # attach_inside_text=True,
        # attach_inside_text_inset_px=2,

        # output_margin_px=15,
        # min_detail_width_px=120,
        # min_detail_height_px=120,
        # min_detail_area_px=15_000,
        # max_detail_area_ratio=0.70,

        # absorb_contain_margin_px=8,
        # absorb_overlap_ratio_threshold=0.5,

        # dedupe_smaller_overlap_threshold=0.80,

        # save_debug=True,
        # export_crops=True,

        # write_pdf_annotations=True,
        # annotation_opacity=0.5,

        # Minimalna liczba pikseli w komponencie atramentu.
        # Mniejsze komponenty są traktowane jako noise.
        min_component_pixels=20,

        # Minimalna liczba pikseli aby komponent został uznany za geometrię.
        # Za nisko = dużo śmieci jako geometria.
        min_geometry_pixels=40,

        # Maksymalny udział powierzchni strony jaki może mieć pojedynczy komponent.
        # Chroni przed uznaniem całej strony / ramki za geometrię.
        max_component_area_ratio=0.80,

        # KLUCZOWE: jeśli >80% atramentu komponentu leży w strefie tekstu PDF,
        # traktuj jako tekst, nie geometrię.
        # Za nisko = litery stają się geometrią.
        # Za wysoko = tekst może przejść do geometrii.
        text_ink_overlap_ratio=0.80,


        # Minimalna szerokość bbox, przy której komponent może zostać
        # wymuszony jako geometria mimo małej liczby pikseli.
        geometry_override_min_width_px=150,

        # Minimalna wysokość bbox dla override geometrii.
        geometry_override_min_height_px=80,

        # Mnożnik pozwalający przepuścić "rzadką" geometrię
        # (np. kreskowanie, cienkie linie).
        geometry_override_min_pixels_multiplier=5,


        # KLUCZOWE: rozmiar okna adaptive threshold.
        # Większe = bardziej globalne progowanie.
        # Mniejsze = więcej lokalnych detali, ale też więcej szumu.
        adaptive_threshold_block_size=31,

        # KLUCZOWE: offset adaptive threshold.
        # Większe = mniej atramentu.
        # Mniejsze = więcej atramentu / grubsze wykrycie.
        adaptive_threshold_c=12,


        # KLUCZOWE: odległość (px), przy której geometria skleja się
        # z inną geometrią w klaster.
        # Za duże = dwa detale mogą stać się jednym.
        # Za małe = jeden detal rozpadnie się na kilka klastrów.
        geometry_touch_px=2,

        # KLUCZOWE: offset styku tekstu z geometrią.
        # Jeśli bbox tekstu jest w tej odległości od geometrii,
        # tekst zostanie dołączony do detalu.
        # Za duże = łapie obce opisy.
        # Za małe = gubi poprawne opisy.
        text_geometry_touch_px=3,


        # KLUCZOWE: maksymalna odległość tekstu po lewej/prawej stronie geometrii.
        # Główna reguła dla opisów na odnośnikach.
        # Za duże = tekst z sąsiednich detali.
        side_text_search_px=45,

        # KLUCZOWE: pionowa tolerancja dla tekstów bocznych.
        # Pozwala tekstowi być trochę wyżej/niżej niż geometria.
        # Za duże = zaczyna łapać obce opisy.
        side_text_vertical_slack_px=25,

        # Maksymalna odległość tekstu nad/pod geometrią.
        # Mniejszy priorytet niż tekst boczny.
        top_bottom_text_search_px=80,

        # Pozioma tolerancja dla tekstów nad/pod geometrią.
        top_bottom_text_horizontal_slack_px=60,

        # Czy tekst całkowicie wewnątrz geometrii ma być dołączony.
        # Przydatne np. opisy w środku detalu.
        attach_inside_text=True,

        # Jak głęboko tekst musi być wewnątrz geometrii
        # (ujemny offset containment).
        attach_inside_text_inset_px=2,


        # KLUCZOWE: margines dodawany do finalnego bbox.
        # Tylko kosmetyczny, ale za duży daje wrażenie złego scalenia.
        output_margin_px=10,

        # Minimalna szerokość finalnego detalu.
        # Mniejsze bbox są odrzucane.
        min_detail_width_px=120,

        # Minimalna wysokość finalnego detalu.
        min_detail_height_px=120,

        # Minimalna powierzchnia finalnego bbox.
        # Chroni przed małymi śmieciami.
        min_detail_area_px=15_000,

        # Maksymalna powierzchnia finalnego bbox jako % strony.
        # Chroni przed bbox obejmującym pół strony.
        max_detail_area_ratio=0.70,


        # KLUCZOWE: jeśli mały bbox geometrii jest prawie w środku większego,
        # tolerancja containment.
        # Pozwala dużym obszarom "wchłaniać" mniejsze.
        absorb_contain_margin_px=8,

        # KLUCZOWE: jeśli overlap małego bbox z dużym przekroczy ten próg,
        # mały bbox zostaje wchłonięty.
        # Za nisko = zbyt agresywne scalanie.
        # Za wysoko = zostają boxy wewnętrzne.
        absorb_overlap_ratio_threshold=0.60,


        # KLUCZOWE: jeśli mniejszy finalny bbox pokrywa się z większym
        # w tym stopniu, zostaje usunięty jako duplikat.
        # Za nisko = gubisz poprawne sąsiednie detale.
        # Za wysoko = zostają duplikaty.
        dedupe_smaller_overlap_threshold=0.80,


        # Czy zapisać debug PNG.
        save_debug=True,

        # Czy zapisać cropy finalnych kandydatów.
        export_crops=False,


        # Czy wstawić rectangle annotations do PDF.
        write_pdf_annotations=True,

        # Przezierność annotations PDF.
        # 0.5 = 50%
        annotation_opacity=0.5,

        use_ocr_fallback=True,
        ocr_lang="eng+pol",
        ocr_min_pdf_text_count=3,

        # Windows OCR
        tesseract_cmd=r"C:\Users\przemyslaw.stepien\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        )

    detect_details_in_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        cfg=config
    )
