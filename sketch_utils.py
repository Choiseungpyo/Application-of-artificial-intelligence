# -*- coding: utf-8 -*-
"""
Semantic UI Sketch Generator

Combines OpenCV-based UI region detection with PIL sketch-style rendering
to produce annotated wireframe visualizations of game UI layouts.

Architecture:
  Original Image
    → simplify_to_sketch_style(): PIL threshold_edges (from Auto Encoder approach)
    → extract_ui_regions_opencv(): OpenCV contour + bounding box detection
    → _create_blueprint_canvas(): Light grid background
    → _blend_sketch_onto_canvas(): Overlay sketch lines on grid
    → draw_semantic_overlay(): Colored regions, corner marks, role labels
    → _draw_legend(): Color coding legend
    → Output: Semantic UI Sketch
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
from typing import List, Dict, Tuple, Optional


# ──────────────────────────────────────────────────────────────
#  Role Configuration
# ──────────────────────────────────────────────────────────────

ROLE_CONFIG = {
    # Fine-tuned model grouped roles (highest priority)
    "navigation_quest": {"label": "Map & Quest",       "tag": "MAP", "color": (35, 160, 75)},
    "status_resource":  {"label": "Status & Resource",  "tag": "HUD", "color": (210, 50, 50)},
    "combat_skill":     {"label": "Combat & Skill",     "tag": "SKL", "color": (45, 90, 210)},
    "inventory_shop":   {"label": "Inventory & Shop",   "tag": "INV", "color": (130, 60, 190)},
    "character_select": {"label": "Character Info",     "tag": "CHR", "color": (190, 130, 35)},
    "system_narrative": {"label": "System & Dialog",    "tag": "SYS", "color": (120, 100, 80)},

    # Fallbacks
    "health":        {"label": "Health",    "tag": "HP",  "color": (210, 50, 50)},
    "combat":        {"label": "Skill",     "tag": "SK",  "color": (45, 90, 210)},
    "navigation":    {"label": "Map",       "tag": "MAP", "color": (35, 160, 75)},
    "communication": {"label": "Chat",      "tag": "CH",  "color": (190, 130, 35)},
    "inventory":     {"label": "Inventory", "tag": "INV", "color": (130, 60, 190)},
    "statistics":    {"label": "Stats",     "tag": "ST",  "color": (55, 145, 165)},
    "report":        {"label": "Report",    "tag": "RPT", "color": (155, 105, 65)},
    "quest":         {"label": "Quest",     "tag": "QST", "color": (175, 135, 45)},
    "status":        {"label": "Status",    "tag": "STS", "color": (85, 125, 195)},
    "info":          {"label": "Info",       "tag": "INF", "color": (105, 105, 155)},
    "menu":          {"label": "Menu",       "tag": "MN",  "color": (120, 100, 80)},
    "unknown":       {"label": "Panel",      "tag": "UI",  "color": (95, 95, 105)},
}


# ──────────────────────────────────────────────────────────────
#  Font Loading
# ──────────────────────────────────────────────────────────────

_font_cache = {}

def _load_font(size: int):
    """Load a system font with fallback. Results are cached."""
    if size in _font_cache:
        return _font_cache[size]
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font
        except (OSError, IOError):
            continue
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

def role_to_label_and_icon(role: str) -> Tuple[str, str, Tuple[int, int, int]]:
    """
    Returns (display_label, tag_prefix, accent_color) for a given role.

    Examples:
        role_to_label_and_icon("health")  → ("Health", "HP", (210, 50, 50))
        role_to_label_and_icon("combat")  → ("Skill",  "SK", (45, 90, 210))
    """
    role_lower = role.lower().strip()
    for key, cfg in ROLE_CONFIG.items():
        if key in role_lower:
            return cfg["label"], cfg["tag"], cfg["color"]
    cfg = ROLE_CONFIG["unknown"]
    return cfg["label"], cfg["tag"], cfg["color"]


def simplify_to_sketch_style(image: Image.Image) -> Image.Image:
    """
    Convert an image to clean sketch lines using threshold_edges approach.

    Based on the Auto Encoder project's generate_sketch(mode='threshold_edges').
    Preserves UI structural lines (panel borders, button edges, bars)
    while removing textures, gradients, and background noise.

    Returns a grayscale (mode 'L') image: black lines on white background.
    """
    # Enhance contrast to make UI panel edges pop
    enhanced = ImageEnhance.Contrast(image).enhance(3.5)
    # Sharpen to reinforce thin lines (1px borders, grid lines)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(4.0)

    # Convert to grayscale
    gray = enhanced.convert("L")

    # Slight blur to smooth noise before edge detection
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=0.4))

    # PIL edge detection
    edges = blurred.filter(ImageFilter.FIND_EDGES)

    # Binary threshold: keep only strong edges
    binary = edges.point(lambda p: 255 if p > 35 else 0)

    # Invert → black lines on white
    inverted = ImageOps.invert(binary)

    return inverted


def extract_ui_regions_opencv(
    image: Image.Image,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.85,
    max_regions: int = 15,
) -> List[Dict]:
    """
    Detect UI panel / button / bar regions using OpenCV.

    Multi-strategy detection:
      1. Adaptive threshold → morphological close → external contours
      2. Canny edge → dilation → external contours

    Returns list of region dicts sorted by area (largest first), capped at *max_regions*.
    Each dict contains: bbox, area, area_fraction, center_norm, aspect_ratio, position_label.
    """
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    total_area = w * h
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio
    min_dim = max(15, int(min(w, h) * 0.03))

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    all_rects: List[Tuple[int, int, int, int]] = []

    # ── Strategy 1: Adaptive threshold ──
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        15, 3,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, rw, rh = cv2.boundingRect(cnt)
            if rw > min_dim and rh > min_dim:
                all_rects.append((x, y, rw, rh))

    # ── Strategy 2: Canny edges ──
    edges = cv2.Canny(blurred, 30, 120)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours2, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, rw, rh = cv2.boundingRect(cnt)
            if rw > min_dim and rh > min_dim:
                all_rects.append((x, y, rw, rh))

    # Merge overlapping rectangles
    merged = _merge_overlapping_rects(all_rects, overlap_thresh=0.3)

    # Build region list
    regions = []
    for x, y, rw, rh in merged:
        cx = (x + rw / 2) / w
        cy = (y + rh / 2) / h
        regions.append({
            "bbox": (x, y, rw, rh),
            "area": rw * rh,
            "area_fraction": (rw * rh) / total_area,
            "center_norm": (cx, cy),
            "aspect_ratio": rw / max(rh, 1),
            "position_label": _classify_position(cx, cy),
        })

    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions[:max_regions]


def draw_semantic_overlay(
    base_image: Image.Image,
    regions: List[Dict],
    layout_tokens: Optional[List[str]] = None,
) -> Image.Image:
    """
    Draw semantic annotations on the sketch base image.

    Overlays:
      - Semi-transparent colored fills for each detected region
      - Colored borders with architectural corner marks
      - Role labels with accent colors
    """
    result = base_image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = result.size
    scale = min(w, h) / 500.0
    font_size = max(10, min(18, int(13 * scale)))
    border_w = max(2, int(2 * scale))
    corner_len = max(8, int(14 * scale))

    font = _load_font(font_size)

    # Parse layout tokens into a position→role lookup
    token_map = _parse_layout_tokens(layout_tokens)

    # Assign roles to each region
    for region in regions:
        region["role"] = _assign_role(region, token_map)

    # Draw regions
    for region in regions:
        _draw_single_region(draw, region, font, border_w, corner_len, w, h)

    result = Image.alpha_composite(result, overlay)
    return result


def generate_semantic_ui_sketch(
    image: Image.Image,
    layout_tokens: Optional[List[str]] = None,
    crop_classifier_func = None,
) -> Tuple[Image.Image, List[Dict]]:
    """
    Main entry point – generate a Semantic UI Sketch from a game UI image.

    Pipeline:
      1. simplify_to_sketch_style  → black-on-white edge sketch
      2. extract_ui_regions_opencv → bounding boxes of UI elements
      3. Blueprint canvas          → light grid background
      4. Blend sketch lines        → overlay edges onto grid
      5. Semantic overlay           → coloured regions + labels
      6. Legend                     → role colour key
    """
    w, h = image.size

    # 1  Sketch lines
    sketch_bw = simplify_to_sketch_style(image)

    # 2  UI regions
    regions = extract_ui_regions_opencv(image)

    # If crop classifier is provided, run it for each detected region
    if crop_classifier_func is not None:
        for region in regions:
            try:
                x, y, rw, rh = region["bbox"]
                crop_img = image.crop((x, y, x + rw, y + rh))
                elem_type, role = crop_classifier_func(crop_img)
                region["role"] = role
                region["element_type"] = elem_type
            except Exception as e:
                print(f"[!] Error classifying region crop: {e}")

    # 3  Blueprint background
    canvas = _create_blueprint_canvas(w, h)

    # 4  Blend sketch onto canvas
    canvas = _blend_sketch_onto_canvas(canvas, sketch_bw)

    # 5  Semantic overlay
    result = draw_semantic_overlay(canvas, regions, layout_tokens)

    return result, regions


# ──────────────────────────────────────────────────────────────
#  Internal Helpers
# ──────────────────────────────────────────────────────────────

def _create_blueprint_canvas(width: int, height: int) -> Image.Image:
    """Light blueprint grid background."""
    bg = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(bg)

    # Fine grid (every 20 px)
    fine = (238, 242, 246)
    for x in range(0, width, 20):
        draw.line([(x, 0), (x, height)], fill=fine, width=1)
    for y in range(0, height, 20):
        draw.line([(0, y), (width, y)], fill=fine, width=1)

    # Major grid (every 80 px)
    major = (222, 230, 240)
    for x in range(0, width, 80):
        draw.line([(x, 0), (x, height)], fill=major, width=1)
    for y in range(0, height, 80):
        draw.line([(0, y), (width, y)], fill=major, width=1)

    return bg


def _blend_sketch_onto_canvas(canvas: Image.Image, sketch_bw: Image.Image) -> Image.Image:
    """Composite black sketch lines onto the blueprint canvas.

    White pixels in the sketch become transparent (blueprint shows through).
    Black pixels become dark slate lines.
    """
    canvas_np = np.array(canvas).astype(np.float32)
    # sketch_bw is mode 'L': 0 = line, 255 = background
    sketch_np = np.array(sketch_bw.convert("L")).astype(np.float32) / 255.0

    line_color = np.array([45, 50, 60], dtype=np.float32)

    for c in range(3):
        canvas_np[:, :, c] = (
            canvas_np[:, :, c] * sketch_np
            + line_color[c] * (1.0 - sketch_np)
        )

    return Image.fromarray(canvas_np.astype(np.uint8))


def _parse_layout_tokens(tokens: Optional[List[str]]) -> Dict[str, str]:
    """Parse 'position:element:role' tokens into {position: role}."""
    if not tokens:
        return {}
    mapping: Dict[str, str] = {}
    for token in tokens:
        parts = token.strip().split(":")
        if len(parts) >= 3:
            mapping[parts[0].lower()] = parts[2].lower()
        elif len(parts) == 2:
            mapping[parts[0].lower()] = parts[1].lower()
    return mapping


def _assign_role(region: Dict, token_map: Dict[str, str]) -> str:
    """Assign a semantic role to a region via token lookup or heuristic."""
    # If a manual token map override exists for this position, use it
    pos = region["position_label"]
    if pos in token_map:
        return token_map[pos]

    for tok_pos, role in token_map.items():
        if tok_pos in pos or pos in tok_pos:
            return role

    # Otherwise, if the region has already been classified (e.g. by crop classifier), use it
    if "role" in region and region["role"] != "unknown":
        return region["role"]

    # Heuristic fallback
    return _guess_role(region)


def _guess_role(region: Dict) -> str:
    """Heuristic: infer UI role from shape + position."""
    pos = region["position_label"]
    ar = region["aspect_ratio"]
    af = region.get("area_fraction", 0)

    # Wide horizontal bars (aspect ratio > 3)
    if ar > 3.0:
        if "top" in pos:
            return "health"
        if "bottom" in pos:
            return "combat"
        return "status"

    # Near-square, small–medium
    if 0.7 < ar < 1.5 and af < 0.08:
        if "right" in pos and "top" in pos:
            return "navigation"
        if "left" in pos and "bottom" in pos:
            return "communication"
        return "info"

    # Tall panels (ar < 0.6)
    if ar < 0.6:
        if "right" in pos:
            return "statistics"
        if "left" in pos:
            return "inventory"
        return "quest"

    # Large centre panel
    if "center" in pos and af > 0.15:
        return "inventory"

    # Bottom-left medium
    if "bottom" in pos and "left" in pos:
        return "communication"

    return "unknown"


def _classify_position(cx: float, cy: float) -> str:
    """Map normalised centre (0-1) to a position label."""
    v = "top" if cy < 0.33 else ("bottom" if cy > 0.67 else "center")
    h = "left" if cx < 0.35 else ("right" if cx > 0.65 else "center")

    if v == "center" and h == "center":
        return "center"
    if h == "center":
        return f"{v}_center"
    if v == "center":
        return h
    return f"{v}_{h}"


def _merge_overlapping_rects(
    rects: List[Tuple[int, int, int, int]],
    overlap_thresh: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    """Greedy merge of overlapping bounding rectangles."""
    if not rects:
        return []

    used = [False] * len(rects)
    merged = []

    for i in range(len(rects)):
        if used[i]:
            continue
        x, y, w, h = rects[i]
        gx1, gy1, gx2, gy2 = x, y, x + w, y + h
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(rects)):
                if used[j]:
                    continue
                bx, by, bw, bh = rects[j]
                bx2, by2 = bx + bw, by + bh

                ix1, iy1 = max(gx1, bx), max(gy1, by)
                ix2, iy2 = min(gx2, bx2), min(gy2, by2)

                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    box_area = bw * bh
                    if box_area > 0 and inter / box_area > overlap_thresh:
                        gx1 = min(gx1, bx)
                        gy1 = min(gy1, by)
                        gx2 = max(gx2, bx2)
                        gy2 = max(gy2, by2)
                        used[j] = True
                        changed = True

        merged.append((gx1, gy1, gx2 - gx1, gy2 - gy1))

    return merged


def _draw_single_region(
    draw: ImageDraw.ImageDraw,
    region: Dict,
    font,
    border_w: int,
    corner_len: int,
    img_w: int,
    img_h: int,
) -> None:
    """Draw one region: fill, border, corner marks, label."""
    x, y, rw, rh = region["bbox"]
    role = region.get("role", "unknown")
    label_text, tag, accent = role_to_label_and_icon(role)

    # Semi-transparent fill
    draw.rectangle([x, y, x + rw, y + rh], fill=(*accent, 22))

    # Border
    draw.rectangle([x, y, x + rw, y + rh], outline=(*accent, 150), width=border_w)

    # Corner marks (architectural / crop-mark style)
    cl = min(corner_len, rw // 4, rh // 4)
    cw = border_w + 1
    cc = (*accent, 220)

    # top-left
    draw.line([(x, y), (x + cl, y)], fill=cc, width=cw)
    draw.line([(x, y), (x, y + cl)], fill=cc, width=cw)
    # top-right
    draw.line([(x + rw, y), (x + rw - cl, y)], fill=cc, width=cw)
    draw.line([(x + rw, y), (x + rw, y + cl)], fill=cc, width=cw)
    # bottom-left
    draw.line([(x, y + rh), (x + cl, y + rh)], fill=cc, width=cw)
    draw.line([(x, y + rh), (x, y + rh - cl)], fill=cc, width=cw)
    # bottom-right
    draw.line([(x + rw, y + rh), (x + rw - cl, y + rh)], fill=cc, width=cw)
    draw.line([(x + rw, y + rh), (x + rw, y + rh - cl)], fill=cc, width=cw)

    # ── Label ──
    display = f"[{tag}] {label_text}"
    try:
        bbox = font.getbbox(display)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = len(display) * 8, 14

    pad_x, pad_y = 6, 3
    lw, lh = tw + pad_x * 2, th + pad_y * 2

    # Place label above box when room exists, otherwise inside top
    lx = x + 4
    ly = y - lh - 3 if y > lh + 6 else y + 4

    # Keep within image bounds
    if lx + lw > img_w:
        lx = max(0, img_w - lw - 2)
    if ly < 0:
        ly = y + 4

    # White background label box
    draw.rectangle(
        [lx, ly, lx + lw, ly + lh],
        fill=(255, 255, 255, 230),
        outline=(*accent, 180),
        width=1,
    )
    draw.text((lx + pad_x, ly + pad_y), display, fill=(*accent, 255), font=font)


def _draw_legend(image: Image.Image, regions: List[Dict]) -> Image.Image:
    """Draw a colour-coded legend in the bottom-right corner."""
    if not regions:
        return image

    # Collect unique roles (preserve order of first appearance)
    seen: set = set()
    items: List[Tuple[str, Tuple[int, int, int]]] = []
    for r in regions:
        role = r.get("role", "unknown")
        if role not in seen:
            seen.add(role)
            label, tag, colour = role_to_label_and_icon(role)
            items.append((f"[{tag}] {label}", colour))

    if not items:
        return image

    result = image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = result.size
    scale = min(w, h) / 500.0
    font_sz = max(9, min(14, int(11 * scale)))
    font = _load_font(font_sz)
    title_font = _load_font(font_sz + 1)

    item_h = max(14, font_sz + 4)
    pad = 8

    # Measure widest text
    max_tw = 0
    for text, _ in items:
        try:
            bb = font.getbbox(text)
            tw = bb[2] - bb[0]
        except Exception:
            tw = len(text) * 7
        max_tw = max(max_tw, tw)

    legend_w = max_tw + 24 + pad * 2
    legend_h = len(items) * (item_h + 3) + pad * 2 + item_h + 4

    lx = w - legend_w - 10
    ly = h - legend_h - 10

    # Background
    draw.rounded_rectangle(
        [lx, ly, lx + legend_w, ly + legend_h],
        radius=6,
        fill=(255, 255, 255, 210),
        outline=(180, 190, 200, 200),
        width=1,
    )

    # Title
    draw.text((lx + pad, ly + pad), "Legend", fill=(50, 55, 65, 255), font=title_font)

    # Items
    cy = ly + pad + item_h + 6
    dot_r = max(3, item_h // 4)
    for text, colour in items:
        dot_cx = lx + pad + dot_r
        dot_cy = cy + item_h // 2
        draw.ellipse(
            [dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r],
            fill=(*colour, 200),
        )
        draw.text(
            (lx + pad + dot_r * 2 + 8, cy + 1),
            text,
            fill=(50, 55, 65, 255),
            font=font,
        )
        cy += item_h + 3

    return Image.alpha_composite(result, overlay).convert("RGB")


# ──────────────────────────────────────────────────────────────
#  Backward Compatibility
# ──────────────────────────────────────────────────────────────

def draw_layout_sketch(*args, **kwargs) -> Image.Image:
    """Legacy stub – keeps existing app.py import from breaking."""
    return Image.new("RGB", (1024, 768), color="#F8FAFC")


def generate_opencv_sketch(image: Image.Image) -> Image.Image:
    """Legacy redirect – routes to the new semantic pipeline."""
    return generate_semantic_ui_sketch(image)
