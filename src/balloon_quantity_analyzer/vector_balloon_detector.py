"""Vector-based balloon detector for PDF files with embedded vector graphics.

This detector finds balloon circles by analyzing PDF vector paths (curves)
directly via pdfplumber, rather than rasterizing and running Hough Circle
Transform. This is much more accurate for CAD-generated PDFs (SOLIDWORKS,
AutoCAD, etc.) where balloons are drawn as vector circles.

It also uses PDF bookmarks (outlines) to identify drawing view boundaries,
enabling view-level multipliers like "BOTH SIDES" to be applied correctly
to all balloons within a specific view.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import pdfplumber
from pypdf import PdfReader

from balloon_quantity_analyzer.models import (
    BoundingBox,
    DetectedBalloon,
    ParsedMultiplier,
    Warning,
    WarningType,
)

logger = logging.getLogger("balloon_quantity_analyzer")


# ---------------------------------------------------------------------------
# View boundary extraction from PDF bookmarks
# ---------------------------------------------------------------------------

@dataclass
class ViewBoundary:
    """A drawing view's bounding rectangle from a PDF bookmark."""
    title: str
    page_number: int  # 1-indexed
    left: float
    bottom: float
    right: float
    top: float

    def contains_point(self, x: float, y: float, margin: float = 20.0) -> bool:
        """Check if a point (in PDF coordinates) is inside this view, with margin."""
        return (
            self.left - margin <= x <= self.right + margin
            and self.bottom - margin <= y <= self.top + margin
        )

    def contains_point_pdfplumber(self, x: float, y: float, page_height: float, margin: float = 20.0) -> bool:
        """Check containment using pdfplumber coordinates (y=0 at top)."""
        # pdfplumber y = page_height - pdf_y
        pdf_y = page_height - y
        return (
            self.left - margin <= x <= self.right + margin
            and self.bottom - margin <= pdf_y <= self.top + margin
        )


def _extract_view_boundaries(pdf_path: str) -> list[ViewBoundary]:
    """Extract drawing view boundaries from PDF bookmarks using pypdf."""
    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        logger.warning("Failed to read PDF bookmarks: %s", exc)
        return []

    views: list[ViewBoundary] = []
    page_objects = list(reader.pages)

    def _walk(items: list, parent_page_idx: int = 0) -> None:
        for item in items:
            if isinstance(item, list):
                _walk(item, parent_page_idx)
                continue

            title = item.get("/Title", "")
            fit_type = item.get("/Type", "")

            # Determine which page this bookmark is on
            page_obj = item.get("/Page")
            page_idx = parent_page_idx
            if page_obj is not None:
                for i, p in enumerate(page_objects):
                    if p.indirect_reference == page_obj.indirect_reference:
                        page_idx = i
                        break

            # Sheet-level bookmarks — track page for children
            if fit_type == "/Fit":
                child_count = item.get("/Count", 0)
                is_open = item.get("/%is_open%", False)
                # This is a sheet bookmark; children will use this page
                continue

            # View-level bookmarks with FitR have bounding rectangles
            if fit_type == "/FitR":
                left = float(item.get("/Left", 0))
                bottom = float(item.get("/Bottom", 0))
                right = float(item.get("/Right", 0))
                top = float(item.get("/Top", 0))

                views.append(ViewBoundary(
                    title=title,
                    page_number=page_idx + 1,
                    left=left,
                    bottom=bottom,
                    right=right,
                    top=top,
                ))

    if reader.outline:
        _walk(reader.outline)

    return views


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_circle_curve(curve: dict, min_r: float = 8.0, max_r: float = 20.0) -> bool:
    """Check if a pdfplumber curve is roughly circular and balloon-sized."""
    w = curve["x1"] - curve["x0"]
    h = curve["bottom"] - curve["top"]
    if w < 1 or h < 1:
        return False
    r = w / 2
    if abs(w - h) > 3:
        return False
    if r < min_r or r > max_r:
        return False
    return True


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_balloons_from_pdf(
    pdf_path: str,
    min_balloon_radius: float = 8.0,
    max_balloon_radius: float = 20.0,
    text_search_radius: float = 5.0,
    multiplier_search_radius: float = 40.0,
) -> tuple[list[DetectedBalloon], list[ParsedMultiplier], list[Warning]]:
    """Detect balloons from a vector PDF using pdfplumber curve analysis.

    Uses PDF bookmarks to identify drawing view boundaries, enabling
    view-level multipliers like "BOTH SIDES" to be applied correctly.
    """
    # Step 1: Extract view boundaries from bookmarks
    view_boundaries = _extract_view_boundaries(pdf_path)
    if view_boundaries:
        logger.info("  Found %d view boundaries from bookmarks", len(view_boundaries))
        for vb in view_boundaries:
            logger.info("    %s (page %d): L=%.0f B=%.0f R=%.0f T=%.0f",
                        vb.title, vb.page_number, vb.left, vb.bottom, vb.right, vb.top)

    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as exc:
        logger.warning("Failed to open PDF for vector detection: %s", exc)
        return [], [], []

    all_balloons: list[DetectedBalloon] = []
    all_multipliers: list[ParsedMultiplier] = []
    all_warnings: list[Warning] = []
    # Track which view each balloon belongs to (index into all_balloons -> view title)
    balloon_view_map: dict[int, str] = {}

    try:
        for page_idx, page in enumerate(pdf.pages):
            page_number = page_idx + 1
            page_height = float(page.height)
            logger.info("  Vector detection on page %d...", page_number)

            # Get views for this page
            page_views = [v for v in view_boundaries if v.page_number == page_number]

            # Find balloon-sized circles
            balloon_circles: list[tuple[float, float, float]] = []
            for curve in page.curves:
                if _is_circle_curve(curve, min_balloon_radius, max_balloon_radius):
                    cx = (curve["x0"] + curve["x1"]) / 2
                    cy = (curve["top"] + curve["bottom"]) / 2
                    r = (curve["x1"] - curve["x0"]) / 2
                    is_dup = any(
                        _distance(cx, cy, ex, ey) < 3
                        for ex, ey, _ in balloon_circles
                    )
                    if not is_dup:
                        balloon_circles.append((cx, cy, r))

            logger.info("  Found %d balloon-sized circles", len(balloon_circles))

            # Get all words on the page
            words = page.extract_words()

            # Track multiplier indices created on this page (for view-level multiplier application)
            page_mult_start = len(all_multipliers)

            # Match each circle with nearby text
            for cx, cy, r in balloon_circles:
                find_number = ""
                find_confidence = 0.0
                balloon_idx = len(all_balloons)

                # Look for Find number text inside/near the circle
                for w in words:
                    wcx = (w["x0"] + w["x1"]) / 2
                    wcy = (w["top"] + w["bottom"]) / 2
                    dist = _distance(wcx, wcy, cx, cy)
                    if dist < r + text_search_radius:
                        candidate = w["text"].strip()
                        if re.match(r"^\d{1,3}[A-Z]?$", candidate):
                            find_number = candidate
                            find_confidence = 1.0

                # Look for multiplier text near the circle
                for w in words:
                    wcx = (w["x0"] + w["x1"]) / 2
                    wcy = (w["top"] + w["bottom"]) / 2
                    dist = _distance(wcx, wcy, cx, cy)
                    if r + text_search_radius < dist < r + multiplier_search_radius:
                        candidate = w["text"].strip()
                        mult_match = re.match(r"^(\d+)X$", candidate, re.IGNORECASE)
                        if mult_match:
                            mult_value = int(mult_match.group(1))
                            if mult_value > 0:
                                all_multipliers.append(
                                    ParsedMultiplier(
                                        value=mult_value,
                                        raw_text=candidate,
                                        bounding_box=BoundingBox(
                                            x=w["x0"], y=w["top"],
                                            width=w["x1"] - w["x0"],
                                            height=w["bottom"] - w["top"],
                                        ),
                                        page_number=page_number,
                                        confidence=1.0,
                                    )
                                )

                bbox = BoundingBox(x=cx - r, y=cy - r, width=r * 2, height=r * 2)

                balloon = DetectedBalloon(
                    find_number=find_number,
                    page_number=page_number,
                    bounding_box=bbox,
                    confidence=find_confidence if find_number else 0.5,
                )
                all_balloons.append(balloon)

                # Associate balloon with a view
                # Associate balloon with the closest containing view
                best_view: str | None = None
                best_dist = float("inf")
                for vb in page_views:
                    if vb.contains_point_pdfplumber(cx, cy, page_height, margin=50.0):
                        vcx = (vb.left + vb.right) / 2
                        vcy = page_height - (vb.bottom + vb.top) / 2
                        dist = _distance(cx, cy, vcx, vcy)
                        if dist < best_dist:
                            best_dist = dist
                            best_view = vb.title
                if best_view:
                    balloon_view_map[balloon_idx] = best_view

                if not find_number:
                    all_warnings.append(
                        Warning(
                            warning_type=WarningType.UNREADABLE_FIND_NUMBER,
                            message=(
                                f"Balloon at ({cx:.0f}, {cy:.0f}) on page "
                                f"{page_number} has no matching Find number text."
                            ),
                            page_number=page_number,
                            related_items=[f"bbox({bbox.x:.0f},{bbox.y:.0f},{bbox.width:.0f},{bbox.height:.0f})"],
                        )
                    )

            # --- Detect view-level multipliers (e.g., "BOTH SIDES") ---
            _apply_view_level_multipliers(
                words, page_views, page_height, page_number,
                all_balloons, all_multipliers, balloon_view_map,
                page_mult_start,
            )
    finally:
        pdf.close()

    return all_balloons, all_multipliers, all_warnings


def _apply_view_level_multipliers(
    words: list[dict],
    page_views: list[ViewBoundary],
    page_height: float,
    page_number: int,
    all_balloons: list[DetectedBalloon],
    all_multipliers: list[ParsedMultiplier],
    balloon_view_map: dict[int, str],
    page_mult_start: int,
) -> None:
    """Find view-level multiplier notes and apply them to balloons in the same view.

    Currently handles:
    - "BOTH SIDES" → multiply all balloon multipliers in the view by 2
    """
    # Find "BOTH SIDES" text on this page
    both_sides_positions: list[tuple[float, float]] = []
    for i, w in enumerate(words):
        if w["text"].strip().upper() == "BOTH":
            for j in range(i + 1, min(i + 3, len(words))):
                if words[j]["text"].strip().upper() == "SIDES":
                    bx = (w["x0"] + words[j]["x1"]) / 2
                    by = (w["top"] + words[j]["bottom"]) / 2
                    both_sides_positions.append((bx, by))
                    break

    if not both_sides_positions:
        return

    for bsx, bsy in both_sides_positions:
        logger.info("  Found 'BOTH SIDES' at (%.0f, %.0f) on page %d", bsx, bsy, page_number)

        # Find which view this "BOTH SIDES" belongs to
        target_view: str | None = None
        for vb in page_views:
            if vb.contains_point_pdfplumber(bsx, bsy, page_height, margin=50.0):
                target_view = vb.title
                break

        if target_view is None:
            # Fallback: find the closest view
            min_dist = float("inf")
            for vb in page_views:
                vcx = (vb.left + vb.right) / 2
                vcy_pdf = (vb.bottom + vb.top) / 2
                vcy = page_height - vcy_pdf  # convert to pdfplumber coords
                dist = _distance(bsx, bsy, vcx, vcy)
                if dist < min_dist:
                    min_dist = dist
                    target_view = vb.title

        if target_view is None:
            logger.warning("  Could not associate 'BOTH SIDES' with any view")
            continue

        logger.info("  'BOTH SIDES' belongs to view: %s", target_view)

        # Find all balloons in this view and double their multipliers
        for mi in range(page_mult_start, len(all_multipliers)):
            m = all_multipliers[mi]
            # Find which balloon this multiplier is near
            mcx = m.bounding_box.x + m.bounding_box.width / 2
            mcy = m.bounding_box.y + m.bounding_box.height / 2

            # Check if this multiplier's associated balloon is in the target view
            for bi, view_title in balloon_view_map.items():
                if view_title != target_view:
                    continue
                b = all_balloons[bi]
                bcx = b.bounding_box.x + b.bounding_box.width / 2
                bcy = b.bounding_box.y + b.bounding_box.height / 2
                dist = _distance(mcx, mcy, bcx, bcy)
                if dist < 50.0:  # multiplier is near this balloon
                    doubled = ParsedMultiplier(
                        value=m.value * 2,
                        raw_text=f"{m.raw_text} (BOTH SIDES)",
                        bounding_box=m.bounding_box,
                        page_number=m.page_number,
                        confidence=m.confidence,
                    )
                    all_multipliers[mi] = doubled
                    logger.info(
                        "  Applied BOTH SIDES to balloon %s multiplier '%s' -> %dX",
                        all_balloons[bi].find_number, m.raw_text, m.value * 2,
                    )
                    break
