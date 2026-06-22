"""Detail View Resolver — identifies Detail views, their regions, multipliers,
and which balloons are contained within each Detail view."""

from __future__ import annotations

import re

from balloon_quantity_analyzer.models import (
    BoundingBox,
    DetectedBalloon,
    DetailView,
    NormalizedPage,
    Warning,
    WarningType,
)

# Regex for Detail view labels.
# Matches: "DETAIL A", "DETAIL B (3 PLACES)", "DETAIL C (2 PLCS)", "DETAIL D (4 PL)"
# Also tolerates missing parentheses: "DETAIL E 5 PLACES"
_DETAIL_LABEL_RE = re.compile(
    r"DETAIL\s+([A-Z])"
    r"(?:\s*\(?\s*(\d+)\s*(?:PLACES|PLCS|PL)\s*\)?)?"
    r"\s*$",
    re.IGNORECASE,
)

# Default region size (pixels) inferred around a Detail view label when no
# contour information is available.
_DEFAULT_REGION_PADDING = 200.0


def parse_detail_label(text: str) -> tuple[str, int | None] | None:
    """Parse a Detail view label string.

    Returns ``(identifier, multiplier_or_None)`` when *text* matches a
    Detail view label pattern, or ``None`` if it does not match.

    Examples::

        >>> parse_detail_label("DETAIL A")
        ('A', None)
        >>> parse_detail_label("DETAIL B (3 PLACES)")
        ('B', 3)
        >>> parse_detail_label("hello world")
        None
    """
    m = _DETAIL_LABEL_RE.match(text.strip())
    if m is None:
        return None

    identifier = m.group(1).upper()
    raw_mult = m.group(2)
    multiplier = int(raw_mult) if raw_mult is not None else None
    return identifier, multiplier


def _bbox_center(bb: BoundingBox) -> tuple[float, float]:
    """Return the center ``(cx, cy)`` of a bounding box."""
    return bb.x + bb.width / 2.0, bb.y + bb.height / 2.0


def _point_in_bbox(px: float, py: float, bb: BoundingBox) -> bool:
    """Return ``True`` if point ``(px, py)`` lies inside *bb*."""
    return (
        bb.x <= px <= bb.x + bb.width
        and bb.y <= py <= bb.y + bb.height
    )


def _infer_region(label_box: BoundingBox, padding: float = _DEFAULT_REGION_PADDING) -> BoundingBox:
    """Infer a Detail view region from the label position.

    The region is a padded rectangle centered on the label, extending
    *padding* pixels in every direction.
    """
    cx, cy = _bbox_center(label_box)
    x = max(cx - padding, 0.0)
    y = max(cy - padding, 0.0)
    return BoundingBox(x=x, y=y, width=padding * 2, height=padding * 2)


class DetailViewResolver:
    """Identifies Detail views on a page and determines which balloons they contain."""

    def __init__(self, region_padding: float = _DEFAULT_REGION_PADDING) -> None:
        self._region_padding = region_padding

    def resolve(
        self, page: NormalizedPage, balloons: list[DetectedBalloon]
    ) -> tuple[list[DetailView], list[Warning]]:
        """Detect Detail view labels, infer regions, and classify balloon containment.

        Returns ``(detail_views, warnings)``.
        """
        warnings: list[Warning] = []

        # --- Step 1: Scan text regions for Detail view labels ---
        raw_details: list[tuple[str, int | None, BoundingBox]] = []
        for tr in page.text_regions:
            parsed = parse_detail_label(tr.text)
            if parsed is not None:
                identifier, multiplier = parsed
                raw_details.append((identifier, multiplier, tr.bounding_box))

        # --- Step 2: Infer region for each Detail view ---
        detail_views: list[DetailView] = []
        for identifier, multiplier, label_box in raw_details:
            region = _infer_region(label_box, self._region_padding)

            # --- Step 3: Determine contained balloons ---
            contained: list[int] = []
            for bi, balloon in enumerate(balloons):
                if balloon.page_number != page.page_number:
                    continue
                bcx, bcy = _bbox_center(balloon.bounding_box)
                if _point_in_bbox(bcx, bcy, region):
                    contained.append(bi)

            detail_views.append(
                DetailView(
                    identifier=identifier,
                    page_number=page.page_number,
                    region=region,
                    multiplier=multiplier,
                    contained_balloon_indices=contained,
                )
            )

        # --- Step 4: Warn on overlapping Detail views with multipliers ---
        # Build a map: balloon_index -> list of detail views (with multipliers) containing it
        balloon_dv_map: dict[int, list[str]] = {}
        for dv in detail_views:
            if dv.multiplier is not None:
                for bi in dv.contained_balloon_indices:
                    balloon_dv_map.setdefault(bi, []).append(dv.identifier)

        for bi, dv_ids in balloon_dv_map.items():
            if len(dv_ids) >= 2:
                balloon = balloons[bi]
                fn = balloon.find_number or f"balloon@{bi}"
                warnings.append(
                    Warning(
                        warning_type=WarningType.OVERLAPPING_DETAIL_VIEWS,
                        message=(
                            f"Balloon '{fn}' lies inside overlapping Detail views "
                            f"with multipliers: {dv_ids}"
                        ),
                        page_number=page.page_number,
                        related_items=[fn] + dv_ids,
                    )
                )

        return detail_views, warnings
