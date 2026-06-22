"""Confidence scoring utilities — threshold-based low-confidence warning generation."""

from __future__ import annotations

from balloon_quantity_analyzer.models import (
    AssociatedBalloon,
    DetectedBalloon,
    ParsedMultiplier,
    Warning,
    WarningType,
)


def check_low_confidence(
    items: list[DetectedBalloon | ParsedMultiplier | AssociatedBalloon],
    threshold: float,
    item_type: str,
) -> list[Warning]:
    """Check items for low confidence and return warnings for those below threshold.

    Parameters
    ----------
    items:
        A list of confidence-bearing objects.
    threshold:
        Confidence threshold in [0.0, 1.0]. Items with confidence strictly
        below this value trigger a warning.
    item_type:
        Human-readable label for the kind of item (e.g. "detected balloon",
        "parsed multiplier", "association").

    Returns
    -------
    list[Warning]
        One warning per item whose confidence is below *threshold*.
    """
    warnings: list[Warning] = []

    for item in items:
        if item.confidence < threshold:
            page = item.page_number

            # Build a short identifier for the related item
            if isinstance(item, DetectedBalloon):
                identifier = item.find_number or f"balloon@({item.bounding_box.x:.0f},{item.bounding_box.y:.0f})"
            elif isinstance(item, ParsedMultiplier):
                identifier = item.raw_text
            elif isinstance(item, AssociatedBalloon):
                identifier = item.find_number or f"association@({item.bounding_box.x:.0f},{item.bounding_box.y:.0f})"
            else:
                identifier = str(item)

            warnings.append(
                Warning(
                    warning_type=WarningType.LOW_CONFIDENCE,
                    message=(
                        f"Low confidence {item_type} '{identifier}': "
                        f"confidence={item.confidence:.3f}, threshold={threshold:.3f}"
                    ),
                    page_number=page,
                    related_items=[identifier, f"{item.confidence:.3f}"],
                )
            )

    return warnings
