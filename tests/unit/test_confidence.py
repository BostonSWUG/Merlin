"""Unit tests for the confidence scoring / low-confidence warning utility."""

from __future__ import annotations

from balloon_quantity_analyzer.confidence import check_low_confidence
from balloon_quantity_analyzer.models import (
    AssociatedBalloon,
    BoundingBox,
    DetectedBalloon,
    ParsedMultiplier,
    WarningType,
)

_BOX = BoundingBox(x=10.0, y=20.0, width=30.0, height=30.0)


class TestCheckLowConfidence:
    """Tests for check_low_confidence utility."""

    def test_no_items_returns_empty(self) -> None:
        assert check_low_confidence([], threshold=0.5, item_type="balloon") == []

    def test_all_above_threshold_returns_empty(self) -> None:
        items = [
            DetectedBalloon(find_number="1", page_number=1, bounding_box=_BOX, confidence=0.9),
            DetectedBalloon(find_number="2", page_number=1, bounding_box=_BOX, confidence=0.8),
        ]
        warnings = check_low_confidence(items, threshold=0.5, item_type="detected balloon")
        assert warnings == []

    def test_item_at_threshold_no_warning(self) -> None:
        items = [
            DetectedBalloon(find_number="1", page_number=1, bounding_box=_BOX, confidence=0.5),
        ]
        warnings = check_low_confidence(items, threshold=0.5, item_type="detected balloon")
        assert warnings == []

    def test_item_below_threshold_generates_warning(self) -> None:
        items = [
            DetectedBalloon(find_number="42", page_number=3, bounding_box=_BOX, confidence=0.3),
        ]
        warnings = check_low_confidence(items, threshold=0.5, item_type="detected balloon")
        assert len(warnings) == 1
        w = warnings[0]
        assert w.warning_type == WarningType.LOW_CONFIDENCE
        assert "42" in w.message
        assert "0.300" in w.message
        assert "0.500" in w.message
        assert w.page_number == 3

    def test_multiple_items_mixed_confidence(self) -> None:
        items = [
            DetectedBalloon(find_number="1", page_number=1, bounding_box=_BOX, confidence=0.9),
            DetectedBalloon(find_number="2", page_number=1, bounding_box=_BOX, confidence=0.2),
            DetectedBalloon(find_number="3", page_number=2, bounding_box=_BOX, confidence=0.4),
        ]
        warnings = check_low_confidence(items, threshold=0.5, item_type="detected balloon")
        assert len(warnings) == 2
        find_numbers = [w.related_items[0] for w in warnings]
        assert "2" in find_numbers
        assert "3" in find_numbers

    def test_parsed_multiplier_low_confidence(self) -> None:
        items = [
            ParsedMultiplier(value=3, raw_text="3X", bounding_box=_BOX, page_number=1, confidence=0.1),
        ]
        warnings = check_low_confidence(items, threshold=0.5, item_type="parsed multiplier")
        assert len(warnings) == 1
        assert "3X" in warnings[0].message
        assert warnings[0].warning_type == WarningType.LOW_CONFIDENCE

    def test_associated_balloon_low_confidence(self) -> None:
        items = [
            AssociatedBalloon(
                find_number="7",
                page_number=2,
                bounding_box=_BOX,
                adjacent_multiplier_text=None,
                adjacent_multiplier_value=1,
                detail_view_id=None,
                detail_view_multiplier=1,
                effective_multiplier=1,
                confidence=0.05,
            ),
        ]
        warnings = check_low_confidence(items, threshold=0.5, item_type="association")
        assert len(warnings) == 1
        assert "7" in warnings[0].message
        assert warnings[0].page_number == 2

    def test_empty_find_number_uses_bbox_identifier(self) -> None:
        items = [
            DetectedBalloon(find_number="", page_number=1, bounding_box=_BOX, confidence=0.1),
        ]
        warnings = check_low_confidence(items, threshold=0.5, item_type="detected balloon")
        assert len(warnings) == 1
        # Should use bounding box coordinates as identifier
        assert "balloon@(10,20)" in warnings[0].message

    def test_related_items_contain_identifier_and_score(self) -> None:
        items = [
            DetectedBalloon(find_number="5", page_number=1, bounding_box=_BOX, confidence=0.25),
        ]
        warnings = check_low_confidence(items, threshold=0.5, item_type="detected balloon")
        assert len(warnings) == 1
        assert warnings[0].related_items[0] == "5"
        assert warnings[0].related_items[1] == "0.250"
