# Feature: balloon-quantity-analyzer, Property 11: Confidence scores in valid range
# Feature: balloon-quantity-analyzer, Property 12: Low confidence triggers warning
# Feature: balloon-quantity-analyzer, Property 13: Warning completeness
"""Property-based tests for confidence scoring and warning pipeline.

Tests cover:
- Property 11: Confidence scores in valid range (Validates: Requirements 2.4, 8.1)
- Property 12: Low confidence triggers warning (Validates: Requirements 8.2)
- Property 13: Warning completeness (Validates: Requirements 8.3)
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from balloon_quantity_analyzer.balloon_detector import BalloonDetector
from balloon_quantity_analyzer.confidence import check_low_confidence
from balloon_quantity_analyzer.models import (
    AnalysisReport,
    AssociatedBalloon,
    BalloonBreakdown,
    BoundingBox,
    DetectedBalloon,
    ParsedMultiplier,
    TallyResult,
    Warning,
    WarningType,
)
from balloon_quantity_analyzer.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_bounding_boxes = st.builds(
    BoundingBox,
    x=st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    width=st.floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False),
    height=st.floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False),
)

_find_numbers = st.text(
    alphabet="0123456789ABCDEF",
    min_size=1,
    max_size=4,
)

_valid_confidences = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

_detected_balloons = st.builds(
    DetectedBalloon,
    find_number=_find_numbers,
    page_number=st.integers(min_value=1, max_value=20),
    bounding_box=_bounding_boxes,
    confidence=_valid_confidences,
)

_parsed_multipliers = st.builds(
    ParsedMultiplier,
    value=st.integers(min_value=1, max_value=99),
    raw_text=st.text(min_size=1, max_size=10),
    bounding_box=_bounding_boxes,
    page_number=st.integers(min_value=1, max_value=20),
    confidence=_valid_confidences,
)

_associated_balloons = st.builds(
    AssociatedBalloon,
    find_number=_find_numbers,
    page_number=st.integers(min_value=1, max_value=20),
    bounding_box=_bounding_boxes,
    adjacent_multiplier_text=st.one_of(st.none(), st.text(min_size=1, max_size=10)),
    adjacent_multiplier_value=st.integers(min_value=1, max_value=99),
    detail_view_id=st.one_of(st.none(), st.sampled_from(list("ABCDEFGHIJ"))),
    detail_view_multiplier=st.integers(min_value=1, max_value=10),
    effective_multiplier=st.integers(min_value=1, max_value=999),
    confidence=_valid_confidences,
)

_warning_types = st.sampled_from(list(WarningType))

_warnings = st.builds(
    Warning,
    warning_type=_warning_types,
    message=st.text(min_size=1, max_size=80),
    page_number=st.one_of(st.none(), st.integers(min_value=1, max_value=20)),
    related_items=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5),
)


# ===========================================================================
# Property 11: Confidence scores in valid range
# **Validates: Requirements 2.4, 8.1**
# ===========================================================================


@settings(max_examples=200)
@given(
    hough_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ocr_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property11_combine_confidence_in_valid_range(
    hough_score: float,
    ocr_confidence: float,
) -> None:
    """BalloonDetector._combine_confidence always produces values in [0.0, 1.0]."""
    result = BalloonDetector._combine_confidence(hough_score, ocr_confidence)
    assert 0.0 <= result <= 1.0, (
        f"Combined confidence {result} out of range for "
        f"hough_score={hough_score}, ocr_confidence={ocr_confidence}"
    )


@settings(max_examples=200)
@given(balloon=_detected_balloons)
def test_property11_detected_balloon_confidence_in_range(
    balloon: DetectedBalloon,
) -> None:
    """DetectedBalloon objects with confidence in [0.0, 1.0] maintain that invariant."""
    assert 0.0 <= balloon.confidence <= 1.0, (
        f"DetectedBalloon confidence {balloon.confidence} out of [0.0, 1.0]"
    )


@settings(max_examples=200)
@given(multiplier=_parsed_multipliers)
def test_property11_parsed_multiplier_confidence_in_range(
    multiplier: ParsedMultiplier,
) -> None:
    """ParsedMultiplier objects with confidence in [0.0, 1.0] maintain that invariant."""
    assert 0.0 <= multiplier.confidence <= 1.0, (
        f"ParsedMultiplier confidence {multiplier.confidence} out of [0.0, 1.0]"
    )


@settings(max_examples=200)
@given(assoc=_associated_balloons)
def test_property11_associated_balloon_confidence_in_range(
    assoc: AssociatedBalloon,
) -> None:
    """AssociatedBalloon objects with confidence in [0.0, 1.0] maintain that invariant."""
    assert 0.0 <= assoc.confidence <= 1.0, (
        f"AssociatedBalloon confidence {assoc.confidence} out of [0.0, 1.0]"
    )


# ===========================================================================
# Property 12: Low confidence triggers warning
# **Validates: Requirements 8.2**
# ===========================================================================


@settings(max_examples=200)
@given(
    balloon=_detected_balloons,
    threshold=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property12_low_confidence_balloon_triggers_warning(
    balloon: DetectedBalloon,
    threshold: float,
) -> None:
    """A DetectedBalloon with confidence below threshold produces a LOW_CONFIDENCE warning."""
    # Force confidence below threshold
    low_conf = min(balloon.confidence, threshold * 0.99) if threshold > 0.0 else 0.0
    low_balloon = DetectedBalloon(
        find_number=balloon.find_number,
        page_number=balloon.page_number,
        bounding_box=balloon.bounding_box,
        confidence=low_conf,
    )

    warnings = check_low_confidence([low_balloon], threshold, "detected balloon")

    assert len(warnings) >= 1, (
        f"Expected warning for balloon with confidence={low_conf} < threshold={threshold}"
    )
    assert all(w.warning_type == WarningType.LOW_CONFIDENCE for w in warnings)
    # Verify the warning message references the confidence score
    for w in warnings:
        assert f"{low_conf:.3f}" in w.message


@settings(max_examples=200)
@given(
    multiplier=_parsed_multipliers,
    threshold=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property12_low_confidence_multiplier_triggers_warning(
    multiplier: ParsedMultiplier,
    threshold: float,
) -> None:
    """A ParsedMultiplier with confidence below threshold produces a LOW_CONFIDENCE warning."""
    low_conf = min(multiplier.confidence, threshold * 0.99) if threshold > 0.0 else 0.0
    low_mult = ParsedMultiplier(
        value=multiplier.value,
        raw_text=multiplier.raw_text,
        bounding_box=multiplier.bounding_box,
        page_number=multiplier.page_number,
        confidence=low_conf,
    )

    warnings = check_low_confidence([low_mult], threshold, "parsed multiplier")

    assert len(warnings) >= 1, (
        f"Expected warning for multiplier with confidence={low_conf} < threshold={threshold}"
    )
    assert all(w.warning_type == WarningType.LOW_CONFIDENCE for w in warnings)


@settings(max_examples=200)
@given(
    assoc=_associated_balloons,
    threshold=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property12_low_confidence_association_triggers_warning(
    assoc: AssociatedBalloon,
    threshold: float,
) -> None:
    """An AssociatedBalloon with confidence below threshold produces a LOW_CONFIDENCE warning."""
    low_conf = min(assoc.confidence, threshold * 0.99) if threshold > 0.0 else 0.0
    low_assoc = AssociatedBalloon(
        find_number=assoc.find_number,
        page_number=assoc.page_number,
        bounding_box=assoc.bounding_box,
        adjacent_multiplier_text=assoc.adjacent_multiplier_text,
        adjacent_multiplier_value=assoc.adjacent_multiplier_value,
        detail_view_id=assoc.detail_view_id,
        detail_view_multiplier=assoc.detail_view_multiplier,
        effective_multiplier=assoc.effective_multiplier,
        confidence=low_conf,
    )

    warnings = check_low_confidence([low_assoc], threshold, "association")

    assert len(warnings) >= 1, (
        f"Expected warning for association with confidence={low_conf} < threshold={threshold}"
    )
    assert all(w.warning_type == WarningType.LOW_CONFIDENCE for w in warnings)


@settings(max_examples=200)
@given(
    balloon=_detected_balloons,
    threshold=st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False),
)
def test_property12_high_confidence_no_warning(
    balloon: DetectedBalloon,
    threshold: float,
) -> None:
    """Items with confidence >= threshold should NOT produce a LOW_CONFIDENCE warning."""
    # Force confidence at or above threshold
    high_conf = max(balloon.confidence, threshold)
    high_balloon = DetectedBalloon(
        find_number=balloon.find_number,
        page_number=balloon.page_number,
        bounding_box=balloon.bounding_box,
        confidence=high_conf,
    )

    warnings = check_low_confidence([high_balloon], threshold, "detected balloon")

    assert len(warnings) == 0, (
        f"No warning expected for balloon with confidence={high_conf} >= threshold={threshold}, "
        f"but got {len(warnings)} warning(s)"
    )


# ===========================================================================
# Property 13: Warning completeness
# **Validates: Requirements 8.3**
# ===========================================================================


@settings(max_examples=200)
@given(
    subsystem_warnings=st.lists(
        st.lists(_warnings, min_size=0, max_size=5),
        min_size=1,
        max_size=5,
    ),
)
def test_property13_all_warnings_present_in_json_report(
    subsystem_warnings: list[list[Warning]],
) -> None:
    """All warnings from multiple subsystems appear in the final JSON report."""
    # Flatten all subsystem warnings into a single list
    all_warnings = [w for subsystem in subsystem_warnings for w in subsystem]

    # Create a minimal tally result for the report
    tally_result = TallyResult(
        tally={},
        balloon_breakdown=[],
        excluded_balloon_count=0,
    )

    generator = ReportGenerator()
    json_str = generator.generate_json(tally_result, all_warnings)

    # Parse the JSON and verify all warnings are present
    parsed = json.loads(json_str)
    report_warnings = parsed["warnings"]

    assert len(report_warnings) == len(all_warnings), (
        f"Expected {len(all_warnings)} warnings in report, got {len(report_warnings)}"
    )

    # Verify each original warning appears in the report
    for original in all_warnings:
        matching = [
            w for w in report_warnings
            if w["warning_type"] == original.warning_type.value
            and w["message"] == original.message
            and w["page_number"] == original.page_number
        ]
        assert len(matching) >= 1, (
            f"Warning not found in report: type={original.warning_type.value}, "
            f"message='{original.message}'"
        )


@settings(max_examples=200)
@given(
    subsystem_warnings=st.lists(
        st.lists(_warnings, min_size=0, max_size=5),
        min_size=1,
        max_size=5,
    ),
    tally=st.dictionaries(
        keys=_find_numbers,
        values=st.integers(min_value=1, max_value=999),
        min_size=0,
        max_size=5,
    ),
    breakdowns=st.lists(
        st.builds(
            BalloonBreakdown,
            find_number=_find_numbers,
            page_number=st.integers(min_value=1, max_value=20),
            bounding_box=_bounding_boxes,
            adjacent_multiplier_text=st.one_of(st.none(), st.text(min_size=1, max_size=10)),
            detail_view_id=st.one_of(st.none(), st.sampled_from(list("ABCDEFGHIJ"))),
            effective_multiplier=st.integers(min_value=1, max_value=999),
        ),
        min_size=0,
        max_size=5,
    ),
)
def test_property13_warning_completeness_with_nonempty_report(
    subsystem_warnings: list[list[Warning]],
    tally: dict[str, int],
    breakdowns: list[BalloonBreakdown],
) -> None:
    """Warnings are preserved even when the report has tally data and breakdowns."""
    all_warnings = [w for subsystem in subsystem_warnings for w in subsystem]

    tally_result = TallyResult(
        tally=tally,
        balloon_breakdown=breakdowns,
        excluded_balloon_count=0,
    )

    generator = ReportGenerator()
    json_str = generator.generate_json(tally_result, all_warnings)

    parsed = json.loads(json_str)
    report_warnings = parsed["warnings"]

    # Every original warning must be present
    assert len(report_warnings) == len(all_warnings)

    for i, original in enumerate(all_warnings):
        rw = report_warnings[i]
        assert rw["warning_type"] == original.warning_type.value
        assert rw["message"] == original.message
        assert rw["page_number"] == original.page_number
        assert rw["related_items"] == list(original.related_items)
