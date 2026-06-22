# Feature: balloon-quantity-analyzer, Property 10: JSON report round-trip
"""Property-based tests for ReportGenerator JSON round-trip.

Tests cover:
- Property 10: JSON report round-trip (Validates: Requirements 7.3)

For any valid AnalysisReport, serializing it to JSON via the Report_Generator
and then parsing the resulting JSON string back into an AnalysisReport SHALL
produce an object equivalent to the original.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from balloon_quantity_analyzer.models import (
    AnalysisReport,
    BalloonBreakdown,
    BoundingBox,
    TallyResult,
    Warning,
    WarningType,
)
from balloon_quantity_analyzer.report_generator import (
    ReportGenerator,
    parse_json_report,
)


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

_balloon_breakdowns = st.builds(
    BalloonBreakdown,
    find_number=_find_numbers,
    page_number=st.integers(min_value=1, max_value=20),
    bounding_box=_bounding_boxes,
    adjacent_multiplier_text=st.one_of(st.none(), st.text(min_size=1, max_size=10)),
    detail_view_id=st.one_of(st.none(), st.sampled_from(list("ABCDEFGHIJ"))),
    effective_multiplier=st.integers(min_value=1, max_value=999),
)

_warning_types = st.sampled_from(list(WarningType))

_warnings = st.builds(
    Warning,
    warning_type=_warning_types,
    message=st.text(min_size=1, max_size=80),
    page_number=st.one_of(st.none(), st.integers(min_value=1, max_value=20)),
    related_items=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5),
)

# Build a tally dict: mapping of find_number -> positive int count
_tally_dicts = st.dictionaries(
    keys=_find_numbers,
    values=st.integers(min_value=1, max_value=9999),
    min_size=0,
    max_size=10,
)

_analysis_reports = st.builds(
    AnalysisReport,
    tally=_tally_dicts,
    balloon_breakdown=st.lists(_balloon_breakdowns, min_size=0, max_size=15),
    excluded_balloon_count=st.integers(min_value=0, max_value=100),
    warnings=st.lists(_warnings, min_size=0, max_size=10),
)


# ===========================================================================
# Property 10: JSON report round-trip
# **Validates: Requirements 7.3**
# ===========================================================================


@settings(max_examples=200)
@given(report=_analysis_reports)
def test_property10_json_round_trip_produces_equivalent_report(
    report: AnalysisReport,
) -> None:
    """Serialize → parse → re-serialize produces an equivalent AnalysisReport."""
    generator = ReportGenerator()

    # Build a TallyResult from the report fields for serialization
    tally_result = TallyResult(
        tally=report.tally,
        balloon_breakdown=report.balloon_breakdown,
        excluded_balloon_count=report.excluded_balloon_count,
    )

    # First serialization
    json_str_1 = generator.generate_json(tally_result, list(report.warnings))

    # Parse back
    parsed_report = parse_json_report(json_str_1)

    # Verify equivalence of all fields
    assert parsed_report.tally == report.tally
    assert parsed_report.balloon_breakdown == report.balloon_breakdown
    assert parsed_report.excluded_balloon_count == report.excluded_balloon_count
    assert parsed_report.warnings == report.warnings


@settings(max_examples=200)
@given(report=_analysis_reports)
def test_property10_double_round_trip_stable(
    report: AnalysisReport,
) -> None:
    """Serialize → parse → re-serialize → parse produces the same result as single round-trip."""
    generator = ReportGenerator()

    tally_result = TallyResult(
        tally=report.tally,
        balloon_breakdown=report.balloon_breakdown,
        excluded_balloon_count=report.excluded_balloon_count,
    )

    # First round-trip
    json_str_1 = generator.generate_json(tally_result, list(report.warnings))
    parsed_1 = parse_json_report(json_str_1)

    # Second serialization from parsed result
    tally_result_2 = TallyResult(
        tally=parsed_1.tally,
        balloon_breakdown=list(parsed_1.balloon_breakdown),
        excluded_balloon_count=parsed_1.excluded_balloon_count,
    )
    json_str_2 = generator.generate_json(tally_result_2, list(parsed_1.warnings))

    # Second parse
    parsed_2 = parse_json_report(json_str_2)

    # Both parsed results should be equivalent
    assert parsed_1.tally == parsed_2.tally
    assert parsed_1.balloon_breakdown == parsed_2.balloon_breakdown
    assert parsed_1.excluded_balloon_count == parsed_2.excluded_balloon_count
    assert parsed_1.warnings == parsed_2.warnings
