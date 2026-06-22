"""Unit tests for ReportGenerator.

Tests cover:
- JSON output validity and schema conformance (Requirements 7.1, 7.2, 7.3)
- Tabular output format (Requirement 7.4)
- Empty warnings case (Requirement 8.4)
"""

from __future__ import annotations

import json

from balloon_quantity_analyzer.models import (
    BalloonBreakdown,
    BoundingBox,
    TallyResult,
    Warning,
    WarningType,
)
from balloon_quantity_analyzer.report_generator import ReportGenerator, parse_json_report


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_BOX = BoundingBox(x=10.0, y=20.0, width=30.0, height=40.0)


def _sample_tally_result() -> TallyResult:
    return TallyResult(
        tally={"1": 3, "2A": 5},
        balloon_breakdown=[
            BalloonBreakdown(
                find_number="1",
                page_number=1,
                bounding_box=_BOX,
                adjacent_multiplier_text="3X",
                detail_view_id=None,
                effective_multiplier=3,
            ),
            BalloonBreakdown(
                find_number="2A",
                page_number=2,
                bounding_box=BoundingBox(x=50.0, y=60.0, width=15.0, height=15.0),
                adjacent_multiplier_text=None,
                detail_view_id="A",
                effective_multiplier=5,
            ),
        ],
        excluded_balloon_count=1,
    )


def _sample_warnings() -> list[Warning]:
    return [
        Warning(
            warning_type=WarningType.AMBIGUOUS_MULTIPLIER,
            message="Multiplier near multiple balloons",
            page_number=1,
            related_items=["1", "2A"],
        ),
        Warning(
            warning_type=WarningType.LOW_CONFIDENCE,
            message="Low confidence detection",
            page_number=2,
            related_items=["2A"],
        ),
    ]


# ===========================================================================
# JSON output validity and schema conformance (Requirements 7.1, 7.2, 7.3)
# ===========================================================================


class TestGenerateJson:
    def test_json_is_valid(self) -> None:
        """generate_json produces valid JSON."""
        gen = ReportGenerator()
        result = gen.generate_json(_sample_tally_result(), _sample_warnings())
        parsed = json.loads(result)  # should not raise
        assert isinstance(parsed, dict)

    def test_json_has_required_top_level_keys(self) -> None:
        """JSON report contains tally, balloon_breakdown, excluded_balloon_count, warnings."""
        gen = ReportGenerator()
        result = json.loads(gen.generate_json(_sample_tally_result(), _sample_warnings()))
        assert "tally" in result
        assert "balloon_breakdown" in result
        assert "excluded_balloon_count" in result
        assert "warnings" in result

    def test_json_tally_values(self) -> None:
        """Tally section maps find numbers to correct counts."""
        gen = ReportGenerator()
        result = json.loads(gen.generate_json(_sample_tally_result(), []))
        assert result["tally"] == {"1": 3, "2A": 5}

    def test_json_balloon_breakdown_structure(self) -> None:
        """Each balloon breakdown entry has the expected fields."""
        gen = ReportGenerator()
        result = json.loads(gen.generate_json(_sample_tally_result(), []))
        breakdown = result["balloon_breakdown"]
        assert len(breakdown) == 2

        entry = breakdown[0]
        assert entry["find_number"] == "1"
        assert entry["page_number"] == 1
        assert entry["effective_multiplier"] == 3
        assert entry["adjacent_multiplier_text"] == "3X"
        assert entry["detail_view_id"] is None
        assert "bounding_box" in entry
        bb = entry["bounding_box"]
        assert bb == {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}

    def test_json_excluded_balloon_count(self) -> None:
        """excluded_balloon_count is serialized correctly."""
        gen = ReportGenerator()
        result = json.loads(gen.generate_json(_sample_tally_result(), []))
        assert result["excluded_balloon_count"] == 1

    def test_json_warnings_structure(self) -> None:
        """Warnings are serialized with correct fields."""
        gen = ReportGenerator()
        warnings = _sample_warnings()
        result = json.loads(gen.generate_json(_sample_tally_result(), warnings))
        assert len(result["warnings"]) == 2

        w = result["warnings"][0]
        assert w["warning_type"] == "ambiguous_multiplier"
        assert w["message"] == "Multiplier near multiple balloons"
        assert w["page_number"] == 1
        assert w["related_items"] == ["1", "2A"]

    def test_json_round_trip(self) -> None:
        """Serialize → parse → compare produces equivalent data."""
        gen = ReportGenerator()
        tally_result = _sample_tally_result()
        warnings = _sample_warnings()
        json_str = gen.generate_json(tally_result, warnings)
        parsed = parse_json_report(json_str)

        assert parsed.tally == tally_result.tally
        assert parsed.balloon_breakdown == tally_result.balloon_breakdown
        assert parsed.excluded_balloon_count == tally_result.excluded_balloon_count
        assert parsed.warnings == warnings


# ===========================================================================
# Empty warnings case (Requirement 8.4)
# ===========================================================================


class TestEmptyWarnings:
    def test_json_empty_warnings_list(self) -> None:
        """When no warnings, JSON report contains an empty warnings list."""
        gen = ReportGenerator()
        result = json.loads(gen.generate_json(_sample_tally_result(), []))
        assert result["warnings"] == []

    def test_tabular_empty_warnings_shows_none(self) -> None:
        """When no warnings, tabular report shows '(none)' in warnings section."""
        gen = ReportGenerator()
        output = gen.generate_tabular(_sample_tally_result(), [])
        assert "(none)" in output


# ===========================================================================
# Tabular output format (Requirement 7.4)
# ===========================================================================


class TestGenerateTabular:
    def test_tabular_contains_tally_section(self) -> None:
        """Tabular output has a TALLY section with find numbers and counts."""
        gen = ReportGenerator()
        output = gen.generate_tabular(_sample_tally_result(), [])
        assert "TALLY" in output
        assert "1" in output
        assert "2A" in output

    def test_tabular_contains_balloon_breakdown_section(self) -> None:
        """Tabular output has a BALLOON BREAKDOWN section."""
        gen = ReportGenerator()
        output = gen.generate_tabular(_sample_tally_result(), [])
        assert "BALLOON BREAKDOWN" in output
        assert "Find=1" in output
        assert "Find=2A" in output

    def test_tabular_contains_warnings_section(self) -> None:
        """Tabular output has a WARNINGS section with warning details."""
        gen = ReportGenerator()
        output = gen.generate_tabular(_sample_tally_result(), _sample_warnings())
        assert "WARNINGS" in output
        assert "ambiguous_multiplier" in output
        assert "low_confidence" in output

    def test_tabular_shows_excluded_count(self) -> None:
        """Tabular output shows excluded balloon count."""
        gen = ReportGenerator()
        output = gen.generate_tabular(_sample_tally_result(), [])
        assert "Excluded balloons (unreadable): 1" in output

    def test_tabular_empty_tally(self) -> None:
        """Tabular output handles empty tally gracefully."""
        gen = ReportGenerator()
        empty_tally = TallyResult(tally={}, balloon_breakdown=[], excluded_balloon_count=0)
        output = gen.generate_tabular(empty_tally, [])
        assert "(no items)" in output
        assert "(no balloons)" in output

    def test_tabular_warning_page_none(self) -> None:
        """Tabular output shows N/A for warnings with no page number."""
        gen = ReportGenerator()
        warning = Warning(
            warning_type=WarningType.INVALID_CONFIGURATION,
            message="Bad config",
            page_number=None,
        )
        output = gen.generate_tabular(_sample_tally_result(), [warning])
        assert "N/A" in output
