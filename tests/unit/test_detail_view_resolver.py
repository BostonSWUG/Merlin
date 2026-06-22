"""Unit tests for DetailViewResolver and parse_detail_label."""

from __future__ import annotations

import pytest

from balloon_quantity_analyzer.detail_view_resolver import (
    DetailViewResolver,
    parse_detail_label,
)
from balloon_quantity_analyzer.models import (
    BoundingBox,
    DetectedBalloon,
    NormalizedPage,
    TextRegion,
    WarningType,
)


# -----------------------------------------------------------------------
# parse_detail_label tests
# -----------------------------------------------------------------------


class TestParseDetailLabel:
    """Tests for the standalone parse_detail_label function."""

    def test_simple_detail_label(self) -> None:
        assert parse_detail_label("DETAIL A") == ("A", None)

    def test_detail_with_places_multiplier(self) -> None:
        assert parse_detail_label("DETAIL B (3 PLACES)") == ("B", 3)

    def test_detail_with_plcs(self) -> None:
        assert parse_detail_label("DETAIL C (2 PLCS)") == ("C", 2)

    def test_detail_with_pl(self) -> None:
        assert parse_detail_label("DETAIL D (4 PL)") == ("D", 4)

    def test_case_insensitive(self) -> None:
        assert parse_detail_label("detail e") == ("E", None)
        assert parse_detail_label("Detail F (5 places)") == ("F", 5)

    def test_extra_whitespace(self) -> None:
        assert parse_detail_label("  DETAIL   G  ") == ("G", None)
        assert parse_detail_label("DETAIL H  ( 2  PLACES )") == ("H", 2)

    def test_no_parentheses_multiplier(self) -> None:
        assert parse_detail_label("DETAIL J 7 PLACES") == ("J", 7)

    def test_non_matching_text(self) -> None:
        assert parse_detail_label("hello world") is None
        assert parse_detail_label("SECTION A") is None
        assert parse_detail_label("3X") is None
        assert parse_detail_label("") is None

    def test_identifier_is_single_letter(self) -> None:
        # Only single uppercase letters are valid identifiers
        assert parse_detail_label("DETAIL Z") == ("Z", None)


# -----------------------------------------------------------------------
# Helper to build test pages
# -----------------------------------------------------------------------


def _make_page(
    text_regions: list[TextRegion],
    page_number: int = 1,
) -> NormalizedPage:
    return NormalizedPage(
        page_number=page_number,
        image=b"",
        text_regions=text_regions,
    )


def _make_balloon(
    find_number: str,
    x: float,
    y: float,
    page_number: int = 1,
) -> DetectedBalloon:
    return DetectedBalloon(
        find_number=find_number,
        page_number=page_number,
        bounding_box=BoundingBox(x=x, y=y, width=20, height=20),
        confidence=0.9,
    )


def _make_text_region(
    text: str,
    x: float,
    y: float,
) -> TextRegion:
    return TextRegion(
        text=text,
        bounding_box=BoundingBox(x=x, y=y, width=60, height=20),
        confidence=0.95,
    )


# -----------------------------------------------------------------------
# DetailViewResolver tests
# -----------------------------------------------------------------------


class TestDetailViewResolver:
    """Tests for the DetailViewResolver.resolve method."""

    def test_no_detail_views(self) -> None:
        page = _make_page([_make_text_region("3X", 100, 100)])
        balloons = [_make_balloon("1", 110, 110)]
        resolver = DetailViewResolver()
        dvs, warnings = resolver.resolve(page, balloons)
        assert dvs == []
        assert warnings == []

    def test_single_detail_view_no_multiplier(self) -> None:
        page = _make_page([_make_text_region("DETAIL A", 100, 100)])
        # Balloon center at (110, 110) — within default 200px padding of label center
        balloons = [_make_balloon("5", 110, 110)]
        resolver = DetailViewResolver()
        dvs, warnings = resolver.resolve(page, balloons)

        assert len(dvs) == 1
        assert dvs[0].identifier == "A"
        assert dvs[0].multiplier is None
        assert 0 in dvs[0].contained_balloon_indices
        assert warnings == []

    def test_single_detail_view_with_multiplier(self) -> None:
        page = _make_page([_make_text_region("DETAIL B (3 PLACES)", 100, 100)])
        balloons = [_make_balloon("7", 110, 110)]
        resolver = DetailViewResolver()
        dvs, warnings = resolver.resolve(page, balloons)

        assert len(dvs) == 1
        assert dvs[0].identifier == "B"
        assert dvs[0].multiplier == 3
        assert 0 in dvs[0].contained_balloon_indices
        assert warnings == []

    def test_balloon_outside_detail_view(self) -> None:
        page = _make_page([_make_text_region("DETAIL C", 100, 100)])
        # Balloon far away from the label — outside the inferred region
        balloons = [_make_balloon("2", 900, 900)]
        resolver = DetailViewResolver(region_padding=50.0)
        dvs, warnings = resolver.resolve(page, balloons)

        assert len(dvs) == 1
        assert dvs[0].contained_balloon_indices == []

    def test_overlapping_detail_views_with_multipliers_warns(self) -> None:
        """A balloon inside 2+ Detail views that each declare a multiplier triggers a warning."""
        page = _make_page([
            _make_text_region("DETAIL A (2 PLACES)", 100, 100),
            _make_text_region("DETAIL B (3 PLACES)", 120, 100),
        ])
        # Balloon near both labels
        balloons = [_make_balloon("10", 110, 110)]
        resolver = DetailViewResolver(region_padding=200.0)
        dvs, warnings = resolver.resolve(page, balloons)

        assert len(dvs) == 2
        assert len(warnings) == 1
        assert warnings[0].warning_type == WarningType.OVERLAPPING_DETAIL_VIEWS
        assert "10" in warnings[0].message

    def test_overlapping_detail_views_without_multipliers_no_warning(self) -> None:
        """Overlapping Detail views without multipliers should not warn."""
        page = _make_page([
            _make_text_region("DETAIL A", 100, 100),
            _make_text_region("DETAIL B", 120, 100),
        ])
        balloons = [_make_balloon("10", 110, 110)]
        resolver = DetailViewResolver(region_padding=200.0)
        dvs, warnings = resolver.resolve(page, balloons)

        assert len(dvs) == 2
        assert warnings == []

    def test_balloon_on_different_page_not_contained(self) -> None:
        page = _make_page([_make_text_region("DETAIL A", 100, 100)], page_number=1)
        balloons = [_make_balloon("3", 110, 110, page_number=2)]
        resolver = DetailViewResolver()
        dvs, warnings = resolver.resolve(page, balloons)

        assert len(dvs) == 1
        assert dvs[0].contained_balloon_indices == []

    def test_multiple_balloons_containment(self) -> None:
        page = _make_page([_make_text_region("DETAIL D (2 PL)", 200, 200)])
        balloons = [
            _make_balloon("1", 210, 210),  # inside
            _make_balloon("2", 220, 220),  # inside
            _make_balloon("3", 900, 900),  # outside
        ]
        resolver = DetailViewResolver(region_padding=100.0)
        dvs, warnings = resolver.resolve(page, balloons)

        assert len(dvs) == 1
        assert 0 in dvs[0].contained_balloon_indices
        assert 1 in dvs[0].contained_balloon_indices
        assert 2 not in dvs[0].contained_balloon_indices
