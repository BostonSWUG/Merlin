"""Unit tests for TallyAggregator.

Requirements: 6.1, 6.2, 6.3, 6.4
"""

from balloon_quantity_analyzer.models import (
    AssociatedBalloon,
    BalloonBreakdown,
    BoundingBox,
    TallyResult,
)
from balloon_quantity_analyzer.tally_aggregator import TallyAggregator


_BOX = BoundingBox(x=10.0, y=20.0, width=30.0, height=30.0)
_BOX2 = BoundingBox(x=100.0, y=200.0, width=30.0, height=30.0)


def _ab(
    find_number: str,
    effective_multiplier: int = 1,
    page: int = 1,
    box: BoundingBox = _BOX,
    adj_text: str | None = None,
    adj_value: int = 1,
    dv_id: str | None = None,
    dv_mult: int = 1,
) -> AssociatedBalloon:
    return AssociatedBalloon(
        find_number=find_number,
        page_number=page,
        bounding_box=box,
        adjacent_multiplier_text=adj_text,
        adjacent_multiplier_value=adj_value,
        detail_view_id=dv_id,
        detail_view_multiplier=dv_mult,
        effective_multiplier=effective_multiplier,
        confidence=0.9,
    )


class TestTallySums:
    """Requirement 6.1 — sum effective multipliers per Find number."""

    def test_single_balloon(self):
        agg = TallyAggregator()
        result = agg.aggregate([_ab("1", effective_multiplier=3)])

        assert result.tally == {"1": 3}

    def test_same_find_number_sums(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("1", effective_multiplier=2),
            _ab("1", effective_multiplier=5),
        ])

        assert result.tally == {"1": 7}

    def test_multiple_find_numbers(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("1", effective_multiplier=2),
            _ab("2", effective_multiplier=3),
            _ab("1", effective_multiplier=4),
            _ab("3", effective_multiplier=1),
        ])

        assert result.tally == {"1": 6, "2": 3, "3": 1}

    def test_alphanumeric_find_numbers(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("1A", effective_multiplier=2),
            _ab("1A", effective_multiplier=3),
            _ab("2B", effective_multiplier=1),
        ])

        assert result.tally == {"1A": 5, "2B": 1}

    def test_across_multiple_pages(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("5", effective_multiplier=2, page=1),
            _ab("5", effective_multiplier=3, page=2),
            _ab("5", effective_multiplier=1, page=3),
        ])

        assert result.tally == {"5": 6}


class TestEmptyFindNumberExclusion:
    """Requirement 6.3 — exclude balloons with empty Find number."""

    def test_empty_find_number_excluded_from_tally(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("1", effective_multiplier=2),
            _ab("", effective_multiplier=3),
        ])

        assert "" not in result.tally
        assert result.tally == {"1": 2}
        assert result.excluded_balloon_count == 1

    def test_all_empty_find_numbers(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("", effective_multiplier=1),
            _ab("", effective_multiplier=2),
            _ab("", effective_multiplier=3),
        ])

        assert result.tally == {}
        assert result.excluded_balloon_count == 3

    def test_no_empty_find_numbers(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("1", effective_multiplier=1),
            _ab("2", effective_multiplier=2),
        ])

        assert result.excluded_balloon_count == 0

    def test_mixed_empty_and_valid(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("", effective_multiplier=5),
            _ab("10", effective_multiplier=2),
            _ab("", effective_multiplier=1),
            _ab("10", effective_multiplier=3),
            _ab("20", effective_multiplier=1),
        ])

        assert result.tally == {"10": 5, "20": 1}
        assert result.excluded_balloon_count == 2


class TestBalloonBreakdown:
    """Requirement 6.4 — per-balloon breakdown."""

    def test_breakdown_contains_all_valid_balloons(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("1", effective_multiplier=2),
            _ab("2", effective_multiplier=3),
        ])

        assert len(result.balloon_breakdown) == 2

    def test_breakdown_excludes_empty_find_numbers(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("1", effective_multiplier=2),
            _ab("", effective_multiplier=3),
        ])

        assert len(result.balloon_breakdown) == 1
        assert result.balloon_breakdown[0].find_number == "1"

    def test_breakdown_preserves_fields(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab(
                "7",
                effective_multiplier=6,
                page=2,
                box=_BOX2,
                adj_text="3X",
                dv_id="A",
            ),
        ])

        bd = result.balloon_breakdown[0]
        assert bd.find_number == "7"
        assert bd.page_number == 2
        assert bd.bounding_box == _BOX2
        assert bd.adjacent_multiplier_text == "3X"
        assert bd.detail_view_id == "A"
        assert bd.effective_multiplier == 6

    def test_breakdown_order_matches_input(self):
        agg = TallyAggregator()
        result = agg.aggregate([
            _ab("3", effective_multiplier=1),
            _ab("1", effective_multiplier=2),
            _ab("2", effective_multiplier=3),
        ])

        find_numbers = [bd.find_number for bd in result.balloon_breakdown]
        assert find_numbers == ["3", "1", "2"]


class TestEmptyInput:
    """Edge case — no balloons at all."""

    def test_empty_list(self):
        agg = TallyAggregator()
        result = agg.aggregate([])

        assert result.tally == {}
        assert result.balloon_breakdown == []
        assert result.excluded_balloon_count == 0
