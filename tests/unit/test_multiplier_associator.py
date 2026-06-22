"""Unit tests for MultiplierAssociator."""

from balloon_quantity_analyzer.models import (
    AssociatedBalloon,
    BoundingBox,
    DetectedBalloon,
    DetailView,
    ParsedMultiplier,
    Warning,
    WarningType,
)
from balloon_quantity_analyzer.multiplier_associator import MultiplierAssociator


def _balloon(find_number: str, x: float, y: float, page: int = 1) -> DetectedBalloon:
    return DetectedBalloon(
        find_number=find_number,
        page_number=page,
        bounding_box=BoundingBox(x=x, y=y, width=10, height=10),
        confidence=0.9,
    )


def _multiplier(value: int, raw: str, x: float, y: float, page: int = 1) -> ParsedMultiplier:
    return ParsedMultiplier(
        value=value,
        raw_text=raw,
        bounding_box=BoundingBox(x=x, y=y, width=10, height=10),
        page_number=page,
        confidence=0.9,
    )


class TestSingleAssociation:
    """A multiplier near exactly one balloon is associated with it."""

    def test_multiplier_within_radius_associates(self):
        assoc = MultiplierAssociator(proximity_radius=50.0)
        balloons = [_balloon("1", 100, 100)]
        multipliers = [_multiplier(3, "3X", 120, 100)]

        results, warnings = assoc.associate(balloons, multipliers, [])

        assert len(results) == 1
        assert results[0].adjacent_multiplier_value == 3
        assert results[0].adjacent_multiplier_text == "3X"
        assert results[0].effective_multiplier == 3
        assert len(warnings) == 0

    def test_multiplier_outside_radius_not_associated(self):
        assoc = MultiplierAssociator(proximity_radius=10.0)
        balloons = [_balloon("1", 100, 100)]
        multipliers = [_multiplier(3, "3X", 200, 200)]

        results, warnings = assoc.associate(balloons, multipliers, [])

        assert len(results) == 1
        assert results[0].adjacent_multiplier_value == 1
        assert results[0].adjacent_multiplier_text is None
        assert results[0].effective_multiplier == 1

    def test_different_pages_not_associated(self):
        assoc = MultiplierAssociator(proximity_radius=50.0)
        balloons = [_balloon("1", 100, 100, page=1)]
        multipliers = [_multiplier(3, "3X", 105, 100, page=2)]

        results, warnings = assoc.associate(balloons, multipliers, [])

        assert results[0].adjacent_multiplier_value == 1


class TestAmbiguousMultiplier:
    """Multiplier near 2+ balloons → warning, not applied."""

    def test_ambiguous_multiplier_warning(self):
        assoc = MultiplierAssociator(proximity_radius=50.0)
        balloons = [_balloon("1", 100, 100), _balloon("2", 120, 100)]
        multipliers = [_multiplier(3, "3X", 110, 100)]

        results, warnings = assoc.associate(balloons, multipliers, [])

        assert len(results) == 2
        # Neither balloon should get the multiplier
        assert results[0].adjacent_multiplier_value == 1
        assert results[1].adjacent_multiplier_value == 1
        # Warning should be produced
        assert len(warnings) == 1
        assert warnings[0].warning_type == WarningType.AMBIGUOUS_MULTIPLIER


class TestMultipleMultipliers:
    """Multiple multipliers near one balloon → warning, none applied."""

    def test_multiple_multipliers_warning(self):
        assoc = MultiplierAssociator(proximity_radius=50.0)
        balloons = [_balloon("1", 100, 100)]
        multipliers = [
            _multiplier(3, "3X", 110, 100),
            _multiplier(5, "5X", 100, 110),
        ]

        results, warnings = assoc.associate(balloons, multipliers, [])

        assert len(results) == 1
        assert results[0].adjacent_multiplier_value == 1
        assert len(warnings) == 1
        assert warnings[0].warning_type == WarningType.MULTIPLE_MULTIPLIERS


class TestDefaultMultiplier:
    """Balloons with no adjacent or Detail view multiplier get effective = 1."""

    def test_no_multipliers_default_one(self):
        assoc = MultiplierAssociator(proximity_radius=50.0)
        balloons = [_balloon("1", 100, 100)]

        results, warnings = assoc.associate(balloons, [], [])

        assert len(results) == 1
        assert results[0].adjacent_multiplier_value == 1
        assert results[0].detail_view_multiplier == 1
        assert results[0].effective_multiplier == 1
        assert len(warnings) == 0


class TestDetailViewMultiplier:
    """Detail view multipliers combine multiplicatively with adjacent."""

    def test_detail_view_multiplier_only(self):
        assoc = MultiplierAssociator(proximity_radius=50.0)
        balloons = [_balloon("1", 100, 100)]
        detail_views = [
            DetailView(
                identifier="A",
                page_number=1,
                region=BoundingBox(x=50, y=50, width=200, height=200),
                multiplier=4,
                contained_balloon_indices=[0],
            )
        ]

        results, warnings = assoc.associate(balloons, [], detail_views)

        assert results[0].detail_view_multiplier == 4
        assert results[0].detail_view_id == "A"
        assert results[0].effective_multiplier == 4

    def test_combined_adjacent_and_detail_view(self):
        assoc = MultiplierAssociator(proximity_radius=50.0)
        balloons = [_balloon("1", 100, 100)]
        multipliers = [_multiplier(3, "3X", 110, 100)]
        detail_views = [
            DetailView(
                identifier="B",
                page_number=1,
                region=BoundingBox(x=50, y=50, width=200, height=200),
                multiplier=2,
                contained_balloon_indices=[0],
            )
        ]

        results, warnings = assoc.associate(balloons, multipliers, detail_views)

        assert results[0].adjacent_multiplier_value == 3
        assert results[0].detail_view_multiplier == 2
        assert results[0].effective_multiplier == 6  # 3 × 2

    def test_detail_view_no_multiplier_treated_as_one(self):
        assoc = MultiplierAssociator(proximity_radius=50.0)
        balloons = [_balloon("1", 100, 100)]
        multipliers = [_multiplier(3, "3X", 110, 100)]
        detail_views = [
            DetailView(
                identifier="C",
                page_number=1,
                region=BoundingBox(x=50, y=50, width=200, height=200),
                multiplier=None,
                contained_balloon_indices=[0],
            )
        ]

        results, warnings = assoc.associate(balloons, multipliers, detail_views)

        assert results[0].detail_view_multiplier == 1
        assert results[0].effective_multiplier == 3  # 3 × 1


class TestEmptyInputs:
    """Edge cases with empty inputs."""

    def test_no_balloons(self):
        assoc = MultiplierAssociator(proximity_radius=50.0)
        results, warnings = assoc.associate([], [], [])
        assert results == []
        assert warnings == []

    def test_multipliers_but_no_balloons(self):
        assoc = MultiplierAssociator(proximity_radius=50.0)
        multipliers = [_multiplier(3, "3X", 100, 100)]
        results, warnings = assoc.associate([], multipliers, [])
        assert results == []
