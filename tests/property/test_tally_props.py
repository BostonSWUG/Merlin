# Feature: balloon-quantity-analyzer, Property 9: Tally correctness
"""Property-based tests for TallyAggregator.

Tests cover:
- Property 9: Tally correctness (Validates: Requirements 6.1, 6.2, 6.3)

For any list of AssociatedBalloons, the Tally_Aggregator SHALL produce a tally
where:
  (a) the set of keys equals exactly the set of non-empty Find numbers,
  (b) each key's value equals the sum of effective multipliers of all balloons
      bearing that Find number, and
  (c) the excluded_balloon_count equals the number of balloons with an empty
      Find number.
"""

from __future__ import annotations

from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from balloon_quantity_analyzer.models import (
    AssociatedBalloon,
    BoundingBox,
)
from balloon_quantity_analyzer.tally_aggregator import TallyAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_BOX = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)


def _make_associated_balloon(
    find_number: str,
    effective_multiplier: int,
    page: int = 1,
) -> AssociatedBalloon:
    """Create an AssociatedBalloon with the given find_number and effective multiplier."""
    return AssociatedBalloon(
        find_number=find_number,
        page_number=page,
        bounding_box=_DEFAULT_BOX,
        adjacent_multiplier_text=None,
        adjacent_multiplier_value=1,
        detail_view_id=None,
        detail_view_multiplier=1,
        effective_multiplier=effective_multiplier,
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Find numbers: non-empty alphanumeric strings (used for balloons with valid IDs)
_find_numbers = st.text(
    alphabet="0123456789ABCDEF",
    min_size=1,
    max_size=4,
)

# Effective multiplier values (positive integers)
_eff_mult = st.integers(min_value=1, max_value=999)

# Page numbers
_pages = st.integers(min_value=1, max_value=20)

# Strategy for a single AssociatedBalloon with a non-empty find number
_balloon_with_find = st.builds(
    _make_associated_balloon,
    find_number=_find_numbers,
    effective_multiplier=_eff_mult,
    page=_pages,
)

# Strategy for a single AssociatedBalloon with an empty find number (excluded)
_balloon_empty_find = st.builds(
    _make_associated_balloon,
    find_number=st.just(""),
    effective_multiplier=_eff_mult,
    page=_pages,
)

# Strategy for a mixed list of balloons (some with find numbers, some empty)
_balloon_list = st.lists(
    st.one_of(_balloon_with_find, _balloon_empty_find),
    min_size=0,
    max_size=50,
)


# ===========================================================================
# Property 9: Tally correctness
# **Validates: Requirements 6.1, 6.2, 6.3**
# ===========================================================================


@settings(max_examples=200)
@given(balloons=_balloon_list)
def test_property9_tally_key_set_equals_nonempty_find_numbers(
    balloons: list[AssociatedBalloon],
) -> None:
    """(a) The set of tally keys equals exactly the set of non-empty Find numbers."""
    aggregator = TallyAggregator()
    result = aggregator.aggregate(balloons)

    expected_keys = {b.find_number for b in balloons if b.find_number != ""}
    assert set(result.tally.keys()) == expected_keys


@settings(max_examples=200)
@given(balloons=_balloon_list)
def test_property9_tally_values_equal_sum_of_effective_multipliers(
    balloons: list[AssociatedBalloon],
) -> None:
    """(b) Each tally value equals the sum of effective multipliers for that Find number."""
    aggregator = TallyAggregator()
    result = aggregator.aggregate(balloons)

    # Compute expected sums manually
    expected: dict[str, int] = {}
    for b in balloons:
        if b.find_number != "":
            expected[b.find_number] = expected.get(b.find_number, 0) + b.effective_multiplier

    assert result.tally == expected


@settings(max_examples=200)
@given(balloons=_balloon_list)
def test_property9_excluded_count_equals_empty_find_number_balloons(
    balloons: list[AssociatedBalloon],
) -> None:
    """(c) excluded_balloon_count equals the number of balloons with empty Find number."""
    aggregator = TallyAggregator()
    result = aggregator.aggregate(balloons)

    expected_excluded = sum(1 for b in balloons if b.find_number == "")
    assert result.excluded_balloon_count == expected_excluded
