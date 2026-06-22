# Feature: balloon-quantity-analyzer, Property 4: Proximity association correctness
# Feature: balloon-quantity-analyzer, Property 5: Ambiguous multiplier produces warning
# Feature: balloon-quantity-analyzer, Property 6: Multiple multipliers on one balloon produces warning
# Feature: balloon-quantity-analyzer, Property 3: Effective multiplier computation
"""Property-based tests for MultiplierAssociator.

Tests cover:
- Property 4: Proximity association correctness (Validates: Requirements 4.1)
- Property 5: Ambiguous multiplier produces warning (Validates: Requirements 4.4)
- Property 6: Multiple multipliers on one balloon produces warning (Validates: Requirements 4.5)
- Property 3: Effective multiplier computation (Validates: Requirements 4.3, 5.3, 5.4)
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from balloon_quantity_analyzer.models import (
    BoundingBox,
    DetectedBalloon,
    DetailView,
    ParsedMultiplier,
    WarningType,
)
from balloon_quantity_analyzer.multiplier_associator import MultiplierAssociator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOX_SIZE = 10.0


def _make_balloon(
    find_number: str, x: float, y: float, page: int = 1
) -> DetectedBalloon:
    """Create a DetectedBalloon centred at (x + BOX_SIZE/2, y + BOX_SIZE/2)."""
    return DetectedBalloon(
        find_number=find_number,
        page_number=page,
        bounding_box=BoundingBox(x=x, y=y, width=_BOX_SIZE, height=_BOX_SIZE),
        confidence=0.9,
    )


def _make_multiplier(
    value: int, x: float, y: float, page: int = 1
) -> ParsedMultiplier:
    """Create a ParsedMultiplier centred at (x + BOX_SIZE/2, y + BOX_SIZE/2)."""
    return ParsedMultiplier(
        value=value,
        raw_text=f"{value}X",
        bounding_box=BoundingBox(x=x, y=y, width=_BOX_SIZE, height=_BOX_SIZE),
        page_number=page,
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Reasonable coordinate range for positions on a drawing page
_coords = st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False)

# Positive multiplier values
_pos_ints = st.integers(min_value=1, max_value=999)

# Proximity radius
_radius = st.floats(min_value=20.0, max_value=500.0, allow_nan=False, allow_infinity=False)

# Find numbers for balloons
_find_numbers = st.text(
    alphabet="0123456789ABCDEF",
    min_size=1,
    max_size=4,
)



# ===========================================================================
# Property 4: Proximity association correctness
# **Validates: Requirements 4.1**
#
# For any set of balloons and multipliers on a page, when a multiplier's
# bounding box center is within the configured proximity radius of exactly
# one balloon's bounding box center, the MultiplierAssociator SHALL
# associate that multiplier with that balloon and no other.
# ===========================================================================


@settings(max_examples=100)
@given(
    bx=_coords,
    by=_coords,
    radius=_radius,
    mult_value=_pos_ints,
    find_number=_find_numbers,
    offset_frac=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
)
def test_property4_proximity_association_correctness(
    bx: float,
    by: float,
    radius: float,
    mult_value: int,
    find_number: str,
    offset_frac: float,
) -> None:
    """A multiplier within radius of exactly one balloon is associated with it.

    We place one balloon and one multiplier close together (within radius),
    and a second balloon far away (well outside radius). The multiplier
    should be associated only with the nearby balloon.
    """
    # Place the multiplier within radius of the first balloon
    # offset_frac in [0, 0.9] ensures the multiplier center is within radius
    offset = offset_frac * radius
    mx = bx + offset

    # Place a second balloon far away — at least 3× radius from the multiplier
    far_x = bx + 4 * radius + 100.0

    balloon_near = _make_balloon(find_number, bx, by, page=1)
    balloon_far = _make_balloon("FAR", far_x, by, page=1)
    multiplier = _make_multiplier(mult_value, mx, by, page=1)

    assoc = MultiplierAssociator(proximity_radius=radius)
    results, warnings = assoc.associate(
        [balloon_near, balloon_far], [multiplier], []
    )

    assert len(results) == 2

    # The near balloon should have the multiplier associated
    near_result = results[0]
    assert near_result.adjacent_multiplier_value == mult_value, (
        f"Expected near balloon to have multiplier {mult_value}, "
        f"got {near_result.adjacent_multiplier_value}"
    )

    # The far balloon should NOT have the multiplier
    far_result = results[1]
    assert far_result.adjacent_multiplier_value == 1, (
        f"Expected far balloon to have default multiplier 1, "
        f"got {far_result.adjacent_multiplier_value}"
    )

    # No ambiguity warnings should be produced
    ambiguous_warnings = [
        w for w in warnings if w.warning_type == WarningType.AMBIGUOUS_MULTIPLIER
    ]
    assert len(ambiguous_warnings) == 0, (
        f"No ambiguous warnings expected, got {ambiguous_warnings}"
    )



# ===========================================================================
# Property 5: Ambiguous multiplier produces warning
# **Validates: Requirements 4.4**
#
# For any multiplier whose bounding box center is within the configured
# proximity radius of two or more balloons, the MultiplierAssociator SHALL
# record a warning identifying the ambiguous multiplier and the candidate
# balloons, and SHALL NOT apply the multiplier to any balloon.
# ===========================================================================


@settings(max_examples=100)
@given(
    cx=_coords,
    cy=_coords,
    radius=_radius,
    mult_value=_pos_ints,
    spread_frac=st.floats(min_value=0.05, max_value=0.4, allow_nan=False, allow_infinity=False),
)
def test_property5_ambiguous_multiplier_warning(
    cx: float,
    cy: float,
    radius: float,
    mult_value: int,
    spread_frac: float,
) -> None:
    """A multiplier near 2+ balloons produces a warning and is not applied.

    We place a multiplier at (cx, cy) and two balloons both within radius
    of the multiplier. The multiplier should not be applied to either balloon,
    and an AMBIGUOUS_MULTIPLIER warning should be produced.
    """
    # Place two balloons on opposite sides of the multiplier, both within radius
    spread = spread_frac * radius  # small fraction of radius
    balloon_a = _make_balloon("A1", cx - spread, cy, page=1)
    balloon_b = _make_balloon("B2", cx + spread, cy, page=1)
    multiplier = _make_multiplier(mult_value, cx, cy, page=1)

    assoc = MultiplierAssociator(proximity_radius=radius)
    results, warnings = assoc.associate(
        [balloon_a, balloon_b], [multiplier], []
    )

    assert len(results) == 2

    # Neither balloon should have the multiplier applied
    assert results[0].adjacent_multiplier_value == 1, (
        f"Balloon A should not have multiplier, got {results[0].adjacent_multiplier_value}"
    )
    assert results[1].adjacent_multiplier_value == 1, (
        f"Balloon B should not have multiplier, got {results[1].adjacent_multiplier_value}"
    )

    # An AMBIGUOUS_MULTIPLIER warning must be produced
    ambiguous_warnings = [
        w for w in warnings if w.warning_type == WarningType.AMBIGUOUS_MULTIPLIER
    ]
    assert len(ambiguous_warnings) >= 1, (
        f"Expected AMBIGUOUS_MULTIPLIER warning, got warnings: {warnings}"
    )



# ===========================================================================
# Property 6: Multiple multipliers on one balloon produces warning
# **Validates: Requirements 4.5**
#
# For any balloon that has two or more multipliers within the configured
# proximity radius, the MultiplierAssociator SHALL record a warning
# identifying the balloon and the competing multipliers, and SHALL NOT
# combine them into an effective multiplier.
# ===========================================================================


@settings(max_examples=100)
@given(
    bx=_coords,
    by=_coords,
    radius=_radius,
    val_a=_pos_ints,
    val_b=_pos_ints,
    find_number=_find_numbers,
    offset_frac=st.floats(min_value=0.05, max_value=0.4, allow_nan=False, allow_infinity=False),
)
def test_property6_multiple_multipliers_warning(
    bx: float,
    by: float,
    radius: float,
    val_a: int,
    val_b: int,
    find_number: str,
    offset_frac: float,
) -> None:
    """2+ multipliers near one balloon produces a warning; none are applied.

    We place one balloon and two multipliers both within radius of that
    balloon, but far from each other's "other" balloons (there are none).
    The balloon should get effective_multiplier == 1 and a
    MULTIPLE_MULTIPLIERS warning should be produced.
    """
    offset = offset_frac * radius
    balloon = _make_balloon(find_number, bx, by, page=1)
    mult_a = _make_multiplier(val_a, bx + offset, by, page=1)
    mult_b = _make_multiplier(val_b, bx - offset, by, page=1)

    # Ensure no other balloon is nearby — only the single balloon
    assoc = MultiplierAssociator(proximity_radius=radius)
    results, warnings = assoc.associate([balloon], [mult_a, mult_b], [])

    assert len(results) == 1

    # The balloon should NOT have either multiplier applied
    assert results[0].adjacent_multiplier_value == 1, (
        f"Balloon should have default multiplier 1 when multiple are nearby, "
        f"got {results[0].adjacent_multiplier_value}"
    )
    assert results[0].effective_multiplier == 1, (
        f"Effective multiplier should be 1, got {results[0].effective_multiplier}"
    )

    # A MULTIPLE_MULTIPLIERS warning must be produced
    multi_warnings = [
        w for w in warnings if w.warning_type == WarningType.MULTIPLE_MULTIPLIERS
    ]
    assert len(multi_warnings) >= 1, (
        f"Expected MULTIPLE_MULTIPLIERS warning, got warnings: {warnings}"
    )



# ===========================================================================
# Property 3: Effective multiplier computation
# **Validates: Requirements 4.3, 5.3, 5.4**
#
# For any balloon with an adjacent multiplier value `a` (defaulting to 1 if
# absent) and an enclosing Detail view multiplier value `d` (defaulting to
# 1 if absent or if the Detail view declares no multiplier), the effective
# multiplier SHALL equal `a × d`.
# ===========================================================================


@settings(max_examples=100)
@given(
    adj_value=_pos_ints,
    dv_value=st.one_of(st.none(), _pos_ints),
    find_number=_find_numbers,
)
def test_property3_effective_multiplier_with_both(
    adj_value: int,
    dv_value: int | None,
    find_number: str,
) -> None:
    """effective_multiplier == adjacent × detail_view (both present).

    Place one balloon with one adjacent multiplier within radius, and one
    Detail view containing the balloon. Verify effective = a × d.
    """
    radius = 100.0
    bx, by = 500.0, 500.0

    balloon = _make_balloon(find_number, bx, by, page=1)
    multiplier = _make_multiplier(adj_value, bx + 20.0, by, page=1)

    detail_views = [
        DetailView(
            identifier="A",
            page_number=1,
            region=BoundingBox(x=400.0, y=400.0, width=300.0, height=300.0),
            multiplier=dv_value,
            contained_balloon_indices=[0],
        )
    ]

    assoc = MultiplierAssociator(proximity_radius=radius)
    results, _ = assoc.associate([balloon], [multiplier], detail_views)

    assert len(results) == 1
    result = results[0]

    expected_dv = dv_value if dv_value is not None else 1
    expected_effective = adj_value * expected_dv

    assert result.adjacent_multiplier_value == adj_value, (
        f"Expected adjacent={adj_value}, got {result.adjacent_multiplier_value}"
    )
    assert result.detail_view_multiplier == expected_dv, (
        f"Expected detail_view={expected_dv}, got {result.detail_view_multiplier}"
    )
    assert result.effective_multiplier == expected_effective, (
        f"Expected effective={expected_effective}, got {result.effective_multiplier}"
    )


@settings(max_examples=100)
@given(
    adj_value=_pos_ints,
    find_number=_find_numbers,
)
def test_property3_effective_multiplier_adjacent_only(
    adj_value: int,
    find_number: str,
) -> None:
    """effective_multiplier == adjacent × 1 when no Detail view is present."""
    radius = 100.0
    bx, by = 500.0, 500.0

    balloon = _make_balloon(find_number, bx, by, page=1)
    multiplier = _make_multiplier(adj_value, bx + 20.0, by, page=1)

    assoc = MultiplierAssociator(proximity_radius=radius)
    results, _ = assoc.associate([balloon], [multiplier], [])

    assert len(results) == 1
    result = results[0]

    assert result.adjacent_multiplier_value == adj_value
    assert result.detail_view_multiplier == 1
    assert result.effective_multiplier == adj_value * 1


@settings(max_examples=100)
@given(
    dv_value=_pos_ints,
    find_number=_find_numbers,
)
def test_property3_effective_multiplier_detail_view_only(
    dv_value: int,
    find_number: str,
) -> None:
    """effective_multiplier == 1 × detail_view when no adjacent multiplier."""
    radius = 100.0
    bx, by = 500.0, 500.0

    balloon = _make_balloon(find_number, bx, by, page=1)

    detail_views = [
        DetailView(
            identifier="B",
            page_number=1,
            region=BoundingBox(x=400.0, y=400.0, width=300.0, height=300.0),
            multiplier=dv_value,
            contained_balloon_indices=[0],
        )
    ]

    assoc = MultiplierAssociator(proximity_radius=radius)
    results, _ = assoc.associate([balloon], [], detail_views)

    assert len(results) == 1
    result = results[0]

    assert result.adjacent_multiplier_value == 1
    assert result.detail_view_multiplier == dv_value
    assert result.effective_multiplier == 1 * dv_value


@settings(max_examples=100)
@given(find_number=_find_numbers)
def test_property3_effective_multiplier_defaults_to_one(
    find_number: str,
) -> None:
    """effective_multiplier == 1 when no adjacent multiplier and no Detail view."""
    radius = 100.0
    bx, by = 500.0, 500.0

    balloon = _make_balloon(find_number, bx, by, page=1)

    assoc = MultiplierAssociator(proximity_radius=radius)
    results, _ = assoc.associate([balloon], [], [])

    assert len(results) == 1
    result = results[0]

    assert result.adjacent_multiplier_value == 1
    assert result.detail_view_multiplier == 1
    assert result.effective_multiplier == 1
