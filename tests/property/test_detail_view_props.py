# Feature: balloon-quantity-analyzer, Property 15: Detail view label parsing
# Feature: balloon-quantity-analyzer, Property 7: Detail view balloon containment
# Feature: balloon-quantity-analyzer, Property 8: Overlapping Detail views produce warning
"""Property-based tests for DetailViewResolver.

Tests cover:
- Property 15: Detail view label parsing (Validates: Requirements 5.1)
- Property 7: Detail view balloon containment (Validates: Requirements 5.2)
- Property 8: Overlapping Detail views produce warning (Validates: Requirements 5.5)
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from balloon_quantity_analyzer.detail_view_resolver import (
    DetailViewResolver,
    parse_detail_label,
    _point_in_bbox,
)
from balloon_quantity_analyzer.models import (
    BoundingBox,
    DetectedBalloon,
    NormalizedPage,
    TextRegion,
    WarningType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOX_SIZE = 10.0


def _make_balloon(
    find_number: str, x: float, y: float, page: int = 1
) -> DetectedBalloon:
    """Create a DetectedBalloon with top-left at (x, y)."""
    return DetectedBalloon(
        find_number=find_number,
        page_number=page,
        bounding_box=BoundingBox(x=x, y=y, width=_BOX_SIZE, height=_BOX_SIZE),
        confidence=0.9,
    )


def _make_page(
    text_regions: list[TextRegion],
    page_number: int = 1,
) -> NormalizedPage:
    return NormalizedPage(
        page_number=page_number,
        image=b"",
        text_regions=text_regions,
    )


def _make_text_region(
    text: str, x: float, y: float,
) -> TextRegion:
    return TextRegion(
        text=text,
        bounding_box=BoundingBox(x=x, y=y, width=60, height=20),
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Single uppercase letter identifiers
_identifiers = st.sampled_from(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

# Positive integer multipliers (reasonable range for drawings)
_pos_ints = st.integers(min_value=1, max_value=999)

# Coordinate values on a drawing page
_coords = st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False)

# Positive dimensions for bounding boxes
_dimensions = st.floats(min_value=10.0, max_value=2000.0, allow_nan=False, allow_infinity=False)

# PLACES keyword variants
_places_variants = st.sampled_from(["PLACES", "PLCS", "PL"])


# ===========================================================================
# Property 15: Detail view label parsing
# **Validates: Requirements 5.1**
#
# For any Detail view identifier (a letter A-Z) and optional positive
# integer multiplier, a label string of the form "DETAIL <id>" or
# "DETAIL <id> (<n> PLACES)" SHALL be parsed into a DetailView with the
# correct identifier and multiplier value (or None if no multiplier).
# ===========================================================================


@settings(max_examples=100)
@given(identifier=_identifiers)
def test_property15_parse_detail_label_no_multiplier(
    identifier: str,
) -> None:
    """'DETAIL X' parses to (identifier, None)."""
    label = f"DETAIL {identifier}"
    result = parse_detail_label(label)

    assert result is not None, f"Expected match for '{label}'"
    parsed_id, parsed_mult = result
    assert parsed_id == identifier
    assert parsed_mult is None


@settings(max_examples=100)
@given(
    identifier=_identifiers,
    multiplier=_pos_ints,
    variant=_places_variants,
)
def test_property15_parse_detail_label_with_multiplier(
    identifier: str,
    multiplier: int,
    variant: str,
) -> None:
    """'DETAIL X (n PLACES)' parses to (identifier, n)."""
    label = f"DETAIL {identifier} ({multiplier} {variant})"
    result = parse_detail_label(label)

    assert result is not None, f"Expected match for '{label}'"
    parsed_id, parsed_mult = result
    assert parsed_id == identifier
    assert parsed_mult == multiplier


@settings(max_examples=100)
@given(
    identifier=_identifiers,
    multiplier=_pos_ints,
    variant=_places_variants,
)
def test_property15_parse_detail_label_without_parens(
    identifier: str,
    multiplier: int,
    variant: str,
) -> None:
    """'DETAIL X n PLACES' (no parentheses) also parses correctly."""
    label = f"DETAIL {identifier} {multiplier} {variant}"
    result = parse_detail_label(label)

    assert result is not None, f"Expected match for '{label}'"
    parsed_id, parsed_mult = result
    assert parsed_id == identifier
    assert parsed_mult == multiplier


@settings(max_examples=100)
@given(
    identifier=_identifiers,
    multiplier=st.one_of(st.none(), _pos_ints),
    variant=_places_variants,
)
def test_property15_parse_detail_label_case_insensitive(
    identifier: str,
    multiplier: int | None,
    variant: str,
) -> None:
    """Parsing is case-insensitive; identifier is always returned uppercase."""
    if multiplier is not None:
        label = f"detail {identifier.lower()} ({multiplier} {variant.lower()})"
    else:
        label = f"detail {identifier.lower()}"

    result = parse_detail_label(label)

    assert result is not None, f"Expected match for '{label}'"
    parsed_id, parsed_mult = result
    assert parsed_id == identifier  # always uppercase
    assert parsed_mult == multiplier



# ===========================================================================
# Property 7: Detail view balloon containment
# **Validates: Requirements 5.2**
#
# For any Detail view region and any set of balloons, the
# Detail_View_Resolver SHALL identify exactly those balloons whose bounding
# box centers lie inside the Detail view region as contained balloons, and
# no others.
# ===========================================================================


@settings(max_examples=100)
@given(
    region_x=_coords,
    region_y=_coords,
    region_w=_dimensions,
    region_h=_dimensions,
    # Balloon center offsets as fractions of region dimensions
    inside_frac_x=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    inside_frac_y=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
)
def test_property7_balloon_inside_region_is_contained(
    region_x: float,
    region_y: float,
    region_w: float,
    region_h: float,
    inside_frac_x: float,
    inside_frac_y: float,
) -> None:
    """A balloon whose center is inside the region is classified as contained."""
    region = BoundingBox(x=region_x, y=region_y, width=region_w, height=region_h)

    # Place balloon center inside the region using fractional offsets
    balloon_cx = region_x + inside_frac_x * region_w
    balloon_cy = region_y + inside_frac_y * region_h
    # Offset so the center of the balloon box lands at (balloon_cx, balloon_cy)
    balloon_x = balloon_cx - _BOX_SIZE / 2.0
    balloon_y = balloon_cy - _BOX_SIZE / 2.0

    assert _point_in_bbox(balloon_cx, balloon_cy, region) is True


@settings(max_examples=100)
@given(
    region_x=_coords,
    region_y=_coords,
    region_w=_dimensions,
    region_h=_dimensions,
    # How far outside the region the balloon center is placed
    outside_offset=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    direction=st.sampled_from(["left", "right", "above", "below"]),
)
def test_property7_balloon_outside_region_not_contained(
    region_x: float,
    region_y: float,
    region_w: float,
    region_h: float,
    outside_offset: float,
    direction: str,
) -> None:
    """A balloon whose center is outside the region is NOT classified as contained."""
    region = BoundingBox(x=region_x, y=region_y, width=region_w, height=region_h)

    # Place balloon center outside the region in the given direction
    mid_x = region_x + region_w / 2.0
    mid_y = region_y + region_h / 2.0

    if direction == "left":
        balloon_cx = region_x - outside_offset
        balloon_cy = mid_y
    elif direction == "right":
        balloon_cx = region_x + region_w + outside_offset
        balloon_cy = mid_y
    elif direction == "above":
        balloon_cx = mid_x
        balloon_cy = region_y - outside_offset
    else:  # below
        balloon_cx = mid_x
        balloon_cy = region_y + region_h + outside_offset

    assert _point_in_bbox(balloon_cx, balloon_cy, region) is False


@settings(max_examples=100)
@given(
    label_x=st.floats(min_value=300.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    label_y=st.floats(min_value=300.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    padding=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    n_inside=st.integers(min_value=0, max_value=5),
    n_outside=st.integers(min_value=0, max_value=5),
)
def test_property7_resolver_containment_matches_point_in_bbox(
    label_x: float,
    label_y: float,
    padding: float,
    n_inside: int,
    n_outside: int,
) -> None:
    """The resolver's containment result matches _point_in_bbox for each balloon.

    We place a Detail view label, infer its region, then place some balloons
    inside and some outside. The resolver should contain exactly the inside ones.
    """
    assume(n_inside + n_outside > 0)

    # Label center is approximately (label_x + 30, label_y + 10) given 60x20 box
    label_cx = label_x + 30.0
    label_cy = label_y + 10.0

    # Inferred region: centered on label center, extending padding in each direction
    region_x = max(label_cx - padding, 0.0)
    region_y = max(label_cy - padding, 0.0)
    region = BoundingBox(x=region_x, y=region_y, width=padding * 2, height=padding * 2)

    balloons: list[DetectedBalloon] = []

    # Place inside balloons: center within the region
    for i in range(n_inside):
        frac = 0.3 + 0.05 * i  # stay well inside
        bcx = region_x + frac * region.width
        bcy = region_y + frac * region.height
        bx = bcx - _BOX_SIZE / 2.0
        by = bcy - _BOX_SIZE / 2.0
        balloons.append(_make_balloon(f"IN{i}", bx, by, page=1))

    # Place outside balloons: center well outside the region
    for i in range(n_outside):
        bcx = region_x + region.width + padding + 100.0 + i * 50.0
        bcy = region_y + region.height + padding + 100.0 + i * 50.0
        bx = bcx - _BOX_SIZE / 2.0
        by = bcy - _BOX_SIZE / 2.0
        balloons.append(_make_balloon(f"OUT{i}", bx, by, page=1))

    page = _make_page([_make_text_region("DETAIL A", label_x, label_y)])
    resolver = DetailViewResolver(region_padding=padding)
    dvs, _ = resolver.resolve(page, balloons)

    assert len(dvs) == 1
    contained = set(dvs[0].contained_balloon_indices)

    # Verify: inside balloons are contained, outside are not
    for idx in range(n_inside):
        assert idx in contained, f"Balloon IN{idx} should be contained"
    for idx in range(n_inside, n_inside + n_outside):
        assert idx not in contained, f"Balloon OUT{idx - n_inside} should NOT be contained"


# ===========================================================================
# Property 8: Overlapping Detail views produce warning
# **Validates: Requirements 5.5**
#
# For any balloon whose bounding box center lies inside two or more Detail
# views that each declare a Quantity_Multiplier, the Detail_View_Resolver
# SHALL record a warning identifying the balloon and the overlapping Detail
# views, and the Multiplier_Associator SHALL NOT apply the overlapping
# Detail view multipliers.
# ===========================================================================


@settings(max_examples=100)
@given(
    label_x=st.floats(min_value=300.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    label_y=st.floats(min_value=300.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    padding=st.floats(min_value=100.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    mult_a=_pos_ints,
    mult_b=_pos_ints,
    find_number=st.text(alphabet="0123456789", min_size=1, max_size=3),
)
def test_property8_overlapping_detail_views_warning(
    label_x: float,
    label_y: float,
    padding: float,
    mult_a: int,
    mult_b: int,
    find_number: str,
) -> None:
    """A balloon inside 2+ Detail views with multipliers produces a warning.

    We place two Detail view labels close together (both with multipliers)
    so their inferred regions overlap, and a balloon inside both regions.
    The resolver should produce an OVERLAPPING_DETAIL_VIEWS warning.
    """
    # Two labels close together — their inferred regions will overlap
    label_offset = 20.0  # small offset so regions overlap heavily
    tr_a = _make_text_region(f"DETAIL A ({mult_a} PLACES)", label_x, label_y)
    tr_b = _make_text_region(
        f"DETAIL B ({mult_b} PLACES)", label_x + label_offset, label_y
    )

    # Place balloon near both labels so its center is inside both inferred regions
    balloon_cx = label_x + 30.0 + label_offset / 2.0  # between the two labels
    balloon_cy = label_y + 10.0
    balloon_x = balloon_cx - _BOX_SIZE / 2.0
    balloon_y = balloon_cy - _BOX_SIZE / 2.0
    balloon = _make_balloon(find_number, balloon_x, balloon_y, page=1)

    page = _make_page([tr_a, tr_b])
    resolver = DetailViewResolver(region_padding=padding)
    dvs, warnings = resolver.resolve(page, [balloon])

    # Both detail views should be detected
    assert len(dvs) == 2

    # The balloon should be contained in both detail views
    dv_ids_containing_balloon = [
        dv.identifier for dv in dvs if 0 in dv.contained_balloon_indices
    ]
    assert len(dv_ids_containing_balloon) >= 2, (
        f"Balloon should be in 2+ detail views, but is in {dv_ids_containing_balloon}"
    )

    # An OVERLAPPING_DETAIL_VIEWS warning must be produced
    overlap_warnings = [
        w for w in warnings if w.warning_type == WarningType.OVERLAPPING_DETAIL_VIEWS
    ]
    assert len(overlap_warnings) >= 1, (
        f"Expected OVERLAPPING_DETAIL_VIEWS warning, got: {warnings}"
    )


@settings(max_examples=100)
@given(
    label_x=st.floats(min_value=300.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    label_y=st.floats(min_value=300.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    padding=st.floats(min_value=100.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    identifier_a=st.sampled_from(list("ABCDEFGHIJ")),
    identifier_b=st.sampled_from(list("KLMNOPQRST")),
    find_number=st.text(alphabet="0123456789", min_size=1, max_size=3),
)
def test_property8_overlapping_without_multipliers_no_warning(
    label_x: float,
    label_y: float,
    padding: float,
    identifier_a: str,
    identifier_b: str,
    find_number: str,
) -> None:
    """Overlapping Detail views WITHOUT multipliers should NOT produce a warning.

    The warning is only triggered when both overlapping views declare multipliers.
    """
    label_offset = 20.0
    tr_a = _make_text_region(f"DETAIL {identifier_a}", label_x, label_y)
    tr_b = _make_text_region(
        f"DETAIL {identifier_b}", label_x + label_offset, label_y
    )

    balloon_cx = label_x + 30.0 + label_offset / 2.0
    balloon_cy = label_y + 10.0
    balloon_x = balloon_cx - _BOX_SIZE / 2.0
    balloon_y = balloon_cy - _BOX_SIZE / 2.0
    balloon = _make_balloon(find_number, balloon_x, balloon_y, page=1)

    page = _make_page([tr_a, tr_b])
    resolver = DetailViewResolver(region_padding=padding)
    dvs, warnings = resolver.resolve(page, [balloon])

    # No OVERLAPPING_DETAIL_VIEWS warning when neither view has a multiplier
    overlap_warnings = [
        w for w in warnings if w.warning_type == WarningType.OVERLAPPING_DETAIL_VIEWS
    ]
    assert len(overlap_warnings) == 0, (
        f"No overlap warning expected without multipliers, got: {overlap_warnings}"
    )
