# Feature: balloon-quantity-analyzer, Property 1: Multiplier parsing correctness
"""Property-based tests for multiplier parsing correctness.

**Validates: Requirements 3.1, 3.2, 3.4, 3.5**

Property 1: For any positive integer n and any recognized multiplier pattern
variant, the MultiplierParser SHALL parse the token and return a
ParsedMultiplier with value equal to n. Additionally, for any custom phrase
dictionary mapping phrase p to integer v, configuring the parser with that
dictionary and parsing p SHALL return value v.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from balloon_quantity_analyzer.models import BoundingBox, TextRegion
from balloon_quantity_analyzer.multiplier_parser import MultiplierParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_BBOX = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)


def _make_region(text: str) -> TextRegion:
    """Create a TextRegion with dummy bounding box for testing."""
    return TextRegion(text=text, bounding_box=_DUMMY_BBOX, confidence=1.0)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Positive integers that are reasonable for multiplier values
_positive_ints = st.integers(min_value=1, max_value=9999)

# Custom phrase: alphabetic words that won't collide with built-in patterns
_custom_phrase = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwz"),  # no 'x' or 'y' to avoid X/TYP collisions
    min_size=3,
    max_size=12,
).filter(lambda s: s.upper() not in {"TYP", "TYPICAL", "PLACES", "PLCS", "BOTH", "SIDES"})


# ---------------------------------------------------------------------------
# Property: Prefix multiplier "{n}X" and "{n} X"
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(n=_positive_ints)
def test_prefix_multiplier_no_space(n: int) -> None:
    """'{n}X' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"{n}X")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for '{n}X'"


@settings(max_examples=100)
@given(n=_positive_ints)
def test_prefix_multiplier_with_space(n: int) -> None:
    """'{n} X' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"{n} X")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for '{n} X'"


# ---------------------------------------------------------------------------
# Property: Suffix multiplier "X{n}" and "X {n}"
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(n=_positive_ints)
def test_suffix_multiplier_no_space(n: int) -> None:
    """'X{n}' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"X{n}")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for 'X{n}'"


@settings(max_examples=100)
@given(n=_positive_ints)
def test_suffix_multiplier_with_space(n: int) -> None:
    """'X {n}' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"X {n}")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for 'X {n}'"


# ---------------------------------------------------------------------------
# Property: PLACES variants
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(n=_positive_ints)
def test_places_paren_full(n: int) -> None:
    """'({n}) PLACES' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"({n}) PLACES")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for '({n}) PLACES'"


@settings(max_examples=100)
@given(n=_positive_ints)
def test_places_paren_plcs(n: int) -> None:
    """'({n}) PLCS' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"({n}) PLCS")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for '({n}) PLCS'"


@settings(max_examples=100)
@given(n=_positive_ints)
def test_places_paren_pl(n: int) -> None:
    """'({n}) PL' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"({n}) PL")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for '({n}) PL'"


@settings(max_examples=100)
@given(n=_positive_ints)
def test_places_no_paren(n: int) -> None:
    """'{n} PLACES' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"{n} PLACES")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for '{n} PLACES'"


# ---------------------------------------------------------------------------
# Property: BOTH SIDES always equals 2
# ---------------------------------------------------------------------------

def test_both_sides_equals_two() -> None:
    """'BOTH SIDES' is always parsed as multiplier with value 2."""
    parser = MultiplierParser()
    region = _make_region("BOTH SIDES")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert 2 in values, f"Expected 2 in {values} for 'BOTH SIDES'"


# ---------------------------------------------------------------------------
# Property: TYP / TYPICAL variants
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(n=_positive_ints)
def test_typ_prefix(n: int) -> None:
    """'TYP {n}' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"TYP {n}")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for 'TYP {n}'"


@settings(max_examples=100)
@given(n=_positive_ints)
def test_typical_prefix(n: int) -> None:
    """'TYPICAL {n}' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"TYPICAL {n}")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for 'TYPICAL {n}'"


@settings(max_examples=100)
@given(n=_positive_ints)
def test_typ_suffix(n: int) -> None:
    """'{n} TYP' is parsed as multiplier with value n."""
    parser = MultiplierParser()
    region = _make_region(f"{n} TYP")
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert n in values, f"Expected {n} in {values} for '{n} TYP'"


# ---------------------------------------------------------------------------
# Property: Custom phrase dictionary parsing
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    phrase=_custom_phrase,
    value=_positive_ints,
)
def test_custom_phrase_parsing(phrase: str, value: int) -> None:
    """A custom phrase mapped to a positive integer is parsed correctly."""
    parser = MultiplierParser(custom_phrases={phrase: value})
    region = _make_region(phrase)
    multipliers, _ = parser.parse([region])
    values = [m.value for m in multipliers]
    assert value in values, f"Expected {value} in {values} for custom phrase '{phrase}'"


# Feature: balloon-quantity-analyzer, Property 2: Non-positive multiplier rejection
# ---------------------------------------------------------------------------
# Property 2: For any integer value that is zero or negative, when a
# multiplier token containing that value is parsed, the MultiplierParser
# SHALL reject the token and record a warning.
#
# **Validates: Requirements 3.6**
#
# NOTE: The built-in regex patterns use \d+ which only matches non-negative
# digits. Negative numbers (e.g. "-3X") won't match at all. Zero ("0X",
# "X0", etc.) WILL match and must be rejected. Custom phrases can map to
# any integer, so we test zero and negative values there.
# ---------------------------------------------------------------------------

from balloon_quantity_analyzer.models import WarningType


# ---------------------------------------------------------------------------
# Strategy: zero-value tokens across all built-in pattern formats
# ---------------------------------------------------------------------------

# All built-in patterns that embed a digit — with value fixed to 0
_ZERO_PATTERN_TEMPLATES = [
    "0X",           # prefix, no space
    "0 X",          # prefix, with space
    "X0",           # suffix, no space
    "X 0",          # suffix, with space
    "(0) PLACES",   # PLACES with parens
    "(0) PLCS",     # PLCS with parens
    "(0) PL",       # PL with parens
    "0 PLACES",     # PLACES without parens
    "TYP 0",        # TYP prefix
    "TYPICAL 0",    # TYPICAL prefix
    "0 TYP",        # TYP suffix
]


@settings(max_examples=100)
@given(template=st.sampled_from(_ZERO_PATTERN_TEMPLATES))
def test_zero_multiplier_rejected_builtin(template: str) -> None:
    """Zero-value tokens in any built-in pattern format are rejected with a warning."""
    parser = MultiplierParser()
    region = _make_region(template)
    multipliers, warnings = parser.parse([region])

    # No ParsedMultiplier should be produced for a zero value
    assert all(
        m.value != 0 for m in multipliers
    ), f"Zero-value multiplier should not be accepted, got {multipliers}"

    # A REJECTED_MULTIPLIER warning must be present
    rejected = [w for w in warnings if w.warning_type == WarningType.REJECTED_MULTIPLIER]
    assert len(rejected) >= 1, (
        f"Expected REJECTED_MULTIPLIER warning for '{template}', "
        f"got warnings: {warnings}"
    )


# ---------------------------------------------------------------------------
# Strategy: custom phrases mapped to zero
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(phrase=_custom_phrase)
def test_custom_phrase_zero_rejected(phrase: str) -> None:
    """A custom phrase mapped to 0 is rejected with a REJECTED_MULTIPLIER warning."""
    parser = MultiplierParser(custom_phrases={phrase: 0})
    region = _make_region(phrase)
    multipliers, warnings = parser.parse([region])

    # No ParsedMultiplier with value 0 should be produced
    assert all(
        m.value != 0 for m in multipliers
    ), f"Zero-value custom phrase should not produce a multiplier, got {multipliers}"

    rejected = [w for w in warnings if w.warning_type == WarningType.REJECTED_MULTIPLIER]
    assert len(rejected) >= 1, (
        f"Expected REJECTED_MULTIPLIER warning for custom phrase '{phrase}' → 0, "
        f"got warnings: {warnings}"
    )


# ---------------------------------------------------------------------------
# Strategy: custom phrases mapped to negative values
# ---------------------------------------------------------------------------

_negative_ints = st.integers(min_value=-9999, max_value=-1)


@settings(max_examples=100)
@given(phrase=_custom_phrase, value=_negative_ints)
def test_custom_phrase_negative_rejected(phrase: str, value: int) -> None:
    """A custom phrase mapped to a negative integer is rejected with a warning."""
    parser = MultiplierParser(custom_phrases={phrase: value})
    region = _make_region(phrase)
    multipliers, warnings = parser.parse([region])

    # No ParsedMultiplier with the negative value should be produced
    assert all(
        m.value != value for m in multipliers
    ), f"Negative-value custom phrase should not produce a multiplier, got {multipliers}"

    rejected = [w for w in warnings if w.warning_type == WarningType.REJECTED_MULTIPLIER]
    assert len(rejected) >= 1, (
        f"Expected REJECTED_MULTIPLIER warning for custom phrase '{phrase}' → {value}, "
        f"got warnings: {warnings}"
    )
