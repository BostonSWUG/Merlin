"""Unit tests for MultiplierParser.

Tests each built-in pattern with concrete examples, case-insensitivity,
whitespace variations, unrecognized candidate recording, custom phrase
dictionary integration, and rejected multiplier warnings.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from __future__ import annotations

import pytest

from balloon_quantity_analyzer.models import (
    BoundingBox,
    TextRegion,
    WarningType,
)
from balloon_quantity_analyzer.multiplier_parser import MultiplierParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BBOX = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)


def _region(text: str) -> TextRegion:
    return TextRegion(text=text, bounding_box=_BBOX, confidence=1.0)


def _parse_values(text: str, custom_phrases: dict[str, int] | None = None) -> list[int]:
    """Parse a single text region and return the list of multiplier values."""
    parser = MultiplierParser(custom_phrases=custom_phrases)
    multipliers, _ = parser.parse([_region(text)])
    return [m.value for m in multipliers]


# ---------------------------------------------------------------------------
# Requirement 3.1 — Prefix "nX" and suffix "Xn"
# ---------------------------------------------------------------------------


class TestPrefixMultiplier:
    """Requirement 3.1: '<integer>X' tokens."""

    def test_3x(self):
        assert 3 in _parse_values("3X")

    def test_12x(self):
        assert 12 in _parse_values("12X")

    def test_1x(self):
        assert 1 in _parse_values("1X")


class TestSuffixMultiplier:
    """Requirement 3.1: 'X<integer>' tokens."""

    def test_x5(self):
        assert 5 in _parse_values("X5")

    def test_x10(self):
        assert 10 in _parse_values("X10")

    def test_x1(self):
        assert 1 in _parse_values("X1")


# ---------------------------------------------------------------------------
# Requirement 3.2 — PLACES variants
# ---------------------------------------------------------------------------


class TestPlacesVariants:
    """Requirement 3.2: '(n) PLACES', '(n) PLCS', '(n) PL', 'n PLACES'."""

    def test_paren_places(self):
        assert 4 in _parse_values("(4) PLACES")

    def test_paren_plcs(self):
        assert 2 in _parse_values("(2) PLCS")

    def test_paren_pl(self):
        assert 6 in _parse_values("(6) PL")

    def test_no_paren_places(self):
        assert 3 in _parse_values("3 PLACES")


# ---------------------------------------------------------------------------
# Requirement 3.3 — BOTH SIDES
# ---------------------------------------------------------------------------


class TestBothSides:
    """Requirement 3.3: 'BOTH SIDES' → 2."""

    def test_both_sides(self):
        assert 2 in _parse_values("BOTH SIDES")


# ---------------------------------------------------------------------------
# Requirement 3.4 — TYP / TYPICAL
# ---------------------------------------------------------------------------


class TestTypVariants:
    """Requirement 3.4: 'TYP n', 'TYPICAL n', 'n TYP'."""

    def test_typ_3(self):
        assert 3 in _parse_values("TYP 3")

    def test_typical_4(self):
        assert 4 in _parse_values("TYPICAL 4")

    def test_5_typ(self):
        assert 5 in _parse_values("5 TYP")


# ---------------------------------------------------------------------------
# Requirement 3.1 — Case insensitivity
# ---------------------------------------------------------------------------


class TestCaseInsensitivity:
    """All patterns should be case-insensitive."""

    def test_lowercase_3x(self):
        assert 3 in _parse_values("3x")

    def test_lowercase_x5(self):
        assert 5 in _parse_values("x5")

    def test_lowercase_both_sides(self):
        assert 2 in _parse_values("both sides")

    def test_mixed_case_both_sides(self):
        assert 2 in _parse_values("Both Sides")

    def test_lowercase_typ(self):
        assert 3 in _parse_values("typ 3")

    def test_lowercase_typical(self):
        assert 4 in _parse_values("typical 4")

    def test_lowercase_places(self):
        assert 2 in _parse_values("(2) places")

    def test_lowercase_plcs(self):
        assert 3 in _parse_values("(3) plcs")


# ---------------------------------------------------------------------------
# Requirement 3.1 — Whitespace variations
# ---------------------------------------------------------------------------


class TestWhitespaceVariations:
    """Patterns with extra whitespace should still parse."""

    def test_prefix_extra_space(self):
        assert 3 in _parse_values("3  X")

    def test_suffix_extra_space(self):
        assert 5 in _parse_values("X  5")

    def test_both_sides_extra_space(self):
        assert 2 in _parse_values("BOTH  SIDES")

    def test_typ_extra_space(self):
        assert 3 in _parse_values("TYP  3")

    def test_places_extra_space(self):
        assert 4 in _parse_values("(4)  PLACES")


# ---------------------------------------------------------------------------
# Requirement 3.7 — Unrecognized multiplier candidates
# ---------------------------------------------------------------------------


class TestUnrecognizedCandidates:
    """Requirement 3.7: tokens resembling multipliers but not matching."""

    def test_typ_alone_produces_warning(self):
        parser = MultiplierParser()
        _, warnings = parser.parse([_region("TYP")])
        unrecognized = [
            w for w in warnings
            if w.warning_type == WarningType.UNRECOGNIZED_MULTIPLIER_CANDIDATE
        ]
        assert len(unrecognized) >= 1

    def test_typical_alone_produces_warning(self):
        parser = MultiplierParser()
        _, warnings = parser.parse([_region("TYPICAL")])
        unrecognized = [
            w for w in warnings
            if w.warning_type == WarningType.UNRECOGNIZED_MULTIPLIER_CANDIDATE
        ]
        assert len(unrecognized) >= 1

    def test_unrecognized_includes_context(self):
        parser = MultiplierParser()
        _, warnings = parser.parse([_region("SEE TYP NOTE")])
        unrecognized = [
            w for w in warnings
            if w.warning_type == WarningType.UNRECOGNIZED_MULTIPLIER_CANDIDATE
        ]
        assert len(unrecognized) >= 1
        # The warning should include surrounding context
        assert any("TYP" in w.message for w in unrecognized)


# ---------------------------------------------------------------------------
# Requirement 3.5 — Custom phrase dictionary
# ---------------------------------------------------------------------------


class TestCustomPhrases:
    """Requirement 3.5: configurable custom phrase dictionary."""

    def test_each_side(self):
        assert 2 in _parse_values("EACH SIDE", custom_phrases={"EACH SIDE": 2})

    def test_per_assembly(self):
        assert 4 in _parse_values("PER ASSEMBLY", custom_phrases={"PER ASSEMBLY": 4})

    def test_custom_case_insensitive(self):
        assert 2 in _parse_values("each side", custom_phrases={"EACH SIDE": 2})

    def test_custom_does_not_interfere_with_builtin(self):
        """Custom phrases should work alongside built-in patterns."""
        parser = MultiplierParser(custom_phrases={"EACH SIDE": 2})
        multipliers, _ = parser.parse([_region("3X"), _region("EACH SIDE")])
        values = [m.value for m in multipliers]
        assert 3 in values
        assert 2 in values


# ---------------------------------------------------------------------------
# Requirement 3.6 — Rejected multipliers (non-positive values)
# ---------------------------------------------------------------------------


class TestRejectedMultipliers:
    """Requirement 3.6: non-positive values produce REJECTED_MULTIPLIER warning."""

    def test_0x_rejected(self):
        parser = MultiplierParser()
        multipliers, warnings = parser.parse([_region("0X")])
        assert all(m.value != 0 for m in multipliers)
        rejected = [w for w in warnings if w.warning_type == WarningType.REJECTED_MULTIPLIER]
        assert len(rejected) >= 1

    def test_x0_rejected(self):
        parser = MultiplierParser()
        multipliers, warnings = parser.parse([_region("X0")])
        assert all(m.value != 0 for m in multipliers)
        rejected = [w for w in warnings if w.warning_type == WarningType.REJECTED_MULTIPLIER]
        assert len(rejected) >= 1

    def test_0_places_rejected(self):
        parser = MultiplierParser()
        multipliers, warnings = parser.parse([_region("(0) PLACES")])
        assert all(m.value != 0 for m in multipliers)
        rejected = [w for w in warnings if w.warning_type == WarningType.REJECTED_MULTIPLIER]
        assert len(rejected) >= 1

    def test_custom_phrase_zero_rejected(self):
        parser = MultiplierParser(custom_phrases={"EACH SIDE": 0})
        multipliers, warnings = parser.parse([_region("EACH SIDE")])
        assert all(m.value != 0 for m in multipliers)
        rejected = [w for w in warnings if w.warning_type == WarningType.REJECTED_MULTIPLIER]
        assert len(rejected) >= 1

    def test_custom_phrase_negative_rejected(self):
        parser = MultiplierParser(custom_phrases={"EACH SIDE": -3})
        multipliers, warnings = parser.parse([_region("EACH SIDE")])
        assert all(m.value != -3 for m in multipliers)
        rejected = [w for w in warnings if w.warning_type == WarningType.REJECTED_MULTIPLIER]
        assert len(rejected) >= 1
