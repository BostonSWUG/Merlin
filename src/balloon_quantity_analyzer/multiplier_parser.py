"""Multiplier Parser — recognizes quantity multiplier tokens from text regions.

Supports built-in regex patterns (prefix nX, suffix Xn, PLACES variants,
BOTH SIDES, TYP/TYPICAL) and a configurable custom phrase dictionary.
"""

from __future__ import annotations

import re

from balloon_quantity_analyzer.models import (
    ParsedMultiplier,
    TextRegion,
    Warning,
    WarningType,
)


# ---------------------------------------------------------------------------
# Built-in patterns
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, group index that holds the integer, fixed value or None)
# When fixed_value is not None the pattern always yields that value (e.g. BOTH SIDES → 2).
_BUILTIN_PATTERNS: list[tuple[re.Pattern[str], list[int], int | None]] = [
    # Prefix multiplier: "3X", "3 X"  — but NOT "X3" (handled separately)
    (re.compile(r"(\d+)\s*X\b", re.IGNORECASE), [1], None),
    # Suffix multiplier: "X3", "X 3"
    (re.compile(r"\bX\s*(\d+)", re.IGNORECASE), [1], None),
    # PLACES variants: "(4) PLACES", "4 PLCS", "(4) PL"
    (re.compile(r"\(?(\d+)\)?\s*(?:PLACES|PLCS|PL)\b", re.IGNORECASE), [1], None),
    # BOTH SIDES → always 2
    (re.compile(r"\bBOTH\s+SIDES\b", re.IGNORECASE), [], 2),
    # TYP / TYPICAL with count: "TYP 3", "TYPICAL 3", "3 TYP", "3 TYPICAL"
    (re.compile(r"\bTYP(?:ICAL)?\s+(\d+)\b", re.IGNORECASE), [1], None),
    (re.compile(r"\b(\d+)\s+TYP(?:ICAL)?\b", re.IGNORECASE), [1], None),
]

# Heuristic pattern to detect tokens that *look like* multiplier candidates
# but didn't match any built-in or custom pattern.
_CANDIDATE_PATTERN = re.compile(
    r"""
    (?:\d+\s*X\b)          |   # nX
    (?:\bX\s*\d+)          |   # Xn
    (?:\(?\d+\)?\s*(?:PLACES|PLCS|PL)\b) |  # PLACES variants
    (?:\bBOTH\s+SIDES\b)  |   # BOTH SIDES
    (?:\bTYP(?:ICAL)?\b)       # TYP / TYPICAL (without a number)
    """,
    re.IGNORECASE | re.VERBOSE,
)


class MultiplierParser:
    """Recognizes quantity multiplier tokens from text regions."""

    def __init__(self, custom_phrases: dict[str, int] | None = None) -> None:
        """
        Parameters
        ----------
        custom_phrases:
            Optional mapping of phrase → integer multiplier value.
            Phrases are matched case-insensitively as whole words.
        """
        self._custom_phrases: dict[str, int] = custom_phrases or {}
        # Pre-compile custom phrase patterns (escaped, case-insensitive)
        self._custom_patterns: list[tuple[re.Pattern[str], int]] = [
            (re.compile(re.escape(phrase), re.IGNORECASE), value)
            for phrase, value in self._custom_phrases.items()
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self, text_regions: list[TextRegion]
    ) -> tuple[list[ParsedMultiplier], list[Warning]]:
        """Parse multiplier tokens from *text_regions*.

        Returns
        -------
        tuple of (parsed_multipliers, warnings)
        """
        multipliers: list[ParsedMultiplier] = []
        warnings: list[Warning] = []

        for region in text_regions:
            text = region.text
            matched_spans: list[tuple[int, int]] = []

            # --- Built-in patterns ---
            for pattern, groups, fixed_value in _BUILTIN_PATTERNS:
                for m in pattern.finditer(text):
                    if fixed_value is not None:
                        value = fixed_value
                    else:
                        # Extract the integer from the first non-None group
                        raw_str = next(
                            m.group(g) for g in groups if m.group(g) is not None
                        )
                        value = int(raw_str)

                    result = self._validate_and_record(
                        value=value,
                        raw_text=m.group(0),
                        region=region,
                        multipliers=multipliers,
                        warnings=warnings,
                    )
                    if result:
                        matched_spans.append(m.span())

            # --- Custom phrase patterns ---
            for pattern, value in self._custom_patterns:
                for m in pattern.finditer(text):
                    result = self._validate_and_record(
                        value=value,
                        raw_text=m.group(0),
                        region=region,
                        multipliers=multipliers,
                        warnings=warnings,
                    )
                    if result:
                        matched_spans.append(m.span())

            # --- Unrecognized candidate detection ---
            self._detect_unrecognized(
                text=text,
                matched_spans=matched_spans,
                region=region,
                warnings=warnings,
            )

        return multipliers, warnings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_and_record(
        self,
        value: int,
        raw_text: str,
        region: TextRegion,
        multipliers: list[ParsedMultiplier],
        warnings: list[Warning],
    ) -> bool:
        """Validate *value* and append to *multipliers* or *warnings*.

        Returns True when the span was "consumed" (matched), regardless of
        whether the value was accepted or rejected.
        """
        if value <= 0:
            warnings.append(
                Warning(
                    warning_type=WarningType.REJECTED_MULTIPLIER,
                    message=(
                        f"Rejected non-positive multiplier value {value} "
                        f"in token '{raw_text}'"
                    ),
                    page_number=None,
                    related_items=[raw_text],
                )
            )
            return True

        multipliers.append(
            ParsedMultiplier(
                value=value,
                raw_text=raw_text,
                bounding_box=region.bounding_box,
                page_number=0,  # caller may override via region metadata
                confidence=region.confidence,
            )
        )
        return True

    @staticmethod
    def _detect_unrecognized(
        text: str,
        matched_spans: list[tuple[int, int]],
        region: TextRegion,
        warnings: list[Warning],
    ) -> None:
        """Record tokens that look like multiplier candidates but were not matched."""
        for m in _CANDIDATE_PATTERN.finditer(text):
            span = m.span()
            # Skip if this span overlaps with any already-matched span
            if any(
                not (span[1] <= ms[0] or span[0] >= ms[1])
                for ms in matched_spans
            ):
                continue

            # Build surrounding context (up to 20 chars each side)
            start = max(0, span[0] - 20)
            end = min(len(text), span[1] + 20)
            context = text[start:end]

            warnings.append(
                Warning(
                    warning_type=WarningType.UNRECOGNIZED_MULTIPLIER_CANDIDATE,
                    message=(
                        f"Unrecognized multiplier candidate '{m.group(0)}' "
                        f"in context: '{context}'"
                    ),
                    page_number=None,
                    related_items=[m.group(0), context],
                )
            )
