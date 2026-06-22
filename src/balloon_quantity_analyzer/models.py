"""Data models, enums, and custom exceptions for the Balloon Quantity Analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class UnsupportedFormatError(Exception):
    """Raised when the input file format is not supported (not PDF, PNG, JPEG, or TIFF)."""


class UnreadableFileError(Exception):
    """Raised when the input file cannot be opened or is corrupted."""


class InvalidConfigurationError(Exception):
    """Raised when configuration parameters are invalid (e.g. non-positive proximity radius)."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WarningType(Enum):
    """Types of warnings that can be recorded during analysis."""

    UNREADABLE_FIND_NUMBER = "unreadable_find_number"
    AMBIGUOUS_MULTIPLIER = "ambiguous_multiplier"
    MULTIPLE_MULTIPLIERS = "multiple_multipliers"
    OVERLAPPING_DETAIL_VIEWS = "overlapping_detail_views"
    LOW_CONFIDENCE = "low_confidence"
    REJECTED_MULTIPLIER = "rejected_multiplier"
    UNRECOGNIZED_MULTIPLIER_CANDIDATE = "unrecognized_multiplier_candidate"
    INVALID_CONFIGURATION = "invalid_configuration"


# ---------------------------------------------------------------------------
# Data Models (frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding rectangle in pixel coordinates."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class TextRegion:
    """A text string with its location on the page."""

    text: str
    bounding_box: BoundingBox
    confidence: float  # [0.0, 1.0]


@dataclass(frozen=True)
class NormalizedPage:
    """A single page of the drawing, normalized for analysis."""

    page_number: int
    image: bytes  # rasterized page image (PNG bytes)
    text_regions: list[TextRegion]


@dataclass(frozen=True)
class DetectedBalloon:
    """A balloon found on a page."""

    find_number: str  # empty string if unreadable
    page_number: int
    bounding_box: BoundingBox
    confidence: float  # [0.0, 1.0]


@dataclass(frozen=True)
class ParsedMultiplier:
    """A recognized quantity multiplier token."""

    value: int
    raw_text: str
    bounding_box: BoundingBox
    page_number: int
    confidence: float  # [0.0, 1.0]


@dataclass(frozen=True)
class DetailView:
    """A Detail view region on a page."""

    identifier: str  # e.g., "A", "B"
    page_number: int
    region: BoundingBox
    multiplier: int | None  # None if no multiplier declared
    contained_balloon_indices: list[int]


@dataclass(frozen=True)
class AssociatedBalloon:
    """A balloon with its resolved effective multiplier."""

    find_number: str
    page_number: int
    bounding_box: BoundingBox
    adjacent_multiplier_text: str | None
    adjacent_multiplier_value: int  # default 1
    detail_view_id: str | None
    detail_view_multiplier: int  # default 1
    effective_multiplier: int  # adjacent × detail_view
    confidence: float


@dataclass(frozen=True)
class Warning:
    """A warning recorded during analysis."""

    warning_type: WarningType
    message: str
    page_number: int | None
    related_items: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BalloonBreakdown:
    """Per-balloon detail for the report."""

    find_number: str
    page_number: int
    bounding_box: BoundingBox
    adjacent_multiplier_text: str | None
    detail_view_id: str | None
    effective_multiplier: int


@dataclass(frozen=True)
class TallyResult:
    """Aggregated tally and per-balloon breakdown."""

    tally: dict[str, int]  # find_number -> total count
    balloon_breakdown: list[BalloonBreakdown]
    excluded_balloon_count: int  # balloons with empty find numbers


@dataclass(frozen=True)
class AnalyzerConfig:
    """Configuration for the Balloon Analyzer."""

    proximity_radius: float = 50.0  # pixels
    confidence_threshold: float = 0.5  # [0.0, 1.0]
    custom_multiplier_phrases: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisReport:
    """The complete analysis report."""

    tally: dict[str, int]
    balloon_breakdown: list[BalloonBreakdown]
    excluded_balloon_count: int
    warnings: list[Warning]
