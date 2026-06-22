# Feature: balloon-quantity-analyzer, Property 14: Configuration validation
"""Property-based tests for configuration validation.

**Validates: Requirements 9.1, 9.3**

Property 14: For any AnalyzerConfig, the Balloon_Analyzer SHALL accept the
configuration if and only if the proximity radius is positive and the confidence
threshold is in the range [0.0, 1.0]. For any configuration that violates either
constraint, the Balloon_Analyzer SHALL reject it with an error identifying the
invalid parameter.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from balloon_quantity_analyzer.config import validate_config
from balloon_quantity_analyzer.models import AnalyzerConfig, InvalidConfigurationError


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_positive_floats = st.floats(min_value=1e-9, max_value=1e6, allow_nan=False, allow_infinity=False)
_valid_thresholds = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_non_positive_floats = st.floats(max_value=0.0, allow_nan=False, allow_infinity=False)
_below_zero_thresholds = st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False)
_above_one_thresholds = st.floats(min_value=1.0 + 1e-9, max_value=1e6, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property: valid configs are accepted
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    radius=_positive_floats,
    threshold=_valid_thresholds,
)
def test_valid_config_is_accepted(radius: float, threshold: float) -> None:
    """A config with positive radius and threshold in [0.0, 1.0] is accepted."""
    cfg = AnalyzerConfig(proximity_radius=radius, confidence_threshold=threshold)
    result = validate_config(cfg)
    assert result is cfg


# ---------------------------------------------------------------------------
# Property: non-positive radius is rejected
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    radius=_non_positive_floats,
    threshold=_valid_thresholds,
)
def test_non_positive_radius_rejected(radius: float, threshold: float) -> None:
    """A config with radius <= 0 is rejected with an error mentioning proximity_radius."""
    cfg = AnalyzerConfig(proximity_radius=radius, confidence_threshold=threshold)
    try:
        validate_config(cfg)
        raise AssertionError("Expected InvalidConfigurationError")  # noqa: TRY301
    except InvalidConfigurationError as exc:
        assert "proximity_radius" in str(exc)


# ---------------------------------------------------------------------------
# Property: threshold below 0 is rejected
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    radius=_positive_floats,
    threshold=_below_zero_thresholds,
)
def test_threshold_below_zero_rejected(radius: float, threshold: float) -> None:
    """A config with confidence_threshold < 0 is rejected with an appropriate error."""
    cfg = AnalyzerConfig(proximity_radius=radius, confidence_threshold=threshold)
    try:
        validate_config(cfg)
        raise AssertionError("Expected InvalidConfigurationError")  # noqa: TRY301
    except InvalidConfigurationError as exc:
        assert "confidence_threshold" in str(exc)


# ---------------------------------------------------------------------------
# Property: threshold above 1 is rejected
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    radius=_positive_floats,
    threshold=_above_one_thresholds,
)
def test_threshold_above_one_rejected(radius: float, threshold: float) -> None:
    """A config with confidence_threshold > 1.0 is rejected with an appropriate error."""
    cfg = AnalyzerConfig(proximity_radius=radius, confidence_threshold=threshold)
    try:
        validate_config(cfg)
        raise AssertionError("Expected InvalidConfigurationError")  # noqa: TRY301
    except InvalidConfigurationError as exc:
        assert "confidence_threshold" in str(exc)
