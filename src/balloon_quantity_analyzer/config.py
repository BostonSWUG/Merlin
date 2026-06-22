"""Configuration validation for the Balloon Quantity Analyzer."""

from __future__ import annotations

from balloon_quantity_analyzer.models import AnalyzerConfig, InvalidConfigurationError


def validate_config(config: AnalyzerConfig | None = None) -> AnalyzerConfig:
    """Validate an AnalyzerConfig and return a valid instance.

    Args:
        config: An AnalyzerConfig to validate, or None to use defaults.

    Returns:
        A valid AnalyzerConfig with defaults applied if *config* was None.

    Raises:
        InvalidConfigurationError: If proximity_radius <= 0 or
            confidence_threshold is outside [0.0, 1.0].
    """
    if config is None:
        return AnalyzerConfig()

    if config.proximity_radius <= 0:
        raise InvalidConfigurationError(
            f"proximity_radius must be positive, got {config.proximity_radius}"
        )

    if not (0.0 <= config.confidence_threshold <= 1.0):
        raise InvalidConfigurationError(
            f"confidence_threshold must be in [0.0, 1.0], got {config.confidence_threshold}"
        )

    return config
