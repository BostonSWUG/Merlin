"""Unit tests for configuration validation."""

import pytest

from balloon_quantity_analyzer.config import validate_config
from balloon_quantity_analyzer.models import AnalyzerConfig, InvalidConfigurationError


class TestValidateConfigDefaults:
    """Tests for default value application (Requirement 9.2)."""

    def test_none_returns_default_config(self):
        result = validate_config(None)
        assert result == AnalyzerConfig()

    def test_default_proximity_radius(self):
        result = validate_config(None)
        assert result.proximity_radius == 50.0

    def test_default_confidence_threshold(self):
        result = validate_config(None)
        assert result.confidence_threshold == 0.5

    def test_default_custom_multiplier_phrases(self):
        result = validate_config(None)
        assert result.custom_multiplier_phrases == {}


class TestValidateConfigAcceptsValid:
    """Tests for accepting valid configurations (Requirement 9.1)."""

    def test_valid_config_returned_unchanged(self):
        cfg = AnalyzerConfig(proximity_radius=100.0, confidence_threshold=0.8)
        assert validate_config(cfg) is cfg

    def test_boundary_confidence_zero(self):
        cfg = AnalyzerConfig(confidence_threshold=0.0)
        assert validate_config(cfg) is cfg

    def test_boundary_confidence_one(self):
        cfg = AnalyzerConfig(confidence_threshold=1.0)
        assert validate_config(cfg) is cfg

    def test_small_positive_radius(self):
        cfg = AnalyzerConfig(proximity_radius=0.001)
        assert validate_config(cfg) is cfg


class TestValidateConfigRejectsInvalid:
    """Tests for rejecting invalid configurations (Requirement 9.3)."""

    def test_zero_proximity_radius(self):
        cfg = AnalyzerConfig(proximity_radius=0.0)
        with pytest.raises(InvalidConfigurationError, match="proximity_radius"):
            validate_config(cfg)

    def test_negative_proximity_radius(self):
        cfg = AnalyzerConfig(proximity_radius=-10.0)
        with pytest.raises(InvalidConfigurationError, match="proximity_radius"):
            validate_config(cfg)

    def test_confidence_below_zero(self):
        cfg = AnalyzerConfig(confidence_threshold=-0.1)
        with pytest.raises(InvalidConfigurationError, match="confidence_threshold"):
            validate_config(cfg)

    def test_confidence_above_one(self):
        cfg = AnalyzerConfig(confidence_threshold=1.01)
        with pytest.raises(InvalidConfigurationError, match="confidence_threshold"):
            validate_config(cfg)

    def test_error_message_includes_invalid_value(self):
        cfg = AnalyzerConfig(proximity_radius=-5.0)
        with pytest.raises(InvalidConfigurationError, match="-5.0"):
            validate_config(cfg)
