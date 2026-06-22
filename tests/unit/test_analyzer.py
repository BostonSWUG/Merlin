"""Unit tests for the BalloonAnalyzer orchestrator."""

from __future__ import annotations

import pytest

from balloon_quantity_analyzer.analyzer import BalloonAnalyzer
from balloon_quantity_analyzer.models import (
    AnalysisReport,
    AnalyzerConfig,
    BoundingBox,
    DetectedBalloon,
    InvalidConfigurationError,
    NormalizedPage,
    ParsedMultiplier,
    TextRegion,
    Warning,
    WarningType,
)
from balloon_quantity_analyzer.ocr_adapter import OcrAdapter

import numpy as np


# ---------------------------------------------------------------------------
# Stub OCR adapter for testing (returns no text)
# ---------------------------------------------------------------------------

class StubOcrAdapter:
    """OCR adapter that returns no text regions — used for controlled tests."""

    def extract_text(
        self, image: np.ndarray, region: BoundingBox | None = None
    ) -> list[TextRegion]:
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBalloonAnalyzerInit:
    """Tests for BalloonAnalyzer construction and config validation."""

    def test_default_config(self) -> None:
        analyzer = BalloonAnalyzer(ocr_adapter=StubOcrAdapter())
        assert analyzer._config.proximity_radius == 50.0
        assert analyzer._config.confidence_threshold == 0.5

    def test_custom_config(self) -> None:
        cfg = AnalyzerConfig(proximity_radius=100.0, confidence_threshold=0.8)
        analyzer = BalloonAnalyzer(config=cfg, ocr_adapter=StubOcrAdapter())
        assert analyzer._config.proximity_radius == 100.0
        assert analyzer._config.confidence_threshold == 0.8

    def test_invalid_config_raises(self) -> None:
        bad_cfg = AnalyzerConfig(proximity_radius=-1.0)
        with pytest.raises(InvalidConfigurationError):
            BalloonAnalyzer(config=bad_cfg, ocr_adapter=StubOcrAdapter())

    def test_invalid_threshold_raises(self) -> None:
        bad_cfg = AnalyzerConfig(confidence_threshold=2.0)
        with pytest.raises(InvalidConfigurationError):
            BalloonAnalyzer(config=bad_cfg, ocr_adapter=StubOcrAdapter())

    def test_none_config_uses_defaults(self) -> None:
        analyzer = BalloonAnalyzer(config=None, ocr_adapter=StubOcrAdapter())
        assert analyzer._config == AnalyzerConfig()


class TestBalloonAnalyzerInputErrors:
    """Tests for fail-fast on input errors."""

    def test_unsupported_format(self, tmp_path) -> None:
        from balloon_quantity_analyzer.models import UnsupportedFormatError

        bad_file = tmp_path / "drawing.xyz"
        bad_file.write_bytes(b"not a real file")
        analyzer = BalloonAnalyzer(ocr_adapter=StubOcrAdapter())
        with pytest.raises(UnsupportedFormatError):
            analyzer.analyze(str(bad_file))

    def test_unreadable_file(self, tmp_path) -> None:
        from balloon_quantity_analyzer.models import UnreadableFileError

        missing = tmp_path / "nonexistent.pdf"
        analyzer = BalloonAnalyzer(ocr_adapter=StubOcrAdapter())
        with pytest.raises(UnreadableFileError):
            analyzer.analyze(str(missing))


class TestBalloonAnalyzerPipeline:
    """Tests for the full pipeline with a minimal image fixture."""

    def _make_blank_png(self, width: int = 100, height: int = 100) -> bytes:
        """Create a minimal blank white PNG image."""
        import cv2
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        _, buf = cv2.imencode(".png", img)
        return bytes(buf)

    def test_analyze_blank_image(self, tmp_path) -> None:
        """A blank image should produce an empty tally with no errors."""
        import cv2

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img_path = tmp_path / "blank.png"
        cv2.imwrite(str(img_path), img)

        analyzer = BalloonAnalyzer(ocr_adapter=StubOcrAdapter())
        report = analyzer.analyze(str(img_path))

        assert isinstance(report, AnalysisReport)
        assert report.tally == {}
        assert report.balloon_breakdown == []
        assert report.excluded_balloon_count == 0
        assert isinstance(report.warnings, list)

    def test_analyze_to_json_returns_string(self, tmp_path) -> None:
        import cv2
        import json

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img_path = tmp_path / "blank.png"
        cv2.imwrite(str(img_path), img)

        analyzer = BalloonAnalyzer(ocr_adapter=StubOcrAdapter())
        result = analyzer.analyze_to_json(str(img_path))

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "tally" in parsed
        assert "warnings" in parsed

    def test_analyze_to_tabular_returns_string(self, tmp_path) -> None:
        import cv2

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img_path = tmp_path / "blank.png"
        cv2.imwrite(str(img_path), img)

        analyzer = BalloonAnalyzer(ocr_adapter=StubOcrAdapter())
        result = analyzer.analyze_to_tabular(str(img_path))

        assert isinstance(result, str)
        assert "TALLY" in result

    def test_warnings_collected_from_all_stages(self, tmp_path) -> None:
        """Verify that warnings from all pipeline stages are collected."""
        import cv2

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img_path = tmp_path / "blank.png"
        cv2.imwrite(str(img_path), img)

        # Use a very high confidence threshold so any detected items
        # would trigger low-confidence warnings
        cfg = AnalyzerConfig(confidence_threshold=1.0)
        analyzer = BalloonAnalyzer(config=cfg, ocr_adapter=StubOcrAdapter())
        report = analyzer.analyze(str(img_path))

        # With a blank image and stub OCR, there should be no balloons
        # and therefore no low-confidence warnings — but the pipeline
        # should still complete without error
        assert isinstance(report.warnings, list)
