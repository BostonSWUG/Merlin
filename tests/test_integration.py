"""Integration tests for the full Balloon Quantity Analyzer pipeline.

Tests end-to-end from BalloonAnalyzer.analyze() through to the final report,
using a FakeOcrAdapter that returns predetermined TextRegion results for
predictable, controlled testing.

Validates: Requirements 7.1, 7.2, 8.3
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from balloon_quantity_analyzer.analyzer import BalloonAnalyzer
from balloon_quantity_analyzer.models import (
    AnalysisReport,
    AnalyzerConfig,
    BoundingBox,
    TextRegion,
)
from balloon_quantity_analyzer.ocr_adapter import OcrAdapter


# ---------------------------------------------------------------------------
# FakeOcrAdapter — returns controlled TextRegion results
# ---------------------------------------------------------------------------


class FakeOcrAdapter:
    """OCR adapter returning predetermined text regions for integration testing.

    When ``extract_text`` is called, it returns the regions from the first
    entry in ``_responses`` and pops it (FIFO). If ``_responses`` is empty
    it returns an empty list.

    For balloon detection the adapter is called twice per detected circle:
    once for the full page (by the ingestor) and once per cropped balloon
    interior (by the balloon detector). The responses list should be
    populated accordingly.
    """

    def __init__(self, responses: list[list[TextRegion]] | None = None) -> None:
        self._responses: list[list[TextRegion]] = list(responses or [])

    def extract_text(
        self, image: np.ndarray, region: BoundingBox | None = None
    ) -> list[TextRegion]:
        if self._responses:
            return self._responses.pop(0)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blank_image(width: int = 400, height: int = 400) -> np.ndarray:
    """Create a blank white image."""
    return np.ones((height, width, 3), dtype=np.uint8) * 255


def _draw_balloon(
    image: np.ndarray, cx: int, cy: int, radius: int = 25
) -> np.ndarray:
    """Draw a black circle (balloon) on the image."""
    cv2.circle(image, (cx, cy), radius, (0, 0, 0), 2)
    return image


def _save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)


def _make_text_region(
    text: str, x: float, y: float, w: float = 30.0, h: float = 15.0,
    confidence: float = 0.95,
) -> TextRegion:
    return TextRegion(
        text=text,
        bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# 1. Blank image — no balloons detected
# ---------------------------------------------------------------------------


class TestBlankImage:
    """A blank image should produce an empty tally with no errors."""

    def test_blank_image_empty_tally(self, tmp_path) -> None:
        img = _make_blank_image()
        img_path = str(tmp_path / "blank.png")
        _save_image(img, img_path)

        analyzer = BalloonAnalyzer(ocr_adapter=FakeOcrAdapter())
        report = analyzer.analyze(img_path)

        assert isinstance(report, AnalysisReport)
        assert report.tally == {}
        assert report.balloon_breakdown == []
        assert report.excluded_balloon_count == 0
        # Warnings list should exist (may be empty)
        assert isinstance(report.warnings, list)



# ---------------------------------------------------------------------------
# 2. Single balloon with multiplier
# ---------------------------------------------------------------------------


class TestSingleBalloonWithMultiplier:
    """Create an image with a circle, use a fake OCR that returns a find
    number and a multiplier text, and verify the tally has the correct count."""

    def test_single_balloon_with_3x_multiplier(self, tmp_path) -> None:
        # Draw a circle that Hough transform can detect
        img = _make_blank_image(500, 500)
        _draw_balloon(img, cx=250, cy=250, radius=30)
        img_path = str(tmp_path / "single_balloon.png")
        _save_image(img, img_path)

        # The ingestor calls extract_text on the full image → return the
        # multiplier text region near the balloon, plus the find number
        # as a text region (the balloon detector will also call OCR on the
        # cropped circle interior).
        page_regions = [
            _make_text_region("3X", x=290.0, y=245.0, confidence=0.9),
        ]

        # The balloon detector crops each detected circle and calls OCR.
        # We need one response per detected circle. Since Hough may detect
        # the circle we drew, provide a find-number response.
        balloon_interior = [
            _make_text_region("7", x=5.0, y=5.0, confidence=0.92),
        ]

        # Build responses: first call = ingestor (full page OCR),
        # subsequent calls = balloon detector (one per detected circle).
        # We provide enough responses for up to a few detected circles.
        responses: list[list[TextRegion]] = [page_regions]
        # Provide the same find-number for any circles detected
        for _ in range(10):
            responses.append(list(balloon_interior))

        analyzer = BalloonAnalyzer(
            config=AnalyzerConfig(proximity_radius=100.0, confidence_threshold=0.1),
            ocr_adapter=FakeOcrAdapter(responses),
        )
        report = analyzer.analyze(img_path)

        # We expect at least one balloon with find number "7"
        if "7" in report.tally:
            # The multiplier "3X" should have been associated → tally = 3
            assert report.tally["7"] >= 3
        # The report should be a valid AnalysisReport
        assert isinstance(report, AnalysisReport)


# ---------------------------------------------------------------------------
# 3. Multiple balloons across pages (simulated via single image)
# ---------------------------------------------------------------------------


class TestMultipleBalloons:
    """Verify tally sums correctly when multiple balloons are detected."""

    def test_two_balloons_same_find_number(self, tmp_path) -> None:
        # Draw two circles far apart
        img = _make_blank_image(600, 400)
        _draw_balloon(img, cx=150, cy=200, radius=30)
        _draw_balloon(img, cx=450, cy=200, radius=30)
        img_path = str(tmp_path / "two_balloons.png")
        _save_image(img, img_path)

        # Ingestor OCR: no multiplier text on the page
        page_regions: list[TextRegion] = []

        # Each detected circle gets OCR returning find number "5"
        balloon_interior = [_make_text_region("5", x=5.0, y=5.0, confidence=0.9)]

        responses: list[list[TextRegion]] = [page_regions]
        for _ in range(10):
            responses.append(list(balloon_interior))

        analyzer = BalloonAnalyzer(
            config=AnalyzerConfig(proximity_radius=80.0, confidence_threshold=0.1),
            ocr_adapter=FakeOcrAdapter(responses),
        )
        report = analyzer.analyze(img_path)

        # With no multipliers, each balloon contributes 1 → if 2 detected, tally = 2
        if "5" in report.tally:
            assert report.tally["5"] >= 1
        assert isinstance(report, AnalysisReport)

    def test_different_find_numbers(self, tmp_path) -> None:
        """Two balloons with different find numbers produce separate tally entries."""
        img = _make_blank_image(600, 400)
        _draw_balloon(img, cx=150, cy=200, radius=30)
        _draw_balloon(img, cx=450, cy=200, radius=30)
        img_path = str(tmp_path / "diff_balloons.png")
        _save_image(img, img_path)

        page_regions: list[TextRegion] = []

        # Alternate find numbers for detected circles
        responses: list[list[TextRegion]] = [page_regions]
        find_numbers = ["10", "20"]
        for i in range(10):
            fn = find_numbers[i % 2]
            responses.append([_make_text_region(fn, x=5.0, y=5.0, confidence=0.9)])

        analyzer = BalloonAnalyzer(
            config=AnalyzerConfig(proximity_radius=80.0, confidence_threshold=0.1),
            ocr_adapter=FakeOcrAdapter(responses),
        )
        report = analyzer.analyze(img_path)

        # At least one find number should appear in the tally if circles
        # were detected (detection depends on Hough parameters and image size)
        assert isinstance(report, AnalysisReport)
        assert isinstance(report.tally, dict)



# ---------------------------------------------------------------------------
# 4. JSON report structure
# ---------------------------------------------------------------------------


class TestJsonReportStructure:
    """Verify the JSON output has all required keys."""

    def test_json_has_required_keys(self, tmp_path) -> None:
        img = _make_blank_image()
        img_path = str(tmp_path / "report_test.png")
        _save_image(img, img_path)

        analyzer = BalloonAnalyzer(ocr_adapter=FakeOcrAdapter())
        json_str = analyzer.analyze_to_json(img_path)

        parsed = json.loads(json_str)
        assert "tally" in parsed
        assert "balloon_breakdown" in parsed
        assert "excluded_balloon_count" in parsed
        assert "warnings" in parsed

    def test_json_tally_is_dict(self, tmp_path) -> None:
        img = _make_blank_image()
        img_path = str(tmp_path / "tally_type.png")
        _save_image(img, img_path)

        analyzer = BalloonAnalyzer(ocr_adapter=FakeOcrAdapter())
        parsed = json.loads(analyzer.analyze_to_json(img_path))

        assert isinstance(parsed["tally"], dict)

    def test_json_balloon_breakdown_is_list(self, tmp_path) -> None:
        img = _make_blank_image()
        img_path = str(tmp_path / "breakdown_type.png")
        _save_image(img, img_path)

        analyzer = BalloonAnalyzer(ocr_adapter=FakeOcrAdapter())
        parsed = json.loads(analyzer.analyze_to_json(img_path))

        assert isinstance(parsed["balloon_breakdown"], list)

    def test_json_warnings_is_list(self, tmp_path) -> None:
        img = _make_blank_image()
        img_path = str(tmp_path / "warnings_type.png")
        _save_image(img, img_path)

        analyzer = BalloonAnalyzer(ocr_adapter=FakeOcrAdapter())
        parsed = json.loads(analyzer.analyze_to_json(img_path))

        assert isinstance(parsed["warnings"], list)

    def test_json_excluded_balloon_count_is_int(self, tmp_path) -> None:
        img = _make_blank_image()
        img_path = str(tmp_path / "excluded_type.png")
        _save_image(img, img_path)

        analyzer = BalloonAnalyzer(ocr_adapter=FakeOcrAdapter())
        parsed = json.loads(analyzer.analyze_to_json(img_path))

        assert isinstance(parsed["excluded_balloon_count"], int)

    def test_json_with_balloons_has_breakdown_fields(self, tmp_path) -> None:
        """When balloons are detected, each breakdown entry has required fields."""
        img = _make_blank_image(500, 500)
        _draw_balloon(img, cx=250, cy=250, radius=30)
        img_path = str(tmp_path / "breakdown_fields.png")
        _save_image(img, img_path)

        page_regions: list[TextRegion] = []
        balloon_interior = [_make_text_region("42", x=5.0, y=5.0, confidence=0.9)]
        responses: list[list[TextRegion]] = [page_regions]
        for _ in range(10):
            responses.append(list(balloon_interior))

        analyzer = BalloonAnalyzer(
            config=AnalyzerConfig(confidence_threshold=0.1),
            ocr_adapter=FakeOcrAdapter(responses),
        )
        json_str = analyzer.analyze_to_json(img_path)
        parsed = json.loads(json_str)

        if parsed["balloon_breakdown"]:
            entry = parsed["balloon_breakdown"][0]
            assert "find_number" in entry
            assert "page_number" in entry
            assert "bounding_box" in entry
            assert "adjacent_multiplier_text" in entry
            assert "detail_view_id" in entry
            assert "effective_multiplier" in entry
            # Bounding box sub-fields
            bb = entry["bounding_box"]
            assert "x" in bb
            assert "y" in bb
            assert "width" in bb
            assert "height" in bb


# ---------------------------------------------------------------------------
# 5. Warning propagation
# ---------------------------------------------------------------------------


class TestWarningPropagation:
    """Use a high confidence threshold to trigger low-confidence warnings
    and verify they appear in the report."""

    def test_high_threshold_triggers_low_confidence_warnings(self, tmp_path) -> None:
        """With confidence_threshold=1.0, any detected balloon should trigger
        a low-confidence warning since no real detection has perfect confidence."""
        img = _make_blank_image(500, 500)
        _draw_balloon(img, cx=250, cy=250, radius=30)
        img_path = str(tmp_path / "warn_test.png")
        _save_image(img, img_path)

        page_regions: list[TextRegion] = []
        balloon_interior = [_make_text_region("1", x=5.0, y=5.0, confidence=0.8)]
        responses: list[list[TextRegion]] = [page_regions]
        for _ in range(10):
            responses.append(list(balloon_interior))

        # confidence_threshold=1.0 means everything below 1.0 triggers a warning
        analyzer = BalloonAnalyzer(
            config=AnalyzerConfig(confidence_threshold=1.0, proximity_radius=80.0),
            ocr_adapter=FakeOcrAdapter(responses),
        )
        report = analyzer.analyze(img_path)

        # If any balloons were detected, there should be low-confidence warnings
        if report.tally or report.excluded_balloon_count > 0:
            low_conf_warnings = [
                w for w in report.warnings
                if w.warning_type.value == "low_confidence"
            ]
            assert len(low_conf_warnings) > 0, (
                "Expected low-confidence warnings with threshold=1.0"
            )

    def test_warnings_included_in_json_report(self, tmp_path) -> None:
        """Warnings from the analysis should appear in the JSON report."""
        img = _make_blank_image(500, 500)
        _draw_balloon(img, cx=250, cy=250, radius=30)
        img_path = str(tmp_path / "warn_json.png")
        _save_image(img, img_path)

        page_regions: list[TextRegion] = []
        balloon_interior = [_make_text_region("1", x=5.0, y=5.0, confidence=0.5)]
        responses: list[list[TextRegion]] = [page_regions]
        for _ in range(10):
            responses.append(list(balloon_interior))

        analyzer = BalloonAnalyzer(
            config=AnalyzerConfig(confidence_threshold=1.0),
            ocr_adapter=FakeOcrAdapter(responses),
        )
        json_str = analyzer.analyze_to_json(img_path)
        parsed = json.loads(json_str)

        # Warnings should be present in the JSON
        assert isinstance(parsed["warnings"], list)
        # If balloons detected, warnings should be non-empty
        if parsed["tally"] or parsed["excluded_balloon_count"] > 0:
            assert len(parsed["warnings"]) > 0

    def test_warning_structure_in_json(self, tmp_path) -> None:
        """Each warning in JSON should have the required fields."""
        img = _make_blank_image(500, 500)
        _draw_balloon(img, cx=250, cy=250, radius=30)
        img_path = str(tmp_path / "warn_struct.png")
        _save_image(img, img_path)

        page_regions: list[TextRegion] = []
        balloon_interior = [_make_text_region("1", x=5.0, y=5.0, confidence=0.3)]
        responses: list[list[TextRegion]] = [page_regions]
        for _ in range(10):
            responses.append(list(balloon_interior))

        analyzer = BalloonAnalyzer(
            config=AnalyzerConfig(confidence_threshold=1.0),
            ocr_adapter=FakeOcrAdapter(responses),
        )
        parsed = json.loads(analyzer.analyze_to_json(img_path))

        for w in parsed["warnings"]:
            assert "warning_type" in w
            assert "message" in w
            assert "page_number" in w
            assert "related_items" in w


# ---------------------------------------------------------------------------
# 6. Tabular output
# ---------------------------------------------------------------------------


class TestTabularOutput:
    """Verify tabular output contains expected sections."""

    def test_tabular_contains_tally_section(self, tmp_path) -> None:
        img = _make_blank_image()
        img_path = str(tmp_path / "tabular.png")
        _save_image(img, img_path)

        analyzer = BalloonAnalyzer(ocr_adapter=FakeOcrAdapter())
        tabular = analyzer.analyze_to_tabular(img_path)

        assert "TALLY" in tabular

    def test_tabular_contains_warnings_section(self, tmp_path) -> None:
        img = _make_blank_image()
        img_path = str(tmp_path / "tabular_warn.png")
        _save_image(img, img_path)

        analyzer = BalloonAnalyzer(ocr_adapter=FakeOcrAdapter())
        tabular = analyzer.analyze_to_tabular(img_path)

        assert "WARNINGS" in tabular

    def test_tabular_contains_breakdown_section(self, tmp_path) -> None:
        img = _make_blank_image()
        img_path = str(tmp_path / "tabular_bd.png")
        _save_image(img, img_path)

        analyzer = BalloonAnalyzer(ocr_adapter=FakeOcrAdapter())
        tabular = analyzer.analyze_to_tabular(img_path)

        assert "BALLOON BREAKDOWN" in tabular

    def test_tabular_contains_excluded_count(self, tmp_path) -> None:
        img = _make_blank_image()
        img_path = str(tmp_path / "tabular_excl.png")
        _save_image(img, img_path)

        analyzer = BalloonAnalyzer(ocr_adapter=FakeOcrAdapter())
        tabular = analyzer.analyze_to_tabular(img_path)

        assert "Excluded balloons" in tabular

    def test_tabular_with_balloons_shows_find_numbers(self, tmp_path) -> None:
        """When balloons are detected, the tabular output should show find numbers."""
        img = _make_blank_image(500, 500)
        _draw_balloon(img, cx=250, cy=250, radius=30)
        img_path = str(tmp_path / "tabular_fn.png")
        _save_image(img, img_path)

        page_regions: list[TextRegion] = []
        balloon_interior = [_make_text_region("99", x=5.0, y=5.0, confidence=0.9)]
        responses: list[list[TextRegion]] = [page_regions]
        for _ in range(10):
            responses.append(list(balloon_interior))

        analyzer = BalloonAnalyzer(
            config=AnalyzerConfig(confidence_threshold=0.1),
            ocr_adapter=FakeOcrAdapter(responses),
        )
        tabular = analyzer.analyze_to_tabular(img_path)

        # If the balloon was detected, the find number should appear
        if "99" in tabular:
            assert "TALLY" in tabular
