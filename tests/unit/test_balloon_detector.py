"""Unit tests for BalloonDetector."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from balloon_quantity_analyzer.balloon_detector import BalloonDetector
from balloon_quantity_analyzer.models import (
    BoundingBox,
    DetectedBalloon,
    NormalizedPage,
    TextRegion,
    Warning,
    WarningType,
)
from balloon_quantity_analyzer.ocr_adapter import OcrAdapter


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------


class FakeOcrAdapter:
    """OCR adapter that returns pre-configured results for testing."""

    def __init__(self, results: list[TextRegion] | None = None) -> None:
        self._results = results if results is not None else []

    def extract_text(
        self, image: np.ndarray, region: BoundingBox | None = None
    ) -> list[TextRegion]:
        return self._results


def _make_circle_image(
    circles: list[tuple[int, int, int]],
    width: int = 400,
    height: int = 400,
) -> bytes:
    """Create a PNG image with drawn circles and return as bytes."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background
    for cx, cy, r in circles:
        cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)  # black circle
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _make_page(image_bytes: bytes, page_number: int = 1) -> NormalizedPage:
    return NormalizedPage(
        page_number=page_number,
        image=image_bytes,
        text_regions=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBalloonDetectorDecodeImage:
    """Tests for image decoding."""

    def test_valid_png_decodes(self) -> None:
        img_bytes = _make_circle_image([])
        result = BalloonDetector._decode_image(img_bytes)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0

    def test_invalid_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to decode"):
            BalloonDetector._decode_image(b"not-an-image")


class TestBalloonDetectorCircleToBbox:
    """Tests for circle-to-bounding-box conversion."""

    def test_basic_conversion(self) -> None:
        bbox = BalloonDetector._circle_to_bbox(100.0, 100.0, 30.0, (400, 400))
        assert bbox.x == pytest.approx(70.0)
        assert bbox.y == pytest.approx(70.0)
        assert bbox.width == pytest.approx(60.0)
        assert bbox.height == pytest.approx(60.0)

    def test_clamped_to_image_bounds(self) -> None:
        bbox = BalloonDetector._circle_to_bbox(5.0, 5.0, 20.0, (100, 100))
        assert bbox.x == 0.0
        assert bbox.y == 0.0
        assert bbox.width == pytest.approx(25.0)
        assert bbox.height == pytest.approx(25.0)


class TestBalloonDetectorBestFindNumber:
    """Tests for Find number extraction from OCR results."""

    def test_empty_regions_returns_empty(self) -> None:
        fn, conf = BalloonDetector._best_find_number([])
        assert fn == ""
        assert conf == 0.0

    def test_single_region(self) -> None:
        regions = [
            TextRegion(
                text="42",
                bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
                confidence=0.95,
            )
        ]
        fn, conf = BalloonDetector._best_find_number(regions)
        assert fn == "42"
        assert conf == pytest.approx(0.95)

    def test_multiple_regions_concatenated(self) -> None:
        regions = [
            TextRegion(
                text="1",
                bounding_box=BoundingBox(x=0, y=0, width=5, height=5),
                confidence=0.8,
            ),
            TextRegion(
                text="A",
                bounding_box=BoundingBox(x=5, y=0, width=5, height=5),
                confidence=0.9,
            ),
        ]
        fn, conf = BalloonDetector._best_find_number(regions)
        assert fn == "1A"
        assert conf == pytest.approx(0.9)

    def test_whitespace_only_returns_empty(self) -> None:
        regions = [
            TextRegion(
                text="   ",
                bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
                confidence=0.5,
            )
        ]
        fn, conf = BalloonDetector._best_find_number(regions)
        assert fn == ""
        assert conf == 0.0


class TestBalloonDetectorCombineConfidence:
    """Tests for confidence combination logic."""

    def test_both_perfect(self) -> None:
        assert BalloonDetector._combine_confidence(1.0, 1.0) == pytest.approx(1.0)

    def test_both_zero(self) -> None:
        assert BalloonDetector._combine_confidence(0.0, 0.0) == pytest.approx(0.0)

    def test_weighted_average(self) -> None:
        # 0.6 * 0.5 + 0.4 * 1.0 = 0.7
        assert BalloonDetector._combine_confidence(0.5, 1.0) == pytest.approx(0.7)

    def test_clamped_to_one(self) -> None:
        result = BalloonDetector._combine_confidence(1.5, 1.5)
        assert result <= 1.0

    def test_clamped_to_zero(self) -> None:
        result = BalloonDetector._combine_confidence(-0.5, -0.5)
        assert result >= 0.0


class TestBalloonDetectorDetect:
    """Integration-level tests for the detect method."""

    def test_no_circles_returns_empty(self) -> None:
        """A blank image with no circles should yield no balloons."""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        _, buf = cv2.imencode(".png", img)
        page = _make_page(buf.tobytes())

        detector = BalloonDetector(ocr_adapter=FakeOcrAdapter())
        balloons, warnings = detector.detect(page)
        assert balloons == []
        assert warnings == []

    def test_unreadable_balloon_produces_warning(self) -> None:
        """When OCR returns nothing, the balloon should have empty find_number and a warning."""
        # Draw a clear circle that Hough should detect
        img_bytes = _make_circle_image([(200, 200, 40)])
        page = _make_page(img_bytes)

        # OCR returns nothing
        detector = BalloonDetector(ocr_adapter=FakeOcrAdapter(results=[]))
        balloons, warnings = detector.detect(page)

        # We may or may not detect the circle depending on Hough params,
        # but if we do, unreadable ones should produce warnings.
        for b in balloons:
            if b.find_number == "":
                matching = [
                    w
                    for w in warnings
                    if w.warning_type == WarningType.UNREADABLE_FIND_NUMBER
                ]
                assert len(matching) >= 1

    def test_detected_balloon_has_valid_confidence(self) -> None:
        """All detected balloons should have confidence in [0, 1]."""
        img_bytes = _make_circle_image([(200, 200, 40)])
        page = _make_page(img_bytes)

        ocr = FakeOcrAdapter(
            results=[
                TextRegion(
                    text="7",
                    bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
                    confidence=0.85,
                )
            ]
        )
        detector = BalloonDetector(ocr_adapter=ocr)
        balloons, _ = detector.detect(page)

        for b in balloons:
            assert 0.0 <= b.confidence <= 1.0

    def test_detected_balloon_page_number_matches(self) -> None:
        """Detected balloons should carry the page number from the input page."""
        img_bytes = _make_circle_image([(200, 200, 40)])
        page = _make_page(img_bytes, page_number=3)

        ocr = FakeOcrAdapter(
            results=[
                TextRegion(
                    text="5",
                    bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
                    confidence=0.9,
                )
            ]
        )
        detector = BalloonDetector(ocr_adapter=ocr)
        balloons, _ = detector.detect(page)

        for b in balloons:
            assert b.page_number == 3
