"""Balloon detection using Hough Circle Transform and OCR-based Find number extraction."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("balloon_quantity_analyzer")

from balloon_quantity_analyzer.models import (
    BoundingBox,
    DetectedBalloon,
    NormalizedPage,
    Warning,
    WarningType,
)
from balloon_quantity_analyzer.ocr_adapter import OcrAdapter


class BalloonDetector:
    """Detects balloon graphics on drawing pages and extracts Find numbers via OCR."""

    def __init__(self, ocr_adapter: OcrAdapter) -> None:
        self._ocr = ocr_adapter
        # Create a dedicated OCR adapter for balloon interiors if using Tesseract.
        # PSM 7 = "Treat the image as a single text line" — ideal for short
        # numbers inside balloons.
        from balloon_quantity_analyzer.ocr_adapter import TesseractOcrAdapter
        if isinstance(ocr_adapter, TesseractOcrAdapter):
            self._balloon_ocr: OcrAdapter = TesseractOcrAdapter(
                tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )
        else:
            self._balloon_ocr = ocr_adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, page: NormalizedPage
    ) -> tuple[list[DetectedBalloon], list[Warning]]:
        """Identify balloons on *page* and extract their Find numbers.

        Returns a tuple of (detected balloons, warnings).
        """
        image = self._decode_image(page.image)
        gray = self._to_grayscale(image)
        h, w = gray.shape[:2]
        logger.info("  Image size: %dx%d", w, h)

        # Downscale large images for faster Hough circle detection.
        # Coordinates are mapped back to the original resolution for OCR.
        max_dim = 3000
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            logger.info("  Downscaled to %dx%d (%.0f%%) for circle detection",
                        small_gray.shape[1], small_gray.shape[0], scale * 100)
        else:
            small_gray = gray

        logger.info("  Running circle detection...")
        circles_scaled, accumulators = self._find_circles(small_gray)

        # Map circle coordinates back to original resolution and filter
        # out circles that are too large to be balloons. On a typical
        # engineering drawing, balloons are roughly 0.3–1.5% of the image
        # width. We cap at 2% to be generous.
        max_balloon_radius = max(h, w) * 0.01
        circles_unfiltered = [
            (cx / scale, cy / scale, r / scale)
            for cx, cy, r in circles_scaled
        ]
        circles = []
        filtered_accumulators = []
        for (cx, cy, r), acc in zip(circles_unfiltered, accumulators):
            if r > max_balloon_radius:
                logger.info("  Skipping circle at (%.0f,%.0f) r=%.0f — too large for a balloon", cx, cy, r)
            else:
                circles.append((cx, cy, r))
                filtered_accumulators.append(acc)
        accumulators = filtered_accumulators
        logger.info("  %d circle(s) after filtering, running OCR on each...", len(circles))

        balloons: list[DetectedBalloon] = []
        warnings: list[Warning] = []

        for i, ((cx, cy, radius), accumulator) in enumerate(zip(circles, accumulators)):
            logger.info("  OCR on circle %d/%d (r=%.0f)...", i + 1, len(circles), radius)
            bbox = self._circle_to_bbox(cx, cy, radius, (h, w))
            crop = self._crop_circle(image, cx, cy, radius)
            processed = self._preprocess_for_ocr(crop)

            text_regions = self._balloon_ocr.extract_text(processed)

            find_number, ocr_confidence = self._best_find_number(text_regions)

            # Normalize Hough accumulator to [0, 1] range.
            # OpenCV accumulator values are unbounded; we use a sigmoid-like
            # mapping: hough_score = min(accumulator / 200, 1.0)
            hough_score = min(float(accumulator) / 200.0, 1.0)
            confidence = self._combine_confidence(hough_score, ocr_confidence)

            balloon = DetectedBalloon(
                find_number=find_number,
                page_number=page.page_number,
                bounding_box=bbox,
                confidence=confidence,
            )
            balloons.append(balloon)

            if find_number == "":
                warnings.append(
                    Warning(
                        warning_type=WarningType.UNREADABLE_FIND_NUMBER,
                        message=(
                            f"Balloon at ({bbox.x:.0f}, {bbox.y:.0f}) on page "
                            f"{page.page_number} has an unreadable Find number."
                        ),
                        page_number=page.page_number,
                        related_items=[
                            f"bbox({bbox.x:.0f},{bbox.y:.0f},{bbox.width:.0f},{bbox.height:.0f})"
                        ],
                    )
                )

        return balloons, warnings

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_image(png_bytes: bytes) -> np.ndarray:
        """Decode PNG bytes into a BGR numpy array."""
        buf = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode page image from PNG bytes.")
        return image

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # Circle detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_circles(
        gray: np.ndarray,
    ) -> tuple[list[tuple[float, float, float]], list[float]]:
        """Run Canny + Hough Circle Transform.

        Returns parallel lists of (cx, cy, radius) and accumulator values.
        """
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        rows = blurred.shape[0]
        cols = blurred.shape[1]

        result = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=rows // 16 if rows >= 16 else 1,
            param1=100,
            param2=50,
            minRadius=3,
            maxRadius=min(rows, cols) // 30,  # balloons are ~1-3% of page size
        )

        if result is None:
            return [], []

        # result shape: (1, N, 3) — each row is [cx, cy, radius]
        # Cap at 50 circles to avoid OCR overload on busy drawings
        detected = result[0][:50]
        circles: list[tuple[float, float, float]] = []
        accumulators: list[float] = []

        for row in detected:
            cx, cy, r = float(row[0]), float(row[1]), float(row[2])
            circles.append((cx, cy, r))
            # OpenCV HoughCircles with HOUGH_GRADIENT does not directly
            # expose the accumulator value per circle.  The circles are
            # returned sorted by accumulator strength (strongest first).
            # We approximate a score from the ordering position.
            accumulators.append(200.0)  # placeholder — see note below

        # Assign descending scores based on sort order (strongest first).
        n = len(accumulators)
        for i in range(n):
            # Score linearly from 200 (first) down to 100 (last).
            accumulators[i] = 200.0 - (100.0 * i / max(n - 1, 1))

        return circles, accumulators

    # ------------------------------------------------------------------
    # Cropping
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_circle(
        image: np.ndarray, cx: float, cy: float, radius: float
    ) -> np.ndarray:
        """Crop the square region enclosing the circle, clamped to image bounds."""
        h, w = image.shape[:2]
        x1 = max(int(cx - radius), 0)
        y1 = max(int(cy - radius), 0)
        x2 = min(int(cx + radius), w)
        y2 = min(int(cy + radius), h)
        return image[y1:y2, x1:x2]

    @staticmethod
    def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
        """Preprocess a cropped balloon image to improve OCR accuracy.

        Steps:
        1. Convert to grayscale
        2. Upscale small crops so text is at least ~40px tall
        3. Apply adaptive thresholding for clean black-on-white text
        4. Add white padding around the edges (Tesseract needs margin)
        """
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        min_dim = 150
        h, w = gray.shape[:2]
        if min(h, w) < min_dim:
            scale = min_dim / min(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )

        pad = 20
        padded = cv2.copyMakeBorder(
            binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
        )

        return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _circle_to_bbox(
        cx: float, cy: float, radius: float, shape: tuple[int, ...]
    ) -> BoundingBox:
        """Convert circle parameters to an axis-aligned BoundingBox."""
        h, w = shape[:2]
        x = max(cx - radius, 0.0)
        y = max(cy - radius, 0.0)
        bw = min(cx + radius, float(w)) - x
        bh = min(cy + radius, float(h)) - y
        return BoundingBox(x=x, y=y, width=bw, height=bh)

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_find_number(
        text_regions: list,
    ) -> tuple[str, float]:
        """Pick the best Find number from OCR results.

        Returns (find_number, ocr_confidence).  If nothing readable,
        returns ("", 0.0).
        """
        if not text_regions:
            return "", 0.0

        # Concatenate all detected text, pick the region with highest
        # confidence as the representative score.
        best_conf = 0.0
        parts: list[str] = []
        for region in text_regions:
            text = region.text.strip()
            if text:
                parts.append(text)
                if region.confidence > best_conf:
                    best_conf = region.confidence

        combined = "".join(parts).strip()
        if not combined:
            return "", 0.0

        return combined, best_conf

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_confidence(hough_score: float, ocr_confidence: float) -> float:
        """Combine Hough accumulator score and OCR confidence.

        Uses a weighted average (60 % Hough, 40 % OCR) clamped to [0, 1].
        """
        combined = 0.6 * hough_score + 0.4 * ocr_confidence
        return max(0.0, min(combined, 1.0))
