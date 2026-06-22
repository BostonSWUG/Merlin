"""OCR adapter with strategy pattern for swapping between Tesseract and Textract."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pytesseract

from balloon_quantity_analyzer.models import BoundingBox, TextRegion


class OcrAdapter(Protocol):
    """Abstract OCR interface allowing different OCR backends."""

    def extract_text(
        self, image: np.ndarray, region: BoundingBox | None = None
    ) -> list[TextRegion]:
        """Extract text from an image, optionally within a specific region.

        Args:
            image: Input image as a numpy ndarray (BGR or grayscale).
            region: Optional bounding box to restrict OCR to a sub-region.

        Returns:
            List of TextRegion objects with text, bounding boxes, and confidence.
        """
        ...


class TesseractOcrAdapter:
    """Local OCR using pytesseract."""

    def __init__(self, tesseract_config: str = "") -> None:
        """
        Parameters
        ----------
        tesseract_config:
            Extra Tesseract CLI config string, e.g. ``"--psm 7"`` for
            single-line mode or ``"--psm 10"`` for single-character mode.
        """
        self._config = tesseract_config

    def _crop_image(
        self, image: np.ndarray, region: BoundingBox
    ) -> tuple[np.ndarray, float, float]:
        """Crop image to the given region, returning the crop and its origin offset."""
        x = int(region.x)
        y = int(region.y)
        w = int(region.width)
        h = int(region.height)
        cropped = image[y : y + h, x : x + w]
        return cropped, float(x), float(y)

    def extract_text(
        self, image: np.ndarray, region: BoundingBox | None = None
    ) -> list[TextRegion]:
        """Extract text from an image using Tesseract OCR.

        Args:
            image: Input image as a numpy ndarray (BGR or grayscale).
            region: Optional bounding box to restrict OCR to a sub-region.

        Returns:
            List of TextRegion objects with text, bounding boxes, and confidence.
        """
        offset_x = 0.0
        offset_y = 0.0
        target = image

        if region is not None:
            target, offset_x, offset_y = self._crop_image(image, region)

        # pytesseract.image_to_data returns a dict with parallel lists
        data = pytesseract.image_to_data(target, output_type=pytesseract.Output.DICT, config=self._config)

        results: list[TextRegion] = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue

            conf = float(data["conf"][i])
            # pytesseract returns -1 for entries it couldn't score
            if conf < 0:
                continue

            # Normalize confidence from 0-100 to 0.0-1.0
            confidence = conf / 100.0

            bbox = BoundingBox(
                x=float(data["left"][i]) + offset_x,
                y=float(data["top"][i]) + offset_y,
                width=float(data["width"][i]),
                height=float(data["height"][i]),
            )

            results.append(
                TextRegion(text=text, bounding_box=bbox, confidence=confidence)
            )

        return results


class TextractOcrAdapter:
    """AWS Textract-based OCR for higher accuracy.

    This is a stub implementation. Full Textract integration requires
    AWS credentials and the boto3 SDK.
    """

    def extract_text(
        self, image: np.ndarray, region: BoundingBox | None = None
    ) -> list[TextRegion]:
        """Extract text using AWS Textract.

        Raises:
            NotImplementedError: Textract integration is not yet implemented.
        """
        raise NotImplementedError(
            "AWS Textract OCR adapter is not yet implemented. "
            "Use TesseractOcrAdapter for local OCR processing."
        )
