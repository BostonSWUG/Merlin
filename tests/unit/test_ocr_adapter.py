"""Unit tests for the OCR adapter module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from balloon_quantity_analyzer.models import BoundingBox, TextRegion
from balloon_quantity_analyzer.ocr_adapter import (
    OcrAdapter,
    TesseractOcrAdapter,
    TextractOcrAdapter,
)


class TestOcrAdapterProtocol:
    """Verify that concrete adapters satisfy the OcrAdapter protocol."""

    def test_tesseract_adapter_is_ocr_adapter(self) -> None:
        adapter: OcrAdapter = TesseractOcrAdapter()
        assert hasattr(adapter, "extract_text")

    def test_textract_adapter_is_ocr_adapter(self) -> None:
        adapter: OcrAdapter = TextractOcrAdapter()
        assert hasattr(adapter, "extract_text")


class TestTextractOcrAdapter:
    """TextractOcrAdapter should raise NotImplementedError."""

    def test_extract_text_raises_not_implemented(self) -> None:
        adapter = TextractOcrAdapter()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            adapter.extract_text(image)

    def test_extract_text_with_region_raises_not_implemented(self) -> None:
        adapter = TextractOcrAdapter()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        region = BoundingBox(x=10, y=10, width=50, height=50)
        with pytest.raises(NotImplementedError):
            adapter.extract_text(image, region=region)


class TestTesseractOcrAdapter:
    """TesseractOcrAdapter tests using mocked pytesseract calls."""

    def _make_tesseract_data(
        self,
        texts: list[str],
        lefts: list[int],
        tops: list[int],
        widths: list[int],
        heights: list[int],
        confs: list[float],
    ) -> dict:
        return {
            "text": texts,
            "left": lefts,
            "top": tops,
            "width": widths,
            "height": heights,
            "conf": confs,
        }

    @patch("balloon_quantity_analyzer.ocr_adapter.pytesseract")
    def test_extract_text_returns_text_regions(self, mock_pytesseract) -> None:
        mock_pytesseract.Output.DICT = "dict"
        mock_pytesseract.image_to_data.return_value = self._make_tesseract_data(
            texts=["Hello", "World"],
            lefts=[10, 100],
            tops=[20, 50],
            widths=[80, 60],
            heights=[30, 25],
            confs=[95.0, 87.5],
        )

        adapter = TesseractOcrAdapter()
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        results = adapter.extract_text(image)

        assert len(results) == 2
        assert results[0].text == "Hello"
        assert results[0].bounding_box == BoundingBox(x=10.0, y=20.0, width=80.0, height=30.0)
        assert results[0].confidence == pytest.approx(0.95)
        assert results[1].text == "World"
        assert results[1].confidence == pytest.approx(0.875)

    @patch("balloon_quantity_analyzer.ocr_adapter.pytesseract")
    def test_extract_text_skips_empty_text(self, mock_pytesseract) -> None:
        mock_pytesseract.Output.DICT = "dict"
        mock_pytesseract.image_to_data.return_value = self._make_tesseract_data(
            texts=["Hello", "", "  ", "World"],
            lefts=[10, 20, 30, 40],
            tops=[10, 20, 30, 40],
            widths=[50, 50, 50, 50],
            heights=[20, 20, 20, 20],
            confs=[90.0, 80.0, 70.0, 85.0],
        )

        adapter = TesseractOcrAdapter()
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        results = adapter.extract_text(image)

        assert len(results) == 2
        assert results[0].text == "Hello"
        assert results[1].text == "World"

    @patch("balloon_quantity_analyzer.ocr_adapter.pytesseract")
    def test_extract_text_skips_negative_confidence(self, mock_pytesseract) -> None:
        mock_pytesseract.Output.DICT = "dict"
        mock_pytesseract.image_to_data.return_value = self._make_tesseract_data(
            texts=["Hello", "Bad"],
            lefts=[10, 50],
            tops=[10, 50],
            widths=[40, 40],
            heights=[20, 20],
            confs=[92.0, -1.0],
        )

        adapter = TesseractOcrAdapter()
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        results = adapter.extract_text(image)

        assert len(results) == 1
        assert results[0].text == "Hello"

    @patch("balloon_quantity_analyzer.ocr_adapter.pytesseract")
    def test_extract_text_with_region_applies_offset(self, mock_pytesseract) -> None:
        mock_pytesseract.Output.DICT = "dict"
        mock_pytesseract.image_to_data.return_value = self._make_tesseract_data(
            texts=["Cropped"],
            lefts=[5],
            tops=[8],
            widths=[30],
            heights=[15],
            confs=[88.0],
        )

        adapter = TesseractOcrAdapter()
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        region = BoundingBox(x=50.0, y=60.0, width=100.0, height=80.0)
        results = adapter.extract_text(image, region=region)

        assert len(results) == 1
        # Bounding box should be offset by the region origin
        assert results[0].bounding_box.x == pytest.approx(55.0)  # 5 + 50
        assert results[0].bounding_box.y == pytest.approx(68.0)  # 8 + 60
        assert results[0].bounding_box.width == pytest.approx(30.0)
        assert results[0].bounding_box.height == pytest.approx(15.0)

    @patch("balloon_quantity_analyzer.ocr_adapter.pytesseract")
    def test_extract_text_empty_image_returns_empty(self, mock_pytesseract) -> None:
        mock_pytesseract.Output.DICT = "dict"
        mock_pytesseract.image_to_data.return_value = self._make_tesseract_data(
            texts=[], lefts=[], tops=[], widths=[], heights=[], confs=[]
        )

        adapter = TesseractOcrAdapter()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = adapter.extract_text(image)

        assert results == []

    @patch("balloon_quantity_analyzer.ocr_adapter.pytesseract")
    def test_confidence_normalized_to_0_1(self, mock_pytesseract) -> None:
        mock_pytesseract.Output.DICT = "dict"
        mock_pytesseract.image_to_data.return_value = self._make_tesseract_data(
            texts=["A", "B"],
            lefts=[0, 0],
            tops=[0, 0],
            widths=[10, 10],
            heights=[10, 10],
            confs=[100.0, 0.0],
        )

        adapter = TesseractOcrAdapter()
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        results = adapter.extract_text(image)

        assert results[0].confidence == pytest.approx(1.0)
        assert results[1].confidence == pytest.approx(0.0)
