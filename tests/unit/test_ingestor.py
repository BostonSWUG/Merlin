"""Unit tests for the DrawingIngestor module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from balloon_quantity_analyzer.ingestor import (
    DrawingIngestor,
    _detect_format,
)
from balloon_quantity_analyzer.models import (
    BoundingBox,
    NormalizedPage,
    TextRegion,
    UnreadableFileError,
    UnsupportedFormatError,
)
from balloon_quantity_analyzer.ocr_adapter import OcrAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeOcrAdapter:
    """A simple OCR adapter that returns a fixed list of TextRegions."""

    def __init__(self, regions: list[TextRegion] | None = None) -> None:
        self._regions = regions or []

    def extract_text(
        self, image: np.ndarray, region: BoundingBox | None = None
    ) -> list[TextRegion]:
        return self._regions


def _write_png(path: str, width: int = 40, height: int = 30) -> None:
    """Write a minimal valid PNG file."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


def _write_jpeg(path: str, width: int = 40, height: int = 30) -> None:
    """Write a minimal valid JPEG file."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
# Format detection tests
# ---------------------------------------------------------------------------


class TestDetectFormat:
    """Tests for _detect_format helper."""

    def test_png_by_extension_and_magic(self, tmp_path: Path) -> None:
        p = tmp_path / "drawing.png"
        _write_png(str(p))
        assert _detect_format(str(p)) == "image"

    def test_jpeg_by_extension_and_magic(self, tmp_path: Path) -> None:
        p = tmp_path / "photo.jpg"
        _write_jpeg(str(p))
        assert _detect_format(str(p)) == "image"

    def test_jpeg_alternate_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "photo.jpeg"
        _write_jpeg(str(p))
        assert _detect_format(str(p)) == "image"

    def test_unsupported_extension_and_no_magic(self, tmp_path: Path) -> None:
        p = tmp_path / "data.xyz"
        p.write_text("not a real image")
        with pytest.raises(UnsupportedFormatError):
            _detect_format(str(p))

    def test_file_not_found(self, tmp_path: Path) -> None:
        p = tmp_path / "missing.png"
        with pytest.raises(UnreadableFileError, match="not found"):
            _detect_format(str(p))

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.png"
        p.write_bytes(b"")
        with pytest.raises(UnreadableFileError, match="empty"):
            _detect_format(str(p))

    def test_magic_bytes_override_extension(self, tmp_path: Path) -> None:
        """A file with .txt extension but PNG magic bytes should be detected as image."""
        p = tmp_path / "sneaky.txt"
        # Write a valid PNG to a .txt file
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "temp.png"), img)
        png_data = (tmp_path / "temp.png").read_bytes()
        p.write_bytes(png_data)
        assert _detect_format(str(p)) == "image"

    def test_known_extension_unknown_magic(self, tmp_path: Path) -> None:
        """A file with .png extension but garbage content should still detect by extension."""
        p = tmp_path / "fake.png"
        p.write_bytes(b"this is not a png at all!!")
        assert _detect_format(str(p)) == "image"


# ---------------------------------------------------------------------------
# Image ingestion tests
# ---------------------------------------------------------------------------


class TestIngestImage:
    """Tests for image file ingestion (PNG, JPEG, TIFF)."""

    def test_ingest_png_returns_single_page(self, tmp_path: Path) -> None:
        p = tmp_path / "test.png"
        _write_png(str(p), width=50, height=40)

        adapter = FakeOcrAdapter()
        ingestor = DrawingIngestor(ocr_adapter=adapter)
        pages = ingestor.ingest(str(p))

        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert isinstance(pages[0].image, bytes)
        assert len(pages[0].image) > 0

    def test_ingest_jpeg_returns_single_page(self, tmp_path: Path) -> None:
        p = tmp_path / "test.jpg"
        _write_jpeg(str(p))

        adapter = FakeOcrAdapter()
        ingestor = DrawingIngestor(ocr_adapter=adapter)
        pages = ingestor.ingest(str(p))

        assert len(pages) == 1
        assert pages[0].page_number == 1

    def test_ingest_image_uses_ocr_adapter(self, tmp_path: Path) -> None:
        p = tmp_path / "test.png"
        _write_png(str(p))

        regions = [
            TextRegion(
                text="3X",
                bounding_box=BoundingBox(x=10, y=20, width=30, height=15),
                confidence=0.9,
            )
        ]
        adapter = FakeOcrAdapter(regions=regions)
        ingestor = DrawingIngestor(ocr_adapter=adapter)
        pages = ingestor.ingest(str(p))

        assert len(pages[0].text_regions) == 1
        assert pages[0].text_regions[0].text == "3X"

    def test_ingest_image_png_bytes_are_valid(self, tmp_path: Path) -> None:
        """The image field should contain valid PNG bytes."""
        p = tmp_path / "test.png"
        _write_png(str(p), width=20, height=15)

        adapter = FakeOcrAdapter()
        ingestor = DrawingIngestor(ocr_adapter=adapter)
        pages = ingestor.ingest(str(p))

        # PNG magic bytes
        assert pages[0].image[:4] == b"\x89PNG"

    def test_ingest_corrupted_image_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.png"
        # Write PNG magic bytes but garbage content
        p.write_bytes(b"\x89PNG\r\n\x1a\ncorrupted data here")

        adapter = FakeOcrAdapter()
        ingestor = DrawingIngestor(ocr_adapter=adapter)
        with pytest.raises(UnreadableFileError):
            ingestor.ingest(str(p))


# ---------------------------------------------------------------------------
# Unsupported format tests
# ---------------------------------------------------------------------------


class TestUnsupportedFormat:
    """Tests for unsupported file format handling."""

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "data.docx"
        p.write_bytes(b"PK\x03\x04some zip content")

        adapter = FakeOcrAdapter()
        ingestor = DrawingIngestor(ocr_adapter=adapter)
        with pytest.raises(UnsupportedFormatError):
            ingestor.ingest(str(p))

    def test_missing_file_raises_unreadable(self, tmp_path: Path) -> None:
        adapter = FakeOcrAdapter()
        ingestor = DrawingIngestor(ocr_adapter=adapter)
        with pytest.raises(UnreadableFileError):
            ingestor.ingest(str(tmp_path / "nonexistent.png"))


# ---------------------------------------------------------------------------
# PDF ingestion tests (mocked)
# ---------------------------------------------------------------------------


class TestIngestPdf:
    """Tests for PDF ingestion using mocked pdf2image and pdfplumber."""

    @patch("balloon_quantity_analyzer.ingestor.pdfplumber")
    @patch("balloon_quantity_analyzer.ingestor.pdf2image")
    def test_single_page_pdf(self, mock_pdf2image, mock_pdfplumber, tmp_path: Path) -> None:
        # Create a fake PDF file with correct magic bytes
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

        # Mock pdf2image to return one PIL-like image
        pil_img = MagicMock()
        pil_img.__array__ = MagicMock(
            return_value=np.zeros((100, 200, 3), dtype=np.uint8)
        )
        # Make np.array(pil_img) work
        pil_img.__array_interface__ = None
        mock_pdf2image.convert_from_path.return_value = [pil_img]

        # Mock pdfplumber
        mock_page = MagicMock()
        mock_page.extract_words.return_value = [
            {"text": "DETAIL", "x0": 10, "top": 20, "x1": 60, "bottom": 35},
        ]
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        adapter = FakeOcrAdapter()
        ingestor = DrawingIngestor(ocr_adapter=adapter)

        # We need np.array to work on the mock — patch it
        with patch("balloon_quantity_analyzer.ingestor.np.array", return_value=np.zeros((100, 200, 3), dtype=np.uint8)):
            pages = ingestor.ingest(str(pdf_path))

        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert len(pages[0].text_regions) == 1
        assert pages[0].text_regions[0].text == "DETAIL"
        assert pages[0].text_regions[0].confidence == 1.0

    @patch("balloon_quantity_analyzer.ingestor.pdfplumber")
    @patch("balloon_quantity_analyzer.ingestor.pdf2image")
    def test_multi_page_pdf_preserves_order(self, mock_pdf2image, mock_pdfplumber, tmp_path: Path) -> None:
        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake multi page")

        # 3 pages
        imgs = []
        for _ in range(3):
            m = MagicMock()
            imgs.append(m)
        mock_pdf2image.convert_from_path.return_value = imgs

        mock_pages = []
        for i in range(3):
            mp = MagicMock()
            mp.extract_words.return_value = [
                {"text": f"Page{i+1}", "x0": 0, "top": 0, "x1": 50, "bottom": 20}
            ]
            mock_pages.append(mp)
        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages
        mock_pdfplumber.open.return_value = mock_pdf

        adapter = FakeOcrAdapter()
        ingestor = DrawingIngestor(ocr_adapter=adapter)

        with patch("balloon_quantity_analyzer.ingestor.np.array", return_value=np.zeros((100, 200, 3), dtype=np.uint8)):
            pages = ingestor.ingest(str(pdf_path))

        assert len(pages) == 3
        assert [p.page_number for p in pages] == [1, 2, 3]
        assert pages[0].text_regions[0].text == "Page1"
        assert pages[1].text_regions[0].text == "Page2"
        assert pages[2].text_regions[0].text == "Page3"

    @patch("balloon_quantity_analyzer.ingestor.pdf2image")
    def test_pdf_rasterize_failure_raises(self, mock_pdf2image, tmp_path: Path) -> None:
        pdf_path = tmp_path / "bad.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 corrupted")

        mock_pdf2image.convert_from_path.side_effect = RuntimeError("poppler error")

        adapter = FakeOcrAdapter()
        ingestor = DrawingIngestor(ocr_adapter=adapter)
        with pytest.raises(UnreadableFileError, match="rasterize"):
            ingestor.ingest(str(pdf_path))

    @patch("balloon_quantity_analyzer.ingestor.pdf2image")
    @patch("balloon_quantity_analyzer.ingestor.pdfplumber")
    def test_pdf_pdfplumber_failure_raises(self, mock_pdfplumber, mock_pdf2image, tmp_path: Path) -> None:
        pdf_path = tmp_path / "bad2.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 corrupted")

        mock_pdf2image.convert_from_path.return_value = [MagicMock()]
        mock_pdfplumber.open.side_effect = RuntimeError("pdfplumber error")

        adapter = FakeOcrAdapter()
        ingestor = DrawingIngestor(ocr_adapter=adapter)
        with pytest.raises(UnreadableFileError, match="text extraction"):
            ingestor.ingest(str(pdf_path))
