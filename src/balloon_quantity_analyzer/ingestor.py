"""Drawing Ingestor — normalizes PDF and image files into NormalizedPage objects."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pdf2image
import pdfplumber

from balloon_quantity_analyzer.models import (
    BoundingBox,
    NormalizedPage,
    TextRegion,
    UnreadableFileError,
    UnsupportedFormatError,
)
from balloon_quantity_analyzer.ocr_adapter import OcrAdapter


# Supported extensions mapped to format category
_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
}

# Magic byte signatures for format detection
_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"%PDF", "pdf"),
    (b"\x89PNG", "image"),
    (b"\xff\xd8\xff", "image"),  # JPEG
    (b"II\x2a\x00", "image"),  # TIFF little-endian
    (b"MM\x00\x2a", "image"),  # TIFF big-endian
]


def _detect_format(file_path: str) -> str:
    """Detect file format using extension and magic bytes.

    Returns:
        "pdf" or "image".

    Raises:
        UnsupportedFormatError: If the format cannot be determined or is unsupported.
        UnreadableFileError: If the file cannot be read.
    """
    path = Path(file_path)

    if not path.exists():
        raise UnreadableFileError(f"File not found: {file_path}")

    # Try extension first
    ext = path.suffix.lower()
    fmt_from_ext = _EXTENSION_MAP.get(ext)

    # Try magic bytes
    fmt_from_magic: str | None = None
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
    except OSError as exc:
        raise UnreadableFileError(f"Cannot read file: {file_path}") from exc

    if len(header) == 0:
        raise UnreadableFileError(f"File is empty: {file_path}")

    for magic, fmt in _MAGIC_BYTES:
        if header[: len(magic)] == magic:
            fmt_from_magic = fmt
            break

    # Prefer magic bytes when available; fall back to extension
    detected = fmt_from_magic or fmt_from_ext
    if detected is None:
        raise UnsupportedFormatError(
            f"Unsupported format for file: {file_path} (extension={ext!r})"
        )

    return detected


def _image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a numpy image array as PNG bytes."""
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise UnreadableFileError("Failed to encode image as PNG")
    return bytes(buf)


def _pdfplumber_text_regions(pdf_page: pdfplumber.page.Page) -> list[TextRegion]:
    """Extract text regions from a pdfplumber page using word-level bounding boxes."""
    regions: list[TextRegion] = []
    words = pdf_page.extract_words()
    for w in words:
        text = w.get("text", "").strip()
        if not text:
            continue
        x0 = float(w["x0"])
        top = float(w["top"])
        x1 = float(w["x1"])
        bottom = float(w["bottom"])
        bbox = BoundingBox(
            x=x0,
            y=top,
            width=x1 - x0,
            height=bottom - top,
        )
        # pdfplumber doesn't provide per-word confidence; use 1.0 for embedded text
        regions.append(TextRegion(text=text, bounding_box=bbox, confidence=1.0))
    return regions


class DrawingIngestor:
    """Accepts a drawing file and produces normalized page representations.

    Supports PDF (via pdf2image + pdfplumber) and image files
    (PNG, JPEG, TIFF via OpenCV + OCR adapter).
    """

    def __init__(self, ocr_adapter: OcrAdapter) -> None:
        self._ocr = ocr_adapter

    def ingest(self, file_path: str) -> list[NormalizedPage]:
        """Ingest a drawing file and return one NormalizedPage per page.

        Args:
            file_path: Path to a PDF or image file.

        Returns:
            List of NormalizedPage objects, one per page, in page order.

        Raises:
            UnsupportedFormatError: If the file format is not supported.
            UnreadableFileError: If the file cannot be opened or is corrupted.
        """
        fmt = _detect_format(file_path)

        if fmt == "pdf":
            return self._ingest_pdf(file_path)
        else:
            return self._ingest_image(file_path)

    # ------------------------------------------------------------------
    # PDF ingestion
    # ------------------------------------------------------------------

    def _ingest_pdf(self, file_path: str) -> list[NormalizedPage]:
        """Rasterize PDF pages and extract embedded text."""
        try:
            pil_images = pdf2image.convert_from_path(file_path)
        except Exception as exc:
            raise UnreadableFileError(
                f"Failed to rasterize PDF: {file_path}"
            ) from exc

        try:
            pdf = pdfplumber.open(file_path)
        except Exception as exc:
            raise UnreadableFileError(
                f"Failed to open PDF for text extraction: {file_path}"
            ) from exc

        pages: list[NormalizedPage] = []
        try:
            for idx, pil_img in enumerate(pil_images):
                # Convert PIL image to numpy BGR for consistency
                np_img = np.array(pil_img)
                if np_img.ndim == 3 and np_img.shape[2] == 3:
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

                png_bytes = _image_to_png_bytes(np_img)

                # Extract text from pdfplumber (embedded text)
                text_regions: list[TextRegion] = []
                if idx < len(pdf.pages):
                    text_regions = _pdfplumber_text_regions(pdf.pages[idx])

                pages.append(
                    NormalizedPage(
                        page_number=idx + 1,
                        image=png_bytes,
                        text_regions=text_regions,
                    )
                )
        finally:
            pdf.close()

        return pages

    # ------------------------------------------------------------------
    # Image ingestion
    # ------------------------------------------------------------------

    def _ingest_image(self, file_path: str) -> list[NormalizedPage]:
        """Load an image file and extract text via OCR."""
        try:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise UnreadableFileError(
                f"Failed to read image: {file_path}"
            ) from exc

        if img is None:
            raise UnreadableFileError(
                f"Failed to decode image (corrupted or unreadable): {file_path}"
            )

        png_bytes = _image_to_png_bytes(img)
        text_regions = self._ocr.extract_text(img)

        return [
            NormalizedPage(
                page_number=1,
                image=png_bytes,
                text_regions=text_regions,
            )
        ]
