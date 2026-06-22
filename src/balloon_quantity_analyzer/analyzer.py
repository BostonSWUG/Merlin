"""BalloonAnalyzer — main orchestrator wiring all pipeline stages together.

Pipeline: Ingestor → BalloonDetector + MultiplierParser → DetailViewResolver
          → MultiplierAssociator → TallyAggregator → ReportGenerator
"""

from __future__ import annotations

import logging
import sys

from balloon_quantity_analyzer.balloon_detector import BalloonDetector

logger = logging.getLogger("balloon_quantity_analyzer")
from balloon_quantity_analyzer.confidence import check_low_confidence
from balloon_quantity_analyzer.config import validate_config
from balloon_quantity_analyzer.detail_view_resolver import DetailViewResolver
from balloon_quantity_analyzer.ingestor import DrawingIngestor
from balloon_quantity_analyzer.models import (
    AnalysisReport,
    AnalyzerConfig,
    DetectedBalloon,
    DetailView,
    ParsedMultiplier,
    TallyResult,
    Warning,
)
from balloon_quantity_analyzer.multiplier_associator import MultiplierAssociator
from balloon_quantity_analyzer.multiplier_parser import MultiplierParser
from balloon_quantity_analyzer.ocr_adapter import OcrAdapter, TesseractOcrAdapter
from balloon_quantity_analyzer.vector_balloon_detector import detect_balloons_from_pdf
from balloon_quantity_analyzer.report_generator import ReportGenerator
from balloon_quantity_analyzer.tally_aggregator import TallyAggregator


class BalloonAnalyzer:
    """Orchestrates the full balloon quantity analysis pipeline.

    Accepts an optional ``AnalyzerConfig`` (defaults applied when *None*)
    and an optional ``OcrAdapter`` (defaults to ``TesseractOcrAdapter``).

    The pipeline is:
        Ingest → Detect balloons + Parse multipliers → Resolve detail views
        → Associate multipliers → Check low confidence → Tally → Report
    """

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        ocr_adapter: OcrAdapter | None = None,
    ) -> None:
        self._config = validate_config(config)
        self._ocr: OcrAdapter = ocr_adapter or TesseractOcrAdapter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, file_path: str) -> AnalysisReport:
        """Run the full analysis pipeline on *file_path*.

        Raises:
            UnsupportedFormatError: If the file format is not supported.
            UnreadableFileError: If the file cannot be opened or is corrupted.
            InvalidConfigurationError: If the config is invalid (caught at init).

        Returns:
            An ``AnalysisReport`` containing the tally, per-balloon breakdown,
            excluded balloon count, and all warnings collected from every stage.
        """
        all_warnings: list[Warning] = []

        # --- Stage 1: Ingest ---
        logger.info("Ingesting file: %s", file_path)
        ingestor = DrawingIngestor(self._ocr)
        pages = ingestor.ingest(file_path)
        logger.info("Ingested %d page(s)", len(pages))

        # --- Try vector-based detection for PDFs first ---
        is_pdf = file_path.lower().endswith(".pdf")
        vector_balloons: list[DetectedBalloon] = []
        vector_multipliers: list[ParsedMultiplier] = []
        vector_warnings: list[Warning] = []

        if is_pdf:
            logger.info("Attempting vector-based balloon detection...")
            vector_balloons, vector_multipliers, vector_warnings = detect_balloons_from_pdf(file_path)
            named_balloons = [b for b in vector_balloons if b.find_number]
            logger.info("Vector detection found %d balloons (%d with Find numbers)",
                        len(vector_balloons), len(named_balloons))

        use_vector = is_pdf and any(b.find_number for b in vector_balloons)

        # --- Stage 2 & 3: Detect balloons + Parse multipliers + Resolve detail views ---
        all_balloons: list[DetectedBalloon] = []
        all_multipliers: list[ParsedMultiplier] = []
        all_detail_views: list[DetailView] = []

        if use_vector:
            # Use vector detection results
            logger.info("Using vector-based detection results")
            all_balloons = vector_balloons
            all_multipliers = vector_multipliers
            all_warnings.extend(vector_warnings)

            # Still parse multipliers from text regions for any the vector
            # detector might have missed
            multiplier_parser = MultiplierParser(
                custom_phrases=self._config.custom_multiplier_phrases or None,
            )
            for page in pages:
                text_multipliers, mult_warnings = multiplier_parser.parse(page.text_regions)
                all_warnings.extend(mult_warnings)
                # Only add text multipliers that aren't already found by vector detection
                for tm in text_multipliers:
                    is_dup = any(
                        abs(tm.bounding_box.x - vm.bounding_box.x) < 5
                        and abs(tm.bounding_box.y - vm.bounding_box.y) < 5
                        for vm in all_multipliers
                    )
                    if not is_dup:
                        all_multipliers.append(tm)

            # Resolve detail views
            detail_view_resolver = DetailViewResolver()
            for page in pages:
                detail_views, dv_warnings = detail_view_resolver.resolve(page, all_balloons)
                all_detail_views.extend(detail_views)
                all_warnings.extend(dv_warnings)
        else:
            # Fall back to raster-based Hough + OCR detection
            if is_pdf:
                logger.info("Vector detection found no balloons, falling back to raster detection")

            balloon_detector = BalloonDetector(self._ocr)
            multiplier_parser = MultiplierParser(
                custom_phrases=self._config.custom_multiplier_phrases or None,
            )
            detail_view_resolver = DetailViewResolver()

            for page in pages:
                # Detect balloons on this page
                logger.info("Page %d: detecting balloons...", page.page_number)
                balloons, balloon_warnings = balloon_detector.detect(page)
                logger.info("Page %d: found %d candidate circle(s)", page.page_number, len(balloons))
                all_warnings.extend(balloon_warnings)

                # Parse multipliers from this page's text regions
                multipliers, mult_warnings = multiplier_parser.parse(page.text_regions)
                all_warnings.extend(mult_warnings)

                # Fix page_number on parsed multipliers (MultiplierParser sets 0)
                fixed_multipliers: list[ParsedMultiplier] = []
                for m in multipliers:
                    fixed_multipliers.append(
                        ParsedMultiplier(
                            value=m.value,
                            raw_text=m.raw_text,
                            bounding_box=m.bounding_box,
                            page_number=page.page_number,
                            confidence=m.confidence,
                        )
                    )

                all_balloons.extend(balloons)
                all_multipliers.extend(fixed_multipliers)

                # Detail view resolver
                detail_views, dv_warnings = detail_view_resolver.resolve(page, all_balloons)
                all_detail_views.extend(detail_views)
                all_warnings.extend(dv_warnings)
            all_warnings.extend(dv_warnings)

        # --- Stage 4: Associate multipliers with balloons ---
        associator = MultiplierAssociator(
            proximity_radius=self._config.proximity_radius,
        )
        associated_balloons, assoc_warnings = associator.associate(
            all_balloons, all_multipliers, all_detail_views,
        )
        all_warnings.extend(assoc_warnings)

        # --- Stage 5: Check low confidence ---
        all_warnings.extend(
            check_low_confidence(
                list(all_balloons),
                self._config.confidence_threshold,
                "detected balloon",
            )
        )
        all_warnings.extend(
            check_low_confidence(
                list(all_multipliers),
                self._config.confidence_threshold,
                "parsed multiplier",
            )
        )
        all_warnings.extend(
            check_low_confidence(
                list(associated_balloons),
                self._config.confidence_threshold,
                "association",
            )
        )

        # --- Stage 6: Aggregate tally ---
        aggregator = TallyAggregator()
        tally_result = aggregator.aggregate(associated_balloons)

        # --- Stage 7: Build report ---
        return AnalysisReport(
            tally=tally_result.tally,
            balloon_breakdown=tally_result.balloon_breakdown,
            excluded_balloon_count=tally_result.excluded_balloon_count,
            warnings=all_warnings,
        )

    def analyze_to_json(self, file_path: str) -> str:
        """Run analysis and return the result as a JSON string."""
        report = self.analyze(file_path)
        return self._report_to_json(report)

    def analyze_to_tabular(self, file_path: str) -> str:
        """Run analysis and return the result as human-readable tabular text."""
        report = self.analyze(file_path)
        return self._report_to_tabular(report)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _report_to_json(report: AnalysisReport) -> str:
        generator = ReportGenerator()
        tally_result = TallyResult(
            tally=report.tally,
            balloon_breakdown=report.balloon_breakdown,
            excluded_balloon_count=report.excluded_balloon_count,
        )
        return generator.generate_json(tally_result, report.warnings)

    @staticmethod
    def _report_to_tabular(report: AnalysisReport) -> str:
        generator = ReportGenerator()
        tally_result = TallyResult(
            tally=report.tally,
            balloon_breakdown=report.balloon_breakdown,
            excluded_balloon_count=report.excluded_balloon_count,
        )
        return generator.generate_tabular(tally_result, report.warnings)
