"""Command-line interface for the Balloon Quantity Analyzer."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from balloon_quantity_analyzer.analyzer import BalloonAnalyzer
from balloon_quantity_analyzer.models import (
    AnalyzerConfig,
    InvalidConfigurationError,
    UnreadableFileError,
    UnsupportedFormatError,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="balloon-analyzer",
        description="Analyze mechanical assembly drawings to detect Find number "
        "balloons, interpret quantity multipliers, and produce a per-Find-number tally.",
    )
    parser.add_argument(
        "file_path",
        help="Path to the drawing file to analyze (PDF, PNG, JPEG, or TIFF).",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="CONFIG_FILE",
        help="Path to a JSON configuration file with keys: proximity_radius, "
        "confidence_threshold, custom_multiplier_phrases.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "tabular"],
        default="json",
        dest="output_format",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="OUTPUT_FILE",
        help="Write output to this file instead of stdout.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show progress messages during analysis.",
    )
    return parser


def _load_config(config_path: str) -> AnalyzerConfig:
    """Load an AnalyzerConfig from a JSON file.

    Expected JSON keys (all optional):
      - proximity_radius (float)
      - confidence_threshold (float)
      - custom_multiplier_phrases (dict[str, int])
    """
    with open(config_path) as fh:
        data = json.load(fh)

    return AnalyzerConfig(
        proximity_radius=float(data.get("proximity_radius", 50.0)),
        confidence_threshold=float(data.get("confidence_threshold", 0.5)),
        custom_multiplier_phrases=data.get("custom_multiplier_phrases", {}),
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``balloon-analyzer`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Set up logging ----------------------------------------------------
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stderr,
        )

    # Load configuration ------------------------------------------------
    config: AnalyzerConfig | None = None
    if args.config is not None:
        try:
            config = _load_config(args.config)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Error loading config file: {exc}", file=sys.stderr)
            sys.exit(1)

    # Create analyzer and run -------------------------------------------
    try:
        analyzer = BalloonAnalyzer(config=config)
        if args.output_format == "tabular":
            result = analyzer.analyze_to_tabular(args.file_path)
        else:
            result = analyzer.analyze_to_json(args.file_path)
    except (UnsupportedFormatError, UnreadableFileError, InvalidConfigurationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Write output ------------------------------------------------------
    if args.output is not None:
        try:
            with open(args.output, "w") as fh:
                fh.write(result)
        except OSError as exc:
            print(f"Error writing output file: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print(result)
