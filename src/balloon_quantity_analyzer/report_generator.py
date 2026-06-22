"""Report Generator — produces JSON and tabular text reports."""

from __future__ import annotations

import json
from typing import Any

from balloon_quantity_analyzer.models import (
    AnalysisReport,
    BalloonBreakdown,
    BoundingBox,
    TallyResult,
    Warning,
    WarningType,
)


class ReportGenerator:
    """Serializes tally results and warnings into JSON or tabular text."""

    def generate_json(
        self, tally_result: TallyResult, warnings: list[Warning]
    ) -> str:
        """Produce a JSON report with tally, per-balloon breakdown, and warnings.

        The output is deterministic: keys are emitted in a fixed order and the
        JSON is pretty-printed with an indent of 2 spaces.
        """
        report_dict = _build_report_dict(tally_result, warnings)
        return json.dumps(report_dict, indent=2)

    def generate_tabular(
        self, tally_result: TallyResult, warnings: list[Warning]
    ) -> str:
        """Produce a human-readable tabular text report."""
        lines: list[str] = []

        # --- Tally section ---
        lines.append("TALLY")
        lines.append("-" * 40)
        if tally_result.tally:
            max_fn_len = max(len(fn) for fn in tally_result.tally)
            col_width = max(max_fn_len, len("Find Number"))
            lines.append(f"{'Find Number':<{col_width}}  Count")
            lines.append(f"{'-' * col_width}  -----")
            for find_number, count in sorted(tally_result.tally.items(), key=lambda item: _sort_tally_key(item[0])):
                lines.append(f"{find_number:<{col_width}}  {count}")
        else:
            lines.append("(no items)")
        lines.append("")

        # --- Excluded balloons ---
        lines.append(
            f"Excluded balloons (unreadable): {tally_result.excluded_balloon_count}"
        )
        lines.append("")

        # --- Balloon breakdown section ---
        lines.append("BALLOON BREAKDOWN")
        lines.append("-" * 40)
        if tally_result.balloon_breakdown:
            for i, b in enumerate(tally_result.balloon_breakdown, 1):
                lines.append(
                    f"  {i}. Find={b.find_number}  Page={b.page_number}  "
                    f"Multiplier={b.effective_multiplier}  "
                    f"Adj={b.adjacent_multiplier_text or '-'}  "
                    f"Detail={b.detail_view_id or '-'}"
                )
        else:
            lines.append("  (no balloons)")
        lines.append("")

        # --- Warnings section ---
        lines.append("WARNINGS")
        lines.append("-" * 40)
        if warnings:
            for w in warnings:
                page_str = f"Page {w.page_number}" if w.page_number is not None else "N/A"
                lines.append(f"  [{w.warning_type.value}] {w.message} ({page_str})")
        else:
            lines.append("  (none)")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _sort_tally_key(find_number: str) -> tuple[int, str]:
    """Sort key that puts numeric Find numbers first in numerical order,
    then alphanumeric ones alphabetically."""
    try:
        return (0, find_number.zfill(10))
    except ValueError:
        return (1, find_number)


def _build_report_dict(
    tally_result: TallyResult, warnings: list[Warning]
) -> dict[str, Any]:
    """Build the dictionary that maps to the JSON schema."""
    sorted_tally = dict(
        sorted(tally_result.tally.items(), key=lambda item: _sort_tally_key(item[0]))
    )
    return {
        "tally": sorted_tally,
        "balloon_breakdown": [
            _balloon_breakdown_to_dict(b) for b in tally_result.balloon_breakdown
        ],
        "excluded_balloon_count": tally_result.excluded_balloon_count,
        "warnings": [_warning_to_dict(w) for w in warnings],
    }


def _balloon_breakdown_to_dict(b: BalloonBreakdown) -> dict[str, Any]:
    return {
        "find_number": b.find_number,
        "page_number": b.page_number,
        "bounding_box": {
            "x": b.bounding_box.x,
            "y": b.bounding_box.y,
            "width": b.bounding_box.width,
            "height": b.bounding_box.height,
        },
        "adjacent_multiplier_text": b.adjacent_multiplier_text,
        "detail_view_id": b.detail_view_id,
        "effective_multiplier": b.effective_multiplier,
    }


def _warning_to_dict(w: Warning) -> dict[str, Any]:
    return {
        "warning_type": w.warning_type.value,
        "message": w.message,
        "page_number": w.page_number,
        "related_items": list(w.related_items),
    }


# ---------------------------------------------------------------------------
# Round-trip parsing
# ---------------------------------------------------------------------------


def parse_json_report(json_str: str) -> AnalysisReport:
    """Parse a JSON report string back into an AnalysisReport.

    This is the inverse of ``ReportGenerator.generate_json`` and is used to
    verify the round-trip property (Property 10).
    """
    data = json.loads(json_str)

    tally: dict[str, int] = {
        str(k): int(v) for k, v in data["tally"].items()
    }

    balloon_breakdown = [
        BalloonBreakdown(
            find_number=b["find_number"],
            page_number=b["page_number"],
            bounding_box=BoundingBox(
                x=b["bounding_box"]["x"],
                y=b["bounding_box"]["y"],
                width=b["bounding_box"]["width"],
                height=b["bounding_box"]["height"],
            ),
            adjacent_multiplier_text=b["adjacent_multiplier_text"],
            detail_view_id=b["detail_view_id"],
            effective_multiplier=b["effective_multiplier"],
        )
        for b in data["balloon_breakdown"]
    ]

    warnings_list = [
        Warning(
            warning_type=WarningType(w["warning_type"]),
            message=w["message"],
            page_number=w["page_number"],
            related_items=w.get("related_items", []),
        )
        for w in data["warnings"]
    ]

    return AnalysisReport(
        tally=tally,
        balloon_breakdown=balloon_breakdown,
        excluded_balloon_count=data["excluded_balloon_count"],
        warnings=warnings_list,
    )
