"""Tally Aggregator — sums effective multipliers per Find number."""

from __future__ import annotations

from balloon_quantity_analyzer.models import (
    AssociatedBalloon,
    BalloonBreakdown,
    BoundingBox,
    TallyResult,
)


class TallyAggregator:
    """Aggregates associated balloons into a per-Find-number tally."""

    def aggregate(self, associated_balloons: list[AssociatedBalloon]) -> TallyResult:
        """Produce a tally mapping each Find number to the sum of effective
        multipliers, excluding balloons with empty Find numbers (counted
        separately), and a per-balloon breakdown."""

        tally: dict[str, int] = {}
        breakdown: list[BalloonBreakdown] = []
        excluded_count = 0

        for balloon in associated_balloons:
            if balloon.find_number == "":
                excluded_count += 1
                continue

            tally[balloon.find_number] = (
                tally.get(balloon.find_number, 0) + balloon.effective_multiplier
            )

            breakdown.append(
                BalloonBreakdown(
                    find_number=balloon.find_number,
                    page_number=balloon.page_number,
                    bounding_box=balloon.bounding_box,
                    adjacent_multiplier_text=balloon.adjacent_multiplier_text,
                    detail_view_id=balloon.detail_view_id,
                    effective_multiplier=balloon.effective_multiplier,
                )
            )

        return TallyResult(
            tally=tally,
            balloon_breakdown=breakdown,
            excluded_balloon_count=excluded_count,
        )
