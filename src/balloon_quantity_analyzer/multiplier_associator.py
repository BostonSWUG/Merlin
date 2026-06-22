"""Multiplier Associator — links multipliers to balloons by spatial proximity
and combines balloon-adjacent multipliers with Detail view multipliers."""

from __future__ import annotations

import math

from balloon_quantity_analyzer.models import (
    AssociatedBalloon,
    BoundingBox,
    DetectedBalloon,
    DetailView,
    ParsedMultiplier,
    Warning,
    WarningType,
)


def _center(bb: BoundingBox) -> tuple[float, float]:
    """Return the center (cx, cy) of a bounding box."""
    return bb.x + bb.width / 2.0, bb.y + bb.height / 2.0


def _distance(a: BoundingBox, b: BoundingBox) -> float:
    """Euclidean distance between the centers of two bounding boxes."""
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


class MultiplierAssociator:
    """Associates parsed multipliers with detected balloons using spatial proximity."""

    def __init__(self, proximity_radius: float) -> None:
        self.proximity_radius = proximity_radius

    def associate(
        self,
        balloons: list[DetectedBalloon],
        multipliers: list[ParsedMultiplier],
        detail_views: list[DetailView],
    ) -> tuple[list[AssociatedBalloon], list[Warning]]:
        """Associate multipliers with balloons and return results with warnings."""
        warnings: list[Warning] = []

        # --- Step 1: Build proximity mappings (same page only) ---
        # For each multiplier index, which balloon indices are within radius?
        mult_to_balloons: dict[int, list[int]] = {}
        # For each balloon index, which multiplier indices are within radius?
        balloon_to_mults: dict[int, list[int]] = {}

        for mi, mult in enumerate(multipliers):
            nearby: list[int] = []
            for bi, balloon in enumerate(balloons):
                if mult.page_number != balloon.page_number:
                    continue
                if _distance(mult.bounding_box, balloon.bounding_box) <= self.proximity_radius:
                    nearby.append(bi)
            mult_to_balloons[mi] = nearby

        for bi in range(len(balloons)):
            balloon_to_mults[bi] = []
        for mi, nearby in mult_to_balloons.items():
            for bi in nearby:
                balloon_to_mults[bi].append(mi)

        # --- Step 2: Flag ambiguous multipliers (near 2+ balloons) ---
        ambiguous_mults: set[int] = set()
        for mi, nearby in mult_to_balloons.items():
            if len(nearby) >= 2:
                ambiguous_mults.add(mi)
                mult = multipliers[mi]
                candidate_fns = [balloons[bi].find_number or f"balloon@{bi}" for bi in nearby]
                warnings.append(
                    Warning(
                        warning_type=WarningType.AMBIGUOUS_MULTIPLIER,
                        message=(
                            f"Multiplier '{mult.raw_text}' (value={mult.value}) is within "
                            f"proximity radius of multiple balloons: {candidate_fns}"
                        ),
                        page_number=mult.page_number,
                        related_items=[mult.raw_text] + candidate_fns,
                    )
                )

        # --- Step 3: Flag balloons with multiple nearby multipliers ---
        multi_mult_balloons: set[int] = set()
        for bi, mults_nearby in balloon_to_mults.items():
            # Filter out already-ambiguous multipliers
            valid = [mi for mi in mults_nearby if mi not in ambiguous_mults]
            if len(valid) >= 2:
                multi_mult_balloons.add(bi)
                balloon = balloons[bi]
                competing = [multipliers[mi].raw_text for mi in valid]
                warnings.append(
                    Warning(
                        warning_type=WarningType.MULTIPLE_MULTIPLIERS,
                        message=(
                            f"Balloon '{balloon.find_number or f'balloon@{bi}'}' has multiple "
                            f"nearby multipliers: {competing}"
                        ),
                        page_number=balloon.page_number,
                        related_items=[balloon.find_number or f"balloon@{bi}"] + competing,
                    )
                )

        # --- Step 4: Determine adjacent multiplier per balloon ---
        adjacent_mult: dict[int, tuple[int, str]] = {}  # bi -> (value, raw_text)
        for bi, mults_nearby in balloon_to_mults.items():
            if bi in multi_mult_balloons:
                continue  # ambiguous — none applied
            valid = [mi for mi in mults_nearby if mi not in ambiguous_mults]
            if len(valid) == 1:
                mi = valid[0]
                adjacent_mult[bi] = (multipliers[mi].value, multipliers[mi].raw_text)

        # --- Step 5: Determine Detail view multiplier per balloon ---
        detail_mult: dict[int, tuple[int, str]] = {}  # bi -> (value, identifier)
        for dv in detail_views:
            dv_value = dv.multiplier if dv.multiplier is not None else 1
            for bi in dv.contained_balloon_indices:
                if bi < 0 or bi >= len(balloons):
                    continue
                if bi in detail_mult and dv_value != 1:
                    # Already has a detail view multiplier — overlapping
                    # The DetailViewResolver should have warned, but we still
                    # don't apply overlapping DV multipliers.
                    existing_val, _ = detail_mult[bi]
                    if existing_val != 1:
                        # Both have non-trivial multipliers → don't apply either
                        detail_mult[bi] = (1, "")
                    else:
                        detail_mult[bi] = (dv_value, dv.identifier)
                else:
                    detail_mult[bi] = (dv_value, dv.identifier)

        # --- Step 6: Build AssociatedBalloon list ---
        results: list[AssociatedBalloon] = []
        for bi, balloon in enumerate(balloons):
            adj_val, adj_text = adjacent_mult.get(bi, (1, None))  # type: ignore[assignment]
            dv_val, dv_id = detail_mult.get(bi, (1, None))  # type: ignore[assignment]

            effective = adj_val * dv_val

            results.append(
                AssociatedBalloon(
                    find_number=balloon.find_number,
                    page_number=balloon.page_number,
                    bounding_box=balloon.bounding_box,
                    adjacent_multiplier_text=adj_text if adj_val != 1 else adj_text,
                    adjacent_multiplier_value=adj_val,
                    detail_view_id=dv_id,
                    detail_view_multiplier=dv_val,
                    effective_multiplier=effective,
                    confidence=balloon.confidence,
                )
            )

        return results, warnings
