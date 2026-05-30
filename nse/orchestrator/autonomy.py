"""Competence-aware autonomy (Phase 12.2).

NSE earns autonomy where it has *proven* itself calibrated and precise, and asks
for a human everywhere else. We slice logged (prediction, outcome) history into
**segments** (by default, the edited file), measure each segment's calibration
(ECE) and execute-precision, and map that competence to an autonomy level:

  AUTO_MERGE    enough evidence AND well-calibrated AND high execute-precision
  ASSISTED      an acted-on segment that isn't (yet) trustworthy enough to auto-merge
  HUMAN_REVIEW  too little evidence, or miscalibrated / low precision

The policy is **advisory**: it never changes what the cascade executes — it labels
*how much to trust the result*, so autonomy expands only where the data earns it.
Pure functions over logged rows; reuses the canonical ECE from ``calibrate``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from nse.models.calibrate import compute_ece


class AutonomyLevel(str, Enum):
    AUTO_MERGE = "auto_merge"
    ASSISTED = "assisted"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class AutonomyPolicy:
    """Thresholds that gate autonomy. Conservative by default."""

    min_n: int = 20              # evidence required before any autonomy is granted
    max_ece: float = 0.10        # calibration bar for auto-merge
    min_precision: float = 0.90  # execute-precision bar for auto-merge


DEFAULT_POLICY = AutonomyPolicy()


@dataclass
class SegmentCompetence:
    segment: str
    n: int
    ece: float
    precision: float | None      # P(passed | predicted pass); None if it never predicts pass
    pass_rate: float

    def level(self, policy: AutonomyPolicy = DEFAULT_POLICY) -> AutonomyLevel:
        if self.n < policy.min_n:
            return AutonomyLevel.HUMAN_REVIEW
        if (
            self.ece <= policy.max_ece
            and self.precision is not None
            and self.precision >= policy.min_precision
        ):
            return AutonomyLevel.AUTO_MERGE
        return AutonomyLevel.ASSISTED


def default_segment_fn(planner_json: str) -> str:
    """Segment by the branch's first edited file — a natural competence unit
    (the system can be trustworthy in one module and not another)."""
    try:
        files = (json.loads(planner_json) or {}).get("edited_files") or []
    except (ValueError, TypeError):
        return "unknown"
    return files[0] if files else "unknown"


def compute_competence(
    rows: list[dict],
    segment_fn: Callable[[str], str] = default_segment_fn,
    threshold: float = 0.5,
) -> dict[str, SegmentCompetence]:
    """Group labeled predictions by segment and score each segment's calibration +
    execute-precision. ``rows`` are dicts with ``p_t``, ``tests_passed``, and
    ``planner_json`` (from :meth:`DBClient.labeled_predictions`)."""
    buckets: dict[str, list[tuple[float, int]]] = {}
    for r in rows:
        seg = segment_fn(r.get("planner_json", ""))
        buckets.setdefault(seg, []).append((float(r["p_t"]), int(r["tests_passed"])))

    out: dict[str, SegmentCompetence] = {}
    for seg, pairs in buckets.items():
        probs = [p for p, _ in pairs]
        labels = [y for _, y in pairs]
        predicted_pass = [y for p, y in pairs if p >= threshold]
        precision = (sum(predicted_pass) / len(predicted_pass)) if predicted_pass else None
        out[seg] = SegmentCompetence(
            segment=seg,
            n=len(pairs),
            ece=compute_ece(probs, labels),
            precision=precision,
            pass_rate=sum(labels) / len(pairs),
        )
    return out


def autonomy_for(
    segment: str,
    competence: dict[str, SegmentCompetence],
    policy: AutonomyPolicy = DEFAULT_POLICY,
) -> AutonomyLevel:
    """Autonomy level for a branch in ``segment``. Unknown/never-seen segments
    default to HUMAN_REVIEW — autonomy is *earned*, not assumed."""
    comp = competence.get(segment)
    return comp.level(policy) if comp is not None else AutonomyLevel.HUMAN_REVIEW
