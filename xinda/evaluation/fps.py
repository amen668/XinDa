"""FPS — Fidelity Preservation Score aggregation.

FPS_unit = (verified claims) / (total claims)
FPS_paper = mean(FPS_unit) across all units
fps_<type> = per-claim-type breakdown

Computed from claim_verifications rows.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from xinda.db.models import ClaimType, DriftType


def fps_from_verifications(
    verifications: Iterable[tuple[ClaimType, DriftType]],
) -> dict[str, float | dict[str, float]]:
    """Given a stream of (claim_type, drift) tuples, return aggregate FPS dict.

    `drift == verified` counts as 1; everything else (drift detected or
    missing) counts as 0.
    """
    total = 0
    verified = 0
    per_type_total: dict[str, int] = defaultdict(int)
    per_type_verified: dict[str, int] = defaultdict(int)
    drift_counts: dict[str, int] = defaultdict(int)

    for ctype, drift in verifications:
        total += 1
        per_type_total[ctype.value] += 1
        drift_counts[drift.value] += 1
        if drift == DriftType.verified:
            verified += 1
            per_type_verified[ctype.value] += 1

    return {
        "total": total,
        "verified": verified,
        "fps": (verified / total) if total else 1.0,
        "per_type": {
            t: (per_type_verified.get(t, 0) / cnt) if cnt else 1.0
            for t, cnt in per_type_total.items()
        },
        "drift_counts": dict(drift_counts),
    }


def fps_unit(total: int, verified: int) -> float:
    """Compute FPS for one unit given its claim count and verified count."""
    return (verified / total) if total > 0 else 1.0
