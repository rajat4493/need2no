"""Resolve a pack's declared rules against detected findings.

A pack declares must_hide and must_preserve field IDs. A finding whose
field_id is in both sets is a policy conflict and is never auto-resolved —
it forces NEEDS_REVIEW (spec 5.4 step 4).
"""

from __future__ import annotations

from dataclasses import dataclass

from n2n.models import Finding


@dataclass(frozen=True)
class Pack:
    pack_id: str
    version: str
    description: str
    must_hide: frozenset[str]
    must_preserve: frozenset[str]
    # auto-tier fields still require the "structural" tier on the finding
    # itself; must_hide fields that only ever arrive as review-tier (e.g.
    # name_header) can never reach PASS_AUTO under this pack.
    redaction_technique: str = "content_stream_rewrite"

    def __post_init__(self) -> None:
        overlap = self.must_hide & self.must_preserve
        if overlap:
            raise ValueError(
                f"Pack {self.pack_id} declares fields in both must_hide and "
                f"must_preserve: {sorted(overlap)}"
            )


@dataclass
class PolicyResolution:
    to_remove: list[Finding]
    to_preserve: list[Finding]
    needs_review: list[Finding]
    conflicts: list[str]


def resolve(findings: list[Finding], pack: Pack) -> PolicyResolution:
    to_remove: list[Finding] = []
    to_preserve: list[Finding] = []
    needs_review: list[Finding] = []
    conflicts: list[str] = []

    for finding in findings:
        must_hide = finding.field_id in pack.must_hide
        must_preserve = finding.field_id in pack.must_preserve

        if must_hide and must_preserve:
            # Can't happen given Pack.__post_init__, but never silently
            # resolve a conflict even if a future pack loosens that check.
            conflicts.append(finding.field_id)
            needs_review.append(finding)
            continue

        if must_hide:
            if finding.tier == "structural":
                to_remove.append(Finding(**{**finding.__dict__, "action": "removed"}))
            else:
                # A must-hide field detected only at review tier can never
                # be auto-removed — free-text candidates always need a human.
                needs_review.append(Finding(**{**finding.__dict__, "action": "flagged"}))
            continue

        if must_preserve:
            to_preserve.append(Finding(**{**finding.__dict__, "action": "preserved"}))
            continue

        # Field not declared by the pack at all: unknown structural content
        # that touches a sensitive field family is safer to route to review
        # than to silently ignore or silently remove.
        needs_review.append(Finding(**{**finding.__dict__, "action": "flagged"}))

    return PolicyResolution(
        to_remove=to_remove,
        to_preserve=to_preserve,
        needs_review=needs_review,
        conflicts=conflicts,
    )
