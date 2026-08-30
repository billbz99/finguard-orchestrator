from dataclasses import dataclass
from enum import StrEnum

from src.graph.schemas import EvidenceGap


class DeficiencyType(StrEnum):
    NONE = "NONE"
    TRANSACTION = "TRANSACTION"
    REGULATORY = "REGULATORY"
    MATERIAL_CONFLICT = "MATERIAL_CONFLICT"


TRANSACTION_EVIDENCE_GAPS = frozenset(
    {
        "AMOUNT",
        "TIMING",
        "COUNTERPARTIES",
        "JURISDICTION",
        "TRANSACTION_RELATIONSHIP",
        "TRANSACTION_HISTORY",
        "PURPOSE",
        "INTENT",
    }
)


@dataclass(frozen=True)
class EvidencePolicyDecision:
    is_finalizable: bool
    deficiency_type: DeficiencyType
    required_gaps: tuple[EvidenceGap, ...]


def evaluate_evidence_policy(
    required_gaps: list[EvidenceGap],
) -> EvidencePolicyDecision:
    """Map validated evidence gaps to deterministic workflow policy."""
    gaps = tuple(dict.fromkeys(required_gaps))
    gap_set = set(gaps)

    if "MATERIAL_CONFLICT" in gap_set:
        deficiency_type = DeficiencyType.MATERIAL_CONFLICT
    elif gap_set & TRANSACTION_EVIDENCE_GAPS:
        deficiency_type = DeficiencyType.TRANSACTION
    elif "REGULATORY_CONTEXT" in gap_set:
        deficiency_type = DeficiencyType.REGULATORY
    else:
        deficiency_type = DeficiencyType.NONE

    return EvidencePolicyDecision(
        is_finalizable=deficiency_type == DeficiencyType.NONE,
        deficiency_type=deficiency_type,
        required_gaps=gaps,
    )
