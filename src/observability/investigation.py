"""Request-scoped numeric diagnostics, independent of graph evidence/state."""

import threading
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CandidateDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_index: int = Field(ge=1)
    vector_distance: float | None = None
    rerank_score: float | None = None
    shortlisted: bool
    admitted: bool | None = None


class RetrievalDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pass_index: int = Field(ge=1)
    candidate_count: int = Field(ge=0)
    reranked_count: int = Field(ge=0)
    shortlist_count: int = Field(ge=0)
    admitted_count: int | None = Field(default=None, ge=0)
    candidates: list[CandidateDiagnostics] = Field(default_factory=list)


class InvestigationDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["reported", "partial", "unavailable", "not_applicable"]
    critic_pass_count: int | None = Field(default=None, ge=0)
    refinement_count: int | None = Field(default=None, ge=0)
    retrieval_passes: list[RetrievalDiagnostics] = Field(default_factory=list)


class InvestigationCollector:
    """One collector per request; never attached to a shared retriever."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._passes: dict[int, RetrievalDiagnostics] = {}

    def record_retrieval(self, pass_index: int, **values: Any) -> None:
        observation = RetrievalDiagnostics(pass_index=pass_index, **values)
        with self._lock:
            self._passes[pass_index] = observation

    def record_admission(self, pass_index: int, admitted_count: int) -> None:
        with self._lock:
            observation = self._passes.get(pass_index)
            if observation is not None:
                observation.admitted_count = admitted_count
                for candidate in observation.candidates:
                    candidate.admitted = (
                        candidate.shortlisted
                        and candidate.rerank_score is not None
                        and candidate.rerank_score >= 0.15
                    )

    def snapshot(self, state: dict[str, Any] | None) -> InvestigationDiagnostics:
        if state is None:
            return InvestigationDiagnostics(
                status="not_applicable", critic_pass_count=0, refinement_count=0
            )
        count = state.get("loop_count")
        completed = state.get("is_audit_complete") is True
        valid_count = isinstance(count, int) and not isinstance(count, bool) and count >= 1
        critic_count = count if valid_count else None
        refinements = count - 1 if valid_count and completed else None
        with self._lock:
            passes = [value.model_copy(deep=True) for _, value in sorted(self._passes.items())]
        complete = (
            refinements is not None
            and len(passes) == critic_count
            and [p.pass_index for p in passes] == list(range(1, critic_count + 1))
            and all(p.admitted_count is not None for p in passes)
        )
        return InvestigationDiagnostics(
            status="reported" if complete else "partial" if passes or valid_count else "unavailable",
            critic_pass_count=critic_count,
            refinement_count=refinements,
            retrieval_passes=passes,
        )


def observe(observer: Any, method: str, *args: Any, **kwargs: Any) -> None:
    """Telemetry failures must not affect evidence or investigation execution."""
    if observer is not None:
        try:
            getattr(observer, method)(*args, **kwargs)
        except Exception:
            pass
