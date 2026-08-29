from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SUPPORTED_SCHEMA_VERSION = "1.0"
AssessmentStatus = Literal["COMPLETE", "INSUFFICIENT_EVIDENCE"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExactMatcher(StrictModel):
    match: Literal["exact"]
    value: Any


class SubsetMatcher(StrictModel):
    match: Literal["subset"]
    values: list[Any] = Field(min_length=1)


class AllowedMatcher(StrictModel):
    match: Literal["allowed"]
    values: list[Any] = Field(min_length=1)


class RangeMatcher(StrictModel):
    match: Literal["range"]
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.max < self.min:
            raise ValueError("range matcher max must be greater than or equal to min")
        return self


Matcher = Annotated[
    ExactMatcher | SubsetMatcher | AllowedMatcher | RangeMatcher,
    Field(discriminator="match"),
]


class AssessmentStatusMatcher(StrictModel):
    match: Literal["exact"]
    value: AssessmentStatus


class ScenarioReference(StrictModel):
    scenario_id: str = Field(min_length=1)
    file: str = Field(min_length=1)


class DatasetManifest(StrictModel):
    dataset_id: Literal["finguard-synthetic-aml-golden"]
    dataset_version: Literal["1.1.0"]
    schema_version: str
    expectation_profile: Literal["aml-golden-v1"]
    scenarios: list[ScenarioReference] = Field(min_length=1)


class ScenarioInput(StrictModel):
    query: str = Field(min_length=1)
    amount: float | None = None
    is_cross_border: bool = False
    max_loops: int = Field(default=2, ge=1)


class SyntheticFacts(StrictModel):
    transaction_ids: list[str] = Field(default_factory=list)
    amounts: list[float] = Field(default_factory=list)
    transaction_type: str | None = None
    jurisdictions: list[str] = Field(default_factory=list)
    counterparties: list[str] = Field(default_factory=list)
    purpose: str | None = None
    time_window: str | None = None
    explicitly_absent: list[str] = Field(default_factory=list)


class RetrievalDocument(StrictModel):
    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, str]
    rerank_score: float


class RetrievalPass(StrictModel):
    documents: list[RetrievalDocument] = Field(default_factory=list)
    expected_query_contains: list[str] = Field(default_factory=list)


class RetrievalPlan(StrictModel):
    passes: list[RetrievalPass] = Field(min_length=1)


class ExtractionExpectation(StrictModel):
    transaction_ids: Matcher
    amount: Matcher
    transaction_type: Matcher
    regulations: Matcher
    suspected_patterns: Matcher
    jurisdiction: Matcher
    doc_type: Matcher


class AMLAssessmentExpectation(StrictModel):
    risk_rating: Matcher
    suspicious_patterns: Matcher
    flagged_transactions: Matcher
    applicable_regulations: Matcher
    insufficient_evidence: Matcher


class CriticExpectation(StrictModel):
    actions: Matcher
    failure_types: Matcher


class ReportExpectation(StrictModel):
    assessment_status: AssessmentStatusMatcher
    risk_rating: Matcher
    flagged_wires: Matcher
    applicable_regulations: Matcher
    source_document_hashes: Matcher


class ExecutionExpectation(StrictModel):
    retrieval_count: Matcher
    critic_count: Matcher
    final_loop_count: Matcher
    terminates: Matcher


class ExpectedOutcomes(StrictModel):
    extraction: ExtractionExpectation
    aml_assessment: AMLAssessmentExpectation
    critic: CriticExpectation
    report: ReportExpectation
    execution: ExecutionExpectation


class ProhibitedOutcomes(StrictModel):
    transaction_ids: list[str] = Field(default_factory=list)
    regulations: list[str] = Field(default_factory=list)
    jurisdictions: list[str] = Field(default_factory=list)
    suspicious_patterns: list[str] = Field(default_factory=list)
    unsupported_fact_terms: list[str] = Field(default_factory=list)


class GoldenScenario(StrictModel):
    schema_version: str
    scenario_id: str = Field(min_length=1)
    scenario_version: int = Field(gt=0)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tags: list[str] = Field(min_length=1)
    synthetic_data: Literal[True]
    input: ScenarioInput
    synthetic_facts: SyntheticFacts
    retrieval: RetrievalPlan
    expected: ExpectedOutcomes
    prohibited: ProhibitedOutcomes

    @model_validator(mode="after")
    def validate_synthetic_transaction_ids(self):
        invalid_ids = [
            transaction_id
            for transaction_id in self.synthetic_facts.transaction_ids
            if not transaction_id.startswith("TXN-SYN-")
        ]
        if invalid_ids:
            raise ValueError(
                "synthetic transaction IDs must use the TXN-SYN- prefix: "
                f"{invalid_ids}"
            )
        return self
