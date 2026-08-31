from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import os
from typing import Any, Literal
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, ConfigDict, Field


CallUsageStatus = Literal["reported", "unavailable", "failed"]
AuditUsageStatus = Literal[
    "reported",
    "partial",
    "unavailable",
    "not_applicable",
]
CostStatus = Literal[
    "estimated",
    "pricing_not_configured",
    "model_mismatch",
    "usage_unavailable",
    "not_applicable",
]


@dataclass(frozen=True)
class LLMPricing:
    """Validated per-million-token prices for one configured model."""

    model: str
    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None
    revision: str | None = None


def _price_from_env(name: str, *, required: bool) -> Decimal | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        if required:
            raise ValueError(f"{name} is required")
        return None
    try:
        value = Decimal(raw.strip())
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a decimal") from exc
    if not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be a non-negative finite decimal")
    return value


def load_xai_pricing() -> LLMPricing | None:
    """Load optional pricing without allowing bad configuration to affect audits."""

    names = (
        "XAI_PRICE_MODEL",
        "XAI_INPUT_PRICE_PER_MILLION",
        "XAI_OUTPUT_PRICE_PER_MILLION",
        "XAI_CACHED_INPUT_PRICE_PER_MILLION",
        "XAI_PRICING_REVISION",
    )
    if not any(os.getenv(name, "").strip() for name in names):
        return None
    try:
        model = os.getenv("XAI_PRICE_MODEL", "").strip()
        if not model:
            raise ValueError("XAI_PRICE_MODEL is required")
        return LLMPricing(
            model=model,
            input_per_million=_price_from_env(
                "XAI_INPUT_PRICE_PER_MILLION", required=True
            ),
            output_per_million=_price_from_env(
                "XAI_OUTPUT_PRICE_PER_MILLION", required=True
            ),
            cached_input_per_million=_price_from_env(
                "XAI_CACHED_INPUT_PRICE_PER_MILLION", required=False
            ),
            revision=os.getenv("XAI_PRICING_REVISION") or None,
        )
    except (ValueError, TypeError):
        return None


class LLMCallUsage(BaseModel):
    """Safe usage metadata for one logical chat-model invocation."""

    model_config = ConfigDict(extra="forbid")

    node: str | None = None
    call_index: int = Field(ge=1)
    usage_status: CallUsageStatus
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    provider_request_id: str | None = None


class AuditLLMUsage(BaseModel):
    """Aggregate usage for the logical LLM calls made by one audit request."""

    model_config = ConfigDict(extra="forbid")

    usage_status: AuditUsageStatus
    provider: str
    model: str
    logical_call_count: int = Field(ge=0)
    completed_call_count: int = Field(ge=0)
    failed_call_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    cost_status: CostStatus
    pricing_revision: str | None = None
    calls: list[LLMCallUsage] = Field(default_factory=list)


class AuditObservability(BaseModel):
    """Additive API observability envelope kept outside the compliance report."""

    model_config = ConfigDict(extra="forbid")

    llm_usage: AuditLLMUsage


@dataclass(frozen=True)
class _ActiveCall:
    call_index: int
    started_at: float
    node: str | None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if converted >= 0 else None


def _mapping_value(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(mapping, dict):
            value = mapping.get(key)
        else:
            value = getattr(mapping, key, None)
        if value is not None:
            return value
    return None


def _usage_values(usage: Any) -> dict[str, int | None]:
    input_details = _mapping_value(
        usage,
        "input_token_details",
        "prompt_tokens_details",
        "input_tokens_details",
    ) or {}
    output_details = _mapping_value(
        usage,
        "output_token_details",
        "completion_tokens_details",
        "output_tokens_details",
    ) or {}
    return {
        "input_tokens": _non_negative_int(
            _mapping_value(usage, "input_tokens", "prompt_tokens")
        ),
        "output_tokens": _non_negative_int(
            _mapping_value(usage, "output_tokens", "completion_tokens")
        ),
        "total_tokens": _non_negative_int(_mapping_value(usage, "total_tokens")),
        "cached_input_tokens": _non_negative_int(
            _mapping_value(input_details, "cache_read", "cached_tokens")
        ),
        "reasoning_tokens": _non_negative_int(
            _mapping_value(output_details, "reasoning", "reasoning_tokens")
        ),
    }


def _message_from_result(response: LLMResult) -> Any:
    for batch in response.generations:
        for generation in batch:
            message = getattr(generation, "message", None)
            if message is not None:
                return message
    return None


def _extract_usage(response: LLMResult) -> tuple[dict[str, int | None], str | None]:
    message = _message_from_result(response)
    usage = getattr(message, "usage_metadata", None)
    response_metadata = getattr(message, "response_metadata", None) or {}
    llm_output = response.llm_output or {}

    if not usage:
        usage = llm_output.get("token_usage") or response_metadata.get("token_usage")

    request_id = (
        getattr(message, "id", None)
        or response_metadata.get("request_id")
        or response_metadata.get("id")
        or llm_output.get("request_id")
        or llm_output.get("id")
    )
    values = _usage_values(usage) if usage else {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
    }
    return values, str(request_id) if request_id else None


def _node_from_metadata(
    metadata: dict[str, Any] | None,
    tags: list[str] | None,
) -> str | None:
    metadata = metadata or {}
    for key in ("langgraph_node", "node"):
        value = metadata.get(key)
        if value in {"extraction", "aml_audit", "auditor_critic"}:
            return str(value)

    for tag in tags or []:
        for prefix in ("langgraph_node:", "node:"):
            if tag.startswith(prefix):
                value = tag.removeprefix(prefix)
                if value in {"extraction", "aml_audit", "auditor_critic"}:
                    return value
    return None


def no_llm_usage(*, provider: str, model: str) -> AuditLLMUsage:
    """Return explicit zero new usage for a path that cannot call an LLM."""

    return AuditLLMUsage(
        usage_status="not_applicable",
        provider=provider,
        model=model,
        logical_call_count=0,
        completed_call_count=0,
        failed_call_count=0,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cached_input_tokens=0,
        reasoning_tokens=0,
        estimated_cost_usd=Decimal("0"),
        cost_status="not_applicable",
        calls=[],
    )


def _estimate_cost(
    *,
    usage_status: AuditUsageStatus,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None,
    pricing: LLMPricing | None,
) -> tuple[Decimal | None, CostStatus, str | None]:
    if usage_status != "reported" or input_tokens is None or output_tokens is None:
        return None, "usage_unavailable", pricing.revision if pricing else None
    if pricing is None:
        return None, "pricing_not_configured", None
    if pricing.model != model:
        return None, "model_mismatch", pricing.revision

    cached = cached_input_tokens or 0
    if cached > input_tokens:
        return None, "usage_unavailable", pricing.revision
    if cached and pricing.cached_input_per_million is None:
        return None, "pricing_not_configured", pricing.revision

    uncached = input_tokens - cached
    cached_rate = pricing.cached_input_per_million or Decimal("0")
    cost = (
        Decimal(uncached) * pricing.input_per_million
        + Decimal(cached) * cached_rate
        + Decimal(output_tokens) * pricing.output_per_million
    ) / Decimal("1000000")
    return cost, "estimated", pricing.revision


class LLMUsageCollector(BaseCallbackHandler):
    """Collect usage for one audit without retaining prompts or model content."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        pricing: LLMPricing | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.pricing = pricing
        self._lock = threading.RLock()
        self._active: dict[UUID, _ActiveCall] = {}
        self._calls: list[LLMCallUsage] = []
        self._next_call_index = 1

    def _start(
        self,
        *,
        run_id: UUID,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        with self._lock:
            if run_id in self._active:
                return
            active = _ActiveCall(
                call_index=self._next_call_index,
                started_at=time.monotonic(),
                node=_node_from_metadata(metadata, tags),
            )
            self._next_call_index += 1
            self._active[run_id] = active

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized, messages, parent_run_id, kwargs
        try:
            self._start(run_id=run_id, tags=tags, metadata=metadata)
        except Exception:
            return

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized, prompts, parent_run_id, kwargs
        try:
            self._start(run_id=run_id, tags=tags, metadata=metadata)
        except Exception:
            return

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, tags, kwargs
        try:
            with self._lock:
                active = self._active.pop(run_id, None)
            if active is None:
                return
            usage, request_id = _extract_usage(response)
            reported = all(
                usage[key] is not None
                for key in ("input_tokens", "output_tokens", "total_tokens")
            )
            call = LLMCallUsage(
                node=active.node,
                call_index=active.call_index,
                usage_status="reported" if reported else "unavailable",
                latency_ms=round((time.monotonic() - active.started_at) * 1000, 3),
                provider_request_id=request_id,
                **usage,
            )
            with self._lock:
                self._calls.append(call)
        except Exception:
            return

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del error, parent_run_id, tags, kwargs
        try:
            with self._lock:
                active = self._active.pop(run_id, None)
            if active is None:
                return
            call = LLMCallUsage(
                node=active.node,
                call_index=active.call_index,
                usage_status="failed",
                latency_ms=round((time.monotonic() - active.started_at) * 1000, 3),
            )
            with self._lock:
                self._calls.append(call)
        except Exception:
            return

    def snapshot(self) -> AuditLLMUsage:
        """Return an immutable validated summary of the observations so far."""

        with self._lock:
            calls = [call.model_copy(deep=True) for call in self._calls]
            active_count = len(self._active)
            logical_call_count = self._next_call_index - 1

        calls.sort(key=lambda call: call.call_index)
        if logical_call_count == 0:
            return no_llm_usage(provider=self.provider, model=self.model)

        completed = [call for call in calls if call.usage_status != "failed"]
        failed = [call for call in calls if call.usage_status == "failed"]
        reported = [call for call in completed if call.usage_status == "reported"]
        exact = (
            active_count == 0
            and not failed
            and len(reported) == logical_call_count
        )

        if exact:
            usage_status: AuditUsageStatus = "reported"
            input_tokens = sum(call.input_tokens or 0 for call in reported)
            output_tokens = sum(call.output_tokens or 0 for call in reported)
            total_tokens = sum(call.total_tokens or 0 for call in reported)
            cached_input_tokens = sum(
                call.cached_input_tokens or 0 for call in reported
            )
            reasoning_tokens = sum(call.reasoning_tokens or 0 for call in reported)
        else:
            usage_status = "partial" if reported else "unavailable"
            input_tokens = None
            output_tokens = None
            total_tokens = None
            cached_input_tokens = None
            reasoning_tokens = None

        estimated_cost, cost_status, pricing_revision = _estimate_cost(
            usage_status=usage_status,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            pricing=self.pricing,
        )

        return AuditLLMUsage(
            usage_status=usage_status,
            provider=self.provider,
            model=self.model,
            logical_call_count=logical_call_count,
            completed_call_count=len(completed),
            failed_call_count=len(failed),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            estimated_cost_usd=estimated_cost,
            cost_status=cost_status,
            pricing_revision=pricing_revision,
            calls=calls,
        )
