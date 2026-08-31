from __future__ import annotations

import threading
import time
from dataclasses import dataclass
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
        calls=[],
    )


class LLMUsageCollector(BaseCallbackHandler):
    """Collect usage for one audit without retaining prompts or model content."""

    def __init__(self, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
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
            calls=calls,
        )
