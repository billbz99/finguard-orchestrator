from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tests.evaluation.scenario_models import (
    AllowedMatcher,
    AssessmentStatusMatcher,
    ExactMatcher,
    Matcher,
    RangeMatcher,
    SubsetMatcher,
)


CASE_INSENSITIVE_FIELDS = {
    "assessment_status",
    "risk_rating",
    "transaction_type",
    "regulations",
    "applicable_regulations",
    "suspected_patterns",
    "suspicious_patterns",
    "jurisdiction",
    "failure_types",
}


@dataclass(frozen=True)
class MatchResult:
    passed: bool
    expected: Any
    actual: Any
    message: str


def _normalize_scalar(field: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    if field in CASE_INSENSITIVE_FIELDS:
        normalized = normalized.casefold()
    return normalized


def _normalized_values(field: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_scalar(field, item) for item in value]
    return _normalize_scalar(field, value)


def _order_insensitive_equal(expected: list[Any], actual: list[Any]) -> bool:
    unmatched = list(actual)
    for expected_item in expected:
        try:
            unmatched.remove(expected_item)
        except ValueError:
            return False
    return not unmatched


def match_value(
    field: str,
    matcher: Matcher,
    actual: Any,
    *,
    order_sensitive: bool = False,
) -> MatchResult:
    """Evaluates one declared matcher with narrow deterministic normalization."""
    normalized_actual = _normalized_values(field, actual)

    if isinstance(matcher, (ExactMatcher, AssessmentStatusMatcher)):
        expected = _normalized_values(field, matcher.value)
        if isinstance(expected, list) and isinstance(normalized_actual, list):
            passed = (
                expected == normalized_actual
                if order_sensitive
                else _order_insensitive_equal(expected, normalized_actual)
            )
        else:
            passed = expected == normalized_actual
        message = (
            f"{field} matched exact expectation"
            if passed
            else f"{field} expected exactly {matcher.value!r}, got {actual!r}"
        )
        return MatchResult(passed, matcher.value, actual, message)

    if isinstance(matcher, SubsetMatcher):
        expected = _normalized_values(field, matcher.values)
        if not isinstance(normalized_actual, list):
            passed = False
        else:
            passed = all(item in normalized_actual for item in expected)
        message = (
            f"{field} contained expected subset"
            if passed
            else f"{field} expected subset {matcher.values!r}, got {actual!r}"
        )
        return MatchResult(passed, matcher.values, actual, message)

    if isinstance(matcher, AllowedMatcher):
        expected = _normalized_values(field, matcher.values)
        passed = normalized_actual in expected
        message = (
            f"{field} matched an allowed value"
            if passed
            else f"{field} expected one of {matcher.values!r}, got {actual!r}"
        )
        return MatchResult(passed, matcher.values, actual, message)

    if isinstance(matcher, RangeMatcher):
        passed = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and matcher.min <= actual <= matcher.max
        )
        message = (
            f"{field} was within the expected range"
            if passed
            else f"{field} expected range [{matcher.min}, {matcher.max}], got {actual!r}"
        )
        return MatchResult(
            passed,
            {"min": matcher.min, "max": matcher.max},
            actual,
            message,
        )

    raise TypeError(f"unsupported matcher type: {type(matcher).__name__}")
