"""Fetch from the Overpass API without trusting a single slow server.

Tries several mirrors, retries with backoff, honours Retry-After, and rejects
HTTP 200 answers that carry a runtime-error ``remark`` or no element list.
Network access goes through ``opener`` and ``sleep`` so tests can replace both.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Sequence

DEFAULT_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "delineo-v2"

TIMEOUT = "timeout"
RATE_LIMITED = "rate_limited"
UNAVAILABLE = "unavailable"
BAD_DATA = "bad_data"
BAD_QUERY = "bad_query"

_RETRYABLE_STATUS = {429: RATE_LIMITED, 500: UNAVAILABLE, 502: UNAVAILABLE, 503: UNAVAILABLE, 504: TIMEOUT}
_MAX_RETRY_AFTER_SECONDS = 60.0
_MAX_BACKOFF_SECONDS = 30.0


class OverpassError(RuntimeError):
    """Every server and retry failed, or the query itself is wrong."""

    def __init__(self, reason: str, message: str, attempts: Sequence[str] = ()):
        super().__init__(message)
        self.reason = reason
        self.attempts = list(attempts)


class _Failure(Exception):
    def __init__(self, reason: str, detail: str, retry: bool = True, wait: float | None = None):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.retry = retry
        self.wait = wait


def fetch_overpass(
    query: str,
    *,
    endpoints: Sequence[str] = DEFAULT_ENDPOINTS,
    attempts_per_endpoint: int = 3,
    request_timeout: float = 200.0,
    backoff_seconds: float = 5.0,
    deadline_seconds: float = 900.0,
    min_elements: int = 0,
    opener: Callable = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """POST ``query`` and return a payload that passed ``validate_payload``."""
    if not endpoints:
        raise ValueError("fetch_overpass needs at least one endpoint")
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    started = clock()
    attempts: list[str] = []
    last_reason = UNAVAILABLE
    tries_made = 0

    for endpoint in endpoints:
        for attempt in range(attempts_per_endpoint):
            if tries_made and clock() - started >= deadline_seconds:
                attempts.append(f"gave up after {deadline_seconds:.0f}s")
                raise OverpassError(last_reason, _summary(last_reason, attempts), attempts)
            tries_made += 1
            try:
                return _one_try(endpoint, body, request_timeout, min_elements, opener)
            except _Failure as failure:
                last_reason = failure.reason
                attempts.append(f"{endpoint}: {failure.reason}: {failure.detail}")
                if not failure.retry:
                    raise OverpassError(failure.reason, _summary(failure.reason, attempts), attempts)
                if attempt < attempts_per_endpoint - 1:
                    sleep(_delay(failure.wait, backoff_seconds, attempt))

    raise OverpassError(last_reason, _summary(last_reason, attempts), attempts)


def validate_payload(payload: object, min_elements: int = 0) -> dict:
    """Return ``payload`` if it looks like real Overpass data, else raise OverpassError."""
    try:
        _check_payload(payload, min_elements)
    except _Failure as failure:
        raise OverpassError(failure.reason, failure.detail, [failure.detail])
    return payload  # type: ignore[return-value]


def _one_try(endpoint: str, body: bytes, timeout: float, min_elements: int, opener: Callable) -> dict:
    request = urllib.request.Request(endpoint, data=body, headers={"User-Agent": USER_AGENT})
    try:
        with opener(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise _http_failure(exc) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise _Failure(TIMEOUT, f"no answer within {timeout:.0f}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise _Failure(TIMEOUT, f"no answer within {timeout:.0f}s") from exc
        raise _Failure(UNAVAILABLE, f"connection failed: {exc.reason}") from exc
    except OSError as exc:
        raise _Failure(UNAVAILABLE, f"connection failed: {exc}") from exc

    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise _Failure(BAD_DATA, f"response was not JSON: {bytes(raw[:80])!r}") from exc
    _check_payload(payload, min_elements)
    return payload


def _http_failure(exc: urllib.error.HTTPError) -> _Failure:
    code = exc.code
    if code == 400:
        return _Failure(BAD_QUERY, "HTTP 400, the query was rejected", retry=False)
    reason = _RETRYABLE_STATUS.get(code)
    if reason is not None:
        return _Failure(reason, f"HTTP {code}", wait=_retry_after(exc))
    return _Failure(UNAVAILABLE, f"HTTP {code}")


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    headers = getattr(exc, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(seconds, _MAX_RETRY_AFTER_SECONDS))


def _check_payload(payload: object, min_elements: int) -> None:
    if not isinstance(payload, dict):
        raise _Failure(BAD_DATA, "response JSON was not an object")
    remark = str(payload.get("remark") or "")
    if "runtime error" in remark.lower():
        reason = TIMEOUT if "timed out" in remark.lower() else UNAVAILABLE
        raise _Failure(reason, f"Overpass reported: {remark[:120]}")
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise _Failure(BAD_DATA, "response had no element list")
    if len(elements) < min_elements:
        raise _Failure(BAD_DATA, f"{len(elements)} elements returned, expected at least {min_elements}")


def _delay(wait: float | None, backoff_seconds: float, attempt: int) -> float:
    if wait is not None:
        return wait
    return min(backoff_seconds * (2 ** attempt), _MAX_BACKOFF_SECONDS)


def _summary(reason: str, attempts: Sequence[str]) -> str:
    lines = "; ".join(attempts[-4:])
    return f"Overpass request failed ({reason}) after {len(attempts)} attempt(s): {lines}"
