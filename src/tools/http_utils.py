"""Retry-aware HTTP helpers for tool implementations.

Retries on 5xx responses and network errors only.
4xx errors (bad API key, not found) fail immediately — retrying won't help.
"""
import time

import httpx


def http_get(
    url: str,
    *,
    retries: int = 2,
    backoff: tuple[float, ...] = (0.5, 1.0),
    **kwargs,
) -> httpx.Response:
    """GET with automatic retry. Raises on final failure."""
    last_err: Exception = RuntimeError("no attempts made")
    for i in range(retries):
        if i > 0:
            time.sleep(backoff[i - 1])
        try:
            r = httpx.get(url, **kwargs)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise
            last_err = e
        except httpx.RequestError as e:
            last_err = e
    raise last_err


def http_post(
    url: str,
    *,
    retries: int = 2,
    backoff: tuple[float, ...] = (0.5, 1.0),
    **kwargs,
) -> httpx.Response:
    """POST with automatic retry. Raises on final failure."""
    last_err: Exception = RuntimeError("no attempts made")
    for i in range(retries):
        if i > 0:
            time.sleep(backoff[i - 1])
        try:
            r = httpx.post(url, **kwargs)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise
            last_err = e
        except httpx.RequestError as e:
            last_err = e
    raise last_err


def http_put(
    url: str,
    *,
    retries: int = 2,
    backoff: tuple[float, ...] = (0.5, 1.0),
    **kwargs,
) -> httpx.Response:
    """PUT with automatic retry. Raises on final failure."""
    last_err: Exception = RuntimeError("no attempts made")
    for i in range(retries):
        if i > 0:
            time.sleep(backoff[i - 1])
        try:
            r = httpx.put(url, **kwargs)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise
            last_err = e
        except httpx.RequestError as e:
            last_err = e
    raise last_err
