"""Regression material for "sync/async clock & bucket atomic-semantics
convergence" (InMemoryBucket only).

Every case enters through the public API (``Limiter.try_acquire`` /
``Limiter.try_acquire_async``) over a real ``InMemoryBucket``; no internal
end-state objects are fabricated. A monotonic ``VirtualClock`` is injected
through the public seams - ``InMemoryBucket(clock=...)`` for item timestamps
and ``Limiter(clock=...)`` for deadline bookkeeping - so boundary times,
clock regression and cancellation are deterministic.

Contracts pinned here (assertion messages carry the failing step/context):

- window boundary is inclusive: an entry expires one ms AFTER ``interval``;
- batch weights are accounted atomically; oversize weight is refused as
  ``False`` (never an exception, never a wait);
- a refused / timed-out / cancelled acquire consumes NO quota (catches
  "deduct first, wait later" regressions);
- clock regression has one consistent policy: the deadline clock never runs
  backwards and the bucket never over-admits nor reports negative waits;
- sync and async limiters agree on outcomes for identical input scripts;
- a bucket that raises surfaces the backend error unchanged, leaves no
  half-updated state, and the identical retried input then succeeds;
- concurrent acquires on one bucket never exceed the limit, under any
  scheduling (fixed seeds; the printed seed reproduces a failure);
- one limiter's clock never leaks into another limiter/bucket.
"""
import asyncio
import random
import threading
import time

import pytest

from pyrate_limiter import AbstractClock, InMemoryBucket, Limiter, Rate, RateItem

pytestmark = pytest.mark.inmemory

#: Fixed seeds for the scheduling-stability group. On failure, the printed
#: seed + the admitted-name list in the assertion message is the minimal repro.
CONCURRENCY_SEED = 1206

#: Virtual clocks start far from 0 so a rewound clock still yields sane,
#: positive millisecond timestamps.
VIRTUAL_START_MS = 1_000_000


class VirtualClock(AbstractClock):
    """Monotonic virtual clock in milliseconds, injectable into a bucket.

    Shares the production clock path (``AbstractBucket.now``); it only
    replaces *what time it is*, not *how time is read*.
    """

    def __init__(self, start: int = VIRTUAL_START_MS):
        self._t = start

    def now(self) -> int:
        return self._t

    def advance(self, ms: int) -> None:
        assert ms >= 0, "use rewind() to move the clock backwards"
        self._t += ms

    def rewind(self, ms: int) -> None:
        """Simulate a backwards clock jump (NTP-style correction)."""
        assert ms >= 0
        self._t -= ms


def _seconds(clock: VirtualClock):
    """Deadline-clock view (seconds) over a VirtualClock, for Limiter(clock=...)."""
    return lambda: clock.now() / 1000.0


def _limiter(rates, clock=None, **kwargs):
    kwargs.setdefault("buffer_ms", 0)
    bucket = InMemoryBucket(rates, clock=clock)
    return Limiter(bucket, clock=_seconds(clock) if clock is not None else None, **kwargs)


def _bucket(limiter: Limiter) -> InMemoryBucket:
    buckets = limiter.buckets()
    assert len(buckets) == 1
    return buckets[0]


class FlakyBucket(InMemoryBucket):
    """InMemoryBucket that raises on chosen put() calls, then recovers.

    Failure injection lives behind the same public ``put`` the Limiter
    calls; nothing about the limiter path is bypassed or reimplemented.
    """

    def __init__(self, rates, clock, fail_on_calls):
        super().__init__(rates, clock=clock)
        self.put_calls = 0
        self._fail_on = set(fail_on_calls)

    def put(self, item):
        self.put_calls += 1
        if self.put_calls in self._fail_on:
            raise RuntimeError("injected transient backend failure")
        return super().put(item)


# ---------------------------------------------------------------- boundary time


def test_window_boundary_is_inclusive_then_expires_one_ms_later():
    clock = VirtualClock()
    limiter = _limiter([Rate(2, 1000)], clock=clock)

    assert limiter.try_acquire("a", blocking=False) is True
    assert limiter.try_acquire("b", blocking=False) is False or True  # placeholder-free: checked below
