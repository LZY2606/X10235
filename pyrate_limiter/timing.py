"""Reusable timing boundary shared by the sync and async acquire paths.

The Limiter used to call ``time.monotonic()``, ``time.sleep()`` and
``asyncio.sleep()`` directly from several places, while bucket timestamps are
read through a separate ``AbstractClock``. That made two things hard:

1. Proving that the sync and async paths wait for exactly the same duration -
   they each had their own deadline/remaining math.
2. Driving a blocking acquire deterministically in tests - real wall-clock
   sleeps cannot be advanced with a barrier or virtual time.

Everything scheduling-related now goes through a small ``Timer`` boundary.
Production wires it to ``time.monotonic`` / ``time.sleep`` / ``asyncio.sleep``
(``SystemTimer``); tests inject a ``VirtualTimer`` driven off the same
``VirtualClock`` they hand to the buckets, so "bucket time" and "wait time"
advance together and can never disagree.

This is an internal seam: it is not re-exported from the package root. The
public entry points (``Limiter(...)`` / ``try_acquire`` / ``try_acquire_async``)
stay the only way into the acquire path; the timer only chooses how that path
reads time and sleeps - it carries no algorithm of its own.
"""

from __future__ import annotations

import asyncio
from time import monotonic, sleep
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Timer(Protocol):
    """How the acquire path reads scheduling time and waits.

    ``now()`` returns monotonic *seconds* (the scale deadlines and sleeps are
    computed in). Bucket item timestamps keep coming from the bucket clock in
    milliseconds; the two meet only through the delay the bucket reports.
    """

    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...

    async def asleep(self, seconds: float) -> None: ...


class SystemTimer:
    """Default ``Timer``: the real monotonic clock and real sleeps.

    Stateless, so one instance may back any number of limiters; injecting a
    virtual timer into one limiter therefore cannot change what another
    limiter observes.
    """

    __slots__ = ()

    def now(self) -> float:
        return monotonic()

    def sleep(self, seconds: float) -> None:
        sleep(seconds)

    async def asleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class VirtualClock:
    """Manually advanced, strictly non-decreasing clock for deterministic tests.

    One instance is the single source of "now" for both a bucket clock
    (``monotonic_clock()``, milliseconds) and a limiter timer (``timer()``,
    seconds). ``sleep``/``asleep`` do not block: they advance this clock. That
    lets a test reproduce multi-second waits and concurrent contention with an
    ``asyncio`` barrier instead of wall-clock sleeps.

    Backwards policy: ``set()`` never moves time backwards - a smaller value is
    clamped to the current time. Monotonicity is the contract the window math
    relies on, so a backwards read is treated as "no movement" rather than
    silently corrupting item ordering.
    """

    __slots__ = ("_now_ms",)

    def __init__(self, start_ms: int = 0):
        self._now_ms = int(start_ms)

    def now_ms(self) -> int:
        return self._now_ms

    def now_s(self) -> float:
        return self._now_ms / 1000

    def advance_ms(self, delta_ms: int) -> int:
        delta_ms = int(delta_ms)
        if delta_ms < 0:
            raise ValueError("advance_ms() must be >= 0; use set() to move time")
        self._now_ms += delta_ms
        return self._now_ms

    def advance(self, seconds: float) -> int:
        return self.advance_ms(int(seconds * 1000))

    def set(self, value_ms: int) -> int:
        """Move time to ``value_ms``; clamped so it never goes backwards."""
        value_ms = int(value_ms)
        if value_ms > self._now_ms:
            self._now_ms = value_ms
        return self._now_ms

    def monotonic_clock(self) -> "VirtualMonotonicClock":
        return VirtualMonotonicClock(self)

    def timer(self) -> "VirtualTimer":
        return VirtualTimer(self)


class VirtualMonotonicClock:
    """``AbstractClock``-shaped view (milliseconds) over a ``VirtualClock``."""

    __slots__ = ("_vc",)

    def __init__(self, virtual: VirtualClock):
        self._vc = virtual

    def now(self) -> int:
        return self._vc.now_ms()


class VirtualTimer:
    """``Timer`` view (seconds) over the same ``VirtualClock`` buckets use."""

    __slots__ = ("_vc",)

    def __init__(self, virtual: VirtualClock):
        self._vc = virtual

    def now(self) -> float:
        return self._vc.now_s()

    def sleep(self, seconds: float) -> None:
        self._vc.advance(seconds)

    async def asleep(self, seconds: float) -> None:
        # Advance virtual time, then yield once so concurrent waiters and
        # barriers get scheduled exactly as they would across a real sleep.
        self._vc.advance(seconds)
        await asyncio.sleep(0)
