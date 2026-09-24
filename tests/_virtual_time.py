"""Deterministic virtual time for exercising the real Limiter acquire paths.

This module contains NO rate-limiting logic. ``VirtualTimeService`` only
implements the two time primitives of the private ``_TimeService`` boundary
in ``pyrate_limiter.limiter``:

- ``now()`` reads the shared :class:`~pyrate_limiter.VirtualClock` (seconds),
- ``sleep()`` / ``sleep_async()`` advance that clock instead of suspending the
  real thread/task.

Everything else - bucket acquire, delay planning, deadline enforcement,
exception translation - runs unchanged through the public Limiter API, so the
tests below genuinely cover the production code. ``make_env`` returns a fresh
``(limiter, bucket, clock)`` triple per test: clocks are never shared between
limiters.
"""

import asyncio
from typing import List, Optional, Tuple

from pyrate_limiter import InMemoryBucket, Limiter, Rate, VirtualClock
from pyrate_limiter.limiter import _TimeService


class VirtualTimeService(_TimeService):
    """Virtual-clock-backed time boundary.

    Sync ``sleep`` advances time immediately (no real wait). The async sleep
    yields once to the event loop before advancing, so concurrently scheduled
    tasks observe an interleaving point at every wait instead of a busy loop.
    """

    def __init__(self, clock: VirtualClock):
        self._clock = clock

    def now(self) -> float:
        return self._clock.seconds()

    def sleep(self, seconds: float) -> None:
        self._clock.advance(int(round(seconds * 1000)))

    async def sleep_async(self, seconds: float) -> None:
        # One scheduler turn so waiters interleave like real asyncio.sleep.
        await asyncio.sleep(0)
        self._clock.advance(int(round(seconds * 1000)))


def make_env(
    rates: List[Rate],
    buffer_ms: int = 0,
    start_ms: int = 0,
    algorithm=None,
) -> Tuple[Limiter, InMemoryBucket, VirtualClock]:
    """Fresh limiter + in-memory bucket sharing one private virtual clock."""
    clock = VirtualClock(start_ms)
    bucket = InMemoryBucket(rates, algorithm=algorithm, clock=clock) if algorithm is not None else InMemoryBucket(
        rates, clock=clock
    )
    limiter = Limiter(bucket, buffer_ms=buffer_ms, _time=VirtualTimeService(clock))
    return limiter, bucket, clock
