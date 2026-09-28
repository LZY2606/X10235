"""Reusable time boundary shared by the sync and async acquire paths.

``Limiter`` used to read ``time.monotonic()`` and call ``time.sleep`` /
``asyncio.sleep`` directly from half a dozen branches, each with its own copy
of the deadline and delay math.  That duplication made the two paths drift
(the async branch asserted ``delay >= 0`` while the sync branch clamped it)
and made the wait behavior impossible to exercise deterministically.

A :class:`Timing` is the single seam used by every acquire path for:

- reading the current time (:meth:`now` seconds and :meth:`now_ms` ms),
- sleeping (:meth:`sleep` for the sync path, :meth:`asleep` for the async
  path - they read the same clock and share the same duration rules),
- deadline math (:meth:`deadline`, :meth:`remaining_ms`).

The production implementation, :class:`SystemTiming`, delegates to the
``time`` module exactly like the old inline calls, so default behavior is
unchanged.  Tests inject an alternative implementation (a virtual monotonic
clock) through ``Limiter(..., _timing=...)``; the limiter logic exercised in
that case is the *same* code the production path runs - no test-only
algorithm.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional


class Timing:
    """Clock + sleep boundary used by ``Limiter``.

    Subclasses (or duck-typed equivalents used by tests) override the clock
    reads and the two sleep primitives.  Every method here is intentionally
    tiny and side-effect free apart from the sleep: this object is the only
    place where acquire code touches the concept of "current time".
    """

    def now(self) -> float:
        """Current monotonic time in seconds."""
        raise NotImplementedError

    def now_ms(self) -> int:
        """Current monotonic time in whole milliseconds.

        Buckets timestamp ``RateItem``\\ s in ms; this must be the same clock
        (and scale) as :meth:`now`, just rounded, so that bucket timestamps
        and limiter deadlines are directly comparable.
        """
        return int(self.now() * 1000)

    def sleep(self, seconds: float) -> None:
        """Blocking sleep used by the sync acquire path."""
        raise NotImplementedError

    async def asleep(self, seconds: float) -> None:
        """Awaitable sleep used by the async acquire path."""
        raise NotImplementedError

    def deadline(self, timeout: float) -> Optional[float]:
        """Absolute deadline for a relative timeout.

        ``timeout == -1`` means "wait forever" and yields ``None``; any other
        timeout is resolved against this timing's clock.
        """
        if timeout == -1:
            return None
        return self.now() + timeout

    def remaining_ms(self, deadline: Optional[float]) -> Optional[float]:
        """Whole ms left until ``deadline``; ``None`` if there is no deadline.

        A non-positive result means the deadline has been reached.
        """
        if deadline is None:
            return None
        return (deadline - self.now()) * 1000


class SystemTiming(Timing):
    """Production timing: ``time.monotonic`` + ``time.sleep``/``asyncio.sleep``.

    Identical to the reads and sleeps the limiter performed inline before the
    sync/async paths were consolidated; it is module-level-shared because it
    holds no per-limiter state.
    """

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    async def asleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


# Single shared instance: stateless and safe across limiters and threads.
SYSTEM_TIMING = SystemTiming()
