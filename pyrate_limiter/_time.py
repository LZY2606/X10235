"""Shared time boundary for the sync and async acquire paths.

The Limiter used to read the clock (`time.monotonic`), sleep through waits
(`time.sleep` vs `asyncio.sleep`) and translate timeouts (`asyncio.TimeoutError`
-> built-in `TimeoutError`) separately in each branch. Those primitives are the
one boundary where the sync and async paths necessarily differ; everything else
(deadline arithmetic, delay conversion, exception conversion) is identical and
lives in `pyrate_limiter.limiter`.

Production code uses the module-level ``SYSTEM_TIME`` singleton, which is a thin
wrapper over the real monotonic clock and carries no mutable state - so sharing
it between limiters cannot leak time between them.

``VirtualTimeSource`` is a deterministic, strictly-monotonic virtual clock for
tests. It is deliberately **not** exported from the package and is documented as
test-only; production callers always get ``SYSTEM_TIME`` unless they explicitly
hand one to ``Limiter(..., _time_source=...)`` (itself a private, underscored
parameter). A virtual clock drives the *same* production code paths - only the
four primitives above are replaced; no test-only algorithm is duplicated.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, List, Optional, Tuple

#: A monotonic clock call result smaller than the previous one by more than
#: this many ms is treated as a real clock regression rather than rounding.
_CLOCK_REGRESSION_TOLERANCE_MS = 0


class TimeSource:
    """The four clock primitives shared by the sync and async acquire paths."""

    def now_ms(self) -> int:
        """Current monotonic time in milliseconds (bucket items use this)."""
        raise NotImplementedError

    def monotonic(self) -> float:
        """Current monotonic time in seconds (deadlines use this)."""
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        """Blocking sleep."""
        raise NotImplementedError

    async def async_sleep(self, seconds: float) -> None:
        """Event-loop-friendly sleep."""
        raise NotImplementedError


class SystemTimeSource(TimeSource):
    """The production time source: real monotonic clock, real sleeps.

    Stateless - safe to share between every Limiter (no per-limiter clock).
    """

    __slots__ = ()

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    async def async_sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


#: The single production time source.
SYSTEM_TIME = SystemTimeSource()


class ClockBackwardError(ValueError):
    """Raised when a strictly-monotonic virtual clock is asked to go backwards.

    Subclasses ``ValueError`` so the consistent clock-regression policy ("the
    clock only moves forward; going back is an input error") participates in
    the library's normal error layering instead of being swallowed like a rate
    decision.
    """


class VirtualTimeSource(TimeSource):
    """Deterministic strictly-monotonic virtual clock - TEST ONLY.

    Time starts at ``start_ms`` and only advances when a test calls
    :meth:`advance` or :meth:`advance_async`. Both the sync blocking loop
    (``time.sleep`` replacement) and the async loop (``asyncio.sleep``
    replacement) park waiters here, so a virtual clock drives the *real*
    blocking-retry code without real wall-clock delays.

    Backward movement is rejected with :class:`ClockBackwardError`
    (a ``ValueError``); the monotonic invariant is enforced rather than
    papered over, matching what ``time.monotonic`` promises in production.
    """

    #: Real-time guard for a sync virtual wait that is never resumed. A test
    #: that forgets to advance the clock fails loudly here instead of hanging.
    sync_wait_real_timeout = 30.0

    def __init__(self, start_ms: int = 0):
        self._now_ms = int(start_ms)
        self._cond = threading.Condition()
        self._sync_waiters: List[Tuple[float, threading.Event]] = []
        self._async_waiters: List[Tuple[float, "asyncio.Future[None]"]] = []

    # ------------------------------------------------------------------ clock
    def now_ms(self) -> int:
        with self._cond:
            return self._now_ms

    def monotonic(self) -> float:
        return self.now_ms() / 1000

    def _bump(self, delta_ms: int) -> int:
        if delta_ms < -_CLOCK_REGRESSION_TOLERANCE_MS:
            raise ClockBackwardError(
                f"virtual clock cannot move backwards by {-delta_ms} ms "
                f"(now={self._now_ms})"
            )
        self._now_ms += delta_ms
        return self._now_ms

    def advance(self, delta_ms: int) -> int:
        """Advance the clock synchronously by ``delta_ms``, waking due sync
        waiters. Returns the new time. Async waiters are NOT resumed here (an
        event loop must run to schedule their callbacks); use
        :meth:`advance_async` inside a running loop for those."""
        with self._cond:
            new_ms = self._bump(delta_ms)
            due, pending = self._partition_due(self._sync_waiters, new_ms)
            self._sync_waiters = pending
        for event in due:
            event.set()
        return new_ms

    async def advance_async(self, delta_ms: int) -> int:
        """Advance the clock, wake due async waiters, and pump the event loop
        until they (and anything they unblock) have settled. Returns the new
        time."""
        with self._cond:
            new_ms = self._bump(delta_ms)
            due, pending = self._partition_due(self._async_waiters, new_ms)
            self._async_waiters = pending

        for fut in due:
            if not fut.done():
                fut.set_result(None)

        if due:
            # Pump control until every waiter's continuation (put, retry,
            # possibly another sleep registration) has run. Bounded so a
            # self-rescheduling task cannot loop forever.
            for _ in range(1000):
                await asyncio.sleep(0)
                with self._cond:
                    still_pending = any(not fut.done() for _, fut in due)
                if not still_pending:
                    break
        return new_ms

    @staticmethod
    def _partition_due(waiters, now_ms):
        due = []
        pending = []
        for wake_at, handle in waiters:
            (due if wake_at <= now_ms else pending).append((wake_at, handle))
        return [handle for _, handle in due], [(wake_at, handle) for wake_at, handle in pending]

    # ------------------------------------------------------------------ sleep
    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        wake_at = self.monotonic() + seconds
        event = threading.Event()
        with self._cond:
            self._sync_waiters.append((wake_at, event))
        released = event.wait(timeout=self.sync_wait_real_timeout)
        if not released:
            raise TimeoutError(
                "virtual clock was not advanced far enough within "
                f"{self.sync_wait_real_timeout:.0f}s; a sync waiter at "
                f"t={wake_at:.3f}s is still parked (forgot advance()?)"
            )

    async def async_sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        loop = asyncio.get_running_loop()
        wake_at = self.monotonic() + seconds
        fut: "asyncio.Future[None]" = loop.create_future()
        with self._cond:
            self._async_waiters.append((wake_at, fut))
        try:
            await fut
        except asyncio.CancelledError:
            # Cancellation must not leave a parked sleeper that a later
            # advance() would resurrect.
            with self._cond:
                self._async_waiters = [
                    (w, f) for w, f in self._async_waiters if f is not fut
                ]
            raise

    # --------------------------------------------------------------- test aid
    def pending_sync_waiters(self) -> int:
        with self._cond:
            return len(self._sync_waiters)

    def pending_async_waiters(self) -> int:
        with self._cond:
            return sum(1 for _, fut in self._async_waiters if not fut.done())
