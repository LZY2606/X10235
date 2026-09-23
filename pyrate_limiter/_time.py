"""Shared time boundary used by every Limiter acquire path.

The sync and async acquire code used to read ``time.monotonic()`` and call
``time.sleep`` / ``asyncio.sleep`` directly, so deadline math and waits could
diverge between the two paths. All of that now goes through a per-Limiter
:class:`TimeService`:

- production code uses :class:`SystemTimeService`, a thin wrapper over the
  same ``time.monotonic`` / ``time.sleep`` primitives as before;
- tests may inject :class:`VirtualTimeService`, a hand-advanced monotonic
  clock shared verbatim by the sync and async paths (no test-only copy of the
  wait math).

The virtual clock is deliberately restricted to the package internals: it is
an accepted type of the private ``_time_service`` seam, not part of the
documented public API.
"""

from __future__ import annotations

import asyncio
import heapq
import threading
import time
from typing import List, Optional, Tuple


class ClockMovedBackwardError(ValueError):
    """Raised when an injected clock reports a value older than its previous
    read.

    Subclasses :class:`ValueError` so callers that already reject invalid
    configuration via ``ValueError`` keep the same error layer. The policy is
    deliberately strict: waits are computed against a monotonic reference, so a
    backward jump cannot be honored without risking a negative wait or a
    broken deadline.
    """


class TimeService:
    """Monotonic time source plus the sleep primitives the acquire loop needs."""

    def now(self) -> float:
        """Current monotonic time, in seconds."""
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        """Blocking sleep."""
        raise NotImplementedError

    async def asleep(self, seconds: float) -> None:
        """Awaitable sleep."""
        raise NotImplementedError

    def deadline(self, timeout: float) -> Optional[float]:
        """Translate a public ``timeout`` argument into an absolute deadline.
        ``-1`` means "no deadline" (the only accepted negative value; argument
        validation happens at the public entry points).
        """
        if timeout == -1:
            return None
        return self.now() + timeout

    def remaining(self, deadline: Optional[float]) -> Optional[float]:
        """Seconds left until ``deadline``; a ``None`` deadline -> ``None``."""
        if deadline is None:
            return None
        return deadline - self.now()

    def plan_delay_step(self, deadline: Optional[float], delay_ms: float) -> Tuple[float, bool]:
        """Plan one delay step for the blocking retry loops.

        Returns ``(seconds_to_sleep, raise_timeout_after_sleep)``. Raises
        :class:`TimeoutError` immediately, without sleeping, when the deadline
        has already passed. This is the single source of deadline math shared
        by the sync and async retry loops; the callers only choose which sleep
        primitive performs the wait.
        """
        if deadline is None:
            return delay_ms / 1000, False
        remaining_ms = (deadline - self.now()) * 1000
        if remaining_ms <= 0:
            raise TimeoutError()
        if remaining_ms < delay_ms:
            return remaining_ms / 1000, True
        return delay_ms / 1000, False


class SystemTimeService(TimeService):
    """Production time service: ``time.monotonic`` plus the real sleep calls."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    async def asleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class VirtualTimeService(TimeService):
    """Hand-advanced monotonic time for deterministic tests.

    Time only moves when the driver calls :meth:`advance` / :meth:`set_time`.
    Both :meth:`sleep` (a real blocking thread wait) and :meth:`asleep`
    (parked until time advances) observe the same virtual clock, so a sync and
    an async limiter sharing one service are driven by identical timing.

    The clock rejects backward moves with :class:`ClockMovedBackwardError`,
    which pins the "clock moves backwards" policy: never silently sleep a
    negative amount and never extend a deadline.
    """

    def __init__(self, start: float = 0.0):
        self._now = float(start)
        self._cond = threading.Condition()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # (wake_at, arrival_seq, future)
        self._async_waiters: List[Tuple[float, int, "asyncio.Future[None]"]] = []
        self._seq = 0

    # ------------------------------------------------------------------ clock

    def now(self) -> float:
        with self._cond:
            return self._now

    def now_ms(self) -> int:
        """Current virtual time in whole milliseconds."""
        with self._cond:
            return int(self._now * 1000)

    def set_time(self, value: float) -> None:
        """Set an absolute virtual time; backward moves are rejected.

        Idempotent at the current value, and wakes every async sleeper whose
        wake-up time is reached in a stable time-then-arrival order.
        """
        with self._cond:
            if value < self._now:
                raise ClockMovedBackwardError(
                    f"virtual clock cannot move backwards: {value} < {self._now}"
                )
            if value > self._now:
                self._now = float(value)
                self._cond.notify_all()
            due = self._take_due_async_locked()
        self._wake_async(due)

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds`` (must be >= 0)."""
        if seconds < 0:
            raise ClockMovedBackwardError(f"virtual clock cannot advance by a negative delta: {seconds}")
        self.set_time(self.now() + seconds)

    def pending_async_sleeps(self) -> int:
        """Number of async sleepers still parked (diagnostic for tests)."""
        with self._cond:
            return len(self._async_waiters)

    # ------------------------------------------------------------------ sleeps

    def sleep(self, seconds: float) -> None:
        wake_at = self.now() + max(0.0, seconds)
        with self._cond:
            while self._now < wake_at:
                self._cond.wait()

    async def asleep(self, seconds: float) -> None:
        wake_at = self.now() + max(0.0, seconds)
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[None]" = loop.create_future()

        def _remove(done_future: "asyncio.Future[None]") -> None:
            # Cancellation or wake-up: drop the waiter so a later advance()
            # cannot resolve an already-finished future.
            with self._cond:
                for entry in self._async_waiters:
                    if entry[2] is done_future:
                        self._async_waiters.remove(entry)
                        heapq.heapify(self._async_waiters)
                        break

        future.add_done_callback(_remove)

        with self._cond:
            if self._loop is None:
                self._loop = loop
            self._seq += 1
            heapq.heappush(self._async_waiters, (wake_at, self._seq, future))

        if wake_at <= self.now():
            self._wake_async([future])

        await future

    def _take_due_async_locked(self) -> List["asyncio.Future[None]"]:
        due: List["asyncio.Future[None]"] = []
        while self._async_waiters and self._async_waiters[0][0] <= self._now:
            _wake_at, _seq, future = heapq.heappop(self._async_waiters)
            if not future.done():
                due.append(future)
        return due

    def _wake_async(self, futures: List["asyncio.Future[None]"]) -> None:
        """Resolve due futures on their event loop in wake-up order."""
        if not futures:
            return
        loops = {future.get_loop() for future in futures}
        loop = self._loop if self._loop is not None and self._loop in loops else next(iter(loops))

        def _resolve() -> None:
            for future in futures:
                if not future.done():
                    future.set_result(None)

        if loop.is_running():
            loop.call_soon_threadsafe(_resolve)
        else:
            loop.call_soon(_resolve)
