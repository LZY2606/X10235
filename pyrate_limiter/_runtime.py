"""Shared runtime boundary for the sync and async acquire paths.

Everything the Limiter needs that touches *wall/monotonic time or a sleep
primitive lives here, behind one ``ScheduleClock`` shape:

- clock reads (deadline computation, deadline checks),
- the sync sleep (``time.sleep``) and async sleep (``asyncio.sleep``),
- the delay/deadline math (``plan_delay_step`` / ``prepare_wait``),
- exception translation (``timeout_error`` / ``TIMEOUT_EXCEPTIONS``).

``Limiter`` keeps a single clock instance (``SystemScheduleClock`` by default),
so both its own deadline bookkeeping and, when a bucket is built with the same
clock, the items' timestamps are read from one source. A clock instance is
never global: every Limiter gets its own default and injected clocks are not
shared behind the caller's back, so two limiters cannot leak clocks into each
other.

``VirtualClock`` is the deterministic, strictly monotonic clock used by tests.
It drives the very same methods as production - no test-only copy of the
waiting/deadline algorithm exists.
"""

from __future__ import annotations

import asyncio
import heapq
import threading
from time import monotonic, sleep
from typing import Any, List, Optional, Tuple, Union

# Everything Python 3.8+ treats as a timeout: the builtin OSError subclass
# TimeoutError and asyncio's CancelledError-shaped timeout. On 3.11+
# asyncio.TimeoutError IS TimeoutError; the tuple is harmless there and keeps
# the except-clauses identical on older interpreters.
TIMEOUT_EXCEPTIONS: Tuple[type, ...] = (TimeoutError, asyncio.TimeoutError)


def timeout_error() -> TimeoutError:
    """Single constructor for the canonical timeout exception.

    Sync and async paths must raise the same type so callers catching
    ``TimeoutError`` see identical behaviour from ``try_acquire`` and
    ``try_acquire_async``.
    """
    return TimeoutError()


class ScheduleClock:
    """Clock + scheduler boundary consumed by ``Limiter``.

    ``now()`` is returned in SECONDS (matching the historical
    ``time.monotonic()`` deadlines); ``now_ms()`` gives integer milliseconds
    for buckets' ``AbstractClock`` interface. The two are the same reading on
    the same scale.
    """

    def now(self) -> float:
        return monotonic()

    def now_ms(self) -> int:
        return int(self.now() * 1000)

    def sleep(self, seconds: float) -> None:
        sleep(seconds)

    async def asleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def wait_for(self, awaitable: Any, deadline: Optional[float]) -> Any:
        """Await ``awaitable`` bounded by an absolute ``deadline`` on this clock.

        ``deadline is None`` awaits without a clock bound. On timeout the
        pending awaitable is cancelled and the canonical ``TimeoutError`` is
        raised. Same semantics for the system clock and injected virtual
        clocks; the virtual implementation races on its own waiter events
        instead of an event-loop wall timer.
        """
        if deadline is None:
            return await awaitable

        task = asyncio.ensure_future(awaitable)
        timeout_task = asyncio.ensure_future(self.asleep(max(0.0, deadline - self.now())))
        try:
            await asyncio.wait({task, timeout_task}, return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                timeout_task.cancel()
                return task.result()
            task.cancel()
            raise timeout_error()
        finally:
            if not timeout_task.done():
                timeout_task.cancel()
            if not task.done():
                task.cancel()


# One immutable default is safe because it holds no per-limiter state.
SYSTEM_CLOCK = ScheduleClock()


def plan_delay_step(
    deadline: Optional[float],
    delay_ms: Union[int, float],
    now_func=monotonic,
) -> Tuple[float, bool]:
    """Plan one delay step of a blocking retry loop.

    Returns ``(seconds_to_sleep, raise_timeout_after_sleep)``. Raises
    ``TimeoutError`` immediately (no sleep) when the deadline has already
    passed. This is the deadline math shared verbatim by the sync and async
    retry loops; callers perform the actual sleep with ``clock.sleep`` /
    ``clock.asleep`` and re-raise afterwards.
    """
    if deadline is None:
        return delay_ms / 1000, False
    remaining_ms = (deadline - now_func()) * 1000
    if remaining_ms <= 0:
        raise timeout_error()
    if remaining_ms < delay_ms:
        return remaining_ms / 1000, True
    return delay_ms / 1000, False


def prepare_wait(
    delay_ms: Optional[int],
    buffer_ms: int,
) -> Optional[int]:
    """Translate a ``bucket.waiting()`` result into the wait until re-put.

    Returns ``None`` when the item can never fit (``waiting() == -1``: the
    weight exceeds the rate limit). Shared by every retry loop so the
    ``-1`` / negative / ``+buffer_ms`` handling is identical between sync and
    async callers - historically the async branch asserted ``d >= 0`` while
    the sync branch clamped negatives to 0.
    """
    if delay_ms is None or delay_ms == -1:
        return None
    if delay_ms < 0:
        delay_ms = 0
    return delay_ms + buffer_ms


class VirtualClock(ScheduleClock):
    """Deterministic, strictly monotonic clock for tests.

    Time only moves when the test calls :meth:`advance` (sync driving) or
    :meth:`aadvance` (event-loop driving). ``asleep()`` registers a deadline
    event instead of yielding real time; ``aadvance`` resumes every waiter
    whose deadline has been reached. ``sleep()`` is driven cooperatively with
    :meth:`drive_sync` (the blocking sync path sleeps without holding the GIL
    via a real thread in tests), so a sync caller and the async loop can be
    scripted against one timeline.

    Going backwards is a defined policy, not undefined behaviour: attempts to
    move before the current reading are clamped to the current reading and
    reported via :meth:`rewind_attempts`; every observed reading is therefore
    non-decreasing.
    """

    def __init__(self, start_ms: int = 1_000_000):
        self._now_ms = int(start_ms)
        self._cond = threading.Condition()
        # (deadline_ms, seq, event) for sync sleep() callers.
        self._sync_waiters: List[tuple] = []
        # (deadline_ms, seq, asyncio.Event) for asleep() callers.
        self._async_waiters: List[tuple] = []
        self._seq = 0
        self.rewind_attempts = 0
        self.sync_sleeps = 0
        self.async_sleeps = 0

    # ------------------------------------------------------------ readings

    def now(self) -> float:
        return self._now_ms / 1000

    def now_ms(self) -> int:
        with self._cond:
            return self._now_ms

    # ------------------------------------------------------------ sleeping

    def sleep(self, seconds: float) -> None:
        """Blocking virtual sleep.

        Parks until a :meth:`drive_sync`/ :meth:`advance` crosses the
        deadline. Registration is rolled back if interrupted (e.g. by an
        exception propagating through the caller), so a cancelled waiter can
        never leave a stale entry behind.
        """
        deadline_ms = self._now_ms + int(round(seconds * 1000))
        with self._cond:
            self._seq += 1
            seq = self._seq
            event = threading.Event()
            heapq.heappush(self._sync_waiters, (deadline_ms, seq, event))
            self.sync_sleeps += 1
        interrupted = False
        try:
            event.wait()
        except BaseException:
            interrupted = True
            raise
        finally:
            if interrupted:
                with self._cond:
                    self._sync_waiters = [w for w in self._sync_waiters if w[1] != seq]
                    heapq.heapify(self._sync_waiters)

    async def asleep(self, seconds: float) -> None:
        deadline_ms = self._now_ms + int(round(seconds * 1000))
        event = asyncio.Event()
        with self._cond:
            self._seq += 1
            seq = self._seq
            heapq.heappush(self._async_waiters, (deadline_ms, seq, event))
            self.async_sleeps += 1
        try:
            await event.wait()
        except BaseException:
            # Cancelled (e.g. timeout/caller cancel): remove the registration
            # so no later advance can resolve a dead waiter.
            with self._cond:
                self._async_waiters = [w for w in self._async_waiters if w[1] != seq]
                heapq.heapify(self._async_waiters)
            raise

    async def wait_for(self, awaitable: Any, deadline: Optional[float]) -> Any:
        if deadline is None:
            return await awaitable
        if self.now() >= deadline:
            awaitable = self._cancel(awaitable)
            raise timeout_error()

        task = asyncio.ensure_future(awaitable)
        # A sleeping coroutine parks on the virtual clock; racing it against
        # the task makes the bound fire exactly when virtual time crosses the
        # deadline (never on real wall time).
        timer = asyncio.ensure_future(self.asleep(max(0.0, deadline - self.now())))
        try:
            await asyncio.wait({task, timer}, return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                return task.result()
            task.cancel()
            raise timeout_error()
        finally:
            if not timer.done():
                timer.cancel()
            if not task.done():
                task.cancel()

    # ----------------------------------------------------- moving the clock

    def _due(self, heap: List[tuple]) -> List[tuple]:
        due = []
        while heap and heap[0][0] <= self._now_ms:
            due.append(heapq.heappop(heap))
        return due

    def advance(self, delta_ms: int) -> int:
        """Move time forward for SYNC drivers; wake sync sleepers only."""
        with self._cond:
            self._bump(delta_ms)
            for _, _, event in self._due(self._sync_waiters):
                event.set()
            return self._now_ms

    def drive_sync(self, delta_ms: int, timeout: float = 5.0) -> int:
        """Advance and block until every sync sleeper woken by the move has
        re-entered a wait (or finished), i.e. the sync caller has observed the
        new time and either re-put or scheduled its next sleep."""
        deadline = monotonic() + timeout
        self.advance(delta_ms)
        while monotonic() < deadline:
            with self._cond:
                if not self._sync_waiters or self._sync_waiters[0][0] > self._now_ms:
                    return self._now_ms
            threading.Event().wait(0.002)
        raise AssertionError("drive_sync: sync waiter did not settle in time")

    async def aadvance(self, delta_ms: int) -> int:
        """Move time forward on the event loop; wake async sleepers and let
        each woken waiter re-put and re-register before the next pop, so one
        ``aadvance`` settles the whole chain of retries at the new time."""
        with self._cond:
            self._bump(delta_ms)
        while True:
            with self._cond:
                due = self._due(self._async_waiters)
            if not due:
                break
            for _, _, event in due:
                event.set()
            # Let every resumed waiter run: re-put, and either return or
            # register a new (later) waiter.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        with self._cond:
            return self._now_ms

    def pending(self) -> int:
        """Total registered waiters (diagnostic for resource-release checks)."""
        with self._cond:
            return len(self._sync_waiters) + len(self._async_waiters)

    # ------------------------------------------------------------- rewind

    def _bump(self, delta_ms: int) -> None:
        if delta_ms < 0:
            self.rewind_attempts += 1
            return
        self._now_ms += delta_ms

    def jump_to(self, target_ms: int) -> bool:
        """Explicit rewind attempt.

        Policy: a backwards jump never moves the reading backwards; it is
        clamped to the current reading and counted. Returns True when time
        actually moved."""
        with self._cond:
            if target_ms < self._now_ms:
                self.rewind_attempts += 1
                return False
            self._now_ms = int(target_ms)
            return True
