"""Clock implementation using different backend"""

from __future__ import annotations

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from time import monotonic_ns, time_ns
from typing import TYPE_CHECKING, Awaitable, Union

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


def isawaitable_value(value) -> bool:
    # Local helper to avoid importing inspect into every clock call.
    return hasattr(value, "__await__")


class AbstractClock(ABC):
    """Clock that return timestamp for `now`"""

    @abstractmethod
    def now(self) -> Union[int, Awaitable[int]]:
        """Get time as of now, in milliseconds"""

    @staticmethod
    def _get_monotonic_ms() -> int:
        """Get monotonic time in milliseconds"""
        return monotonic_ns() // 1_000_000

    @staticmethod
    def _get_wall_ms() -> int:
        """Get wall-clock (epoch) time in milliseconds.

        Used as the local fallback for clocks whose stored timestamps are
        wall-clock epoch ms (e.g. PostgresClock); must NOT be monotonic, which
        is on an unrelated, far smaller scale.
        """
        return time_ns() // 1_000_000


class MonotonicClock(AbstractClock):
    def now(self) -> int:
        """Get monotonic time in milliseconds"""
        return self._get_monotonic_ms()


class WallClock(AbstractClock):
    """Wall-clock epoch milliseconds.

    Needed where timestamps are compared across processes or hosts - a
    monotonic clock is only meaningful within one machine's boot. Use it for
    any bucket whose state is shared through Redis.
    """

    def now(self) -> int:
        return self._get_wall_ms()


class MonotonicAsyncClock(AbstractClock):
    """Monotonic Async Clock, meant for testing only"""

    async def now(self) -> int:
        """Get monotonic time in milliseconds"""
        return self._get_monotonic_ms()


class StableMonotonicClock(AbstractClock):
    """Monotonic view over another clock with a single consistent rollback policy.

    Every read is clamped to ``max(underlying.now(), last_read)``: an observed
    clock can never move backwards. Both the bucket (item timestamps) and the
    limiter (deadlines) read through this one wrapper, so an injected clock that
    hiccups cannot extend a deadline or put an item "before" already-admitted
    items. The default `MonotonicClock` never decreases, so the clamp is a
    no-op there.
    """

    def __init__(self, clock: AbstractClock):
        self._clock = clock
        self._last: Union[int, None] = None
        self._lock = threading.RLock()

    def now(self) -> Union[int, Awaitable[int]]:
        value = self._clock.now()

        if isawaitable_value(value):

            async def _stabilized():
                observed = await value
                return self._observe(observed)

            return _stabilized()

        return self._observe(value)

    def _observe(self, observed: int) -> int:
        with self._lock:
            if self._last is not None and observed < self._last:
                logger.debug("Clock moved backwards (%s -> %s); clamping to %s", observed, self._last, self._last)
                observed = self._last
            self._last = observed
            return observed

class VirtualMonotonicClock(AbstractClock):
    """Deterministic, explicitly-advanced monotonic clock for tests.

    Time only moves via ``advance()``/``advance_to()``; ``now()`` never
    decreases. Waiters register through ``sleep_ms()`` / ``asleep_ms()`` -
    these are the SAME hooks the production ``Limiter`` calls (via
    ``_TimeControl``), so a test drives the genuine acquire path rather than a
    copy: timestamp reading, deadline math, bucket re-put and delay retry all
    see identical virtual readings.

    ``advance()`` wakes every waiter scheduled at or before the new time
    synchronously - both threads parked in ``sleep_ms`` and event-loop tasks
    parked in ``asleep_ms`` - in scheduled-time order, giving deterministic
    barrier-style concurrency tests.
    """

    def __init__(self, start_ms: int = 0):
        assert start_ms >= 0, "virtual clock start must be >= 0"
        self._now_ms = int(start_ms)
        self._lock = threading.RLock()
        self._waiters: list = []

    def now(self) -> int:
        return self._now_ms

    def advance(self, delta_ms: int) -> int:
        """Move forward by ``delta_ms`` (must be >= 0) and wake due waiters."""
        assert delta_ms >= 0, "virtual clock can only advance forward"
        return self.advance_to(self._now_ms + delta_ms)

    def advance_to(self, target_ms: int) -> int:
        assert target_ms >= self._now_ms, "virtual clock can only advance forward"
        with self._lock:
            self._now_ms = int(target_ms)
            due = [entry for entry in self._waiters if entry[0] <= self._now_ms]
            self._waiters = [entry for entry in self._waiters if entry[0] > self._now_ms]

        for at_ms, signal, loop, fut in sorted(due, key=lambda entry: entry[0]):
            if fut is not None:
                loop.call_soon_threadsafe(_resolve_future, fut)
            else:
                signal.set()

        return self._now_ms

    def sleep_ms(self, milliseconds: Union[int, float]) -> None:
        """Block (real thread) until the virtual clock reaches the target time."""
        if milliseconds <= 0:
            return

        target_ms = self._now_ms + milliseconds
        signal = threading.Event()

        with self._lock:
            if target_ms <= self._now_ms:
                return
            self._waiters.append((target_ms, signal, None, None))

        signal.wait()

    async def asleep_ms(self, milliseconds: Union[int, float]) -> None:
        """Park the current event-loop task until the virtual clock is advanced."""
        if milliseconds <= 0:
            return

        target_ms = self._now_ms + milliseconds

        with self._lock:
            if target_ms <= self._now_ms:
                return
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            entry = (target_ms, None, asyncio.get_running_loop(), fut)
            self._waiters.append(entry)

        try:
            await fut
        except asyncio.CancelledError:
            with self._lock:
                if entry in self._waiters:
                    self._waiters.remove(entry)
            raise


def _resolve_future(fut: "asyncio.Future") -> None:
    if not fut.done():
        fut.set_result(None)


class PostgresClock(AbstractClock):
    """Get timestamp using Postgres as remote clock backend"""

    def __init__(self, pool: "ConnectionPool"):
        self.pool = pool

    def now(self) -> int:
        """Get current time in milliseconds using Postgres.

        Falls back to local time if the DB query fails for any reason.
        """
        # used `clock_timestamp` instead of `current_timestamp`,
        # because we want the actual current time
        query = "SELECT (extract(epoch FROM clock_timestamp()) * 1000)::bigint"

        try:
            with (
                self.pool.connection() as connection,
                # the cursor should be explicitly closed to avoid
                # potential resource leaks. E.g. psycopg3, cursors
                # should either be used with a context manager or
                # closed explicitly after use.
                connection.cursor() as cursor,
            ):
                cursor.execute(query)
                row = cursor.fetchone()
                if row is None:
                    msg = "Postgres time query returned no rows"
                    raise RuntimeError(msg)
                return row[0]
        except Exception:
            logger.exception("Postgres time query failed, falling back to local clock")

        # Fall back to local WALL-CLOCK epoch time in milliseconds - NOT
        # monotonic. Stored item timestamps are wall-clock epoch ms (the bucket
        # writes TO_TIMESTAMP(now/1000)); monotonic_ns() is seconds-since-boot,
        # ~5 orders of magnitude smaller, so mixing it with epoch timestamps in
        # the same bucket would corrupt every window comparison and leak bound.
        return self._get_wall_ms()
