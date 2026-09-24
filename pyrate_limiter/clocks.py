"""Clock implementation using different backend"""

from __future__ import annotations

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from time import monotonic_ns, time_ns
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional, Union

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


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


class VirtualClock(AbstractClock):
    """Manually controlled monotonic clock for deterministic tests.

    Time is an integer millisecond counter that only moves when a test tells
    it to (``advance``) or when a limiter driven by this clock waits
    (``sleep``/``asleep`` move it forward by the waited duration instead of
    blocking real time). This keeps the bucket's item timestamps and the
    Limiter's deadline math on one shared timeline, so blocking acquire paths
    can be exercised instantly and exactly.

    Monotonicity is enforced by default: ``advance`` rejects a negative delta.
    NTP-style corrections are represented explicitly by ``move_back``; callers
    that allow them choose the policy at construction (``on_backwards``).
    ``sleep``/``asleep`` never move time backwards.

    ``history`` records every observation/move in call order; assertions on it
    pin the first-violating position rather than a bare true/false.

    Production code uses the same path: Limiter wraps this clock in its normal
    time-source boundary (no test-only copy of the deadline/delay math).
    """

    #: backwards movement is rejected (default monotonic policy)
    RAISE = "raise"
    #: backwards movement is accepted (simulates a corrected wall clock)
    ALLOW = "allow"
    #: backwards movement is clamped to a no-op
    CLAMP = "clamp"

    def __init__(
        self,
        start_ms: int = 0,
        *,
        on_backwards: str = RAISE,
        on_sleep: Optional[Callable[[float], None]] = None,
    ):
        if on_backwards not in (self.RAISE, self.ALLOW, self.CLAMP):
            raise ValueError(f"invalid on_backwards policy: {on_backwards!r}")
        self._now_ms = int(start_ms)
        self._on_backwards = on_backwards
        self._on_sleep = on_sleep
        self._cond = threading.Condition()
        self.history: List[str] = []

    def now(self) -> int:
        with self._cond:
            value = self._now_ms
        self.history.append(f"now->{value}")
        return value

    def advance(self, delta_ms: Union[int, float]) -> int:
        """Move time forward by ``delta_ms`` (non-negative under RAISE)."""
        if delta_ms < 0:
            if self._on_backwards == self.RAISE:
                raise ValueError("VirtualClock is monotonic: cannot advance backwards")
            if self._on_backwards == self.CLAMP:
                delta_ms = 0
        return self._move(int(delta_ms))

    def move_back(self, delta_ms: Union[int, float]) -> int:
        """Explicitly model a clock correction of ``-delta_ms``.

        Applies the ``on_backwards`` policy chosen at construction and returns
        the resulting time, so callers can assert the chosen strategy instead
        of guessing it.
        """
        if delta_ms < 0:
            raise ValueError("Invalid move_back delta: must be >= 0")
        if self._on_backwards == self.RAISE:
            raise ValueError("VirtualClock policy rejects backwards movement")
        if self._on_backwards == self.CLAMP:
            return self._move(0)
        return self._move(-int(delta_ms))

    def _move(self, delta_ms: int) -> int:
        with self._cond:
            self._now_ms += delta_ms
            value = self._now_ms
            self._cond.notify_all()
        self.history.append(f"advance->{value}")
        return value

    def wait_until(self, target_ms: int, timeout: Optional[float] = None) -> bool:
        """Block a real test thread until virtual time reaches ``target_ms``.

        Used with barriers so concurrent blocking acquirers can be parked at an
        exact virtual instant. Returns False on timeout.
        """
        with self._cond:
            return self._cond.wait_for(lambda: self._now_ms >= target_ms, timeout=timeout)

    def sleep(self, seconds: Union[int, float]) -> None:
        """Limiter sync-wait hook: advance virtual time, never block real time."""
        if seconds < 0:
            raise ValueError("sleep length must be non-negative")
        delta_ms = int(seconds * 1000)
        self._move(delta_ms)
        if self._on_sleep is not None:
            self._on_sleep(seconds)

    async def asleep(self, seconds: Union[int, float]) -> None:
        """Limiter async-wait hook: advance virtual time, yield to the loop.

        Time advances one millisecond per event-loop yield, so other tasks
        blocked on this acquire get scheduled at each new virtual instant and
        nothing blocks the event loop for real. Cancellation propagates at a
        yield boundary, so a cancelled wait can never push the clock past the
        point the cancelled task had actually reached (and hence can never
        spend a permit on its behalf).
        """
        if seconds < 0:
            raise ValueError("sleep duration must be non-negative")
        remaining_ms = int(seconds * 1000)
        if remaining_ms == 0:
            # Still yields once, matching asyncio.sleep(0).
            await asyncio.sleep(0)
            return
        while remaining_ms > 0:
            self._move(1)
            remaining_ms -= 1
            if self._on_sleep is not None:
                self._on_sleep(0.001)
            await asyncio.sleep(0)


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
