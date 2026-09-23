"""Clock implementation using different backend"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from time import monotonic, monotonic_ns, time_ns
from typing import TYPE_CHECKING, Awaitable, Union

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


class AbstractClock(ABC):
    """Clock that return timestamp for `now`"""

    @abstractmethod
    def now(self) -> Union[int, Awaitable[int]]:
        """Get time as of now, in milliseconds"""

    def now_seconds(self) -> float:
        """Get current time in seconds, on the same scale as ``now()``.

        This is the single seam through which a ``Limiter`` reads time for
        deadline planning, while the bucket's ``now()`` stamps items in
        milliseconds; deriving seconds from ``now()`` keeps both on one
        injected time source instead of reaching for a second system clock.
        Millisecond-only custom clocks get this for free.
        """
        return self.now() / 1000  # type: ignore[operator]

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

    @staticmethod
    def _get_monotonic_seconds() -> float:
        """Get monotonic time in seconds (``time.monotonic``)."""
        return monotonic()


class MonotonicClock(AbstractClock):
    def now(self) -> int:
        """Get monotonic time in milliseconds"""
        return self._get_monotonic_ms()


class SystemMonotonicClock(MonotonicClock):
    """Process-local monotonic clock in both milliseconds (``now``) and
    seconds (``now_seconds``).

    Seconds come straight from ``time.monotonic`` rather than a truncated
    millisecond value, keeping sub-millisecond deadline precision, while both
    methods share one underlying monotonic source. Every ``Limiter`` gets a
    fresh instance by default - limiters must not share clock state.
    """

    def now_seconds(self) -> float:
        return self._get_monotonic_seconds()


class VirtualClock(MonotonicClock):
    """Deterministic, strictly non-decreasing monotonic clock for tests.

    Time starts at ``start_ms`` and only moves forward through ``advance``;
    ``now()`` never observes wall-clock time. Backward movement is refused
    with ``ValueError`` rather than silently clamped - the consistent
    clock-regression policy shared by every caller.

    ``Limiter(..., clock=...)`` and ``InMemoryBucket(..., clock=...)`` accept
    it, so item timestamps, deadline planning and delay steps all run on the
    one injected time source.

    Note: blocking waits still sleep in real time (``time.sleep`` /
    ``asyncio.sleep``); a frozen clock only lets a deadline expire once the
    test advances it. Use non-blocking acquires or advance the clock from
    another task/thread when exercising timeout paths deterministically.
    """

    def __init__(self, start_ms: int = 0):
        self._now_ms = int(start_ms)

    def now(self) -> int:
        """Current virtual time, in milliseconds"""
        return self._now_ms

    def now_seconds(self) -> float:
        return self._now_ms / 1000

    def advance(self, milliseconds: int) -> int:
        """Move the clock forward; return the new millisecond time.

        Backward movement raises ``ValueError`` and leaves the clock
        untouched, so callers cannot corrupt monotonicity by accident.
        """
        if milliseconds < 0:
            msg = f"VirtualClock cannot move backwards: {milliseconds} ms"
            raise ValueError(msg)

        self._now_ms += int(milliseconds)
        return self._now_ms


class AsyncVirtualClock(VirtualClock):
    """:class:`VirtualClock` with an awaitable ``now()`` (async buckets)."""

    async def now(self) -> int:  # type: ignore[override]
        """Get virtual time as of now, in milliseconds"""
        return self._now_ms


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
