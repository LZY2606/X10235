"""Clock implementation using different backend"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from time import monotonic_ns, time_ns
from typing import TYPE_CHECKING, Awaitable, Union

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


class ClockMovedBackward(ValueError):
    """Raised when a strictly monotonic virtual clock is asked to move back.

    Time moving backwards is never silently accepted: a bucket that compared
    item timestamps against an earlier ``now`` would resurrect expired windows
    and grant permits that were already spent. This is the single, explicit
    policy for every virtual clock - see ``VirtualMonotonicClock.advance``.
    """


class VirtualMonotonicClock(AbstractClock):
    """Deterministic, strictly monotonic clock for tests and simulations.

    Time is held in an integer millisecond counter and only moves forward via
    ``advance()``; ``now()`` never consults the real clock, so sync and async
    limiters driven by the same instance observe exactly the same timeline.
    Every instance owns its own counter, so separate limiters cannot advance
    (or otherwise leak) each other's clock.

    Going backwards is rejected with ``ClockMovedBackward`` (a ``ValueError``)
    rather than clamped: callers that intended a rollback must state it, and
    stored item timestamps must never be compared against an earlier clock.
    """

    def __init__(self, start_ms: int = 0):
        self._now_ms = int(start_ms)

    def now(self) -> int:
        """Current virtual time, in milliseconds."""
        return self._now_ms

    def advance(self, milliseconds: int) -> int:
        """Move the clock forward by ``milliseconds`` (>= 0) and return new time."""
        if milliseconds < 0:
            raise ClockMovedBackward(
                f"Virtual clock cannot move backward by {-milliseconds}ms: now={self._now_ms}"
            )

        self._now_ms += int(milliseconds)
        return self._now_ms

    def set(self, milliseconds: int) -> int:
        """Jump to an absolute timestamp; it must not precede the current one."""
        if milliseconds < self._now_ms:
            raise ClockMovedBackward(
                f"Virtual clock cannot be set to {milliseconds}ms: now={self._now_ms}"
            )

        self._now_ms = int(milliseconds)
        return self._now_ms


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
