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


class VirtualClock(AbstractClock):
    """Deterministic monotonic virtual clock, meant for testing only.

    Time only moves when a test calls :meth:`advance` (or constructs a larger
    value through :meth:`set`); nothing observes the real machine clock. The
    same instance can drive both a bucket (``bucket.now()`` -> :meth:`now`)
    and the Limiter's waiting deadlines/sleeps (via a time service bound to
    it), so sync and async paths can be tested against one shared, injectable
    time boundary without real sleeps.

    Consistent clock-rollback policy: an attempt to move the clock backwards
    never rewrites history - readings are clamped to the last observed value
    (``last >= current`` semantics, matching ``time.monotonic()``). Both
    :meth:`now` and its async form apply the same shared boundary.
    """

    def __init__(self, start_ms: int = 0):
        self._now_ms = int(start_ms)

    def now(self) -> int:
        """Current virtual time, in milliseconds."""
        return self._now_ms

    async def now_async(self) -> int:
        """Async form of :meth:`now`; same reading, no suspension."""
        return self._now_ms

    def advance(self, delta_ms: int) -> int:
        """Move time forward by ``delta_ms`` (must be >= 0) and return the new time."""
        if delta_ms < 0:
            raise ValueError("VirtualClock.advance() delta must be >= 0; use set() to replay time")
        self._now_ms += int(delta_ms)
        return self._now_ms

    def set(self, value_ms: int) -> int:
        """Assign the next reading; values in the past are clamped forward.

        This is the single place the clock-rollback policy lives, so every
        writer (tests replaying scenarios, the time service driving sleeps)
        gets identical behaviour.
        """
        self._now_ms = max(self._now_ms, int(value_ms))
        return self._now_ms

    def seconds(self) -> float:
        """Current virtual time, in seconds (the Limiter deadline scale)."""
        return self._now_ms / 1000


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
