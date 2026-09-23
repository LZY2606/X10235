"""Clock implementation using different backend"""

from __future__ import annotations

import asyncio
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
    """Manually advanced monotonic clock, meant for testing only.

    Time moves only when the test says so, which makes limiter/bucket timing
    behavior deterministic: inject the same instance into a ``Limiter`` (its
    deadline/wait scheduling) and into a bucket (``bucket._clock``) and drive
    both with ``advance()``.

    Backwards policy (consistent, by construction): ``now()`` never goes
    backwards. ``set()`` to an earlier timestamp is a clamped no-op (logged),
    and ``advance()`` with a negative delta is rejected with ``ValueError``.

    Wait seam: ``sleep_ms``/``asleep_ms`` are the test hooks ``Limiter``
    delegates to when its clock defines them, instead of really sleeping.
    They advance virtual time by the requested wait, so a blocking acquire
    completes without real waiting while exercising the exact production
    delay loop. ``MonotonicClock`` defines no such methods, so production
    keeps sleeping in real time - the seam only reroutes *how long* the
    limiter waits, never *whether* it waits.
    """

    def __init__(self, start_ms: int = 0):
        self._now_ms = start_ms

    def now(self) -> int:
        return self._now_ms

    def set(self, timestamp_ms: int) -> int:
        """Move to an absolute timestamp; refuses to go backwards."""
        if timestamp_ms < self._now_ms:
            logger.warning(
                "VirtualClock: ignoring backwards set(%d) at %d; monotonic clock never goes backwards",
                timestamp_ms,
                self._now_ms,
            )
            return self._now_ms

        self._now_ms = timestamp_ms
        return self._now_ms

    def advance(self, delta_ms: int) -> int:
        """Move forward by ``delta_ms``; negative deltas are unsupported."""
        if delta_ms < 0:
            raise ValueError(f"VirtualClock cannot advance backwards: {delta_ms}")

        self._now_ms += delta_ms
        return self._now_ms

    def sleep_ms(self, delay_ms: int) -> None:
        """Sync wait seam: a virtual sleep is exactly an advance of the wait."""
        self.advance(delay_ms)

    async def asleep_ms(self, delay_ms: int) -> None:
        """Async wait seam: advance, then yield so peers can observe the wait."""
        self.advance(delay_ms)
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
