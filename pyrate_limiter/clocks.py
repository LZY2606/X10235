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

    def time(self) -> float:
        """Get time as of now, in seconds.

        Seconds (fractional) is the scale the Limiter uses for deadlines and
        sleep planning; milliseconds is the scale stamped onto RateItems. A
        clock implementing ``now()`` gets ``time()`` for free, and - crucially -
        both views read the SAME underlying source, so an injected virtual clock
        drives bucket timestamps and limiter deadlines coherently instead of the
        two drifting on independent clocks.
        """
        return self._get_monotonic_seconds()

    @staticmethod
    def _get_monotonic_ms() -> int:
        """Get monotonic time in milliseconds"""
        return monotonic_ns() // 1_000_000

    @staticmethod
    def _get_monotonic_seconds() -> float:
        """Get monotonic time in (fractional) seconds."""
        return monotonic_ns() / 1_000_000_000

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

    def time(self) -> float:
        """Monotonic time in seconds; same source (``time.monotonic_ns``) as
        ``now()``, keeping the ms and s views consistent."""
        return monotonic_ns() / 1_000_000_000


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


class VirtualMonotonicClock(AbstractClock):
    """Deterministic monotonic clock that moves only when a test advances it.

    Production code reads it through the same boundary as every other clock:
    ``now()`` for millisecond RateItem timestamps and ``time()`` for the
    Limiter's second-scale deadlines. Both share one internal counter, so an
    in-memory bucket constructed with this clock and a Limiter constructed with
    the same instance observe one coherent virtual timeline.

    Policy on clock movement: the clock is strictly non-decreasing. Attempting
    to move it backwards is an explicit ``ValueError`` rather than a silent
    clamp - a backward jump is a test/configuration mistake and swallowing it
    would hide a broken schedule. This is the repository-wide consistent
    strategy for clock regression.
    """

    def __init__(self, start_ms: int = 0):
        if start_ms < 0:
            raise ValueError("virtual clock start must be >= 0")
        self._now_ms = int(start_ms)

    def now(self) -> int:
        """Current virtual time in milliseconds."""
        return self._now_ms

    def time(self) -> float:
        """Current virtual time in seconds; the same counter as ``now()``."""
        return self._now_ms / 1000

    def advance(self, delta_ms: int) -> int:
        """Move the clock forward by ``delta_ms`` milliseconds and return the
        new time. ``delta_ms`` must be non-negative (see class policy)."""
        if delta_ms < 0:
            raise ValueError("virtual clock cannot move backwards")
        self._now_ms += int(delta_ms)
        return self._now_ms

    def move_to(self, timestamp_ms: int) -> int:
        """Move the clock to an absolute millisecond timestamp.

        Only forward moves are accepted; moving before the current time raises
        ``ValueError`` and leaves the clock untouched.
        """
        if timestamp_ms < self._now_ms:
            raise ValueError("virtual clock cannot move backwards")
        self._now_ms = int(timestamp_ms)
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
