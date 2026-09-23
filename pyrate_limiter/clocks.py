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
    """Raised when a monotonic clock is asked to move backwards.

    Monotonic clocks only advance; callers that observe time going backwards
    (e.g. a test driver feeding a smaller timestamp) get an explicit, stable
    error instead of silently corrupting window math. Subclasses ``ValueError``
    so it stays inside the library's built-in error layering.
    """


class VirtualClock(AbstractClock):
    """Deterministic monotonic clock for tests and simulations.

    Time stands still until :meth:`advance` moves it forward by a fixed number
    of milliseconds; both the sync and async entry points then report that same
    value. This is what makes blocking acquire loops reproducible: pair it with
    a virtual sleep seam (the Limiter's test clock seam) and waiting only
    happens when the driver explicitly advances time.

    ``now()`` is synchronous (matching :class:`MonotonicClock`); awaiting the
    int is harmless because the async limiter normalizes non-awaitable values.
    Moving backwards is rejected with :class:`ClockMovedBackward` so the
    monotonic policy is consistent across every consumer.
    """

    def __init__(self, start_ms: int = 0):
        self._now_ms = int(start_ms)

    def now(self) -> int:
        """Current virtual time in milliseconds."""
        return self._now_ms

    def advance(self, delta_ms: int) -> int:
        """Advance virtual time by ``delta_ms`` milliseconds (must be >= 0).

        Returns the new virtual time. A negative delta raises
        :class:`ClockMovedBackward` rather than rewinding: stored bucket
        timestamps and limiter deadlines share this one clock, so a rewind
        could make admitted items appear "in the future" and silently relax
        limits.
        """
        if delta_ms < 0:
            msg = f"virtual clock cannot move backwards by {-delta_ms}ms"
            raise ClockMovedBackward(msg)
        self._now_ms += int(delta_ms)
        return self._now_ms

    def reset(self, start_ms: int = 0) -> None:
        """Reset to an explicit timestamp.

        Intended for fresh-test setup only - unlike :meth:`advance`, resetting
        to an earlier timestamp raises :class:`ClockMovedBackward`, so even an
        explicit reset cannot silently turn the clock backwards.
        """
        if start_ms < self._now_ms:
            msg = f"virtual clock cannot reset backwards: {start_ms} < {self._now_ms}"
            raise ClockMovedBackward(msg)
        self._now_ms = int(start_ms)


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
