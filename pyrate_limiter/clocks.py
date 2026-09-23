"""Clock implementation using different backend"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from time import monotonic_ns, sleep, time_ns
from typing import TYPE_CHECKING, Awaitable, Optional, Union

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


class ClockRewoundError(ValueError):
    """A clock was asked to move backwards.

    Rate-limit windows rely on monotonically non-decreasing timestamps; rather
    than silently corrupting bucket state on a backwards jump, this explicit
    error (a ``ValueError``, like the other argument-validation errors in the
    package) is raised and the clock is left unchanged.
    """


class AbstractClock(ABC):
    """Clock that return timestamp for `now`"""

    @abstractmethod
    def now(self) -> Union[int, Awaitable[int]]:
        """Get time as of now, in milliseconds"""

    def sleep(self, seconds: float) -> None:
        """Blocking wait boundary shared by synchronous limiter paths.

        Defaults to real time; virtual clocks override it to advance their own
        state instead, so production code never needs a second wait code path
        for tests.
        """
        sleep(seconds)

    async def asleep(self, seconds: float) -> None:
        """Awaitable counterpart of :meth:`sleep` for asynchronous paths."""
        await asyncio.sleep(seconds)

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


class VirtualMonotonicClock(AbstractClock):
    """Deterministic, manually-advanced monotonic clock - test support only.

    Time starts at a fixed epoch (0 ms by default) and only moves forward,
    either via :meth:`advance` or automatically when the Limiter waits through
    :meth:`sleep`/:meth:`asleep`. It implements the same ``AbstractClock``
    boundary used in production (``now`` + waiting), so code under test runs the
    unmodified production algorithm - no parallel test-only scheduling.

    Both :meth:`now` and the waits are synchronous-compatible; the waits still
    yield once to the event loop in async code so concurrent tasks get their
    scheduling point.
    """

    def __init__(self, start_ms: int = 0):
        self._now_ms = int(start_ms)

    def now(self) -> int:
        """Current virtual time in milliseconds."""
        return self._now_ms

    def seconds(self) -> float:
        """Current virtual time in seconds (deadline-math view of the clock)."""
        return self._now_ms / 1000

    def advance_ms(self, delta_ms: int) -> int:
        """Advance virtual time by ``delta_ms`` milliseconds; returns the new time."""
        if delta_ms < 0:
            msg = f"virtual clock cannot rewind by {delta_ms} ms"
            raise ClockRewoundError(msg)
        self._now_ms += int(delta_ms)
        return self._now_ms

    def advance(self, seconds: float) -> int:
        """Advance virtual time by ``seconds`` (may be fractional)."""
        return self.advance_ms(int(round(seconds * 1000)))

    def set_ms(self, value_ms: int) -> None:
        """Set absolute virtual time, rejecting backwards moves.

        On a backwards attempt the clock is left untouched and
        :class:`ClockRewoundError` is raised - the single, consistent
        rewind policy shared by every virtual-clock entry point.
        """
        value_ms = int(value_ms)
        if value_ms < self._now_ms:
            msg = f"virtual clock rewind refused: {value_ms} < {self._now_ms}"
            raise ClockRewoundError(msg)
        self._now_ms = value_ms

    def sleep(self, seconds: float) -> None:
        """Advance virtual time instead of a real blocking sleep."""
        self.advance(seconds)

    async def asleep(self, seconds: float) -> None:
        """Advance virtual time and yield once so peers can be scheduled."""
        self.advance(seconds)
        await asyncio.sleep(0)
