"""Clock implementation using different backend"""

from __future__ import annotations

import asyncio
import logging
from threading import Event, RLock
from abc import ABC, abstractmethod
from time import monotonic_ns, time_ns
from typing import TYPE_CHECKING, Awaitable, List, Optional, Tuple, Union

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
    """Deterministic monotonic clock for tests.

    Time is virtual milliseconds; it moves only through ``advance()`` or
    through sleeps registered against this clock (sync ``Event`` waits from
    worker threads, async futures from the event loop). ``now()`` can never go
    backwards by itself, so the same clock instance can stamp bucket items and
    drive a Limiter's deadlines/waits with zero real delay - the sync and
    async acquire paths then run their production logic unchanged.

    This is part of the public test surface; the Limiter recognises the
    ``virtual_time`` marker (see ``Limiter``) and routes all its time reads and
    sleeps through this clock.
    """

    #: Marker the Limiter looks for when choosing its time boundary.
    virtual_time: bool = True

    def __init__(self, start_ms: int = 0):
        if not isinstance(start_ms, int):
            msg = "VirtualClock start_ms must be an int"
            raise TypeError(msg)

        self._now_ms = start_ms
        # Sync blocking waits run in worker threads while a test thread drives
        # advance(); every mutation is therefore guarded.
        self._lock = RLock()
        self._sync_waits: List[Tuple[int, Event]] = []
        self._async_waits: List[Tuple[int, "asyncio.AbstractEventLoop", "asyncio.Future"]] = []

    def now(self) -> int:
        """Current virtual time, in milliseconds."""
        with self._lock:
            return self._now_ms

    def advance(self, ms: int) -> int:
        """Move time forward by ``ms`` milliseconds (>= 0), waking every sleep
        registered at or before the new time. Returns the new time.

        Going backwards is refused: a clock that cannot go back is exactly the
        monotonic guarantee callers rely on.
        """
        if not isinstance(ms, int):
            msg = "VirtualClock.advance() expects an int number of milliseconds"
            raise TypeError(msg)

        if ms < 0:
            msg = "VirtualClock cannot move backwards"
            raise ValueError(msg)

        with self._lock:
            return self._fire_due(self._now_ms + ms)

    # -------------------------------------------------------------- internal

    def _register_sync(self, wake_ms: int) -> Event:
        event = Event()
        with self._lock:
            self._sync_waits.append((wake_ms, event))
        return event

    def _unregister_sync(self, event: Event) -> None:
        with self._lock:
            self._sync_waits = [entry for entry in self._sync_waits if entry[1] is not event]

    def _register_async(self, wake_ms: int) -> "asyncio.Future":
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._lock:
            self._async_waits.append((wake_ms, loop, future))
        return future

    def _unregister_async(self, future: "asyncio.Future") -> None:
        with self._lock:
            self._async_waits = [entry for entry in self._async_waits if entry[2] is not future]

    def earliest_wake(self) -> Optional[int]:
        """Smallest registered wake time, or None when nobody is sleeping."""
        with self._lock:
            wakes = [entry[0] for entry in self._sync_waits]
            wakes.extend(entry[0] for entry in self._async_waits)
            return min(wakes) if wakes else None

    def drive_to_next_wake(self) -> Optional[int]:
        """Advance to the earliest registered sleep and wake what is due."""
        with self._lock:
            target = self.earliest_wake_locked()
            if target is None:
                return None
            return self._fire_due(target)

    def earliest_wake_locked(self) -> Optional[int]:
        wakes = [entry[0] for entry in self._sync_waits]
        wakes.extend(entry[0] for entry in self._async_waits)
        return min(wakes) if wakes else None

    def _fire_due(self, target_ms: int) -> int:
        """Commit ``target_ms`` as the new time and release due sleepers.

        Caller must hold ``self._lock``; futures/events are resolved only after
        releasing it to avoid re-entering it from a woken thread.
        """
        if target_ms < self._now_ms:
            msg = "VirtualClock cannot move backwards"
            raise ValueError(msg)

        due_sync = [entry for entry in self._sync_waits if entry[0] <= target_ms]
        due_async = [entry for entry in self._async_waits if entry[0] <= target_ms]
        self._sync_waits = [entry for entry in self._sync_waits if entry[0] > target_ms]
        self._async_waits = [entry for entry in self._async_waits if entry[0] > target_ms]
        self._now_ms = target_ms

        for _, event in due_sync:
            event.set()

        for _, loop, future in due_async:
            if not future.done():
                loop.call_soon_threadsafe(future.set_result, None)

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
