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


# ---------------------------------------------------------------------------
# Timing backends for the Limiter acquire path
#
# The Limiter reads time (deadline/remaining math) and sleeps (blocking retry
# loops) in BOTH its sync and async branches. Funneling both primitives behind
# one ``LimiterClock`` protocol is what lets those branches share a single
# code path - and what lets tests drive that path with a virtual clock instead
# of real ``time.monotonic()`` sleeps.
# ---------------------------------------------------------------------------

import asyncio
import threading
from time import monotonic as _system_monotonic
from time import sleep as _system_sleep
from typing import Protocol, Union, runtime_checkable


@runtime_checkable
class LimiterClock(Protocol):
    """Timing primitives consumed by :class:`Limiter` (seconds, float).

    ``now`` MUST be monotonic: implementations must never report a value
    smaller than one they reported earlier. The production implementation
    wraps ``time.monotonic``; tests use :class:`VirtualClock`.
    """

    def now(self) -> Union[int, float]:
        """Monotonic current time, in seconds."""

    def sleep(self, seconds: float) -> None:
        """Blocking sleep (used by the sync acquire loop)."""

    async def asleep(self, seconds: float) -> None:
        """Awaitable sleep (used by the async acquire loop)."""


class SystemClock:
    """Production ``LimiterClock``: real monotonic time and real sleeps.

    Stateless, so every Limiter that does not inject a clock shares the same
    behaviour without sharing any mutable timing state - advancing one
    Limiter's injected clock can never move another Limiter's clock.
    """

    __slots__ = ()

    def now(self) -> float:
        return _system_monotonic()

    def sleep(self, seconds: float) -> None:
        _system_sleep(seconds)

    async def asleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


#: Default shared instance; ``SystemClock`` holds no state, so a singleton is safe.
SYSTEM_CLOCK = SystemClock()


class VirtualClock(AbstractClock):
    """Deterministic monotonic clock for tests, serving both clock contracts.

    As :class:`AbstractClock` it reports integer MILLISECONDS for bucket item
    timestamps (``wrap_item``/``now``); as :class:`LimiterClock` it reports
    float SECONDS via :meth:`seconds` and provides virtual ``sleep``/``asleep``
    that complete only when test code advances the clock with :meth:`advance`.
    One instance shared between a bucket and its Limiter therefore drives both
    timestamping and waiting on a single timeline.

    Time can only move forward: ``advance`` with a negative delta raises
    ``ValueError`` instead of moving time backwards - the consistent
    clock-regression policy (monotonic timestamps can never be un-written).

    A blocked ``sleep``/``asleep`` that is never woken within
    ``deadlock_timeout`` real seconds raises ``RuntimeError`` so a test that
    deadlocks its virtual timeline fails loudly at the first offending wait
    rather than hanging forever.
    """

    def __init__(self, start_ms: int = 0, deadlock_timeout: float = 30.0):
        self._now_ms = int(start_ms)
        self._deadlock_timeout = deadlock_timeout
        self._condition = threading.Condition()
        # Sorted list of (deadline_ms, future); entries are pruned on advance.
        self._async_waiters: list = []

    # -- AbstractClock (milliseconds, consumed by buckets) ------------------

    def now(self) -> int:
        with self._condition:
            return self._now_ms

    # -- LimiterClock (seconds, consumed by the Limiter acquire path) -------

    def seconds(self) -> float:
        with self._condition:
            return self._now_ms / 1000

    def advance(self, delta_ms: int) -> None:
        """Move virtual time forward by ``delta_ms`` milliseconds.

        Negative deltas raise ``ValueError``; zero is allowed and yields to
        already-parked event-loop tasks whose deadline is due.
        """
        if delta_ms < 0:
            msg = f"VirtualClock is monotonic and cannot move backwards (delta_ms={delta_ms})"
            raise ValueError(msg)

        due_futures = []
        with self._condition:
            self._now_ms += int(delta_ms)
            remaining = []
            for deadline_ms, fut in self._async_waiters:
                if fut.done():
                    continue
                if deadline_ms <= self._now_ms:
                    due_futures.append(fut)
                else:
                    remaining.append((deadline_ms, fut))
            self._async_waiters = remaining
            self._condition.notify_all()

        # Resolve futures from the event-loop thread; call_soon_threadsafe is
        # required because advance() is typically called from the test thread.
        for fut in due_futures:
            if not fut.done():
                fut.get_loop().call_soon_threadsafe(_set_result_if_pending, fut, None)

    def sleep(self, seconds: float) -> None:
        deadline_ms = self.now() + max(0, int(round(seconds * 1000)))
        with self._condition:
            woke = self._condition.wait_for(
                lambda: self._now_ms >= deadline_ms, timeout=self._deadlock_timeout
            )
        if not woke:
            msg = f"VirtualClock.sleep({seconds}s) timed out: virtual time never reached {deadline_ms}ms"
            raise RuntimeError(msg)

    async def asleep(self, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        deadline_ms = self.now() + max(0, int(round(seconds * 1000)))
        fut = loop.create_future()
        with self._condition:
            self._async_waiters.append((deadline_ms, fut))
            self._async_waiters.sort(key=lambda entry: entry[0])

        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout=self._deadlock_timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"VirtualClock.asleep({seconds}s) timed out: virtual time never reached {deadline_ms}ms"
            ) from exc
        finally:
            with self._condition:
                self._async_waiters = [entry for entry in self._async_waiters if entry[1] is not fut]
            if not fut.done():
                fut.cancel()

    # -- LimiterClock view (seconds) sharing this exact timeline ------------

    @property
    def timer(self) -> "_VirtualTimer":
        """Seconds-since-epoch :class:`LimiterClock` view of this clock.

        The view holds no timeline of its own: ``now``/``sleep``/``asleep``
        read and park on the VirtualClock's single millisecond timeline, which
        is why one instance can timestamp bucket items AND drive Limiter waits.
        """
        return _VirtualTimer(self)

    # -- introspection for test invariants ----------------------------------

    def pending_async_waits(self) -> int:
        """How many async sleepers are still parked on this clock."""
        with self._condition:
            return sum(1 for _, fut in self._async_waiters if not fut.done())


def _set_result_if_pending(fut, value) -> None:
    if not fut.done():
        fut.set_result(value)


class _VirtualTimer:
    """Adapter exposing a :class:`VirtualClock` as a seconds-based
    :class:`LimiterClock`. Stateless apart from its reference - all timing
    state stays on the shared VirtualClock."""

    __slots__ = ("_vc",)

    def __init__(self, vc: "VirtualClock"):
        self._vc = vc

    def now(self) -> float:
        return self._vc.seconds()

    def sleep(self, seconds: float) -> None:
        self._vc.sleep(seconds)

    async def asleep(self, seconds: float) -> None:
        await self._vc.asleep(seconds)
