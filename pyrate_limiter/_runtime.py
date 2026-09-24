"""Converged sync/async acquire boundary.

The sync and async limiter paths used to read the clock, plan retry sleeps and
turn a deadline expiry into ``TimeoutError`` independently, with subtly
different rules (the async branch asserted ``delay >= 0`` while the sync
branch clamped it). Everything both paths need lives here:

- :class:`TimeService` - the single place that reads a monotonic clock,
  builds deadlines, clamps a regressing clock and performs the actual wait
  (``time.sleep`` in sync code, ``asyncio.sleep`` in async code). In
  production it wraps the process monotonic clock; tests inject a
  :class:`~pyrate_limiter.clocks.VirtualMonotonicClock` and waits become
  instant clock advances driven by virtual time.
- :func:`wait_to_sleep_ms` - the one ``waiting()`` -> sleep-duration mapping
  (``-1`` means "can never fit", negatives clamp to zero, ``buffer_ms`` is
  added once).
- :func:`plan_delay_step` - the one deadline/sleep-step computation.

The production ``Limiter`` and its tests share exactly these functions; there
is no test-only copy of the algorithm.
"""

from __future__ import annotations

import asyncio
import logging
from time import monotonic as _system_monotonic
from time import sleep as _system_sleep
from typing import Any, Awaitable, Callable, Optional, Tuple, Union

from .clocks import AbstractClock

logger = logging.getLogger(__name__)

#: Bucket.waiting() reports this when the weight can never fit.
NEVER_FITS: int = -1


class TimeService:
    """Monotonic time + waiting boundary shared by sync and async acquires.

    ``clock`` is the millisecond monotonic source stamped onto items (usually
    the bucket's clock). ``monotonic()`` is a float-seconds source used for
    deadline bookkeeping; in production it is ``time.monotonic``. When a
    virtual clock is injected both are the same virtual source, so virtual
    deadline time and virtual item timestamps move in lockstep.

    Clock-regression policy is enforced once, here: readings are never allowed
    to move backwards (clamped to the latest observed value). The stored item
    timestamps may therefore stay equal across a regression, but never shrink,
    so a window cannot be "re-opened" and an over-limit admit cannot occur.
    """

    def __init__(
        self,
        clock: Optional[AbstractClock] = None,
        *,
        virtual: Optional[bool] = None,
        monotonic: Callable[[], float] = _system_monotonic,
        sleep_sync: Callable[[float], None] = _system_sleep,
        sleep_async: Optional[Callable[..., Awaitable[None]]] = None,
    ):
        self._clock = clock
        self._monotonic = monotonic
        self._sleep_sync = sleep_sync
        self._sleep_async = sleep_async or asyncio.sleep
        # An injected/virtual clock means waits advance virtual time instead
        # of sleeping on a wall clock.
        self.virtual = bool(virtual) if virtual is not None else clock is not None
        self._last_seconds: Optional[float] = None

    def now_ms(self) -> int:
        """Current monotonic time in milliseconds, clamped non-regressing."""
        now = self.monotonic()
        return int(now * 1000)

    def monotonic(self) -> float:
        """Monotonic float seconds, clamped non-regressing."""
        value = float(self._monotonic())
        if self._last_seconds is not None and value < self._last_seconds:
            logger.debug("monotonic clock regressed; clamping %f -> %f", value, self._last_seconds)
            value = self._last_seconds
        self._last_seconds = value
        return value

    def deadline(self, timeout: float) -> Optional[float]:
        """Absolute deadline for ``timeout`` seconds; ``None`` when unbounded."""
        if timeout == -1:
            return None
        return self.monotonic() + timeout

    def remaining(self, deadline: Optional[float]) -> Optional[float]:
        """Seconds left until the deadline; ``None`` if there is no deadline."""
        if deadline is None:
            return None
        return deadline - self.monotonic()

    def is_expired(self, deadline: Optional[float]) -> bool:
        return deadline is not None and deadline - self.monotonic() <= 0

    def plan_step(self, deadline: Optional[float], delay_ms: Union[int, float]) -> Tuple[float, bool]:
        return plan_delay_step(deadline, delay_ms, monotonic=self.monotonic)

    def sleep(self, seconds: float) -> None:
        """Sync wait. Under virtual time, advance the clock by the duration."""
        if self.virtual:
            self._advance_virtual_ms(seconds * 1000)
            return
        self._sleep_sync(seconds)

    async def async_sleep(self, seconds: float) -> None:
        """Async wait. Under virtual time, advance the clock and yield control
        twice so peer tasks scheduled at the same instant get their turn
        before the woken task re-attempts."""
        if self.virtual:
            self._advance_virtual_ms(seconds * 1000)
            await self._sleep_async(0)
            await self._sleep_async(0)
            return
        await self._sleep_async(seconds)

    def _advance_virtual_ms(self, delta_ms: float) -> None:
        clock = self._clock
        advance = getattr(clock, "advance", None)
        if clock is None or not callable(advance):
            # Defensive: a virtual service must have an advanceable clock; a
            # plain read-only clock under virtual time would make waits hang
            # forever. Report explicitly rather than spinning.
            raise RuntimeError("virtual TimeService requires an advanceable clock")
        advance(int(delta_ms))
        # Keep the float-seconds view aligned with the same virtual source.
        self._last_seconds = clock.now() / 1000


def wait_to_sleep_ms(delay: int, buffer_ms: int) -> Optional[int]:
    """Translate a ``bucket.waiting()`` result into milliseconds to sleep.

    Returns ``None`` when the item can never fit (``delay == -1``: the weight
    exceeds the rate limit). This mapping is shared verbatim by the sync and
    async retry loops.
    """
    if delay == NEVER_FITS:
        return None
    if delay < 0:
        delay = 0
    return delay + buffer_ms


def plan_delay_step(
    deadline: Optional[float],
    delay_ms: Union[int, float],
    *,
    monotonic: Callable[[], float] = _system_monotonic,
) -> Tuple[float, bool]:
    """Plan one delay step.

    Returns ``(seconds_to_sleep, raise_timeout_after_sleep)``. Raises
    ``TimeoutError`` immediately (no sleep) when the deadline has passed.
    """
    if deadline is None:
        return delay_ms / 1000, False
    remaining_ms = (deadline - monotonic()) * 1000
    if remaining_ms <= 0:
        raise TimeoutError()
    if remaining_ms < delay_ms:
        return remaining_ms / 1000, True
    return delay_ms / 1000, False

