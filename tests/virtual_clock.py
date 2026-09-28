"""Deterministic virtual monotonic clock for the clock/acquire contract tests.

This is TEST SUPPORT, not a second rate-limiting algorithm: it implements the
SAME boundaries production code uses (``pyrate_limiter.clocks.AbstractClock``
for bucket timestamps and ``pyrate_limiter.timing.Timing`` for limiter reads
and sleeps). Injecting it changes *when time advances* and *what sleeps block
on* - never how the limiter computes admits, delays or deadlines.

One pair of facades (:class:`VirtualMonotonicClock` + :class:`VirtualTiming`)
shares a single :class:`_VirtualTimeState`, so bucket timestamps (int ms) and
limiter deadlines (float seconds) always describe the same virtual instant.

Backwards-time policy (deliberate and consistent):
- ``advance(-x)`` is an invalid duration and raises ``ValueError`` (never
  guessed as success).
- Any attempt to set the clock to an earlier instant is CLAMPED to the current
  value - reads are monotonic by construction, mirroring ``time.monotonic``.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from typing import List, Tuple

from pyrate_limiter.clocks import AbstractClock
from pyrate_limiter.timing import Timing


class _VirtualTimeState:
    """Shared virtual instant plus registries for virtual sleepers."""

    def __init__(self, start_ms: int = 1_000_000_000_000):
        self._now_ms = int(start_ms)
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        # (deadline_ms, seq) -> asyncio.Event, in registration order.
        self._async_waiters: List[Tuple[int, int, asyncio.Event]] = []
        self._seq = itertools.count()

    def now_ms(self) -> int:
        with self._lock:
            return self._now_ms

    def clamp(self, ms: int) -> int:
        """Set an absolute time, never moving backwards.

        Returns the resulting time (== current when ``ms`` lay in the past).
        """
        with self._cond:
            if ms > self._now_ms:
                self._now_ms = ms
                due_ms = ms
            else:
                due_ms = self._now_ms
            self._wake_sync()
            self._wake_async_due(due_ms)
            return self._now_ms

    def advance(self, delta_ms: int) -> int:
        """Move time forward by ``delta_ms`` (must be >= 0)."""
        if delta_ms < 0:
            raise ValueError(f"virtual clock cannot move backwards: advance({delta_ms})")
        with self._cond:
            self._now_ms += int(delta_ms)
            due_ms = self._now_ms
            self._wake_sync()
            self._wake_async_due(due_ms)
            return self._now_ms

    def _wake_sync(self) -> None:
        # Caller holds self._cond.
        self._cond.notify_all()

    def _wake_async_due(self, now_ms: int) -> None:
        """Set every async waiter due by ``now_ms`` in (deadline, seq) order.

        Deterministic wake order is what lets scheduling-order tests be
        reproducible.
        """
        due = [w for w in self._async_waiters if w[0] <= now_ms]
        if not due:
            return
        due.sort(key=lambda w: (w[0], w[1]))
        for deadline_ms, _seq, event in due:
            self._async_waiters.remove((deadline_ms, _seq, event))
            event.set()

    def sync_wait_until(self, deadline_ms: int) -> None:
        with self._cond:
            while self._now_ms < deadline_ms:
                self._cond.wait()

    async def async_wait_until(self, deadline_ms: int) -> None:
        if deadline_ms <= self.now_ms():
            return
        event = asyncio.Event()
        with self._lock:
            seq = next(self._seq)
            self._async_waiters.append((deadline_ms, seq, event))
            # Re-check under the lock: time may have advanced while registering.
            if deadline_ms <= self._now_ms:
                self._async_waiters.remove((deadline_ms, seq, event))
                event.set()
        await event.wait()


class VirtualTiming(Timing):
    """``Timing`` facade over a shared virtual state (used by the Limiter)."""

    def __init__(self, state: _VirtualTimeState):
        self._state = state

    def now(self) -> float:
        return self._state.now_ms() / 1000

    def sleep(self, seconds: float) -> None:
        target_ms = self._state.now_ms() + int(round(seconds * 1000))
        self._state.sync_wait_until(target_ms)

    async def asleep(self, seconds: float) -> None:
        target_ms = self._state.now_ms() + int(round(seconds * 1000))
        await self._state.async_wait_until(target_ms)


class VirtualMonotonicClock(AbstractClock):
    """``AbstractClock`` facade (int ms) + factory for the matching ``Timing``."""

    def __init__(self, start_ms: int = 1_000_000_000_000):
        self._state = _VirtualTimeState(start_ms=start_ms)

    def now(self) -> int:
        return self._state.now_ms()

    def timing(self) -> VirtualTiming:
        """The limiter-side facade for this exact clock (shared state)."""
        return VirtualTiming(self._state)

    def advance(self, delta_ms: int) -> int:
        return self._state.advance(delta_ms)

    def clamp(self, abs_ms: int) -> int:
        return self._state.clamp(abs_ms)

    async def pump(self, steps: int = 25) -> None:
        """Let async tasks resumed by the last ``advance`` run to completion.

        A waiter that is still denied re-registers for a later deadline, so a
        few zero-duration yields are needed per window.
        """
        for _ in range(steps):
            await asyncio.sleep(0)
