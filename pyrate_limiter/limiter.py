"""Limiter class implementation"""

import asyncio
import logging
from contextlib import contextmanager
from functools import wraps
from inspect import isawaitable, iscoroutine, iscoroutinefunction
from threading import RLock, local
from time import monotonic, sleep
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Protocol, Tuple, Union

from .abstracts import AbstractBucket, BucketFactory, Rate, RateItem
from .buckets import InMemoryBucket
from .clocks import AbstractClock, MonotonicClock

logger = logging.getLogger(__name__)

ItemMapping = Callable[[Any], Tuple[str, int]]
DecoratorWrapper = Callable[[Callable[[Any], Any]], Callable[[Any], Any]]

# Sentinel: "no bucket obtained yet" for _acquire_co (None is a valid arg).
_UNSET: Any = object()


class LockLike(Protocol):
    def acquire(self, blocking: bool = ..., timeout: Union[float, int, None] = ...) -> bool: ...
    def release(self) -> None: ...


class SingleBucketFactory(BucketFactory):
    """Single-bucket factory for quick use with Limiter"""

    bucket: AbstractBucket

    def __init__(self, bucket: AbstractBucket, schedule_leak: bool = True):
        """
        Initialize the SingleBucketFactory with a bucket and an optional leak scheduling flag.

        schedule_leak (bool): If True, the factory will schedule periodic leaks for the bucket. Default is True. Disable only if you plan to handle leaking manually.
        """
        self.bucket = bucket

        if schedule_leak:
            self.schedule_leak(bucket)

    def wrap_item(self, name: str, weight: int = 1):
        now = self.bucket.now()

        if isawaitable(now):

            async def wrap_async():
                return RateItem(name, await now, weight=weight)

            return wrap_async()

        # Sync fast path (every blocking-free acquire): no closures allocated.
        return RateItem(name, now, weight=weight)

    def get(self, _: RateItem) -> AbstractBucket:
        return self.bucket

    def owned_buckets(self) -> List[AbstractBucket]:
        """This factory holds ``self.bucket`` for its whole life, so it must be
        closed even when it was never registered for leaking
        (``schedule_leak=False``) or has since been disposed.
        """
        buckets = super().owned_buckets()

        if all(bucket is not self.bucket for bucket in buckets):
            buckets.append(self.bucket)

        return buckets


@contextmanager
def combined_lock(locks: Union[Iterable[LockLike], RLock], blocking: bool, timeout: int | float = -1):
    if not isinstance(locks, Iterable):
        acquired_ok = locks.acquire(blocking=blocking, timeout=timeout)
        if not acquired_ok:
            raise TimeoutError("acquire failed")
        try:
            yield
        finally:
            locks.release()

    else:
        acquired_locks = []
        deadline = monotonic() + timeout if blocking and timeout != -1 else None
        try:
            for lock in locks:
                if not blocking:
                    ok = lock.acquire(False)
                elif timeout == -1:
                    ok = lock.acquire()
                else:
                    assert deadline is not None
                    remaining = max(0.0, deadline - monotonic())
                    ok = lock.acquire(timeout=remaining)

                if not ok:
                    raise TimeoutError("Timeout while acquiring combined lock.")
                acquired_locks.append(lock)
            yield
        finally:
            for lock in reversed(acquired_locks):
                lock.release()


def _plan_delay_step(now_ms: float, deadline_ms: Optional[float], delay_ms: Union[int, float]) -> Tuple[float, bool]:
    """Plan one delay step for `_delay_waiter`.

    Returns ``(seconds_to_sleep, raise_timeout_after_sleep)``. Raises
    ``TimeoutError`` immediately (no sleep) when the deadline has already
    passed. This is the deadline math shared verbatim by the sync and async
    branches of `_delay_waiter`; the caller performs the actual sleep with its
    own primitive (`time.sleep` vs `asyncio.sleep`) and re-raises afterwards.
    All times are milliseconds on the limiter's (possibly injected) clock.
    """
    if deadline_ms is None:
        return delay_ms / 1000, False
    remaining_ms = deadline_ms - now_ms
    if remaining_ms <= 0:
        raise TimeoutError()
    if remaining_ms < delay_ms:
        return remaining_ms / 1000, True
    return delay_ms / 1000, False


class Limiter:
    """This class responsibility is to sum up all underlying logic
    and make working with async/sync functions easily
    """

    bucket_factory: BucketFactory
    lock: Union[RLock, Iterable]
    buffer_ms: int
    _clock: AbstractClock

    # async_lock is thread local, created on first use
    _thread_local: local

    def __init__(
        self,
        argument: Union[BucketFactory, AbstractBucket, Rate, List[Rate]],
        buffer_ms: int = 50,
        clock: Optional[AbstractClock] = None,
    ):
        """Init Limiter using either a single bucket / multiple-bucket factory
        / single rate / rate list.

        Parameters:
            argument (Union[BucketFactory, AbstractBucket, Rate, List[Rate]]): The bucket or rate configuration.
            clock (Optional[AbstractClock]): Clock backing all deadline/timeout
                accounting. Defaults to MonotonicClock. Must be synchronous;
                tests may inject a VirtualClock to drive deadlines manually.
                Actual sleeping always uses the real sync/async sleep
                primitives regardless of the injected clock.
        """

        self.buffer_ms = buffer_ms
        self._clock = clock or MonotonicClock()
        self.bucket_factory = self._init_bucket_factory(argument)
        self.lock = RLock()
        self._thread_local = local()

        if isinstance(argument, AbstractBucket):
            limiter_lock = argument.limiter_lock()
            if limiter_lock is not None:
                self.lock = (limiter_lock, self.lock)

    def buckets(self) -> List[AbstractBucket]:
        """Get list of active buckets"""
        return self.bucket_factory.get_buckets()

    def dispose(self, bucket: Union[int, AbstractBucket]) -> bool:
        """Dispose/Remove a specific bucket,
        using bucket-id or bucket object as param
        """
        return self.bucket_factory.dispose(bucket)

    def _init_bucket_factory(
        self,
        argument: Union[BucketFactory, AbstractBucket, Rate, List[Rate]],
    ) -> BucketFactory:
        if isinstance(argument, Rate):
            argument = [argument]

        if isinstance(argument, list):
            assert len(argument) > 0, "Rates must not be empty"
            assert isinstance(argument[0], Rate), "Not valid rates list"
            rates = argument
            logger.info("Initializing default bucket(InMemoryBucket) with rates: %s", rates)
            argument = InMemoryBucket(rates)

        if isinstance(argument, AbstractBucket):
            argument = SingleBucketFactory(argument)

        assert isinstance(argument, BucketFactory), "Not a valid bucket/bucket-factory"

        return argument

    def _now_ms(self) -> float:
        """The single clock read behind every deadline computation.

        The injected clock must be synchronous: an awaitable reading is
        unsupported and rejected explicitly rather than silently mis-timed.
        """
        now = self._clock.now()

        if isawaitable(now):
            if iscoroutine(now):
                now.close()
            raise RuntimeError("Limiter clock must be synchronous")

        return now

    @staticmethod
    def _validate_acquire_args(blocking: bool, timeout: Union[int, float]) -> None:
        """Entry validation shared by ``try_acquire`` and ``try_acquire_async``."""
        if timeout < 0 and timeout != -1:
            raise ValueError("timeout must be -1 or >= 0")

        if not blocking and timeout != -1:
            raise RuntimeError("Can't set timeout with non-blocking")

    def _delay_to_sleep_ms(self, delay: int) -> Optional[int]:
        """Translate a ``bucket.waiting()`` result into milliseconds to sleep
        before re-attempting the put.

        Returns ``None`` when the item can never fit (``waiting() == -1``: the
        weight exceeds the rate limit). Shared by both branches of
        ``_delay_waiter`` so the ``-1`` / negative / ``+buffer_ms`` handling is
        identical - previously the async branch asserted ``d >= 0`` while the
        sync branch clamped negatives to 0.
        """
        if delay == -1:
            return None
        if delay < 0:
            delay = 0
        return delay + self.buffer_ms

    def _delay_plan(self, item: RateItem, wait_ms: int, deadline: Optional[float]):
        """The blocking-wait schedule shared by the sync and async delay loops.

        A generator driving the acquire-retry protocol: each yield hands the
        caller ``(seconds_to_sleep, raise_timeout_after_sleep)``; the caller
        sleeps with its own primitive (``time.sleep`` vs ``asyncio.sleep``),
        re-puts the item however its path requires, and sends back
        ``(acquired, next_wait_ms)`` with both values already resolved (never
        awaitables). The final outcome arrives as ``StopIteration.value``.
        ``TimeoutError`` propagates out of ``next()``/``send()`` when the
        deadline passes. Centralizing the buffer translation, deadline math
        and timestamp bump here keeps the sync and async drivers line-for-line
        equivalent.
        """
        while True:
            sleep_ms = self._delay_to_sleep_ms(wait_ms)

            if sleep_ms is None:
                return False

            secs, timed_out = _plan_delay_step(self._now_ms(), deadline, sleep_ms)
            item.timestamp += sleep_ms
            acquired, wait_ms = yield secs, timed_out

            if acquired:
                return True

    def _delay_waiter(
        self, bucket: AbstractBucket, item: RateItem, blocking: bool, _force_async: bool = False, deadline: Optional[float] = None
    ) -> Union[bool, Awaitable[bool]]:
        """On `try_acquire` failed, handle delay"""
        assert bucket.failing_rate is not None

        if not blocking:
            return False

        wait = bucket.waiting(item)

        if _force_async or isawaitable(wait):
            return self._delay_wait_async(bucket, item, wait, deadline=deadline)

        assert isinstance(wait, int)
        return self._blocking_retry_sync(bucket, item, wait, blocking=blocking, deadline=deadline)

    async def _delay_wait_async(
        self, bucket: AbstractBucket, item: RateItem, wait: Union[int, Awaitable[int]], deadline: Optional[float] = None
    ) -> bool:
        """Async driver over the shared `_delay_plan` schedule.

        Resolves awaitable ``waiting()``/``put()`` results each round and
        sleeps with ``asyncio.sleep``; the schedule itself (buffer, deadline,
        timestamp bump) comes from the same generator the sync path uses.
        """
        wait = await self._handle_async_result(wait, deadline=deadline)
        assert isinstance(wait, int)

        plan = self._delay_plan(item, wait, deadline)

        try:
            step = next(plan)
        except StopIteration as stop:
            return stop.value

        while True:
            secs, timed_out = step
            await asyncio.sleep(secs)

            if timed_out:
                plan.close()
                raise TimeoutError()

            acquired = await self._handle_async_result(bucket.put(item), deadline=deadline)

            if not acquired:
                wait = await self._handle_async_result(bucket.waiting(item), deadline=deadline)
                assert isinstance(wait, int)

            try:
                step = plan.send((bool(acquired), wait))
            except StopIteration as stop:
                return stop.value

    def handle_bucket_put(
        self, bucket: AbstractBucket, item: RateItem, blocking: bool, _force_async: bool = False, deadline: Optional[float] = None
    ) -> Union[bool, Awaitable[bool]]:
        """Putting item into bucket"""
        acquire = bucket.put(item)

        if isawaitable(acquire):
            return self._wait_after_async_put(bucket, item, acquire, blocking=blocking, deadline=deadline)

        if acquire:
            return True

        return self._delay_waiter(bucket, item, blocking=blocking, _force_async=_force_async, deadline=deadline)

    async def _wait_after_async_put(self, bucket, item, acquire, blocking, deadline=None):
        """Resolve an awaitable ``put`` result, then run the (async) delay loop
        on failure.

        Shared by ``handle_bucket_put``, the async-put branch of
        ``_try_acquire``, and the rare "sync bucket returned an awaitable
        mid-wait" case in ``_blocking_retry_sync``. The delay loop is always
        forced async here because we are already inside a coroutine.
        """
        acquire_result = await self._handle_async_result(acquire, deadline=deadline)
        if acquire_result:
            return True

        result = self._delay_waiter(bucket, item, blocking=blocking, _force_async=True, deadline=deadline)
        return await self._handle_async_result(result, deadline=deadline)

    def _blocking_retry_sync(
        self, bucket: AbstractBucket, item: RateItem, wait_ms: int, blocking: bool, deadline: Optional[float] = None
    ) -> Union[bool, Awaitable[bool]]:
        """Sync blocking wait with the limiter lock RELEASED during the sleep.

        The initial check-then-put already happened under the lock in
        ``_try_acquire``; this loop sleeps WITHOUT the lock, then re-acquires it
        only to re-put and recompute the wait. Because every ``put()`` +
        ``waiting()`` pair runs inside a single lock hold, ``failing_rate`` stays
        consistent; only the already-computed sleep duration spans the unlocked
        window, and the loop re-checks on wake, so a stale duration costs at most
        one extra retry - never correctness. Releasing the lock means a long
        wait on one key no longer serializes acquisitions for other keys (#301).
        The schedule itself comes from `_delay_plan`, the same generator the
        async delay loop drives.
        """
        plan = self._delay_plan(item, wait_ms, deadline)

        try:
            step = next(plan)
        except StopIteration as stop:
            return stop.value

        while True:
            secs, timed_out = step
            sleep(secs)  # lock intentionally NOT held during the sleep

            if timed_out:
                plan.close()
                raise TimeoutError()

            remaining: Union[int, float] = -1 if deadline is None else max(0.0, (deadline - self._now_ms()) / 1000)

            with combined_lock(self.lock, blocking=True, timeout=remaining):
                re_acquire = bucket.put(item)

                if isawaitable(re_acquire):
                    # Pathological: a nominally-sync bucket returned an awaitable
                    # mid-wait. Hand off to the async tail (lock released on return).
                    plan.close()
                    return self._handle_async_result(
                        self._wait_after_async_put(bucket, item, re_acquire, blocking=blocking, deadline=deadline),
                        deadline=deadline,
                    )

                assert isinstance(re_acquire, bool)

                if not re_acquire:
                    next_wait = bucket.waiting(item)
                    assert isinstance(next_wait, int)
                    wait_ms = next_wait

            try:
                step = plan.send((re_acquire, wait_ms))
            except StopIteration as stop:
                return stop.value

    def _get_async_lock(self):
        """Returns thread_local, loop-specific lock"""
        loop = asyncio.get_running_loop()
        try:
            # The async loop *can* change in a given thread
            lock = self._thread_local.async_lock
            if self._thread_local.async_lock_loop is loop:
                return lock
        except AttributeError:
            pass
        lock = asyncio.Lock()
        self._thread_local.async_lock = lock
        self._thread_local.async_lock_loop = loop
        return lock

    def try_acquire(self, name: str = "pyrate", weight: int = 1, blocking: bool = True, timeout: int | float = -1) -> Union[bool, Awaitable[bool]]:
        """
        Attempt to acquire a permit from the limiter.

        Parameters
        ----------
        name : str, default "pyrate"
            The bucket key to acquire from.
        weight : int, default 1
            Number of permits to consume.
        timeout : int | float, default -1
            Maximum time (in seconds) to wait; -1 means wait indefinitely.
        blocking : bool, default True
            If True, block until a permit is available (subject to timeout);
            if False, return immediately.

        Returns
        -------
        bool or Awaitable[bool]
            True if the permit was acquired, False otherwise. Async limiters
            return an awaitable resolving to the same.
        """
        self._validate_acquire_args(blocking, timeout)

        try:
            result = self._try_acquire(name=name, weight=weight, timeout=timeout, blocking=blocking)
        except TimeoutError:
            logger.debug("Acquisition TimeoutError")
            return False

        if not isawaitable(result):
            return result

        return self._await_acquire(result)

    async def _await_acquire(self, result: Awaitable[bool]) -> bool:
        """Resolve an awaitable acquire result, converting TimeoutError to
        False - the public acquire contract: timeouts are reported as False,
        never raised. Shared by ``try_acquire``'s async tail and
        ``try_acquire_async``."""
        try:
            return await self._handle_async_result(result)
        except TimeoutError:
            logger.debug("Acquisition TimeoutError")
            return False

    async def _acquire_async(self, blocking, name, weight, timeout=-1):
        return await self._handle_async_result(self._try_acquire(name, weight, blocking=blocking, timeout=timeout, _force_async=True))

    async def try_acquire_async(self, name: str = "pyrate", weight: int = 1, blocking: bool = True, timeout: int | float = -1) -> bool:
        """
        Attempt to asynchronously acquire a permit from the limiter.

        Parameters
        ----------
        name : str, default "pyrate"
            The bucket key to acquire from.
        weight : int, default 1
            Number of permits to consume.
        blocking : bool, default True
            If True, wait until a permit is available (subject to timeout);
            if False, return immediately.
        timeout : int | float, default -1
            Maximum time (in seconds) to wait; -1 means wait indefinitely.

        Returns
        -------
        bool
            True if the permit was acquired, False otherwise.

        Notes
        -----
        This is the async variant of ``try_acquire``. A top-level, thread-local
        async lock is used to prevent blocking the event loop.
        """

        if weight == 0:
            return True

        self._validate_acquire_args(blocking, timeout)

        async def run():
            lock = self._get_async_lock()
            async with lock:
                # Pass timeout through so the internal deadline governs the wait
                # (mirrors sync try_acquire). This makes timeout=0 a non-waiting
                # attempt instead of asyncio.wait_for(timeout=0) failing before
                # the acquire can even run.
                return await self._acquire_async(blocking=blocking, name=name, weight=weight, timeout=timeout)

        try:
            if timeout > 0:
                # wait_for additionally bounds time spent waiting on the async lock.
                return await asyncio.wait_for(run(), timeout=timeout)
            return await run()
        except (asyncio.TimeoutError, TimeoutError):
            return False

    async def _acquire_co(
        self,
        item,
        bucket: Any = _UNSET,
        *,
        blocking: bool,
        _force_async: bool = False,
        deadline: Optional[float] = None,
    ):
        """Async tail shared by every path where wrap_item()/get()/put() turned
        out to be awaitable.

        Resolves the (maybe-awaitable) ``item``, routes to a bucket if one was
        not already obtained, then drives the put - awaiting at each stage via
        the single normalizer ``_handle_async_result``. ``_handle_async_result``
        is a no-op on already-resolved values, so the same coroutine serves both
        "wrap_item was async" (bucket _UNSET, get() done here) and "wrap_item was
        sync but get() was async" (resolved item + awaitable bucket passed in).
        Replaces the former _handle_async_acquire / _handle_async_bucket pair.
        """
        this_item = await self._handle_async_result(item, deadline=deadline)
        assert isinstance(this_item, RateItem)

        if bucket is _UNSET:
            bucket = self.bucket_factory.get(this_item)
        bucket = await self._handle_async_result(bucket, deadline=deadline)
        assert isinstance(bucket, AbstractBucket), f"Invalid bucket: item: {this_item.name}"

        result = self.handle_bucket_put(bucket, this_item, blocking=blocking, _force_async=_force_async, deadline=deadline)
        return await self._handle_async_result(result, deadline=deadline)

    async def _handle_async_result(self, result, deadline: Optional[float] = None):
        timed_out = False

        def _cancel_if_pending(awaitable: Any) -> None:
            if asyncio.isfuture(awaitable) and not awaitable.done():
                awaitable.cancel()

        try:
            while isawaitable(result):
                if deadline is None:
                    result = await result
                    continue

                remaining = (deadline - self._now_ms()) / 1000
                if remaining <= 0:
                    timed_out = True
                    _cancel_if_pending(result)
                    raise TimeoutError()

                try:
                    result = await asyncio.wait_for(result, timeout=remaining)
                except asyncio.TimeoutError as exc:
                    timed_out = True
                    _cancel_if_pending(result)
                    raise TimeoutError() from exc

            return result
        finally:
            if timed_out:
                _cancel_if_pending(result)

            if iscoroutine(result):
                result.close()

    def _cleanup_awaitable(self, awaitable: Any) -> None:
        if asyncio.isfuture(awaitable):
            awaitable.cancel()
            return

        if iscoroutine(awaitable):
            awaitable.close()
            return

        cancel = getattr(awaitable, "cancel", None)
        if callable(cancel):
            cancel()

        close = getattr(awaitable, "close", None)
        if callable(close):
            close()

    def _try_acquire(
        self,
        name: str,
        weight: int,
        blocking: bool,
        timeout: int | float = -1,
        _force_async: bool = False,
        _allow_async_result: bool = True,
    ) -> Union[bool, Awaitable[bool]]:
        """Try acquiring an item with name & weight
        Return true on success, false on failure
        """

        deadline: Optional[float] = self._now_ms() + timeout * 1000 if timeout != -1 else None

        with combined_lock(self.lock, blocking=blocking, timeout=timeout):
            assert weight >= 0, "item's weight must be >= 0"

            if weight == 0:
                # NOTE: if item is weightless, just let it go through
                # NOTE: this might change in the future
                return True

            item = self.bucket_factory.wrap_item(name, weight)

            if isawaitable(item):
                if not _force_async and not _allow_async_result:
                    self._cleanup_awaitable(item)
                    raise RuntimeError("Can't use async bucket with sync decorator")
                return self._acquire_co(item, blocking=blocking, _force_async=_force_async, deadline=deadline)

            assert isinstance(item, RateItem)

            bucket = self.bucket_factory.get(item)
            if isawaitable(bucket):
                if not _force_async and not _allow_async_result:
                    self._cleanup_awaitable(bucket)
                    raise RuntimeError("Can't use async bucket with sync decorator")
                return self._acquire_co(item, bucket, blocking=blocking, _force_async=_force_async, deadline=deadline)

            assert isinstance(bucket, AbstractBucket), f"Invalid bucket: item: {name}"

            if (
                not _force_async
                and not _allow_async_result
                and self.bucket_factory._leaker
                and id(bucket) in self.bucket_factory._leaker.async_buckets
            ):
                raise RuntimeError("Can't use async bucket with sync decorator")

            acquire = bucket.put(item)

            if isawaitable(acquire):
                if not _force_async and not _allow_async_result:
                    self._cleanup_awaitable(acquire)
                    raise RuntimeError("Can't use async bucket with sync decorator")
                return self._handle_async_result(
                    self._wait_after_async_put(bucket, item, acquire, blocking=blocking, deadline=deadline),
                    deadline=deadline,
                )

            if acquire:
                return True

            if not blocking:
                return False

            if _force_async:
                # Async caller (try_acquire_async) over a sync bucket: keep the
                # async delay loop (asyncio.sleep). The coroutine runs after this
                # `with` exits, so the lock is not held during the wait anyway.
                result = self._delay_waiter(bucket, item, blocking=blocking, _force_async=True, deadline=deadline)
                return self._handle_async_result(result, deadline=deadline)

            # Sync caller: the first put failed and we will block. Capture the
            # wait here (failing_rate is consistent under the lock), then leave
            # the lock so the blocking sleep does not serialize other keys (#301).
            wait_ms = bucket.waiting(item)
            if isawaitable(wait_ms):
                # Defensive parity with the old sync path: a bucket with a sync
                # put() but an async waiting(). Fall back to the async delay loop.
                result = self._delay_waiter(bucket, item, blocking=blocking, _force_async=True, deadline=deadline)
                return self._handle_async_result(result, deadline=deadline)
            assert isinstance(wait_ms, int)

        return self._blocking_retry_sync(bucket, item, wait_ms, blocking=blocking, deadline=deadline)

    def as_decorator(self, *, name="ratelimiter", weight=1):
        def deco(func: Callable[..., Any]) -> Callable[..., Any]:
            if iscoroutinefunction(func):

                @wraps(func)
                async def wrapper(*args, **kwargs):
                    r = await self.try_acquire_async(name=name, weight=weight)
                    while isawaitable(r):
                        r = await r
                    return await func(*args, **kwargs)

                return wrapper
            else:

                @wraps(func)
                def wrapper(*args, **kwargs):
                    try:
                        r = self._try_acquire(name=name, weight=weight, blocking=True, _allow_async_result=False)
                    except TimeoutError:
                        r = False
                    if isawaitable(r):
                        try:
                            self._cleanup_awaitable(r)
                        finally:
                            raise RuntimeError("Can't use async bucket with sync decorator")
                    return func(*args, **kwargs)

                return wrapper

        return deco

    def close(self) -> None:
        self.bucket_factory.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __getstate__(self):
        """Get state for pickling"""
        state = self.__dict__.copy()
        state.pop("lock", None)
        state.pop("_thread_local", None)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling"""
        self.__dict__.update(state)
        self.lock = RLock()
        self._thread_local = local()
