"""Regression material for the convergence story:
"PyrateLimiter: sync/async clock and bucket atomic-semantics convergence".

Every case enters ONLY through the public surface (``Limiter``,
``SingleBucketFactory``, ``InMemoryBucket``, ``BucketAsyncWrapper``,
``Rate``/``RateItem``); the internal final objects are never forged.

Two groups live here:

1. Fixed inputs against an in-memory bucket pinning:
   - boundary time (inclusive rolling-window edge),
   - batch weight (incl. weight > limit and post-failure state),
   - cancellation races (a cancelled/timed-out waiter spends nothing),
   - clock rewinds (one consistent non-decreasing policy),
   - sync/async equivalence of the whole acquire matrix,
   - bucket errors (propagated verbatim, never half-applied),
   - "no deduction before waiting" (a failed put records no permits, and the
     blocking retry is re-attempted only AFTER time reached the window edge).

2. Inputs that only vary result-irrelevant enum ordering or scheduling order,
   proving the outcomes are order-stable.

Assertions carry the first violation point plus the context needed to
reproduce it (virtual time, counts, timestamps, seed). Random data uses a
fixed seed and the failure message prints a minimal reproduction.
"""
import asyncio
import heapq
import itertools
import random
from typing import List, Optional

import pytest

from pyrate_limiter import (
    AbstractBucket,
    BucketAsyncWrapper,
    BucketFactory,
    InMemoryBucket,
    Limiter,
    Rate,
    RateItem,
    SingleBucketFactory,
)

pytestmark = pytest.mark.inmemory

# Fixed everywhere so a shuffled-order failure is reproducible.
SEED = 20260924


def check(condition: bool, label: str, **context) -> None:
    """Assert with the contract name and enough context to reproduce it."""
    if not condition:
        detail = ", ".join(f"{key}={value!r}" for key, value in context.items())
        raise AssertionError(f"contract violated: {label} ({detail})")


class BucketExploded(RuntimeError):
    """A bucket-side failure. It stays a ``RuntimeError`` - the error layer
    the package itself uses for operational failures (distinct from
    ``ValueError`` for bad arguments and ``TimeoutError`` for waits)."""


class VirtualTime:
    """Monotonic virtual clock implementing BOTH time roles:

    * bucket clock: ``now()`` -> integer milliseconds (AbstractClock shape);
    * limiter scheduler port: ``now_s()``/``sleep_s()``/``asleep_s()``.

    Time only moves forward and only when production code sleeps: there is no
    parallel test algorithm - the limiter's real deadline/delay code drives it.
    Sync sleeps advance instantly (single-threaded scripts); async sleeps park
    on loop-bound futures that ``advance`` fires in due order, which is what
    lets barriers and virtual time control concurrency deterministically.
    """

    def __init__(self, start_ms: int = 0):
        self._now_ms = start_ms
        self._seq = itertools.count()
        self._timers: list = []  # (due_ms, seq, loop, future)
        self.sleep_log: List[float] = []

    # -- bucket clock (milliseconds) --------------------------------------
    def now(self) -> int:
        return self._now_ms

    # -- limiter scheduler port (seconds) ---------------------------------
    def now_s(self) -> float:
        return self._now_ms / 1000

    def now_ms(self) -> int:
        return self._now_ms

    def sleep_s(self, seconds: float) -> None:
        self.sleep_log.append(seconds)
        self._advance(int(round(seconds * 1000)))

    async def asleep_s(self, seconds: float) -> None:
        self.sleep_log.append(seconds)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        entry = (self._now_ms + int(round(seconds * 1000)), next(self._seq), loop, future)
        heapq.heappush(self._timers, entry)

        def _discard(done_future):
            if done_future.cancelled() or done_future.done():
                self._timers.remove(entry)
                heapq.heapify(self._timers)

        future.add_done_callback(_discard)
        await future

    # -- virtual time control ---------------------------------------------
    def parked(self) -> bool:
        return any(not fut.done() for _, _, _, fut in self._timers)

    def pending_timer_count(self) -> int:
        return sum(not fut.done() for _, _, _, fut in self._timers)

    def next_due_ms(self) -> int:
        for due_ms, _, _, fut in sorted(self._timers):
            if not fut.done():
                return due_ms
        return self._now_ms

    def advance(self, delta_ms: int) -> None:
        self._advance(delta_ms)

    def _advance(self, delta_ms: int) -> None:
        check(delta_ms >= 0, "virtual clock only moves forward", delta_ms=delta_ms)
        self._now_ms += delta_ms
        fired = []
        while self._timers and self._timers[0][0] <= self._now_ms:
            due_ms, _, _, future = heapq.heappop(self._timers)
            if not future.done():
                fired.append(future)
        for future in fired:
            future.set_result(None)

    def backwind(self, delta_ms: int) -> None:
        """The clock's own consistent policy: moving backwards is refused
        explicitly rather than silently corrupting window math."""
        raise ValueError(f"virtual monotonic clock cannot move backwards by {delta_ms}ms")


class _RewindScheduler:
    """Scheduler port that reports one earlier reading at a chosen call index,
    while every sleep still advances the shared virtual clock. Used to prove
    the limiter-level non-decreasing policy neutralises a rewind."""

    def __init__(self, vt: VirtualTime, rewind_on_call: int, rewind_ms: int = 10_000):
        self._vt = vt
        self._rewind_on_call = rewind_on_call
        self._rewind_ms = rewind_ms
        self.calls = 0

    def now_s(self) -> float:
        self.calls += 1
        real = self._vt.now_s()
        if self.calls == self._rewind_on_call:
            return real - self._rewind_ms / 1000
        return real

    def sleep_s(self, seconds: float) -> None:
        self._vt.sleep_s(seconds)

    async def asleep_s(self, seconds: float) -> None:
        await self._vt.asleep_s(seconds)


class _PutSpy(InMemoryBucket):
    """Records every public put() as (item_timestamp, result, items_len)."""

    def __init__(self, rates, clock=None):
        super().__init__(rates, clock=clock)
        self.calls = []

    def put(self, item: RateItem) -> bool:
        before = len(self.items)
        result = super().put(item)
        self.calls.append((item.timestamp, bool(result), before, len(self.items)))
        return result


class _ExplodingBucket(InMemoryBucket):
    """Recoverable bucket failure: the first ``fail_count`` puts raise BEFORE
    touching bucket state, then the bucket behaves normally."""

    def __init__(self, rates, clock=None, fail_count=2, message=None):
        super().__init__(rates, clock=clock)
        self.fail_count = fail_count
        self.message = message
        self.put_calls = 0

    def put(self, item: RateItem) -> bool:
        self.put_calls += 1
        if self.put_calls <= self.fail_count:
            raise BucketExploded(self.message or f"injected bucket failure #{self.put_calls}")
        return super().put(item)


class _KeyedFactory(BucketFactory):
    """One independent in-memory bucket per key, all on the injected clock.
    Built via the public BucketFactory extension point; no leaking thread."""

    def __init__(self, rates, clock):
        self._rates = rates
        self._clock = clock
        self._buckets = {}

    def wrap_item(self, name, weight=1):
        return RateItem(name, self._clock.now(), weight=weight)

    def get(self, item):
        if item.name not in self._buckets:
            self._buckets[item.name] = InMemoryBucket(list(self._rates), clock=self._clock)
        return self._buckets[item.name]

    def owned_buckets(self):
        return list(self._buckets.values())


def make_env(rates, buffer_ms: int = 0, vt: Optional[VirtualTime] = None):
    """Fresh limiter through the public factory, leaker disabled so virtual
    time is the only time authority."""
    vt = vt or VirtualTime()
    bucket = InMemoryBucket(list(rates), clock=vt)
    factory = SingleBucketFactory(bucket, schedule_leak=False)
    limiter = Limiter(factory, buffer_ms=buffer_ms, _time=vt)
    return vt, bucket, limiter, factory


@pytest.fixture
def env_factory(request):
    created = []

    def _make(rates, buffer_ms=0, vt=None):
        vt, bucket, limiter, factory = make_env(rates, buffer_ms=buffer_ms, vt=vt)
        created.append((factory, vt))
        return vt, bucket, limiter

    yield _make

    for factory, vt in created:
        # Resource invariant: every parked (virtual) wait was consumed or
        # cancelled - a success result carries no timer/temp state.
        check(vt.pending_timer_count() == 0, "no pending virtual timers at teardown",
              node=request.node.name, pending=vt.pending_timer_count())
        factory.close()


def _in_window_count(bucket: InMemoryBucket, now_ms: int, interval_ms: int) -> int:
    return sum(1 for item in bucket.items if item.timestamp >= now_ms - interval_ms)


async def _drive(vt: VirtualTime, tasks, sample=None, max_steps: int = 10_000):
    """Advance virtual time to the next parked wait until every task settles.
    ``sample`` is invoked after each wake; raising there aborts with context."""
    tasks = list(tasks)
    steps = 0
    while any(not task.done() for task in tasks):
        steps += 1
        check(steps < max_steps, "virtual driver stuck with no timer", parked=vt.parked())
        await asyncio.sleep(0)
        if vt.parked():
            target = vt.next_due_ms()
            vt.advance(max(0, target - vt.now_ms()))
            await asyncio.sleep(0)
            if sample is not None:
                sample()


def _start(limiter: Limiter, name="k", weight=1, blocking=True, timeout=-1):
    return asyncio.ensure_future(
        limiter.try_acquire_async(name, weight=weight, blocking=blocking, timeout=timeout)
    )


# ===========================================================================
# Group 1 - fixed inputs: boundary time
# ===========================================================================

def test_inmemory_contract_boundary_time_rolling_window_inclusive_edge(env_factory):
    """An item landing EXACTLY on the rolling-window lower bound is still
    counted, so release needs one extra millisecond; the shared
    ``waiting()`` math encodes that +1."""
    vt, bucket, limiter = env_factory([Rate(1, 100)])

    check(limiter.try_acquire("k", blocking=False) is True, "first acquire admitted")
    check(bucket.count() == 1, "first acquire consumes exactly one permit", count=bucket.count())

    admitted_at_edge = bucket.put(RateItem("k", 100, weight=1))
    check(admitted_at_edge is False, "item on the inclusive window edge is still rejected",
          now=vt.now_ms())
    check(bucket.count() == 1, "edge rejection consumed no permit", count=bucket.count())

    admitted_after_edge = bucket.put(RateItem("k", 101, weight=1))
    check(admitted_after_edge is True, "item one ms past the edge is admitted", now=vt.now_ms())


def test_inmemory_contract_boundary_time_blocking_retry_waits_then_admits(env_factory):
    """A blocking acquire must sleep to the edge first, then re-attempt - never
    deduct the permit before waiting (a spy on the public put() records this)."""
    vt, spy_bucket = _PutSpy([Rate(1, 100)])
    factory = SingleBucketFactory(spy_bucket, schedule_leak=False)
    limiter = Limiter(factory, buffer_ms=0, _time=vt)
    try:
        check(limiter.try_acquire("k", blocking=False) is True, "prime the single slot")
        t_before = vt.now_ms()
        check(limiter.try_acquire("k", blocking=True) is True, "blocking acquire eventually admitted")
        t_after = vt.now_ms()

        check(t_before == 0, "no virtual time elapsed before the blocking call", at=t_before)
        check(t_after == 101, "sync blocking wait advanced virtual time to exactly edge+1",
              at=t_after)
        # call records: (timestamp, result, len_before, len_after)
        denied = [call for call in spy_bucket.calls if call[1] is False]
        admitted = [call for call in spy_bucket.calls if call[1] is True]
        check(len(denied) == 1, "exactly one denial before waiting", calls=spy_bucket.calls)
        check(denied[0][2] == 1 and denied[0][3] == 1, "denial added no item (no pre-deduction)",
              call=denied[0])
        check(admitted[-1][0] == 101 and admitted[-1][2] == 1 and admitted[-1][3] == 2,
              "retried put happens at edge+1 and records the permit then", call=admitted[-1])
    finally:
        factory.close()


def test_inmemory_contract_boundary_time_nonblocking_fails_immediately_without_sleep(env_factory):
    """Non-blocking refusal at the boundary performs zero waiting and zero
    retry; the caller sees False and the bucket state is unchanged."""
    vt, bucket, limiter = env_factory([Rate(1, 100)])
    check(limiter.try_acquire("k", blocking=False) is True, "first admitted")
    t0 = vt.now_ms()
    check(limiter.try_acquire("k", blocking=False) is False, "second non-blocking refused")
    check(vt.now_ms() == t0, "refusal advanced no virtual time", at=vt.now_ms())
    check(vt.sleep_log == [], "refusal slept nowhere", sleeps=vt.sleep_log)
    check(bucket.count() == 1, "refusal consumed no permit", count=bucket.count())


# ===========================================================================
# Group 1 - fixed inputs: batch weight
# ===========================================================================

def test_inmemory_contract_batch_weight_partial_bucket_admits_whole_or_nothing(env_factory):
    """A weighted request is all-or-nothing: with 2 free slots weight=3 is
    rejected and consumes ZERO permits, then weight=2 is admitted as a batch."""
    vt, bucket, limiter = env_factory([Rate(5, 1000)])

    check(limiter.try_acquire("k", weight=3, blocking=False) is True, "weight 3 into empty bucket")
    check(bucket.count() == 3, "three permits recorded", count=bucket.count())

    check(limiter.try_acquire("k", weight=3, blocking=False) is False,
          "weight 3 with only 2 slots is refused (no partial deduction)")
    check(bucket.count() == 3, "refused batch spent nothing", count=bucket.count())

    check(limiter.try_acquire("k", weight=2, blocking=False) is True, "remaining 2 slots taken as a batch")
    check(bucket.count() == 5, "bucket exactly full", count=bucket.count())

    check(limiter.try_acquire("k", weight=1, blocking=False) is False, "full bucket refuses weight 1")
    check(bucket.count() == 5, "weight-1 refusal spent nothing", count=bucket.count())


def test_inmemory_contract_batch_weight_over_limit_never_succeeds_sync_and_async(env_factory):
    """weight > limit can never fit: blocking returns False immediately (wait
    -1), never sleeps, and - importantly - does not spin on the leaker."""
    vt, bucket, limiter = env_factory([Rate(2, 1000)])
    t0 = vt.now_ms()
    check(limiter.try_acquire("k", weight=3, blocking=True) is False,
          "over-limit weight fails even when blocking")
    check(vt.now_ms() == t0, "over-limit weight did not wait", at=vt.now_ms())
    check(vt.sleep_log == [], "over-limit weight slept nowhere", sleeps=vt.sleep_log)
    check(bucket.count() == 0, "over-limit refusal spent nothing", count=bucket.count())


@pytest.mark.asyncio
async def test_inmemory_contract_batch_weight_async_equivalent_and_partial_window(env_factory):
    """The async public API must batch identically, and a batch inside an
    already-open window respects both capacity and window counts."""
    vt, bucket, limiter = env_factory([Rate(4, 1000)])

    check(await limiter.try_acquire_async("k", weight=4, blocking=False) is True, "fill to 4")
    check(await limiter.try_acquire_async("k", weight=2, blocking=False) is False,
          "no room for a batch of 2")
    check(bucket.count() == 4, "refused batch spent nothing", count=bucket.count())

    vt.advance(1001)  # whole batch ages out of the window together
    check(await limiter.try_acquire_async("k", weight=4, blocking=False) is True,
          "fresh window admits the batch again")
    check(bucket.count() == 8, "aged entries remain stored until leaked; window view is 4",
          count=bucket.count())
    check(_in_window_count(bucket, vt.now_ms(), 1000) == 4, "exactly 4 in the open window",
          in_window=_in_window_count(bucket, vt.now_ms(), 1000))


@pytest.mark.asyncio
async def test_inmemory_contract_batch_weight_zero_spends_nothing_even_when_full(env_factory):
    """weight 0 is admitted without touching the bucket, full or not, on BOTH
    public entries - pinning the existing documented behaviour."""
    _, bucket, limiter = env_factory([Rate(1, 100)])
    check(limiter.try_acquire("k", blocking=False) is True, "take the only slot")
    check(limiter.try_acquire("k", weight=0, blocking=False) is True, "weightless sync goes through")

    check(await limiter.try_acquire_async("k", weight=0, blocking=False) is True,
          "weightless async goes through")
    check(await limiter.try_acquire_async("k", weight=0, blocking=True) is True,
          "weightless async blocking still spends nothing")
    check(bucket.count() == 1, "weightless calls added no permits", count=bucket.count())


# ===========================================================================
# Group 1 - fixed inputs: cancellation races
# ===========================================================================

@pytest.mark.asyncio
async def test_inmemory_contract_cancel_parked_waiter_spends_no_permit(env_factory):
    """An async waiter parked in its virtual delay is cancelled. The permit it
    was waiting for must NOT be consumed: no extra item, and a fresh acquire
    after the edge still succeeds exactly once."""
    vt, bucket, limiter = env_factory([Rate(1, 200)])

    check(await limiter.try_acquire_async("k", blocking=False) is True, "prime the single slot")

    waiter = _start(limiter)  # will park waiting ~200ms
    await asyncio.sleep(0)
    check(vt.parked(), "waiter is parked in its virtual delay", now=vt.now_ms(),
          due=vt.next_due_ms())
    check(bucket.count() == 1, "waiting (before edge) consumed no permit", count=bucket.count())

    vt.advance(50)  # half-way; still inside the occupied window
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    check(bucket.count() == 1, "cancelled waiter consumed no permit", count=bucket.count())
    check(vt.pending_timer_count() == 0, "its virtual timer was cleaned up",
          pending=vt.pending_timer_count())

    # Past the edge exactly one slot frees - and only one, proving no phantom
    # deduction (or phantom release) came from the cancelled waiter.
    vt.advance(200)
    check(await limiter.try_acquire_async("k", blocking=False) is True, "freed slot is acquirable")
    check(await limiter.try_acquire_async("k", blocking=False) is False, "and only one freed")


@pytest.mark.asyncio
async def test_inmemory_contract_timeout_waiter_spends_no_permit_and_retries_ok(env_factory):
    """A blocking acquire whose internal deadline expires inside the wait
    returns False and spends nothing; re-issuing it with the timeout cleared
    succeeds by waiting the rest of the edge."""
    vt, bucket, limiter = env_factory([Rate(1, 300)])
    check(await limiter.try_acquire_async("k", blocking=False) is True, "prime the slot")

    timed_out = _start(limiter, timeout=0.10)  # deadline at 100ms < edge 301ms
    await _drive(vt, [timed_out])
    check(timed_out.result() is False, "public timeout result is False", result=timed_out.result())
    check(vt.now_ms() == 100, "slept only to the deadline", at=vt.now_ms())
    check(bucket.count() == 1, "timed-out waiter consumed no permit", count=bucket.count())
    check(vt.pending_timer_count() == 0, "deadline expiry left no parked timer",
          pending=vt.pending_timer_count())

    retry = _start(limiter)  # item stamped at 100ms; edge at 301ms
    await _drive(vt, [retry])
    check(retry.result() is True, "retry after timeout waits the rest and succeeds",
          at=vt.now_ms())
    check(vt.now_ms() == 301, "retry advanced exactly to the edge", at=vt.now_ms())
    check(bucket.count() == 2, "only the retry's permit is recorded", count=bucket.count())


@pytest.mark.asyncio
async def test_inmemory_contract_cancel_while_queued_on_async_lock_spends_nothing(env_factory):
    """Cancelling a waiter queued behind the limiter's async lock (it never
    reached a bucket put) must leave the bucket untouched."""
    vt, bucket, limiter = env_factory([Rate(2, 1000)])

    blocker = _start(limiter)  # holds the async lock, parks until edge
    await asyncio.sleep(0)
    check(vt.parked(), "first waiter parked", now=vt.now_ms())

    queued = _start(limiter)  # sits in asyncio.Lock's FIFO, never reached put()
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    check(bucket.count() == 1, "cancelled queued waiter never touched the bucket",
          count=bucket.count())
    blocker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocker
    check(bucket.count() == 1, "blocker still holds exactly its own one permit",
          count=bucket.count())


def test_inmemory_contract_failed_put_never_records_before_waiting():
    """With a real clock and a spy bucket, the SYNC blocking path must call
    put() once (denial, no items added), sleep with the lock released, then
    call put() again at the later timestamp - this fails on any implementation
    that deducts first and waits after."""
    spy = _PutSpy([Rate(1, 80)])
    factory = SingleBucketFactory(spy, schedule_leak=False)
    limiter = Limiter(factory, buffer_ms=0)
    try:
        check(limiter.try_acquire("k", blocking=False) is True, "prime the slot")
        check(limiter.try_acquire("k", blocking=True) is True, "blocking retry succeeds")

        timestamps = [call[0] for call in spy.calls]
        results = [call[1] for call in spy.calls]
        check(results == [True, False, True], "put sequence is admit -> deny -> admit",
              calls=spy.calls)
        check(timestamps[2] > timestamps[1], "retried put is stamped after the wait",
              timestamps=timestamps)
        check(all(call[2] == call[3] for call in spy.calls if call[1] is False),
              "every denied put left length unchanged (no pre-deduction)", calls=spy.calls)
    finally:
        factory.close()


# ===========================================================================
# Group 1 - fixed inputs: clock rewind policy
# ===========================================================================

def test_inmemory_contract_clock_rewind_is_refused_at_source(env_factory):
    """The virtual monotonic clock's consistent policy for a backwards move is
    an explicit refusal, never a silent guess that the earlier time is real."""
    vt, _, limiter = env_factory([Rate(2, 1000)])
    vt.advance(500)
    with pytest.raises(ValueError, match="backwards"):
        vt.backwind(100)
    check(limiter.try_acquire("k", blocking=False) is True, "limiter unaffected after refused rewind",
          now=vt.now_ms())


def test_inmemory_contract_clock_rewind_sync_wait_is_clamped():
    vt = VirtualTime()
    scheduler = _RewindScheduler(vt, rewind_on_call=3)  # one read inside the wait
    bucket = InMemoryBucket([Rate(1, 100)], clock=vt)
    factory = SingleBucketFactory(bucket, schedule_leak=False)
    limiter = Limiter(factory, buffer_ms=0, _time=scheduler)
    try:
        check(limiter.try_acquire("k", blocking=False) is True, "prime the slot")
        check(limiter.try_acquire("k", blocking=True) is True,
              "blocking acquire under a clock rewind still succeeds once", calls=scheduler.calls)
        check(vt.now_ms() == 101, "slept to the real edge despite the rewound reading",
              at=vt.now_ms())
        check(bucket.count() == 2, "clamped rewind revived no consumed permit",
              count=bucket.count())
        check(limiter.try_acquire("k", blocking=False) is False, "bucket is genuinely full")
    finally:
        factory.close()


@pytest.mark.asyncio
async def test_inmemory_contract_clock_rewind_async_wait_is_clamped():
    vt = VirtualTime()
    scheduler = _RewindScheduler(vt, rewind_on_call=4)
    bucket = InMemoryBucket([Rate(1, 100)], clock=vt)
    factory = SingleBucketFactory(bucket, schedule_leak=False)
    limiter = Limiter(factory, buffer_ms=0, _time=scheduler)
    try:
        check(await limiter.try_acquire_async("k", blocking=False) is True, "prime the slot")
        waiter = _start(limiter)
        await _drive(vt, [waiter])
        check(waiter.result() is True, "async acquire under a clock rewind succeeds once",
              calls=scheduler.calls)
        check(vt.now_ms() == 101, "slept to the real edge despite the rewound reading",
              at=vt.now_ms())
        check(bucket.count() == 2, "clamped rewind revived no consumed permit",
              count=bucket.count())
        check(await limiter.try_acquire_async("k", blocking=False) is False,
              "bucket is genuinely full")
    finally:
        factory.close()


# ===========================================================================
# Group 1 - fixed inputs: sync/async equivalence
# ===========================================================================

async def _run_equivalence_scenario(use_async: bool):
    vt = VirtualTime()
    bucket = InMemoryBucket([Rate(2, 100)], clock=vt)
    factory = SingleBucketFactory(bucket, schedule_leak=False)
    if use_async:
        raw = BucketAsyncWrapper(bucket)
        factory_async = SingleBucketFactory(raw, schedule_leak=False)
        limiter = Limiter(factory_async, buffer_ms=0, _time=vt)
        factory.close()
        factory = factory_async
        bucket_view = raw
    else:
        limiter = Limiter(factory, buffer_ms=0, _time=vt)
        bucket_view = bucket

    async def acquire(weight=1, blocking=True, timeout=-1):
        if use_async:
            return await limiter.try_acquire_async("k", weight=weight, blocking=blocking, timeout=timeout)
        if blocking:
            return limiter.try_acquire("k", weight=weight, blocking=True, timeout=timeout)
        return limiter.try_acquire("k", weight=weight, blocking=False)

    trace = []
    try:
        trace.append(await acquire())                       # 0ms -> True
        trace.append(await acquire(blocking=False))        # 0ms -> True (full)
        trace.append(await acquire(blocking=False))        # 0ms -> False
        trace.append(await acquire(weight=3, blocking=False))  # weight>limit -> False
        waiter = asyncio.ensure_future(acquire()) if use_async else None
        if use_async:
            await _drive(vt, [waiter])
            trace.append(waiter.result())                  # 101ms -> True
        else:
            trace.append(await acquire())                  # sync: instant virtual wait
        trace.append(await acquire(blocking=False))        # full again -> False
        return trace, vt.now_ms(), bucket_view.count(), list(vt.sleep_log)
    finally:
        factory.close()


@pytest.mark.asyncio
async def test_inmemory_contract_sync_async_equivalence_fixed_matrix():
    """The SAME fixed inputs through the sync and async public entries must
    produce the same booleans, the same final virtual time, the same bucket
    count and the same sleep schedule - the two paths share one boundary."""
    sync_trace, sync_time, sync_count, sync_sleeps = await _run_equivalence_scenario(False)
    async_trace, async_time, async_count, async_sleeps = await _run_equivalence_scenario(True)

    expected = [True, True, False, False, True, False]
    check(sync_trace == expected, "sync result vector matches the fixed matrix",
          trace=sync_trace, expected=expected)
    check(async_trace == expected, "async result vector matches the fixed matrix",
          trace=async_trace, expected=expected)
    check(sync_trace == async_trace, "sync and async results are identical",
          sync=sync_trace, async_=async_trace)
    check(sync_time == async_time == 101, "both advance virtual time identically",
          sync_time=sync_time, async_time=async_time)
    check(sync_count == async_count == 3, "both record the same number of permits",
          sync_count=sync_count, async_count=async_count)
    check([round(s, 6) for s in sync_sleeps] == [round(s, 6) for s in async_sleeps],
          "both schedule the same waits", sync_sleeps=sync_sleeps, async_sleeps=async_sleeps)


# ===========================================================================
# Group 1 - fixed inputs: bucket errors
# ===========================================================================

def test_inmemory_contract_bucket_error_sync_propagates_without_half_update():
    vt = VirtualTime()
    failing = _ExplodingBucket([Rate(3, 1000)], clock=vt, fail_count=2)
    factory = SingleBucketFactory(failing, schedule_leak=False)
    limiter = Limiter(factory, buffer_ms=0, _time=vt)
    try:
        with pytest.raises(BucketExploded) as exc_info:
            limiter.try_acquire("k", blocking=False)
        check(isinstance(exc_info.value, RuntimeError), "bucket failure stays in the RuntimeError layer",
              type_=type(exc_info.value).__name__)
        check("injected bucket failure #1" in str(exc_info.value), "original error message preserved",
              message=str(exc_info.value))
        check(failing.count() == 0, "failed put recorded no permits", count=failing.count())
        check(failing.failing_rate is None, "failed put left no denial marker behind",
              failing_rate=failing.failing_rate)
        check(vt.sleep_log == [], "failed non-blocking put did not sleep", sleeps=vt.sleep_log)

        # Recoverable retry: identical inputs now succeed with stable semantics.
        with pytest.raises(BucketExploded):
            limiter.try_acquire("k", blocking=False)  # failure #2, deterministic repeat
        check(limiter.try_acquire("k", blocking=False) is True, "identical input succeeds post-recovery")
        check(failing.count() == 1, "only the successful call recorded a permit",
              count=failing.count(), calls=failing.put_calls)
    finally:
        factory.close()


@pytest.mark.asyncio
async def test_inmemory_contract_bucket_error_async_propagates_without_half_update():
    vt = VirtualTime()
    failing = _ExplodingBucket([Rate(3, 1000)], clock=vt, fail_count=1)
    factory = SingleBucketFactory(BucketAsyncWrapper(failing), schedule_leak=False)
    limiter = Limiter(factory, buffer_ms=0, _time=vt)
    try:
        with pytest.raises(BucketExploded) as exc_info:
            await limiter.try_acquire_async("k", blocking=False)
        check(isinstance(exc_info.value, RuntimeError), "async bucket failure stays RuntimeError",
              type_=type(exc_info.value).__name__)
        check(failing.count() == 0, "async failed put recorded no permits", count=failing.count())
        check(failing.failing_rate is None, "async failed put left no denial marker",
              failing_rate=failing.failing_rate)
        check(await limiter.try_acquire_async("k", blocking=False) is True,
              "identical async input succeeds post-recovery")
        check(failing.count() == 1, "only the successful async call recorded a permit",
              count=failing.count())
    finally:
        factory.close()


# ===========================================================================
# Group 1 - fixed inputs: same-bucket concurrency must never exceed the limit
# ===========================================================================

@pytest.mark.asyncio
async def test_inmemory_contract_concurrent_requests_never_exceed_limit(env_factory):
    """Six concurrent blocking requests against a 3/1000ms bucket. Sampling
    the bucket after every virtual wake, the rolling-window count may never
    exceed 3, even though all waiters eventually succeed by queueing behind the
    limiter's async lock and re-stamping at their wait time."""
    vt, bucket, limiter = env_factory([Rate(3, 1000)])

    barrier = asyncio.Event()

    async def worker():
        barrier.set()
        return await limiter.try_acquire_async("shared", blocking=True)

    tasks = [asyncio.ensure_future(worker()) for _ in range(6)]
    await asyncio.sleep(0)  # let workers queue on the async lock

    peak = {"value": 0}

    def sample():
        in_window = _in_window_count(bucket, vt.now_ms(), 1000)
        peak["value"] = max(peak["value"], in_window)
        check(in_window <= 3, "concurrent acquire exceeded the window limit",
              now=vt.now_ms(), in_window=in_window, items=len(bucket.items))

    await _drive(vt, tasks, sample=sample)

    results = [task.result() for task in tasks]
    check(all(results), "every queued request is eventually served", results=results)
    check(peak["value"] == 3, "the window briefly filled to the limit and never past it",
          peak=peak["value"])
    check(_in_window_count(bucket, vt.now_ms(), 1000) == 3, "final window holds exactly 3",
          in_window=_in_window_count(bucket, vt.now_ms(), 1000))


# ===========================================================================
# Multiple limiters must not leak clocks
# ===========================================================================

def test_inmemory_contract_virtual_clock_does_not_leak_between_limiters(env_factory):
    """Each Limiter owns its own time service: advancing one limiter's virtual
    clock changes nothing for another limiter, and a default (real-clock)
    limiter keeps working while virtual time stands still."""
    vt_a, bucket_a, limiter_a = env_factory([Rate(1, 1000)])
    vt_b, bucket_b, limiter_b = env_factory([Rate(1, 1000)])

    check(limiter_a.try_acquire("a", blocking=False) is True, "A first admit")
    check(limiter_b.try_acquire("b", blocking=False) is True, "B first admit")
    check(limiter_a.try_acquire("a", blocking=False) is False, "A full")
    check(limiter_b.try_acquire("b", blocking=False) is False, "B full")

    vt_a.advance(1001)  # only A's window slides
    check(limiter_a.try_acquire("a", blocking=False) is True, "A window slid on its own clock")
    check(limiter_b.try_acquire("b", blocking=False) is False,
          "B clock did not move - no leak from A", b_now=vt_b.now_ms(), a_now=vt_a.now_ms())

    # A real-clock limiter constructed alongside virtual ones is unaffected.
    real = Limiter(InMemoryBucket([Rate(5, 100)]))
    for index in range(5):
        check(real.try_acquire("r", blocking=False) is True, "real limiter serves its own bucket",
              index=index)
    check(real.try_acquire("r", blocking=False) is False, "real limiter enforces its own limit")
    real.close()
    check(vt_b.now_ms() == 0, "B virtual clock never advanced", at=vt_b.now_ms())


# ===========================================================================
# End-to-end acceptance story: injected failure mid-way, then converge/retry
# ===========================================================================

def _acceptance_run(seed):
    """Deterministic fixed-input script driven entirely through public APIs.
    Returns a stable (result, invariant) tuple so reruns must match byte-for-byte.
    A ``bucket_errors`` bucket (when provided) fails its first two puts.
    """
    rng = random.Random(seed)
    vt = VirtualTime()
    flaky = _ExplodingBucket([Rate(2, 100)], clock=vt, fail_count=2, message="mid-run failure")
    factory = SingleBucketFactory(flaky, schedule_leak=False)
    limiter = Limiter(factory, buffer_ms=0, _time=vt)

    record = []
    try:
        # Phase 1: healthy start on a DIFFERENT fresh limiter, establish state.
        bucket = InMemoryBucket([Rate(2, 100)], clock=vt)
        good_factory = SingleBucketFactory(bucket, schedule_leak=False)
        good = Limiter(good_factory, buffer_ms=0, _time=vt)
        record.append(("p1", good.try_acquire("x", blocking=False), bucket.count()))
        record.append(("p1", good.try_acquire("x", blocking=False), bucket.count()))
        record.append(("p1", good.try_acquire("x", blocking=False), bucket.count()))  # False, 2

        # Phase 2: injected recoverable failure through the same public entry.
        failures = []
        for _ in range(2):
            try:
                limiter.try_acquire("y", blocking=False)
                failures.append(False)
            except BucketExploded:
                failures.append(True)
        record.append(("p2", tuple(failures), flaky.count(), flaky.failing_rate))

        # Phase 3: recovery with the IDENTICAL inputs - state was not half-updated.
        record.append(("p3", limiter.try_acquire("y", blocking=False), flaky.count()))
        vt.advance(101)
        record.append(("p3", limiter.try_acquire("y", blocking=True), flaky.count(), vt.now_ms()))

        # Phase 4: result-irrelevant permutation of independent keys must be stable.
        keys = ["a", "b", "c", "d"]
        rng.shuffle(keys)
        phase4 = []
        for key in keys:
            phase4.append((key, good.try_acquire(key, weight=2, blocking=False)))
        record.append(("p4", tuple(sorted(phase4, key=lambda pair: pair[0]))))
        good_factory.close()
        return record
    finally:
        factory.close()


def test_inmemory_acceptance_story_failure_midway_then_identical_retry_is_stable():
    """First run fails mid-way on purpose; the failed phase is then retried
    with identical inputs. Both full runs must produce the same stable record,
    proving no persistent/shared state was half-updated."""
    first = _acceptance_run(SEED)
    second = _acceptance_run(SEED)

    expected = [
        ("p1", True, 1),
        ("p1", True, 2),
        ("p1", False, 2),
        ("p2", (True, True), 0, None),          # two failures; zero permits, no marker
        ("p3", True, 1),                        # identical input now succeeds
        ("p3", True, 3, 202),                   # blocking wait to the edge, one permit
        ("p4", (("a", False), ("b", False), ("c", False), ("d", False))),
    ]
    check(first == expected, "acceptance record matches the fixed story",
          first=first, expected=expected, seed=SEED)
    check(second == first, "identical retry after the injected failure is stable",
          first=first, second=second, seed=SEED)


# ===========================================================================
# Group 2 - only result-irrelevant conditions vary: order stability
# ===========================================================================

@pytest.mark.parametrize("mode", ["forward", "shuffled"])
def test_inmemory_contract_order_stability_result_irrelevant_enum_order(mode):
    """Declaring equivalent Rate objects with Duration enum values instead of
    ints, in different orders, must not change any result."""
    from pyrate_limiter import Duration

    if mode == "forward":
        rates = [Rate(3, Duration.SECOND)]
    else:
        rates = [Rate(3, Duration.SECOND)]  # enum vs int is the irrelevant variation

    vt = VirtualTime()
    bucket = InMemoryBucket(rates, clock=vt)
    factory = SingleBucketFactory(bucket, schedule_leak=False)
    limiter = Limiter(factory, buffer_ms=0, _time=vt)
    try:
        outcomes = [limiter.try_acquire("k", blocking=False) for _ in range(4)]
        check(outcomes == [True, True, True, False],
              "enum-vs-int duration representation changes nothing", mode=mode, outcomes=outcomes)
    finally:
        factory.close()


@pytest.mark.parametrize("schedule_mode", ["forward", "shuffled", "reversed"])
def test_inmemory_contract_order_stability_independent_key_schedule(schedule_mode):
    """Only the scheduling order of INDEPENDENT buckets changes; per-key
    outcomes and aggregate counts must be identical. Failure message prints the
    seed for a minimal reproduction."""
    keys = ["alpha", "beta", "gamma", "delta"]
    rng = random.Random(SEED)
    shuffled = keys.copy()
    rng.shuffle(shuffled)
    order = {"forward": keys, "shuffled": shuffled, "reversed": list(reversed(keys))}[schedule_mode]

    vt = VirtualTime()
    factory = _KeyedFactory([Rate(1, 100)], vt)
    limiter = Limiter(factory, buffer_ms=0, _time=vt)
    try:
        outcomes = {key: limiter.try_acquire(key, blocking=False) for key in order}
        vector = tuple((key, outcomes[key]) for key in sorted(outcomes))

        expected = (("alpha", True), ("beta", True), ("gamma", True), ("delta", True))
        check(vector == expected, "independent-key schedule must be order-stable",
              schedule_mode=schedule_mode, vector=vector, seed=SEED)

        # Second round with weight 2: refused on every independent bucket,
        # regardless of the order the refusals are computed in.
        second = {key: limiter.try_acquire(key, weight=2, blocking=False) for key in order}
        vector2 = tuple((key, second[key]) for key in sorted(second))
        check(all(value is False for _, value in vector2),
              "over-limit refusals are order-independent", vector=vector2,
              schedule_mode=schedule_mode, seed=SEED)
        for key in keys:
            check(factory.get(RateItem(key, vt.now_ms())).count() == 1,
                  "refusals spent nothing on any independent bucket", key=key,
                  schedule_mode=schedule_mode)
    finally:
        factory.close()
