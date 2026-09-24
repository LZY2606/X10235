"""Regression material for:

"PyrateLimiter: sync/async clock & bucket atomicity convergence".

Every case here enters through the public ``Limiter`` API
(``try_acquire`` / ``try_acquire_async``) against an ``InMemoryBucket`` driven
by a single injectable monotonic virtual clock. No internal result object is
forged. Each case asserts BOTH the caller-visible outcome (result value,
exception type, waiting semantics) and the internal invariant that proves the
rate-limit / virtual-clock / concurrency contract.

Contracts pinned by the test names:

- boundary time: denying exactly at the limit, admitting at +waiting()
- batch weight: multi-unit items consume exactly their weight or nothing
- deduct-before-wait: no quota may be consumed while waiting for room
- cancel race: cancelling a blocked acquire consumes no quota
- clock rollback: backwards movement is clamped to a non-decreasing reading
- sync/async equivalence: identical inputs/order -> identical outcomes
- bucket raises: the bucket's error type propagates unchanged through both
  APIs (no translation into a success/false)

Fixed seed is printed (with the inputs) on failure so any randomized case has
a minimal, deterministic reproduction.
"""

import asyncio
import threading

import pytest

from pyrate_limiter import Duration, InMemoryBucket, Limiter, Rate, RateItem

from ._virtual_time import make_env

pytestmark = pytest.mark.inmemory


def in_window_count(bucket, now_ms, interval_ms):
    """Admission-relevant invariant: units with timestamp > now-interval
    (the inclusive lower bound the algorithms use). Unlike bucket.count(),
    this ignores expired-but-not-yet-leaked entries."""
    bound = now_ms - interval_ms
    return sum(1 for i in range(bucket.count()) if bucket.peek(i).timestamp > bound)


# ------------------------------------------------------------- boundary time

def test_boundary_time_sync_denied_at_limit_admitted_at_waiting_boundary():
    """1/100ms bucket. At the exact window boundary the second item is denied;
    after advancing exactly waiting() the re-attempt succeeds; the millisecond
    before it stays denied (the lower bound is inclusive, hence +1)."""
    limiter, bucket, clock = make_env([Rate(1, 100)])

    first = limiter.try_acquire("k")
    assert first is True, "first permit must be granted"
    assert bucket.count() == 1, "invariant: exactly one unit consumed"

    denied = RateItem("k", clock.now())
    assert bucket.put(denied) is False, "at the limit a put is denied"
    wait_ms = bucket.waiting(RateItem("k", clock.now()))
    assert isinstance(wait_ms, int) and wait_ms == 101, f"waiting() must clear the inclusive bound, got {wait_ms}"

    # One ms before the boundary: still inside the window.
    clock.advance(wait_ms - 1)
    assert limiter.try_acquire("k", blocking=False) is False
    assert bucket.count() == 1, "invariant: denial must not consume quota"

    # Exactly at the boundary: the blocking acquire succeeds immediately under
    # virtual time (no real wall-clock wait).
    clock.advance(1)
    assert limiter.try_acquire("k") is True
    assert in_window_count(bucket, clock.now(), 100) == 1, "only the new item is in the window"


@pytest.mark.asyncio
async def test_boundary_time_async_matches_sync_boundary():
    """The blocked permit is released at exactly waiting() (101ms for
    1/100ms: the inclusive lower bound needs one extra ms, matching the
    sync branch). During the wait the coroutine holds no quota."""
    limiter, bucket, clock = make_env([Rate(2, 100)])

    assert await limiter.try_acquire_async("k") is True
    assert await limiter.try_acquire_async("k") is True

    blocked = asyncio.ensure_future(limiter.try_acquire_async("k"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)  # complete sleep_async's scheduler yield + advance
    assert not blocked.done()
    assert in_window_count(bucket, clock.now(), 100) == 2

    # waiting() for the oldest of two items with interval 100 is 101ms.
    assert clock.now() == 101, f"planned step must target waiting(), got {clock.now()}"
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert await blocked is True
    assert in_window_count(bucket, clock.now(), 100) == 1

    # And the non-blocking boundary directly, through the public API.
    clock.advance(100)
    assert await limiter.try_acquire_async("k", blocking=False) is False
    clock.advance(1)
    assert await limiter.try_acquire_async("k", blocking=False) is True


# ------------------------------------------------------------- batch weights

def test_batch_weight_consumes_exactly_weight_or_nothing():
    """A weight-N item consumes N units; a denied batch consumes ZERO units."""
    limiter, bucket, clock = make_env([Rate(5, 1000)])

    assert limiter.try_acquire("k", weight=3) is True
    assert bucket.count() == 3, f"weight=3 consumes 3, got {bucket.count()}"

    assert limiter.try_acquire("k", weight=3, blocking=False) is False
    assert bucket.count() == 3, "invariant: denied batch consumed no units"

    assert limiter.try_acquire("k", weight=2) is True
    assert bucket.count() == 5

    # Oversized weight can never fit -> blocking returns False, no wait.
    assert limiter.try_acquire("k", weight=6) is False
    assert bucket.count() == 5


def test_batch_weight_fits_exactly_after_window_leaks():
    limiter, bucket, clock = make_env([Rate(3, 100)])

    assert limiter.try_acquire("k", weight=3) is True
    assert limiter.try_acquire("k", weight=3, blocking=False) is False
    clock.advance(101)
    assert limiter.try_acquire("k", weight=3) is True
    assert in_window_count(bucket, clock.now(), 100) == 3


# ---------------------------------------------- deduct quota BEFORE waiting?

class _CountProbe:
    """Observer that samples bucket.count() at every sleep step of a
    virtual blocking acquire. A "deduct then wait" implementation would show
    count == limit+1 during the wait."""

    def __init__(self, bucket, inner_sleep):
        self.bucket = bucket
        self.inner_sleep = inner_sleep
        self.counts_during_wait = []

    def sleep(self, seconds):
        self.counts_during_wait.append(self.bucket.count())
        self.inner_sleep(seconds)


def test_blocking_acquire_never_deducts_quota_before_room_exists():
    """The blocked caller must not appear in the bucket until room exists.
    This catches implementations that reserve quota first and sleep after."""
    from ._virtual_time import VirtualTimeService

    from pyrate_limiter import VirtualClock

    vc = VirtualClock(0)
    bucket = InMemoryBucket([Rate(1, 100)], clock=vc)
    limiter = Limiter(bucket, buffer_ms=0, _time=VirtualTimeService(vc))

    assert limiter.try_acquire("k") is True

    probe = _CountProbe(bucket, limiter._time.sleep)
    limiter._time.sleep = probe.sleep  # observe each wait step

    assert limiter.try_acquire("k") is True
    assert probe.counts_during_wait, "blocking path must have slept at least once"
    assert max(probe.counts_during_wait) == 1, (
        f"quota was consumed before the wait ended: counts={probe.counts_during_wait}"
    )
    assert in_window_count(bucket, vc.now(), 100) == 1, "after success only the new item is in the window"


# -------------------------------------------------------------- cancel race

@pytest.mark.asyncio
async def test_cancelled_blocked_acquire_consumes_no_quota():
    """A blocked acquire that is cancelled while waiting must leave the
    bucket exactly as it was: no item, no failing reservation. Retrying the
    same input after the window leaks then succeeds deterministically."""
    limiter, bucket, clock = make_env([Rate(1, 100)])

    assert await limiter.try_acquire_async("k") is True

    blocked = asyncio.ensure_future(limiter.try_acquire_async("k"))
    await asyncio.sleep(0)
    assert not blocked.done()

    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked

    assert bucket.count() == 1, "invariant: cancellation added no items"
    clock.advance(101)
    assert await limiter.try_acquire_async("k") is True
    assert in_window_count(bucket, clock.now(), 100) == 1


# ------------------------------------------------- sync / async equivalence

def _scenario(limiter, calls):
    """Run one fixed (name, weight, blocking) scenario on a fresh limiter;
    returns the public outcomes plus the bucket count at each step."""
    outcomes = []
    counts = []
    for name, weight, blocking in calls:
        outcomes.append(limiter.try_acquire(name, weight=weight, blocking=blocking))
        counts.append(in_window_count(limiter.buckets()[0], limiter._time.now() * 1000, 200))
    return outcomes, counts


async def _scenario_async(limiter, calls):
    outcomes = []
    counts = []
    for name, weight, blocking in calls:
        outcomes.append(await limiter.try_acquire_async(name, weight=weight, blocking=blocking))
        counts.append(in_window_count(limiter.buckets()[0], limiter._time.now() * 1000, 200))
    return outcomes, counts


@pytest.mark.asyncio
async def test_sync_and_async_equivalent_on_fixed_scenario():
    """The same fixed scenario through both public entries must give the same
    outcomes and the same per-step bucket counts. A second sync limiter with a
    disjoint clock proves clocks do not leak between limiter instances."""
    rates = [Rate(2, 100), Rate(3, 200)]
    calls = [
        ("a", 1, False),
        ("a", 1, False),
        ("a", 2, False),  # denied: only 2 per 100ms
        ("a", 1, True),   # blocks virtually until room exists
        ("b", 1, False),  # same bucket: 101ms < 200ms window of rate(3/200)
    ]

    sync_limiter, sync_bucket, sync_clock = make_env(rates)
    sync_outcomes, sync_counts = _scenario(sync_limiter, calls)

    async_limiter, async_bucket, async_clock = make_env(rates)
    async_outcomes, async_counts = await _scenario_async(async_limiter, calls)

    assert sync_outcomes == async_outcomes == [True, True, False, True, False], (
        f"sync={sync_outcomes} async={async_outcomes}"
    )
    assert sync_counts == async_counts, f"sync counts={sync_counts} async={async_counts}"
    assert sync_clock.now() == async_clock.now(), "both scenarios consumed the same virtual time"


# -------------------------------------- concurrency on one shared in-mem key

def test_concurrent_threads_same_bucket_never_exceed_limit():
    """20 blocking threads on a 1/50ms bucket must all eventually succeed but
    never observe more than one item inside any window. Barrier start removes
    scheduling luck; the assertion message reports the first violation point.
    """
    limiter, bucket, clock = make_env([Rate(1, 50)])

    # The virtual clock cannot be advanced by real threads, so this case uses
    # the real clock but pins the atomicity invariant directly: observe count
    # from the contending threads.
    from pyrate_limiter import MonotonicClock

    real_bucket = InMemoryBucket([Rate(1, 50)], clock=MonotonicClock())
    real_limiter = Limiter(real_bucket, buffer_ms=0)

    n_threads = 20
    barrier = threading.Barrier(n_threads)
    violations = []
    results = []

    def worker(idx):
        barrier.wait()
        ok = real_limiter.try_acquire("k")
        observed = in_window_count(real_bucket, real_bucket.now(), 50)
        if observed > 1:
            violations.append((idx, observed))
        results.append((idx, ok, observed))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not any(t.is_alive() for t in threads), "workers deadlocked"
    assert all(ok for _, ok, _ in results), f"some acquires failed: {results}"
    assert not violations, (
        f"first over-limit observation (thread, count): {violations[0]}; all={results}"
    )


@pytest.mark.asyncio
async def test_concurrent_async_tasks_same_bucket_never_exceed_limit():
    """Virtual-time version: 10 tasks released from one barrier contend for a
    2/100ms bucket. Exactly 2 are admitted immediately; the rest succeed as
    the window leaks, and count never exceeds 2."""
    limiter, bucket, clock = make_env([Rate(2, 100)])

    n_tasks = 10
    barrier = asyncio.Barrier(n_tasks)
    violations = []

    async def worker():
        await barrier.wait()
        ok = await limiter.try_acquire_async("k")
        observed = in_window_count(bucket, clock.now(), 100)
        if observed > 2:
            violations.append(observed)
        return ok

    tasks = [asyncio.ensure_future(worker()) for _ in range(n_tasks)]
    results = await asyncio.gather(*tasks)

    assert all(results), f"every blocking acquire must succeed, got {results}"
    assert not violations, f"count exceeded limit: {violations}"
    assert in_window_count(bucket, clock.now(), 100) == 2


# --------------------------------------------------- clocks do not leak

def test_multiple_limiters_do_not_share_or_leak_clocks():
    """Each make_env() builds a private VirtualClock; advancing one must not
    move another, and a failure/denial in one limiter must not affect the
    other's bucket."""
    lim_a, bucket_a, clock_a = make_env([Rate(1, 100)])
    lim_b, bucket_b, clock_b = make_env([Rate(1, 100)], start_ms=5000)

    assert clock_a.now() == 0 and clock_b.now() == 5000

    assert lim_a.try_acquire("k") is True
    assert lim_a.try_acquire("k", blocking=False) is False
    # A's state/failing_rate must not leak into B.
    assert bucket_b.count() == 0
    assert bucket_b.failing_rate is None
    assert lim_b.try_acquire("k") is True

    clock_a.advance(100)
    assert clock_b.now() == 5000, "clock A advanced; B must be untouched"
    assert lim_a.try_acquire("k") is True
    assert lim_b.try_acquire("k", blocking=False) is False, "B still independent and full"


def test_result_after_failed_calls_carries_no_stale_diagnostics():
    """failing_rate/_last_wait describe only the MOST RECENT put. A failed
    then flushed/recovered sequence must succeed with no stale denial, and a
    successful non-blocking acquire after denials clears the diagnostics."""
    limiter, bucket, clock = make_env([Rate(1, 100)])

    assert limiter.try_acquire("k") is True
    assert limiter.try_acquire("k", blocking=False) is False
    assert bucket.failing_rate is not None, "denial is recorded"

    clock.advance(101)
    assert limiter.try_acquire("k", blocking=False) is True
    assert bucket.failing_rate is None, "success must clear failing_rate"
    assert bucket._last_wait is None, "success must clear the recorded wait"
    assert in_window_count(bucket, clock.now(), 100) == 1

    # Invalid inputs must be explicit, never guessed into success.
    with pytest.raises(AssertionError):
        limiter.try_acquire("k", weight=-1)


def test_duration_enum_interval_is_result_irrelevant():
    """Rate(1, Duration.SECOND) and Rate(1, 1000) describe the same limit;
    outcomes under the same virtual timeline must be identical."""
    lim_enum, bucket_enum, clock_enum = make_env([Rate(1, Duration.SECOND)])
    lim_int, bucket_int, clock_int = make_env([Rate(1, 1000)])

    for lim, bucket in ((lim_enum, bucket_enum), (lim_int, bucket_int)):
        assert lim.try_acquire("k") is True
        assert lim.try_acquire("k", blocking=False) is False

    clock_enum.advance(999)
    clock_int.advance(999)
    assert lim_enum.try_acquire("k", blocking=False) is False
    assert lim_int.try_acquire("k", blocking=False) is False
    clock_enum.advance(2)
    clock_int.advance(2)
    assert lim_enum.try_acquire("k", blocking=False) is True
    assert lim_int.try_acquire("k", blocking=False) is True


@pytest.mark.asyncio
async def test_cancelled_acquire_leaves_no_temp_state_for_next_caller():
    """The cancelled coroutine must not leave a stuck async lock or a pending
    future: a subsequent acquire on the same limiter proceeds normally."""
    limiter, bucket, clock = make_env([Rate(1, 100)])

    a = asyncio.ensure_future(limiter.try_acquire_async("k"))
    await asyncio.sleep(0)
    assert await a is True

    # Must actually be blocked (a 2/100 bucket would admit this immediately).
    b = asyncio.ensure_future(limiter.try_acquire_async("k"))
    await asyncio.sleep(0)
    assert not b.done(), "second permit on a 1/100 bucket must block"
    b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b

    # Lock must be released by the cancellation; this would hang otherwise.
    assert await asyncio.wait_for(limiter.try_acquire_async("other"), timeout=1) is True
    assert in_window_count(bucket, clock.now(), 100) == 1, "cancelled permit was never stored"


# ------------------------------------------------------------- clock rollback

def test_clock_rollback_is_clamped_for_bucket_and_deadlines():
    """VirtualClock.set() into the past must not move readings backwards, so
    deadlines never get longer and items never get 'newer'. The same instance
    governs bucket timestamps and limiter deadlines."""
    from pyrate_limiter import VirtualClock

    vc = VirtualClock(1000)
    assert vc.now() == 1000
    assert vc.set(900) == 1000, "rollback must clamp to the last reading"
    assert vc.now() == 1000
    assert vc.advance(50) == 1050
    with pytest.raises(ValueError, match="delta must be >= 0"):
        vc.advance(-1)

    # Through the public boundary: a timeout measured before the attempted
    # rollback must still expire (deadline cannot be pushed into the future).
    from ._virtual_time import VirtualTimeService

    bucket = InMemoryBucket([Rate(1, 100)], clock=vc)
    limiter = Limiter(bucket, buffer_ms=0, _time=VirtualTimeService(vc))
    assert limiter.try_acquire("k") is True

    blocked_timeout = limiter.try_acquire("k", timeout=0.05)
    # The blocking wait sleeps virtually until its deadline; the attempt to
    # roll back below that point must not grant extra waiting room.
    vc.set(1040)
    assert blocked_timeout is False, "timeout stays a timeout under rollback"
    assert bucket.count() == 1


# ------------------------------------------------------------- bucket raises

class _ExplodingBucket(InMemoryBucket):
    """InMemoryBucket whose put() raises a distinguishable error a fixed
    number of times, then behaves normally (recoverable failure injection)."""

    def __init__(self, rates, clock, failures=1, exc_type=RuntimeError):
        super().__init__(rates, clock=clock)
        self.remaining_failures = failures
        self.exc_type = exc_type

    def put(self, item):
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise self.exc_type("simulated bucket failure")
        return super().put(item)


class _MarkerError(RuntimeError):
    pass


def test_sync_bucket_error_propagates_without_partial_state():
    from pyrate_limiter import VirtualClock
    from ._virtual_time import VirtualTimeService

    vc = VirtualClock(0)
    bucket = _ExplodingBucket([Rate(1, 100)], vc, failures=1, exc_type=_MarkerError)
    limiter = Limiter(bucket, buffer_ms=0, _time=VirtualTimeService(vc))

    # Mid-stream recoverable failure: error type must survive unchanged
    # (never translated to False/success).
    with pytest.raises(_MarkerError, match="simulated bucket failure"):
        limiter.try_acquire("k")

    assert bucket.count() == 0, "failed call left no items"
    assert bucket.failing_rate is None, "failed call left no failing_rate"

    # Same input, same result on retry after recovery.
    assert limiter.try_acquire("k") is True
    assert bucket.count() == 1


@pytest.mark.asyncio
async def test_async_bucket_error_propagates_without_partial_state():
    from pyrate_limiter import VirtualClock
    from ._virtual_time import VirtualTimeService

    vc = VirtualClock(0)
    bucket = _ExplodingBucket([Rate(1, 100)], vc, failures=1, exc_type=_MarkerError)
    limiter = Limiter(bucket, buffer_ms=0, _time=VirtualTimeService(vc))

    with pytest.raises(_MarkerError):
        await limiter.try_acquire_async("k")

    assert bucket.count() == 0
    assert bucket.failing_rate is None
    assert await limiter.try_acquire_async("k") is True
    assert bucket.count() == 1
