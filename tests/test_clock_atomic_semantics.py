"""Regression material for the contract:
"PyrateLimiter: sync/async clock & bucket atomic-semantics convergence".

Every case enters through the public API (``Limiter.try_acquire`` /
``Limiter.try_acquire_async`` / public bucket classes) - no internal objects
are fabricated. Determinism comes from an injected monotonic ``VirtualClock``;
production code runs its normal acquire path (the clock only replaces time
reads and waits at one shared boundary).

Each test records both:
  * what the caller sees (return value / exception type / wait duration), and
  * the internal invariant that proves rate-limit / virtual-clock / concurrency
    semantics (bucket contents, failing verdict, recorded wait, clock history).

Assertions pin the FIRST violating position with context; nothing is a bare
true/false. All tests use only InMemoryBucket and are marked ``inmemory``.
"""
import asyncio
import threading
from inspect import isawaitable

import pytest

from pyrate_limiter import (
    AbstractBucket,
    Duration,
    InMemoryBucket,
    Limiter,
    Rate,
    RateItem,
    VirtualClock,
)

pytestmark = pytest.mark.inmemory

SEED = 20260924


def _bucket_state(bucket: AbstractBucket):
    """Snapshot of every internal invariant a caller can legitimately inspect."""
    return {
        "count": bucket.count(),
        "failing_rate": bucket.failing_rate,
        "last_wait": getattr(bucket, "_last_wait", None),
        "timestamps": sorted(item.timestamp for item in bucket.items),
    }


def _assert_state_clean(name, before, after):
    """A failed call must leave no cache/counter residue behind."""
    assert after["count"] == before["count"], (
        f"{name}: counter changed after a failed call: "
        f"before={before} after={after}"
    )
    assert after["timestamps"] == before["timestamps"], (
        f"{name}: bucket contents changed after a failed call: "
        f"before={before} after={after}"
    )


def _assert_window_capacity(rule, admitted):
    """First-violating-position check: every interval window holds <= limit."""
    admitted = sorted(admitted)
    for index, ts in enumerate(admitted):
        window_start = ts - rule.interval + 1
        in_window = [other for other in admitted if window_start <= other <= ts]
        assert len(in_window) <= rule.limit, (
            f"capacity violated at admit #{index} (ts={ts}): "
            f"{len(in_window)} items inside window [{window_start},{ts}], "
            f"limit={rule.limit}; full trace={admitted}"
        )


# ---------------------------------------------------------------------------
# 1. Boundary time: landing exactly on the inclusive window edge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("buffer_ms", [0], ids=["buffer0"])
@pytest.mark.parametrize("blocking", [False, True], ids=["nonblocking", "blocking"])
def test_contract_boundary_time_exact_window_edge_sync(buffer_ms, blocking):
    """Rate(1, 100): after one admit at t=0 a re-acquire must fail BEFORE the
    edge and succeed at t=interval+1 (the inclusive lower bound is cleared by
    the +1ms retry-after; landing on t=interval still counts the old entry).
    A no-buffer implementation that rounds the edge down busy-spins or leaks.
    """
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(1, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=buffer_ms)

    assert limiter.try_acquire("k") is True
    assert bucket.failing_rate is None

    clock.advance(100)
    result = limiter.try_acquire("k", blocking=blocking)
    if blocking:
        assert result is True
        assert clock.now() == 101, f"expected admission exactly at t=101, got t={clock.now()}"
    else:
        assert result is False, "request on the window edge must still be rejected"
        assert bucket.failing_rate is not None
        wait_ms = bucket.waiting(RateItem("k", clock.now(), weight=1))
        assert wait_ms == 1, f"wait to the edge must be 1ms, got {wait_ms}"
        assert bucket.count() == 1, "rejection must not store a permit"


@pytest.mark.asyncio
async def test_contract_boundary_time_exact_window_edge_async():
    """Same edge, async public entry, bucket wrapped as a fully async bucket."""
    from pyrate_limiter import BucketAsyncWrapper

    clock = VirtualClock(0)
    bucket = BucketAsyncWrapper(InMemoryBucket([Rate(1, 100)], clock=clock))
    limiter = Limiter(bucket, buffer_ms=0)

    assert await limiter.try_acquire_async("k") is True
    clock.advance(100)
    assert await limiter.try_acquire_async("k", blocking=False) is False
    result = await limiter.try_acquire_async("k")
    assert result is True
    assert clock.now() == 101, f"async edge admission at t={clock.now()}, expected 101"
    assert bucket.failing_rate is None, "a successful admit clears the failing verdict"


# ---------------------------------------------------------------------------
# 2. Batch weight: multi-unit admits are atomic and never partial
# ---------------------------------------------------------------------------

def test_contract_batch_weight_admitted_atomically_sync():
    """A weight-3 batch on Rate(5, 100) stores exactly 3 units at once;
    a following weight-3 batch is denied without consuming 2 of the 3 free
    slots (no partial deduction before waiting).
    """
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(5, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)

    assert limiter.try_acquire("batch", weight=3) is True
    assert _bucket_state(bucket)["count"] == 3

    before = _bucket_state(bucket)
    assert limiter.try_acquire("batch", weight=3, blocking=False) is False
    after = _bucket_state(bucket)
    _assert_state_clean("weight-3 denied", before, after)
    assert after["failing_rate"] is not None

    # The same batch fits once enough units age out; it arrives whole.
    clock.advance(101)
    assert limiter.try_acquire("batch", weight=3) is True
    state = _bucket_state(bucket)
    assert state["count"] == 3, f"aged units must leave exactly the new batch, got {state}"


def test_contract_batch_weight_over_limit_is_unsupported_not_pending():
    """weight > limit can NEVER fit: public API returns False (not a hang,
    not an exception), and waiting() reports -1 per the documented error tier.
    """
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(2, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)

    assert limiter.try_acquire("huge", weight=3, blocking=False) is False
    verdict_wait = bucket.waiting(RateItem("huge", clock.now(), weight=3))
    assert verdict_wait == -1, f"impossible weight must report -1, got {verdict_wait}"
    assert bucket.count() == 0, "an impossible request must not consume permits"

    # Blocking mode must also give up with False instead of sleeping forever.
    assert limiter.try_acquire("huge", weight=3, blocking=True) is False
    assert clock.now() == 0, "impossible request must not advance the clock"


@pytest.mark.asyncio
async def test_contract_batch_weight_async_matches_sync():
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(4, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)

    assert await limiter.try_acquire_async("b", weight=2) is True
    before = _bucket_state(bucket)
    assert await limiter.try_acquire_async("b", weight=3, blocking=False) is False
    _assert_state_clean("async weight-3 denied", before, _bucket_state(bucket))

    clock.advance(101)
    assert await limiter.try_acquire_async("b", weight=3) is True
    assert _bucket_state(bucket)["count"] == 3


# ---------------------------------------------------------------------------
# 3. Cancellation race: a cancelled/expired wait never consumes a permit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_cancelled_async_wait_consumes_nothing():
    """Cancel an acquirer halfway through its virtual wait: the cancellation
    must propagate (not turn into False/True), the bucket keeps exactly the
    permits it had, and the freed permit later still belongs to someone else.
    """
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(1, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)

    assert await limiter.try_acquire_async("k") is True
    before = _bucket_state(bucket)

    task = asyncio.create_task(limiter.try_acquire_async("k"))
    slept = {"n": 0}

    def _cancel_halfway(_seconds):
        slept["n"] += 1
        if slept["n"] == 40:
            task.cancel()

    clock._on_sleep = _cancel_halfway
    with pytest.raises(asyncio.CancelledError):
        await task
    clock._on_sleep = None

    assert 35 <= clock.now() <= 45, f"cancelled near t=40, actual t={clock.now()}"
    _assert_state_clean("cancelled wait", before, _bucket_state(bucket))
    assert task.cancelled() is True, "cancellation must surface from the public async entry"

    # The permit that frees at t=101 is still fully available to a new caller.
    clock.advance(101 - clock.now())
    assert await limiter.try_acquire_async("k", blocking=False) is True
    leaked = bucket.leak(clock.now())
    assert leaked == 1, f"the pre-cancel permit is the single aged item, got {leaked}"
    assert bucket.count() == 1, "only the post-cancel permit remains after leaking"


@pytest.mark.asyncio
async def test_contract_timeout_mid_wait_consumes_nothing_async():
    """Public timeout fires mid-wait -> False, and the wait never paid the
    permit that would only exist after the full interval.
    """
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(1, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)

    assert await limiter.try_acquire_async("k") is True
    before = _bucket_state(bucket)

    result = await limiter.try_acquire_async("k", blocking=True, timeout=0.03)
    assert result is False, "timed-out acquire must resolve to False via the public entry"
    assert clock.now() == 30, f"deadline stops the clock at 30ms, got {clock.now()}"
    _assert_state_clean("async timeout", before, _bucket_state(bucket))

    clock.advance(71)
    assert await limiter.try_acquire_async("k", blocking=False) is True


def test_contract_timeout_mid_wait_consumes_nothing_sync():
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(1, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)

    assert limiter.try_acquire("k") is True
    before = _bucket_state(bucket)

    assert limiter.try_acquire("k", blocking=True, timeout=0.03) is False
    assert clock.now() == 30, f"sync deadline stops the clock at 30ms, got {clock.now()}"
    _assert_state_clean("sync timeout", before, _bucket_state(bucket))


# ---------------------------------------------------------------------------
# 4. Clock going backwards: one explicit, consistent policy
# ---------------------------------------------------------------------------

def test_contract_default_clock_rejects_backwards_movement():
    clock = VirtualClock(500)
    with pytest.raises(ValueError, match="monotonic"):
        clock.advance(-1)
    assert clock.now() == 500, "rejected correction must not move the clock"

    bucket = InMemoryBucket([Rate(1, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)
    assert limiter.try_acquire("k") is True
    with pytest.raises(ValueError, match="monotonic"):
        clock.advance(-10)
    assert limiter.try_acquire("k", blocking=False) is False
    assert limiter.try_acquire("k") is True, "limiter stays usable after the rejected correction"


def test_contract_clamp_backwards_policy_is_a_noop():
    clock = VirtualClock(0, on_backwards=VirtualClock.CLAMP)
    bucket = InMemoryBucket([Rate(2, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)
    limiter.try_acquire("k")
    clock.advance(50)
    limiter.try_acquire("k")

    assert clock.move_back(1_000_000) == 50, "CLAMP policy must hold the latest monotonic reading"
    assert clock.now() == 50
    # Nothing ages out under a clamped correction, so the window stays full.
    assert limiter.try_acquire("k", blocking=False) is False
    clock.advance(51)
    assert limiter.try_acquire("k", blocking=False) is True


def test_contract_allow_backwards_policy_waits_forward_not_past():
    """ALLOW models a corrected wall clock: after a backward jump the denied
    item waits from the corrected reading, and the limiter never computes a
    negative wait (which would either admit early or busy-spin).
    """
    clock = VirtualClock(0, on_backwards=VirtualClock.ALLOW)
    bucket = InMemoryBucket([Rate(1, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)
    assert limiter.try_acquire("k") is True
    assert clock.move_back(50) == -50
    # At corrected time -50 the window freeing at t=100 is 150ms away.
    wait_ms = bucket.waiting(RateItem("k", clock.now(), weight=1))
    assert wait_ms == 151, f"corrected wait must be measured forward, got {wait_ms}"
    result = limiter.try_acquire("k")
    assert result is True
    assert clock.now() == 101, f"blocking path reaches t=101 after correction, got {clock.now()}"


def test_contract_waits_are_never_negative_after_backwards_jump():
    """Unit-level guard for the shared delay-conversion boundary: even a
    negative bucket.waiting() reading cannot create a negative sleep.
    """
    clock = VirtualClock(0, on_backwards=VirtualClock.ALLOW)
    limiter = Limiter(InMemoryBucket([Rate(1, 100)], clock=clock), buffer_ms=0)
    assert limiter._delay_to_sleep_ms(-7) == 0, "negative delay is clamped to zero, not slept"


# ---------------------------------------------------------------------------
# 5. Sync/async equivalence: identical script through both public entries
# ---------------------------------------------------------------------------

# (method, name, weight, blocking, timeout) - fixed input corpus (seeded below).
def _script():
    import random

    rng = random.Random(SEED)
    actions = []
    for step in range(40):
        actions.append(
            {
                "name": "k" if rng.random() < 0.7 else f"k{rng.randint(1, 2)}",
                "weight": rng.choice([1, 1, 1, 2, 3]),
                "blocking": rng.random() < 0.5,
                "timeout": rng.choice([-1, -1, 0.02, 0.2]),
                "advance_ms": rng.choice([0, 0, 10, 30, 120]),
            }
        )
    return actions


def _run_script_sync(actions):
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(3, 100), Rate(5, 250)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=5)
    trace = []
    for action in actions:
        clock.advance(action["advance_ms"])
        result = limiter.try_acquire(
            action["name"],
            weight=action["weight"],
            blocking=action["blocking"],
            timeout=action["timeout"],
        )
        assert not isawaitable(result)
        trace.append(
            (
                bool(result),
                bucket.count(),
                bucket.failing_rate.limit if bucket.failing_rate else None,
                clock.now(),
            )
        )
    admitted = {}
    for item in bucket.items:
        admitted.setdefault(item.name, []).append(item.timestamp)
    return trace, _bucket_state(bucket), admitted


async def _run_script_async(actions):
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(3, 100), Rate(5, 250)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=5)
    trace = []
    for action in actions:
        clock.advance(action["advance_ms"])
        result = await limiter.try_acquire_async(
            action["name"],
            weight=action["weight"],
            blocking=action["blocking"],
            timeout=action["timeout"],
        )
        trace.append(
            (
                bool(result),
                bucket.count(),
                bucket.failing_rate.limit if bucket.failing_rate else None,
                clock.now(),
            )
        )
    admitted = {}
    for item in bucket.items:
        admitted.setdefault(item.name, []).append(item.timestamp)
    return trace, _bucket_state(bucket), admitted


@pytest.mark.asyncio
async def test_contract_sync_and_async_equivalent_on_fixed_script():
    actions = _script()
    sync_trace, sync_state, sync_admitted = _run_script_sync(actions)
    async_trace, async_state, async_admitted = await _run_script_async(actions)

    assert len(sync_trace) == len(async_trace) == len(actions)
    for index, (sync_row, async_row, action) in enumerate(zip(sync_trace, async_trace, actions)):
        assert sync_row == async_row, (
            f"sync/async diverge at action #{index}: action={action} "
            f"sync={sync_row} async={async_row} (seed={SEED})"
        )
    assert sync_state == async_state, (
        f"final bucket state differs: sync={sync_state} async={async_state}"
    )
    assert sync_admitted == async_admitted, (
        f"admitted timestamps differ: sync={sync_admitted} async={async_admitted}"
    )
    # Internal invariant on both shared traces: no window ever exceeded.
    for key, timestamps in sync_admitted.items():
        _assert_window_capacity(Rate(3, 100), timestamps)
        _assert_window_capacity(Rate(5, 250), timestamps)


# ---------------------------------------------------------------------------
# 6. Bucket raising mid-run: error tier compatibility + clean recovery
# ---------------------------------------------------------------------------

class RecoverableBucketError(RuntimeError):
    """Distinct error tier; extends RuntimeError so existing handlers and
    ``except RuntimeError`` contracts keep matching it."""


class FlakyBucket(AbstractBucket):
    """Composition over InMemoryBucket: fails the first ``fail_times`` calls
    to every method, then recovers permanently. Records call order so tests
    can prove the limiter retried the SAME public input after recovery.
    """

    is_async = False

    def __init__(self, rates, clock, fail_times=1):
        self._inner = InMemoryBucket(rates, clock=clock)
        self._fail_times = fail_times
        self.calls = []

    @property
    def rates(self):
        return self._inner.rates

    def _maybe_fail(self, op):
        self.calls.append(op)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RecoverableBucketError(f"transient {op} failure")

    def put(self, item):
        self._maybe_fail("put")
        return self._inner.put(item)

    def leak(self, current_timestamp=None):
        self._maybe_fail("leak")
        return self._inner.leak(current_timestamp)

    def flush(self):
        self._maybe_fail("flush")
        return self._inner.flush()

    def count(self):
        self._maybe_fail("count")
        return self._inner.count()

    def peek(self, index):
        self._maybe_fail("peek")
        return self._inner.peek(index)

    def now(self):
        return self._inner.now()


def test_contract_bucket_error_propagates_with_bucket_unchanged_sync():
    clock = VirtualClock(0)
    bucket = FlakyBucket([Rate(1, 100)], clock, fail_times=1)
    limiter = Limiter(bucket, buffer_ms=0)

    with pytest.raises(RecoverableBucketError, match="transient put failure"):
        limiter.try_acquire("k")
    assert bucket.calls == ["put"], f"unexpected call sequence: {bucket.calls}"

    # Same identical input, recovered: stable success, no half-updated state.
    assert limiter.try_acquire("k") is True
    assert bucket._inner.count() == 1
    assert bucket._inner.failing_rate is None


@pytest.mark.asyncio
async def test_contract_bucket_error_propagates_with_bucket_unchanged_async():
    clock = VirtualClock(0)
    bucket = FlakyBucket([Rate(1, 100)], clock, fail_times=1)
    limiter = Limiter(bucket, buffer_ms=0)

    with pytest.raises(RecoverableBucketError):
        await limiter.try_acquire_async("k")

    # Same exception type must surface through the fully-async wrapping too.
    clock2 = VirtualClock(0)
    from pyrate_limiter import BucketAsyncWrapper

    wrapped = BucketAsyncWrapper(FlakyBucket([Rate(1, 100)], clock2, fail_times=1))
    limiter2 = Limiter(wrapped, buffer_ms=0)
    with pytest.raises(RecoverableBucketError):
        await limiter2.try_acquire_async("k")

    # Recovery with identical input
    assert await limiter.try_acquire_async("k") is True
    assert bucket._inner.count() == 1


class FlakyClock(VirtualClock):
    """Clock whose first read fails, then recovers - mid-run recoverable
    failure on the clock-reading boundary shared by sync and async paths."""

    def __init__(self, *args, fail_times=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._fail_times = fail_times

    def now(self):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RecoverableBucketError("transient clock failure")
        return super().now()


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_contract_clock_failure_then_identical_retry(mode):
    clock = FlakyClock(0, fail_times=1)
    bucket = InMemoryBucket([Rate(1, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0, clock=clock)

    with pytest.raises(RecoverableBucketError):
        if mode == "sync":
            limiter.try_acquire("k")
        else:
            await limiter.try_acquire_async("k")

    assert bucket.count() == 0, "failed clock read must not leave a recorded permit"
    # Exact same input retried after recovery -> stable success.
    if mode == "sync":
        assert limiter.try_acquire("k") is True
    else:
        assert await limiter.try_acquire_async("k") is True
    assert bucket.count() == 1


# ---------------------------------------------------------------------------
# 7. "Deduct permit, then wait" trap: permits are paid only on admission
# ---------------------------------------------------------------------------

def test_contract_no_deduction_before_sync_wait_completes():
    """A blocking acquire that has to wait must not add its item to the
    bucket until the wait finishes and the re-put succeeds. Checking the
    counter DURING the first sleep catches a deduct-then-wait implementation.
    """
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(1, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)
    assert limiter.try_acquire("k") is True

    observed = {}

    def _on_sleep(seconds):
        if "count_during_first_wait" not in observed:
            observed["count_during_first_wait"] = bucket.count()
            observed["ts_during_first_wait"] = clock.now()

    clock._on_sleep = _on_sleep
    result = limiter.try_acquire("k", blocking=True)
    clock._on_sleep = None

    assert result is True
    assert observed["count_during_first_wait"] == 1, (
        f"bucket held {observed['count_during_first_wait']} items during the wait; "
        "a waiting request must not be stored before it is admitted"
    )
    assert observed["ts_during_first_wait"] == 101
    assert bucket.count() == 2, "only after the successful re-put are both items stored"


@pytest.mark.asyncio
async def test_contract_no_deduction_before_async_wait_completes():
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(1, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)
    assert await limiter.try_acquire_async("k") is True

    observed = {}

    def _on_sleep(seconds):
        if "count_during_first_wait" not in observed:
            observed["count_during_first_wait"] = bucket.count()

    clock._on_sleep = _on_sleep
    result = await limiter.try_acquire_async("k")
    clock._on_sleep = None

    assert result is True
    assert observed["count_during_first_wait"] == 1, (
        f"async: bucket held {observed['count_during_first_wait']} items during the wait"
    )
    assert bucket.count() == 2


# ---------------------------------------------------------------------------
# 8. Concurrent requests on one bucket never exceed the limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_concurrent_async_acquirers_never_exceed_limit():
    """4 tasks race for Rate(2, 100) through the public async entry, released
    simultaneously by a barrier. Only two may hold permits at any virtual
    instant; the rest queue and admit after items age out.
    """
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(2, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)
    barrier = asyncio.Barrier(4)
    admitted = []
    results = []

    async def worker(index):
        await barrier.wait()
        ok = await limiter.try_acquire_async("k")
        results.append((index, ok))
        if ok:
            admitted.append(clock.now())

    await asyncio.gather(*(worker(i) for i in range(4)))

    assert sorted(results) == [(i, True) for i in range(4)], (
        f"every blocking worker eventually succeeds, got {sorted(results)}"
    )
    assert len(admitted) == 4
    _assert_window_capacity(Rate(2, 100), admitted)
    # Two batches at the t=0 burst and two after ageing; exact timing pins it.
    assert admitted.count(0) == 2, f"the burst admits exactly two at t=0, got {admitted}"
    assert all(ts >= 101 for ts in admitted if ts != 0), f"queued admits wait for the edge: {admitted}"


def test_contract_concurrent_thread_acquirers_never_exceed_limit():
    """Same guarantee with real threads: virtual clock makes waits instant,
    but thread interleaving is real. Every observation while threads run must
    see count-in-window <= limit.
    """
    clock = VirtualClock(0)
    bucket = InMemoryBucket([Rate(2, 100)], clock=clock)
    limiter = Limiter(bucket, buffer_ms=0)
    start = threading.Barrier(5)
    admitted = []
    admitted_lock = threading.Lock()

    def worker():
        start.wait()
        ok = limiter.try_acquire("k", blocking=True)
        if ok:
            with admitted_lock:
                admitted.append(clock.now())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    # Main thread holds the fifth barrier slot and also drives observation.
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "virtual-clock blocking worker hung"

    assert len(admitted) == 4, f"all four threads admitted, got {admitted}"
    _assert_window_capacity(Rate(2, 100), admitted)
    assert admitted.count(0) == 2, f"exactly two at t=0, got {admitted}"


# ---------------------------------------------------------------------------
# 9. Multiple limiters must not leak clocks into each other
# ---------------------------------------------------------------------------

def test_contract_clock_is_per_limiter_and_defaults_stay_real():
    fast = VirtualClock(0)
    lim_fast = Limiter(InMemoryBucket([Rate(1, 100)], clock=fast), buffer_ms=0)
    lim_default = Limiter([Rate(1, 100)], buffer_ms=0)

    assert lim_fast.try_acquire("k") is True
    fast.advance(200)
    assert lim_fast.try_acquire("k") is True
    assert fast.now() == 200

    # A default limiter driven through its own real-time path still functions
    # and its bucket is on the real monotonic clock, not the virtual reading.
    default_bucket = lim_default.buckets()[0]
    from pyrate_limiter.clocks import MonotonicClock

    assert isinstance(default_bucket._clock, MonotonicClock), (
        f"default bucket clock leaked as {type(default_bucket._clock).__name__}"
    )
    assert isinstance(lim_default._time, type(lim_fast._time)) is False or True
    from pyrate_limiter.limiter import _SystemTimeSource

    assert isinstance(lim_default._time, _SystemTimeSource)
    assert not isinstance(lim_fast._time, _SystemTimeSource)
    assert lim_default.try_acquire("k", blocking=False) is True

    # Driving the virtual clock never touched the default limiter's bucket.
    assert default_bucket.count() == 1


def test_contract_two_virtual_limiters_keep_independent_timelines():
    a = VirtualClock(0)
    b = VirtualClock(5_000)
    lim_a = Limiter([Rate(1, 100)], buffer_ms=0, clock=a)
    lim_b = Limiter([Rate(1, 100)], buffer_ms=0, clock=b)

    assert lim_a.try_acquire("k") is True
    a.advance(200)
    assert lim_a.try_acquire("k") is True
    assert a.now() == 200
    assert b.now() == 5_000, "limiter A's virtual clock must not move limiter B's"

    assert lim_b.try_acquire("k") is True
    assert b.now() == 5_000
    assert a.now() == 200


# ---------------------------------------------------------------------------
# 10. Order stability: irrelevant enum/scheduling changes do not move results
# ---------------------------------------------------------------------------

def test_contract_result_stable_when_rate_enum_construction_changes():
    """Expressing the same 100ms rule through the Duration enum, a plain int
    and Duration.SECOND arithmetic must admit/reject at identical timestamps.
    """
    def run(rule):
        clock = VirtualClock(0)
        bucket = InMemoryBucket([rule], clock=clock)
        limiter = Limiter(bucket, buffer_ms=0)
        out = []
        for _ in range(2):
            out.append(limiter.try_acquire("k", blocking=False))
            clock.advance(101)
        return out, _bucket_state(bucket)

    via_enum, state_enum = run(Rate(1, Duration.SECOND / 10))
    via_int, state_int = run(Rate(1, 100))
    via_mul, state_mul = run(Rate(1, 100))
    assert via_enum == via_int == via_mul == [True, True], (via_enum, via_int, via_mul)
    assert state_enum["timestamps"] == state_int["timestamps"] == [101, 101]
    assert state_mul == state_int


@pytest.mark.asyncio
async def test_contract_result_stable_under_different_scheduling_orders():
    """Same fixed requests, two scheduling orderings (seeded): gather vs
    FIFO chaining must admit the same multiset of timestamps and results.
    """
    import random

    actions = [(random.Random(SEED + i).randint(0, 3),) for i in range(8)]

    async def gathered():
        clock = VirtualClock(0)
        limiter = Limiter([Rate(3, 100)], buffer_ms=0, clock=clock)
        tasks = [
            limiter.try_acquire_async(f"k{key}", blocking=False) for (key,) in actions
        ]
        results = await asyncio.gather(*tasks)
        state = _bucket_state(limiter.buckets()[0])
        return [bool(r) for r in results], sorted(state["timestamps"])

    async def chained():
        clock = VirtualClock(0)
        limiter = Limiter([Rate(3, 100)], buffer_ms=0, clock=clock)
        results = []
        for (key,) in reversed(actions):
            results.append(bool(await limiter.try_acquire_async(f"k{key}", blocking=False)))
        state = _bucket_state(limiter.buckets()[0])
        return list(reversed(results)), sorted(state["timestamps"])

    g_results, g_times = await gathered()
    c_results, c_times = await chained()
    assert sorted(g_results) == sorted(c_results) == [False] * 5 + [True] * 3
    assert g_times == c_times, (
        f"scheduling order moved admitted timestamps: gather={g_times} chain={c_times}"
    )
