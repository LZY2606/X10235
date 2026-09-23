"""Duh....
"""
from asyncio import sleep
from concurrent.futures import ThreadPoolExecutor
from inspect import isawaitable
from time import time
from typing import List
from typing import Tuple

from .conftest import logger
from pyrate_limiter import AbstractBucket
from pyrate_limiter import AbstractClock
from pyrate_limiter import Limiter
from pyrate_limiter import RateItem


class VirtualClock:
    """Deterministic monotonic virtual clock for tests.

    Exposes the SAME virtual timeline in milliseconds (bucket view, via
    ``VirtualMonotonicClock``) and in seconds (``Limiter`` deadline view,
    via ``now_s``), so a bucket and its limiter can never disagree about
    "now". Tests drive time explicitly with ``advance``/``regress``.
    """

    def __init__(self, start_ms: int = 1_000_000):
        self._ms = start_ms

    def now_ms(self) -> int:
        return self._ms

    def now_s(self) -> float:
        return self._ms / 1000

    def advance(self, ms: int) -> None:
        assert ms >= 0, "advance() only moves time forward"
        self._ms += ms

    def regress(self, ms: int) -> None:
        assert ms >= 0, "regress() only moves time backward"
        self._ms -= ms


class VirtualMonotonicClock(AbstractClock):
    """AbstractClock adapter exposing a VirtualClock to a bucket.

    Injected through the public ``InMemoryBucket(rates, clock=...)`` entry
    point; it shares no code path with production clocks beyond the
    ``AbstractClock.now()`` interface itself.
    """

    def __init__(self, clock: VirtualClock):
        self._clock = clock

    def now(self) -> int:
        return self._clock.now_ms()


async def inspect_bucket_items(bucket: AbstractBucket, expected_item_count: int):
    """Inspect items in the bucket
    - Assert number of item == expected-item-count
    - Assert that items are ordered by timestamps, from latest to earliest
    """
    collected_items = []

    for idx in range(expected_item_count):
        item = bucket.peek(idx)

        if isawaitable(item):
            item = await item

        assert isinstance(item, RateItem)
        collected_items.append(item)

    item_names = [item.name for item in collected_items]

    for i in range(1, expected_item_count):
        item = collected_items[i]
        prev_item = collected_items[i - 1]
        assert item.timestamp <= prev_item.timestamp

    return item_names


async def concurrent_acquire(limiter: Limiter, items: List[str], blocking: bool = True):
    with ThreadPoolExecutor() as executor:
        result = list(executor.map(limiter.try_acquire, items))
        for idx, coro in enumerate(result):
            while isawaitable(coro):
                coro = await coro
                result[idx] = coro

        return result


async def async_acquire(limiter: Limiter, item: str, weight: int = 1, blocking: bool = True) -> Tuple[bool, int]:
    start = time()
    acquire = await limiter.try_acquire_async(item, weight=weight, blocking = blocking)

    time_cost_in_ms = int((time() - start) * 1000)
    assert isinstance(acquire, bool)
    return acquire, time_cost_in_ms


async def async_count(bucket: AbstractBucket) -> int:
    count = bucket.count()

    if isawaitable(count):
        count = await count

    assert isinstance(count, int)
    return count


async def prefilling_bucket(limiter: Limiter, sleep_interval: float, item: str):
    """Pre-filling bucket to the limit before testing
    the time cost might vary depending on the bucket's backend
    - For in-memory bucket, this should be less than a 1ms
    - For external bucket's source ie Redis, this mostly depends on the network latency
    """
    acquire_ok, cost = await async_acquire(limiter, item)
    logger.info("cost = %s", cost)
    assert cost <= 50
    assert acquire_ok
    await sleep(sleep_interval)

    acquire_ok, cost = await async_acquire(limiter, item)
    logger.info("cost = %s", cost)
    assert cost <= 50
    assert acquire_ok
    await sleep(sleep_interval)

    acquire_ok, cost = await async_acquire(limiter, item)
    logger.info("cost = %s", cost)
    assert cost <= 50
    assert acquire_ok


async def flushing_bucket(bucket: AbstractBucket):
    flush = bucket.flush()

    if isawaitable(flush):
        await flush
