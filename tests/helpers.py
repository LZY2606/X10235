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
from pyrate_limiter import InMemoryBucket
from pyrate_limiter import Limiter
from pyrate_limiter import RateItem


class TransientBucketError(RuntimeError):
    """Recoverable bucket failure injected by the regression suite."""


class FlakyBucket(InMemoryBucket):
    """InMemoryBucket whose put() raises scripted, recoverable errors.

    Failure-injection seam for the regression suite: errors are queued with
    ``pending_errors`` and raised from the public ``put()``, so every failure
    still enters through the limiter's normal acquire path - no internal
    objects are forged.
    """

    def __init__(self, rates, clock=None):
        super().__init__(rates, clock=clock)
        self.pending_errors: List[Exception] = []

    def put(self, item):
        if self.pending_errors:
            raise self.pending_errors.pop(0)
        return super().put(item)


def bucket_snapshot(bucket: AbstractBucket) -> Tuple:
    """Comparable, printable snapshot of the invariants that prove
    rate-limit / virtual-clock semantics: how many units are stored, which
    rate last denied (if any), and every stored item newest-first.
    """
    count = bucket.count()
    assert isinstance(count, int)
    items = []

    for index in range(count):
        item = bucket.peek(index)
        assert isinstance(item, RateItem)
        items.append((item.name, item.timestamp, item.weight))

    failing = None if bucket.failing_rate is None else (bucket.failing_rate.limit, bucket.failing_rate.interval)
    return count, failing, tuple(items)


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
