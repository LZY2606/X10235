# PyrateLimiter 同步/异步时钟与 bucket 原子语义收束

## 契约（不可漂移）
- 公共入口: Limiter(rate/bucket/factory)、try_acquire、try_acquire_async、as_decorator
- 异常分层: 超时 -> 公共入口统一吞 TimeoutError 返回 False；weight 非法 ValueError；weight>limit 等待返回 False；不支持 -> RuntimeError
- 等待语义: 先 put 判定，被拒才等待；取消不消耗额度；buffer_ms/-1 语义不变

## Steps
1. clocks.py: 增加 VirtualMonotonicClock（单调虚拟时钟，回退显式抛错——一致的“不支持回退”策略）
2. in_memory_bucket.py: 构造支持 clock= 注入（与 StateBucket 同形，默认 MonotonicClock）
3. limiter.py:
   - 新增内部边界 _TimePrimitives（now 秒 / sleep / async_sleep），默认绑定真实 monotonic；Limiter(..., _time=None) 仅测试注入
   - monotonic 读取全部经由 self._time.now；_plan_delay_step/combined_lock 接收 now 参数
   - delay 计算三处循环（async _handle_async / sync _blocking_retry_sync / legacy sync）抽出共享 _next_delay_step；bucket acquire 已共享 handle_bucket_put
   - 异常转换公共入口统一走 _wrap_timeout
4. 修复测试收集: test_aiohttp_limiter.py 缺 aiohttp 时 importorskip
5. 新增 tests/test_clock_bucket_atomicity.py（全部 @pytest.mark.inmemory），经公共入口：
   - 边界时间 / 批量 weight / weight>limit 拒绝分层
   - 取消竞态不扣额度（async cancel + sync 线程中断）
   - 先扣额度再等待的回归（sleep 前断言 bucket 未增长）
   - 时钟倒退一致策略（VirtualMonotonicClock 显式拒绝）
   - 同步异步等价（同虚拟时钟同输入 -> 同结果/时间戳轨迹）
   - bucket 抛错同型传播 + 成功后无残留（failing_rate/_last_wait/计数）
   - 同 bucket 并发不越限（barrier + 虚拟时间 + 准入轨迹）
   - 多 limiter 时钟不泄漏（独立虚拟时钟、互不推进）
   - 可恢复失败中段注入：重试结果稳定，共享状态无半更新
   - 顺序稳定组：仅改变枚举/调度顺序，聚合不变量一致；断言消息含首个违约位置与上下文
6. 固定 seed 的随机小工具（直接 random.Random(SEED)，失败时打印最小复现参数）
7. 验收: python3 -m venv .venv && .venv/bin/pip install -e . pytest pytest-asyncio
        .venv/bin/python -m pytest -q -m inmemory
