# DiskANN 动态缓存策略 QPS Review

本文基于已有实验结果和当前代码路径解释：为什么部分动态策略提高了 cache hit rate、降低了平均 I/O 次数，但 QPS 仍低于 `StaticBFS(original)`。

输入数据：

- `experiments/cache_eval/sift1m/results/current_cache_policy_summary.csv`
- `experiments/cache_eval/sift1m/results/warm_bfs_cache_policy_summary.csv`
- `experiments/cache_eval/sift1m/results/current_trace_replay_summary.csv`

实验口径：

- dataset: SIFT1M
- queries: 10,000
- `search_list=100`
- `beam_width=4`
- cache capacity: 10,000 nodes
- recall: `Recall@10 = 97.04%` across all shown live-search rows

注意：下列表格来自现有 CSV。后续代码已修复 `2Q`/`SLRU` segment 容量维护，并让 `TinyLFU/W-TinyLFU` warmup 训练 frequency sketch；因此涉及这些策略的 warmup 和 hit/QPS 结论需要重跑实验后更新。

## 核心结论

`StaticBFS(original)` 的 QPS 更高，不是因为它 hit rate 最高。它的优势来自查询期路径极短：

1. static cache 只做 immutable hash lookup 和引用返回，不维护 replacement metadata。
2. dynamic cache 每个访问会进入全局 `Mutex`，更新策略状态，并且 dynamic hit 会 clone `CachedNode` 到当前 batch。
3. 当前统计中的 `Mean IO(us)` 并非纯 disk I/O。`disk_provider.rs` 在 `ensure_loaded` 外层计时，这段时间包含 cache lookup、policy update、payload clone、disk load、admission 和 store insert/remove。因此很多动态策略的 CPU 维护成本会显示成 `Mean IO(us)` 上升，而不是 `CPU(us)` 上升。
4. 复杂策略虽然减少了 1 到 2.5 次平均 I/O，但每次 query 仍有约 100 次 vertex load attempt；如果每次 attempt 多做锁、hash、队列移动或 heap/sketch 更新，节省的少量 I/O 很容易被抵消。

## Baseline 指标

| Policy | QPS | Mean latency us | P999 latency us | Mean IOs/query | Mean IO(us) | CPU(us) | Cache hit % | Recall@10 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NoCache` | 322.06 | 3103.78 | 4395.00 | 114.08 | 2788.92 | 311.36 | 0.00 | 97.04 |
| `StaticBFS(original)` | 349.69 | 2858.38 | 4035.00 | 104.32 | 2539.79 | 315.13 | 8.56 | 97.04 |

`StaticBFS(original)` 相比 `NoCache`：

- hit rate: `+8.56 pp`
- mean I/O: `-9.76` vertices/query
- QPS: `+8.6%`

这说明当前实验中缓存确实能改善性能；问题是动态策略的维护成本高于其额外 hit rate 收益。

## Cold dynamic 与 BFS warmup 对比

表中 `Warm` 指 `DynamicNodeCacheWithBfsWarmup`，即先用 medoid BFS payload 填充动态 cache，再让策略继续在线替换。它不是 static cache。

| Policy | Cold hit % | Cold IOs/q | Cold Mean IO(us) | Cold QPS | Warm hit % | Warm IOs/q | Warm Mean IO(us) | Warm QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 8.58 | 104.29 | 2681.55 | 334.72 | 8.58 | 104.28 | 2654.99 | 339.24 |
| LRU | 9.21 | 103.58 | 2725.00 | 329.34 | 9.21 | 103.57 | 2686.61 | 335.49 |
| LFU | 10.08 | 102.57 | 3136.12 | 290.06 | 10.10 | 102.55 | 3061.42 | 298.02 |
| Random | 8.42 | 104.47 | 2977.30 | 304.29 | 8.44 | 104.45 | 2936.26 | 309.23 |
| TinyLFU | 9.52 | 103.22 | 2645.50 | 339.83 | 9.54 | 103.19 | 2627.90 | 342.81 |
| W-TinyLFU | 10.36 | 102.26 | 3640.02 | 253.12 | 10.34 | 102.28 | 3595.30 | 256.79 |
| CLOCK | 9.03 | 103.78 | 2682.24 | 334.65 | 9.04 | 103.77 | 2662.93 | 338.20 |
| 2Q | 10.74 | 101.82 | 3807.16 | 242.63 | 10.81 | 101.75 | 3748.65 | 247.13 |
| SLRU | 10.63 | 101.96 | 3383.83 | 270.89 | 10.64 | 101.93 | 3384.45 | 271.40 |
| LIRS | 10.46 | 102.14 | 5300.06 | 178.31 | 10.50 | 102.10 | 5343.90 | 177.16 |
| ARC | 8.38 | 104.51 | 3428.92 | 267.44 | 7.92 | 105.05 | 3430.94 | 268.12 |
| GDSF | 9.92 | 102.76 | 2858.13 | 315.67 | 9.94 | 102.74 | 2829.92 | 319.91 |
| LeCaR | 9.22 | 103.56 | 2836.07 | 316.12 | 9.22 | 103.55 | 2765.58 | 326.39 |
| Cacheus | 10.61 | 101.97 | 3611.57 | 254.89 | 10.69 | 101.89 | 3595.81 | 256.73 |

相对 `StaticBFS(original)`，cold dynamic 中“命中率提高但 QPS 下降”的策略为：

| Policy | Hit gain vs StaticBFS | Mean IOs/query change | Mean IO(us) change | QPS change |
|---|---:|---:|---:|---:|
| FIFO | +0.02 pp | -0.03 | +141.76 | -4.3% |
| LRU | +0.65 pp | -0.74 | +185.21 | -5.8% |
| LFU | +1.52 pp | -1.75 | +596.33 | -17.1% |
| TinyLFU | +0.96 pp | -1.10 | +105.71 | -2.8% |
| W-TinyLFU | +1.80 pp | -2.06 | +1100.23 | -27.6% |
| CLOCK | +0.47 pp | -0.54 | +142.45 | -4.3% |
| 2Q | +2.18 pp | -2.50 | +1267.37 | -30.6% |
| SLRU | +2.07 pp | -2.36 | +844.04 | -22.5% |
| LIRS | +1.90 pp | -2.18 | +2760.27 | -49.0% |
| GDSF | +1.36 pp | -1.56 | +318.34 | -9.7% |
| LeCaR | +0.66 pp | -0.76 | +296.28 | -9.6% |
| Cacheus | +2.05 pp | -2.35 | +1071.78 | -27.1% |

`Random` 和 `ARC` 不在这张清单中，因为 cold hit rate 低于 `StaticBFS`；但它们的 QPS 也低于 `StaticBFS`。

## Trace replay 的含义

`current_trace_replay_summary.csv` 只重放 vertex id 序列，不执行 live DiskANN search。它验证的是 policy metadata 的 hit/miss 行为：

| Policy | Replay hit % | Admissions | Evictions | Rejections |
|---|---:|---:|---:|---:|
| no_cache | 0.00 | 0 | 0 | 1,140,699 |
| fifo | 8.58 | 1,042,840 | 1,032,840 | 0 |
| lru | 9.21 | 1,035,691 | 1,025,691 | 0 |
| lfu | 10.08 | 1,025,676 | 1,015,676 | 0 |
| tiny_lfu | 9.51 | 112,248 | 102,248 | 919,958 |
| w_tiny_lfu | 10.35 | 1,022,687 | 1,012,687 | 0 |
| random | 8.42 | 1,044,601 | 1,034,601 | 0 |
| clock | 9.03 | 1,037,692 | 1,027,692 | 0 |
| 2q | 10.74 | 1,018,192 | 1,008,192 | 0 |
| slru | 10.62 | 1,019,505 | 1,009,505 | 0 |
| lirs | 10.47 | 1,021,322 | 1,011,322 | 0 |
| arc | 8.38 | 1,045,079 | 1,035,079 | 0 |
| gdsf | 9.92 | 1,027,557 | 1,017,557 | 0 |
| lecar | 9.22 | 1,035,529 | 1,025,529 | 0 |
| cacheus | 10.58 | 1,020,040 | 1,010,040 | 0 |
| belady_opt | 24.70 | 858,924 | 848,924 | 0 |

关键解释：

- Replay hit rate 与 live `cache_hit_percent` 基本一致，说明 live 低 QPS 不是因为 replay 统计完全失真。
- `BeladyOptimal` 的 `24.70%` 是 offline 上界，说明这条访问序列还有缓存空间；但达到上界需要知道未来，不能作为 live 策略 QPS 目标。
- `TinyLFU` replay rejections 很高。live search 中这些 reject 发生在 disk read 之后，因为当前 admission 需要先把 miss node 读出来再决定是否插入；因此 reject 减少 store churn，但不减少本次 miss I/O。

## 代码级原因

### 1. Dynamic cache 使用单个全局 `Mutex`

`SharedNodeCache::Dynamic` 是 `Arc<Mutex<DynamicNodeCache<Data>>>`。每个 vertex lookup 调一次 `lookup_dynamic`，每个 miss admission 再调一次 `admit_dynamic`。

影响：

- 多线程查询会争用同一个锁。
- 锁内执行 `policy.record_access`、`store.get_node` clone、`policy.admit`、`store.remove/insert`。
- 高命中策略并不一定更快，因为命中越多，锁内 policy hit maintenance 越多。

相关代码：

- `cached_disk_vertex_provider.rs:45` `SharedNodeCache::Dynamic`
- `cached_disk_vertex_provider.rs:129` `lookup_dynamic`
- `cached_disk_vertex_provider.rs:144` `admit_dynamic`

### 2. `Mean IO(us)` 包含 cache 维护成本

`DiskSearchStrategy::ensure_loaded` 在调用 `ensure_vertex_loaded` 前后计时，并把整段写入 `io_time_us`。

这段包含：

- cache lookup/filter
- dynamic policy update
- hit payload clone
- disk provider load
- process loaded node
- dynamic admission and store mutation

因此动态策略的 CPU 维护成本会表现为 `Mean IO(us)` 变大，而 `CPU(us)` 反而可能下降。CSV 中很多动态策略 `CPU(us)` 低于 StaticBFS，但 `Mean IO(us)` 大幅上升，正是这个计量口径造成的。

相关代码：

- `disk_provider.rs:620` 到 `disk_provider.rs:635`
- `search_disk_index.rs:300` 到 `search_disk_index.rs:313`

### 3. `VecDeque` 的 `remove_from_order` 是线性扫描

`remove_from_order` 使用 `order.iter().position(...)`，然后 `order.remove(idx)`。

受影响策略：

- LRU: 每次 hit 移动到 MRU。
- TinyLFU: 基础 victim 是 LRU，hit 也移动。
- 2Q/SLRU/LIRS/ARC/W-TinyLFU/Cacheus: hit、promotion、demotion、history hit 都可能在线性队列中删除。

容量 10,000 时，每 query 约 100 次 vertex access，百万级 trace 下 `O(C)` 维护成本足以抵消 1 到 2 次 I/O 节省。

相关代码：

- `cache_policy.rs:1684` `move_to_back`
- `cache_policy.rs:1690` `remove_from_order`

### 4. Dynamic hit 会 clone payload

`Cache::get_node` 返回 `CachedNode` by value：

- vector: `to_vec()`
- adjacency list: `clone()`
- associated data: copy

然后 `CachedDiskVertexProvider` 把 node 放入 `cached_nodes_for_current_read`，使后续 accessor 能返回引用而不持有锁。

这是一个合理的 Rust 生命周期取舍，但 static cache hit 是直接返回 immutable cache 中的引用，成本明显更低。

相关代码：

- `cache.rs:71` `Cache::get_node`
- `cached_disk_vertex_provider.rs:320` dynamic hit insert into current-read map

### 5. Admission 后仍需 store insert/remove

动态 miss 路径先从 disk provider 读 node，再执行 admission：

- evicted 时 `store.remove`，内部会 `swap_remove` adjacency/associated data，并在 vector flat buffer 中 `copy_within` 移动最后一个 slot。
- admitted 时 `store.insert`，复制 full vector 和 adjacency。
- rejected 时本次 disk read 已经发生，只是避免进入 cache。

相关代码：

- `cached_disk_vertex_provider.rs:242` `process_loaded_node`
- `cache_policy.rs:1747` `DynamicNodeCache::admit_node`
- `cache.rs:105` `Cache::insert`
- `cache.rs:137` `Cache::remove`

### 6. Lazy heaps 在 LFU/GDSF/LeCaR/Cacheus 中会膨胀

LFU 和 GDSF 每次 hit 都 push 新 heap entry，victim 时再跳过 stale entries。

受影响策略：

- `LFU`
- `GDSF`
- `LeCaR` 的 LFU expert
- `Cacheus` 的 LFU expert

如果热点页频繁命中，heap 长度可能远大于 resident 数量，增加内存和 victim scan 成本。

相关代码：

- `cache_policy.rs:635` `push_lfu_entry`
- `cache_policy.rs:648` `lfu_victim`
- `cache_policy.rs:1083` `record_gdsf_hit`
- `cache_policy.rs:1093` `gdsf_victim`

### 7. TinyLFU/W-TinyLFU 每次 access 更新 sketch/doorkeeper

`record_access` 对 `TinyLfu` 和 `WTinyLfu` 在 resident 判断前调用 `record_tiny_lfu_access`。这意味着 hit 和 miss 都更新 frequency model。

成本包括：

- exact doorkeeper `HashSet` insert/lookup
- 4 行 sketch counter hash/update
- sample reset 时全表衰减
- admission 比较时额外 estimate

这些成本可以换来 hit rate，但在当前实验中 W-TinyLFU 的 hit gain `+1.80 pp` 没有抵消 `Mean IO(us) +1100.23`。

相关代码：

- `cache_policy.rs:394` `record_access`
- `cache_policy.rs:673` `record_tiny_lfu_access`
- `cache_policy.rs:1199` `admit_wtiny_lfu`

### 8. Trace writer 是隐藏开销源

如果设置 `DISKANN_CACHE_TRACE_PATH`，每个 vertex access 都会写 JSONL，并用单个 `Mutex<BufWriter<File>>`。

当前 CSV 是否开启 trace 需要看运行脚本；如果开启，dynamic/static 都受影响，但 dynamic 的访问路径已经更长，额外写 trace 会放大差异。

相关代码：

- `cached_disk_vertex_provider.rs:25` `trace_cache_access`

## 为什么 ARC warmup 后 hit rate 下降

Cold ARC:

- query trace 自己逐步填充 `T1/T2/B1/B2`。
- ghost hits 根据实际访问序列调整 `p`。

Warm ARC:

- `DynamicNodeCache::from_warm_cache` 调 `policy.warm`，等价于按 BFS id 顺序做 admission。
- 所有 warm nodes 初始进入 `T1`，`T2/B1/B2` 为空，`p=0`。
- 如果 BFS warm nodes 与后续 query trace 的 ARC recent/frequent 划分不匹配，早期 miss 会把 warm `T1` 节点推到 `B1`，并用这些 BFS ghost 干扰 `p` 的学习。

结果：

- cold ARC hit: `8.38%`
- warm ARC hit: `7.92%`
- warm mean I/O: `105.05`，比 StaticBFS 的 `104.32` 更差

这不是 ARC 论文机制必然失败，而是当前 warmup state 与 ARC 的 online learned state 不等价。

建议验证：

- 变体 A：BFS warm nodes 插入 `T2`。
- 变体 B：BFS depth 近的进 `T2`，远的进 `T1`。
- 变体 C：先 replay 一段训练 trace 来形成 `B1/B2/p`，再 live search。

## 为什么旧 CSV 中 W-TinyLFU warmup 后 hit rate 下降

旧 CSV 对应的实现中，Warm W-TinyLFU 的 payload 被填入 cache，但 frequency model 没有被训练：

- `from_warm_cache` 调 `warm_node`。
- `warm_node` 调 `policy.warm`。
- 当时 `PolicyCache::warm` 只是 `admit`，不会调用 `record_tiny_lfu_access`。
- 因此旧结果里的 sketch/doorkeeper 对 warm BFS nodes 没有频率记忆。

结果：

- cold W-TinyLFU hit: `10.36%`
- warm W-TinyLFU hit: `10.34%`
- warm mean I/O 从 `102.26` 变为 `102.28`

下降很小，但方向符合旧实现：warmup 初始化了 window/probation/protected payload，却没有初始化 TinyLFU 的 admission 知识。后续 window candidate 与 main victim 比较时，频率估计仍接近 cold start。

当前代码已改为 warmup 时同步调用 `record_tiny_lfu_access`。因此这段解释仅适用于旧 CSV；修复后的 W-TinyLFU warmup 需要重跑 live search 和 trace replay。

建议验证：

- warmup 时对每个 BFS id 调一次或多次 `record_tiny_lfu_access`，再 `admit`。
- 用 `current_access_trace.cleaned.jsonl` 的前 N% 训练 sketch，但不写 payload store，比较 trained sketch 与 payload warmup 的独立影响。
- sweep window ratio；当前固定 1% 可能不是 DiskANN trace 的最佳点。

## 优化建议

### 低风险优化

| 优化 | 风险 | 验证方式 |
|---|---|---|
| 将 `VecDeque + linear remove` 替换为 O(1) linked hash structure 或 index-linked list | 中低。需要保持所有 segment/history set 一致。 | 对每个策略跑现有 unit tests、新增短 trace 状态测试；比较 replay hit rate 必须逐行一致；用 `perf`/criterion 测每百万 access 时间。 |
| 在 `filter_cached_nodes` 中按 batch 获取一次 dynamic cache lock，而不是每个 vertex lock/unlock 一次 | 低到中。需要避免扩大锁内 disk I/O；只批量 lookup，不批量 load。 | 统计 lock 次数；live search 比较 QPS、p999 latency、hit rate、Recall@10。 |
| 为 LFU/GDSF/LeCaR/Cacheus 增加 heap compaction 或 indexed heap | 中低。victim tie-breaker 可能变化。 | 记录 heap len/resident len；设置 compaction threshold，例如 `heap.len() > 4 * resident.len()`；验证 replay hit rate 或允许声明 tie-breaker 变体。 |
| 减少 duplicate hash lookup：store lookup 与 policy contains/admit 尽量共用结果 | 低。主要是代码整理。 | 使用 `perf stat` 或 flamegraph 看 hashbrown lookup 占比；确保 policy/store 一致性测试通过。 |
| trace 写入默认完全旁路，或改成 per-thread buffered trace/sample trace | 低。只影响 tracing。 | 分别在 trace on/off 下跑同一策略；报告 QPS 差异，确保 JSONL 字段兼容。 |
| 预分配 `cached_nodes_for_current_read` 和 `nodes_to_fetch_local_idx_to_filtered_idx` 到 batch size | 低。 | 比较 alloc count；QPS 小幅提升即可接受。 |

### 中风险优化

| 优化 | 风险 | 验证方式 |
|---|---|---|
| Sharded dynamic cache：按 vertex id 分片锁和 payload store | 中。全局容量、全局 victim 策略会变成近似。 | 每片容量固定或按负载调节；比较 hit rate/QPS/latency；记录 shard imbalance。 |
| 分离 policy metadata 与 payload store 锁 | 中。需要严格维护 evict/admit 原子性。 | 增加不变量断言：policy resident ids 与 store ids 一致；压力测试多线程查询。 |
| Per-thread local cache + global admission queue | 中高。可能改变 hit rate 和内存占用。 | 统计 local/global hit；总内存限制；比较 p999 latency，避免个别线程 cache 污染。 |
| 批量 admission：一个 query/batch 的 misses 汇总后更新策略 | 中高。策略语义变为 batch online。 | replay 中实现 batch mode 对照；关注 Recall@10 不变、hit rate 与 QPS 的 tradeoff。 |

### 策略级优化

| 优化 | 适用策略 | 风险 | 验证方式 |
|---|---|---|---|
| Hybrid cache：保留 20% 到 80% 容量给 immutable StaticBFS，其余容量给动态策略 | 所有动态策略 | 中。总容量切分需要 sweep。 | sweep static fraction；报告 hit rate、QPS、Mean IO(us)、Recall@10；重点看是否保留 StaticBFS 低开销。 |
| TinyLFU/W-TinyLFU 训练 trace warmup | TinyLFU, W-TinyLFU | 低到中。BFS warmup 已训练 sketch，但真实 trace 训练可能过拟合。 | 比较 BFS payload+sketch warmup 与训练 trace sketch warmup；报告 rejections 与 hit rate。 |
| W-TinyLFU window size 可调 | W-TinyLFU | 低。只是参数化。 | sweep 0.5%、1%、2%、5%、10%；关注 hit gain 是否抵消 Mean IO(us)。 |
| ARC warmup placement 变体 | ARC | 中。不是标准 ARC cold start。 | BFS 全进 `T1`、全进 `T2`、depth split、trace replay warmup 四组对比。 |
| 2Q/SLRU/LIRS segment ratio 参数化并修复容量不变量 | 2Q, SLRU, LIRS | 中。会改变当前结果。 | 先用 unit tests 锁定论文状态机，再重跑 trace replay 和 live search。 |
| GDSF 使用 node payload byte size | GDSF | 中。需要估算 adjacency list 大小并改容量单位。 | 比较 node-count capacity 与 byte capacity；报告 memory footprint 和 hit/QPS。 |

## 推荐验证矩阵

每个优化至少报告以下指标：

- QPS
- mean latency us
- p999 latency us
- `Mean IO(us)`
- `CPU(us)`，并注明当前 CPU 不含 `ensure_loaded` 中的 cache 维护成本
- mean I/O per query
- cache hit %
- Recall@10
- policy-specific metrics，例如 heap len、lock wait time、segment sizes、rejections

建议优先做三个最小实验：

1. LRU O(1) order structure: 验证最基础动态策略能否接近 StaticBFS QPS。
2. Batch dynamic lookup lock: 验证全局锁开销占比。
3. Hybrid StaticBFS + TinyLFU or LRU: 验证“保留静态低成本热点 + 动态补充长尾”是否优于全动态。

## 最终判断

当前数据支持以下判断：

- 动态策略的 policy hit 行为基本工作，trace replay 与 live hit rate 互相印证。
- QPS 下降主要来自 live path 工程成本，而不是策略完全无效。
- `StaticBFS(original)` 是很强的系统基线，因为它查询期几乎零维护成本。
- 对复杂策略，当前实现应先优化数据结构和锁，再讨论论文级 hit rate 优劣。
- `ARC`、`2Q`、`SLRU`、`Cacheus` 等策略在 correctness 层面仍有近似或 bug-risk；这些策略的性能结果不应作为原论文策略的最终结论。

## 相关代码位置

- `diskann-disk/src/search/provider/cached_disk_vertex_provider.rs:129` dynamic lookup lock
- `diskann-disk/src/search/provider/cached_disk_vertex_provider.rs:242` miss 后 admission
- `diskann-disk/src/search/provider/cached_disk_vertex_provider.rs:307` cache filter path
- `diskann-disk/src/search/provider/disk_provider.rs:620` `ensure_loaded` 计时范围
- `diskann-disk/src/data_model/cache.rs:71` dynamic hit clone payload
- `diskann-disk/src/data_model/cache.rs:105` store insert
- `diskann-disk/src/data_model/cache.rs:137` store remove
- `diskann-disk/src/data_model/cache_policy.rs:394` policy `record_access`
- `diskann-disk/src/data_model/cache_policy.rs:448` policy `admit`
- `diskann-disk/src/data_model/cache_policy.rs:635` LFU heap update
- `diskann-disk/src/data_model/cache_policy.rs:673` TinyLFU sketch update
- `diskann-disk/src/data_model/cache_policy.rs:1690` linear `remove_from_order`
