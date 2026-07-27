# QASC v1 实现验证与 SIFT1M 初步结果

## 1. 目的和结论边界

本文记录 `feature/qasc-cache` 分支上 QASC v1 的正确性测试和一次可复现的
SIFT1M 单线程 smoke experiment。它回答两个不同问题：

1. QASC 是否已正确接入 DiskANN 搜索，并保持结果、容量和逻辑计费不变量？
2. 当前 v1 启发式是否已经优于通用 LRU？

第一个问题的答案是“是”。第二个问题的答案是“尚不能”。原始 query 顺序下，
QASC 的 cache hit 比 LRU 高 `0.45` 个百分点，但 QPS 更低；k-means 连续顺序下，
LRU 明显优于 QASC。该结果说明 query-conditioned cache 的接口和数据通路已经落地，
但当前 admission、quota 和 victim 启发式还不是一个成熟的性能策略。

## 2. 被验证的实现版本

实验运行于以下提交之后：

```text
278c1fa3 Fix bounded QASC affinity classification
11152013 Test QASC integration and invariants
8c875571 Expose QASC search configuration
5abc8d19 Route search queries into QASC
9c8de5b6 Add query-affinity cache core
0baccd7d Document QASC implementation design
```

实现契约见 `qasc_implementation_design_zh.md`。实验日期为 2026-07-27。

## 3. 自动化测试覆盖

QASC 核心测试覆盖：

- prototype fbin 加载、维度校验和有限值校验；
- L2、cosine、cosine-normalized 和 inner-product query 路由；
- soft top-m 的确定性和等距离 tie；
- 一个图节点只有一份物理 payload；
- fractional charge 总和为 1；
- global promotion、elastic quota、借用和超配额回收；
- affinity 项数上限、周期衰减和 bounded ghost metadata；
- 低 utility、低 observation candidate 的拒绝和成功 replacement；
- `store.len <= capacity`、逻辑 usage 和 resident metadata 一致性。

真实磁盘索引集成测试使用 `test_data/disk_index_search`，验证：

- QASC 与 `NoCache` 返回完全相同的 top-k ID；
- 同一个 query 第二次搜索能够复用节点；
- 第二次搜索的物理 I/O 少于第一次搜索。

CLI 测试验证 prototype 文件必填、QASC 拒绝 sharded backend，以及合法参数能够
构造 `CachingStrategy::QueryAffinityCache`。既有 27 个 cache-policy 测试继续通过。

执行命令：

```bash
cargo fmt --all -- --check
cargo check -p diskann-tools --bins
cargo test -p diskann-disk query_affinity_cache --no-fail-fast
cargo test -p diskann-disk qasc_preserves_results_and_reuses_cached_nodes --no-fail-fast
cargo test -p diskann-tools --bin search_disk_index --no-fail-fast
cargo test -p diskann-disk data_model::cache_policy::tests --no-fail-fast
```

## 4. 实验环境和固定参数

硬件与系统：

```text
CPU: 2 x Intel Xeon Gold 6330, 56 physical cores, 112 logical CPUs
L3: 2 x 42 MiB
OS: Linux 6.8.0-117-generic x86_64
Build: cargo --release
```

所有测量串行执行，固定：

```text
dataset=SIFT1M
metric=L2
PQ chunks=16
num_threads=1
search_list=100
beam_width=4
recall_at=10
search_io_limit=1000
cache_capacity=10000 nodes
QASC prototypes=100
QASC top_m=2
QASC global_fraction=0.25
```

QASC prototype 使用既有 `K=100, seed=42` query 聚类中心。这里只用于实现 smoke
test，属于对测试分布的乐观设定，不应被解释为可部署的 prototype 来源。正式评估应
改用 train/base/history-only prototypes，并单独检查 distribution drift。

## 5. 输入可追溯性

prototype 从 NumPy 矩阵转换成标准 DiskANN fbin：

```bash
python3 -c '
import pathlib, struct, numpy as np
src = pathlib.Path("experiments/cache_eval/results/vector_patterns/sift1m/k100_seed42/query.centers.npy")
dst = pathlib.Path("experiments/cache_eval/sift1m/results/qasc_validation/query_k100_seed42.prototypes.fbin")
a = np.asarray(np.load(src), dtype="<f4", order="C")
assert a.ndim == 2 and np.isfinite(a).all()
dst.write_bytes(struct.pack("<II", *a.shape) + a.tobytes(order="C"))
'
```

关键输入的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `query.centers.npy` | `c19c88eb014f38c9baa9ee699aa3fef8056ed1c0108f1ea9fbfcc3465e1037d0` |
| `query_k100_seed42.prototypes.fbin` | `46b90945498409ebed1bb21839d22cbca4f94f6e6193816cea8ef3487a0970af` |
| `query.fbin` | `9b0082b67d0ac55b4c7d42216560344567ad87ce3e75a9d5214a0762f1c15d65` |
| `groundtruth.bin` | `047418d19f3981e4fc2dd28ae3b6f72fa7108f40d9336bd2d80b07829885ba13` |
| `query.kmeans_k100_seed42.fbin` | `b2c486f01888aecd1649ddc54ea8c38cc828bbc6e0038318329a4b5f79d7bf20` |
| `groundtruth.kmeans_k100_seed42.bin` | `79e8f29ea71009cd13da5cdec84c2e8a34b902f71e714e16bb332315d098695d` |
| `sift_disk_disk.index` | `f99ea76ef4a3c0d1658e96e76aed65b194dc930bf7570236c6e1dbee46736907` |
| `sift_disk_pq_pivots.bin` | `1ac335ae59fb9cec5903a9376e96f97f1bf73092c36b4d09444fec2c9d415cf0` |
| `sift_disk_pq_compressed.bin` | `4aa817923bc7669538d2a4e2182681eca91cb0a2245b048aa5a06e712578d8e6` |

## 6. 运行方式

以 QASC 原始 query 顺序为例：

```bash
target/release/search_disk_index \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix experiments/cache_eval/sift1m/sift_disk \
  --result_output_prefix experiments/cache_eval/sift1m/results/qasc_validation/qasc_10000_k100_entropyfix \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --ground_truth_file experiments/cache_eval/sift1m/groundtruth.bin \
  --search_list 100 \
  --beam_width 4 \
  --recall_at 10 \
  --search_io_limit 1000 \
  --num_threads 1 \
  --cache_policy qasc \
  --cache_capacity 10000 \
  --qasc_prototypes_file experiments/cache_eval/sift1m/results/qasc_validation/query_k100_seed42.prototypes.fbin \
  --qasc_top_m 2 \
  --qasc_global_fraction 0.25
```

LRU 将 policy 改为 `lru` 并删除 QASC 参数。k-means workload 将 query 和 truth
分别替换为 `workloads/query.kmeans_k100_seed42.fbin` 与
`workloads/groundtruth.kmeans_k100_seed42.bin`。

## 7. 结果

### 7.1 原始 query 顺序

| Policy | QPS | Mean latency us | P99.9 us | Mean I/O | CPU us | Cache hit | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NoCache | 313.51 | 3188.41 | 4268 | 114.08 | 303.36 | 0.00% | 97.04% |
| LRU | 325.16 | 3074.05 | 4277 | 103.58 | 311.64 | 9.21% | 97.04% |
| QASC | 245.26 | 4075.84 | 15509 | 103.06 | 395.92 | 9.66% | 97.04% |

QASC 比 LRU 多 `0.45` 个百分点命中率，平均 I/O 再减少 `0.52`，但 QPS 低
`24.6%`。当前收益不足以覆盖 profile 更新、全局 mutex 和线性 victim 搜索。

### 7.2 k-means 连续 query 顺序

| Policy | QPS | Mean latency us | P99.9 us | Mean I/O | CPU us | Cache hit | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LRU | 376.22 | 2656.69 | 4376 | 71.77 | 302.64 | 37.09% | 97.04% |
| QASC | 127.54 | 7839.09 | 21491 | 99.36 | 420.14 | 12.90% | 97.04% |

在 cluster 连续到达时，LRU 能把几乎整个 cache 暂时用于当前 cluster。QASC v1
则维护跨 cluster 的长期逻辑配额和历史 resident，不能自动等价为 burst-local LRU。

## 8. 解释和后续实现优先级

本次结果支持以下结论：

1. **Query context 通路正确。** QASC 不改变搜索路径的语义，Recall 在所有运行中
   保持 `97.04%`，且真实索引集成测试证明重复 query 能产生物理 I/O 复用。
2. **空间分区不应替代时间自适应。** 强连续 locality 下，纯 LRU 是很强的基线。
   QASC 更可能适用于多个 query region 交错、每个 region 会在较长间隔后再次出现的
   workload；需要构造 interleaved-cluster trace 验证这个假设。
3. **逻辑 quota 需要工作集压力信号。** 只按 decayed query mass 的平方根分配容量，
   不能识别某个 cluster 当前处于 burst，也不能估计各 cluster 的 marginal hit gain。
4. **admission 需要同时看 novelty 和 predicted reuse。** 固定 observation 阈值过于粗糙。
   后续应使用 ghost hit、cluster reuse distance 或短期 route pressure，而不是简单改成 1。
5. **victim 查找必须索引化。** 当前满缓存 admission 可能扫描大量 resident；应为 global
   和每个 cluster 维护有界候选堆或 CLOCK 队列，把常见路径从 `O(capacity)` 降低到
   摊销 `O(log capacity)` 或 `O(1)`。
6. **并发优化尚未开始。** v1 使用单个 `Mutex<QueryAffinityCache>`。在完善单线程策略
   前不应宣称多线程可扩展性，之后需要分片 payload store 并周期合并 quota/profile。

下一轮策略实验至少应包括：原始、随机、cluster-contiguous 和 cluster-interleaved 四类
workload；使用 train/base/history-only prototypes；同时报告 hit、I/O、Recall、QPS、
QASC CPU、锁等待、admission/rejection/eviction、global occupancy 和 cluster quota usage。
