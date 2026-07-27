# 四类静态缓存节点来源对比实验

日期：2026-07-27
实验分支：`eval/static-cache-node-sources`
搜索实现 commit：`50b05ccaa21f09d95dde7151c5e1e80425775903`

## 1. 实验问题与结论

本实验比较四种等容量静态缓存节点来源：

1. `bfs`：从图 medoid 出发，按 DiskANN 原生顺序进行 BFS。
2. `all_base_hot`：把所有 base 向量作为 query 后，按节点拓展次数降序选取。
3. `sample_base_hot`：固定 seed 42，从 base 中随机采样 1% 作为 query 后选取热点。
4. `test_hot`：用测试集 query 统计热点。这是使用未来负载信息的 oracle，只用于给出上界，不能直接作为无泄漏的部署方案。

主要结论：

- **节点来源会显著影响静态缓存价值，而且效果高度依赖数据集。** 相对 BFS，all-base 热点的命中率在 SIFT1M、Fashion-MNIST、GloVe-25、Last.fm 上分别提高 `4.63`、`8.11`、`1.90`、`3.41` 个百分点。
- **Fashion-MNIST 的 base pattern 可以很好迁移到 test。** all-base 与 test oracle 节点重合 `86.50%`，命中率只差 `0.11` 个百分点；1% base sample 也达到 `16.52%` 命中率和 `+6.39%` QPS。
- **Last.fm 的 test pattern 与 base pattern 严重失配。** all-base 与 test oracle 仅重合 `11.18%`，命中率为 `15.38%`，而 oracle 达到 `44.55%`；oracle 相对 BFS 减少 `37.01%` Mean I/O，并提高 `25.11%` QPS。
- **只看节点集合重合度不够。** SIFT1M 中 BFS 与 all-base 仅重合 `12.58%`，但这部分节点承载全部测试拓展的 `7.76%`，占 BFS 所有命中价值的 `92.39%`。四个来源都包含一个很小但非常高价值的共享导航核心，其余容量才决定来源之间的主要差异。
- **更高命中率不保证 QPS 同比例提高。** GloVe oracle 相对 BFS 将 Mean I/O 降低 `6.57%`，但中位 QPS 基本不变；本机未清空 OS page cache，且静态哈希查找和距离计算仍有固定成本。因此命中率/Mean I/O 是机制指标，QPS 是受存储层和机器状态共同影响的端到端指标。

## 2. 实验口径

每个数据集缓存容量固定为图节点数的 1%，向上取整：SIFT1M `10,000`，Fashion-MNIST `600`，GloVe-25 `11,836`，Last.fm `2,924`。四个来源在同一数据集内容量完全相同。

测试搜索参数沿用此前达到 Recall@10 > 95% 的配置：

| 数据集 | 距离/导航方式 | L | Queries | Recall@10 |
|---|---|---:|---:|---:|
| SIFT1M | L2, PQ=16 | 100 | 10,000 | 97.04% |
| Fashion-MNIST | L2, PQ=49 | 50 | 10,000 | 99.13% |
| GloVe-25 | 归一化余弦, PQ=16 | 20 | 10,000 | 96.79% |
| Last.fm | MIPS 转换后的 cosine, 65D 全精度导航 | 20 | 50,000 | 96.51% |

共同参数为 `beam_width=4`、`recall_at=10`、`num_threads=1`。每个配置重复三次，来源顺序循环移位，报告中使用中位数。DiskANN 输出的 QPS 只覆盖 search phase，不包含索引和静态缓存装载时间。

“节点拓展占比”定义为指定节点集合上的测试 query 拓展次数之和，除以测试 query 的全部节点拓展次数。由于静态缓存中的每次目标节点拓展都会命中，该值应与实际 cache hit 相同；本实验全部 16 个配置在线命中率均与离线拓展占比在两位小数上完全一致。

## 3. 两两节点重合度

下表中的“重合”是等容量集合的 overlap coefficient；由于左右集合大小相同，它也等于交集节点数除以单个集合容量。“交集拓展占比”是交集节点承载的测试总拓展比例。

| 数据集 | 节点来源对 | 重合节点 | 重合 | Jaccard | 交集拓展占比 |
|---|---|---:|---:|---:|---:|
| SIFT1M | BFS / all-base | 1,258 | 12.58% | 6.71% | 7.76% |
| SIFT1M | BFS / sample-base | 1,087 | 10.87% | 5.75% | 7.64% |
| SIFT1M | BFS / test | 1,053 | 10.53% | 5.56% | 7.69% |
| SIFT1M | all-base / sample-base | 4,912 | 49.12% | 32.56% | 10.36% |
| SIFT1M | all-base / test | 4,856 | 48.56% | 32.07% | 11.06% |
| SIFT1M | sample-base / test | 3,415 | 34.15% | 20.59% | 9.90% |
| Fashion | BFS / all-base | 79 | 13.17% | 7.05% | 8.78% |
| Fashion | BFS / sample-base | 76 | 12.67% | 6.76% | 8.69% |
| Fashion | BFS / test | 82 | 13.67% | 7.33% | 8.81% |
| Fashion | all-base / sample-base | 332 | 55.33% | 38.25% | 15.46% |
| Fashion | all-base / test | 519 | **86.50%** | 76.21% | 17.43% |
| Fashion | sample-base / test | 324 | 54.00% | 36.99% | 15.42% |
| GloVe | BFS / all-base | 6,952 | 58.74% | 41.58% | 26.88% |
| GloVe | BFS / sample-base | 5,310 | 44.86% | 28.92% | 25.72% |
| GloVe | BFS / test | 4,945 | 41.78% | 26.41% | 26.19% |
| GloVe | all-base / sample-base | 6,629 | 56.01% | 38.90% | 26.64% |
| GloVe | all-base / test | 6,101 | 51.55% | 34.72% | 27.51% |
| GloVe | sample-base / test | 4,894 | 41.35% | 26.06% | 26.11% |
| Last.fm | BFS / all-base | 511 | 17.48% | 9.57% | 11.22% |
| Last.fm | BFS / sample-base | 385 | 13.17% | 7.05% | 10.79% |
| Last.fm | BFS / test | 164 | **5.61%** | 2.89% | 11.40% |
| Last.fm | all-base / sample-base | 1,606 | 54.92% | 37.86% | 13.27% |
| Last.fm | all-base / test | 327 | 11.18% | 5.92% | 14.32% |
| Last.fm | sample-base / test | 247 | 8.45% | 4.41% | 12.93% |

### 3.1 如何理解“少量交集、很高拓展占比”

以 Last.fm 的 BFS/test 为例，交集只有 164 个节点，占缓存容量 `5.61%`，却承载测试总拓展的 `11.40%`，几乎覆盖 BFS 全部 `11.97%` 命中中的价值。SIFT1M 和 Fashion 也有相同现象：BFS 与其他来源的交集很小，但交集贡献了 BFS 命中价值的约 88% 到 92%。

这表明静态缓存节点至少应区分两层：

- **共享导航核心**：medoid、入口附近节点和跨 query pattern 都会经过的高中心性节点。容量很小，但单位节点收益极高。
- **负载特定热点**：由 query 分布决定，决定缓存从约 10% 命中继续提升到 15%、20% 甚至 45% 的能力。

因此未来方案不应在“BFS 或 query 热点”之间二选一，更合理的结构是保留小型共享核心，再把剩余容量分配给可迁移、可更新的 query-pattern 热点。

## 4. 缓存命中率、I/O 与 QPS

| 数据集 | 来源 | 命中率 | Mean I/O | 相对 BFS I/O | 中位 QPS | 相对 BFS QPS |
|---|---|---:|---:|---:|---:|---:|
| SIFT1M | BFS | 8.40% | 104.50 | - | 348.01 | - |
| SIFT1M | all-base | 13.03% | 99.21 | -5.06% | 349.82 | +0.52% |
| SIFT1M | sample-base | 11.94% | 100.46 | -3.87% | 348.70 | +0.20% |
| SIFT1M | test oracle | **14.54%** | **97.49** | **-6.71%** | **352.94** | **+1.42%** |
| Fashion | BFS | 9.85% | 56.80 | - | 623.93 | - |
| Fashion | all-base | 17.96% | 51.69 | -9.00% | 661.75 | +6.06% |
| Fashion | sample-base | 16.52% | 52.60 | -7.39% | 663.79 | +6.39% |
| Fashion | test oracle | **18.07%** | **51.62** | **-9.12%** | **663.84** | **+6.40%** |
| GloVe | BFS | 27.44% | 24.37 | - | 1,317.45 | - |
| GloVe | all-base | 29.34% | 23.74 | -2.59% | **1,339.16** | **+1.65%** |
| GloVe | sample-base | 27.50% | 24.35 | -0.08% | 1,302.12 | -1.16% |
| GloVe | test oracle | **32.21%** | **22.77** | **-6.57%** | 1,316.51 | -0.07% |
| Last.fm | BFS | 11.97% | 40.91 | - | 776.81 | - |
| Last.fm | all-base | 15.38% | 39.33 | -3.86% | 797.07 | +2.61% |
| Last.fm | sample-base | 13.71% | 40.10 | -1.98% | 772.41 | -0.57% |
| Last.fm | test oracle | **44.55%** | **25.77** | **-37.01%** | **971.88** | **+25.11%** |

### 4.1 各数据集解释

**SIFT1M**：all-base 是可部署来源中最好方案，但命中率相对 BFS 只提高 4.63 个百分点，QPS 仅提高 0.52%。逻辑 I/O 已下降 5.06%，但当前机器上的热 page cache 弱化了物理 I/O 收益。

**Fashion-MNIST**：all-base 与 test pattern 高度一致，all-base 基本达到 oracle；即使只使用 1% base sample，也保留了大部分收益。这是最支持“离线 base-derived 静态热点”方案的数据集。

**GloVe-25**：BFS 本身已经达到 27.44%，all-base 只小幅提高到 29.34%。sample-base 与 BFS 几乎相同，说明 1% 采样不足以稳定提取超出共享导航核心之外的热点。oracle 虽达到 32.21%，但在热 page cache 下未转化为 QPS 提升。

**Last.fm**：test oracle 远高于所有无泄漏来源，证明测试 query 存在强而集中的特定 pattern；然而 all-base 和 sample-base 无法预测这一 pattern。若真实线上 workload 与 base 分布同样失配，依赖 base 预训练的静态缓存会错过主要收益，必须使用近期真实请求进行在线或周期性更新。

## 5. 对缓存设计的启示

本实验最重要的启示不是“用 test 热点替代 BFS”，因为 test oracle 存在未来信息泄漏；真正的启示是：

1. **共享核心与负载特定容量应分离。** 少量 BFS/全局高价值节点应常驻，剩余容量根据 query pattern 分配。
2. **训练来源必须经过迁移性验证。** Fashion 证明 base 可以有效预测 test，Last.fm 则证明这种假设可能完全失败。缓存初始化需要明确的 validation workload，而不能只凭数据集名称或向量空间相似性推断。
3. **1% base sample 可降低画像成本，但不是普遍可靠。** Fashion 中接近 all-base，SIFT 和 Last.fm 有一定损失，GloVe 基本退化到 BFS。
4. **缓存优化目标应直接使用 expected saved I/O，而不是集合重合。** 节点频率、单次节点读取成本、跨 pattern 复用和容量竞争应共同进入评分；低重合并不代表低价值，高重合也不保证 QPS 提升。
5. **静态 oracle 与 Belady 上界回答不同问题。** test-hot oracle 选择全局最频繁节点，是该固定测试分布下的最佳等容量静态集合；Belady 使用未来访问序列进行逐访问替换，是动态缓存的离线上界。两者之间的差距可以量化“仅利用稳定空间 pattern”与“同时利用精确时间顺序”之间的剩余空间。

## 6. 可复现性与数据溯源

运行命令：

```bash
REPETITIONS=3 experiments/cache_eval/scripts/run_static_cache_source_experiments.sh all
```

关键产物：

- `experiments/cache_eval/results/static_cache_sources/experiment_config.json`：commit、机器、重复次数和顺序。
- `experiments/cache_eval/results/static_cache_sources/run_manifest.csv`：48 次运行及原始日志路径。
- `experiments/cache_eval/results/static_cache_sources/summary/benchmark_repetitions.csv`：逐次原始指标。
- `experiments/cache_eval/results/static_cache_sources/summary/benchmark_summary.csv`：本文使用的三次中位数。
- `experiments/cache_eval/results/static_cache_sources/<dataset>/sources/manifest.json`：图索引和三类 expansion-count 输入文件的 SHA-256。
- `experiments/cache_eval/results/static_cache_sources/<dataset>/sources/*.nodes.csv`：实际装入缓存的节点 ID，顺序和来源计数。
- `experiments/cache_eval/results/static_cache_sources/<dataset>/sources/pairwise_overlap.csv`：所有两两重合和交集拓展占比。
- `experiments/cache_eval/results/static_cache_sources/<dataset>/benchmarks/*.log`：DiskANN 原始输出。

测试与验证：

```bash
cargo fmt --all -- --check
cargo check -p diskann-tools --bins
cargo test -p diskann-tools --bin search_disk_index
python -m py_compile \
  experiments/cache_eval/scripts/analyze_static_cache_sources.py \
  experiments/cache_eval/scripts/summarize_static_cache_source_benchmarks.py
bash -n experiments/cache_eval/scripts/run_static_cache_source_experiments.sh
```

CLI 单元测试 5/5 通过。`disk_vertex_provider_factory` 模块的磁盘装载测试依赖仓库外的 VFS test data；当前环境缺少该数据，原有 7 个测试和新增的显式装载测试都会报 `PATH NOT FILLED BY VFS LAYER`。显式节点装载路径已由四个真实索引、48 次完整搜索覆盖，重复 ID/越界 ID 的纯校验测试通过。

## 7. 限制

- 未清空 Linux page cache，也未绑定 CPU/NUMA；因此 QPS 是当前机器环境下的相对结果，不应直接外推为裸盘吞吐。
- test oracle 使用完整测试负载统计，只表示最佳静态频率缓存上界，不是无泄漏部署方案。
- 当前只比较 1% 容量。不同容量下，共享核心和负载特定热点的相对价值可能变化，需要后续做容量曲线。
- base sample 只使用 seed 42 的一次 1% 样本；应增加采样比例和多 seed 方差分析。
- 热点计数来自固定搜索参数。改变 L、beam width、图构建参数或 PQ 精度会改变访问路径，也可能改变节点排序。
