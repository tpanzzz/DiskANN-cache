# DiskANN 图度数、节点扩展热度与 base-query 预测能力实验报告

日期：2026-07-24

## 1. 实验问题

本轮实验针对 SIFT1M、Fashion-MNIST、GloVe-25 和 Last.fm ANN 四个数据集，回答以下问题：

1. DiskANN 图节点的度数如何分布？
2. 官方测试 query 搜索中扩展次数位于前 1%/5% 的节点是否具有更高的度数？
3. 使用全部 base 向量作为 query 时，节点扩展热度如何分布？由此得到的 top 1%/5% 热节点与测试 query 热节点有多少重合？
4. 仅使用固定 seed 随机采样的 1% base 向量时，能否近似全部 base query 或预测测试 query 的热节点？

实验的最终目的不是只描述图结构，而是判断静态图属性和 base-derived query workload 是否能够为缓存准入、全局热节点选择或 query-conditioned cache 提供可靠先验。

## 2. 关键定义

### 2.1 出度与入度

Rust DiskANN 磁盘图是有向图。索引节点记录中保存的是邻接表，因此用户原始问题中的“度数”首先解释为存储出度：

```text
out_degree(v) = 节点 v 的邻接表长度
```

实际解析后发现，四个图的所有节点出度都恰好等于构建参数 `R`，没有任何方差。为了避免实验一和实验二只得到水平线，本报告保留所要求的出度结果，同时补充从全部有向边反向累计得到的入度：

```text
in_degree(v) = 指向节点 v 的有向边数量
```

入度能够描述节点作为图导航汇聚点或 hub 的结构重要性。它不是用户要求的出度替代品，而是对固定出度现象的必要补充。

### 2.2 节点扩展次数

扩展计数沿用此前 spatial-locality pilot 的定义：每当一个节点 id 被提交给 DiskANN `load_vertices`，该节点的计数增加一。该事件代表搜索路径需要读取或使用该节点的全精度向量和邻接信息。

本轮新增 `DISKANN_NODE_ACCESS_COUNTS_PATH` 聚合出口，在内存原子数组中累计计数，搜索结束时只输出非零节点：

```text
node_id,expansion_count
```

这与原逐访问 JSON trace 在同一个代码位置采集，但避免为一亿级节点访问生成数十 GiB 日志。

### 2.3 top 1%/5% 热节点

对所有图节点按以下键排序：

```text
(-expansion_count, node_id)
```

然后取 `ceil(num_nodes * 1%)` 或 `ceil(num_nodes * 5%)` 个节点。`node_id` 是计数相同时的确定性 tie-break，因此所有集合都可复现且具有固定基数。

集合重合度主要报告：

```text
overlap = |A intersection B| / min(|A|, |B|)
```

本实验比较的集合大小相同，所以该值也等于交集占任一集合的比例。原始结果同时保存 Jaccard 指标。

## 3. 数据集与搜索配置

所有搜索均禁用节点缓存，`beam_width=4`，随机采样 seed 为 42，采样大小为 `ceil(1% * num_nodes)`。

| 数据集 | 节点数 | 维度 | 图构建 | 导航向量 | 搜索 L | 官方测试 Recall@10 |
|---|---:|---:|---|---|---:|---:|
| SIFT1M | 1,000,000 | 128 | R=64, Lbuild=100 | PQ16 | 100 | 97.04% |
| Fashion-MNIST | 60,000 | 784 | R=64, Lbuild=100 | PQ49 | 50 | 99.13% |
| GloVe-25 | 1,183,514 | 25 | R=96, Lbuild=200 | normalized cosine, PQ16 | 20 | 96.79% |
| Last.fm ANN | 292,385 | 65 | R=128, Lbuild=400 | exact in-memory cosine | 20 | 96.51% |

GloVe 使用单位归一化向量和 `cosinenormalized`，没有复用此前 raw cosine/PQ5、`L=1600` 的度量不一致配置。

Last.fm 的原 R=96/Lbuild=200、PQ16 图虽然在 `L=100` 达到 95.79% 测试 Recall，但 PQ cosine 导航存在 squared-L2 候选排序警告。最终实验新建了 R=128/Lbuild=400 图，并使用完整 base f32 矩阵执行 exact cosine 导航。新索引使用独立前缀，没有覆盖旧索引。

### 3.1 线程数说明

搜索使用 8 个线程，而不是 query-order/cache 实验中的 1 个线程。原因是本轮缓存完全禁用，每个 query 的搜索状态独立，最终计数是可交换求和；并发不会形成 cache 访问交错，也不会改变单 query 搜索路径。Fashion-MNIST 的 1 线程和 8 线程结果具有完全相同的 Recall、总扩展数和派生分布统计，验证了聚合计数流程的确定性。

### 3.2 base 自查询质量判据

全部 base 和 1% base sample 的精确最近邻至少包含 query 自身。最初使用 identity id 作为 Recall@1 ground truth，但 SIFT 和 Last.fm 中存在完全相同或几乎共线的向量，搜索返回等价向量时会被 identity Recall 错误计为 miss。

因此同时报告两种指标：

- identity Recall@1：返回 id 必须等于 query 对应节点 id；
- distance-equivalent Recall@1：L2 使用 squared-L2 `<= 1e-6`，cosine 使用 `1-cosine <= 1e-6`。

| 数据集 | 全部 base identity R@1 | 全部 base 等价 R@1 | 1% sample identity R@1 | 1% sample 等价 R@1 |
|---|---:|---:|---:|---:|
| SIFT1M | 98.5455% | 99.9993% | 98.6800% | 100.0000% |
| Fashion-MNIST | 99.9367% | 99.9367% | 99.8333% | 99.8333% |
| GloVe-25 | 99.9999% | 99.9999% | 100.0000% | 100.0000% |
| Last.fm ANN | 94.5120% | 99.9018% | 95.3488% | 99.8290% |

Last.fm 的 identity Recall 较低不是普通近邻搜索失败。对于 1% sample 中按 id 错误的结果，大多数返回向量与 query 的 cosine similarity 为 1 或接近 1；全部 base 的距离等价 Recall 为 99.90%。四个数据集均满足大于 95% 的距离质量门槛。

## 4. 实验一：全图节点度数分布

### 4.1 存储出度

| 数据集 | 平均出度 | 中位数 | P95 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|---:|
| SIFT1M | 64 | 64 | 64 | 64 | 64 |
| Fashion-MNIST | 64 | 64 | 64 | 64 | 64 |
| GloVe-25 | 96 | 96 | 96 | 96 | 96 |
| Last.fm ANN | 128 | 128 | 128 | 128 | 128 |

结论非常明确：在这些 Rust DiskANN 索引中，出度完全由 `R` 决定。按出度从大到小排序时，每个节点都相同，因而出度 rank 曲线是水平线。出度不能用于区分 hot node，也不能直接作为缓存 admission score。

### 4.2 补充的入度分布

| 数据集 | 平均入度 | 中位数 | P95 | P99 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| SIFT1M | 64 | 55 | 123 | 169 | 5 | 134,506 |
| Fashion-MNIST | 64 | 50 | 150 | 237 | 2 | 7,964 |
| GloVe-25 | 96 | 89 | 153 | 209 | 19 | 3,639 |
| Last.fm ANN | 128 | 105 | 284 | 443 | 1 | 28,691 |

平均入度必然等于平均出度，因为每条有向边对两个总和各贡献一次；但分布显著不同。四个图都存在长尾入度，SIFT1M 的最大入度达到 134,506，说明少数入口或导航骨干节点被大量邻接表共同引用。

该结果说明必须区分：

- `R` 控制的固定出度是存储预算；
- 长尾入度是静态图拓扑中的 hub 信号；
- query 扩展次数是拓扑与 query 分布共同作用后的负载信号。

## 5. 实验二：测试 hot 节点的度数

由于所有出度固定，测试扩展 top 1%/5% 节点的出度仍然分别为 64、64、96 和 128，没有任何富集。

入度结果如下：

| 数据集 | hot set | 平均入度 | 中位数 | P95 | 相对全图平均值 | 入度/扩展 Spearman |
|---|---|---:|---:|---:|---:|---:|
| SIFT1M | top 1% | 121.92 | 97 | 196 | 1.91x | 0.342 |
| SIFT1M | top 5% | 103.08 | 93 | 178 | 1.61x | 0.342 |
| Fashion-MNIST | top 1% | 174.25 | 116 | 446 | 2.72x | 0.674 |
| Fashion-MNIST | top 5% | 144.46 | 118 | 329 | 2.26x | 0.674 |
| GloVe-25 | top 1% | 130.99 | 121 | 231 | 1.36x | 0.161 |
| GloVe-25 | top 5% | 126.27 | 118 | 215 | 1.32x | 0.161 |
| Last.fm ANN | top 1% | 125.81 | 114 | 176 | 0.98x | -0.030 |
| Last.fm ANN | top 5% | 117.43 | 109 | 194 | 0.92x | -0.030 |

解释：

- Fashion-MNIST 中入度是较强的扩展热度先验，top 1% 热节点平均入度为全图的 2.72 倍。
- SIFT1M 有中等程度关系，说明导航 hub 能解释一部分、但不是全部热度。
- GloVe-25 关系较弱，query 相关的局部路径可能比静态 hub 更重要。
- Last.fm 是明确反例：测试热节点没有入度富集，Spearman 接近零且略为负值。仅按入度选择缓存节点会漏掉其主要 query-demand hot set。

因此，入度可以作为全局共享层候选特征，但不能成为跨数据集通用的缓存策略。

## 6. 实验三：全部 base 向量作为 queries

### 6.1 扩展分布

| 数据集 | 总扩展数 | 非零扩展节点 | Gini | top 1% 扩展占比 | top 5% 扩展占比 | 覆盖 50% 扩展的节点数 |
|---|---:|---:|---:|---:|---:|---:|
| SIFT1M | 113,957,149 | 999,999 | 0.5029 | 12.91% | 25.80% | 173,412 |
| Fashion-MNIST | 3,784,611 | 59,972 | 0.5183 | 18.01% | 29.19% | 10,165 |
| GloVe-25 | 39,400,400 | 1,183,514 | 0.5537 | 29.59% | 39.37% | 139,308 |
| Last.fm ANN | 10,967,602 | 292,358 | 0.6416 | 33.19% | 45.58% | 21,242 |

使用全部 base 向量后，几乎所有图节点都至少被访问一次，这符合 base query 覆盖整个数据空间的预期。但是访问质量仍高度不均匀，特别是 GloVe 和 Last.fm 的 top 1% 节点分别承担 29.59% 和 33.19% 的扩展。

每个 workload 的最大扩展次数等于 query 数量，因为入口 medoid/导航起点会在每次搜索中访问。原始 top hot set 因此包含一个确定的全局入口节点；本报告没有人为移除它，以保持与真实缓存访问一致。后续可增加“排除 medoid/前若干 hop”的消融实验，区分入口骨干与 query-region 热节点。

### 6.2 全部 base hot set 与测试 hot set 的重合

| 数据集 | top 1% 交集占比 | top 1% Jaccard | top 5% 交集占比 | top 5% Jaccard |
|---|---:|---:|---:|---:|
| SIFT1M | 48.56% | 32.07% | 49.56% | 32.94% |
| Fashion-MNIST | 86.50% | 76.21% | 75.10% | 60.13% |
| GloVe-25 | 51.55% | 34.72% | 30.13% | 17.74% |
| Last.fm ANN | 11.18% | 5.92% | 8.93% | 4.68% |

主要观察：

- Fashion-MNIST 的 base 和测试 query 都来自同一图像域，base-derived hot set 对测试 hot set 的预测非常强。
- SIFT1M 的重合约为一半，base workload 可以作为有价值但不完整的 cold-start prior。
- GloVe top 1% 仍有约一半重合，但扩展到 top 5% 后下降到 30.13%，说明最核心导航节点共享，而更大的区域工作集更依赖 query 分布。
- Last.fm 的重合极低。base 是 item factor，测试 query 是 user factor；语料几何与需求几何不同。使用全部 base 也不能替代历史 query。

该结果与此前 base-fitted cluster occupancy 实验一致：SIFT、Fashion-MNIST 和 GloVe 的 base geometry 可提供一定 cold-start 信息，Last.fm 则是重要反例。

## 7. 实验四：随机采样 1% base 向量

### 7.1 扩展集中度是否接近全部 base

| 数据集 | 全部 base top 1% 占比 | 1% sample top 1% 占比 | 全部 base top 5% 占比 | 1% sample top 5% 占比 |
|---|---:|---:|---:|---:|
| SIFT1M | 12.91% | 14.42% | 25.80% | 31.13% |
| Fashion-MNIST | 18.01% | 19.63% | 29.19% | 37.90% |
| GloVe-25 | 29.59% | 32.24% | 39.37% | 52.73% |
| Last.fm ANN | 33.19% | 34.70% | 45.58% | 57.03% |

1% sample 对 top 1% 扩展质量集中度的估计相当接近全部 base，误差约为 1.5 到 2.7 个百分点。但 top 5% 占比普遍高估，因为较少 query 会产生更多零计数和低计数节点，使有限样本下的分布看起来更加集中。

因此，1% sample 适合快速估计“是否存在明显热度长尾”，但不能直接当作精确 miss-ratio curve 或容量规划输入。

### 7.2 1% sample hot set 与测试 hot set 的重合

| 数据集 | top 1% 交集占比 | top 1% Jaccard | top 5% 交集占比 | top 5% Jaccard |
|---|---:|---:|---:|---:|
| SIFT1M | 34.15% | 20.59% | 35.84% | 21.83% |
| Fashion-MNIST | 54.00% | 36.99% | 36.33% | 22.20% |
| GloVe-25 | 41.35% | 26.06% | 23.91% | 13.58% |
| Last.fm ANN | 8.45% | 4.41% | 8.17% | 4.26% |

与全部 base 相比，1% sample 对测试 hot set 的节点级预测明显下降。它仍能捕捉一些非常稳定的全局导航节点，但不足以恢复完整 hot set。

这里必须区分两个问题：

- 分布形状：1% sample 可以较好估计 top 1% 节点承担多少扩展；
- 节点身份：1% sample 对“具体哪些节点属于测试 hot set”的预测能力有限，并且强烈依赖 base/query 分布是否一致。

## 8. 测试 query 扩展集中度

为便于与此前实验比较，官方测试 query 的主要扩展统计如下：

| 数据集 | 总扩展数 | 非零扩展节点占图比例 | Gini | top 1% 扩展占比 | top 5% 扩展占比 |
|---|---:|---:|---:|---:|---:|
| SIFT1M | 1,140,773 | 52.51% | 0.6788 | 14.54% | 31.49% |
| Fashion-MNIST | 630,043 | 91.68% | 0.5492 | 18.07% | 29.92% |
| GloVe-25 normalized | 335,924 | 18.15% | 0.8802 | 32.21% | 53.69% |
| Last.fm ANN | 2,323,649 | 21.28% | 0.9480 | 44.55% | 82.82% |

GloVe 与此前 raw PQ5、`L=1600` trace 的结果不可直接比较。修正为 normalized PQ16、`L=20` 后，搜索不再遍历大部分图，测试扩展呈现明显更强的集中度。这再次说明扩展统计必须在正确度量和固定 Recall 下解释。

Last.fm 的测试 top 5% 节点承担 82.82% 扩展，是四个数据集中最强的全局热度。但这些节点与 base-derived hot set 重合很低，说明其热度来自 user-query demand，而不是 item base geometry 或简单入度 hub。

## 9. 对缓存策略的含义

### 9.1 不应使用出度作为 hot-node proxy

在当前 Rust DiskANN 图中，出度只是构建参数 `R`。任何基于出度的准入、淘汰或静态缓存排序都会退化为全体节点同分。

### 9.2 入度可以帮助构建全局层，但必须校准

Fashion-MNIST 和 SIFT1M 表明，高入度节点更可能频繁扩展。可以将入度、是否为 medoid、早期 hop 频率加入全局共享层评分。

但 GloVe 关系较弱，Last.fm 完全不支持该假设。入度应当只是 QASC utility model 的一个特征，不能替代实际 query trace 或 node-query-cluster affinity。

### 9.3 base-derived workload 只适合作为经过验证的冷启动先验

Fashion-MNIST 的结果说明，当 base 与 query 来自一致分布时，全部 base 搜索能够很好地预测测试 hot set。SIFT 和 GloVe 只能提供中等强度先验，Last.fm 则几乎无效。

因此推荐的 prototype/hot-set 数据源顺序仍然是：

1. 保留到达顺序的历史生产 query；
2. 代表性 learn-query 集；
3. 检索模型训练 query；
4. base sample，仅在通过 occupancy 和 hot-set overlap 验证后作为 cold-start fallback。

### 9.4 1% base sample 适合筛选，不适合最终节点选择

1% sample 能快速判断访问分布是否长尾，也适合比较多个图参数或数据集；但其 exact hot-set overlap 明显低于全部 base，不能直接据此固定静态缓存内容。更合理的用法是初始化统计，然后用线上 query 更新或替换。

## 10. 此次实验带来的最大启发

本轮最大的启发是：**图结构中的“中心性”和 query workload 中的“热度”是两个不同层次的对象。**

更具体地说：

- 出度在 DiskANN 中被 `R` 固定，根本不包含区分信息；
- 入度确实揭示静态图 hub，但它对扩展热度的解释能力从 Spearman 0.674 到 -0.030，不能跨数据集泛化；
- base-query hot set 的预测能力从 Fashion-MNIST top 1% 的 86.5% 重合到 Last.fm 的 11.18%，由 query/base 分布关系决定；
- 真实需要缓存的节点不是“图上最中心的节点”本身，而是“在当前 query 分布和搜索配置下具有最高未来重用概率的节点”。

这进一步支持 query-affinity-aware cache 的方向：保留一个小型全局导航层处理 medoid、早期 hop 和稳定 hub，同时通过历史 query、query cluster 和节点-聚类扩展亲和度学习剩余容量。静态入度和 base sample 可以作为冷启动特征，但不能成为最终策略的核心监督信号。

## 11. 局限与后续实验

- 本轮 1% sample 仅使用 seed 42。论文结果应增加多个 seed，并报告 overlap 和集中度的置信区间。
- top set 采用固定大小和 node-id tie-break。有限 query 下大量低频节点可能并列，top 5% 比 top 1% 更容易受 tie 影响。
- 当前计数记录 `load_vertices` 节点请求，不包含 query id、hop 或页面 id。后续应加入 query boundary 和 hop，区分 medoid/早期导航与后期区域节点。
- 本轮没有移除每次搜索都会访问的 medoid。建议补充排除 medoid、排除前 1/2/3 hop 的 hot-set overlap。
- 当前使用节点级计数。SSD 缓存收益最终还应按 4 KiB block 聚合，分析入度、扩展热度与 page reuse 的关系。
- Last.fm 为满足严格且度量正确的测试 Recall，使用新建 R=128/Lbuild=400 图和 exact cosine 导航；跨数据集绝对扩展数不应被解释为仅由数据分布造成。
- 并发不影响本轮无缓存的路径计数，但 QPS 和延迟没有做隔离重复，本报告不将性能数字作为主要结论。

建议下一步优先完成：

1. 从 trace 中记录 `query_id`、query cluster、hop 和 page id。
2. 将 hot set 分解为全局入口/高入度 hub 与 query-cluster-specific residual。
3. 对 base sample 比例执行 0.1%、0.5%、1%、5%、10% sweep，并使用多个 seed。
4. 计算 sample size 与测试 hot-set overlap 的学习曲线。
5. 对 Last.fm 使用 user-factor 历史 query 或 learn query，而不是 item base，验证 demand-aligned source 是否显著提高 overlap。
6. 将入度、早期 hop 频率、query-cluster affinity 分别加入 QASC admission score，进行消融。

## 12. 数据溯源与产物

统一结果根目录：

```text
experiments/cache_eval/results/graph_hotness_degree/
```

每个数据集目录包含：

```text
workloads/workload_manifest.json
search/*.search.log
search/*.node_expansion_counts.csv
self_query_quality.json
analysis/summary.json
analysis/provenance.json
analysis/node_degrees_by_id.npz
analysis/*top_1.csv
analysis/*top_5.csv
analysis/experiment1_graph_degree_rank.png
analysis/experiment2_test_hot_node_degree_rank.png
analysis/experiments3_4_expansion_rank.png
analysis/experiments3_4_expansion_mass.png
analysis/experiments3_4_hot_set_overlap.png
```

跨数据集汇总：

```text
experiments/cache_eval/results/graph_hotness_degree/summary/dataset_summary.csv
experiments/cache_eval/results/graph_hotness_degree/summary/degree_summary.csv
experiments/cache_eval/results/graph_hotness_degree/summary/expansion_summary.csv
experiments/cache_eval/results/graph_hotness_degree/summary/hot_set_degree_summary.csv
experiments/cache_eval/results/graph_hotness_degree/summary/hot_set_overlap.csv
experiments/cache_eval/results/graph_hotness_degree/summary/source_files_sha256.csv
experiments/cache_eval/results/graph_hotness_degree/summary/test_expansion_concentration.png
experiments/cache_eval/results/graph_hotness_degree/summary/hot_set_overlap_comparison.png
experiments/cache_eval/results/graph_hotness_degree/summary/test_hot_in_degree_enrichment.png
```

`provenance.json` 保存索引、base/query/ground truth、聚合计数和配置文件的绝对路径、字节数与 SHA-256。`source_files_sha256.csv` 保存搜索二进制、Rust 采集代码和所有分析脚本的 SHA-256，可以将关键结果追溯到本轮实际执行的代码状态。

## 13. 复现命令

运行单个数据集：

```bash
experiments/cache_eval/scripts/run_graph_hotness_degree_experiments.sh sift1m
experiments/cache_eval/scripts/run_graph_hotness_degree_experiments.sh fashion_mnist
experiments/cache_eval/scripts/run_graph_hotness_degree_experiments.sh glove_25
experiments/cache_eval/scripts/run_graph_hotness_degree_experiments.sh lastfm_64
```

运行全部数据集：

```bash
NUM_THREADS=8 SEED=42 SAMPLE_FRACTION=0.01 \
  experiments/cache_eval/scripts/run_graph_hotness_degree_experiments.sh all
```

重新生成跨数据集汇总：

```bash
experiments/cache_eval/.venv/bin/python \
  experiments/cache_eval/scripts/summarize_graph_hotness_degree_experiments.py \
  --repo-root . \
  --root experiments/cache_eval/results/graph_hotness_degree \
  --output-dir experiments/cache_eval/results/graph_hotness_degree/summary
```
