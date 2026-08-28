# DiskANN 图向量检索缓存与局部性研究：实验时间线、综合发现与论文路线图

日期：2026-08-28

## 1. 文档目的

本文汇总 `DiskANN/experiments/cache_eval/` 下截至 2026-08-28 的实验、报告、结果摘要和相关 Codex 会话，目标是回答四个问题：

1. 这项研究从最初的“比较缓存替换策略”如何演进到“理解 query 空间、图路径与缓存重用之间的关系”？
2. 已有哪些结论得到多组实验支持，哪些结论只在单一数据集或人工 workload 上成立？
3. 哪些旧结果已被后续正确性修复、公平性修复或距离度量修复取代，论文中应如何引用？
4. 如何把现有结果组织成一篇有明确研究问题、因果链和负面结果价值的论文？

本文不是新的性能实验。文中的数字来自已有 CSV、JSON、日志和报告，时间以 Git commit、报告日期和结果生成时间交叉确定。

## 2. 信息来源与证据分级

主要信息来源包括：

- 当前分支中的实验报告和结果摘要；
- `git log --all -- experiments/cache_eval` 中的提交时间；
- 已归档分支中的缓存策略正确性与 QPS review；
- `~/.codex/sessions/` 中 2026-06-18 至 2026-07-27 的相关会话记录；
- 原始结果 CSV、run manifest、实验配置、输入 SHA-256 和搜索日志。

为避免把不同成熟度的结果混为一谈，本文使用以下证据等级：

| 等级 | 含义 | 典型材料 |
|---|---|---|
| A | 配置公平、结果可追溯、完整性检查通过，并有重复实验或大规模矩阵 | 2026-07-27 静态节点来源实验；2026-07-06 fair split 的完整矩阵，但后者每格仅一次 |
| B | 配置和数据可追溯、质量门槛明确，但每个配置通常只测一次 | query workload、Belady replay、并发 sweep、跨数据集热点分析、GloVe PQ sweep |
| C | 实现 smoke test、设计验证或带乐观信息来源的初步结果 | QASC v1；test-hot 静态 oracle |
| D | 已被后续修复或更正，仅用于解释研究演进 | 旧 2Q/SLRU 结果、20k hybrid 的“公平”解释、raw GloVe PQ5 热点分布 |

论文主表应优先使用 A/B 级结果；C 级结果适合写成 feasibility 或 negative result；D 级结果只能出现在方法演进、威胁或附录中。

## 3. 一页式研究结论

截至目前，最稳固的综合判断是：

> 磁盘图 ANNS 的跨 query 节点重用不是单一的传统时间局部性，而是由稳定的全局导航核心、query 空间区域决定的路径重叠、以及 query 区域到达顺序共同生成。缓存策略只有同时控制系统开销、短期 recency 和 query-conditioned reuse，才可能稳定优于 DiskANN 的静态 BFS 缓存。

这一判断由以下结果共同支持：

1. **StaticBFS 是强系统基线。** SIFT1M 原始顺序、1 线程、10k 节点缓存下，StaticBFS 命中约 `8.56%`，QPS 比 NoCache 高约 `8.6%`；许多动态策略命中略高，却因锁、元数据更新、线性队列操作、payload clone 和 admission 成本而更慢。
2. **人工增强 query 空间连续性会显著增加真实可缓存重用。** SIFT1M 的 LRU 命中率由原始顺序 `9.21%` 提升到 k-means 连续顺序 `37.09%`；Belady 上界同时由 `24.70%` 提升到 `47.12%`。这不是只让 LRU 更聪明，而是改变到达顺序后访问序列本身产生了更多短距离重用。
3. **标准 ANN benchmark 的原始 query 行顺序通常没有生产时间语义。** Fashion、GloVe、Last.fm 的 lag-1 cluster locality 基本等于或低于 shuffled baseline；SIFT 文件顺序有 `1.7483x` lift，但没有 request timestamp，不能解释为真实线上局部性。
4. **所有数据集都存在热点，但热点结构差异很大。** 在修正后的高 recall 配置下，测试 query 的 top 1% 节点承担 SIFT `14.54%`、Fashion `18.07%`、GloVe `32.21%`、Last.fm `44.55%` 的拓展；Last.fm top 5% 更承担 `82.82%`。
5. **图中心性不等于 workload 热度。** 当前 DiskANN 图出度几乎由 `R` 固定，不能排序热点；入度与扩展热度的 Spearman 从 Fashion 的 `0.674` 到 Last.fm 的 `-0.030`，不能作为跨数据集通用 proxy。
6. **base geometry 只能是经过验证的 cold-start prior。** base-fitted cluster occupancy 对 SIFT、Fashion、GloVe 的 query 分布相关性很高，但 Last.fm 完全失配；all-base/test top 1% 热点重合从 Fashion 的 `86.50%` 到 Last.fm 的 `11.18%`。
7. **最优静态节点来源也是 workload dependent。** all-base 热点通常优于 BFS，但改善范围从 GloVe 的 `+1.90` 个命中百分点到 Fashion 的 `+8.11` 个百分点；Last.fm test oracle 可达 `44.55%`，而 all-base 只有 `15.38%`。
8. **空间信息不能替代 recency。** QASC v1 在原始顺序比 LRU 多 `0.45` 个命中百分点但 QPS 低 `24.6%`；在 k-means 连续 workload 上仅 `12.90%` 命中，远低于 LRU 的 `37.09%`。长期 cluster quota 阻碍了缓存快速集中到当前 burst。
9. **并发可扩展性首先是数据结构与锁模型问题。** 32 线程下，sharded CLOCK 相对 global CLOCK 的 QPS 提升为 `41.45%` 到 `69.76%`，命中率和 I/O 基本不变，直接证明全局 mutex 是主要瓶颈。
10. **缓存命中、逻辑 I/O 和 QPS 必须分别报告。** GloVe test-hot oracle 相对 BFS 降低 `6.57%` Mean I/O，却没有提高中位 QPS；OS page cache、CPU 成本和缓存查找路径会改变命中收益到端到端吞吐的转化率。

## 4. 实验时间线总览

| 日期 | 阶段 | 主要问题 | 关键产物 | 研究推进 |
|---|---|---|---|---|
| 2026-06-18 | 缓存评估基础设施与首轮 SIFT1M | 动态策略能否接入 DiskANN，命中和 I/O 是否可测？ | cache harness、live search、trace replay、首轮 CSV | 建立研究平台，发现 StaticBFS 很强、hit 不等于 QPS |
| 2026-06-21 | StaticBFS warmup | 图结构先验能否帮助所有动态策略？ | `warm_bfs_cache_policy_summary.csv` | 得到重要负面结果：错误状态初始化会伤害 ARC 等历史型策略 |
| 2026-06-30 | 正确性/QPS 审计与 query 顺序修复 | 策略实现是否忠于原算法，为什么动态策略慢？ | 两份 archived review、correctness fixes、1-thread rerun | 降低旧结果证据等级，定位全局锁和 O(C) 元数据成本 |
| 2026-06-30 至 07-03 | query workload 构造与 Belady | query 几何相近且连续时，缓存收益是否变化？ | random/k-means order、三 workload CSV、Belady replay | 从传统时间局部性转向 query-space-induced path reuse |
| 2026-07-03 | 并发缓存策略完整 sweep | 多线程下如何避免动态缓存锁竞争？ | 180-run concurrency sweep | 证明 sharding 是主要并发优化；发现 20k hybrid 容量 caveat |
| 2026-07-06 | 公平 hybrid split sweep | equal-capacity 下 static/dynamic 如何分配？ | 252-run fair split sweep | 强 locality 需要更多动态容量；弱 locality 中 StaticBFS 仍强 |
| 2026-07-17 | 数据集和 query 顺序调研 | benchmark query 顺序是否具有真实时间意义？ | dataset survey、4 数据集 K=100 pattern analysis | 得到负面结论：标准 test order 不是生产 trace |
| 2026-07-17 | 跨数据集 expansion pilot | 高 recall 下节点访问是否集中？ | expansion traces、Gini/top share | 证明热点普遍存在但形态不统一，支持 global + conditional 结构 |
| 2026-07-17 | GloVe cosine/PQ 修正 | L=1600 是图差还是 PQ/metric 问题？ | PQ5/10/16/25 + full precision sweep | 发现 raw cosine 与 L2 PQ 导航不一致，重置 GloVe 正确配置 |
| 2026-07-17 至 07-21 | 空间局部性感知缓存设计 | 如何处理跨 cluster 共享节点和容量分配？ | QASC design、dataset survey、中文翻译 | 从独立 cluster cache 转为去重 shared cache + logical quota |
| 2026-07-24 | 图度数、base/sample/test 热点 | 图结构、base 和 test 热点是否一致？ | 四数据集 graph/hotness report | 证明中心性、语料几何和需求热度是不同对象 |
| 2026-07-27 | QASC v1 实现与验证 | query context 是否能落地，v1 是否优于 LRU？ | QASC core、integration tests、SIFT smoke | 接口成功、策略失败；确立“空间增强 recency”方向 |
| 2026-07-27 | 四类静态节点来源实测 | BFS、all-base、sample-base、test-hot 谁更好？ | 48 runs、3 repetitions、provenance | 验证共享导航核心 + workload-specific residual 的两层结构 |

### 4.1 Git 时间锚点

| Commit | 时间 | 作用 |
|---|---|---|
| `b720c463` | 2026-06-18 | 建立 cache policy evaluation harness |
| `a4641157` | 2026-06-21 | 将动态缓存 warmup 接入策略状态 |
| `4842dbb4` | 2026-06-30 | 修复策略正确性和 query ordering，并保存两份 review |
| `08026e1c` / `b3ead6b1` | 2026-07-03 | 并发策略实现与结果报告 |
| `c918815b` | 2026-07-06 | equal-capacity fair hybrid split |
| `6ea99ce3` | 2026-07-17 | 跨数据集空间模式和 expansion workflow |
| `91788fd8` | 2026-07-17 | GloVe metric/PQ/full-precision 修正实验 |
| `2bceb4d6` | 2026-07-27 提交，报告实验日期 2026-07-24 | 图度数和 expansion hot-set 分析 |
| `0baccd7d` 至 `47b80b0b` | 2026-07-27 | QASC 设计、实现、修正、测试和验证报告 |
| `50b05cca` 至 `7a4c857b` | 2026-07-27 | 四类静态节点来源实现、实验和报告 |

query workload/Belady 的 CSV 生成于 2026-07-03，相关工具在后续 spatial-locality 分支中正式归档。因此“结果生成时间”和“工具提交时间”并不完全相同。检查所有本地与远端 Git refs 后，当前可见的 `cache_eval` 实验提交止于 2026-07-27；2026-08-28 之前没有发现尚未纳入本文的后续 cache-eval 结果分支。

## 5. 分阶段实验梳理

### 5.1 2026-06-18：建立缓存评估平台和首轮 SIFT1M 基线

#### 研究问题

- DiskANN 原生缓存到底是什么？
- 是否可以在不改变图搜索语义的前提下接入 FIFO、LRU、LFU、TinyLFU 等动态缓存？
- trace replay 的命中率是否能与 live search 对齐？

#### 实验设置

SIFT1M，`L=100`，`beam_width=4`，cache capacity 10,000 nodes，Recall@10 `97.04%`。首轮实现同时提供 NoCache、count-limited medoid BFS 静态缓存、动态策略和 trace replay。

#### 代表性结果

| 策略 | QPS | Mean I/O | Hit | Recall@10 |
|---|---:|---:|---:|---:|
| NoCache | 约 322 | 114.08 | 0.00% | 97.04% |
| StaticBFS | 约 350 | 104.32 | 8.56% | 97.04% |
| LRU | 约 329 | 103.58 | 9.21% | 97.04% |
| LFU | 约 290 | 102.57 | 10.08% | 97.04% |
| CLOCK | 约 335 | 103.78 | 9.03% | 97.04% |

#### 启示

1. 原生 Rust DiskANN 的 StaticBFS 是从 medoid 开始按 BFS 顺序取前 N 个节点，不是严格按完整 hop boundary 截止。
2. 缓存只改变节点 payload 的来源，不改变搜索路径和 recall；各策略 Recall 一致验证了这一点。
3. StaticBFS 的优势不仅是选点，还包括查询期只读、无替换元数据、无全局动态策略维护。
4. 初始研究假设“更先进策略命中更高，因此系统更快”不成立。必须把策略机制收益和工程路径成本分开。

#### 对后续的影响

首轮结果直接引出两个问题：如何预热动态缓存，以及为什么高 hit 动态策略 QPS 更低。

### 5.2 2026-06-21：StaticBFS warmup 并不普遍有效

#### 研究问题

把 StaticBFS 节点作为动态缓存初始 resident，能否结合图结构先验和在线适应？

#### 代表性结果

- LRU cold/warm hit 都约为 `9.21%`，warmup 几乎不改变稳态结果。
- ARC 从 cold `8.38%` 降到 warm `7.92%`。
- W-TinyLFU 从 `10.36%` 轻微降到 `10.34%`。
- 2Q、SLRU、Cacheus 在旧实现中只有很小改善。

#### 启示

StaticBFS warmup 是“图结构中心性先验”，而 ARC、TinyLFU、W-TinyLFU 等需要的是“与真实 query 访问序列一致的策略状态”。把一批 BFS 节点按一次性 admission 填满缓存，会产生：

- ARC 的满 `T1`、空 `T2/B1/B2` 和非真实 ghost history；
- TinyLFU payload 与 frequency sketch 训练不一致；
- 动态策略从真实 query 流自然学习的空间被预填节点占据。

#### 论文价值

这是一个可保留的负面结果：**结构 warmup 不是 history-aware policy 的通用 warmup。** 但旧 warmup CSV 涉及后续已修复的 2Q/SLRU/TinyLFU 细节，论文应只引用机制结论，不能把全部旧数字当作最终策略对比。

### 5.3 2026-06-30：正确性审计和系统开销审计

#### 正确性发现

对所有策略逐一对照论文或作者实现后，策略被分为 Exact、Mostly consistent、Approximate 和 inspired variants。重要结论包括：

- FIFO、LRU、NoCache、Belady replay 的语义较明确；
- LFU 是无 aging 的 naive LFU；
- 2Q/SLRU 的 segment 容量维护被修复，修复后一次性序列可能只占 probation/protected 的部分容量；
- LIRS、ARC、GDSF、TinyLFU/W-TinyLFU、Cacheus 存在参数固定或机制简化；
- Cacheus 只能称为 Cacheus-inspired，不能直接用 FAST'21 原论文结论解释；
- Belady 只适用于等大小对象的离线 trace replay，不能用于 live QPS。

#### QPS 发现

动态策略 hit 提升未转化为 QPS 的主要代码原因是：

1. 全局 `Mutex<DynamicNodeCache>`；
2. 每个 vertex 分别 lock/unlock；
3. `VecDeque` 删除为 O(C)；
4. dynamic hit clone vector 和 adjacency payload；
5. admission/eviction 需要额外 HashMap remove/insert；
6. LFU/GDSF/LeCaR/Cacheus lazy heap 膨胀；
7. TinyLFU 每次访问更新 sketch/doorkeeper；
8. trace writer 可能成为隐藏开销。

#### 参数压力实验

在 `top-k=100`、`L=1000`、目标 recall 约 `99.31%` 的深搜索中：

- StaticBFS hit 仅 `1.40%`；
- LRU/CLOCK/TinyLFU hit 约 `3.82%` 到 `3.87%`；
- W-TinyLFU/2Q/LIRS hit 约 `4.37%` 到 `4.86%`；
- 但复杂策略 QPS 仍显著低于简单策略。

这说明搜索更深时，固定入口附近的 10k BFS 集合被更广的访问工作集稀释，动态策略有更多命中空间；但如果策略 CPU 成本过高，I/O 减少仍不一定改善吞吐。

#### 对后续的影响

研究由“继续堆叠替换算法”转为两条更有价值的路线：

- 构造可控 query workload，理解访问序列为什么可缓存；
- 为简单策略设计统一、可扩展的并发 backend。

### 5.4 2026-06-30 至 2026-07-03：query 顺序、k-means workload 与 Belady 上界

#### 研究问题

如果向量空间中相近的 query 连续到达，它们的图搜索路径是否更容易重用？

#### 方法

新增可复现的 query workload 工具：

- original：原始 SIFT test 行顺序；
- random seed 42：固定 seed 的随机排列；
- k-means K=100：让相似 query 和相邻 cluster 尽量连续。

query 和 ground truth 使用同一 permutation，所有实验 `num_threads=1`，避免线程调度重排到达顺序。

#### live search 结果

| workload | 策略 | Hit | Mean I/O | QPS |
|---|---|---:|---:|---:|
| original | StaticBFS | 8.56% | 104.32 | 347.11 |
| original | LRU | 9.21% | 103.58 | 324.30 |
| random | LRU | 7.91% | 105.06 | 325.25 |
| k-means grouped | FIFO | 36.45% | 72.50 | 381.84 |
| k-means grouped | LRU | 37.09% | 71.77 | 358.25 |
| k-means grouped | CLOCK | 36.98% | 71.89 | 383.62 |

LFU 是关键反例：k-means grouped 下 hit 只有 `6.66%`。没有 aging 的全局频率会粘住先前 cluster 的热点，无法快速切换到当前 cluster。TinyLFU admission 也只达到 `12.73%`，说明选择性准入会拒绝当前 burst 中即将很快复用、但历史频率尚低的节点。

#### Belady 上界

| 容量 | Original LRU / OPT | Random LRU / OPT | K-means LRU / OPT |
|---:|---:|---:|---:|
| 1,000 | 4.42% / 12.72% | 3.77% / 11.60% | 18.98% / 34.64% |
| 5,000 | 7.50% / 20.01% | 6.30% / 18.87% | 31.55% / 43.62% |
| 10,000 | 9.21% / 24.70% | 7.91% / 23.57% | 37.09% / 47.12% |
| 20,000 | 11.70% / 30.62% | 10.24% / 29.65% | 40.48% / 50.76% |

#### 核心启示

1. K-means 排序同时提高 LRU 和 OPT，说明 query 几何连续性增加了访问序列本身的可缓存性。
2. 原始顺序 LRU 到 OPT 仍有 `15.49` 个百分点，随机顺序有 `15.66` 个百分点，说明当前在线策略没有利用所有可预测结构。
3. K-means 下 gap 降到 `10.03` 个百分点，recency 已利用大部分短期 burst，但仍有跨更长间隔或共享全局节点可利用。
4. 这一实验建立的是受控因果干预，不代表 benchmark 原始顺序具有真实时间语义。

### 5.5 2026-07-03：并发缓存策略优化

#### 方法

3 workloads x 6 thread counts x 10 policies，共 180 次运行；线程数为 1、2、4、8、16、32。实现包括 sharded payload/policy backend、O(1) LRU metadata、sharded CLOCK/FIFO/LRU、TinyLFU admission 和 hybrid StaticBFS + dynamic tier。

#### 关键结果

- 32 线程下 sharded CLOCK 相对 global CLOCK：original `+41.45%`、random `+53.43%`、k-means `+69.76%` QPS；I/O 和 hit 几乎不变。
- sharded LRU 相对 global LRU 在 32 线程提升 `60.5%`、`96.9%`、`145.3%`。
- k-means workload 的动态缓存 hit 随线程数从单线程约 `37%` 降到 32 线程约 `18%`，因为并发 query 交错破坏了人为连续到达顺序。
- TinyLFU 对 original/random 有小幅收益，对 k-means 降低约 `2.6` 到 `6.8` 个命中百分点。

#### 重要 caveat

本轮 hybrid 是 `10k StaticBFS + 10k dynamic = 20k`，而单层 baseline 是 10k。它只能回答“额外增加一个动态 tier 的收益”，不能用于 equal-capacity claim。

#### 启示

1. 并发实现差异足以盖过 replacement policy 差异。
2. 简单、低 mutation 的 CLOCK 更适合作为并发动态 baseline。
3. 多线程 query interleaving 是 workload 定义的一部分；不能用单线程 locality hit 预测高并发 hit。

### 5.6 2026-07-06：equal-capacity hybrid split

#### 方法

总容量固定为 10,000，测试 `2500:7500`、`5000:5000`、`7500:2500` 三种 static/dynamic split，3 workloads x 6 thread counts x 14 policies，共 252 次运行。所有 Recall@10 均为 `97.04%`。

#### 32 线程代表性结果

| workload | 策略 | QPS | Mean I/O | Hit |
|---|---|---:|---:|---:|
| original | StaticBFS 10k | 4479.03 | 104.32 | 8.56% |
| original | 5k/5k CLOCK | 4465.62 | 103.71 | 9.09% |
| k-means | StaticBFS 10k | 4494.86 | 104.32 | 8.56% |
| k-means | sharded CLOCK 10k | 4844.95 | 93.83 | 17.75% |
| k-means | 2.5k/7.5k CLOCK | 4863.65 | 93.40 | 18.13% |
| k-means | 7.5k/2.5k CLOCK | 4539.92 | 100.91 | 11.54% |

#### 启示

1. 旧 20k hybrid 的大收益部分来自额外容量。
2. original/random 中大 static tier 通常更适合，StaticBFS 仍是强基线。
3. k-means locality 中动态容量的边际价值更高，`2.5k static + 7.5k dynamic` 在所有线程数上是最佳 non-admission split。
4. 不存在跨 workload 固定最优的 static/dynamic 比例；容量分配应由 workload 的 miss-ratio 或 marginal hit gain 决定。

### 5.7 2026-07-17：数据集调研和 query-order 语义

#### 数据集

SIFT1M、Fashion-MNIST、GloVe-25 和 Last.fm ANN，均不超过约 1.2M base vectors。

#### query 顺序结果

| 数据集 | Lag-1 observed | Shuffled | Lift | 解释 |
|---|---:|---:|---:|---|
| SIFT1M | 0.020002 | 0.011441 | 1.7483x | 文件顺序有分组，但无 request timestamp |
| Fashion | 0.012101 | 0.011551 | 1.0476x | 近似随机 |
| GloVe | 0.011301 | 0.011181 | 1.0107x | 随机 split 的预期结果 |
| Last.fm | 0.009740 | 0.011006 | 0.8850x | 无正序列局部性 |

#### base/query 空间 pattern

query 分配到 base-fitted centers 后：

| 数据集 | Occupancy Spearman | JS divergence |
|---|---:|---:|
| SIFT1M | 0.9334 | 0.002589 |
| Fashion | 0.9543 | 0.002086 |
| GloVe | 0.9418 | 0.001565 |
| Last.fm | -0.7481 | 0.789898 |

#### 启示

1. “query 集合作为空间分布”与“query 行顺序作为时间负载”必须严格区分。
2. SIFT/Fashion/GloVe 的 base geometry 可用于 aggregate cold-start routing；Last.fm 的 item base 与 user query 是明确反例。
3. 论文不能把 k-means grouped workload 称为真实 temporal locality，只能称为 controlled spatial-locality intervention。
4. 最理想的数据源顺序应是生产历史 query、learn queries、模型训练 queries，最后才是经过验证的 base sample。

### 5.8 2026-07-17：跨数据集节点拓展集中度

初始 pilot 在 Recall@10 > 95% 下统计节点拓展：

| 数据集 | 初始配置 top 1% | top 5% | Gini |
|---|---:|---:|---:|
| SIFT1M | 14.54% | 31.49% | 0.6788 |
| Fashion | 18.07% | 29.92% | 0.5492 |
| raw GloVe PQ5, L=1600 | 6.89% | 22.29% | 0.5139 |
| Last.fm PQ16, L=100 | 42.93% | 86.01% | 0.9521 |

该阶段最重要的不是比较绝对集中度，而是发现：

- 热点不是所有数据集相同形态；
- Last.fm 有极小全局热点 footprint；
- 搜索参数和 PQ 精度会强烈改变访问路径与热点分布；
- 高 recall 必须先用正确 metric 和合理 navigation 达到，再分析 cache trace。

GloVe 初始数字随后被正确 metric/PQ 配置下的结果取代，见下一节。

### 5.9 2026-07-17：GloVe cosine、PQ chunks 与 full precision 修正

#### 问题发现

GloVe HDF5 标记为 angular，但原始向量不单位归一化。图构建和 exact rerank 使用 true cosine；当前 PQ navigation 却把 cosine 映射到 squared L2。raw vector 上二者不等价，导致图目标 metric 与导航 metric 不一致。

#### 关键结果

| 表示/导航 | 达到 95% 的最小 tested L | Recall | Mean I/O | QPS | Navigation memory |
|---|---:|---:|---:|---:|---:|
| Raw PQ5 | 1600 | 95.94% | 1609.87 | 23.44 | 5.64 MiB |
| Raw PQ16 | 150 | 95.68% | 163.35 | 225.50 | 18.06 MiB |
| Raw full precision | 10 | 95.25% | 25.17 | 1191.78 | 112.87 MiB |
| Normalized PQ10 | 40 | 95.20% | 52.75 | 665.60 | 11.29 MiB |
| Normalized PQ16 | 20 | 96.79% | 33.59 | 1007.12 | 18.06 MiB |
| Normalized PQ25 | 20 | 97.98% | 33.50 | 978.91 | 28.22 MiB |
| Normalized full precision | 10 | 95.21% | 24.79 | 1289.83 | 112.87 MiB |

#### 启示

1. L=1600 不是图质量差，而是 PQ navigation 过粗且 metric 不一致。
2. 增加 chunks 能降低量化误差，但不能从数学上修复 metric mismatch。
3. normalized PQ16 是当前 GloVe 的最佳实用点：相对 full precision 节省 `6.25x` 导航内存，仍在 L=20 达到 `96.79%` recall。
4. 论文中缓存结果必须建立在正确 metric 的高 recall operating point 上；否则所谓热点可能只是错误导航导致的广泛探索。

#### 结果替代规则

后续 GloVe 缓存和热点分析应使用 normalized PQ16/L=20 的 `top 1%=32.21%`、`top 5%=53.69%`，而不是 raw PQ5/L=1600 的 `6.89%/22.29%`。

### 5.10 2026-07-17 至 2026-07-21：QASC 概念形成

query cluster 独立物理缓存的直接方案存在五个问题：

1. 同一全局导航节点在多个 cluster cache 中重复存储；
2. hard routing 会在 cluster boundary 错误分配；
3. cluster query 数量不等于所需工作集容量；
4. 固定 cluster 会随模型和流量漂移；
5. 真正 SSD I/O 单元可能是 page，而不是 node。

因此提出 Query-Affinity Shared Cache：

- 单一去重物理节点 store；
- soft top-m query routing；
- decayed node-cluster affinity；
- global/conditional role；
- fractional logical charging；
- elastic quota、borrowing 和 over-quota reclaim；
- 容量按 marginal hit gain 而非 cluster population 分配；
- 最终扩展到 page-aware utility。

该设计阶段的最大理论贡献是把“空间局部性”具体化为：

```text
query geometry
  -> graph path overlap
  -> node-cluster conditional reuse probability
  -> query-conditioned cache utility
```

### 5.11 2026-07-24：图度数、base/sample/test 热点实验

#### 图度数

- 所有图的存储出度几乎固定为 R：SIFT/Fashion 64，GloVe 96，Last.fm 128。
- 入度长尾明显，SIFT 最大入度达到 134,506。
- 测试扩展热度与入度 Spearman：SIFT `0.342`、Fashion `0.674`、GloVe `0.161`、Last.fm `-0.030`。

结论：出度没有热点区分能力；入度可作为某些数据集的 global prior，但不能替代 query trace。

#### 全部 base 与 test hot set

| 数据集 | Top 1% overlap | Top 5% overlap |
|---|---:|---:|
| SIFT1M | 48.56% | 49.56% |
| Fashion | 86.50% | 75.10% |
| GloVe | 51.55% | 30.13% |
| Last.fm | 11.18% | 8.93% |

#### 1% base sample 与 test hot set

| 数据集 | Top 1% overlap | Top 5% overlap |
|---|---:|---:|
| SIFT1M | 34.15% | 35.84% |
| Fashion | 54.00% | 36.33% |
| GloVe | 41.35% | 23.91% |
| Last.fm | 8.45% | 8.17% |

#### 启示

1. 1% base sample 能较好判断“分布是否长尾”，但不能稳定识别“具体哪些节点会成为 test 热点”。
2. 语料几何与需求几何是不同对象；Last.fm 的 item/user factor 分离是最有力反例。
3. hot node 应定义为给定 query 分布、search configuration 和时间窗口下的未来复用价值，不是固定的图属性。
4. 必须分解 medoid/early-hop 全局核心和 query-region residual；当前 trace 还缺 query id、cluster 和 hop。

### 5.12 2026-07-27：QASC v1 落地与负面结果

#### 实现验证

QASC v1 成功打通：

```text
query -> begin_query -> soft route -> cache lookup/admission -> shared payload store
```

自动化测试覆盖 prototype、metric routing、去重、fractional charge、quota、decay、ghost bound、容量不变量和真实磁盘索引结果一致性。实现中还修复了 entropy 归一化问题：K=100、每节点最多 4 个 affinity slot 时，若除以 `ln(100)`，global role 永远无法达到默认阈值；修复后按可表示 support 归一化。

#### SIFT1M 结果

| workload | 策略 | QPS | Mean I/O | Hit | Recall |
|---|---|---:|---:|---:|---:|
| original | LRU | 325.16 | 103.58 | 9.21% | 97.04% |
| original | QASC | 245.26 | 103.06 | 9.66% | 97.04% |
| k-means grouped | LRU | 376.22 | 71.77 | 37.09% | 97.04% |
| k-means grouped | QASC | 127.54 | 99.36 | 12.90% | 97.04% |

#### 启示

1. query context 接口和共享缓存抽象可行，不改变 search semantics。
2. QASC v1 的方形根 quota、固定 global fraction、observation threshold 和 O(C) victim scan 尚不是有效策略。
3. k-means continuous 同时具有空间和强短期时间局部性，LRU 能把近乎整个 cache 给当前 cluster；QASC 长期保护旧 cluster，反而发生容量僵化。
4. QASC 更可能在 cluster interleaved、同一区域经过较长间隔返回的 workload 中体现价值，但该假设尚未实验验证。
5. prototype 来自待测 test query 自身，属于乐观 smoke setting，不能作为正式无泄漏结果。

#### 设计修正方向

后续策略应是 QASC-Hybrid，而不是继续强化硬 quota：

- 20% 到 40% LRU recency window；
- 60% 到 80% query-conditioned main；
- 全局 expected-hit score，空间信息增强而非替代 recency；
- query-conditioned recency 和 cluster transition model；
- 可选 marginal-hit water-filling reservation；
- 先用 query-aware trace replay 验证，再优化 live QPS。

### 5.13 2026-07-27：四类静态缓存节点来源

#### 方法

四种等容量 1% 静态节点集合：medoid BFS、all-base hot、1% base-sample hot、test-hot oracle。4 datasets x 4 sources x 3 repetitions，共 48 次单线程运行，来源顺序轮换。所有运行 Recall@10 > 95%，在线 hit 与离线 test expansion coverage 两位小数完全一致。

#### 核心结果

| 数据集 | BFS hit/QPS | All-base hit/QPS | 1% sample hit/QPS | Test oracle hit/QPS |
|---|---:|---:|---:|---:|
| SIFT1M | 8.40% / 348.01 | 13.03% / 349.82 | 11.94% / 348.70 | 14.54% / 352.94 |
| Fashion | 9.85% / 623.93 | 17.96% / 661.75 | 16.52% / 663.79 | 18.07% / 663.84 |
| GloVe | 27.44% / 1317.45 | 29.34% / 1339.16 | 27.50% / 1302.12 | 32.21% / 1316.51 |
| Last.fm | 11.97% / 776.81 | 15.38% / 797.07 | 13.71% / 772.41 | 44.55% / 971.88 |

#### 共享核心发现

节点集合 overlap 很低时，交集仍可能承担大部分缓存价值：

- SIFT BFS/all-base 只重合 `12.58%`，交集却承担 test 总扩展 `7.76%`，占 BFS 命中价值 `92.39%`。
- Last.fm BFS/test 只重合 `5.61%`，交集承担 test 总扩展 `11.40%`，几乎覆盖 BFS 的全部 `11.97%` hit。

#### 启示

1. 图搜索缓存天然适合“两层”解释：小型共享导航核心 + workload-specific residual。
2. all-base hot 通常比 BFS 有更高 hit，但 QPS 收益取决于 I/O 是否真正在关键路径上。
3. Fashion 证明 base prior 可以接近 oracle；Last.fm 证明它也可能完全错过 demand hot set。
4. test-hot 是最佳频率静态集合的乐观上界，不能作为部署策略；它与逐访问动态 Belady 上界回答不同问题。

## 6. 跨实验统一认识

### 6.1 局部性应拆成四个层次

#### 图结构局部性

medoid、early-hop 和高入度 hub 构成稳定导航骨干。StaticBFS 主要利用这一层。

#### Query 空间局部性

相近 query 更可能共享后期搜索路径和区域节点。这是 k-means grouped workload 大幅提升动态 hit 的根因。

#### 到达顺序局部性

只有相似 query 在时间上接近时，普通 LRU/CLOCK 才能直接利用路径重叠。多线程 interleaving 会削弱这一层。

#### Demand distribution locality

即使 query 顺序不连续，某些 query region 或节点长期更常出现，形成频率型/条件型热点。QASC 的目标是利用这一层，但 v1 尚未成功。

论文中不应把四者统一简称为 temporal locality。更准确的表述是：**query-space locality induces path overlap；arrival process determines whether this overlap appears as short-term temporal reuse or long-horizon conditional reuse。**

### 6.2 最有解释力的缓存分层

现有证据支持以下抽象：

```text
Physical cache capacity C
  = Global navigation core G
  + Recency window W
  + Query-conditioned residual S
```

- `G`：medoid、early-hop、跨 cluster 高频节点，低维护、长期常驻；
- `W`：捕捉当前 burst 和普通时间局部性；
- `S`：保护跨较长 cluster 间隔仍有条件重用的节点。

三部分应共享去重 payload store，逻辑角色可转换，容量根据 marginal saved I/O 动态调整。

### 6.3 为什么传统“高级策略”没有稳定胜出

传统策略的目标与 workload 结构并不完全对齐：

- LRU/CLOCK 只看全局时间；
- naive LFU 不会快速遗忘旧 cluster；
- TinyLFU 用历史频率拒绝 burst newcomer；
- ARC/2Q/SLRU 的 recency/frequency 分段不理解 query region；
- GDSF 当前按 equal-size/equal-cost 节点近似，没有反映 page read cost；
- LeCaR/Cacheus 在当前实现中仍是对 recency/frequency 专家学习，不包含 query context。

同时，它们的系统开销往往比节省的 I/O 更大。因此真正的比较对象不是抽象 hit rate，而是：

```text
expected saved physical I/O
- policy CPU
- lock contention
- metadata memory
- extra cache lookup cost
```

### 6.4 数据集差异不是噪声，而是论文核心证据

| 数据集 | 主要特征 | 缓存含义 |
|---|---|---|
| SIFT1M | 中等热点；base/test 中等一致；人工 grouping 收益强 | 适合研究 global + dynamic/query-conditioned 的平衡 |
| Fashion | base/test 高度一致；all-base 几乎达到 test oracle | 离线 base prior 和低成本 sampling 有现实可行性 |
| GloVe | 正确 PQ 后热点较强；BFS 已强；PQ/metric 极敏感 | 先解决 navigation accuracy，再谈 cache；全局核心较大 |
| Last.fm | test 热点极强；item base 与 user demand 严重失配 | 必须使用 demand-aligned history；最能证明 base prior 的边界 |

## 7. 旧结果、修正结果与引用规则

### 7.1 可以作为主结果引用

- `query_workload_cache_policy_summary.csv` 中 original/random/k-means 单线程冷缓存结果；
- `workload_traces/belady_trace_replay_summary.csv`；
- 2026-07-06 equal-capacity fair split；
- normalized GloVe PQ16/full-precision sweep；
- 2026-07-24 graph degree/hotness；
- 2026-07-27 static cache source 48-run repeated experiment。

### 7.2 需要带 caveat 引用

- 2026-07-03 concurrency full sweep：single-tier 和 sharding结论有效；20k hybrid 只表示 extra-tier，不是公平容量。
- QASC v1：只表示接口可行和负面策略结果；prototype 使用 test centers，且每配置一次。
- 初始 spatial pilot 的 GloVe：只用于说明错误配置如何改变路径，不作为最终热点结果。
- warmup CSV：机制结论可引用，2Q/SLRU/TinyLFU 旧数值需谨慎。

### 7.3 不应作为论文 claim

- “标准 ANN benchmark 原始 query 顺序具有真实生产 temporal locality”；
- “all-base 热点普遍代表线上 query 热点”；
- “入度高的节点就是应该缓存的节点”；
- “QASC v1 优于 LRU”；
- “20k hybrid 在相同容量下优于 10k StaticBFS”；
- “更高 cache hit 必然带来更高 QPS”。

## 8. 当前可以形成的论文贡献

### 8.1 Measurement/characterization paper 路线

现有材料最接近一篇系统测量与 characterization 论文。可主张：

1. 首次系统区分 disk graph ANNS 中的 global navigation locality、query-space path overlap、arrival-order locality 和 demand-conditioned locality。
2. 构建 trace replay + live DiskANN 的统一评估框架，覆盖 replacement、admission、warmup、Belady、并发和 workload construction。
3. 证明 query grouping 同时提高在线 LRU 和离线 OPT，建立 query geometry 改变可缓存性的因果证据。
4. 证明图中心性、base geometry 和 test demand hotness 不能互相替代，并给出 Fashion 与 Last.fm 的强正反例。
5. 证明系统开销与并发 backend 可以主导策略优劣，sharding 比复杂 replacement policy 更关键。
6. 提出由共享导航核心、recency window 和 query-conditioned residual 组成的设计原则。

这条路线不要求 QASC v1 成为正面性能贡献，但必须补真实或半真实时序 workload、重复实验和 page-level 分析。

### 8.2 New cache policy/systems paper 路线

如果论文必须包含新策略，当前 QASC v1 还不够。至少需要实现并证明 QASC-Hybrid：

- 在 original/random 上不显著劣于 sharded CLOCK/LRU；
- 在 cluster-contiguous 上接近 LRU；
- 在 cluster-interleaved 或真实 session trace 上超过 LRU/CLOCK；
- 使用 history/train-only prototype 和 statistics；
- 关闭可观比例的 Belady gap；
- 在多线程下通过分片或近似 victim structure 保持 QPS；
- 在至少 3 个数据集上给出正面结果，同时保留 Last.fm 的 source mismatch 分析。

### 8.3 推荐定位

建议采用“measurement-first, design-second”的论文定位：

> 先以严谨 characterization 解释现有缓存策略为什么在 graph ANNS 上表现不稳定，再提出一个由测量结果直接导出的轻量 hybrid 策略。

这样即使新策略收益只在部分 workload 上显著，论文仍有独立的测量贡献；负面结果也自然成为设计动机，而不是失败记录。

## 9. 建议的论文研究问题

建议用四个 RQ 组织全文：

### RQ1：Disk-resident graph ANNS 中存在什么形式的跨 query 重用？

回答材料：节点 expansion concentration、global core、query grouping、Belady、query-order survey。

### RQ2：传统缓存策略为什么不能稳定利用这些重用？

回答材料：原始/random/k-means policy matrix、LFU/TinyLFU 反例、warmup negative result、QPS review。

### RQ3：图结构或 base 数据能否预测 query-demand hot nodes？

回答材料：in-degree correlations、base/query occupancy、all-base/sample/test overlap、四类静态来源实测。

### RQ4：系统和并发因素如何改变策略选择？

回答材料：global vs sharded backend、fair split、hit/I/O/QPS divergence、page cache caveat。

若实现 QASC-Hybrid，可增加：

### RQ5：空间上下文增强的 recency cache 能否关闭传统策略与 OPT 的差距？

## 10. 建议的论文结构

1. **Introduction**：StaticBFS 强但静态；传统动态缓存面向时间局部性；graph search 的重用由 query geometry 和 graph navigation 共同产生。
2. **Background**：DiskANN graph layout、PQ navigation、StaticBFS、node cache path、query search path。
3. **Methodology**：数据集、正确 metric/high-recall operating point、workload 类型、cache capacity、公平性、trace/live 指标。
4. **Locality Characterization**：query order、expansion concentration、Belady、global core、cluster path overlap。
5. **Why Existing Policies Fall Short**：replacement matrix、warmup、admission negative result、CPU/lock overhead。
6. **Can Offline Data Predict Demand?**：base occupancy、degree、hot-set overlap、static source live search。
7. **Concurrent System Evaluation**：sharding、equal-capacity split、interleaving effect。
8. **Design Implications / Proposed Policy**：G + W + S 三层逻辑、去重 store、expected saved I/O utility。
9. **Evaluation of New Policy**：仅在实现 QASC-Hybrid 后加入。
10. **Related Work**：DiskANN、Starling/SPANN、replacement/admission、semantic/similarity caching。
11. **Limitations and Threats**。

## 11. 推荐图表

论文主文建议优先保留以下图表：

1. **研究概念图**：query geometry -> overlapping graph paths -> cache reuse。
2. **三 workload hit/OPT 图**：original、random、k-means 在多容量下的 LRU/OPT 曲线。
3. **跨数据集 expansion Lorenz/rank curve**：统一使用正确 metric 和 >95% recall 配置。
4. **global core + residual 图**：两两集合 overlap 与交集 expansion mass 并列展示。
5. **base/test transfer heatmap**：SIFT/Fashion/GloVe/Last.fm 的 occupancy 与 hot-set overlap。
6. **hit-I/O-QPS 三轴或并排图**：证明机制指标和系统指标不总同向。
7. **并发 scaling 图**：global CLOCK、sharded CLOCK、StaticBFS，1 到 32 threads。
8. **fair split 图**：不同 workload 下 static fraction 对 hit/I/O/QPS 的影响。
9. **GloVe PQ tradeoff 图**：作为 methodology correctness 或附录案例。
10. **QASC v1 negative result 图**：若正文讨论设计演进，可展示“space-only quota cannot replace recency”。

## 12. 论文级补实验优先级

### P0：发表前必须完成

1. **真实或有时间语义的 query trace**：优先 Last.fm 1K listening histories、MIND、KuaiRand/KuaiRec 或内部日志生成 embedding query；保留 session/timestamp。
2. **query-aware trace schema**：至少记录 `query_id`、timestamp/order、cluster route、vertex id、hop/stage、page id。
3. **所有关键配置重复 3 到 5 次**：随机 run order，报告 mean/median、95% CI、page-cache 状态和 CPU/NUMA binding。
4. **容量曲线**：至少 0.1%、0.5%、1%、2%、5%，同时按 node count 和 bytes 报告。
5. **统一正确 operating point**：每数据集固定 >95% recall，并记录 metric transformation、PQ chunks、L、R、build L。
6. **训练/测试隔离**：prototype、hotness prior、transition model 和 admission statistics 只能来自 train/history；test-hot 只作为 oracle。
7. **页面级分析**：把 node access 聚合到 4 KiB page/sector，报告 node hit、page hit、physical reads 和 read amplification。

### P1：强烈建议完成

1. **base sample learning curve**：0.1%、0.5%、1%、5%、10%，多 seed，报告 hot-set identity overlap 和 expansion-shape error。
2. **排除 medoid/early-hop 消融**：去掉 medoid、前 1/2/3 hop 后重新计算热点和 overlap。
3. **query K 和 route sweep**：K=8/16/32/64/100/256，hard vs soft top-m，避免 K=100 成为未经验证常数。
4. **cluster-contiguous vs interleaved vs drift**：区分短期 recency、周期返回和分布漂移。
5. **policy overhead instrumentation**：lock wait/hold、lookup/admission CPU、victim scan length、metadata bytes、admission rejects。
6. **compulsory/capacity/conflict miss 分解**：解释 OPT gap 的来源。

### P2：新策略路线

1. 实现去重 QASC-Hybrid 的 recency window + spatial main。
2. 用 global expected-hit score 替代硬 cluster quota作为主版本。
3. 加入 query-conditioned logical recency 和 cluster transition EMA。
4. 用 replay 先比较 LRU、CLOCK、StaticTopC、Belady、per-cluster LRU、QASC-Hybrid。
5. 指标使用 Belady gap closure：

```text
gap_closure = (hit_policy - hit_lru) / (hit_opt - hit_lru)
```

6. live 版本再使用 sharded store、CLOCK/heap victim candidates 和延迟 profile 合并优化 QPS。

## 13. 论文 claim 建议

### 可以谨慎写入摘要或 introduction

- Query-space grouping on SIFT1M raises both LRU and Belady hit rate, indicating that query geometry changes the intrinsic cacheability of graph-search accesses.
- Static graph centrality and base-vector geometry are unreliable substitutes for demand traces; their predictive power varies sharply across datasets.
- A small shared navigation core carries disproportionate reuse, while the remaining cache value is workload specific.
- Concurrent cache backend design can matter more than replacement policy sophistication.

### 只能写入结果或讨论，不能泛化

- Fashion 的 all-base static hot set 接近 test oracle。
- Last.fm 的 test-hot oracle 达到 `44.55%` hit 和 `25.11%` QPS uplift。
- SIFT k-means grouped LRU 达到 `37.09%` hit。

这些都必须带数据集、容量、L、线程数和 workload construction。

### 应主动报告的负面结果

- TinyLFU admission 在 locality-heavy burst 上拒绝有用 newcomer。
- StaticBFS warmup 可污染 ARC/history-aware state。
- QASC v1 在 cluster-contiguous workload 上远低于 LRU。
- GloVe 在错误 cosine/PQ 配置下产生误导性深搜索和热点分布。
- all-base prior 在 Last.fm 上失败。

负面结果共同强化一个中心论点：**不能把通用 cache policy、图结构 prior 或 query clustering 中的任何一个单独视为完整答案。**

## 14. 威胁与限制

1. 大部分早期 SIFT 和 concurrency matrix 每个配置只测一次。
2. 多次实验未统一清空 OS page cache 或固定 NUMA/CPU。
3. 当前主要是 node-level cache，而 SSD 实际以 page/sector 读取。
4. K-means grouped workload 是人工干预，不是真实 request trace。
5. 现有公开 benchmark query order 缺乏 timestamp。
6. 当前缓存容量主要按 node count，而 node adjacency/payload bytes 可能不同。
7. 不同数据集使用不同图参数和 PQ/full-precision navigation，绝对 expansion 数不可横向解释为纯数据分布差异。
8. QASC v1 使用 test-derived prototypes，存在乐观信息来源。
9. Test-hot static set 和 Belady 均使用未来信息，只能作为不同类型的 oracle。
10. 当前尚未完成 QASC-Hybrid 或真实漂移 workload，因此新策略的正面 claim 尚不存在。

## 15. 建议的下一步执行顺序

1. 冻结当前结果为 `characterization-v1`，生成一张统一实验配置表。
2. 扩展 query-aware/page-aware trace，不先修改 live replacement policy。
3. 选择至少一个有真实时间顺序的数据集，验证空间距离、session continuity、path overlap 和 reuse distance 的关系。
4. 在 replay 中实现 QASC-Hybrid 和 expected-hit baseline，先证明它能在 contiguous 上接近 LRU、在 interleaved 上超过 LRU。
5. 补容量、K、sample ratio、多 seed 和 medoid/hop 消融。
6. 只将 replay 中稳定有效的策略接入 sharded live cache。
7. 最后做统一硬件环境下的多次 live evaluation 和论文图表。

## 16. 文档与产物索引

### 缓存策略与并发

- `experiments/cache_eval/sift1m/results/current_cache_policy_summary.csv`
- `experiments/cache_eval/sift1m/results/warm_bfs_cache_policy_summary.csv`
- `experiments/cache_eval/sift1m/results/query_workload_cache_policy_summary.csv`
- `experiments/cache_eval/sift1m/results/workload_traces/belady_trace_replay_summary.csv`
- `experiments/cache_eval/docs/concurrent_cache_policy_results_20260703.md`
- `experiments/cache_eval/docs/concurrent_cache_policy_fair_split_results_20260706.md`
- archived `cache_policy_correctness_review.md`
- archived `cache_performance_qps_review.md`

### 数据集、空间模式与图热点

- `experiments/cache_eval/docs/vector_dataset_query_order_survey.md`
- `experiments/cache_eval/docs/spatial_locality_pilot_results_20260717.md`
- `experiments/cache_eval/docs/glove_pq_full_precision_results_20260717.md`
- `experiments/cache_eval/docs/graph_degree_expansion_hotness_results_20260724_zh.md`
- `experiments/cache_eval/results/vector_patterns/summary/`
- `experiments/cache_eval/results/spatial_locality_pilot/summary/`
- `experiments/cache_eval/results/graph_hotness_degree/summary/`

### QASC 与静态节点来源

- `experiments/cache_eval/docs/spatial_locality_aware_cache_design.md`
- `experiments/cache_eval/docs/spatial_locality_aware_cache_design_zh.md`
- `experiments/cache_eval/docs/qasc_implementation_design_zh.md`
- `experiments/cache_eval/docs/qasc_validation_results_20260727_zh.md`
- `experiments/cache_eval/docs/static_cache_node_source_evaluation_20260727_zh.md`
- `experiments/cache_eval/results/static_cache_sources/`

## 17. 最终建议

当前工作最有价值的成果不是某个复杂 replacement policy 已经胜出，而是形成了一条可辩护的因果链：

```text
query distribution and arrival order
  -> overlap and reuse distance of graph-search paths
  -> value of global versus conditional cache capacity
  -> hit and logical I/O
  -> system performance after policy/locking overhead
```

论文应围绕这条链组织，而不是按策略名称罗列结果。

现有证据已经足以支持一篇有潜力的 measurement paper 骨架，但距离扎实投稿仍缺三项核心材料：真实时序 workload、page-level physical-I/O analysis、关键实验的重复与置信区间。若希望把 QASC 作为主要系统贡献，则应放弃“空间 quota 替代 LRU”的方向，转向“recency window + query-conditioned main”的 hybrid，并以关闭 Belady gap 和 interleaved-cluster workload 为主要验证目标。
