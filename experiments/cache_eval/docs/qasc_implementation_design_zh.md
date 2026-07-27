# QASC 落地设计：从 query 路由到共享节点缓存

## 1. 文档目的

本文描述 Query-Affinity Shared Cache（QASC）如何在当前 Rust DiskANN 代码中落地。重点不是再次说明“为什么 query 空间局部性存在”，而是回答以下工程问题：

1. query 向量在搜索调用链的哪个位置被路由？
2. cluster 信息如何传递到节点 cache lookup 和 admission？
3. 如何只保存一份节点 payload，同时表达节点对多个 query cluster 的亲和度？
4. 全局层和 cluster-conditioned 层如何共享物理容量？
5. 配额如何计算、借用和回收？
6. candidate 和 victim 如何比较？
7. 第一版实现哪些功能，哪些功能明确留到后续？
8. 如何验证 QASC 没有破坏 Recall、容量约束或现有缓存策略？

本文也是 QASC v1 的实现契约。代码实现、CLI 参数和测试应与本文保持一致。

## 2. 当前 DiskANN 缓存调用链

当前磁盘图搜索的核心调用链是：

```text
search_disk_index CLI
  -> DiskIndexSearcher::search(query)
    -> DiskAccessor::new(query)
      -> 从 scratch pool 取得一个线程私有 VertexProvider
    -> DiskAccessor::expand_beam(node_ids)
      -> CachedDiskVertexProvider::load_vertices(node_ids)
        -> SharedNodeCache lookup
        -> 未命中节点从磁盘读取
      -> CachedDiskVertexProvider::process_loaded_node(node_id)
        -> SharedNodeCache admission
```

当前问题是：

- `DiskAccessor` 知道 query 向量；
- `CachedDiskVertexProvider` 知道节点访问和缓存状态；
- 但二者之间没有 query context 通道。

所以 QASC 落地的第一个必要改动不是 replacement policy，而是给 `VertexProvider` 增加每次搜索开始时的 query 初始化方法：

```rust
fn begin_query(&mut self, query: &[Data::VectorDataType]) -> ANNResult<()>;
```

默认实现是 no-op。`CachedDiskVertexProvider` 的实现会调用共享 QASC router，得到 `QueryRoute`，并把 route 保存在当前线程私有 provider 中。后续同一个 query 的所有 lookup 和 admission 都使用这份 route。

### 2.1 代码位置

| 文件 | QASC 职责 |
|---|---|
| `diskann-tools/src/bin/search_disk_index.rs` | 解析和校验 `--cache_policy qasc` 及全部 QASC 参数，构造 settings。 |
| `diskann-disk/src/data_model/cache.rs` | 在 `CachingStrategy` 中表示 QASC，使 factory 能区别普通 policy cache 和 query-conditioned cache。 |
| `diskann-disk/src/data_model/query_affinity_cache.rs` | prototype 路由、profile、role、quota、lookup、admission、victim、ghost metadata 和 invariants 的核心实现。 |
| `diskann-disk/src/search/traits/vertex_provider.rs` | 定义默认 no-op 的 `begin_query`，保证非 QASC provider 无需修改行为。 |
| `diskann-disk/src/search/provider/disk_vertex_provider_factory.rs` | 读取图 metadata，校验 prototype 维度，创建一份共享 `QueryAffinityCache`。 |
| `diskann-disk/src/search/provider/cached_disk_vertex_provider.rs` | 保存当前 query 的 `QueryRoute`，并在每次节点 lookup/admission 时把 route 传给共享 QASC。 |
| `diskann-disk/src/search/provider/disk_provider.rs` | 每次取得 search scratch 后调用 `begin_query(query)`，保证复用 provider 时旧 route 被覆盖。 |

`QueryAffinityCache` 是跨 query 共享的：它保存 resident payload 和历史 profile。
`QueryRoute` 是一次搜索私有的：它只保存在当前 scratch 中，不能放进全局变量。这个所有权
划分是理解实现的关键。

### 2.2 一个 query 的完整运行时生命周期

以一次 `DiskIndexSearcher::search(q)` 为例：

1. `search_disk_index` 启动时只创建一次 factory。factory 从图 header 得到维度 `D`，
   加载 `K x D` prototype fbin，并创建共享 QASC。
2. 搜索开始时，从 object pool 取得线程私有 scratch；scratch 内含一个
   `CachedDiskVertexProvider`，它可能被上一次 query 使用过。
3. `DiskAccessor::new` 立即调用 `vertex_provider.begin_query(q)`。QASC 将 query 转成
   f32，计算 top-m route，更新 cluster arrival mass，并覆盖 provider 中的旧 route。
4. 图搜索准备拓展一批 node ID 时，provider 对每个 ID 调用
   `QASC.lookup(node_id, route)`。命中返回共享 `Arc<CachedNode>`，未命中 ID 进入磁盘读。
5. miss 的节点从磁盘 materialize 后，provider 调用
   `QASC.admit_node(node_id, payload, route)`。QASC 可能直接插入、拒绝，或比较
   candidate/victim 后替换；搜索结果不依赖是否成功 admission。
6. 同一个 query 的所有 beam expansion 和 rerank load 都复用步骤 3 得到的 route，
   不会在一次搜索中重复做 prototype 路由。
7. 搜索结束后 scratch 返回 object pool。共享 QASC 保留 resident 和 profile；scratch
   中的 route 即使仍存在，也会在下一次步骤 3 被覆盖，因此不会跨 query 泄漏。

一个简化例子：query 被路由为 `[(cluster 2, 0.8), (cluster 0, 0.2)]`。节点 17
首次 lookup miss 后，其 profile 增加一次 observation，并分别给 cluster 2/0 增加
`0.8/0.2` affinity。节点从磁盘读出并被 admission 后，物理 store 中只有一份节点 17，
逻辑上向两个 cluster 计费 `0.8/0.2`。后续相近 query 命中节点 17 时不发生磁盘 I/O，
但 profile 仍继续更新。如果长期有许多不同 cluster 命中该节点，其 affinity entropy
达到阈值后，节点 17 会改为 global role，原有 conditional charges 同步撤销。

调试 access trace 时，QASC 命中的 `cache_source` 为 `qasc`。命中只影响节点 payload
从内存还是磁盘取得，不改变距离计算、beam 扩展规则或最终 top-k 语义。

## 3. QASC v1 的实现范围

QASC v1 实现以下功能：

- 从 f32 `.fbin` 文件加载固定 query prototypes；
- 支持 L2、cosine、cosinenormalized 和 inner product 路由距离；
- 对每个 query 执行 soft top-`m` 路由；
- 使用一个去重物理节点 store；
- 为节点维护指数衰减的 cluster affinity；
- 根据 affinity entropy 将 resident node 分为 global 或 conditional；
- conditional node 对多个 cluster 做 fractional logical charging；
- 根据近期 query cluster mass 动态计算 elastic logical quotas；
- cluster 空闲时允许其他 cluster 借用物理容量；
- 容量满时优先从过量 cluster 回收；
- 使用 query-conditioned utility 比较 candidate 和 victim；
- 维护有界 ghost profile，避免元数据无限增长；
- 提供容量、去重、配额和命中统计的测试接口。

QASC v1 暂不实现：

- 在线 k-means 或 centroid 漂移更新；
- 基于经验 miss-ratio curve 的真正 water-filling；
- page-level admission 和 prefetch；
- hop-aware global/conditional 分类；
- 无锁或分片并发实现；
- 模型训练型 reuse-distance predictor；
- semantic result cache。

这些项目不是被忽略，而是需要在 v1 证明 query-conditioned node cache 有效后单独实现和消融。

## 4. 核心数据结构

### 4.1 QueryAffinityCacheConfig

```text
capacity
prototype_file
metric
top_m
global_fraction
global_entropy_threshold
global_min_observations
min_admission_observations
max_affinity_clusters
decay_interval
quota_update_interval
ghost_capacity_multiplier
admission_margin
```

默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `top_m` | 2 | 每个 query 最多关联两个 cluster |
| `global_fraction` | 0.25 | 全局层软目标占总容量比例 |
| `global_entropy_threshold` | 0.75 | 节点归一化 affinity entropy 达到该值后可成为 global |
| `global_min_observations` | 8 | global 分类前至少观察的访问次数 |
| `min_admission_observations` | 2 | 满缓存时 candidate 至少被观察的次数 |
| `max_affinity_clusters` | 4 | 每个节点最多保存的 cluster affinity 项数 |
| `decay_interval` | 100,000 | 每多少次节点访问将 affinity 和 query mass 减半 |
| `quota_update_interval` | 1,024 | 每多少个 query 重新计算 logical quota |
| `ghost_capacity_multiplier` | 4 | profile 上限为 `4 * cache_capacity` |
| `admission_margin` | 1.0 | candidate utility 至少达到 victim utility 的倍数 |

`global_fraction` 是软目标，不是物理隔离。global 和 conditional 条目仍在同一个 store 中。

### 4.2 QueryPrototypes

```text
num_clusters: K
dimensions: D
vectors: K x D f32
precomputed_norms: K
```

prototype 文件使用标准 DiskANN fbin 格式：

```text
u32 rows = K
u32 dimensions = D
f32 vectors[K * D]
```

prototype 维度必须与图节点维度一致。文件为空、维度不一致、包含非有限值或 `top_m > K` 时，CLI 直接失败。

### 4.3 QueryRoute

```text
weights: [(cluster_id, weight)]  // 按 weight 降序，最多 top_m 项
confidence: top_1_weight
```

所有 weight 非负且总和为 1。即使是 hard routing，也表示为只有一项、weight 为 1 的 route。

### 4.4 NodeProfile

每个被观察节点维护：

```text
total_observations: f64
affinities: [(cluster_id, decayed_mass)]
last_observed_tick: u64
```

`affinities` 使用 weighted Space-Saving 近似，只保留
`max_affinity_clusters` 项，避免 `num_nodes x K` 稠密矩阵。已有 cluster 直接累加；
槽位已满且出现新 cluster 时，用“最小已有 affinity + 新 weight”替换最小项。这样
长期出现的新 cluster 能进入 profile，不会被早期累计质量永久挡在槽位之外。

每次 query route 为 `[(z_i, w_i)]` 时：

```text
total_observations += 1
affinity[z_i] += w_i
```

达到 `decay_interval` 后，所有 profile affinity、total observations 和 cluster arrival mass 除以 2。非常小且非 resident 的 ghost profile 会被删除。

### 4.5 ResidentMetadata

```text
last_access_tick
inserted_tick
role: Global | Conditional
charges: [(cluster_id, fractional_size)]
```

global node 的 `charges` 为空，物理上占一个 cache entry，并计入 global occupancy。

conditional node 的 charge 由其保留 affinity 归一化得到：

```text
charge(v,z) = affinity(v,z) / sum_j affinity(v,j)
```

所有 charge 总和为 1。一个节点即使对多个 cluster 有用，store 中仍只有一个 payload。

### 4.6 QueryAffinityCache

```text
store: HashMap<node_id, Arc<CachedNode>>
resident_metadata: HashMap<node_id, ResidentMetadata>
profiles: HashMap<node_id, NodeProfile>
ghost_order: VecDeque<(node_id, last_observed_tick)>
cluster_arrival_mass: Vec<f64>
cluster_quotas: Vec<f64>
cluster_usage: Vec<f64>
global_occupancy: usize
stats
tick
query_count
```

`store.len()` 永远不超过 physical capacity。`cluster_usage` 是逻辑计费，允许小数；其总和等于 conditional resident 数量。

## 5. Query 路由

### 5.1 距离

对 query `q` 和 prototype `c_z`：

```text
L2:               d_z = ||q - c_z||^2
cosine:           d_z = 1 - dot(q,c_z) / (||q|| ||c_z||)
cosinenormalized: d_z = 1 - dot(q,c_z)
inner product:    d_z = -dot(q,c_z)
```

选择距离最小的 top-`m` prototype。

### 5.2 距离到权重

为了避免额外 temperature 参数，v1 使用平移后的 inverse-distance 权重：

```text
adjusted_i = d_i - min_j(d_j)
raw_i = 1 / (epsilon + adjusted_i)
weight_i = raw_i / sum_j(raw_j)
```

其中 `epsilon=1e-6`。

如果两个 prototype 距离完全相同，它们获得相同权重。排序 tie-break 使用 cluster id，保证确定性。

## 6. 节点角色分类

对于 profile 中的 affinity 分布：

```text
p(z|v) = affinity(v,z) / sum_j affinity(v,j)
H(v) = -sum_z p(z|v) * ln(p(z|v))
K_profile = min(K, max_affinity_clusters)
H_norm(v) = H(v) / ln(K_profile)
```

当 `K_profile <= 1` 时定义 `H_norm=1`。分母使用 profile 可表示的最大 support，而不是
prototype 总数 `K`。否则默认 `K=100`、`max_affinity_clusters=4` 时，理论最大值只有
`ln(4)/ln(100)=0.301`，`global_entropy_threshold=0.75` 将永远不可达。

节点满足以下条件时分类为 global：

```text
total_observations >= global_min_observations
and H_norm(v) >= global_entropy_threshold
```

否则分类为 conditional。

角色不是永久的。节点随着访问来自更多 cluster，可以从 conditional 晋升为 global；在 decay 后也可能重新变为 conditional。角色变化时必须更新 `global_occupancy` 和 fractional charges，但不能复制或移动 payload。

## 7. Logical quota

conditional 容量软预算：

```text
C_conditional = capacity - round(capacity * global_fraction)
```

query 到达时，将 route weight 加到 `cluster_arrival_mass`。每 `quota_update_interval` 个 query，根据 square-root rule 重新计算 quota：

```text
weight_z = sqrt(cluster_arrival_mass[z] + 1)
quota_z = C_conditional * weight_z / sum_j weight_j
```

使用 `sqrt` 而不是线性比例有两个目的：

- 热 cluster 获得更多容量；
- 避免单个热门 cluster 吞掉全部 conditional 容量。

v1 使用 soft quota：

- cache 未满时，任何 cluster 都可以借用空闲物理容量；
- cache 已满时，优先从 `usage_z - quota_z` 最大的 cluster 回收；
- 没有 cluster 超额时，从 candidate 主要关联的 cluster 中选择 victim；
- 找不到合适 conditional victim 时，只有在 global occupancy 超过 global target 后才从 global 回收；
- 最终 fallback 是全体 resident 中 utility 最低的节点。

因此不会出现某个冷 cluster 的保留配额长期空置、同时热 cluster 发生 thrashing 的硬分区问题。

## 8. Lookup 流程

伪代码：

```text
lookup(node_id, route):
    tick += 1
    maybe_decay()
    observe(profile[node_id], route)

    if node_id not in store:
        stats.misses += 1
        remember_bounded_ghost(node_id)
        return MISS

    stats.hits += 1
    resident[node_id].last_access_tick = tick

    old_role, old_charges = resident metadata
    new_role, new_charges = classify(profile[node_id])
    update global occupancy and cluster usage by delta

    return store[node_id]
```

注意：profile 在 miss 时也更新。否则一个被 admission filter 拒绝的节点永远无法积累“再次出现”的证据。

## 9. Admission 流程

节点从磁盘读出并 materialize 后执行：

```text
admit(node_id, payload, route):
    if node_id already resident:
        replace payload in place
        return

    candidate = profile[node_id]

    if cache not full:
        insert once
        attach role and charges
        return

    if candidate.total_observations < min_admission_observations:
        reject
        return

    victim = choose_victim(candidate, route)
    if no victim:
        reject
        return

    if utility(candidate, route) < admission_margin * utility(victim, route):
        reject
        return

    remove victim payload and all logical charges
    insert candidate payload exactly once
    attach role and charges
```

缓存未满时允许首次访问填充，以避免长时间空缓存。缓存满后才启用 observation 和 utility 门槛。

## 10. Utility

QASC v1 的 utility 是可解释启发式，而不是训练模型：

```text
route_affinity(v,q) = sum_z route_weight(q,z) * affinity(v,z)
frequency_bonus(v) = ln(1 + total_observations(v))
freshness(v) = 1 / (1 + (tick - last_observed_tick) / decay_interval)

utility(v,q) =
    route_affinity(v,q)
    + 0.10 * frequency_bonus(v)
    + 0.01 * freshness(v)
```

对于 global victim，route affinity 替换为：

```text
global_affinity(v) = total_observations(v) * max(H_norm(v), epsilon)
```

选择 cluster victim 时，使用只包含目标 cluster、weight=1 的 synthetic route。这样被 cluster `z` 占用的节点按其对 `z` 的实际价值排序。

`0.10` 和 `0.01` 是 v1 固定常数，必须在后续消融中验证。它们的作用是 tie-break 和轻量稳定化，不应压过 cluster affinity 主项。

## 11. Victim 选择

### 11.1 Candidate 是 global

```text
if global_occupancy >= global_target:
    victim = global resident with minimum global utility
else:
    victim = lowest-utility conditional resident from most over-quota cluster
```

若 conditional 没有候选，再从全体 resident 中选择最低 utility。

### 11.2 Candidate 是 conditional

1. 找出 candidate charge 最大的 primary cluster。
2. 如果存在 `usage_z > quota_z` 的 cluster，选择超额最大的 cluster。
3. 否则选择 primary cluster。
4. eligible victim 是对该 cluster 有正 charge 的 conditional resident。
5. 使用 synthetic single-cluster route 选 utility 最低者。
6. 如果没有 eligible victim：
   - global 超过 target 时从 global 选；
   - 否则从全体 resident 选最低 utility。

### 11.3 Fractional charge 与 eviction

一个节点可能同时对 cluster 3 和 cluster 7 计费 `0.6/0.4`。当 cluster 3 超额时，该节点可以成为候选 victim，但 eviction 会一次性删除物理 payload，并同时减去两个 cluster 的 logical usage。绝不只删除一份“cluster 3 副本”，因为副本不存在。

## 12. Ghost profile 上限

如果为所有曾访问节点永久保留 profile，metadata 最终会增长到图节点数。v1 将 profile 上限设为：

```text
max_profiles = max(capacity, capacity * ghost_capacity_multiplier)
```

使用 lazy FIFO queue 记录 `(node_id, last_observed_tick)`。超出上限时：

- resident profile 永远不删除；
- queue 中 tick 已过期的记录跳过；
- 删除最老的 non-resident profile；
- 如果当前所有 profile 都是 resident，则允许 profile 数等于 resident 数。

## 13. 并发模型

QASC v1 使用：

```text
Arc<Mutex<QueryAffinityCache<Data>>>
```

每个搜索线程拥有自己的 `CachedDiskVertexProvider` 和 `QueryRoute`，但共享一个 QASC store。

这是 correctness-first 选择，原因是：

- fractional usage、role transition 和 quota 是全局状态；
- 直接按 node id 分片会把一个 cluster 的 logical quota 拆散；
- 在语义稳定前实现细粒度并发会增加验证难度。

性能测试必须单独报告 QASC policy CPU 和 lock contention。后续可将 store 分片，而 quota controller 和 profile summary 使用周期性合并。

## 14. 与现有缓存策略的集成

新增 cache policy 名称：

```text
qasc
```

`CachePolicyKind::Qasc` 只用于 CLI 选择。它不能直接构造普通 `PolicyCache`，因为普通 policy 没有 query context 或 prototype。

新增 `CachingStrategy::QueryAffinityCache`，包含 QASC 配置和 prototype 文件路径。

`SharedNodeCache` 新增：

```text
QueryAffinity(Arc<Mutex<QueryAffinityCache<Data>>>）
```

现有 static、dynamic、sharded 和 hybrid cache 行为不变。

### 14.1 CLI 示例

```bash
cargo run -q -p diskann-tools --bin search_disk_index --release -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix experiments/cache_eval/sift1m/sift_disk \
  --result_output_prefix experiments/cache_eval/sift1m/results/qasc \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --ground_truth_file experiments/cache_eval/sift1m/groundtruth.bin \
  --search_list 100 \
  --beam_width 4 \
  --recall_at 10 \
  --num_threads 1 \
  --cache_policy qasc \
  --cache_capacity 10000 \
  --qasc_prototypes_file path/to/query_centroids.fbin \
  --qasc_top_m 2 \
  --qasc_global_fraction 0.25 \
  --qasc_global_entropy_threshold 0.75 \
  --qasc_min_observations 2
```

使用 `qasc` 时：

- `--qasc_prototypes_file` 必填；
- `--cache_backend sharded` 不支持；
- `--cache_admission` 不支持，QASC 自己执行 admission；
- `--cache_warmup_nodes` 和 `--cache_static_nodes` v1 不支持；
- `--cache_capacity` 必须大于 0。

## 15. Stats

QASC stats 至少包含：

```text
accesses
hits
misses
admissions
evictions
rejections
global_resident
conditional_resident
profiles
role_promotions
role_demotions
quota_recomputations
borrowed_admissions
```

通用搜索表继续通过 loaded vertices 和 I/O 估计 cache hit rate。单元测试和 debug 输出使用 QASC 专有 stats 检查内部不变量。

## 16. 必须保持的不变量

每次 lookup、role transition、admission 和 eviction 后都应满足：

```text
store.len <= capacity
store.keys == resident_metadata.keys
global_occupancy + conditional_resident_count == store.len
sum(cluster_usage) ~= conditional_resident_count
每个 conditional node 的 sum(charges) ~= 1
每个 global node 的 charges 为空
resident node 必须有 profile
同一个 node id 在 store 中最多出现一次
```

浮点 charge 比较使用小容差，例如 `1e-6`。

## 17. 测试计划

### 17.1 Router

- fbin metadata 和维度校验；
- L2 最近 prototype；
- cosine 与 inner-product 路由；
- top-2 权重和为 1；
- 等距离 tie 的确定性；
- query 中非有限值被拒绝。

### 17.2 Affinity 与 role

- soft route 按权重增加 affinity；
- affinity slots 被限制为配置上限；
- entropy 低的节点为 conditional；
- 跨多个 cluster 的节点晋升 global；
- decay 后 affinity 和 arrival mass 减半；
- role transition 正确更新 usage。

### 17.3 去重和 fractional charging

- 同一节点被多个 cluster 访问时 store 仍只有一份；
- conditional charges 总和为 1；
- eviction 同时删除所有 logical charges；
- global node 不占 conditional quota。

### 17.4 Quota 与 borrowing

- 初始 query mass 相同时 quota 近似相等；
- 热 cluster 获得更高 quota；
- cache 未满时允许借用；
- cache 满后优先从过量 cluster 回收；
- 没有过量 cluster 时 fallback 不违反容量。

### 17.5 Admission 与 eviction

- cache 未满时可填充；
- 满缓存时 observation 不足会拒绝；
- candidate utility 低于 victim 时拒绝；
- utility 更高时正确替换；
- global 和 conditional victim path 都覆盖。

### 17.6 Ghost metadata

- 大量一次性 miss 后 profile 数不超过上限；
- resident profile 不被 ghost eviction 删除；
- 被移除的 ghost 再访问可以重新创建。

### 17.7 Provider 集成

- `DiskAccessor::new` 对每个 query 调用一次 `begin_query`；
- query route 不会泄漏到下一个 query；
- 无 QASC 时默认 `begin_query` 不改变行为；
- 同一 query 的 lookup 和 admission 使用同一 route。

### 17.8 CLI

- `--cache_policy qasc` 缺 prototype 文件时报错；
- 非 QASC policy 不要求 QASC 参数；
- incompatible cache 参数被拒绝；
- prototype 维度与索引维度不一致时报错。

### 17.9 回归与端到端

- 现有 `cache_policy` 测试全部通过；
- no-cache、LRU 和 sharded cache 构建不受影响；
- 小型磁盘索引上 QASC 搜索 Recall 与 no-cache 一致；
- `store.len <= capacity`；
- k-means grouped query workload 上 QASC 能产生非零命中；
- 输出日志记录完整 QASC 配置。

## 18. 提交拆分

为了保持清晰历史，QASC 实现分为以下提交：

1. `Document QASC implementation design`
2. `Add query-affinity cache core`
3. `Route search queries into QASC`
4. `Expose QASC search configuration`
5. `Test QASC integration and invariants`

如果实现过程中某个提交同时修改策略逻辑和实验输出，应继续拆分，避免不可审查的大提交。

## 19. 验收标准

QASC v1 完成必须满足：

- CLI 能在现有 DiskANN 索引上启用 QASC；
- query prototype 路由真实进入 lookup/admission；
- 节点 payload 物理去重；
- global role、fractional charge、quota borrowing 都有测试覆盖；
- 所有容量和 usage 不变量通过；
- 现有缓存策略回归通过；
- 小型端到端搜索 Recall 不下降；
- 文档中的默认值、CLI 和代码一致；
- git 历史按职责拆分且工作树干净。
