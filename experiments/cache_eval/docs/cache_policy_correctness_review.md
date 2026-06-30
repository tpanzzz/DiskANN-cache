# DiskANN 缓存策略正确性 Review

本文审计当前 Rust 实现中的缓存策略是否可被解释为相应论文或公开实现的实现版本。审计范围以当前工作区代码为准，重点文件为：

- `diskann-disk/src/data_model/cache_policy.rs`
- `diskann-disk/src/search/provider/cached_disk_vertex_provider.rs`
- `diskann-disk/src/search/provider/disk_vertex_provider_factory.rs`
- `diskann-tools/src/bin/search_disk_index.rs`
- `diskann-tools/src/bin/cache_trace_replay.rs`

结论等级含义：

- `Exact`: 对当前实验抽象而言等价，例如固定容量、等大小对象、offline trace replay。
- `Mostly consistent`: 核心机制一致，差异主要是工程参数、数据结构或 tie-breaker。
- `Approximate`: 借用了策略思想，但省略、固定或简化了论文中的关键机制。
- `Divergent`: 与论文机制有本质差别，不能直接用论文结论解释结果。
- `Bug-risk`: 发现实现状态可能长期偏离策略定义，建议补测试或修复。

## 总表

| Policy | Paper / source | Original mechanism summary | Current implementation summary | Consistency rating | Key deviations | Recommended action |
|---|---|---|---|---|---|---|
| `NoCache` | 工程基线，无需追溯单篇论文 | 不缓存节点，每次访问落盘。 | live CLI 将 `no_cache` 映射为 `CachingStrategy::None`；replay 中 `NoCache` admission 直接 reject。 | Exact | live 与 replay 的统计路径不同，但语义一致。 | 保持；在论文中说明这是无缓存基线。 |
| `FIFO` | 工程基线 | 按进入缓存的时间淘汰最早进入的 resident。 | `order.push_back` 插入，hit 不移动，victim 为 `order.front()`。 | Exact | 无 aging/admission；符合 FIFO。 | 保持。 |
| `LRU` | 工程基线 | hit 后提升到 MRU，淘汰 LRU resident。 | hit 调 `move_to_back`，victim 为 `order.front()`。 | Mostly consistent | `VecDeque` 线性删除影响性能，不影响语义。 | 保持语义；优化数据结构。 |
| `LFU` | 工程基线 | 按访问频率淘汰最低频 resident。 | `resident_frequency` + lazy `lfu_heap`，hit 加 1，tie 用插入时间。 | Mostly consistent | 无 aging/decay；tie-breaker 不是所有 LFU 变体通用定义。 | 作为 naive LFU 声明；如需公平比较，增加 aging LFU 变体。 |
| `Random` | 工程基线 | 从 resident 中随机选择 victim。 | 固定 seed LCG，在 `order` 中取随机下标。 | Mostly consistent | 可重复伪随机，不是外部随机源；对实验可复现有利。 | 保持；报告 seed 固定。 |
| `StaticBFS` | DiskANN paper: Subramanya et al., 2019, [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) | DiskANN 将入口点/medoid 附近若干跳的图节点及向量驻留内存，查询期只做静态命中判断。 | `build_cache_via_bfs` 从 header medoid 开始 FIFO BFS，按固定节点数填满 `Cache`；查询期 `contains_static` 命中后跳过 disk load。 | Mostly consistent | 论文描述更接近“若干 hop/入口附近”缓存，当前实现是固定数量节点的 BFS 前缀；不动态维护策略状态。 | 作为 `StaticBFS(original)` 基线保留；文档中明确是 count-limited medoid BFS。 |
| `CLOCK` / Second-Chance | Carr and Hennessy, 1981, [SOSP paper PDF](https://web.stanford.edu/~ouster/cgi-bin/cs140-spring20/papers/clock.pdf)；Second-Chance/CLOCK 为经典近似 LRU | resident 有 reference bit；victim 扫描 clock hand，遇到引用位 1 则清零并给 second chance，遇到 0 则淘汰。 | `clock_refs` 记录 bit；`clock_victim` 通过旋转 `order` 模拟 hand。insert/hit 置 true。 | Mostly consistent | hand 用 `VecDeque` 旋转表达；无 dirty/writeback 等 OS 场景状态。 | 保持；增加小容量 wrap-around 测试。 |
| `2Q` | Johnson and Shasha, 1994, [VLDB PDF](https://www.vldb.org/conf/1994/P439.PDF) | `A1in` 保存首次访问 resident FIFO，`Am` 保存多次访问 LRU，`A1out` 保存被 `A1in` 淘汰的 ghost；ghost hit 进入 `Am`。 | `twoq_a1in`、`twoq_am`、`twoq_a1out_set`；`A1in` 固定 `capacity/4`，history 为 `capacity`；`A1in` 溢出立即进入 `A1out`；`A1out` hit 插入 `Am`。 | Mostly consistent | 严格维护 `A1in` 会让纯一次性 cold/warm 序列只占用 `A1in`，直到 ghost hit 填充 `Am`；这符合分段语义，但会改变 warmup 后 resident 数。 | 保持；重跑 trace/live 实验，报告严格分段后的 hit/QPS 变化。 |
| `SLRU` | Karedla, Love, Wherry, 1994, [IEEE Computer page](https://www.computer.org/csdl/magazine/co/1994/03/r3038/13rRUyYjKdx) | 分为 probationary 与 protected；新对象进 probationary，probationary hit 提升 protected，protected LRU 溢出降回 probationary，通常优先淘汰 probationary LRU。 | `slru_probationary`、`slru_protected`，protected 固定 80% 且最多 `capacity - 1`；hit 触发 promotion/demotion；probationary 溢出立即淘汰。 | Mostly consistent | 固定 80/20 分段，不是自适应；严格 probationary 容量会让纯一次性 warm 序列不一定填满整 cache。 | 保持；重跑实验并报告严格分段带来的 warmup resident 数变化。 |
| `LIRS` | Jiang and Zhang, 2002, [author PDF](https://ranger.uta.edu/~sjiang/pubs/papers/jiang02_LIRS.pdf) | 用 stack S 表达 inter-reference recency，queue Q 保存 resident HIR；LIR/HIR 状态随 hit/miss 转换，保留 non-resident HIR 作为 history 并 prune stack。 | `lirs_stack`、`lirs_queue`、`lirs_status`；HIR resident 目标为 1% 且至少 1 个；stack 限制到 `2 * capacity`，优先裁剪 non-resident HIR。 | Approximate | HIR ratio 仍固定不可由 CLI 配置；多处线性扫描；状态表达比作者实现简化。 | 可作为 LIRS-like 报告；如要复现实验，暴露 HIR ratio 参数并补经典 trace 状态对照。 |
| `ARC` | Megiddo and Modha, 2003, [USENIX FAST paper](https://www.usenix.org/legacy/event/fast03/tech/full_papers/megiddo/megiddo.pdf) | `T1/T2` resident 和 `B1/B2` ghost，自适应目标 `p` 在 recency/frequency 间调节；replacement 遵守 `L1/L2` 总容量约束。 | `arc_t1/t2/b1/b2` + sets + `arc_p`；hit 从 `T1/T2` 移 `T2`；ghost hit 调 `p`；新 miss 多数直接 `arc_replace` 后进 `T1`。 | Approximate | 新 miss 的完整 ARC case analysis 被简化；ghost 总长 trimming 只看 `B1+B2 > capacity`；warmup 全部进入 `T1` 且 `p=0`。 | 报告为 ARC approximation；若研究 ARC，补齐完整 case analysis 和 warmup placement 变体。 |
| `GDSF` / GreedyDual | Cherkasova, 1998, [HP Labs PDF](https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/98/HPL-98-69R1.pdf) | GreedyDual-Size-Frequency 用 `H = L + cost * freq / size` 排序，victim 的 H 成为新的 aging clock `L`。 | `gdsf_frequency`、`gdsf_priority`、lazy heap；priority 为 `gdsf_clock + frequency`，victim 时更新 `gdsf_clock`。 | Approximate | 假设 cost=1、size=1；DiskANN cache 容量按节点数，不按字节；节点 adjacency 大小差异未参与。 | 声明为 equal-size GDSF/GDF；如要按内存公平，加入 payload size 估计。 |
| `TinyLFU` | Einziger, Friedman, Manes, 2017, [arXiv](https://arxiv.org/abs/1512.00727) | TinyLFU 是 admission policy：用频率 sketch 和 doorkeeper 估计候选与 victim 频率，只在候选更热时准入，并周期性 aging。 | `FrequencySketch` + exact `HashSet` doorkeeper；sample size `10 * capacity`；满时比较 candidate 与 LRU victim，`<=` reject；warmup 会训练频率模型。 | Approximate | doorkeeper 不是 Bloom filter；counter 宽度、aging、tie-breaker 与论文/常见实现不同。 | 可作为 TinyLFU-inspired admission；建议加入 sketch 参数和训练 trace warmup 变体。 |
| `W-TinyLFU` | Einziger, Friedman, Manes, 2017/2018, [Adaptive Software Cache Management PDF](https://arxiv.org/abs/1512.00727)；Caffeine W-TinyLFU 实现背景见 [Caffeine efficiency wiki](https://github.com/ben-manes/caffeine/wiki/Efficiency) | 新对象先进入 small window LRU；window victim 与 main probation victim 用 TinyLFU admission 比较；main 是 SLRU；一些实现有 window hill-climbing。 | window 固定 1%；main protected 固定 80%；`wt_window/probationary/protected`；候选频率大于 victim 才进入 main；warmup 会训练频率模型。 | Approximate | 无 adaptive window；doorkeeper/sketch 简化；main segment 容量维护依赖 hit/demotion 和 window drain。 | 报告固定-window W-TinyLFU；新增 window size sweep 和训练 trace warmup。 |
| `LeCaR` | Vietri et al., 2018, [USENIX paper](https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf)；作者仓库 [sylab/LeCaR](https://github.com/sylab/LeCaR) | 用 regret/minimize-loss 在 LRU 与 LFU 两个专家间学习权重；被某专家淘汰的页进入该专家 history，ghost hit 对该专家施加随时间折扣的负 reward。 | `lecar_weights`、LRU `order`、LFU heap、两个 history；学习率 0.45，discount `0.005^(1/C)`；按权重随机选择 LRU/LFU victim。 | Mostly consistent | 与 `sylab/LeCaR` 主实现接近；但 Rust LFU tie 用插入时间，随机源不同；Cacheus 仓库中的 LeCaR 变体默认 history 为 `C/2`，当前为 `C`。 | 可声称实现 LeCaR core；报告 tie-breaker/history-size 差异。 |
| `Cacheus` | Rodriguez et al., 2021, [USENIX FAST paper](https://www.usenix.org/conference/fast21/presentation/valdes)；作者仓库 [sylab/cacheus](https://github.com/sylab/cacheus) | Cacheus 演化自 LeCaR，结合 LRU/LFU 学习、LIRS/DLIRS 风格 S/Q resident 结构、demoted/new object history、adaptive learning rate 和 adaptive Q/S size。 | `cacheus_s/q`、LFU heap、LRU/LFU history、weights、`dem_count/nor_count`、`s_limit/q_limit`；初始 Q 为 1%；history 为 `C/2`。 | Approximate | 缺少作者实现中的 per-period adaptive learning rate、pollution/window 统计；eviction/frequency/history 细节与仓库代码不完全一致；live cache 按节点数而非 page/block。 | 报告为 Cacheus-inspired；若要声称复现 FAST'21，应补 adaptive LR 并对照作者 trace。 |
| `BeladyOptimal` / OPT | Belady, 1966, [IBM Systems Journal paper copy](https://users.informatik.uni-halle.de/~hinnebur/Lehre/Web_DBIIb/uebung3_belady_opt_buffer.pdf) | Offline upper bound：淘汰未来最晚再次访问或不再访问的 resident。 | `replay_belady_optimal` 预构建 future positions，满时选 future index 最大者；CLI 禁止 live search 使用。 | Exact | 只适用于 trace replay 和等容量对象；不适用于 live DiskANN 查询。 | 保持为 offline upper bound；不要与 live QPS 混同。 |

## 集成路径审计

### live search cache path

查询期入口在 `CachedDiskVertexProvider::load_vertices` 和 `filter_cached_nodes`：

- static cache: `contains_static` 命中后直接跳过 disk read。
- dynamic cache: 每个 vertex 调 `lookup_dynamic`，进入 `Arc<Mutex<DynamicNodeCache<Data>>>`，调用 `DynamicNodeCache::lookup`。
- miss 后 `process_loaded_node` 从 disk provider 取 vector/adjacency/associated data，构造 `CachedNode`，再调 `admit_dynamic`。

相关代码：

- `cached_disk_vertex_provider.rs:129` `lookup_dynamic`
- `cached_disk_vertex_provider.rs:144` `admit_dynamic`
- `cached_disk_vertex_provider.rs:307` `filter_cached_nodes`
- `cached_disk_vertex_provider.rs:242` `process_loaded_node`
- `cache_policy.rs:1731` `DynamicNodeCache::lookup`
- `cache_policy.rs:1747` `DynamicNodeCache::admit_node`

正确性结论：

- 正常路径下 policy state 与 payload store 同步：admission 返回 evicted 后先 `store.remove`，admitted 后 `store.insert`。
- 动态 hit 会先 `policy.record_access` 再 `store.get_node`。如果未来出现 policy/store 不一致，可能发生“store 有但 policy 不更新”或“policy 有但 store 缺失后 admission no-op”的情况；当前代码没有主动校验这种不变量。
- `Cache::get_node` 每次 dynamic hit 都 clone vector 和 adjacency list，语义安全但不是 static cache 的零拷贝路径。
- trace replay 只跑 `PolicyCache` 元数据，不涉及 `Cache<Data>` payload、锁、clone、disk read，因此只能验证策略命中行为，不能代表 live search QPS。

### StaticBFS 与 warmup path

`DiskVertexProviderFactory::setup_cache` 中：

- `StaticCacheWithBfsNodes` 调 `build_cache_via_bfs` 后构造 immutable `Cache`。
- `DynamicNodeCacheWithBfsWarmup` 先构造相同 BFS `Cache`，再通过 `DynamicNodeCache::from_warm_cache` 对每个 warm id 调 `policy.warm`。

相关代码：

- `disk_vertex_provider_factory.rs:136` static BFS setup
- `disk_vertex_provider_factory.rs:171` dynamic BFS warmup setup
- `disk_vertex_provider_factory.rs:194` `build_cache_via_bfs`
- `cache_policy.rs:1717` `DynamicNodeCache::from_warm_cache`
- `cache_policy.rs:491` `PolicyCache::warm`

正确性结论：

- warmup 不是“把 StaticBFS cache 冻住”，而是把 BFS 序列作为一串 admission 写入动态策略。
- 对 `FIFO/LRU/LFU/CLOCK/LeCaR`，warm nodes 获得普通 insert 状态：frequency=1、reference bit=true、LRU order 为 BFS order。
- 对 `TinyLFU/W-TinyLFU`，warmup 会调用 `record_tiny_lfu_access` 训练 sketch/doorkeeper；这让 BFS warmup 同时初始化 payload 和 admission 频率模型。
- 对 `ARC`，warmup 全部进入 `T1`，`T2/B1/B2` 为空且 `p=0`；这不是 ARC 根据真实 trace 学出的 warm state。
- 对 `2Q/SLRU`，warmup 按严格 segment 容量维护状态；纯一次性 BFS 序列可能不会填满整 cache，直到后续 ghost hit 或 promotion 形成 main/protected resident。

## 逐策略细节

### `NoCache`

原始机制：无缓存基线，不缓存 vector/adjacency payload。

当前机制：

- live CLI 中 `CachePolicyKind::NoCache` 直接返回 `CachingStrategy::None`。
- trace replay 中 `PolicyCache::admit` 对 `NoCache` 直接 reject，并统计 rejections。

一致性：`Exact`。

建议：保留；在实验图表中把 live `NoCache` 与 replay `no_cache` 的统计口径分开说明。

### `FIFO`

原始机制：按 resident 入队顺序淘汰最早进入者，hit 不改变顺序。

当前机制：

- `insert_resident` 写 `order.push_back`。
- `record_access` 对 FIFO 不做移动。
- `victim` 返回 `order.front()`。

一致性：`Exact`。

建议：保留；性能优化时不要把 FIFO 与 LRU 共用“hit move”路径。

### `LRU`

原始机制：hit 或更新后成为 MRU，淘汰 LRU。

当前机制：

- `record_access` 对 `Lru` 调 `move_to_back`。
- `victim` 返回 `order.front()`。
- `move_to_back` 通过 `remove_from_order` 线性扫描再 push back。

一致性：`Mostly consistent`。

差异影响：命中行为正确，但 `O(C)` 删除会放大 CPU 和 lock hold time；在容量 10,000、百万级 access 下会明显影响 QPS。

建议：改为 linked hash set、slot index + prev/next 数组，或 `hashlink::LinkedHashMap` 一类结构；新增 replay hit sequence 回归测试。

### `LFU`

原始机制：淘汰频率最低对象，常见实现会额外定义 aging 或 tie-breaker。

当前机制：

- `resident_frequency` 在 hit 时递增。
- `lfu_heap` lazy push `(frequency, inserted_at, vertex_id)`。
- `lfu_victim` 弹出过期 heap entry，直到找到当前 resident/frequency/inserted_at 匹配项。

一致性：`Mostly consistent`，准确说是 naive LFU。

差异影响：

- 无 aging，长期热点可能粘住。
- heap 每次 hit 追加 entry，频繁 hit 会造成 heap 膨胀，影响 CPU/内存。
- tie 用插入时间，不用 recency 或 object id。

建议：文档中称 naive LFU；性能优化中增加 heap compaction 或 indexed heap。

### `Random`

原始机制：随机选择 resident victim。

当前机制：

- `next_random_index` 使用固定 seed LCG。
- victim 为 `order[idx]`。

一致性：`Mostly consistent`。

差异影响：确定性随机有助于复现，但不是多 run 平均随机。

建议：报告固定 seed；如发表比较可跑多 seed 方差。

### `StaticBFS`

原始来源：DiskANN 论文使用入口附近节点常驻内存来降低初始导航 I/O。

当前机制：

- `build_cache_via_bfs` 从 medoid BFS，beam width 固定为 32。
- 以 `num_nodes_to_cache` 为停止条件，不按 hop boundary 停止。
- static cache 查询期不维护任何 policy state。

一致性：`Mostly consistent`。

差异影响：

- 如果论文实验按 hop 缓存，则当前 count-limited BFS 的节点集合可能不同。
- StaticBFS 查询期只读，和动态策略有本质成本差异；不能只用 hit rate 判断谁更好。

建议：基线名称保留 `StaticBFS(original)`，并注明“当前 Rust 实现为 medoid BFS 前 N 个节点”。

### `CLOCK`

原始机制：reference bit + clock hand 的 second chance。

当前机制：

- insert/hit 将 `clock_refs[id] = true`。
- victim 扫描 `order`：bit true 则清零并移到尾部，bit false 则选为 victim。

一致性：`Mostly consistent`。

差异影响：没有 OS page dirty/writeback 状态；对 DiskANN node cache 不需要。

建议：保留；补一个多轮 wrap-around 测试，确保连续 second chance 后最终淘汰正确。

### `2Q`

原始机制：`A1in` 捕获一次性访问，`A1out` ghost 识别第二次访问，`Am` 保存多次访问热点。

当前机制：

- `A1in` 容量固定为 `capacity / 4`。
- `A1out` 容量固定为 `capacity`。
- hit in `Am` 移到 MRU；hit in `A1in` 不提升；miss in `A1out` 插入 `Am`。
- `A1in` 溢出后立即将其 LRU resident 推入 `A1out`，保持 `A1in <= Kin`。

一致性：`Mostly consistent`。

关键偏差：

- 严格 `A1in` 容量意味着纯一次性 cold/warm 序列可能只保留 `A1in`，直到 ghost hit 使对象进入 `Am`。
- 当前仍使用 `VecDeque` 线性删除，影响性能而非语义。

影响：

- 修复后不会再出现 `A1in` 长期超过配置容量的问题。
- warmup resident 数可能低于 cache capacity；这会改变先前实验结果，需要重跑。

建议：

- 保留当前不变量测试：cold insert 和 warm insert 后 `A1in <= Kin`。
- 重跑 trace replay 和 live search，更新旧 CSV。

### `SLRU`

原始机制：probationary/protected 双段；新对象进 probationary，第二次命中提升 protected；protected overflow 降级到 probationary；优先淘汰 probationary。

当前机制：

- protected 容量固定 80%，且为 probationary 保留至少 1 个 slot。
- probationary hit 提升 protected；protected overflow 调 `slru_demote_if_needed`。
- probationary overflow 立即淘汰 probationary LRU。

一致性：`Mostly consistent`。

关键偏差：

- 固定 80/20 分段，不随 workload 自适应。
- 纯一次性 cold/warm 序列可能只保留 probationary segment，直到 hit 触发 promotion。

影响：

- 修复后 probationary/protected 不会长期超过配置容量。
- warmup resident 数可能低于 cache capacity；需要重跑实验确认命中率和 QPS。

建议：

- 保留 segment 容量不变量测试。
- 若需要复现实验论文，补充不同 protected ratio 的 sweep。

### `LIRS`

原始机制：用 reuse distance 思想区分 LIR/HIR。LIR 长期驻留；HIR resident 数量很小；non-resident HIR 留在 stack 中帮助识别重新变热对象。

当前机制：

- `lirs_stack` 作为 stack S，`lirs_queue` 作为 resident HIR queue。
- `lirs_status` 记录 LIR/HIR resident/HIR non-resident。
- HIR resident 目标为 1% 且至少 1 个；`lirs_lir_capacity = capacity - hirs_capacity`。
- stack 限制为 `2 * capacity`，优先移除 non-resident HIR。

一致性：`Approximate`。

关键偏差：

- HIR ratio 固定为 1%，尚未暴露为 CLI/实验参数。
- 多个操作使用 `VecDeque` 线性扫描。

影响：

- 固定 HIR ratio 在不同 workload 上可能不是最优。
- CPU 开销明显高于简单策略。

建议：

- 参数化 HIR ratio。
- 补 non-resident HIR hit 回归测试和经典 trace 对照。

### `ARC`

原始机制：ARC 用 `T1` 表示 recent resident，`T2` 表示 frequent resident，`B1/B2` 为 ghost；`p` 自适应控制 T1 目标大小。

当前机制：

- `arc_t1/t2/b1/b2` 和 set 并存。
- hit in `T1/T2` 进入或保持 `T2`。
- ghost hit 调整 `p` 并插入 `T2`。
- new miss 在满时直接 `arc_replace`，然后插入 `T1`。

一致性：`Approximate`。

关键偏差：

- 原 ARC 对 `|T1|+|B1|`、`|T1|+|T2|+|B1|+|B2|` 有完整 case analysis；当前实现简化。
- `B1/B2` trimming 只在 `B1+B2 > capacity` 时执行。
- warmup 全部放入 `T1`，没有已学习的 `T2/B1/B2`。

影响：

- 可以体现 ARC 的 ghost/p 调节思想，但不能保证完全符合 ARC 状态机。
- warmup 结果不能解释为“ARC 已经被 BFS 训练好”。

建议：

- 若 ARC 是重点，按论文伪代码补齐 case analysis。
- 为 warmup 做变体：BFS 节点进 `T2`、按 BFS depth 分配 `T1/T2`、或先 replay 训练 trace。

### `GDSF`

原始机制：priority = aging clock + cost * frequency / size；淘汰最低 priority 并将 aging clock 设为 victim priority。

当前机制：

- `priority = gdsf_clock + frequency`。
- `gdsf_heap` lazy 维护 priority。
- victim 时更新 `gdsf_clock`。

一致性：`Approximate`。

关键偏差：

- size/cost 固定为 1。
- DiskANN cache 容量以节点数计，不以字节计；邻接表长度差异未进入决策。

影响：

- 如果节点 payload 大小近似，当前实现可视为 equal-size GDSF。
- 如果 adjacency list 变长明显，GDSF 的“size-aware”优势不会体现。

建议：

- 报告为 equal-size GDSF。
- 如要按内存约束比较，应把 vector + adjacency payload 字节数作为 size。

### `TinyLFU`

原始机制：TinyLFU 通常不是完整 eviction policy，而是 admission policy；用 sketch 估计 candidate/victim 频率，配合 LRU/SLRU 等 eviction。

当前机制：

- `record_tiny_lfu_access` 在每次访问时更新 exact doorkeeper 和 sketch。
- warmup 也调用 `record_tiny_lfu_access`，因此 BFS warmup 会训练频率模型。
- cache 未满时直接 admit。
- cache 满时选择 LRU victim，若 `candidate_freq <= victim_freq` 则 reject。

一致性：`Approximate`。

关键偏差：

- doorkeeper 是 exact `HashSet`，不是 Bloom filter。
- sketch counter、aging 与论文/常见实现不同。
- victim 固定为 LRU，不支持对接其他 eviction policy。

影响：

- 命中率趋势可作为 TinyLFU admission 的近似参考。
- rejected admission 很多，live search 中仍然要先 disk read 才能决定 reject，I/O 已发生，QPS 收益有限。

建议：

- 记录 rejections 和“rejected after disk read”的成本。
- 增加 trace-trained sketch warmup，与 BFS warmup 分开比较。

### `W-TinyLFU`

原始机制：小 window 接纳新对象，window victim 与 main probation victim 比较频率；main 通常是 SLRU，protected/probationary 管理长期热点。

当前机制：

- window 容量固定 1%。
- main protected 容量固定 main 的 80%。
- 新对象先进入 window；window 溢出候选与 main victim 比较。
- warmup 会训练 TinyLFU sketch/doorkeeper。

一致性：`Approximate`。

关键偏差：

- 无 adaptive/hill-climbing window。
- segment 操作使用线性删除，命中维护成本高。

影响：

- hit rate 通常比 LRU/StaticBFS 高，但 QPS 可能因维护成本和 `mean_io_us` 增长而下降。
- warmup 现在同时初始化 payload 与 frequency model；旧 CSV 中 warmup 后轻微下降的现象需要重跑确认。

建议：

- 做 `window=0.5%,1%,5%,10%` sweep。
- 比较 BFS warmup 与真实训练 trace warmup。

### `LeCaR`

原始机制：LRU/LFU 两个专家，各有权重；eviction 随权重随机选择专家；如果被某专家淘汰的对象在 history 中再次访问，则对该专家给负 reward，reward 随 eviction age 折扣。

当前机制：

- `lecar_weights` 初始 `[0.5, 0.5]`。
- learning rate 固定 `0.45`。
- discount rate `0.005^(1/capacity)`。
- LRU history 和 LFU history 各 `capacity`。
- eviction 记录 victim frequency 和 evicted_at。

一致性：`Mostly consistent`。

关键偏差：

- Rust LFU tie-breaker 是插入时间；作者仓库 LeCaR entry tie-breaker 使用 object id。
- 随机源是 deterministic LCG，不是 NumPy seed 123。
- `sylab/cacheus` 框架中的 LeCaR 默认 history size 是 `C/2`，`sylab/LeCaR` 主仓库是 `C`；当前 Rust 采用 `C`。

影响：

- 核心学习机制基本对齐。
- 低容量或 tie 多的 workload 中，tie-breaker 会影响结果。

建议：

- 文档中注明对齐的是 `sylab/LeCaR` 版本。
- 增加小 trace 对照作者 Python 输出的测试。

### `Cacheus`

原始机制：Cacheus 结合 LeCaR 风格专家学习和 LIRS/DLIRS 风格 S/Q 结构。作者实现还包含 adaptive learning rate、demoted/new object 计数、history、pollution/visualization 统计。

当前机制：

- `cacheus_s`、`cacheus_q`；初始 Q limit 为 1%。
- LRU victim 取 `Q` front，否则 fallback 到 `S` front。
- LFU victim 复用 lazy `lfu_heap`。
- LRU/LFU history 容量为 `capacity/2`。
- 权重按 `sqrt(2 ln 2 / capacity)` 学习率更新。
- `cacheus_adjust_size` 根据 `dem_count/nor_count` 调整 S/Q limit。

一致性：`Approximate`。

关键偏差：

- 作者代码有 `Cacheus_Learning_Rate.update`，按周期 hit rate 变化自适应 learning rate；Rust 没有 period update，只用公式学习率。
- 作者代码中的 visualization/pollution 不影响核心 replacement，但用于论文分析；Rust 没有。
- eviction 和 history hit 细节并非完全逐行对应。

影响：

- 当前结果可称 Cacheus-inspired，不能直接代表 FAST'21 Cacheus 论文结论。
- 因为同时维护 S/Q、LFU heap、history、weights，CPU 成本高，QPS 下降不意外。

建议：

- 如果要复现 Cacheus，优先补 adaptive learning rate。
- 对照作者 repo 的短 trace 输出：hit/miss/evicted/weights/q_size。

### `BeladyOptimal`

原始机制：offline 知道完整未来访问序列，淘汰未来最晚再用或不再用的对象。

当前机制：

- `future_positions` 记录每个 id 的未来位置队列。
- replay 时每步 pop 当前访问位置。
- cache 满时选 future front 最大或 `usize::MAX` 的 resident。
- live search CLI 拒绝 `belady_opt`。

一致性：`Exact`。

差异影响：

- 等大小对象、固定容量下是 OPT hit upper bound。
- 不能用于 live QPS，也不能与 StaticBFS warmup 的 wall-clock 性能比较。

建议：保留；在性能文档中只作为 trace hit upper bound 使用。

## 需要补的测试

建议新增以下小 trace/unit tests，不需要跑 SIFT1M：

1. 已补：`2Q` cold fill/warm fill 后 `A1in` 不超过配置上限。
2. 已补：`SLRU` cold fill/warm fill 后 probationary/protected 容量不变量。
3. 已补：`TinyLFU/W-TinyLFU` warmup 会训练 sketch/doorkeeper。
4. `ARC` 对论文经典短 trace 的 `T1/T2/B1/B2/p` 状态对照。
5. `LeCaR` 与 `sylab/LeCaR` 在短 trace 上的 weights/history/victim 对照。
6. `Cacheus` 与 `sylab/cacheus` 在短 trace 上的 S/Q size、weights、history hit 对照。
7. policy/store 一致性断言：每次 `admit_node` 后 `policy.len() == store.len()`，每个 store id 都在 policy resident 中。

## 外部来源

- DiskANN: Subramanya et al., 2019, [DiskANN / Rand-NSG NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
- CLOCK/Second-Chance: Carr and Hennessy, 1981, [WSClock paper PDF](https://web.stanford.edu/~ouster/cgi-bin/cs140-spring20/papers/clock.pdf)
- 2Q: Johnson and Shasha, 1994, [VLDB PDF](https://www.vldb.org/conf/1994/P439.PDF)
- SLRU: Karedla, Love, Wherry, 1994, [IEEE Computer page](https://www.computer.org/csdl/magazine/co/1994/03/r3038/13rRUyYjKdx)
- LIRS: Jiang and Zhang, 2002, [author PDF](https://ranger.uta.edu/~sjiang/pubs/papers/jiang02_LIRS.pdf)
- ARC: Megiddo and Modha, 2003, [FAST paper PDF](https://www.usenix.org/legacy/event/fast03/tech/full_papers/megiddo/megiddo.pdf)
- TinyLFU / W-TinyLFU: Einziger, Friedman, Manes, [arXiv paper](https://arxiv.org/abs/1512.00727), and [Caffeine efficiency notes](https://github.com/ben-manes/caffeine/wiki/Efficiency)
- GDSF: Cherkasova, 1998, [HP Labs PDF](https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/98/HPL-98-69R1.pdf)
- Belady OPT: Belady, 1966, [paper copy](https://users.informatik.uni-halle.de/~hinnebur/Lehre/Web_DBIIb/uebung3_belady_opt_buffer.pdf)
- LeCaR: Vietri et al., 2018, [USENIX PDF](https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf), [sylab/LeCaR](https://github.com/sylab/LeCaR)
- Cacheus: Rodriguez et al., 2021, [USENIX FAST page](https://www.usenix.org/conference/fast21/presentation/valdes), [sylab/cacheus](https://github.com/sylab/cacheus)
