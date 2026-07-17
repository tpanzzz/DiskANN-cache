# Spatial-locality-aware caching for disk-resident graph ANNS

## 1. Reframing the observed locality

The current SIFT1M experiments show that dynamic node caches obtain much higher
hit rates after test queries are sorted by k-means cluster. At cache capacity
10,000, LRU rises from 9.21% on the original order to 37.09% on the k-means
order. Belady's offline upper bound rises from 24.70% to 47.12%.

Calling this only temporal locality is incomplete. The reuse appears over time,
but its source is geometric:

- nearby queries in the vector space tend to enter similar graph regions;
- their graph traversals expand overlapping full-precision nodes;
- putting nearby queries next to one another shortens the reuse distance of
  those nodes;
- a conventional temporal cache then captures the reuse indirectly.

The useful workload variable is therefore not only the vertex access sequence
`v_1, v_2, ...`, but the joint process `(q_t, E(q_t))`, where `q_t` is the query
vector and `E(q_t)` is the set or ordered sequence of graph nodes expanded for
that query.

For a query-region label `z`, define a node affinity

```text
A[v, z] = P(v is expanded | query belongs to region z).
```

A spatially aligned cache should predict future reuse from `z` and `A[v,z]`,
not only from recency or global frequency of `v`.

## 2. Why physically splitting one cache per cluster is an incomplete design

The proposed cluster-partitioned cache is a strong baseline because it is easy
to explain and directly tests the hypothesis. It nevertheless has several
structural problems.

### 2.1 Duplicate nodes

Entry-region nodes, medoid-near nodes, hubs, and bridge nodes may be useful to
many query clusters. Physically copying them into many cluster caches wastes
capacity. This is particularly harmful when cache entries contain both a
full-precision vector and an adjacency list.

### 2.2 Hard routing errors

A query near a cluster boundary can traverse nodes associated with two or more
regions. A single hard cluster assignment sends all admissions to one cache and
can evict useful nodes from the wrong partition.

### 2.3 Capacity does not follow cluster population alone

The number of historical queries in a cluster estimates arrival probability,
but not the cache capacity needed by that cluster. Two equally popular clusters
can have different graph working-set sizes, path overlap, path lengths, and
node-sharing rates.

### 2.4 Fixed clusters become stale

Production query distributions drift. A fixed offline partition can leave
capacity stranded in inactive clusters while a new cluster thrashes.

### 2.5 The physical I/O unit may be a page

DiskANN logically caches nodes, but SSD reads and graph layouts are page or
sector oriented. A node-optimal partition may not minimize page reads if useful
nodes from one query region are scattered across pages.

For these reasons, physically independent caches should be implemented as an
experimental baseline, not the final architecture.

## 3. Recommended design: Query-Affinity Shared Cache

The recommended design is a single deduplicated physical cache with
query-conditioned admission, eviction, and logical capacity accounting. This
document calls it Query-Affinity Shared Cache, or QASC.

### 3.1 Components

1. **Query router**

   Maintain `K` query prototypes. For each query, return either a hard cluster
   id or a soft top-`m` distribution `p(z | q)`. Soft routing is preferred near
   boundaries.

2. **Node-affinity statistics**

   Maintain exponentially decayed counts:

   ```text
   expansion_count[v, z]
   hit_count[v, z]
   last_access_epoch[v, z]
   ```

   A compressed top-cluster list per node is preferable to a dense
   `num_nodes x K` matrix.

3. **Single physical node store**

   Store each graph node at most once. A cached node can serve every query
   cluster. Cluster ownership is metadata, not a duplicate payload.

4. **Global and conditional logical tiers**

   Reserve part of the capacity for globally reusable nodes and the remainder
   for cluster-specific nodes. These can be two policy lists over one payload
   store rather than two copies of data.

5. **Quota controller**

   Logical cluster quotas constrain which cluster can cause eviction, but unused
   capacity is borrowable. Quotas are periodically updated from measured
   marginal hit gain.

### 3.2 Global versus cluster-specific nodes

For node `v`, normalize its affinity distribution across clusters:

```text
p(z | v) = expansion_count[v,z] / sum_j expansion_count[v,j].
```

Use its entropy to distinguish roles:

```text
H(v) = -sum_z p(z|v) log p(z|v).
```

- high support and high entropy: global/shared node;
- high support and low entropy: cluster-specific node;
- low support: weak admission candidate;
- high early-hop frequency: likely navigation backbone node.

This directly addresses duplicated entry-region nodes: they are stored once in
the global tier. Cluster caches retain only the residual region-specific working
set.

### 3.3 Admission score

A practical first score is

```text
utility(v, q) =
    io_cost(v) / entry_size(v)
    * sum_z p(z | q) * reuse_probability(v, z)
    * confidence(q)
    - admission_penalty(v).
```

`reuse_probability(v,z)` can initially be an exponentially decayed expansion
frequency. Later versions can predict reuse distance or expected hits within a
time horizon.

Admission should be compared against the current victim's utility. This avoids
polluting a cluster quota with every expanded node.

### 3.4 Eviction and logical cost sharing

When a node is useful to multiple clusters, storing it once creates an
accounting question. Charge its logical size fractionally:

```text
charge(v,z) = size(v) * demand(v,z) / sum_j demand(v,j).
```

Physical occupancy remains one entry. Logical charges allow each cluster to
consume a quota without making copies. Eviction chooses a node with low global
marginal utility while respecting minimum cluster service levels.

### 3.5 Capacity allocation

Do not use only `C_z proportional to number_of_queries_z`. That is a useful
baseline, but the preferred objective is marginal hit gain.

For each cluster, measure a hit curve `H_z(c)` from trace replay. Allocate the
next cache unit to the cluster with the largest

```text
arrival_rate_z * (H_z(c + delta) - H_z(c)) / delta.
```

This is a water-filling allocation over empirical miss-ratio curves. It
automatically gives more capacity to clusters with large reusable working sets
and avoids over-allocating clusters whose hit curves saturate early.

Useful baselines are:

```text
C_z proportional to n_z
C_z proportional to sqrt(n_z)
C_z proportional to n_z^alpha, alpha in {0.5, 0.75, 1.0}
C_z proportional to arrival_rate_z * unique_working_set_z
```

A robust implementation should also have:

- a minimum quota per active cluster;
- a shared elastic pool, initially 20-40% of total capacity;
- borrowing when another cluster is idle;
- hysteresis so quotas do not oscillate every epoch.

## 4. Where the query prototypes should come from

The preferred source order is:

1. **Historical production queries with preserved arrival order.**

   This is the only source that directly captures demand and sequence behavior.

2. **A representative offline learn-query set.**

   This is suitable for bootstrap if its distribution is validated against
   live queries.

3. **Training queries from the retrieval model.**

   These can bootstrap semantic regions, but training sampling may deliberately
   differ from serving traffic.

4. **Base vectors or sampled graph nodes as pseudo-queries.**

   This estimates corpus geometry, not demand. It should be a cold-start
   fallback only.

The recommended lifecycle is hybrid:

- initialize prototypes from offline learn queries or a base sample;
- collect a bounded online query window;
- update prototypes with streaming or periodic mini-batch k-means;
- compare old and new cluster distributions;
- switch only when drift exceeds a threshold and the new model is stable.

Useful drift signals include centroid movement, occupancy Jensen-Shannon
divergence, routing confidence, cluster birth/death, and a drop in predicted
versus realized reuse.

## 5. Alternative designs beyond cluster-partitioned caching

### 5.1 Global node-affinity optimization

Avoid logical partitions entirely. Estimate expected future hits for every node
under the current query mixture and keep the globally best nodes under the
capacity constraint. Cluster information is only a feature in the utility
model. This is the cleanest deduplicated design.

### 5.2 Hierarchical query-region cache

Use global, coarse-cluster, and leaf-cluster levels. A node is assigned to the
highest level where it remains reusable. This naturally represents navigation
backbone nodes, broad semantic regions, and fine local neighborhoods.

### 5.3 Query-conditioned reuse-distance prediction

Train a lightweight model to predict whether an expanded node will be reused
within a horizon. Features can include query-centroid distances, node affinity,
hop number, degree, node frequency, current reuse distance, and cache pressure.
This is a learning-augmented replacement policy rather than a fixed heuristic.

### 5.4 Path-sketch cache and prefetch

For each query region, learn a compact sketch of high-probability expansion
nodes or pages. Use it to pre-admit or asynchronously prefetch likely nodes
before the graph reaches them. Prefetch must be evaluated by useful-prefetch
rate and extra I/O, not only latency.

### 5.5 Transition-aware region warming

If the real query stream has stable cluster transitions, learn
`P(z_{t+1} | z_t)` and preserve or prewarm the likely next region. This uses
sequence structure beyond independent cluster popularity.

### 5.6 Page-aware affinity cache

Aggregate `A[v,z]` by disk page and optimize expected page reuse. A page can be
valuable even when no individual node is globally frequent. This design aligns
the policy with actual SSD I/O and Starling-style layouts.

### 5.7 Search-stage-aware cache

Classify expansions by hop or search stage. Early navigation nodes are likely
global; late-stage nodes are more query-region specific. A simple global-early
plus cluster-late policy may capture much of QASC's value with less metadata.

### 5.8 Semantic result caching

Cache final result sets and allow a nearby query to reuse or refine them. This
is conceptually related to semantic/similarity caching, but it changes retrieval
semantics and recall. It should be evaluated separately from graph-node caching.

## 6. Recommended implementation sequence

### Phase A: measurement before policy code

1. Preserve query boundaries and cluster ids in expansion traces.
2. Build the node-cluster affinity matrix.
3. Measure node affinity entropy, cross-cluster overlap, and per-cluster
   miss-ratio curves.
4. Quantify how many bytes independent physical caches would duplicate.

### Phase B: baselines

1. Hard cluster routing with fixed proportional partitions.
2. Hard routing with a global shared tier.
3. Soft top-2 routing with one deduplicated store.
4. Capacity rules `n`, `sqrt(n)`, `n^0.75`, and water-filling.

### Phase C: adaptive policy

1. Decayed node-cluster affinity.
2. Elastic quotas and borrowing.
3. Drift detection and periodic centroid refresh.
4. Page-aware utility and optional prefetch.

## 7. Evaluation matrix

Always report:

- dataset and source;
- whether query order is original, shuffled, k-means grouped, or real temporal;
- query clustering source and update cadence;
- routing confidence and cluster occupancy;
- cache capacity in nodes and bytes;
- global/conditional split;
- duplicate bytes avoided;
- hit rate, mean I/O, QPS, latency percentiles, and Recall@10;
- metadata memory and policy CPU cost;
- cold/warm state and thread count.

Required ablations:

- one physical cache per cluster versus deduplicated shared store;
- no global tier versus global tier;
- hard versus soft routing;
- proportional versus water-filled capacity;
- offline prototypes versus online-updated prototypes;
- node-aware versus page-aware utility;
- stable workload versus abrupt and gradual drift.

## 8. Related research and how it maps to this problem

- Dar et al., *Semantic Data Caching and Replacement*, VLDB 1996. Semantic
  regions rather than object identity determine reuse. This motivates routing
  by query region, although QASC caches graph nodes rather than query answers.
- Garetto, Leonardi, and Neglia, *Similarity Caching: Theory and Algorithms*,
  arXiv:1912.03888. This formalizes caches where similar objects can satisfy a
  request. It is directly relevant to defining query-space neighborhoods and
  approximate reuse.
- Vietri et al., *LeCaR: Learning Cache Replacement with Regret Minimization*,
  HotStorage 2018. LeCaR learns between recency and frequency experts, but its
  state is access-sequence based and does not condition on query geometry.
- Rodriguez et al., *CACHEUS: An Adaptive Cache Replacement Policy for a
  Changing World*, FAST 2021. CACHEUS adapts among temporal policies; query
  region can be added as a contextual feature or expert-selection signal.
- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor
  Search on a Single Node*, NeurIPS 2019. DiskANN's static BFS cache captures
  globally useful navigation nodes, which corresponds to QASC's global tier.
- Chen et al., *SPANN: Highly-efficient Billion-scale Approximate Nearest
  Neighbor Search*, NeurIPS 2021, arXiv:2111.08566. SPANN uses coarse regions
  and boundary replication; the region-routing and overlap issues are analogous
  to query-cluster cache routing.
- Wang et al., *Starling: An I/O-Efficient Disk-Resident Graph Index Framework
  for High-Dimensional Vector Similarity Search on Data Segment*, PACMMOD 2024,
  DOI 10.1145/3639269. Starling shows that graph layout and block I/O locality
  must be considered together with caching.
- Aumuller, Bernhardsson, and Faithfull, *ANN-Benchmarks*, Information Systems
  2019, DOI 10.1016/j.is.2019.02.006. Its datasets are suitable for ANN quality
  and speed comparisons, but most query orders are not temporal workloads.

Useful links:

- https://arxiv.org/abs/1912.03888
- https://www.usenix.org/conference/hotstorage18/presentation/vietri
- https://www.usenix.org/conference/fast21/presentation/rodriguez
- https://arxiv.org/abs/1907.12320
- https://arxiv.org/abs/2111.08566
- https://doi.org/10.1145/3639269

## 9. Current recommendation

Implement the simple per-cluster cache as a falsifiable baseline, but direct the
main design toward a deduplicated Query-Affinity Shared Cache with:

- a global navigation tier;
- soft query routing;
- decayed node-cluster affinity;
- elastic logical quotas;
- water-filling capacity allocation;
- online drift detection;
- eventual page-aware scoring.

This aligns the cache with the actual causal structure observed in the current
experiments: query geometry determines path overlap, and path overlap determines
future node reuse.
