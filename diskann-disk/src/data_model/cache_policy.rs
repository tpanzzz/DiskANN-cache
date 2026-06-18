/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, VecDeque},
    fmt,
    str::FromStr,
};

use diskann::{ANNError, ANNResult};
use hashbrown::{HashMap, HashSet};

use crate::data_model::{Cache, CachedNode, GraphDataType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CachePolicyKind {
    NoCache,
    Fifo,
    Lru,
    Lfu,
    Random,
    TinyLfu,
    BeladyOptimal,
}

impl CachePolicyKind {
    pub fn is_online(self) -> bool {
        !matches!(self, Self::BeladyOptimal)
    }
}

impl fmt::Display for CachePolicyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCache => f.write_str("no_cache"),
            Self::Fifo => f.write_str("fifo"),
            Self::Lru => f.write_str("lru"),
            Self::Lfu => f.write_str("lfu"),
            Self::Random => f.write_str("random"),
            Self::TinyLfu => f.write_str("tiny_lfu"),
            Self::BeladyOptimal => f.write_str("belady_opt"),
        }
    }
}

impl FromStr for CachePolicyKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" | "no-cache" | "no_cache" | "nocache" => Ok(Self::NoCache),
            "fifo" => Ok(Self::Fifo),
            "lru" => Ok(Self::Lru),
            "lfu" => Ok(Self::Lfu),
            "random" => Ok(Self::Random),
            "tinylfu" | "tiny-lfu" | "tiny_lfu" => Ok(Self::TinyLfu),
            "opt" | "belady" | "belady-opt" | "belady_opt" => Ok(Self::BeladyOptimal),
            other => Err(format!("unknown cache policy '{other}'")),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CachePolicyStats {
    pub accesses: u64,
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub evictions: u64,
    pub rejections: u64,
}

impl CachePolicyStats {
    pub fn hit_rate(&self) -> f64 {
        if self.accesses == 0 {
            0.0
        } else {
            self.hits as f64 / self.accesses as f64
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdmissionOutcome {
    pub admitted: bool,
    pub evicted: Option<u32>,
    pub rejected: bool,
}

pub struct PolicyCache {
    kind: CachePolicyKind,
    capacity: usize,
    resident: HashSet<u32>,
    order: VecDeque<u32>,
    resident_frequency: HashMap<u32, u64>,
    inserted_at: HashMap<u32, u64>,
    lfu_heap: BinaryHeap<Reverse<(u64, u64, u32)>>,
    frequency_estimate: HashMap<u32, u64>,
    rng_state: u64,
    tick: u64,
}

impl PolicyCache {
    pub fn new(kind: CachePolicyKind, capacity: usize) -> ANNResult<Self> {
        if !kind.is_online() {
            return Err(ANNError::log_index_error(format!(
                "{kind} is an offline replay policy and cannot be used as a live cache"
            )));
        }

        Ok(Self {
            kind,
            capacity,
            resident: HashSet::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            resident_frequency: HashMap::with_capacity(capacity),
            inserted_at: HashMap::with_capacity(capacity),
            lfu_heap: BinaryHeap::with_capacity(capacity),
            frequency_estimate: HashMap::new(),
            rng_state: 0x9e37_79b9_7f4a_7c15,
            tick: 0,
        })
    }

    pub fn kind(&self) -> CachePolicyKind {
        self.kind
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.resident.len()
    }

    pub fn contains(&self, vertex_id: u32) -> bool {
        self.resident.contains(&vertex_id)
    }

    pub fn record_access(&mut self, vertex_id: u32) {
        self.tick = self.tick.saturating_add(1);

        if matches!(self.kind, CachePolicyKind::TinyLfu) {
            *self.frequency_estimate.entry(vertex_id).or_default() += 1;
        }

        if !self.contains(vertex_id) {
            return;
        }

        match self.kind {
            CachePolicyKind::Lru | CachePolicyKind::TinyLfu => {
                Self::move_to_back(&mut self.order, vertex_id);
            }
            CachePolicyKind::Lfu => {
                *self.resident_frequency.entry(vertex_id).or_insert(1) += 1;
                self.push_lfu_entry(vertex_id);
            }
            CachePolicyKind::NoCache
            | CachePolicyKind::Fifo
            | CachePolicyKind::Random
            | CachePolicyKind::BeladyOptimal => {}
        }
    }

    pub fn admit(&mut self, vertex_id: u32) -> AdmissionOutcome {
        if self.capacity == 0 || self.kind == CachePolicyKind::NoCache {
            return AdmissionOutcome {
                rejected: true,
                ..AdmissionOutcome::default()
            };
        }

        if self.contains(vertex_id) {
            return AdmissionOutcome::default();
        }

        if self.len() < self.capacity {
            self.insert_resident(vertex_id);
            return AdmissionOutcome {
                admitted: true,
                ..AdmissionOutcome::default()
            };
        }

        if self.kind == CachePolicyKind::TinyLfu {
            let Some(victim) = self.victim() else {
                return AdmissionOutcome {
                    rejected: true,
                    ..AdmissionOutcome::default()
                };
            };
            let candidate_freq = self.estimated_frequency(vertex_id);
            let victim_freq = self.estimated_frequency(victim);
            if candidate_freq <= victim_freq {
                return AdmissionOutcome {
                    rejected: true,
                    ..AdmissionOutcome::default()
                };
            }
            self.remove_resident(victim);
            self.insert_resident(vertex_id);
            return AdmissionOutcome {
                admitted: true,
                evicted: Some(victim),
                rejected: false,
            };
        }

        let Some(victim) = self.victim() else {
            return AdmissionOutcome {
                rejected: true,
                ..AdmissionOutcome::default()
            };
        };
        self.remove_resident(victim);
        self.insert_resident(vertex_id);

        AdmissionOutcome {
            admitted: true,
            evicted: Some(victim),
            rejected: false,
        }
    }

    pub fn warm(&mut self, vertex_id: u32) -> AdmissionOutcome {
        if self.capacity == 0 || self.kind == CachePolicyKind::NoCache {
            return AdmissionOutcome {
                rejected: true,
                ..AdmissionOutcome::default()
            };
        }

        if self.contains(vertex_id) {
            return AdmissionOutcome::default();
        }

        let evicted = if self.len() == self.capacity {
            let victim = self.victim();
            if let Some(victim) = victim {
                self.remove_resident(victim);
            }
            victim
        } else {
            None
        };

        self.insert_resident(vertex_id);
        AdmissionOutcome {
            admitted: true,
            evicted,
            rejected: false,
        }
    }

    fn insert_resident(&mut self, vertex_id: u32) {
        self.tick = self.tick.saturating_add(1);
        self.resident.insert(vertex_id);
        self.order.push_back(vertex_id);
        self.resident_frequency.insert(vertex_id, 1);
        self.inserted_at.insert(vertex_id, self.tick);
        if self.kind == CachePolicyKind::Lfu {
            self.push_lfu_entry(vertex_id);
        }
    }

    fn remove_resident(&mut self, vertex_id: u32) -> bool {
        if !self.resident.remove(&vertex_id) {
            return false;
        }
        Self::remove_from_order(&mut self.order, vertex_id);
        self.resident_frequency.remove(&vertex_id);
        self.inserted_at.remove(&vertex_id);
        true
    }

    fn victim(&mut self) -> Option<u32> {
        match self.kind {
            CachePolicyKind::NoCache | CachePolicyKind::BeladyOptimal => None,
            CachePolicyKind::Fifo | CachePolicyKind::Lru | CachePolicyKind::TinyLfu => {
                self.order.front().copied()
            }
            CachePolicyKind::Lfu => self.lfu_victim(),
            CachePolicyKind::Random => {
                if self.order.is_empty() {
                    None
                } else {
                    let idx = self.next_random_index(self.order.len());
                    self.order.get(idx).copied()
                }
            }
        }
    }

    fn push_lfu_entry(&mut self, vertex_id: u32) {
        let frequency = self
            .resident_frequency
            .get(&vertex_id)
            .copied()
            .unwrap_or_default();
        let inserted_at = self
            .inserted_at
            .get(&vertex_id)
            .copied()
            .unwrap_or_default();
        self.lfu_heap
            .push(Reverse((frequency, inserted_at, vertex_id)));
    }

    fn lfu_victim(&mut self) -> Option<u32> {
        while let Some(Reverse((frequency, inserted_at, vertex_id))) = self.lfu_heap.pop() {
            if !self.resident.contains(&vertex_id) {
                continue;
            }
            if self
                .resident_frequency
                .get(&vertex_id)
                .copied()
                .unwrap_or_default()
                != frequency
            {
                continue;
            }
            if self
                .inserted_at
                .get(&vertex_id)
                .copied()
                .unwrap_or_default()
                != inserted_at
            {
                continue;
            }
            return Some(vertex_id);
        }
        None
    }

    fn estimated_frequency(&self, vertex_id: u32) -> u64 {
        self.frequency_estimate
            .get(&vertex_id)
            .copied()
            .unwrap_or_default()
    }

    fn next_random_index(&mut self, len: usize) -> usize {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.rng_state as usize) % len
    }

    fn move_to_back(order: &mut VecDeque<u32>, vertex_id: u32) {
        if Self::remove_from_order(order, vertex_id) {
            order.push_back(vertex_id);
        }
    }

    fn remove_from_order(order: &mut VecDeque<u32>, vertex_id: u32) -> bool {
        let Some(idx) = order.iter().position(|id| *id == vertex_id) else {
            return false;
        };
        order.remove(idx);
        true
    }
}

pub struct DynamicNodeCache<Data: GraphDataType<VectorIdType = u32>> {
    store: Cache<Data>,
    policy: PolicyCache,
    stats: CachePolicyStats,
}

impl<Data> DynamicNodeCache<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    pub fn new(dimension: usize, capacity: usize, policy: CachePolicyKind) -> ANNResult<Self> {
        Ok(Self {
            store: Cache::new(dimension, capacity)?,
            policy: PolicyCache::new(policy, capacity)?,
            stats: CachePolicyStats::default(),
        })
    }

    pub fn from_warm_cache(
        policy: CachePolicyKind,
        capacity: usize,
        warm_cache: &Cache<Data>,
    ) -> ANNResult<Self> {
        let mut cache = Self::new(warm_cache.dimension(), capacity, policy)?;
        for id in warm_cache.ids().iter().take(capacity) {
            if let Some(node) = warm_cache.get_node(id) {
                cache.warm_node(*id, node)?;
            }
        }
        Ok(cache)
    }

    pub fn lookup(&mut self, vertex_id: &Data::VectorIdType) -> Option<CachedNode<Data>> {
        self.stats.accesses += 1;
        self.policy.record_access(*vertex_id);
        let node = self.store.get_node(vertex_id);
        if node.is_some() {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }
        node
    }

    pub fn contains(&self, vertex_id: &Data::VectorIdType) -> bool {
        self.store.contains(vertex_id)
    }

    pub fn admit_node(
        &mut self,
        vertex_id: &Data::VectorIdType,
        node: CachedNode<Data>,
    ) -> ANNResult<()> {
        if self.store.contains(vertex_id) {
            return self.store.insert(
                vertex_id,
                &node.vector,
                node.adjacency_list,
                node.associated_data,
            );
        }

        let outcome = self.policy.admit(*vertex_id);
        if let Some(evicted) = outcome.evicted {
            self.store.remove(&evicted);
            self.stats.evictions += 1;
        }

        if outcome.admitted {
            self.store.insert(
                vertex_id,
                &node.vector,
                node.adjacency_list,
                node.associated_data,
            )?;
            self.stats.admissions += 1;
        } else if outcome.rejected {
            self.stats.rejections += 1;
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.store.capacity()
    }

    pub fn policy_kind(&self) -> CachePolicyKind {
        self.policy.kind()
    }

    pub fn stats(&self) -> CachePolicyStats {
        self.stats
    }

    fn warm_node(
        &mut self,
        vertex_id: Data::VectorIdType,
        node: CachedNode<Data>,
    ) -> ANNResult<()> {
        let outcome = self.policy.warm(vertex_id);
        if let Some(evicted) = outcome.evicted {
            self.store.remove(&evicted);
        }
        if outcome.admitted {
            self.store.insert(
                &vertex_id,
                &node.vector,
                node.adjacency_list,
                node.associated_data,
            )?;
        }
        Ok(())
    }
}

pub fn replay_online_policy(
    policy: CachePolicyKind,
    capacity: usize,
    trace: &[u32],
) -> ANNResult<CachePolicyStats> {
    let mut cache = PolicyCache::new(policy, capacity)?;
    let mut stats = CachePolicyStats::default();

    for vertex_id in trace {
        stats.accesses += 1;
        cache.record_access(*vertex_id);
        if cache.contains(*vertex_id) {
            stats.hits += 1;
            continue;
        }

        stats.misses += 1;
        let outcome = cache.admit(*vertex_id);
        if outcome.admitted {
            stats.admissions += 1;
        }
        if outcome.evicted.is_some() {
            stats.evictions += 1;
        }
        if outcome.rejected {
            stats.rejections += 1;
        }
    }

    Ok(stats)
}

pub fn replay_belady_optimal(capacity: usize, trace: &[u32]) -> CachePolicyStats {
    let mut stats = CachePolicyStats::default();
    if capacity == 0 {
        stats.accesses = trace.len() as u64;
        stats.misses = trace.len() as u64;
        stats.rejections = trace.len() as u64;
        return stats;
    }

    let mut future_positions: HashMap<u32, VecDeque<usize>> = HashMap::new();
    for (idx, vertex_id) in trace.iter().enumerate() {
        future_positions
            .entry(*vertex_id)
            .or_default()
            .push_back(idx);
    }

    let mut resident = HashSet::with_capacity(capacity);
    for (idx, vertex_id) in trace.iter().enumerate() {
        stats.accesses += 1;
        if let Some(positions) = future_positions.get_mut(vertex_id) {
            debug_assert_eq!(positions.front().copied(), Some(idx));
            positions.pop_front();
        }

        if resident.contains(vertex_id) {
            stats.hits += 1;
            continue;
        }

        stats.misses += 1;
        if resident.len() == capacity {
            let Some(victim) = resident.iter().copied().max_by_key(|id| {
                future_positions
                    .get(id)
                    .and_then(|positions| positions.front().copied())
                    .unwrap_or(usize::MAX)
            }) else {
                stats.rejections += 1;
                continue;
            };
            resident.remove(&victim);
            stats.evictions += 1;
        }
        resident.insert(*vertex_id);
        stats.admissions += 1;
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_lru_counts_expected_hits() {
        let trace = [1, 2, 1, 3, 1, 2];
        let stats = replay_online_policy(CachePolicyKind::Lru, 2, &trace).unwrap();

        assert_eq!(stats.accesses, 6);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 4);
    }

    #[test]
    fn replay_fifo_counts_expected_hits() {
        let trace = [1, 2, 1, 3, 1, 2];
        let stats = replay_online_policy(CachePolicyKind::Fifo, 2, &trace).unwrap();

        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 5);
    }

    #[test]
    fn replay_lfu_keeps_frequent_item() {
        let trace = [1, 2, 1, 3, 1, 2];
        let stats = replay_online_policy(CachePolicyKind::Lfu, 2, &trace).unwrap();

        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 4);
    }

    #[test]
    fn belady_is_at_least_lru_on_same_trace() {
        let trace = [1, 2, 3, 1, 2, 4, 1, 2, 3, 4];
        let lru = replay_online_policy(CachePolicyKind::Lru, 3, &trace).unwrap();
        let opt = replay_belady_optimal(3, &trace);

        assert!(opt.hits >= lru.hits);
    }
}
