/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, VecDeque},
    fmt,
    str::FromStr,
    sync::{Arc, Mutex},
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
    Clock,
    TwoQ,
    Slru,
    Lirs,
    Arc,
    Gdsf,
    TinyLfu,
    WTinyLfu,
    LeCar,
    Cacheus,
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
            Self::Clock => f.write_str("clock"),
            Self::TwoQ => f.write_str("2q"),
            Self::Slru => f.write_str("slru"),
            Self::Lirs => f.write_str("lirs"),
            Self::Arc => f.write_str("arc"),
            Self::Gdsf => f.write_str("gdsf"),
            Self::TinyLfu => f.write_str("tiny_lfu"),
            Self::WTinyLfu => f.write_str("w_tiny_lfu"),
            Self::LeCar => f.write_str("lecar"),
            Self::Cacheus => f.write_str("cacheus"),
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
            "clock" | "second-chance" | "second_chance" => Ok(Self::Clock),
            "2q" | "twoq" | "two-q" | "two_q" => Ok(Self::TwoQ),
            "slru" | "segmented-lru" | "segmented_lru" => Ok(Self::Slru),
            "lirs" => Ok(Self::Lirs),
            "arc" => Ok(Self::Arc),
            "gdsf" | "greedydual" | "greedy-dual" | "greedy_dual" => Ok(Self::Gdsf),
            "tinylfu" | "tiny-lfu" | "tiny_lfu" => Ok(Self::TinyLfu),
            "wtinylfu" | "w-tinylfu" | "w_tinylfu" | "w-tiny-lfu" | "w_tiny_lfu" => {
                Ok(Self::WTinyLfu)
            }
            "lecar" | "le-car" | "le_car" => Ok(Self::LeCar),
            "cacheus" => Ok(Self::Cacheus),
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

    pub fn merge(&mut self, other: Self) {
        self.accesses += other.accesses;
        self.hits += other.hits;
        self.misses += other.misses;
        self.admissions += other.admissions;
        self.evictions += other.evictions;
        self.rejections += other.rejections;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdmissionOutcome {
    pub admitted: bool,
    pub evicted: Option<u32>,
    pub rejected: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwoQSegment {
    A1In,
    Am,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlruSegment {
    Probationary,
    Protected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LirsStatus {
    Lir,
    HirResident,
    HirNonResident,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WTinyLfuSegment {
    Window,
    Probationary,
    Protected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeCarExpert {
    Lru,
    Lfu,
}

struct FrequencySketch {
    width: usize,
    counters: Vec<Vec<u16>>,
    seeds: [u64; 4],
}

impl FrequencySketch {
    fn new(capacity: usize) -> Self {
        let width = capacity.next_power_of_two().max(64);
        Self {
            width,
            counters: vec![vec![0; width]; 4],
            seeds: [
                0x9e37_79b9_7f4a_7c15,
                0xbf58_476d_1ce4_e5b9,
                0x94d0_49bb_1331_11eb,
                0x2545_f491_4f6c_dd1d,
            ],
        }
    }

    fn increment(&mut self, id: u32) {
        for (row, seed) in self.counters.iter_mut().zip(self.seeds) {
            let idx = Self::hash(id, seed) & (self.width - 1);
            if row[idx] < u16::MAX {
                row[idx] += 1;
            }
        }
    }

    fn estimate(&self, id: u32) -> u64 {
        self.counters
            .iter()
            .zip(self.seeds)
            .map(|(row, seed)| row[Self::hash(id, seed) & (self.width - 1)] as u64)
            .min()
            .unwrap_or_default()
    }

    fn reset(&mut self) {
        for row in &mut self.counters {
            for counter in row {
                *counter /= 2;
            }
        }
    }

    fn hash(id: u32, seed: u64) -> usize {
        let mut x = id as u64 ^ seed;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (x ^ (x >> 31)) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheusSegment {
    S,
    Q,
}

#[derive(Clone, Copy, Debug)]
struct HistoryEntry {
    frequency: u64,
    evicted_at: u64,
    is_new: bool,
}

pub struct PolicyCache {
    kind: CachePolicyKind,
    capacity: usize,
    resident: HashSet<u32>,
    order: VecDeque<u32>,
    resident_frequency: HashMap<u32, u64>,
    inserted_at: HashMap<u32, u64>,
    lfu_heap: BinaryHeap<Reverse<(u64, u64, u32)>>,
    clock_refs: HashMap<u32, bool>,
    tiny_sketch: FrequencySketch,
    tiny_doorkeeper: HashSet<u32>,
    tiny_sample_count: usize,
    tiny_sample_size: usize,
    arc_t1: VecDeque<u32>,
    arc_t2: VecDeque<u32>,
    arc_b1: VecDeque<u32>,
    arc_b2: VecDeque<u32>,
    arc_t1_set: HashSet<u32>,
    arc_t2_set: HashSet<u32>,
    arc_b1_set: HashSet<u32>,
    arc_b2_set: HashSet<u32>,
    arc_p: usize,
    twoq_a1in: VecDeque<u32>,
    twoq_am: VecDeque<u32>,
    twoq_a1out: VecDeque<u32>,
    twoq_a1out_set: HashSet<u32>,
    twoq_segment: HashMap<u32, TwoQSegment>,
    slru_probationary: VecDeque<u32>,
    slru_protected: VecDeque<u32>,
    slru_segment: HashMap<u32, SlruSegment>,
    lirs_stack: VecDeque<u32>,
    lirs_queue: VecDeque<u32>,
    lirs_status: HashMap<u32, LirsStatus>,
    lirs_lir_count: usize,
    gdsf_frequency: HashMap<u32, u64>,
    gdsf_priority: HashMap<u32, f64>,
    gdsf_heap: BinaryHeap<Reverse<(u64, u32)>>,
    gdsf_clock: f64,
    wt_window: VecDeque<u32>,
    wt_probationary: VecDeque<u32>,
    wt_protected: VecDeque<u32>,
    wt_segment: HashMap<u32, WTinyLfuSegment>,
    lecar_weights: [f64; 2],
    lecar_lru_history: VecDeque<u32>,
    lecar_lru_history_set: HashSet<u32>,
    lecar_lru_history_meta: HashMap<u32, HistoryEntry>,
    lecar_lfu_history: VecDeque<u32>,
    lecar_lfu_history_set: HashSet<u32>,
    lecar_lfu_history_meta: HashMap<u32, HistoryEntry>,
    cacheus_weights: [f64; 2],
    cacheus_s: VecDeque<u32>,
    cacheus_q: VecDeque<u32>,
    cacheus_segment: HashMap<u32, CacheusSegment>,
    cacheus_is_new: HashMap<u32, bool>,
    cacheus_is_demoted: HashMap<u32, bool>,
    cacheus_lru_history: VecDeque<u32>,
    cacheus_lru_history_set: HashSet<u32>,
    cacheus_lru_history_meta: HashMap<u32, HistoryEntry>,
    cacheus_lfu_history: VecDeque<u32>,
    cacheus_lfu_history_set: HashSet<u32>,
    cacheus_lfu_history_meta: HashMap<u32, HistoryEntry>,
    cacheus_s_limit: usize,
    cacheus_q_limit: usize,
    cacheus_dem_count: usize,
    cacheus_nor_count: usize,
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

        let cacheus_q_limit = if capacity == 0 {
            0
        } else {
            ((((capacity as f64) * 0.01) + 0.5) as usize)
                .max(1)
                .min(capacity)
        };
        let cacheus_s_limit = capacity.saturating_sub(cacheus_q_limit);

        Ok(Self {
            kind,
            capacity,
            resident: HashSet::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            resident_frequency: HashMap::with_capacity(capacity),
            inserted_at: HashMap::with_capacity(capacity),
            lfu_heap: BinaryHeap::with_capacity(capacity),
            clock_refs: HashMap::with_capacity(capacity),
            tiny_sketch: FrequencySketch::new(capacity),
            tiny_doorkeeper: HashSet::with_capacity(capacity),
            tiny_sample_count: 0,
            tiny_sample_size: capacity.saturating_mul(10).max(1),
            arc_t1: VecDeque::with_capacity(capacity),
            arc_t2: VecDeque::with_capacity(capacity),
            arc_b1: VecDeque::with_capacity(capacity),
            arc_b2: VecDeque::with_capacity(capacity),
            arc_t1_set: HashSet::with_capacity(capacity),
            arc_t2_set: HashSet::with_capacity(capacity),
            arc_b1_set: HashSet::with_capacity(capacity),
            arc_b2_set: HashSet::with_capacity(capacity),
            arc_p: 0,
            twoq_a1in: VecDeque::with_capacity(capacity),
            twoq_am: VecDeque::with_capacity(capacity),
            twoq_a1out: VecDeque::with_capacity(capacity),
            twoq_a1out_set: HashSet::with_capacity(capacity),
            twoq_segment: HashMap::with_capacity(capacity),
            slru_probationary: VecDeque::with_capacity(capacity),
            slru_protected: VecDeque::with_capacity(capacity),
            slru_segment: HashMap::with_capacity(capacity),
            lirs_stack: VecDeque::with_capacity(capacity),
            lirs_queue: VecDeque::with_capacity(capacity),
            lirs_status: HashMap::with_capacity(capacity),
            lirs_lir_count: 0,
            gdsf_frequency: HashMap::with_capacity(capacity),
            gdsf_priority: HashMap::with_capacity(capacity),
            gdsf_heap: BinaryHeap::with_capacity(capacity),
            gdsf_clock: 0.0,
            wt_window: VecDeque::with_capacity(capacity),
            wt_probationary: VecDeque::with_capacity(capacity),
            wt_protected: VecDeque::with_capacity(capacity),
            wt_segment: HashMap::with_capacity(capacity),
            lecar_weights: [0.5, 0.5],
            lecar_lru_history: VecDeque::with_capacity(capacity),
            lecar_lru_history_set: HashSet::with_capacity(capacity),
            lecar_lru_history_meta: HashMap::with_capacity(capacity),
            lecar_lfu_history: VecDeque::with_capacity(capacity),
            lecar_lfu_history_set: HashSet::with_capacity(capacity),
            lecar_lfu_history_meta: HashMap::with_capacity(capacity),
            cacheus_weights: [0.5, 0.5],
            cacheus_s: VecDeque::with_capacity(capacity),
            cacheus_q: VecDeque::with_capacity(capacity),
            cacheus_segment: HashMap::with_capacity(capacity),
            cacheus_is_new: HashMap::with_capacity(capacity),
            cacheus_is_demoted: HashMap::with_capacity(capacity),
            cacheus_lru_history: VecDeque::with_capacity(capacity),
            cacheus_lru_history_set: HashSet::with_capacity(capacity),
            cacheus_lru_history_meta: HashMap::with_capacity(capacity),
            cacheus_lfu_history: VecDeque::with_capacity(capacity),
            cacheus_lfu_history_set: HashSet::with_capacity(capacity),
            cacheus_lfu_history_meta: HashMap::with_capacity(capacity),
            cacheus_s_limit,
            cacheus_q_limit,
            cacheus_dem_count: 0,
            cacheus_nor_count: 0,
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

        if matches!(
            self.kind,
            CachePolicyKind::TinyLfu | CachePolicyKind::WTinyLfu
        ) {
            self.record_tiny_lfu_access(vertex_id);
        }

        if !self.contains(vertex_id) {
            return;
        }

        match self.kind {
            CachePolicyKind::Lru | CachePolicyKind::TinyLfu => {
                Self::move_to_back(&mut self.order, vertex_id);
            }
            CachePolicyKind::Lfu | CachePolicyKind::LeCar => {
                Self::move_to_back(&mut self.order, vertex_id);
                *self.resident_frequency.entry(vertex_id).or_insert(1) += 1;
                self.push_lfu_entry(vertex_id);
            }
            CachePolicyKind::Clock => {
                self.clock_refs.insert(vertex_id, true);
            }
            CachePolicyKind::TwoQ => {
                self.record_twoq_hit(vertex_id);
            }
            CachePolicyKind::Slru => {
                self.record_slru_hit(vertex_id);
            }
            CachePolicyKind::Lirs => {
                self.record_lirs_hit(vertex_id);
            }
            CachePolicyKind::Arc => {
                self.record_arc_hit(vertex_id);
            }
            CachePolicyKind::Gdsf => {
                self.record_gdsf_hit(vertex_id);
            }
            CachePolicyKind::WTinyLfu => {
                self.record_wtiny_lfu_hit(vertex_id);
            }
            CachePolicyKind::Cacheus => {
                self.record_cacheus_hit(vertex_id);
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

        match self.kind {
            CachePolicyKind::Arc => return self.admit_arc(vertex_id),
            CachePolicyKind::TwoQ => return self.admit_twoq(vertex_id),
            CachePolicyKind::Slru => return self.admit_slru(vertex_id),
            CachePolicyKind::Lirs => return self.admit_lirs(vertex_id),
            CachePolicyKind::Gdsf => return self.admit_gdsf(vertex_id),
            CachePolicyKind::WTinyLfu => return self.admit_wtiny_lfu(vertex_id),
            CachePolicyKind::LeCar => return self.admit_lecar(vertex_id),
            CachePolicyKind::Cacheus => return self.admit_cacheus(vertex_id),
            CachePolicyKind::NoCache
            | CachePolicyKind::Fifo
            | CachePolicyKind::Lru
            | CachePolicyKind::Lfu
            | CachePolicyKind::Random
            | CachePolicyKind::Clock
            | CachePolicyKind::TinyLfu
            | CachePolicyKind::BeladyOptimal => {}
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
        self.admit(vertex_id)
    }

    fn insert_resident(&mut self, vertex_id: u32) {
        self.insert_resident_with_frequency(vertex_id, 1);
    }

    fn insert_resident_with_frequency(&mut self, vertex_id: u32, frequency: u64) {
        self.tick = self.tick.saturating_add(1);
        self.resident.insert(vertex_id);
        self.order.push_back(vertex_id);
        self.resident_frequency.insert(vertex_id, frequency.max(1));
        self.inserted_at.insert(vertex_id, self.tick);
        self.clock_refs.insert(vertex_id, true);
        if matches!(
            self.kind,
            CachePolicyKind::Lfu | CachePolicyKind::LeCar | CachePolicyKind::Cacheus
        ) {
            self.push_lfu_entry(vertex_id);
        }
        if matches!(self.kind, CachePolicyKind::Gdsf) {
            self.insert_gdsf_resident(vertex_id);
        }
    }

    fn remove_resident(&mut self, vertex_id: u32) -> bool {
        if !self.resident.remove(&vertex_id) {
            return false;
        }
        Self::remove_from_order(&mut self.order, vertex_id);
        self.resident_frequency.remove(&vertex_id);
        self.inserted_at.remove(&vertex_id);
        self.clock_refs.remove(&vertex_id);
        self.gdsf_frequency.remove(&vertex_id);
        self.gdsf_priority.remove(&vertex_id);
        self.twoq_segment.remove(&vertex_id);
        Self::remove_from_order(&mut self.twoq_a1in, vertex_id);
        Self::remove_from_order(&mut self.twoq_am, vertex_id);
        self.slru_segment.remove(&vertex_id);
        Self::remove_from_order(&mut self.slru_probationary, vertex_id);
        Self::remove_from_order(&mut self.slru_protected, vertex_id);
        self.wt_segment.remove(&vertex_id);
        Self::remove_from_order(&mut self.wt_window, vertex_id);
        Self::remove_from_order(&mut self.wt_probationary, vertex_id);
        Self::remove_from_order(&mut self.wt_protected, vertex_id);
        self.cacheus_segment.remove(&vertex_id);
        self.cacheus_is_new.remove(&vertex_id);
        if self.cacheus_is_demoted.remove(&vertex_id) == Some(true) {
            self.cacheus_dem_count = self.cacheus_dem_count.saturating_sub(1);
        }
        Self::remove_from_order(&mut self.cacheus_s, vertex_id);
        Self::remove_from_order(&mut self.cacheus_q, vertex_id);
        if matches!(
            self.lirs_status.get(&vertex_id),
            Some(LirsStatus::Lir | LirsStatus::HirResident)
        ) {
            if self.lirs_status.get(&vertex_id) == Some(&LirsStatus::Lir) && self.lirs_lir_count > 0
            {
                self.lirs_lir_count -= 1;
            }
            self.lirs_status
                .insert(vertex_id, LirsStatus::HirNonResident);
        }
        Self::remove_from_order(&mut self.lirs_queue, vertex_id);
        self.arc_t1_set.remove(&vertex_id);
        self.arc_t2_set.remove(&vertex_id);
        Self::remove_from_order(&mut self.arc_t1, vertex_id);
        Self::remove_from_order(&mut self.arc_t2, vertex_id);
        true
    }

    fn victim(&mut self) -> Option<u32> {
        match self.kind {
            CachePolicyKind::NoCache | CachePolicyKind::BeladyOptimal => None,
            CachePolicyKind::Fifo | CachePolicyKind::Lru | CachePolicyKind::TinyLfu => {
                self.order.front().copied()
            }
            CachePolicyKind::Lfu => self.lfu_victim(),
            CachePolicyKind::Clock => self.clock_victim(),
            CachePolicyKind::Random => {
                if self.order.is_empty() {
                    None
                } else {
                    let idx = self.next_random_index(self.order.len());
                    self.order.get(idx).copied()
                }
            }
            CachePolicyKind::Gdsf => self.gdsf_victim(),
            CachePolicyKind::TwoQ => self.twoq_victim(),
            CachePolicyKind::Slru => self.slru_victim(),
            CachePolicyKind::Lirs => self.lirs_victim(),
            CachePolicyKind::Arc => self.arc_victim_for_new(),
            CachePolicyKind::WTinyLfu => self.wtiny_main_victim(),
            CachePolicyKind::LeCar => self.lecar_lru_victim(),
            CachePolicyKind::Cacheus => self.cacheus_lru_victim(),
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
        self.tiny_sketch.estimate(vertex_id) + u64::from(self.tiny_doorkeeper.contains(&vertex_id))
    }

    fn record_tiny_lfu_access(&mut self, vertex_id: u32) {
        self.tiny_sample_count += 1;
        if !self.tiny_doorkeeper.insert(vertex_id) {
            self.tiny_sketch.increment(vertex_id);
        }
        if self.tiny_sample_count >= self.tiny_sample_size {
            self.tiny_sketch.reset();
            self.tiny_doorkeeper.clear();
            self.tiny_sample_count = 0;
        }
    }

    fn clock_victim(&mut self) -> Option<u32> {
        while let Some(vertex_id) = self.order.pop_front() {
            if !self.resident.contains(&vertex_id) {
                continue;
            }
            if self.clock_refs.get(&vertex_id).copied().unwrap_or(false) {
                self.clock_refs.insert(vertex_id, false);
                self.order.push_back(vertex_id);
                continue;
            }
            self.order.push_front(vertex_id);
            return Some(vertex_id);
        }
        None
    }

    fn twoq_a1in_capacity(&self) -> usize {
        (self.capacity / 4).max(1)
    }

    fn twoq_history_capacity(&self) -> usize {
        self.capacity
    }

    fn record_twoq_hit(&mut self, vertex_id: u32) {
        match self.twoq_segment.get(&vertex_id).copied() {
            Some(TwoQSegment::A1In) => {}
            Some(TwoQSegment::Am) => Self::move_to_back(&mut self.twoq_am, vertex_id),
            None => {}
        }
        Self::move_to_back(&mut self.order, vertex_id);
    }

    fn insert_twoq_a1in(&mut self, vertex_id: u32) {
        self.insert_resident(vertex_id);
        self.twoq_segment.insert(vertex_id, TwoQSegment::A1In);
        self.twoq_a1in.push_back(vertex_id);
    }

    fn insert_twoq_am(&mut self, vertex_id: u32) {
        self.insert_resident(vertex_id);
        self.twoq_segment.insert(vertex_id, TwoQSegment::Am);
        self.twoq_am.push_back(vertex_id);
    }

    fn twoq_victim(&mut self) -> Option<u32> {
        if self.twoq_a1in.len() > self.twoq_a1in_capacity() {
            self.twoq_a1in.front().copied()
        } else {
            self.twoq_am
                .front()
                .copied()
                .or_else(|| self.twoq_a1in.front().copied())
        }
    }

    fn admit_twoq(&mut self, vertex_id: u32) -> AdmissionOutcome {
        let mut evicted = None;
        let seen_before = self.twoq_a1out_set.remove(&vertex_id);
        Self::remove_from_order(&mut self.twoq_a1out, vertex_id);

        if self.len() == self.capacity {
            evicted = self.twoq_victim();
            if let Some(victim) = evicted {
                if self.twoq_segment.get(&victim) == Some(&TwoQSegment::A1In) {
                    let history_capacity = self.twoq_history_capacity();
                    Self::push_limited_history(
                        &mut self.twoq_a1out,
                        &mut self.twoq_a1out_set,
                        victim,
                        history_capacity,
                    );
                }
                self.remove_resident(victim);
            }
        }

        if seen_before {
            self.insert_twoq_am(vertex_id);
        } else {
            self.insert_twoq_a1in(vertex_id);
        }

        AdmissionOutcome {
            admitted: true,
            evicted,
            rejected: false,
        }
    }

    fn slru_protected_capacity(&self) -> usize {
        ((self.capacity * 8) / 10).max(1).min(self.capacity)
    }

    fn slru_probationary_capacity(&self) -> usize {
        self.capacity
            .saturating_sub(self.slru_protected_capacity())
            .max(1)
    }

    fn record_slru_hit(&mut self, vertex_id: u32) {
        match self.slru_segment.get(&vertex_id).copied() {
            Some(SlruSegment::Protected) => {
                Self::move_to_back(&mut self.slru_protected, vertex_id);
            }
            Some(SlruSegment::Probationary) => {
                Self::remove_from_order(&mut self.slru_probationary, vertex_id);
                self.slru_segment.insert(vertex_id, SlruSegment::Protected);
                self.slru_protected.push_back(vertex_id);
                self.slru_demote_if_needed();
            }
            None => {}
        }
        Self::move_to_back(&mut self.order, vertex_id);
    }

    fn insert_slru_probationary(&mut self, vertex_id: u32) {
        self.insert_resident(vertex_id);
        self.slru_segment
            .insert(vertex_id, SlruSegment::Probationary);
        self.slru_probationary.push_back(vertex_id);
    }

    fn slru_demote_if_needed(&mut self) {
        while self.slru_protected.len() > self.slru_protected_capacity() {
            let Some(demoted) = self.slru_protected.pop_front() else {
                break;
            };
            self.slru_segment.insert(demoted, SlruSegment::Probationary);
            self.slru_probationary.push_back(demoted);
        }
    }

    fn slru_victim(&mut self) -> Option<u32> {
        self.slru_probationary
            .front()
            .copied()
            .or_else(|| self.slru_protected.front().copied())
    }

    fn admit_slru(&mut self, vertex_id: u32) -> AdmissionOutcome {
        let evicted = if self.len() == self.capacity {
            let victim = self.slru_victim();
            if let Some(victim) = victim {
                self.remove_resident(victim);
            }
            victim
        } else {
            None
        };
        self.insert_slru_probationary(vertex_id);
        while self.slru_probationary.len() > self.slru_probationary_capacity()
            && self.len() > self.capacity
        {
            if let Some(victim) = self.slru_victim() {
                self.remove_resident(victim);
            } else {
                break;
            }
        }
        AdmissionOutcome {
            admitted: true,
            evicted,
            rejected: false,
        }
    }

    fn lirs_lir_capacity(&self) -> usize {
        self.capacity.saturating_sub(1).max(1)
    }

    fn lirs_stack_contains(&self, vertex_id: u32) -> bool {
        self.lirs_stack.iter().any(|id| *id == vertex_id)
    }

    fn lirs_move_to_stack_top(&mut self, vertex_id: u32) {
        Self::remove_from_order(&mut self.lirs_stack, vertex_id);
        self.lirs_stack.push_back(vertex_id);
    }

    fn lirs_prune_stack(&mut self) {
        while let Some(bottom) = self.lirs_stack.front().copied() {
            if self.lirs_status.get(&bottom) == Some(&LirsStatus::Lir) {
                break;
            }
            self.lirs_stack.pop_front();
            if self.lirs_status.get(&bottom) == Some(&LirsStatus::HirNonResident) {
                self.lirs_status.remove(&bottom);
            }
        }
    }

    fn lirs_demote_bottom_lir(&mut self) {
        while self.lirs_lir_count > self.lirs_lir_capacity() {
            let Some(victim) = self
                .lirs_stack
                .iter()
                .copied()
                .find(|id| self.lirs_status.get(id) == Some(&LirsStatus::Lir))
            else {
                break;
            };
            self.lirs_status.insert(victim, LirsStatus::HirResident);
            self.lirs_queue.push_back(victim);
            self.lirs_lir_count -= 1;
            self.lirs_prune_stack();
        }
    }

    fn record_lirs_hit(&mut self, vertex_id: u32) {
        match self.lirs_status.get(&vertex_id).copied() {
            Some(LirsStatus::Lir) => {
                self.lirs_move_to_stack_top(vertex_id);
                self.lirs_prune_stack();
            }
            Some(LirsStatus::HirResident) => {
                let in_stack = self.lirs_stack_contains(vertex_id);
                Self::remove_from_order(&mut self.lirs_queue, vertex_id);
                self.lirs_move_to_stack_top(vertex_id);
                if in_stack {
                    self.lirs_status.insert(vertex_id, LirsStatus::Lir);
                    self.lirs_lir_count += 1;
                    self.lirs_demote_bottom_lir();
                } else {
                    self.lirs_queue.push_back(vertex_id);
                }
                self.lirs_prune_stack();
            }
            Some(LirsStatus::HirNonResident) | None => {}
        }
    }

    fn lirs_victim(&mut self) -> Option<u32> {
        self.lirs_queue.front().copied().or_else(|| {
            self.resident
                .iter()
                .copied()
                .find(|id| self.lirs_status.get(id) != Some(&LirsStatus::Lir))
        })
    }

    fn lirs_evict_hir_if_needed(&mut self) -> Option<u32> {
        if self.len() < self.capacity {
            return None;
        }
        let victim = self.lirs_victim()?;
        self.remove_resident(victim);
        self.lirs_status.insert(victim, LirsStatus::HirNonResident);
        Some(victim)
    }

    fn admit_lirs(&mut self, vertex_id: u32) -> AdmissionOutcome {
        let known_hir = self.lirs_status.get(&vertex_id) == Some(&LirsStatus::HirNonResident);
        let evicted = self.lirs_evict_hir_if_needed();

        self.insert_resident(vertex_id);
        self.lirs_move_to_stack_top(vertex_id);
        if self.lirs_lir_count < self.lirs_lir_capacity() || known_hir {
            self.lirs_status.insert(vertex_id, LirsStatus::Lir);
            self.lirs_lir_count += 1;
            self.lirs_demote_bottom_lir();
        } else {
            self.lirs_status.insert(vertex_id, LirsStatus::HirResident);
            self.lirs_queue.push_back(vertex_id);
        }
        self.lirs_prune_stack();

        AdmissionOutcome {
            admitted: true,
            evicted,
            rejected: false,
        }
    }

    fn record_arc_hit(&mut self, vertex_id: u32) {
        if self.arc_t1_set.remove(&vertex_id) {
            Self::remove_from_order(&mut self.arc_t1, vertex_id);
            self.arc_t2_set.insert(vertex_id);
            self.arc_t2.push_back(vertex_id);
        } else if self.arc_t2_set.contains(&vertex_id) {
            Self::move_to_back(&mut self.arc_t2, vertex_id);
        }
        Self::move_to_back(&mut self.order, vertex_id);
    }

    fn arc_insert_t1(&mut self, vertex_id: u32) {
        self.insert_resident(vertex_id);
        self.arc_t1_set.insert(vertex_id);
        self.arc_t1.push_back(vertex_id);
    }

    fn arc_insert_t2(&mut self, vertex_id: u32) {
        self.insert_resident(vertex_id);
        self.arc_t2_set.insert(vertex_id);
        self.arc_t2.push_back(vertex_id);
    }

    fn arc_push_b1(&mut self, vertex_id: u32) {
        Self::push_limited_history(
            &mut self.arc_b1,
            &mut self.arc_b1_set,
            vertex_id,
            self.capacity,
        );
    }

    fn arc_push_b2(&mut self, vertex_id: u32) {
        Self::push_limited_history(
            &mut self.arc_b2,
            &mut self.arc_b2_set,
            vertex_id,
            self.capacity,
        );
    }

    fn arc_replace(&mut self, incoming: u32) -> Option<u32> {
        let use_t1 = !self.arc_t1.is_empty()
            && (self.arc_t1.len() > self.arc_p
                || (self.arc_b2_set.contains(&incoming) && self.arc_t1.len() == self.arc_p));
        let victim = if use_t1 {
            self.arc_t1.pop_front()
        } else {
            self.arc_t2.pop_front().or_else(|| self.arc_t1.pop_front())
        }?;

        if self.arc_t1_set.remove(&victim) {
            self.arc_push_b1(victim);
        } else if self.arc_t2_set.remove(&victim) {
            self.arc_push_b2(victim);
        }
        self.remove_resident(victim);
        Some(victim)
    }

    fn arc_victim_for_new(&mut self) -> Option<u32> {
        self.arc_t1
            .front()
            .copied()
            .or_else(|| self.arc_t2.front().copied())
    }

    fn admit_arc(&mut self, vertex_id: u32) -> AdmissionOutcome {
        let mut evicted = None;
        if self.arc_b1_set.remove(&vertex_id) {
            Self::remove_from_order(&mut self.arc_b1, vertex_id);
            let delta = if self.arc_b1.len() >= self.arc_b2.len() {
                1
            } else {
                (self.arc_b2.len() / self.arc_b1.len().max(1)).max(1)
            };
            self.arc_p = (self.arc_p + delta).min(self.capacity);
            evicted = self.arc_replace(vertex_id);
            self.arc_insert_t2(vertex_id);
        } else if self.arc_b2_set.remove(&vertex_id) {
            Self::remove_from_order(&mut self.arc_b2, vertex_id);
            let delta = if self.arc_b2.len() >= self.arc_b1.len() {
                1
            } else {
                (self.arc_b1.len() / self.arc_b2.len().max(1)).max(1)
            };
            self.arc_p = self.arc_p.saturating_sub(delta);
            evicted = self.arc_replace(vertex_id);
            self.arc_insert_t2(vertex_id);
        } else {
            if self.len() == self.capacity {
                evicted = self.arc_replace(vertex_id);
            }
            while self.arc_b1.len() + self.arc_b2.len() > self.capacity {
                if let Some(old) = self.arc_b2.pop_front() {
                    self.arc_b2_set.remove(&old);
                } else if let Some(old) = self.arc_b1.pop_front() {
                    self.arc_b1_set.remove(&old);
                } else {
                    break;
                }
            }
            self.arc_insert_t1(vertex_id);
        }

        AdmissionOutcome {
            admitted: true,
            evicted,
            rejected: false,
        }
    }

    fn gdsf_key(priority: f64) -> u64 {
        priority.to_bits()
    }

    fn insert_gdsf_resident(&mut self, vertex_id: u32) {
        self.gdsf_frequency.insert(vertex_id, 1);
        let priority = self.gdsf_clock + 1.0;
        self.gdsf_priority.insert(vertex_id, priority);
        self.gdsf_heap
            .push(Reverse((Self::gdsf_key(priority), vertex_id)));
    }

    fn record_gdsf_hit(&mut self, vertex_id: u32) {
        let frequency = self.gdsf_frequency.entry(vertex_id).or_insert(0);
        *frequency += 1;
        let priority = self.gdsf_clock + *frequency as f64;
        self.gdsf_priority.insert(vertex_id, priority);
        self.gdsf_heap
            .push(Reverse((Self::gdsf_key(priority), vertex_id)));
    }

    fn gdsf_victim(&mut self) -> Option<u32> {
        while let Some(Reverse((priority_key, vertex_id))) = self.gdsf_heap.pop() {
            if !self.resident.contains(&vertex_id) {
                continue;
            }
            let priority = self
                .gdsf_priority
                .get(&vertex_id)
                .copied()
                .unwrap_or_default();
            if Self::gdsf_key(priority) != priority_key {
                continue;
            }
            self.gdsf_clock = priority;
            return Some(vertex_id);
        }
        None
    }

    fn admit_gdsf(&mut self, vertex_id: u32) -> AdmissionOutcome {
        let evicted = if self.len() == self.capacity {
            let victim = self.gdsf_victim();
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

    fn wt_window_capacity(&self) -> usize {
        (self.capacity / 100).max(1)
    }

    fn wt_protected_capacity(&self) -> usize {
        let main = self.capacity.saturating_sub(self.wt_window_capacity());
        ((main * 8) / 10).max(1).min(main.max(1))
    }

    fn record_wtiny_lfu_hit(&mut self, vertex_id: u32) {
        match self.wt_segment.get(&vertex_id).copied() {
            Some(WTinyLfuSegment::Window) => {
                Self::move_to_back(&mut self.wt_window, vertex_id);
            }
            Some(WTinyLfuSegment::Probationary) => {
                Self::remove_from_order(&mut self.wt_probationary, vertex_id);
                self.wt_segment
                    .insert(vertex_id, WTinyLfuSegment::Protected);
                self.wt_protected.push_back(vertex_id);
                self.wt_demote_if_needed();
            }
            Some(WTinyLfuSegment::Protected) => {
                Self::move_to_back(&mut self.wt_protected, vertex_id);
            }
            None => {}
        }
        Self::move_to_back(&mut self.order, vertex_id);
    }

    fn wt_demote_if_needed(&mut self) {
        while self.wt_protected.len() > self.wt_protected_capacity() {
            let Some(demoted) = self.wt_protected.pop_front() else {
                break;
            };
            self.wt_segment
                .insert(demoted, WTinyLfuSegment::Probationary);
            self.wt_probationary.push_back(demoted);
        }
    }

    fn wt_insert_window(&mut self, vertex_id: u32) {
        self.insert_resident(vertex_id);
        self.wt_segment.insert(vertex_id, WTinyLfuSegment::Window);
        self.wt_window.push_back(vertex_id);
    }

    fn wt_insert_probationary(&mut self, vertex_id: u32) {
        self.wt_segment
            .insert(vertex_id, WTinyLfuSegment::Probationary);
        self.wt_probationary.push_back(vertex_id);
    }

    fn wt_drain_window_while_uncontended(&mut self) {
        while self.wt_window.len() > self.wt_window_capacity() && self.len() <= self.capacity {
            let Some(candidate) = self.wt_window.pop_front() else {
                break;
            };
            self.wt_segment.remove(&candidate);
            self.wt_insert_probationary(candidate);
            self.wt_demote_if_needed();
        }
    }

    fn wtiny_main_victim(&mut self) -> Option<u32> {
        self.wt_probationary
            .front()
            .copied()
            .or_else(|| self.wt_protected.front().copied())
    }

    fn admit_wtiny_lfu(&mut self, vertex_id: u32) -> AdmissionOutcome {
        if self.len() < self.capacity {
            self.wt_insert_window(vertex_id);
            self.wt_drain_window_while_uncontended();
            return AdmissionOutcome {
                admitted: true,
                evicted: None,
                rejected: false,
            };
        }

        self.wt_insert_window(vertex_id);
        let Some(candidate) = self.wt_window.pop_front() else {
            return AdmissionOutcome {
                admitted: true,
                evicted: None,
                rejected: false,
            };
        };
        self.wt_segment.remove(&candidate);

        if let Some(victim) = self.wtiny_main_victim() {
            if self.estimated_frequency(candidate) > self.estimated_frequency(victim) {
                self.remove_resident(victim);
                self.wt_insert_probationary(candidate);
                AdmissionOutcome {
                    admitted: true,
                    evicted: Some(victim),
                    rejected: false,
                }
            } else {
                self.remove_resident(candidate);
                AdmissionOutcome {
                    admitted: candidate != vertex_id,
                    evicted: (candidate != vertex_id).then_some(candidate),
                    rejected: candidate == vertex_id,
                }
            }
        } else {
            self.remove_resident(candidate);
            AdmissionOutcome {
                admitted: candidate != vertex_id,
                evicted: (candidate != vertex_id).then_some(candidate),
                rejected: candidate == vertex_id,
            }
        }
    }

    fn lecar_lru_victim(&self) -> Option<u32> {
        self.order.front().copied()
    }

    fn lecar_lfu_victim(&mut self) -> Option<u32> {
        self.lfu_victim()
    }

    fn lecar_discount_rate(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            0.005_f64.powf(1.0 / self.capacity as f64)
        }
    }

    fn adjust_lecar_weights(&mut self, reward_lru: f64, reward_lfu: f64) {
        const LEARNING_RATE: f64 = 0.45;
        self.lecar_weights[0] *= (LEARNING_RATE * reward_lru).exp();
        self.lecar_weights[1] *= (LEARNING_RATE * reward_lfu).exp();
        Self::normalize_weights(&mut self.lecar_weights);
        Self::clamp_binary_weights(&mut self.lecar_weights);
    }

    fn update_lecar_weights(&mut self, vertex_id: u32) -> u64 {
        let mut frequency = 1;
        if self.lecar_lru_history_set.remove(&vertex_id) {
            Self::remove_from_order(&mut self.lecar_lru_history, vertex_id);
            if let Some(entry) = self.lecar_lru_history_meta.remove(&vertex_id) {
                frequency = entry.frequency.saturating_add(1);
                let age = self.tick.saturating_sub(entry.evicted_at) as f64;
                let reward = -(self.lecar_discount_rate().powf(age));
                self.adjust_lecar_weights(reward, 0.0);
            }
        }
        if self.lecar_lfu_history_set.remove(&vertex_id) {
            Self::remove_from_order(&mut self.lecar_lfu_history, vertex_id);
            if let Some(entry) = self.lecar_lfu_history_meta.remove(&vertex_id) {
                frequency = frequency.max(entry.frequency.saturating_add(1));
                let age = self.tick.saturating_sub(entry.evicted_at) as f64;
                let reward = -(self.lecar_discount_rate().powf(age));
                self.adjust_lecar_weights(0.0, reward);
            }
        }
        frequency
    }

    fn choose_lecar_expert(&mut self) -> LeCarExpert {
        let sample = self.next_random_unit();
        if sample < self.lecar_weights[0] {
            LeCarExpert::Lru
        } else {
            LeCarExpert::Lfu
        }
    }

    fn admit_lecar(&mut self, vertex_id: u32) -> AdmissionOutcome {
        let frequency = self.update_lecar_weights(vertex_id);
        let mut evicted = None;
        if self.len() == self.capacity {
            let expert = self.choose_lecar_expert();
            let victim = match expert {
                LeCarExpert::Lru => self.lecar_lru_victim(),
                LeCarExpert::Lfu => self.lecar_lfu_victim().or_else(|| self.lecar_lru_victim()),
            };
            if let Some(victim) = victim {
                let entry = HistoryEntry {
                    frequency: self.resident_frequency.get(&victim).copied().unwrap_or(1),
                    evicted_at: self.tick,
                    is_new: false,
                };
                match expert {
                    LeCarExpert::Lru => {
                        if let Some(old) = Self::push_limited_history(
                            &mut self.lecar_lru_history,
                            &mut self.lecar_lru_history_set,
                            victim,
                            self.capacity,
                        ) {
                            self.lecar_lru_history_meta.remove(&old);
                        }
                        self.lecar_lru_history_meta.insert(victim, entry);
                    }
                    LeCarExpert::Lfu => {
                        if let Some(old) = Self::push_limited_history(
                            &mut self.lecar_lfu_history,
                            &mut self.lecar_lfu_history_set,
                            victim,
                            self.capacity,
                        ) {
                            self.lecar_lfu_history_meta.remove(&old);
                        }
                        self.lecar_lfu_history_meta.insert(victim, entry);
                    }
                }
                self.remove_resident(victim);
                evicted = Some(victim);
            }
        }
        self.insert_resident_with_frequency(vertex_id, frequency);
        AdmissionOutcome {
            admitted: true,
            evicted,
            rejected: false,
        }
    }

    fn cacheus_lru_victim(&self) -> Option<u32> {
        self.cacheus_q
            .front()
            .copied()
            .or_else(|| self.cacheus_s.front().copied())
    }

    fn cacheus_history_capacity(&self) -> usize {
        (self.capacity / 2).max(1)
    }

    fn cacheus_learning_rate(&self) -> f64 {
        if self.capacity == 0 {
            0.001
        } else {
            ((2.0 * std::f64::consts::LN_2) / self.capacity as f64)
                .sqrt()
                .clamp(0.001, 1.0)
        }
    }

    fn adjust_cacheus_weights(&mut self, reward_lru: f64, reward_lfu: f64) {
        let learning_rate = self.cacheus_learning_rate();
        self.cacheus_weights[0] *= (learning_rate * reward_lru).exp();
        self.cacheus_weights[1] *= (learning_rate * reward_lfu).exp();
        Self::normalize_weights(&mut self.cacheus_weights);
        Self::clamp_binary_weights(&mut self.cacheus_weights);
    }

    fn cacheus_adjust_size(&mut self, hit_in_q: bool) {
        if self.capacity <= 1 {
            self.cacheus_q_limit = self.capacity;
            self.cacheus_s_limit = 0;
            return;
        }

        if hit_in_q {
            let dem_count = self.cacheus_dem_count.max(1);
            let delta = ((self.cacheus_nor_count as f64 / dem_count as f64) + 0.5)
                .floor()
                .max(1.0) as usize;
            self.cacheus_s_limit = (self.cacheus_s_limit + delta).min(self.capacity - 1);
            self.cacheus_q_limit = self.capacity - self.cacheus_s_limit;
        } else {
            let nor_count = self.cacheus_nor_count.max(1);
            let delta = ((self.cacheus_dem_count as f64 / nor_count as f64) + 0.5)
                .floor()
                .max(1.0) as usize;
            self.cacheus_q_limit = (self.cacheus_q_limit + delta).min(self.capacity - 1);
            self.cacheus_s_limit = self.capacity - self.cacheus_q_limit;
        }
    }

    fn cacheus_demote_s_lru(&mut self) -> bool {
        let Some(demoted) = self.cacheus_s.pop_front() else {
            return false;
        };
        self.cacheus_segment.insert(demoted, CacheusSegment::Q);
        if self.cacheus_is_demoted.insert(demoted, true) != Some(true) {
            self.cacheus_dem_count += 1;
        }
        self.cacheus_q.push_back(demoted);
        true
    }

    fn cacheus_limit_stack(&mut self) {
        if self.cacheus_s_limit == 0 {
            while self.cacheus_demote_s_lru() {}
            return;
        }

        while self.cacheus_s.len() >= self.cacheus_s_limit {
            if !self.cacheus_demote_s_lru() {
                break;
            }
        }
    }

    fn cacheus_add_to_s(&mut self, vertex_id: u32, frequency: u64, is_new: bool) {
        self.insert_resident_with_frequency(vertex_id, frequency);
        self.cacheus_segment.insert(vertex_id, CacheusSegment::S);
        self.cacheus_is_new.insert(vertex_id, is_new);
        self.cacheus_is_demoted.insert(vertex_id, false);
        self.cacheus_s.push_back(vertex_id);
    }

    fn cacheus_add_to_q(&mut self, vertex_id: u32, frequency: u64, is_new: bool) {
        self.insert_resident_with_frequency(vertex_id, frequency);
        self.cacheus_segment.insert(vertex_id, CacheusSegment::Q);
        self.cacheus_is_new.insert(vertex_id, is_new);
        self.cacheus_is_demoted.insert(vertex_id, false);
        self.cacheus_q.push_back(vertex_id);
    }

    fn record_cacheus_hit(&mut self, vertex_id: u32) {
        Self::move_to_back(&mut self.order, vertex_id);
        *self.resident_frequency.entry(vertex_id).or_insert(1) += 1;
        self.push_lfu_entry(vertex_id);

        match self.cacheus_segment.get(&vertex_id).copied() {
            Some(CacheusSegment::S) => {
                Self::move_to_back(&mut self.cacheus_s, vertex_id);
            }
            Some(CacheusSegment::Q) => {
                if self.cacheus_is_demoted.get(&vertex_id).copied() == Some(true) {
                    self.cacheus_adjust_size(true);
                    self.cacheus_is_demoted.insert(vertex_id, false);
                    self.cacheus_dem_count = self.cacheus_dem_count.saturating_sub(1);
                }
                Self::remove_from_order(&mut self.cacheus_q, vertex_id);
                self.cacheus_segment.insert(vertex_id, CacheusSegment::S);
                if self.cacheus_s.len() >= self.cacheus_s_limit && self.cacheus_s_limit > 0 {
                    self.cacheus_demote_s_lru();
                }
                self.cacheus_s.push_back(vertex_id);
            }
            None => {}
        }
        self.cacheus_limit_stack();
    }

    fn cacheus_choose_expert(&mut self) -> LeCarExpert {
        if self.next_random_unit() < self.cacheus_weights[0] {
            LeCarExpert::Lru
        } else {
            LeCarExpert::Lfu
        }
    }

    fn cacheus_push_lru_history(&mut self, vertex_id: u32, entry: HistoryEntry) {
        if entry.is_new {
            self.cacheus_nor_count += 1;
        }
        let history_capacity = self.cacheus_history_capacity();
        if let Some(old) = Self::push_limited_history(
            &mut self.cacheus_lru_history,
            &mut self.cacheus_lru_history_set,
            vertex_id,
            history_capacity,
        ) {
            if self
                .cacheus_lru_history_meta
                .remove(&old)
                .map(|entry| entry.is_new)
                == Some(true)
            {
                self.cacheus_nor_count = self.cacheus_nor_count.saturating_sub(1);
            }
        }
        self.cacheus_lru_history_meta.insert(vertex_id, entry);
    }

    fn cacheus_push_lfu_history(&mut self, vertex_id: u32, entry: HistoryEntry) {
        let history_capacity = self.cacheus_history_capacity();
        if let Some(old) = Self::push_limited_history(
            &mut self.cacheus_lfu_history,
            &mut self.cacheus_lfu_history_set,
            vertex_id,
            history_capacity,
        ) {
            self.cacheus_lfu_history_meta.remove(&old);
        }
        self.cacheus_lfu_history_meta.insert(vertex_id, entry);
    }

    fn cacheus_evict(&mut self) -> Option<u32> {
        let lru = self.cacheus_lru_victim()?;
        let lfu = self.lfu_victim().unwrap_or(lru);
        let (victim, expert) = if lru == lfu {
            (lru, None)
        } else {
            match self.cacheus_choose_expert() {
                LeCarExpert::Lru => (lru, Some(LeCarExpert::Lru)),
                LeCarExpert::Lfu => (lfu, Some(LeCarExpert::Lfu)),
            }
        };

        let entry = HistoryEntry {
            frequency: self.resident_frequency.get(&victim).copied().unwrap_or(1),
            evicted_at: self.tick,
            is_new: self.cacheus_is_new.get(&victim).copied().unwrap_or(false),
        };

        match expert {
            Some(LeCarExpert::Lru) => self.cacheus_push_lru_history(victim, entry),
            Some(LeCarExpert::Lfu) => self.cacheus_push_lfu_history(victim, entry),
            None => {}
        }

        self.remove_resident(victim);
        Some(victim)
    }

    fn admit_cacheus(&mut self, vertex_id: u32) -> AdmissionOutcome {
        if self.cacheus_lru_history_set.remove(&vertex_id) {
            Self::remove_from_order(&mut self.cacheus_lru_history, vertex_id);
            let entry = self
                .cacheus_lru_history_meta
                .remove(&vertex_id)
                .unwrap_or(HistoryEntry {
                    frequency: 1,
                    evicted_at: self.tick,
                    is_new: false,
                });
            if entry.is_new {
                self.cacheus_nor_count = self.cacheus_nor_count.saturating_sub(1);
                self.cacheus_adjust_size(false);
            }
            self.adjust_cacheus_weights(-1.0, 0.0);
            let evicted = if self.len() >= self.capacity {
                self.cacheus_evict()
            } else {
                None
            };
            self.cacheus_add_to_s(vertex_id, entry.frequency.saturating_add(1), false);
            self.cacheus_limit_stack();
            return AdmissionOutcome {
                admitted: true,
                evicted,
                rejected: false,
            };
        }

        if self.cacheus_lfu_history_set.remove(&vertex_id) {
            Self::remove_from_order(&mut self.cacheus_lfu_history, vertex_id);
            let entry = self
                .cacheus_lfu_history_meta
                .remove(&vertex_id)
                .unwrap_or(HistoryEntry {
                    frequency: 1,
                    evicted_at: self.tick,
                    is_new: false,
                });
            self.adjust_cacheus_weights(0.0, -1.0);
            let evicted = if self.len() >= self.capacity {
                self.cacheus_evict()
            } else {
                None
            };
            self.cacheus_add_to_s(vertex_id, entry.frequency.saturating_add(1), false);
            self.cacheus_limit_stack();
            return AdmissionOutcome {
                admitted: true,
                evicted,
                rejected: false,
            };
        }

        let evicted;
        if self.cacheus_s.len() < self.cacheus_s_limit && self.cacheus_q.is_empty() {
            evicted = None;
            self.cacheus_add_to_s(vertex_id, 1, false);
        } else if self.len() < self.capacity && self.cacheus_q.len() < self.cacheus_q_limit {
            evicted = None;
            self.cacheus_add_to_q(vertex_id, 1, false);
        } else {
            evicted = if self.len() >= self.capacity {
                self.cacheus_evict()
            } else {
                None
            };
            self.cacheus_add_to_q(vertex_id, 1, true);
            self.cacheus_limit_stack();
        }

        AdmissionOutcome {
            admitted: true,
            evicted,
            rejected: false,
        }
    }

    fn next_random_index(&mut self, len: usize) -> usize {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.rng_state as usize) % len
    }

    fn next_random_unit(&mut self) -> f64 {
        let sample = self.next_random_index(usize::MAX);
        sample as f64 / usize::MAX as f64
    }

    fn push_limited_history(
        history: &mut VecDeque<u32>,
        history_set: &mut HashSet<u32>,
        vertex_id: u32,
        capacity: usize,
    ) -> Option<u32> {
        if capacity == 0 {
            return None;
        }
        if history_set.insert(vertex_id) {
            history.push_back(vertex_id);
        } else {
            Self::move_to_back(history, vertex_id);
        }
        let mut evicted = None;
        while history.len() > capacity {
            if let Some(old) = history.pop_front() {
                history_set.remove(&old);
                evicted = Some(old);
            } else {
                break;
            }
        }
        evicted
    }

    fn normalize_weights(weights: &mut [f64; 2]) {
        let sum = weights.iter().copied().sum::<f64>();
        if !sum.is_finite() || sum <= f64::EPSILON {
            *weights = [0.5, 0.5];
            return;
        }
        for weight in weights {
            *weight /= sum;
        }
    }

    fn clamp_binary_weights(weights: &mut [f64; 2]) {
        if weights[0] >= 0.99 {
            *weights = [0.99, 0.01];
        } else if weights[1] >= 0.99 {
            *weights = [0.01, 0.99];
        }
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
    dimension: usize,
    store: HashMap<Data::VectorIdType, Arc<CachedNode<Data>>>,
    policy: PolicyCache,
    stats: CachePolicyStats,
}

impl<Data> DynamicNodeCache<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    pub fn new(dimension: usize, capacity: usize, policy: CachePolicyKind) -> ANNResult<Self> {
        Ok(Self {
            dimension,
            store: HashMap::with_capacity(capacity),
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

    pub fn lookup(&mut self, vertex_id: &Data::VectorIdType) -> Option<Arc<CachedNode<Data>>> {
        self.stats.accesses += 1;
        self.policy.record_access(*vertex_id);
        let node = self.store.get(vertex_id).cloned();
        if node.is_some() {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }
        node
    }

    pub fn contains(&self, vertex_id: &Data::VectorIdType) -> bool {
        self.store.contains_key(vertex_id)
    }

    pub fn admit_node(
        &mut self,
        vertex_id: &Data::VectorIdType,
        node: CachedNode<Data>,
    ) -> ANNResult<()> {
        if node.vector.len() != self.dimension {
            return Err(ANNError::log_index_error(
                "Vector dimension does not match the dimension set in cache.",
            ));
        }

        if self.store.contains_key(vertex_id) {
            self.store.insert(*vertex_id, Arc::new(node));
            return Ok(());
        }

        let outcome = self.policy.admit(*vertex_id);
        if let Some(evicted) = outcome.evicted {
            self.store.remove(&evicted);
            self.stats.evictions += 1;
        }

        if outcome.admitted {
            self.store.insert(*vertex_id, Arc::new(node));
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
        self.policy.capacity()
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
            if node.vector.len() != self.dimension {
                return Err(ANNError::log_index_error(
                    "Vector dimension does not match the dimension set in cache.",
                ));
            }
            self.store.insert(vertex_id, Arc::new(node));
        }
        Ok(())
    }
}

pub struct ShardedDynamicNodeCache<Data: GraphDataType<VectorIdType = u32>> {
    shards: Vec<Mutex<DynamicNodeCache<Data>>>,
    capacity: usize,
}

impl<Data> ShardedDynamicNodeCache<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    pub fn new(
        dimension: usize,
        capacity: usize,
        policy: CachePolicyKind,
        cache_shards: usize,
    ) -> ANNResult<Self> {
        if cache_shards == 0 {
            return Err(ANNError::log_index_error(
                "cache_shards must be greater than 0 for sharded dynamic cache",
            ));
        }

        let mut shards = Vec::with_capacity(cache_shards);
        for shard_id in 0..cache_shards {
            shards.push(Mutex::new(DynamicNodeCache::new(
                dimension,
                Self::capacity_for_shard(capacity, cache_shards, shard_id),
                policy,
            )?));
        }

        Ok(Self { shards, capacity })
    }

    pub fn from_warm_cache(
        policy: CachePolicyKind,
        capacity: usize,
        warm_cache: &Cache<Data>,
        cache_shards: usize,
    ) -> ANNResult<Self> {
        let cache = Self::new(warm_cache.dimension(), capacity, policy, cache_shards)?;
        for id in warm_cache.ids().iter().take(capacity) {
            if let Some(node) = warm_cache.get_node(id) {
                cache.warm_node(*id, node)?;
            }
        }
        Ok(cache)
    }

    pub fn capacity_for_shard(capacity: usize, cache_shards: usize, shard_id: usize) -> usize {
        if cache_shards == 0 {
            return 0;
        }
        (capacity / cache_shards) + usize::from(shard_id < capacity % cache_shards)
    }

    pub fn shard_index_for(vertex_id: u32, cache_shards: usize) -> usize {
        if cache_shards == 0 {
            return 0;
        }
        Self::hash_vertex_id(vertex_id) % cache_shards
    }

    pub fn shard_index(&self, vertex_id: &Data::VectorIdType) -> usize {
        Self::shard_index_for(*vertex_id, self.shards.len())
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn shard_capacities(&self) -> ANNResult<Vec<usize>> {
        self.shards
            .iter()
            .map(|shard| {
                shard
                    .lock()
                    .map(|shard| shard.capacity())
                    .map_err(|_| ANNError::log_index_error("Dynamic node cache lock is poisoned"))
            })
            .collect()
    }

    pub fn lookup(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<Option<Arc<CachedNode<Data>>>> {
        let mut shard = self.lock_shard(vertex_id)?;
        Ok(shard.lookup(vertex_id))
    }

    pub fn contains(&self, vertex_id: &Data::VectorIdType) -> ANNResult<bool> {
        let shard = self.lock_shard(vertex_id)?;
        Ok(shard.contains(vertex_id))
    }

    pub fn admit_node(
        &self,
        vertex_id: &Data::VectorIdType,
        node: CachedNode<Data>,
    ) -> ANNResult<()> {
        let mut shard = self.lock_shard(vertex_id)?;
        shard.admit_node(vertex_id, node)
    }

    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .filter_map(|shard| shard.lock().ok().map(|shard| shard.len()))
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn stats(&self) -> CachePolicyStats {
        let mut stats = CachePolicyStats::default();
        for shard in &self.shards {
            if let Ok(shard) = shard.lock() {
                stats.merge(shard.stats());
            }
        }
        stats
    }

    fn warm_node(&self, vertex_id: Data::VectorIdType, node: CachedNode<Data>) -> ANNResult<()> {
        let mut shard = self.lock_shard(&vertex_id)?;
        shard.warm_node(vertex_id, node)
    }

    fn lock_shard(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<std::sync::MutexGuard<'_, DynamicNodeCache<Data>>> {
        self.shards[self.shard_index(vertex_id)]
            .lock()
            .map_err(|_| ANNError::log_index_error("Dynamic node cache lock is poisoned"))
    }

    fn hash_vertex_id(vertex_id: u32) -> usize {
        let mut x = vertex_id as u64;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (x ^ (x >> 31)) as usize
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
    let mut eviction_heap = BinaryHeap::with_capacity(capacity);

    let current_next_use = |vertex_id: &u32, future_positions: &HashMap<u32, VecDeque<usize>>| {
        future_positions
            .get(vertex_id)
            .and_then(|positions| positions.front().copied())
            .unwrap_or(usize::MAX)
    };

    for (idx, vertex_id) in trace.iter().enumerate() {
        stats.accesses += 1;
        if let Some(positions) = future_positions.get_mut(vertex_id) {
            debug_assert_eq!(positions.front().copied(), Some(idx));
            positions.pop_front();
        }

        if resident.contains(vertex_id) {
            stats.hits += 1;
            eviction_heap.push((current_next_use(vertex_id, &future_positions), *vertex_id));
            continue;
        }

        stats.misses += 1;
        if resident.len() == capacity {
            let Some(victim) = pop_belady_victim(&mut eviction_heap, &resident, &future_positions)
            else {
                stats.rejections += 1;
                continue;
            };
            resident.remove(&victim);
            stats.evictions += 1;
        }
        resident.insert(*vertex_id);
        eviction_heap.push((current_next_use(vertex_id, &future_positions), *vertex_id));
        stats.admissions += 1;
    }

    stats
}

fn pop_belady_victim(
    eviction_heap: &mut BinaryHeap<(usize, u32)>,
    resident: &HashSet<u32>,
    future_positions: &HashMap<u32, VecDeque<usize>>,
) -> Option<u32> {
    while let Some((next_use, vertex_id)) = eviction_heap.pop() {
        if !resident.contains(&vertex_id) {
            continue;
        }

        let current_next_use = future_positions
            .get(&vertex_id)
            .and_then(|positions| positions.front().copied())
            .unwrap_or(usize::MAX);
        if next_use == current_next_use {
            return Some(vertex_id);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::GraphDataF32VectorUnitData;
    use diskann::graph::AdjacencyList;

    fn touch(cache: &mut PolicyCache, vertex_id: u32) -> bool {
        cache.record_access(vertex_id);
        if cache.contains(vertex_id) {
            true
        } else {
            cache.admit(vertex_id);
            false
        }
    }

    fn cached_node(first_value: f32, neighbor: u32) -> CachedNode<GraphDataF32VectorUnitData> {
        CachedNode {
            vector: vec![first_value, first_value + 1.0],
            adjacency_list: AdjacencyList::from_iter_untrusted([neighbor]),
            associated_data: (),
        }
    }

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

    #[test]
    fn belady_counts_expected_hits() {
        let trace = [1, 2, 3, 1, 2, 4, 1, 2, 3, 4];
        let opt = replay_belady_optimal(3, &trace);

        assert_eq!(opt.accesses, 10);
        assert_eq!(opt.hits, 5);
        assert_eq!(opt.misses, 5);
        assert_eq!(opt.evictions, 2);
    }

    #[test]
    fn dynamic_hit_handle_survives_eviction() {
        let mut cache =
            DynamicNodeCache::<GraphDataF32VectorUnitData>::new(2, 1, CachePolicyKind::Fifo)
                .unwrap();

        cache.admit_node(&1, cached_node(1.0, 7)).unwrap();
        let cached = cache.lookup(&1).unwrap();

        cache.admit_node(&2, cached_node(3.0, 8)).unwrap();

        assert!(!cache.contains(&1));
        assert_eq!(cached.vector, vec![1.0, 2.0]);
        assert_eq!(
            cached.adjacency_list.iter().copied().collect::<Vec<_>>(),
            vec![7]
        );
    }

    #[test]
    fn sharded_capacity_split_assigns_remainder_to_early_shards() {
        let cache = ShardedDynamicNodeCache::<GraphDataF32VectorUnitData>::new(
            2,
            10,
            CachePolicyKind::Fifo,
            4,
        )
        .unwrap();

        assert_eq!(cache.shard_capacities().unwrap(), vec![3, 3, 2, 2]);
        assert_eq!(
            (0..4)
                .map(|shard_id| {
                    ShardedDynamicNodeCache::<GraphDataF32VectorUnitData>::capacity_for_shard(
                        10, 4, shard_id,
                    )
                })
                .collect::<Vec<_>>(),
            vec![3, 3, 2, 2]
        );
    }

    #[test]
    fn sharded_cache_maps_same_vertex_to_same_shard() {
        let cache = ShardedDynamicNodeCache::<GraphDataF32VectorUnitData>::new(
            2,
            10,
            CachePolicyKind::Fifo,
            4,
        )
        .unwrap();

        assert_eq!(cache.shard_index(&42), cache.shard_index(&42));
        assert_eq!(
            ShardedDynamicNodeCache::<GraphDataF32VectorUnitData>::shard_index_for(42, 4),
            cache.shard_index(&42)
        );
    }

    #[test]
    fn sharded_fifo_stays_within_total_capacity() {
        let cache = ShardedDynamicNodeCache::<GraphDataF32VectorUnitData>::new(
            2,
            3,
            CachePolicyKind::Fifo,
            2,
        )
        .unwrap();

        for vertex_id in 0..20 {
            cache
                .admit_node(&vertex_id, cached_node(vertex_id as f32, vertex_id + 1))
                .unwrap();
            assert!(cache.len() <= cache.capacity());
        }
    }

    #[test]
    fn sharded_stats_aggregate_shard_stats() {
        let cache = ShardedDynamicNodeCache::<GraphDataF32VectorUnitData>::new(
            2,
            4,
            CachePolicyKind::Fifo,
            2,
        )
        .unwrap();

        cache.admit_node(&1, cached_node(1.0, 7)).unwrap();
        assert!(cache.lookup(&1).unwrap().is_some());
        assert!(cache.lookup(&2).unwrap().is_none());

        let stats = cache.stats();
        assert_eq!(stats.accesses, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn sharded_clock_gives_second_chance_within_shard() {
        let cache = ShardedDynamicNodeCache::<GraphDataF32VectorUnitData>::new(
            2,
            2,
            CachePolicyKind::Clock,
            1,
        )
        .unwrap();
        cache.admit_node(&1, cached_node(1.0, 7)).unwrap();
        cache.admit_node(&2, cached_node(2.0, 8)).unwrap();

        {
            let mut shard = cache.shards[0].lock().unwrap();
            shard.policy.clock_refs.insert(1, false);
            shard.policy.clock_refs.insert(2, false);
        }

        assert!(cache.lookup(&1).unwrap().is_some());
        cache.admit_node(&3, cached_node(3.0, 9)).unwrap();

        assert!(cache.contains(&1).unwrap());
        assert!(cache.contains(&3).unwrap());
        assert!(!cache.contains(&2).unwrap());
    }

    #[test]
    fn clock_gives_second_chance_to_referenced_node() {
        let mut cache = PolicyCache::new(CachePolicyKind::Clock, 2).unwrap();
        cache.admit(1);
        cache.admit(2);
        cache.clock_refs.insert(1, false);
        cache.clock_refs.insert(2, false);

        cache.record_access(1);
        let outcome = cache.admit(3);

        assert_eq!(outcome.evicted, Some(2));
        assert!(cache.contains(1));
        assert!(cache.contains(3));
        assert!(!cache.contains(2));
    }

    #[test]
    fn twoq_promotes_a1out_hit_to_main_queue() {
        let mut cache = PolicyCache::new(CachePolicyKind::TwoQ, 2).unwrap();
        cache.admit(1);
        cache.admit(2);
        cache.admit(3);

        assert!(cache.twoq_a1out_set.contains(&1));

        cache.admit(1);

        assert_eq!(cache.twoq_segment.get(&1), Some(&TwoQSegment::Am));
        assert!(cache.contains(1));
    }

    #[test]
    fn slru_promotes_hit_and_evicts_probationary_first() {
        let mut cache = PolicyCache::new(CachePolicyKind::Slru, 2).unwrap();
        cache.admit(1);
        cache.admit(2);
        cache.record_access(1);
        let outcome = cache.admit(3);

        assert_eq!(outcome.evicted, Some(2));
        assert_eq!(cache.slru_segment.get(&1), Some(&SlruSegment::Protected));
        assert!(cache.contains(1));
        assert!(cache.contains(3));
    }

    #[test]
    fn arc_adapts_after_b1_ghost_hit() {
        let mut cache = PolicyCache::new(CachePolicyKind::Arc, 2).unwrap();
        cache.admit(1);
        cache.admit(2);
        cache.admit(3);

        assert!(cache.arc_b1_set.contains(&1));

        cache.admit(1);

        assert!(cache.arc_t2_set.contains(&1));
        assert!(cache.arc_p > 0);
    }

    #[test]
    fn wtiny_lfu_rejects_weaker_window_candidate() {
        let mut cache = PolicyCache::new(CachePolicyKind::WTinyLfu, 2).unwrap();
        touch(&mut cache, 1);
        touch(&mut cache, 2);
        touch(&mut cache, 1);

        let outcome = cache.admit(3);

        assert_eq!(outcome.evicted, Some(2));
        assert!(cache.contains(1));
        assert!(cache.contains(3));
        assert!(!cache.contains(2));
    }

    #[test]
    fn gdsf_keeps_more_frequent_node() {
        let mut cache = PolicyCache::new(CachePolicyKind::Gdsf, 2).unwrap();
        cache.admit(1);
        cache.admit(2);
        cache.record_access(1);
        cache.record_access(1);

        let outcome = cache.admit(3);

        assert_eq!(outcome.evicted, Some(2));
        assert!(cache.contains(1));
        assert!(cache.contains(3));
    }

    #[test]
    fn cacheus_promotes_q_hit_to_s_segment() {
        let mut cache = PolicyCache::new(CachePolicyKind::Cacheus, 4).unwrap();
        cache.admit(1);
        cache.admit(2);
        cache.admit(3);
        cache.admit(4);

        assert_eq!(cache.cacheus_segment.get(&4), Some(&CacheusSegment::Q));

        cache.record_access(4);

        assert_eq!(cache.cacheus_segment.get(&4), Some(&CacheusSegment::S));
        assert!(cache.contains(4));
    }

    #[test]
    fn added_policies_stay_within_capacity_on_replay() {
        let policies = [
            CachePolicyKind::Clock,
            CachePolicyKind::TwoQ,
            CachePolicyKind::Slru,
            CachePolicyKind::Lirs,
            CachePolicyKind::Arc,
            CachePolicyKind::Gdsf,
            CachePolicyKind::WTinyLfu,
            CachePolicyKind::LeCar,
            CachePolicyKind::Cacheus,
        ];
        let trace = [1, 2, 3, 1, 4, 5, 1, 2, 6, 3, 7, 1, 2, 8, 9, 1];

        for policy in policies {
            let mut cache = PolicyCache::new(policy, 3).unwrap();
            for vertex_id in trace {
                touch(&mut cache, vertex_id);
                assert!(
                    cache.len() <= cache.capacity(),
                    "{policy} exceeded capacity after vertex {vertex_id}"
                );
            }
        }
    }

    #[test]
    fn warmed_policies_initialize_eviction_state() {
        let policies = [
            CachePolicyKind::Fifo,
            CachePolicyKind::Lru,
            CachePolicyKind::Lfu,
            CachePolicyKind::Random,
            CachePolicyKind::TinyLfu,
            CachePolicyKind::WTinyLfu,
            CachePolicyKind::Clock,
            CachePolicyKind::TwoQ,
            CachePolicyKind::Slru,
            CachePolicyKind::Lirs,
            CachePolicyKind::Arc,
            CachePolicyKind::Gdsf,
            CachePolicyKind::LeCar,
            CachePolicyKind::Cacheus,
        ];

        for policy in policies {
            let mut cache = PolicyCache::new(policy, 3).unwrap();
            cache.warm(1);
            cache.warm(2);
            cache.warm(3);
            assert_eq!(cache.len(), 3, "{policy} did not fill warm cache");

            cache.admit(4);
            assert!(
                cache.len() <= cache.capacity(),
                "{policy} exceeded capacity after warm-start admission"
            );
        }
    }
}
