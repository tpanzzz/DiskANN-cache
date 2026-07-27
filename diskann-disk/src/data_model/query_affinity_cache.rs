/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    cmp::Ordering,
    collections::VecDeque,
    fs::File,
    io::{BufReader, Read},
    sync::Arc,
};

use diskann::{ANNError, ANNResult};
use diskann_vector::distance::Metric;
use hashbrown::HashMap;

use crate::data_model::{CachedNode, GraphDataType};

const ROUTE_EPSILON: f64 = 1e-6;
const CHARGE_EPSILON: f64 = 1e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct QueryAffinityCacheSettings {
    pub capacity: usize,
    pub prototype_file: String,
    pub metric: Metric,
    pub top_m: usize,
    pub global_fraction: f64,
    pub global_entropy_threshold: f64,
    pub global_min_observations: f64,
    pub min_admission_observations: f64,
    pub max_affinity_clusters: usize,
    pub decay_interval: u64,
    pub quota_update_interval: u64,
    pub ghost_capacity_multiplier: usize,
    pub admission_margin: f64,
}

impl QueryAffinityCacheSettings {
    pub fn validate(&self) -> ANNResult<()> {
        if self.capacity == 0 {
            return Err(ANNError::log_index_error(
                "QASC capacity must be greater than zero",
            ));
        }
        if self.prototype_file.is_empty() {
            return Err(ANNError::log_index_error(
                "QASC prototype file must not be empty",
            ));
        }
        if self.top_m == 0 {
            return Err(ANNError::log_index_error(
                "QASC top_m must be greater than zero",
            ));
        }
        if !(0.0..=1.0).contains(&self.global_fraction) {
            return Err(ANNError::log_index_error(
                "QASC global_fraction must be in [0, 1]",
            ));
        }
        if !(0.0..=1.0).contains(&self.global_entropy_threshold) {
            return Err(ANNError::log_index_error(
                "QASC global_entropy_threshold must be in [0, 1]",
            ));
        }
        if self.global_min_observations < 0.0 || self.min_admission_observations < 0.0 {
            return Err(ANNError::log_index_error(
                "QASC observation thresholds must be non-negative",
            ));
        }
        if self.max_affinity_clusters == 0 {
            return Err(ANNError::log_index_error(
                "QASC max_affinity_clusters must be greater than zero",
            ));
        }
        if self.decay_interval == 0 || self.quota_update_interval == 0 {
            return Err(ANNError::log_index_error(
                "QASC decay and quota intervals must be greater than zero",
            ));
        }
        if self.ghost_capacity_multiplier == 0 {
            return Err(ANNError::log_index_error(
                "QASC ghost_capacity_multiplier must be greater than zero",
            ));
        }
        if !self.admission_margin.is_finite() || self.admission_margin < 0.0 {
            return Err(ANNError::log_index_error(
                "QASC admission_margin must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct QueryPrototypes {
    dimensions: usize,
    vectors: Vec<f32>,
    norms: Vec<f64>,
}

impl QueryPrototypes {
    pub fn from_fbin(path: &str, expected_dimensions: usize) -> ANNResult<Self> {
        let file = File::open(path).map_err(ANNError::log_index_error)?;
        let mut reader = BufReader::new(file);
        let mut header = [0u8; 8];
        reader
            .read_exact(&mut header)
            .map_err(ANNError::log_index_error)?;
        let rows = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        let dimensions = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
        if rows == 0 {
            return Err(ANNError::log_index_error(
                "QASC prototype file must contain at least one vector",
            ));
        }
        if dimensions != expected_dimensions {
            return Err(ANNError::log_index_error(format!(
                "QASC prototype dimension {dimensions} does not match index dimension {expected_dimensions}"
            )));
        }

        let value_count = rows.checked_mul(dimensions).ok_or_else(|| {
            ANNError::log_index_error("QASC prototype matrix size overflowed usize")
        })?;
        let byte_count = value_count.checked_mul(4).ok_or_else(|| {
            ANNError::log_index_error("QASC prototype byte size overflowed usize")
        })?;
        let mut bytes = vec![0u8; byte_count];
        reader
            .read_exact(&mut bytes)
            .map_err(ANNError::log_index_error)?;
        let mut trailing = [0u8; 1];
        if reader
            .read(&mut trailing)
            .map_err(ANNError::log_index_error)?
            != 0
        {
            return Err(ANNError::log_index_error(
                "QASC prototype file contains trailing bytes",
            ));
        }

        let mut vectors = Vec::with_capacity(value_count);
        for chunk in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().unwrap());
            if !value.is_finite() {
                return Err(ANNError::log_index_error(
                    "QASC prototype vectors must contain only finite values",
                ));
            }
            vectors.push(value);
        }
        Self::new(dimensions, vectors)
    }

    pub fn new(dimensions: usize, vectors: Vec<f32>) -> ANNResult<Self> {
        if dimensions == 0 || vectors.is_empty() || vectors.len() % dimensions != 0 {
            return Err(ANNError::log_index_error(
                "QASC prototype matrix must be non-empty and rectangular",
            ));
        }
        if vectors.iter().any(|value| !value.is_finite()) {
            return Err(ANNError::log_index_error(
                "QASC prototype vectors must contain only finite values",
            ));
        }
        let norms = vectors
            .chunks_exact(dimensions)
            .map(|vector| {
                vector
                    .iter()
                    .map(|value| f64::from(*value) * f64::from(*value))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        Ok(Self {
            dimensions,
            vectors,
            norms,
        })
    }

    pub fn len(&self) -> usize {
        self.norms.len()
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn route(&self, query: &[f32], metric: Metric, top_m: usize) -> ANNResult<QueryRoute> {
        if query.len() != self.dimensions {
            return Err(ANNError::log_index_error(format!(
                "QASC query dimension {} does not match prototype dimension {}",
                query.len(),
                self.dimensions
            )));
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(ANNError::log_index_error(
                "QASC query vectors must contain only finite values",
            ));
        }
        if top_m == 0 || top_m > self.len() {
            return Err(ANNError::log_index_error(format!(
                "QASC top_m {top_m} is outside [1, {}]",
                self.len()
            )));
        }

        let query_norm = query
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if matches!(metric, Metric::Cosine | Metric::CosineNormalized) && query_norm == 0.0 {
            return Err(ANNError::log_index_error(
                "QASC cosine routing does not support zero-norm queries",
            ));
        }

        let mut distances = Vec::with_capacity(self.len());
        for (cluster_id, prototype) in self.vectors.chunks_exact(self.dimensions).enumerate() {
            let dot = query
                .iter()
                .zip(prototype)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum::<f64>();
            let distance = match metric {
                Metric::L2 => query
                    .iter()
                    .zip(prototype)
                    .map(|(left, right)| {
                        let delta = f64::from(*left) - f64::from(*right);
                        delta * delta
                    })
                    .sum(),
                Metric::Cosine => {
                    let prototype_norm = self.norms[cluster_id];
                    if prototype_norm == 0.0 {
                        return Err(ANNError::log_index_error(
                            "QASC cosine prototypes must have non-zero norm",
                        ));
                    }
                    1.0 - dot / (query_norm * prototype_norm)
                }
                Metric::CosineNormalized => 1.0 - dot,
                Metric::InnerProduct => -dot,
            };
            distances.push((cluster_id, distance));
        }
        distances.sort_unstable_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        distances.truncate(top_m);

        let min_distance = distances[0].1;
        let raw_weights: Vec<f64> = distances
            .iter()
            .map(|(_, distance)| 1.0 / (ROUTE_EPSILON + distance - min_distance))
            .collect();
        let total_weight: f64 = raw_weights.iter().sum();
        let mut weights: Vec<(usize, f64)> = distances
            .into_iter()
            .zip(raw_weights)
            .map(|((cluster_id, _), weight)| (cluster_id, weight / total_weight))
            .collect();
        weights.sort_unstable_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        let confidence = weights[0].1;
        Ok(QueryRoute {
            weights,
            confidence,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryRoute {
    pub weights: Vec<(usize, f64)>,
    pub confidence: f64,
}

impl QueryRoute {
    pub fn hard(cluster_id: usize) -> Self {
        Self {
            weights: vec![(cluster_id, 1.0)],
            confidence: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryAffinityRole {
    Global,
    Conditional,
}

#[derive(Clone, Debug, Default)]
struct NodeProfile {
    total_observations: f64,
    affinities: Vec<(usize, f64)>,
    last_observed_tick: u64,
}

#[derive(Clone, Debug)]
struct ResidentMetadata {
    last_access_tick: u64,
    inserted_tick: u64,
    role: QueryAffinityRole,
    charges: Vec<(usize, f64)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueryAffinityCacheStats {
    pub accesses: u64,
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub evictions: u64,
    pub rejections: u64,
    pub role_promotions: u64,
    pub role_demotions: u64,
    pub quota_recomputations: u64,
    pub borrowed_admissions: u64,
}

pub struct QueryAffinityCache<Data: GraphDataType<VectorIdType = u32>> {
    dimension: usize,
    settings: QueryAffinityCacheSettings,
    prototypes: QueryPrototypes,
    store: HashMap<u32, Arc<CachedNode<Data>>>,
    resident_metadata: HashMap<u32, ResidentMetadata>,
    profiles: HashMap<u32, NodeProfile>,
    ghost_order: VecDeque<(u32, u64)>,
    cluster_arrival_mass: Vec<f64>,
    cluster_quotas: Vec<f64>,
    cluster_usage: Vec<f64>,
    global_occupancy: usize,
    stats: QueryAffinityCacheStats,
    tick: u64,
    query_count: u64,
}

impl<Data> QueryAffinityCache<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    pub fn new(
        dimension: usize,
        settings: QueryAffinityCacheSettings,
        prototypes: QueryPrototypes,
    ) -> ANNResult<Self> {
        settings.validate()?;
        if prototypes.dimensions() != dimension {
            return Err(ANNError::log_index_error(format!(
                "QASC prototype dimension {} does not match cache dimension {dimension}",
                prototypes.dimensions()
            )));
        }
        if settings.top_m > prototypes.len() {
            return Err(ANNError::log_index_error(format!(
                "QASC top_m {} exceeds prototype count {}",
                settings.top_m,
                prototypes.len()
            )));
        }
        let cluster_count = prototypes.len();
        let mut cache = Self {
            dimension,
            store: HashMap::with_capacity(settings.capacity),
            resident_metadata: HashMap::with_capacity(settings.capacity),
            profiles: HashMap::with_capacity(
                settings
                    .capacity
                    .saturating_mul(settings.ghost_capacity_multiplier),
            ),
            ghost_order: VecDeque::new(),
            cluster_arrival_mass: vec![0.0; cluster_count],
            cluster_quotas: vec![0.0; cluster_count],
            cluster_usage: vec![0.0; cluster_count],
            global_occupancy: 0,
            stats: QueryAffinityCacheStats::default(),
            tick: 0,
            query_count: 0,
            settings,
            prototypes,
        };
        cache.recompute_quotas();
        Ok(cache)
    }

    pub fn begin_query(&mut self, query: &[f32]) -> ANNResult<QueryRoute> {
        let route = self
            .prototypes
            .route(query, self.settings.metric, self.settings.top_m)?;
        self.query_count = self.query_count.saturating_add(1);
        for (cluster_id, weight) in &route.weights {
            self.cluster_arrival_mass[*cluster_id] += *weight;
        }
        if self.query_count % self.settings.quota_update_interval == 0 {
            self.recompute_quotas();
        }
        Ok(route)
    }

    pub fn lookup(&mut self, vertex_id: u32, route: &QueryRoute) -> Option<Arc<CachedNode<Data>>> {
        self.stats.accesses = self.stats.accesses.saturating_add(1);
        self.observe(vertex_id, route);
        let node = self.store.get(&vertex_id).cloned();
        if node.is_some() {
            self.stats.hits = self.stats.hits.saturating_add(1);
            if let Some(metadata) = self.resident_metadata.get_mut(&vertex_id) {
                metadata.last_access_tick = self.tick;
            }
            self.refresh_resident_accounting(vertex_id);
        } else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            self.ghost_order.push_back((vertex_id, self.tick));
            self.prune_ghost_profiles();
        }
        node
    }

    pub fn admit_node(
        &mut self,
        vertex_id: u32,
        node: CachedNode<Data>,
        route: &QueryRoute,
    ) -> ANNResult<()> {
        if node.vector.len() != self.dimension {
            return Err(ANNError::log_index_error(
                "Vector dimension does not match the dimension set in QASC",
            ));
        }
        if self.store.contains_key(&vertex_id) {
            self.store.insert(vertex_id, Arc::new(node));
            return Ok(());
        }
        self.profiles.entry(vertex_id).or_default();

        if self.store.len() < self.settings.capacity {
            let borrowed = self.is_borrowed_candidate(vertex_id);
            self.insert_resident(vertex_id, node);
            if borrowed {
                self.stats.borrowed_admissions = self.stats.borrowed_admissions.saturating_add(1);
            }
            return Ok(());
        }

        let candidate_observations = self.profiles[&vertex_id].total_observations;
        if candidate_observations < self.settings.min_admission_observations {
            self.stats.rejections = self.stats.rejections.saturating_add(1);
            return Ok(());
        }

        let Some(victim) = self.choose_victim(vertex_id, route) else {
            self.stats.rejections = self.stats.rejections.saturating_add(1);
            return Ok(());
        };
        let candidate_utility = self.route_utility(vertex_id, route);
        let victim_utility = self.route_utility(victim, route);
        if candidate_utility < self.settings.admission_margin * victim_utility {
            self.stats.rejections = self.stats.rejections.saturating_add(1);
            return Ok(());
        }

        self.remove_resident(victim);
        self.stats.evictions = self.stats.evictions.saturating_add(1);
        let borrowed = self.is_borrowed_candidate(vertex_id);
        self.insert_resident(vertex_id, node);
        if borrowed {
            self.stats.borrowed_admissions = self.stats.borrowed_admissions.saturating_add(1);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn capacity(&self) -> usize {
        self.settings.capacity
    }

    pub fn contains(&self, vertex_id: u32) -> bool {
        self.store.contains_key(&vertex_id)
    }

    pub fn stats(&self) -> QueryAffinityCacheStats {
        self.stats
    }

    pub fn global_occupancy(&self) -> usize {
        self.global_occupancy
    }

    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    pub fn cluster_quotas(&self) -> &[f64] {
        &self.cluster_quotas
    }

    pub fn cluster_usage(&self) -> &[f64] {
        &self.cluster_usage
    }

    pub fn role(&self, vertex_id: u32) -> Option<QueryAffinityRole> {
        self.resident_metadata
            .get(&vertex_id)
            .map(|metadata| metadata.role)
    }

    pub fn charges(&self, vertex_id: u32) -> Option<&[(usize, f64)]> {
        self.resident_metadata
            .get(&vertex_id)
            .map(|metadata| metadata.charges.as_slice())
    }

    pub fn validate_invariants(&self) -> ANNResult<()> {
        if self.store.len() > self.settings.capacity {
            return Err(ANNError::log_index_error("QASC physical capacity exceeded"));
        }
        if self.store.len() != self.resident_metadata.len()
            || self
                .store
                .keys()
                .any(|vertex_id| !self.resident_metadata.contains_key(vertex_id))
        {
            return Err(ANNError::log_index_error(
                "QASC store and resident metadata keys differ",
            ));
        }
        let mut expected_usage = vec![0.0; self.prototypes.len()];
        let mut expected_global = 0usize;
        let mut conditional_count = 0usize;
        for (vertex_id, metadata) in &self.resident_metadata {
            if !self.profiles.contains_key(vertex_id) {
                return Err(ANNError::log_index_error(
                    "QASC resident node is missing its affinity profile",
                ));
            }
            match metadata.role {
                QueryAffinityRole::Global => {
                    expected_global += 1;
                    if !metadata.charges.is_empty() {
                        return Err(ANNError::log_index_error(
                            "QASC global node has conditional charges",
                        ));
                    }
                }
                QueryAffinityRole::Conditional => {
                    conditional_count += 1;
                    let total_charge: f64 = metadata.charges.iter().map(|(_, charge)| charge).sum();
                    if (total_charge - 1.0).abs() > 1e-6 {
                        return Err(ANNError::log_index_error(
                            "QASC conditional charges do not sum to one",
                        ));
                    }
                    for (cluster_id, charge) in &metadata.charges {
                        expected_usage[*cluster_id] += *charge;
                    }
                }
            }
        }
        if expected_global != self.global_occupancy
            || expected_global + conditional_count != self.store.len()
        {
            return Err(ANNError::log_index_error(
                "QASC global/conditional occupancy is inconsistent",
            ));
        }
        if expected_usage
            .iter()
            .zip(&self.cluster_usage)
            .any(|(expected, actual)| (expected - actual).abs() > 1e-6)
        {
            return Err(ANNError::log_index_error(
                "QASC logical cluster usage is inconsistent",
            ));
        }
        Ok(())
    }

    fn observe(&mut self, vertex_id: u32, route: &QueryRoute) {
        self.tick = self.tick.saturating_add(1);
        if self.tick % self.settings.decay_interval == 0 {
            self.decay();
        }
        let profile = self.profiles.entry(vertex_id).or_default();
        profile.total_observations += 1.0;
        profile.last_observed_tick = self.tick;
        for (cluster_id, weight) in &route.weights {
            Self::add_affinity(
                profile,
                *cluster_id,
                *weight,
                self.settings.max_affinity_clusters,
            );
        }
    }

    fn add_affinity(
        profile: &mut NodeProfile,
        cluster_id: usize,
        weight: f64,
        max_affinity_clusters: usize,
    ) {
        if let Some((_, affinity)) = profile
            .affinities
            .iter_mut()
            .find(|(existing, _)| *existing == cluster_id)
        {
            *affinity += weight;
            return;
        }
        if profile.affinities.len() < max_affinity_clusters {
            profile.affinities.push((cluster_id, weight));
            return;
        }
        let min_index = profile
            .affinities
            .iter()
            .enumerate()
            .min_by(|left, right| {
                left.1
                     .1
                    .partial_cmp(&right.1 .1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| right.1 .0.cmp(&left.1 .0))
            })
            .map(|(index, _)| index)
            .unwrap();
        if weight > profile.affinities[min_index].1 {
            profile.affinities[min_index] = (cluster_id, weight);
        }
    }

    fn classify_profile(&self, profile: &NodeProfile) -> (QueryAffinityRole, Vec<(usize, f64)>) {
        let total_affinity: f64 = profile.affinities.iter().map(|(_, value)| value).sum();
        if total_affinity <= 0.0 {
            return (QueryAffinityRole::Conditional, vec![(0, 1.0)]);
        }
        let entropy = profile
            .affinities
            .iter()
            .map(|(_, value)| {
                let probability = value / total_affinity;
                if probability > 0.0 {
                    -probability * probability.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        let normalized_entropy = if self.prototypes.len() <= 1 {
            1.0
        } else {
            entropy / (self.prototypes.len() as f64).ln()
        };
        if profile.total_observations >= self.settings.global_min_observations
            && normalized_entropy >= self.settings.global_entropy_threshold
        {
            return (QueryAffinityRole::Global, Vec::new());
        }
        let mut charges: Vec<_> = profile
            .affinities
            .iter()
            .map(|(cluster_id, value)| (*cluster_id, value / total_affinity))
            .collect();
        charges.sort_unstable_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        (QueryAffinityRole::Conditional, charges)
    }

    fn insert_resident(&mut self, vertex_id: u32, node: CachedNode<Data>) {
        let profile = self.profiles.entry(vertex_id).or_default().clone();
        let (role, charges) = self.classify_profile(&profile);
        let metadata = ResidentMetadata {
            last_access_tick: self.tick,
            inserted_tick: self.tick,
            role,
            charges,
        };
        self.add_accounting(&metadata);
        self.resident_metadata.insert(vertex_id, metadata);
        self.store.insert(vertex_id, Arc::new(node));
        self.stats.admissions = self.stats.admissions.saturating_add(1);
    }

    fn remove_resident(&mut self, vertex_id: u32) {
        if let Some(metadata) = self.resident_metadata.remove(&vertex_id) {
            self.remove_accounting(&metadata);
        }
        self.store.remove(&vertex_id);
    }

    fn refresh_resident_accounting(&mut self, vertex_id: u32) {
        let Some(profile) = self.profiles.get(&vertex_id) else {
            return;
        };
        let (new_role, new_charges) = self.classify_profile(profile);
        let Some(old_metadata) = self.resident_metadata.get(&vertex_id).cloned() else {
            return;
        };
        if old_metadata.role == new_role && old_metadata.charges == new_charges {
            return;
        }
        self.remove_accounting(&old_metadata);
        let mut new_metadata = old_metadata;
        if new_metadata.role != new_role {
            match (new_metadata.role, new_role) {
                (QueryAffinityRole::Conditional, QueryAffinityRole::Global) => {
                    self.stats.role_promotions = self.stats.role_promotions.saturating_add(1);
                }
                (QueryAffinityRole::Global, QueryAffinityRole::Conditional) => {
                    self.stats.role_demotions = self.stats.role_demotions.saturating_add(1);
                }
                _ => {}
            }
        }
        new_metadata.role = new_role;
        new_metadata.charges = new_charges;
        self.add_accounting(&new_metadata);
        self.resident_metadata.insert(vertex_id, new_metadata);
    }

    fn add_accounting(&mut self, metadata: &ResidentMetadata) {
        match metadata.role {
            QueryAffinityRole::Global => self.global_occupancy += 1,
            QueryAffinityRole::Conditional => {
                for (cluster_id, charge) in &metadata.charges {
                    self.cluster_usage[*cluster_id] += *charge;
                }
            }
        }
    }

    fn remove_accounting(&mut self, metadata: &ResidentMetadata) {
        match metadata.role {
            QueryAffinityRole::Global => {
                self.global_occupancy = self.global_occupancy.saturating_sub(1)
            }
            QueryAffinityRole::Conditional => {
                for (cluster_id, charge) in &metadata.charges {
                    self.cluster_usage[*cluster_id] =
                        (self.cluster_usage[*cluster_id] - *charge).max(0.0);
                }
            }
        }
    }

    fn global_target(&self) -> usize {
        ((self.settings.capacity as f64 * self.settings.global_fraction).round() as usize)
            .min(self.settings.capacity)
    }

    fn conditional_budget(&self) -> f64 {
        self.settings.capacity.saturating_sub(self.global_target()) as f64
    }

    fn recompute_quotas(&mut self) {
        let weights: Vec<f64> = self
            .cluster_arrival_mass
            .iter()
            .map(|mass| (mass + 1.0).sqrt())
            .collect();
        let total_weight: f64 = weights.iter().sum();
        let budget = self.conditional_budget();
        for (quota, weight) in self.cluster_quotas.iter_mut().zip(weights) {
            *quota = if total_weight > 0.0 {
                budget * weight / total_weight
            } else {
                0.0
            };
        }
        self.stats.quota_recomputations = self.stats.quota_recomputations.saturating_add(1);
    }

    fn is_borrowed_candidate(&self, vertex_id: u32) -> bool {
        let Some(profile) = self.profiles.get(&vertex_id) else {
            return false;
        };
        let (role, charges) = self.classify_profile(profile);
        if role == QueryAffinityRole::Global {
            return self.global_occupancy >= self.global_target();
        }
        charges.iter().any(|(cluster_id, charge)| {
            self.cluster_usage[*cluster_id] + *charge > self.cluster_quotas[*cluster_id]
        })
    }

    fn choose_victim(&self, candidate: u32, route: &QueryRoute) -> Option<u32> {
        let profile = self.profiles.get(&candidate)?;
        let (candidate_role, candidate_charges) = self.classify_profile(profile);
        if candidate_role == QueryAffinityRole::Global {
            if self.global_occupancy >= self.global_target() {
                if let Some(victim) = self.minimum_global_utility() {
                    return Some(victim);
                }
            }
            if let Some(cluster_id) = self.most_overquota_cluster() {
                if let Some(victim) = self.minimum_cluster_utility(cluster_id) {
                    return Some(victim);
                }
            }
        } else {
            let primary_cluster = candidate_charges.first().map(|(cluster_id, _)| *cluster_id);
            let target_cluster = self.most_overquota_cluster().or(primary_cluster);
            if let Some(cluster_id) = target_cluster {
                if let Some(victim) = self.minimum_cluster_utility(cluster_id) {
                    return Some(victim);
                }
            }
            if self.global_occupancy > self.global_target() {
                if let Some(victim) = self.minimum_global_utility() {
                    return Some(victim);
                }
            }
        }
        self.minimum_by(|vertex_id| self.route_utility(vertex_id, route), |_| true)
    }

    fn most_overquota_cluster(&self) -> Option<usize> {
        self.cluster_usage
            .iter()
            .zip(&self.cluster_quotas)
            .enumerate()
            .filter_map(|(cluster_id, (usage, quota))| {
                let overage = usage - quota;
                (overage > CHARGE_EPSILON).then_some((cluster_id, overage))
            })
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(cluster_id, _)| cluster_id)
    }

    fn minimum_global_utility(&self) -> Option<u32> {
        self.minimum_by(
            |vertex_id| self.global_utility(vertex_id),
            |metadata| metadata.role == QueryAffinityRole::Global,
        )
    }

    fn minimum_cluster_utility(&self, cluster_id: usize) -> Option<u32> {
        let route = QueryRoute::hard(cluster_id);
        self.minimum_by(
            |vertex_id| self.route_utility(vertex_id, &route),
            |metadata| {
                metadata.role == QueryAffinityRole::Conditional
                    && metadata
                        .charges
                        .iter()
                        .any(|(id, charge)| *id == cluster_id && *charge > CHARGE_EPSILON)
            },
        )
    }

    fn minimum_by<F, P>(&self, utility: F, predicate: P) -> Option<u32>
    where
        F: Fn(u32) -> f64,
        P: Fn(&ResidentMetadata) -> bool,
    {
        self.resident_metadata
            .iter()
            .filter(|(_, metadata)| predicate(metadata))
            .map(|(vertex_id, metadata)| (*vertex_id, utility(*vertex_id), metadata.inserted_tick))
            .min_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.2.cmp(&right.2))
                    .then_with(|| left.0.cmp(&right.0))
            })
            .map(|(vertex_id, _, _)| vertex_id)
    }

    fn route_utility(&self, vertex_id: u32, route: &QueryRoute) -> f64 {
        let Some(profile) = self.profiles.get(&vertex_id) else {
            return 0.0;
        };
        let route_affinity = route
            .weights
            .iter()
            .map(|(cluster_id, route_weight)| {
                let affinity = profile
                    .affinities
                    .iter()
                    .find(|(id, _)| id == cluster_id)
                    .map(|(_, value)| *value)
                    .unwrap_or(0.0);
                route_weight * affinity
            })
            .sum::<f64>();
        route_affinity
            + 0.10 * (1.0 + profile.total_observations).ln()
            + 0.01 * self.freshness(profile)
    }

    fn global_utility(&self, vertex_id: u32) -> f64 {
        let Some(profile) = self.profiles.get(&vertex_id) else {
            return 0.0;
        };
        let total_affinity: f64 = profile.affinities.iter().map(|(_, value)| value).sum();
        if total_affinity <= 0.0 {
            return 0.0;
        }
        let entropy = profile
            .affinities
            .iter()
            .map(|(_, value)| {
                let probability = value / total_affinity;
                if probability > 0.0 {
                    -probability * probability.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        let normalized_entropy = if self.prototypes.len() <= 1 {
            1.0
        } else {
            entropy / (self.prototypes.len() as f64).ln()
        };
        profile.total_observations * normalized_entropy.max(ROUTE_EPSILON)
            + 0.01 * self.freshness(profile)
    }

    fn freshness(&self, profile: &NodeProfile) -> f64 {
        let age = self.tick.saturating_sub(profile.last_observed_tick) as f64;
        1.0 / (1.0 + age / self.settings.decay_interval as f64)
    }

    fn decay(&mut self) {
        for mass in &mut self.cluster_arrival_mass {
            *mass *= 0.5;
        }
        for profile in self.profiles.values_mut() {
            profile.total_observations *= 0.5;
            for (_, affinity) in &mut profile.affinities {
                *affinity *= 0.5;
            }
            profile
                .affinities
                .retain(|(_, affinity)| *affinity > CHARGE_EPSILON);
        }
        let resident_ids: Vec<u32> = self.resident_metadata.keys().copied().collect();
        for vertex_id in resident_ids {
            self.refresh_resident_accounting(vertex_id);
        }
        self.recompute_quotas();
        self.prune_ghost_profiles();
    }

    fn prune_ghost_profiles(&mut self) {
        let max_profiles = self
            .settings
            .capacity
            .saturating_mul(self.settings.ghost_capacity_multiplier)
            .max(self.settings.capacity);
        while self.profiles.len() > max_profiles {
            let Some((vertex_id, observed_tick)) = self.ghost_order.pop_front() else {
                break;
            };
            if self.store.contains_key(&vertex_id) {
                continue;
            }
            if self
                .profiles
                .get(&vertex_id)
                .is_some_and(|profile| profile.last_observed_tick == observed_tick)
            {
                self.profiles.remove(&vertex_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use diskann::graph::AdjacencyList;
    use tempfile::tempdir;

    use super::*;
    use crate::test_utils::GraphDataF32VectorUnitData;

    fn settings(capacity: usize) -> QueryAffinityCacheSettings {
        QueryAffinityCacheSettings {
            capacity,
            prototype_file: "unused.fbin".to_string(),
            metric: Metric::L2,
            top_m: 2,
            global_fraction: 0.25,
            global_entropy_threshold: 0.45,
            global_min_observations: 4.0,
            min_admission_observations: 2.0,
            max_affinity_clusters: 4,
            decay_interval: 100,
            quota_update_interval: 2,
            ghost_capacity_multiplier: 2,
            admission_margin: 1.0,
        }
    }

    fn prototypes() -> QueryPrototypes {
        QueryPrototypes::new(2, vec![0.0, 0.0, 10.0, 0.0]).unwrap()
    }

    fn node(value: f32) -> CachedNode<GraphDataF32VectorUnitData> {
        CachedNode {
            vector: vec![value, value],
            adjacency_list: AdjacencyList::from_iter_untrusted([1, 2]),
            associated_data: (),
        }
    }

    #[test]
    fn router_returns_deterministic_soft_top_two() {
        let route = prototypes().route(&[5.0, 0.0], Metric::L2, 2).unwrap();
        assert_eq!(route.weights.len(), 2);
        assert_eq!(route.weights[0].0, 0);
        assert_eq!(route.weights[1].0, 1);
        assert!((route.weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() < 1e-9);
        assert!((route.weights[0].1 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn prototype_fbin_loader_validates_and_routes() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("prototypes.fbin");
        let mut file = File::create(&path).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        for value in [0.0f32, 0.0, 10.0, 0.0] {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        drop(file);

        let loaded = QueryPrototypes::from_fbin(path.to_str().unwrap(), 2).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(
            loaded.route(&[9.0, 0.0], Metric::L2, 1).unwrap(),
            QueryRoute::hard(1)
        );
        assert!(QueryPrototypes::from_fbin(path.to_str().unwrap(), 3).is_err());
    }

    #[test]
    fn router_supports_cosine_and_inner_product() {
        let prototypes = QueryPrototypes::new(2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(
            prototypes.route(&[1.0, 0.0], Metric::Cosine, 1).unwrap(),
            QueryRoute::hard(0)
        );
        assert_eq!(
            prototypes
                .route(&[0.0, 2.0], Metric::InnerProduct, 1)
                .unwrap(),
            QueryRoute::hard(1)
        );
        assert!(prototypes.route(&[0.0, 0.0], Metric::Cosine, 1).is_err());
    }

    #[test]
    fn cache_stores_one_payload_for_multi_cluster_node() {
        let mut cache =
            QueryAffinityCache::<GraphDataF32VectorUnitData>::new(2, settings(4), prototypes())
                .unwrap();
        let route0 = QueryRoute::hard(0);
        let route1 = QueryRoute::hard(1);
        assert!(cache.lookup(7, &route0).is_none());
        cache.admit_node(7, node(7.0), &route0).unwrap();
        assert!(cache.lookup(7, &route1).is_some());
        assert_eq!(cache.len(), 1);
        let charges = cache.charges(7).unwrap();
        assert_eq!(charges.len(), 2);
        assert!((charges.iter().map(|(_, charge)| charge).sum::<f64>() - 1.0).abs() < 1e-9);
        cache.validate_invariants().unwrap();
    }

    #[test]
    fn broadly_reused_node_is_promoted_to_global() {
        let mut cache =
            QueryAffinityCache::<GraphDataF32VectorUnitData>::new(2, settings(4), prototypes())
                .unwrap();
        let route0 = QueryRoute::hard(0);
        let route1 = QueryRoute::hard(1);
        cache.lookup(1, &route0);
        cache.admit_node(1, node(1.0), &route0).unwrap();
        cache.lookup(1, &route1);
        cache.lookup(1, &route0);
        cache.lookup(1, &route1);
        assert_eq!(cache.role(1), Some(QueryAffinityRole::Global));
        assert_eq!(cache.global_occupancy(), 1);
        assert!(cache.charges(1).unwrap().is_empty());
        assert_eq!(cache.stats().role_promotions, 1);
        cache.validate_invariants().unwrap();
    }

    #[test]
    fn query_mass_changes_elastic_quotas() {
        let mut cache =
            QueryAffinityCache::<GraphDataF32VectorUnitData>::new(2, settings(8), prototypes())
                .unwrap();
        cache.begin_query(&[0.0, 0.0]).unwrap();
        cache.begin_query(&[0.0, 0.0]).unwrap();
        assert!(cache.cluster_quotas()[0] > cache.cluster_quotas()[1]);
        let quota_sum: f64 = cache.cluster_quotas().iter().sum();
        assert!((quota_sum - cache.conditional_budget()).abs() < 1e-9);
    }

    #[test]
    fn affinity_slots_and_decay_are_bounded() {
        let prototypes = QueryPrototypes::new(2, vec![0.0, 0.0, 10.0, 0.0, 20.0, 0.0]).unwrap();
        let mut config = settings(4);
        config.top_m = 1;
        config.max_affinity_clusters = 2;
        config.decay_interval = 4;
        let mut cache =
            QueryAffinityCache::<GraphDataF32VectorUnitData>::new(2, config, prototypes).unwrap();
        cache.lookup(9, &QueryRoute::hard(0));
        cache.lookup(9, &QueryRoute::hard(1));
        cache.lookup(9, &QueryRoute::hard(2));
        assert_eq!(cache.profiles[&9].affinities.len(), 2);
        let observations_before_decay = cache.profiles[&9].total_observations;
        cache.lookup(10, &QueryRoute::hard(0));
        assert!(cache.profiles[&9].total_observations < observations_before_decay);
    }

    #[test]
    fn unused_capacity_can_be_borrowed_by_hot_cluster() {
        let mut config = settings(4);
        config.global_fraction = 0.25;
        let mut cache =
            QueryAffinityCache::<GraphDataF32VectorUnitData>::new(2, config, prototypes()).unwrap();
        let route = QueryRoute::hard(0);
        for vertex_id in 1..=3 {
            cache.lookup(vertex_id, &route);
            cache
                .admit_node(vertex_id, node(vertex_id as f32), &route)
                .unwrap();
        }
        assert_eq!(cache.len(), 3);
        assert!(cache.cluster_usage()[0] > cache.cluster_quotas()[0]);
        assert!(cache.stats().borrowed_admissions > 0);
        cache.validate_invariants().unwrap();
    }

    #[test]
    fn admission_reclaims_from_overquota_cluster() {
        let mut config = settings(2);
        config.global_fraction = 0.0;
        config.min_admission_observations = 1.0;
        let mut cache =
            QueryAffinityCache::<GraphDataF32VectorUnitData>::new(2, config, prototypes()).unwrap();
        let route0 = QueryRoute::hard(0);
        let route1 = QueryRoute::hard(1);
        for vertex_id in 1..=2 {
            cache.lookup(vertex_id, &route0);
            cache
                .admit_node(vertex_id, node(vertex_id as f32), &route0)
                .unwrap();
        }
        for _ in 0..3 {
            cache.lookup(3, &route1);
        }
        cache.admit_node(3, node(3.0), &route1).unwrap();
        assert!(cache.contains(3));
        assert_eq!(cache.len(), 2);
        assert!(cache.cluster_usage()[1] > 0.9);
        assert!(cache.cluster_usage()[0] < 1.1);
        cache.validate_invariants().unwrap();
    }

    #[test]
    fn lower_utility_candidate_is_rejected() {
        let mut config = settings(1);
        config.global_fraction = 0.0;
        config.min_admission_observations = 1.0;
        let mut cache =
            QueryAffinityCache::<GraphDataF32VectorUnitData>::new(2, config, prototypes()).unwrap();
        let route = QueryRoute::hard(0);
        for _ in 0..5 {
            cache.lookup(1, &route);
        }
        cache.admit_node(1, node(1.0), &route).unwrap();
        cache.lookup(2, &route);
        cache.admit_node(2, node(2.0), &route).unwrap();
        assert!(cache.contains(1));
        assert!(!cache.contains(2));
        assert_eq!(cache.stats().rejections, 1);
    }

    #[test]
    fn full_cache_rejects_unobserved_candidate() {
        let mut config = settings(1);
        config.min_admission_observations = 2.0;
        let mut cache =
            QueryAffinityCache::<GraphDataF32VectorUnitData>::new(2, config, prototypes()).unwrap();
        let route = QueryRoute::hard(0);
        cache.lookup(1, &route);
        cache.admit_node(1, node(1.0), &route).unwrap();
        cache.lookup(2, &route);
        cache.admit_node(2, node(2.0), &route).unwrap();
        assert!(cache.contains(1));
        assert!(!cache.contains(2));
        assert_eq!(cache.stats().rejections, 1);
    }

    #[test]
    fn higher_affinity_candidate_replaces_cluster_victim() {
        let mut config = settings(2);
        config.min_admission_observations = 1.0;
        let mut cache =
            QueryAffinityCache::<GraphDataF32VectorUnitData>::new(2, config, prototypes()).unwrap();
        let route0 = QueryRoute::hard(0);
        let route1 = QueryRoute::hard(1);
        cache.lookup(1, &route0);
        cache.admit_node(1, node(1.0), &route0).unwrap();
        cache.lookup(2, &route1);
        cache.admit_node(2, node(2.0), &route1).unwrap();
        for _ in 0..4 {
            cache.lookup(3, &route0);
        }
        cache.admit_node(3, node(3.0), &route0).unwrap();
        assert!(cache.contains(3));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
        cache.validate_invariants().unwrap();
    }

    #[test]
    fn ghost_profiles_are_bounded_without_removing_residents() {
        let mut cache =
            QueryAffinityCache::<GraphDataF32VectorUnitData>::new(2, settings(2), prototypes())
                .unwrap();
        let route = QueryRoute::hard(0);
        cache.lookup(1, &route);
        cache.admit_node(1, node(1.0), &route).unwrap();
        for vertex_id in 2..30 {
            cache.lookup(vertex_id, &route);
        }
        assert!(cache.contains(1));
        assert!(cache.profile_count() <= 4);
        cache.validate_invariants().unwrap();
    }
}
