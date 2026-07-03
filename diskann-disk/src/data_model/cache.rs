/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::data_model::GraphDataType;
use diskann::{graph::AdjacencyList, ANNError, ANNResult};
use hashbrown::{hash_map::Entry::Occupied, HashMap};

#[derive(Clone)]
pub struct CachedNode<Data: GraphDataType<VectorIdType = u32>> {
    pub vector: Vec<Data::VectorDataType>,
    pub adjacency_list: AdjacencyList<Data::VectorIdType>,
    pub associated_data: Data::AssociatedDataType,
}

pub struct Cache<Data: GraphDataType<VectorIdType = u32>> {
    // Maintains the mapping of vector_id to index in the global cached nodes list.
    mapping: HashMap<Data::VectorIdType, usize>,

    // Maintains vector ids in cache slot order, allowing removals to keep storage dense.
    ids: Vec<Data::VectorIdType>,

    // Flat buffer holding `capacity * dimension` vector elements, laid out row-major.
    vectors: Vec<Data::VectorDataType>,

    // The cached adjacency lists.
    adjacency_lists: Vec<AdjacencyList<Data::VectorIdType>>,

    // The cached associated data list.
    associated_data: Vec<Data::AssociatedDataType>,

    // The dimension of the vectors in the cache.
    dimension: usize,

    // The capacity of the cache.
    capacity: usize,
}

impl<Data> Cache<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    // Creates a new cache with the specified dimension and capacity.
    pub fn new(dimension: usize, capacity: usize) -> ANNResult<Self> {
        Ok(Self {
            mapping: HashMap::new(),
            ids: Vec::with_capacity(capacity),
            vectors: vec![Data::VectorDataType::default(); capacity * dimension],
            adjacency_lists: Vec::with_capacity(capacity),
            associated_data: Vec::with_capacity(capacity),
            dimension,
            capacity,
        })
    }

    // Returns `true` if the cache contains the `vector_id`, otherwise `false`.
    pub fn contains(&self, vector_id: &Data::VectorIdType) -> bool {
        self.mapping.contains_key(vector_id)
    }

    // Returns the vector associated with the `vector_id`, if it exists in the cache otherwise `Option::None`.
    pub fn get_vector(&self, vector_id: &Data::VectorIdType) -> Option<&[Data::VectorDataType]> {
        if let Some(idx) = self.mapping.get(vector_id) {
            Some(&self.vectors[idx * self.dimension..(idx + 1) * self.dimension])
        } else {
            Option::None
        }
    }

    pub fn get_node(&self, vector_id: &Data::VectorIdType) -> Option<CachedNode<Data>> {
        self.mapping.get(vector_id).map(|idx| CachedNode {
            vector: self.vectors[idx * self.dimension..(idx + 1) * self.dimension].to_vec(),
            adjacency_list: self.adjacency_lists[*idx].clone(),
            associated_data: self.associated_data[*idx],
        })
    }

    // Returns the adjacency list associated with the `vector_id``, if it exists in the cache otherwise `Option::None`.
    pub fn get_adjacency_list(
        &self,
        vector_id: &Data::VectorIdType,
    ) -> Option<&AdjacencyList<Data::VectorIdType>> {
        if let Some(idx) = self.mapping.get(vector_id) {
            Some(&self.adjacency_lists[*idx])
        } else {
            Option::None
        }
    }

    // Returns the associated data associated with the `vector_id`, if it exists in the cache otherwise `Option::None`.
    pub fn get_associated_data(
        &self,
        vector_id: &Data::VectorIdType,
    ) -> Option<&Data::AssociatedDataType> {
        if let Some(idx) = self.mapping.get(vector_id) {
            Some(&self.associated_data[*idx])
        } else {
            Option::None
        }
    }

    // Inserts a new node in the cache, if the node already exists in the cache, it updates the node.
    // If the cache is full, it returns an error.
    pub fn insert(
        &mut self,
        vector_id: &Data::VectorIdType,
        vector: &[Data::VectorDataType],
        adjacency_list: AdjacencyList<Data::VectorIdType>,
        associated_data: Data::AssociatedDataType,
    ) -> ANNResult<()> {
        if self.dimension != vector.len() {
            return ANNResult::Err(ANNError::log_index_error(
                "Vector dimension does not match the dimension set in cache.",
            ));
        }

        if let Occupied(occupied_entry) = self.mapping.entry(*vector_id) {
            let idx = *occupied_entry.get();
            self.copy_to_cache(idx, vector, adjacency_list, associated_data);
            return ANNResult::Ok(());
        }

        if self.mapping.len() >= self.capacity {
            return ANNResult::Err(ANNError::log_index_error(
                "Cache is full, cannot insert more nodes",
            ));
        }

        let idx = self.mapping.len();
        self.mapping.insert(*vector_id, idx);
        self.ids.push(*vector_id);
        self.copy_to_cache(idx, vector, adjacency_list, associated_data);
        ANNResult::Ok(())
    }

    pub fn remove(&mut self, vector_id: &Data::VectorIdType) -> bool {
        let Some(idx) = self.mapping.remove(vector_id) else {
            return false;
        };

        let last_idx = self.ids.len() - 1;
        let moved_id = self.ids[last_idx];

        if idx != last_idx {
            let src = last_idx * self.dimension;
            let dst = idx * self.dimension;
            self.vectors.copy_within(src..src + self.dimension, dst);
        }

        self.ids.swap_remove(idx);
        self.adjacency_lists.swap_remove(idx);
        self.associated_data.swap_remove(idx);

        if idx != last_idx {
            self.mapping.insert(moved_id, idx);
        }

        true
    }

    // Returns `true` if the cache is empty, otherwise `false`.
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    // Returns the number of nodes in the cache.
    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn ids(&self) -> &[Data::VectorIdType] {
        &self.ids
    }

    fn copy_to_cache(
        &mut self,
        idx: usize,
        vector: &[Data::VectorDataType],
        adjacency_list: AdjacencyList<Data::VectorIdType>,
        associated_data: Data::AssociatedDataType,
    ) {
        self.vectors[idx * self.dimension..(idx + 1) * self.dimension].copy_from_slice(vector);
        if idx < self.adjacency_lists.len() {
            self.adjacency_lists[idx] = adjacency_list;
            self.associated_data[idx] = associated_data;
        } else {
            self.adjacency_lists.push(adjacency_list);
            self.associated_data.push(associated_data);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachingStrategy {
    None,
    StaticCacheWithBfsNodes(usize),
    DynamicNodeCache {
        policy: super::CachePolicyKind,
        capacity: usize,
    },
    DynamicNodeCacheWithBfsWarmup {
        policy: super::CachePolicyKind,
        capacity: usize,
        warmup_nodes: usize,
    },
    ShardedDynamicNodeCache {
        policy: super::CachePolicyKind,
        capacity: usize,
        cache_shards: usize,
        admission: super::CacheAdmissionKind,
    },
    ShardedDynamicNodeCacheWithBfsWarmup {
        policy: super::CachePolicyKind,
        capacity: usize,
        warmup_nodes: usize,
        cache_shards: usize,
        admission: super::CacheAdmissionKind,
    },
    StaticBfsAndShardedDynamicNodeCache {
        policy: super::CachePolicyKind,
        static_nodes: usize,
        dynamic_capacity: usize,
        cache_shards: usize,
        admission: super::CacheAdmissionKind,
    },
}

#[cfg(test)]
mod tests {
    use crate::test_utils::GraphDataF32VectorUnitData;
    use diskann::graph::AdjacencyList;
    use rstest::rstest;

    use crate::data_model::Cache;

    #[rstest]
    fn test_contains() {
        let mut cache =
            Cache::<GraphDataF32VectorUnitData>::new(/*dimention=*/ 10, /*capacity=*/ 2).unwrap();
        insert_a_random_node(&mut cache);
        let vector_id = 1;
        let vector = vec![1.0; 10];
        let adjacency_list = AdjacencyList::from_iter_untrusted([2, 3, 4]);
        cache
            .insert(&vector_id, &vector, adjacency_list, ())
            .unwrap();

        assert!(cache.contains(&vector_id));

        let not_exist_vector_id = 2;
        assert!(!cache.contains(&not_exist_vector_id));
    }

    #[rstest]
    fn test_get_vector() {
        let mut cache =
            Cache::<GraphDataF32VectorUnitData>::new(/*dimention=*/ 10, /*capacity=*/ 2).unwrap();
        insert_a_random_node(&mut cache);
        let vector_id = 1;
        let vector = vec![1.0; 10];
        let adjacency_list = AdjacencyList::from_iter_untrusted([2, 3, 4]);
        cache
            .insert(&vector_id, &vector, adjacency_list, ())
            .unwrap();

        let result = cache.get_vector(&vector_id).unwrap();
        assert_eq!(result, vector.as_slice());

        let not_exist_vector_id = 2;
        assert!(cache.get_vector(&not_exist_vector_id).is_none());
    }

    #[rstest]
    fn test_get_adjacency_list() {
        let mut cache =
            Cache::<GraphDataF32VectorUnitData>::new(/*dimention=*/ 10, /*capacity=*/ 2).unwrap();
        insert_a_random_node(&mut cache);
        let vector_id = 1;
        let vector = vec![1.0; 10];
        let adjacency_list = AdjacencyList::from_iter_untrusted([2, 3, 4]);
        cache
            .insert(&vector_id, &vector, adjacency_list.clone(), ())
            .unwrap();

        let result = cache.get_adjacency_list(&vector_id).unwrap();
        assert_eq!(*result, adjacency_list);

        let not_exist_vector_id = 2;
        assert!(cache.get_adjacency_list(&not_exist_vector_id).is_none());
    }

    #[rstest]
    fn test_get_associated_data() {
        let mut cache =
            Cache::<GraphDataF32VectorUnitData>::new(/*dimention=*/ 10, /*capacity=*/ 2).unwrap();
        insert_a_random_node(&mut cache);
        let vector_id = 1;
        let vector = vec![1.0; 10];
        let adjacency_list = AdjacencyList::from_iter_untrusted([2, 3, 4]);
        let associated_data = ();
        cache
            .insert(&vector_id, &vector, adjacency_list, associated_data)
            .unwrap();

        let result = cache.get_associated_data(&vector_id);
        assert!(result.is_some());

        let not_exist_vector_id = 2;
        assert!(cache.get_associated_data(&not_exist_vector_id).is_none());
    }

    #[rstest]
    fn test_insert() {
        let mut cache =
            Cache::<GraphDataF32VectorUnitData>::new(/*dimention=*/ 10, /*capacity=*/ 2).unwrap();
        insert_a_random_node(&mut cache);
        let vector_id = 1;
        let vector = vec![1.0; 10];
        let adjacency_list = AdjacencyList::from_iter_untrusted([2, 3, 4]);

        // Insert in cache
        cache
            .insert(&vector_id, &vector, adjacency_list.clone(), ())
            .unwrap();
        assert!(cache.contains(&vector_id));

        // Update in cache
        let updated_vector = vec![2.0; 10];
        cache
            .insert(&vector_id, &updated_vector, adjacency_list.clone(), ())
            .unwrap();
        assert_eq!(
            cache.get_vector(&vector_id).unwrap(),
            updated_vector.as_slice()
        );
        assert_eq!(cache.len(), 2);

        // Cache is Full
        let vector_id_2 = 2;
        let result = cache.insert(&vector_id_2, &vector, adjacency_list.clone(), ());
        assert!(result.is_err());

        // Wrong dimention Insert fails.
        let wrong_dimentions_vector = vec![1.0; 11];
        assert!(cache
            .insert(&vector_id, &wrong_dimentions_vector, adjacency_list, ())
            .is_err());
    }

    #[rstest]
    fn test_is_empty() {
        let mut cache =
            Cache::<GraphDataF32VectorUnitData>::new(/*dimention=*/ 10, /*capacity=*/ 1).unwrap();

        assert!(cache.is_empty());

        insert_a_random_node(&mut cache);

        assert!(!cache.is_empty());
    }

    #[rstest]
    fn test_len() {
        let mut cache =
            Cache::<GraphDataF32VectorUnitData>::new(/*dimention=*/ 10, /*capacity=*/ 5).unwrap();

        assert_eq!(cache.len(), 0);

        let vector_id = 1;
        let vector = vec![1.0; 10];
        let adjacency_list = AdjacencyList::from_iter_untrusted([2, 3, 4]);
        cache
            .insert(&vector_id, &vector, adjacency_list.clone(), ())
            .unwrap();
        let vector_id_2 = 2;
        cache
            .insert(&vector_id_2, &vector, adjacency_list.clone(), ())
            .unwrap();
        let vector_id_3 = 3;
        cache
            .insert(&vector_id_3, &vector, adjacency_list, ())
            .unwrap();

        assert_eq!(cache.len(), 3);
    }

    #[rstest]
    fn test_remove_keeps_cache_dense() {
        let mut cache =
            Cache::<GraphDataF32VectorUnitData>::new(/*dimention=*/ 2, /*capacity=*/ 3).unwrap();
        let adjacency_list = AdjacencyList::from_iter_untrusted([10, 11]);

        cache
            .insert(&1, &[1.0, 1.0], adjacency_list.clone(), ())
            .unwrap();
        cache
            .insert(&2, &[2.0, 2.0], adjacency_list.clone(), ())
            .unwrap();
        cache.insert(&3, &[3.0, 3.0], adjacency_list, ()).unwrap();

        assert!(cache.remove(&2));
        assert!(!cache.contains(&2));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get_vector(&1).unwrap(), &[1.0, 1.0]);
        assert_eq!(cache.get_vector(&3).unwrap(), &[3.0, 3.0]);

        assert!(!cache.remove(&99));
    }

    fn insert_a_random_node(cache: &mut Cache<GraphDataF32VectorUnitData>) {
        let vector_id = 99;
        let vector = vec![9.0; 10];
        cache
            .insert(
                &vector_id,
                &vector,
                AdjacencyList::from_iter_untrusted([20, 30, 40]),
                (),
            )
            .unwrap();
    }
}
