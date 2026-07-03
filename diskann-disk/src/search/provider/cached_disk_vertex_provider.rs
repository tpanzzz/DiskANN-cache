/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::{Arc, Mutex, OnceLock},
};

use crate::data_model::GraphDataType;
use diskann::{graph::AdjacencyList, ANNError, ANNResult};

use crate::utils::aligned_file_reader::traits::AlignedFileReader;
use hashbrown::HashMap;

use crate::{
    data_model::{Cache, CachedNode, DynamicNodeCache, GraphHeader, ShardedDynamicNodeCache},
    search::{provider::disk_vertex_provider::DiskVertexProvider, traits::VertexProvider},
};

static CACHE_TRACE_WRITER: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();

fn trace_cache_access(vertex_id: u32, cache_hit: bool, cache_source: &str) {
    let writer = CACHE_TRACE_WRITER.get_or_init(|| {
        std::env::var_os("DISKANN_CACHE_TRACE_PATH")
            .and_then(|path| File::create(path).ok())
            .map(|file| Mutex::new(BufWriter::new(file)))
    });

    let Some(writer) = writer.as_ref() else {
        return;
    };
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    let _ = writeln!(
        writer,
        "{{\"stage\":\"load_vertices\",\"vertex_id\":{},\"cache_hit\":{},\"cache_source\":\"{}\"}}",
        vertex_id, cache_hit, cache_source
    );
}

pub enum SharedNodeCache<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    Static(Arc<Cache<Data>>),
    Dynamic(Arc<Mutex<DynamicNodeCache<Data>>>),
    ShardedDynamic(Arc<ShardedDynamicNodeCache<Data>>),
    StaticAndShardedDynamic {
        static_cache: Arc<Cache<Data>>,
        dynamic_cache: Arc<ShardedDynamicNodeCache<Data>>,
    },
}

impl<Data> Clone for SharedNodeCache<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    fn clone(&self) -> Self {
        match self {
            Self::Static(cache) => Self::Static(cache.clone()),
            Self::Dynamic(cache) => Self::Dynamic(cache.clone()),
            Self::ShardedDynamic(cache) => Self::ShardedDynamic(cache.clone()),
            Self::StaticAndShardedDynamic {
                static_cache,
                dynamic_cache,
            } => Self::StaticAndShardedDynamic {
                static_cache: static_cache.clone(),
                dynamic_cache: dynamic_cache.clone(),
            },
        }
    }
}

impl<Data> SharedNodeCache<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    pub fn empty() -> ANNResult<Self> {
        Ok(Self::Static(Arc::new(Cache::new(0, 0)?)))
    }

    pub fn static_cache(cache: Cache<Data>) -> Self {
        Self::Static(Arc::new(cache))
    }

    pub fn dynamic_cache(cache: DynamicNodeCache<Data>) -> Self {
        Self::Dynamic(Arc::new(Mutex::new(cache)))
    }

    pub fn sharded_dynamic_cache(cache: ShardedDynamicNodeCache<Data>) -> Self {
        Self::ShardedDynamic(Arc::new(cache))
    }

    pub fn static_and_sharded_dynamic_cache(
        static_cache: Cache<Data>,
        dynamic_cache: ShardedDynamicNodeCache<Data>,
    ) -> Self {
        Self::StaticAndShardedDynamic {
            static_cache: Arc::new(static_cache),
            dynamic_cache: Arc::new(dynamic_cache),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Static(cache) => cache.len(),
            Self::Dynamic(cache) => match cache.lock() {
                Ok(cache) => cache.len(),
                Err(_) => 0,
            },
            Self::ShardedDynamic(cache) => cache.len(),
            Self::StaticAndShardedDynamic {
                static_cache,
                dynamic_cache,
            } => static_cache.len() + dynamic_cache.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get_static_vector(&self, vertex_id: &Data::VectorIdType) -> Option<&[Data::VectorDataType]> {
        match self {
            Self::Static(cache) => cache.get_vector(vertex_id),
            Self::Dynamic(_) => None,
            Self::ShardedDynamic(_) => None,
            Self::StaticAndShardedDynamic { static_cache, .. } => {
                static_cache.get_vector(vertex_id)
            }
        }
    }

    fn get_static_adjacency_list(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> Option<&AdjacencyList<Data::VectorIdType>> {
        match self {
            Self::Static(cache) => cache.get_adjacency_list(vertex_id),
            Self::Dynamic(_) => None,
            Self::ShardedDynamic(_) => None,
            Self::StaticAndShardedDynamic { static_cache, .. } => {
                static_cache.get_adjacency_list(vertex_id)
            }
        }
    }

    fn get_static_associated_data(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> Option<&Data::AssociatedDataType> {
        match self {
            Self::Static(cache) => cache.get_associated_data(vertex_id),
            Self::Dynamic(_) => None,
            Self::ShardedDynamic(_) => None,
            Self::StaticAndShardedDynamic { static_cache, .. } => {
                static_cache.get_associated_data(vertex_id)
            }
        }
    }

    fn contains_static(&self, vertex_id: &Data::VectorIdType) -> bool {
        match self {
            Self::Static(cache) => cache.contains(vertex_id),
            Self::Dynamic(_) => false,
            Self::ShardedDynamic(_) => false,
            Self::StaticAndShardedDynamic { static_cache, .. } => static_cache.contains(vertex_id),
        }
    }

    fn lookup_dynamic(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<Option<Arc<CachedNode<Data>>>> {
        match self {
            Self::Static(_) => Ok(None),
            Self::Dynamic(cache) => {
                let mut cache = cache.lock().map_err(|_| {
                    ANNError::log_index_error("Dynamic node cache lock is poisoned")
                })?;
                Ok(cache.lookup(vertex_id))
            }
            Self::ShardedDynamic(cache) => cache.lookup(vertex_id),
            Self::StaticAndShardedDynamic { dynamic_cache, .. } => dynamic_cache.lookup(vertex_id),
        }
    }

    fn admit_dynamic(
        &self,
        vertex_id: &Data::VectorIdType,
        node: CachedNode<Data>,
    ) -> ANNResult<()> {
        match self {
            Self::Static(_) => Ok(()),
            Self::Dynamic(cache) => {
                let mut cache = cache.lock().map_err(|_| {
                    ANNError::log_index_error("Dynamic node cache lock is poisoned")
                })?;
                cache.admit_node(vertex_id, node)
            }
            Self::ShardedDynamic(cache) => cache.admit_node(vertex_id, node),
            Self::StaticAndShardedDynamic { dynamic_cache, .. } => {
                dynamic_cache.admit_node(vertex_id, node)
            }
        }
    }
}

pub struct CachedDiskVertexProvider<Data, AlignedReaderType>
where
    Data: GraphDataType<VectorIdType = u32>,
    AlignedReaderType: AlignedFileReader,
{
    // Global shared cache. Static caches are immutable; dynamic caches use an internal lock.
    cache: SharedNodeCache<Data>,

    // The underlying disk vertex provider to read from disk.
    vector_provider: DiskVertexProvider<Data, AlignedReaderType>,

    // Dynamic cache hits are held here for the current load batch so accessors can return
    // references without holding a global cache lock.
    cached_nodes_for_current_read: HashMap<Data::VectorIdType, Arc<CachedNode<Data>>>,

    // Maintains the mapping of local index of the uncached vertices to index in filtered list
    // after removing cached nodes.
    nodes_to_fetch_local_idx_to_filtered_idx: HashMap<usize, usize>,

    // The number of vertices loaded by this provider.
    vertices_loaded_count: u32,
}

impl<Data, AlignedReaderType> VertexProvider<Data>
    for CachedDiskVertexProvider<Data, AlignedReaderType>
where
    Data: GraphDataType<VectorIdType = u32>,
    AlignedReaderType: AlignedFileReader,
{
    fn get_vector(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&[<Data as GraphDataType>::VectorDataType]> {
        if let Some(node) = self.cached_nodes_for_current_read.get(vertex_id) {
            return Ok(&node.vector);
        }
        match self.cache.get_static_vector(vertex_id) {
            Some(vector) => Ok(vector),
            None => self.vector_provider.get_vector(vertex_id),
        }
    }

    fn get_adjacency_list(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&[Data::VectorIdType]> {
        if let Some(node) = self.cached_nodes_for_current_read.get(vertex_id) {
            return Ok(&node.adjacency_list);
        }
        match self.cache.get_static_adjacency_list(vertex_id) {
            Some(adj_list) => Ok(adj_list),
            None => self.vector_provider.get_adjacency_list(vertex_id),
        }
    }

    fn get_associated_data(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&Data::AssociatedDataType> {
        if let Some(node) = self.cached_nodes_for_current_read.get(vertex_id) {
            return Ok(&node.associated_data);
        }
        match self.cache.get_static_associated_data(vertex_id) {
            Some(associated_data) => Ok(associated_data),
            None => self.vector_provider.get_associated_data(vertex_id),
        }
    }

    fn load_vertices(&mut self, vertex_ids: &[Data::VectorIdType]) -> ANNResult<()> {
        self.clear_before_next_read();

        self.vertices_loaded_count += vertex_ids.len() as u32;

        let uncached_vertex_ids = self.filter_cached_nodes(vertex_ids)?;
        if uncached_vertex_ids.is_empty() {
            return ANNResult::Ok(());
        }

        self.vector_provider.load_vertices(&uncached_vertex_ids)
    }

    fn process_loaded_node(
        &mut self,
        vertex_id: &Data::VectorIdType,
        mut idx: usize,
    ) -> Result<(), ANNError> {
        if self.cache.contains_static(vertex_id)
            || self.cached_nodes_for_current_read.contains_key(vertex_id)
        {
            return Ok(());
        }
        idx = self.nodes_to_fetch_local_idx_to_filtered_idx[&idx];
        self.vector_provider.process_loaded_node(vertex_id, idx)?;

        let node = CachedNode {
            vector: self.vector_provider.get_vector(vertex_id)?.to_vec(),
            adjacency_list: AdjacencyList::from_iter_untrusted(
                self.vector_provider
                    .get_adjacency_list(vertex_id)?
                    .iter()
                    .copied(),
            ),
            associated_data: *self.vector_provider.get_associated_data(vertex_id)?,
        };
        self.cache.admit_dynamic(vertex_id, node)?;
        Ok(())
    }

    fn io_operations(&self) -> u32 {
        self.vector_provider.io_operations()
    }

    fn clear(&mut self) {
        self.clear_before_next_read();
        self.vector_provider.clear();
        self.vertices_loaded_count = 0;
    }

    fn vertices_loaded_count(&self) -> u32 {
        self.vertices_loaded_count
    }
}

impl<Data, AlignedReaderType> CachedDiskVertexProvider<Data, AlignedReaderType>
where
    Data: GraphDataType<VectorIdType = u32>,
    AlignedReaderType: AlignedFileReader,
{
    /// Create new CachedDiskVertexProvider instance.
    pub fn new(
        header: &GraphHeader,
        max_batch_size: usize,
        sector_reader: AlignedReaderType,
        cache: SharedNodeCache<Data>,
    ) -> ANNResult<Self> {
        let vector_provider = DiskVertexProvider::new(header, max_batch_size, sector_reader)?;
        Ok(CachedDiskVertexProvider {
            cache,
            vector_provider,
            cached_nodes_for_current_read: HashMap::new(),
            nodes_to_fetch_local_idx_to_filtered_idx: HashMap::new(),
            vertices_loaded_count: 0,
        })
    }

    // Filter cached nodes from the list of vertex ids leaving uncached nodes.
    fn filter_cached_nodes(
        &mut self,
        vertex_ids: &[Data::VectorIdType],
    ) -> ANNResult<Vec<Data::VectorIdType>> {
        let mut uncached_vertex_ids = Vec::new();

        for (idx, vertex_id) in vertex_ids.iter().enumerate() {
            if self.cache.contains_static(vertex_id) {
                trace_cache_access(*vertex_id, true, "static");
                continue;
            }

            if let Some(node) = self.cache.lookup_dynamic(vertex_id)? {
                trace_cache_access(*vertex_id, true, "dynamic");
                self.cached_nodes_for_current_read.insert(*vertex_id, node);
                continue;
            }

            trace_cache_access(*vertex_id, false, "disk");
            self.nodes_to_fetch_local_idx_to_filtered_idx
                .insert(idx, uncached_vertex_ids.len());
            uncached_vertex_ids.push(*vertex_id);
        }

        Ok(uncached_vertex_ids)
    }

    fn clear_before_next_read(&mut self) {
        self.cached_nodes_for_current_read.clear();
        self.nodes_to_fetch_local_idx_to_filtered_idx.clear();
    }
}
