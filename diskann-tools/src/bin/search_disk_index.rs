/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use clap::{Parser, ValueEnum};
use diskann_disk::{
    data_model::{CacheAdmissionKind, CachePolicyKind, CachingStrategy},
    utils::AlignedFileReaderFactory,
};
use diskann_providers::storage::{get_disk_index_file, FileStorageProvider};
use diskann_tools::utils::{
    get_num_threads, init_subscriber, search_disk_index, CMDResult, CMDToolError, DataType,
    GraphDataF32Vector, GraphDataHalfVector, GraphDataInt8Vector, GraphDataU8Vector,
    SearchDiskIndexParameters,
};
use diskann_vector::distance::Metric;

fn main() -> CMDResult<()> {
    init_subscriber();
    let args = Args::parse();
    let storage_provider = FileStorageProvider;
    let threads = get_num_threads(args.num_threads);
    let caching_strategy = cache_strategy_from_args(&args)?;

    let parameters = SearchDiskIndexParameters {
        metric: args.dist_fn,
        index_path_prefix: &args.index_path_prefix,
        result_output_prefix: &args.result_output_prefix,
        query_file: &args.query_file,
        truthset_file: &args.ground_truth_file,
        vector_filters_file: args.vector_filters_file.as_deref(),
        full_precision_vector_file: args.full_precision_vector_file.as_deref(),
        num_threads: threads,
        recall_at: args.recall_at,
        beam_width: args.beam_width,
        search_io_limit: args.search_io_limit,
        l_vec: &args.search_list,
        fail_if_recall_below: args.fail_if_recall_below,
        num_nodes_to_cache: args.num_nodes_to_cache,
        caching_strategy: Some(caching_strategy),
        is_flat_search: args.flat_search,
    };

    let aligned_reader_factory =
        AlignedFileReaderFactory::new(get_disk_index_file(&args.index_path_prefix));

    let exit_code = match args.data_type {
        DataType::Float => search_disk_index::<GraphDataF32Vector, _, _>(
            &storage_provider,
            parameters,
            aligned_reader_factory,
        ),
        DataType::Fp16 => search_disk_index::<GraphDataHalfVector, _, _>(
            &storage_provider,
            parameters,
            aligned_reader_factory,
        ),
        DataType::Uint8 => search_disk_index::<GraphDataU8Vector, _, _>(
            &storage_provider,
            parameters,
            aligned_reader_factory,
        ),
        DataType::Int8 => search_disk_index::<GraphDataInt8Vector, _, _>(
            &storage_provider,
            parameters,
            aligned_reader_factory,
        ),
    }?;

    if exit_code == 0 {
        Ok(())
    } else {
        Err(CMDToolError {
            details: format!("search_disk_index completed with failure status {exit_code}"),
        })
    }
}

fn cache_strategy_from_args(args: &Args) -> CMDResult<CachingStrategy> {
    if args.cache_shards == 0 {
        return Err(CMDToolError {
            details: "--cache_shards must be greater than 0".to_string(),
        });
    }

    let Some(policy) = args.cache_policy else {
        return Ok(if args.num_nodes_to_cache > 0 {
            CachingStrategy::StaticCacheWithBfsNodes(args.num_nodes_to_cache)
        } else {
            CachingStrategy::None
        });
    };

    if policy == CachePolicyKind::BeladyOptimal {
        return Err(CMDToolError {
            details: "belady_opt is only available in cache_trace_replay, not live search"
                .to_string(),
        });
    }

    if policy == CachePolicyKind::NoCache {
        return Ok(CachingStrategy::None);
    }

    let capacity = args.cache_capacity.unwrap_or(args.num_nodes_to_cache);
    if capacity == 0 {
        return Err(CMDToolError {
            details: "--cache_capacity or --num_nodes_to_cache must be greater than 0 for dynamic cache policies".to_string(),
        });
    }

    let use_sharded_backend =
        args.cache_backend == CacheBackendArg::Sharded || args.cache_shards > 1;
    let use_configured_dynamic_backend =
        use_sharded_backend || args.cache_admission != CacheAdmissionKind::None;
    let cache_shards = if use_sharded_backend {
        args.cache_shards
    } else {
        1
    };

    if let Some(static_nodes) = args.cache_static_nodes {
        if static_nodes == 0 {
            return Err(CMDToolError {
                details: "--cache_static_nodes must be greater than 0 when supplied".to_string(),
            });
        }
        return Ok(CachingStrategy::StaticBfsAndShardedDynamicNodeCache {
            policy,
            static_nodes,
            dynamic_capacity: capacity,
            cache_shards,
            admission: args.cache_admission,
        });
    }

    if args.cache_warmup_nodes > 0 {
        if use_configured_dynamic_backend {
            Ok(CachingStrategy::ShardedDynamicNodeCacheWithBfsWarmup {
                policy,
                capacity,
                warmup_nodes: args.cache_warmup_nodes,
                cache_shards,
                admission: args.cache_admission,
            })
        } else {
            Ok(CachingStrategy::DynamicNodeCacheWithBfsWarmup {
                policy,
                capacity,
                warmup_nodes: args.cache_warmup_nodes,
            })
        }
    } else if use_configured_dynamic_backend {
        Ok(CachingStrategy::ShardedDynamicNodeCache {
            policy,
            capacity,
            cache_shards,
            admission: args.cache_admission,
        })
    } else {
        Ok(CachingStrategy::DynamicNodeCache { policy, capacity })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CacheBackendArg {
    Global,
    Sharded,
}

#[derive(Debug, Parser)]
#[command(name = "search_disk_index", about = "Search a DiskANN disk index")]
struct Args {
    /// data type <float/int8/uint8/fp16>
    #[arg(long = "data_type", default_value = "float")]
    data_type: DataType,

    /// Distance function to use.
    #[arg(long = "dist_fn", default_value = "l2")]
    dist_fn: Metric,

    /// Disk index path prefix.
    #[arg(long = "index_path_prefix", short = 'i')]
    index_path_prefix: String,

    /// Result output prefix.
    #[arg(long = "result_output_prefix", short = 'o')]
    result_output_prefix: String,

    /// Query vector file in DiskANN binary format.
    #[arg(long = "query_file", short = 'q')]
    query_file: String,

    /// Ground-truth file. Leave empty to skip recall calculation.
    #[arg(long = "ground_truth_file", default_value = "")]
    ground_truth_file: String,

    /// Optional vector filters file.
    #[arg(long = "vector_filters_file")]
    vector_filters_file: Option<String>,

    /// Optional base-vector matrix used for exact in-memory graph-navigation distances.
    #[arg(long = "full_precision_vector_file")]
    full_precision_vector_file: Option<String>,

    /// Number of search threads. Defaults to all logical CPUs.
    #[arg(long = "num_threads", short = 'T')]
    num_threads: Option<usize>,

    /// Recall K.
    #[arg(long = "recall_at", short = 'K', default_value_t = 10)]
    recall_at: u32,

    /// Beam width.
    #[arg(long = "beam_width", default_value_t = 4)]
    beam_width: u32,

    /// Search I/O limit.
    #[arg(long = "search_io_limit", default_value_t = 1000)]
    search_io_limit: u32,

    /// Search list sizes L.
    #[arg(long = "search_list", value_delimiter = ',', required = true)]
    search_list: Vec<u32>,

    /// Return non-zero when best recall is below this value.
    #[arg(long = "fail_if_recall_below", default_value_t = 0.0)]
    fail_if_recall_below: f32,

    /// Static BFS cache size. Also used as dynamic capacity when --cache_capacity is omitted.
    #[arg(long = "num_nodes_to_cache", default_value_t = 0)]
    num_nodes_to_cache: usize,

    /// Dynamic cache policy. Omit to use the legacy static BFS cache selected by --num_nodes_to_cache.
    #[arg(long = "cache_policy")]
    cache_policy: Option<CachePolicyKind>,

    /// Dynamic cache capacity in node payloads.
    #[arg(long = "cache_capacity")]
    cache_capacity: Option<usize>,

    /// Dynamic cache backend. The default keeps the legacy single global lock.
    #[arg(long = "cache_backend", value_enum, default_value = "global")]
    cache_backend: CacheBackendArg,

    /// Number of shards for sharded dynamic caches.
    #[arg(long = "cache_shards", default_value_t = 1)]
    cache_shards: usize,

    /// Optional admission filter for dynamic cache insertions.
    #[arg(long = "cache_admission", default_value_t = CacheAdmissionKind::None)]
    cache_admission: CacheAdmissionKind,

    /// Immutable medoid-BFS Tier-0 nodes to use before the dynamic cache.
    #[arg(long = "cache_static_nodes")]
    cache_static_nodes: Option<usize>,

    /// Number of medoid-BFS nodes used to warm a dynamic cache.
    #[arg(long = "cache_warmup_nodes", default_value_t = 0)]
    cache_warmup_nodes: usize,

    /// Use flat scan search.
    #[arg(long = "flat_search", default_value_t = false)]
    flat_search: bool,
}
