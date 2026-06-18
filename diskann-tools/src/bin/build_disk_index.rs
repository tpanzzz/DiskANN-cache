/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use clap::Parser;
use diskann_disk::QuantizationType;
use diskann_providers::storage::FileStorageProvider;
use diskann_tools::utils::{
    build_disk_index, get_num_threads, init_subscriber, BuildDiskIndexParameters, CMDResult,
    DataType, DimensionValues, GraphDataF32Vector, GraphDataHalfVector, GraphDataInt8Vector,
    GraphDataU8Vector,
};
use diskann_vector::distance::Metric;

fn main() -> CMDResult<()> {
    init_subscriber();
    let args = Args::parse();
    let storage_provider = FileStorageProvider;
    let threads = get_num_threads(args.num_threads);

    let dim_values = DimensionValues::new(args.dimensions, args.full_precision_dimensions);
    let parameters = BuildDiskIndexParameters {
        metric: args.dist_fn,
        data_path: &args.data_path,
        r: args.max_degree,
        l: args.build_search_list_size,
        index_path_prefix: &args.index_path_prefix,
        num_threads: threads,
        num_of_pq_chunks: args.pq_chunks,
        index_build_ram_limit_gb: args.index_build_ram_limit_gb,
        build_quantization_type: args.build_quantization_type,
        chunking_parameters: None,
        dim_values,
    };

    match args.data_type {
        DataType::Float => build_disk_index::<GraphDataF32Vector, _>(&storage_provider, parameters),
        DataType::Fp16 => build_disk_index::<GraphDataHalfVector, _>(&storage_provider, parameters),
        DataType::Uint8 => build_disk_index::<GraphDataU8Vector, _>(&storage_provider, parameters),
        DataType::Int8 => build_disk_index::<GraphDataInt8Vector, _>(&storage_provider, parameters),
    }?;

    Ok(())
}

#[derive(Debug, Parser)]
#[command(
    name = "build_disk_index",
    about = "Build a DiskANN disk index from a DiskANN binary vector file"
)]
struct Args {
    /// data type <float/int8/uint8/fp16>
    #[arg(long = "data_type", default_value = "float")]
    data_type: DataType,

    /// Distance function to use.
    #[arg(long = "dist_fn", default_value = "l2")]
    dist_fn: Metric,

    /// Input vector file in DiskANN binary format.
    #[arg(long = "data_path", short = 'd')]
    data_path: String,

    /// Output index path prefix.
    #[arg(long = "index_path_prefix", short = 'i')]
    index_path_prefix: String,

    /// Stored vector dimensionality.
    #[arg(long = "dimensions")]
    dimensions: usize,

    /// Full-precision dimensionality. Usually the same as --dimensions.
    #[arg(long = "full_precision_dimensions")]
    full_precision_dimensions: usize,

    /// Graph max degree R.
    #[arg(long = "max_degree", short = 'R', default_value_t = 64)]
    max_degree: u32,

    /// Build search list size L.
    #[arg(long = "build_search_list_size", short = 'L', default_value_t = 100)]
    build_search_list_size: u32,

    /// Number of PQ chunks used by the disk index build parameters.
    #[arg(long = "pq_chunks", default_value_t = 16)]
    pq_chunks: usize,

    /// Build RAM limit in GiB.
    #[arg(long = "index_build_ram_limit_gb", default_value_t = 8.0)]
    index_build_ram_limit_gb: f64,

    /// Build quantization type, for example FP, PQ_16, SQ_8, or SQ_8_2.0.
    #[arg(long = "build_quantization_type", default_value = "FP")]
    build_quantization_type: QuantizationType,

    /// Number of build threads. Defaults to all logical CPUs.
    #[arg(long = "num_threads", short = 'T')]
    num_threads: Option<usize>,
}
