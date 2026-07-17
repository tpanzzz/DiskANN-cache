/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};
use diskann_disk::utils::k_means_clustering;
use diskann_providers::utils::create_thread_pool;
use diskann_tools::utils::DataType;
use diskann_utils::io::Metadata;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

#[derive(Debug, Parser)]
#[command(
    name = "query_workload",
    about = "Construct and apply reproducible DiskANN query arrival orders"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate a fixed-seed random query permutation.
    RandomOrder {
        /// Query vector file in DiskANN binary matrix format.
        #[arg(long = "query_file")]
        query_file: PathBuf,

        /// Query vector data type.
        #[arg(long = "data_type", default_value = "float")]
        data_type: DataType,

        /// Random seed used by Fisher-Yates shuffle.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Output one-column CSV order file.
        #[arg(long = "order_output")]
        order_output: PathBuf,
    },

    /// Group f32 queries by k-means region and greedy geometric proximity.
    KmeansOrder {
        /// f32 query vector file in DiskANN binary matrix format.
        #[arg(long = "query_file")]
        query_file: PathBuf,

        /// Number of query clusters.
        #[arg(long, default_value_t = 100)]
        clusters: usize,

        /// Random seed used for k-means initialization.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Maximum number of Lloyd iterations.
        #[arg(long = "max_reps", default_value_t = 20)]
        max_reps: usize,

        /// Rayon thread count used by k-means.
        #[arg(long = "num_threads", default_value_t = 1)]
        num_threads: usize,

        /// Output one-column CSV order file.
        #[arg(long = "order_output")]
        order_output: PathBuf,

        /// Optional query-to-cluster assignment CSV.
        #[arg(long = "assignments_output")]
        assignments_output: Option<PathBuf>,
    },

    /// Apply one permutation to a query matrix and its ground truth.
    ApplyOrder {
        /// Query vector file in DiskANN binary matrix format.
        #[arg(long = "query_file")]
        query_file: PathBuf,

        /// Ground-truth file containing ids and optional distances.
        #[arg(long = "ground_truth_file")]
        ground_truth_file: PathBuf,

        /// Input one-column CSV order file.
        #[arg(long = "order_file")]
        order_file: PathBuf,

        /// Query vector data type.
        #[arg(long = "data_type", default_value = "float")]
        data_type: DataType,

        /// Output reordered query matrix.
        #[arg(long = "query_output")]
        query_output: PathBuf,

        /// Output reordered ground-truth file.
        #[arg(long = "ground_truth_output")]
        ground_truth_output: PathBuf,
    },
}

#[derive(Debug)]
struct MatrixBytes {
    npoints: usize,
    dimensions: usize,
    payload: Vec<u8>,
}

fn element_size(data_type: DataType) -> usize {
    match data_type {
        DataType::Float => size_of::<f32>(),
        DataType::Int8 | DataType::Uint8 => size_of::<u8>(),
        DataType::Fp16 => size_of::<u16>(),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::RandomOrder {
            query_file,
            data_type,
            seed,
            order_output,
        } => random_order(&query_file, data_type, seed, &order_output),
        Command::KmeansOrder {
            query_file,
            clusters,
            seed,
            max_reps,
            num_threads,
            order_output,
            assignments_output,
        } => kmeans_order_command(
            &query_file,
            clusters,
            seed,
            max_reps,
            num_threads,
            &order_output,
            assignments_output.as_deref(),
        ),
        Command::ApplyOrder {
            query_file,
            ground_truth_file,
            order_file,
            data_type,
            query_output,
            ground_truth_output,
        } => apply_order(
            &query_file,
            &ground_truth_file,
            &order_file,
            data_type,
            &query_output,
            &ground_truth_output,
        ),
    }
}

fn read_matrix_bytes(path: &Path, element_size: usize) -> Result<MatrixBytes> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
    );
    let metadata = Metadata::read(&mut reader)
        .with_context(|| format!("failed to read metadata from {}", path.display()))?;
    let (npoints, dimensions) = metadata.into_dims();
    let expected = npoints
        .checked_mul(dimensions)
        .and_then(|count| count.checked_mul(element_size))
        .context("matrix payload size overflow")?;
    let mut payload = Vec::new();
    reader.read_to_end(&mut payload)?;
    ensure!(
        payload.len() == expected,
        "{} has {} payload bytes, expected {} for {}x{} elements of {} bytes",
        path.display(),
        payload.len(),
        expected,
        npoints,
        dimensions,
        element_size
    );
    Ok(MatrixBytes {
        npoints,
        dimensions,
        payload,
    })
}

fn create_output(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Ok(BufWriter::new(File::create(path).with_context(|| {
        format!("failed to create {}", path.display())
    })?))
}

fn write_order(path: &Path, order: &[usize]) -> Result<()> {
    let mut writer = create_output(path)?;
    writeln!(writer, "query_id")?;
    for query_id in order {
        writeln!(writer, "{query_id}")?;
    }
    writer.flush()?;
    Ok(())
}

fn parse_order_csv_text(contents: &str, nqueries: usize) -> Result<Vec<usize>> {
    let mut lines = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    ensure!(
        lines.next() == Some("query_id"),
        "order CSV must start with a query_id header"
    );

    let mut order = Vec::with_capacity(nqueries);
    let mut seen = vec![false; nqueries];
    for (row, value) in lines.enumerate() {
        ensure!(
            !value.contains(','),
            "order CSV row {} must contain exactly one query id",
            row + 2
        );
        let query_id: usize = value
            .parse()
            .with_context(|| format!("invalid query id {value:?} at CSV row {}", row + 2))?;
        ensure!(
            query_id < nqueries,
            "query id {} at CSV row {} is outside [0, {})",
            query_id,
            row + 2,
            nqueries
        );
        ensure!(
            !seen[query_id],
            "duplicate query id {} at CSV row {}",
            query_id,
            row + 2
        );
        seen[query_id] = true;
        order.push(query_id);
    }
    ensure!(
        order.len() == nqueries,
        "order contains {} ids, expected {}",
        order.len(),
        nqueries
    );
    if let Some(missing) = seen.iter().position(|present| !present) {
        bail!("order is missing query id {missing}");
    }
    Ok(order)
}

fn read_order(path: &Path, nqueries: usize) -> Result<Vec<usize>> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    parse_order_csv_text(&contents, nqueries)
}

fn random_permutation(nqueries: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<_> = (0..nqueries).collect();
    order.shuffle(&mut StdRng::seed_from_u64(seed));
    order
}

fn random_order(
    query_file: &Path,
    data_type: DataType,
    seed: u64,
    order_output: &Path,
) -> Result<()> {
    let matrix = read_matrix_bytes(query_file, element_size(data_type))?;
    write_order(order_output, &random_permutation(matrix.npoints, seed))?;
    println!(
        "wrote {} random query ids to {}",
        matrix.npoints,
        order_output.display()
    );
    Ok(())
}

fn decode_f32(payload: &[u8]) -> Vec<f32> {
    payload
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32 chunk")))
        .collect()
}

fn squared_l2(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum()
}

fn greedy_centroid_order(centers: &[f32], dimensions: usize, active: &[bool]) -> Vec<usize> {
    let active_count = active.iter().filter(|active| **active).count();
    let mut order = Vec::with_capacity(active_count);
    let Some(mut current) = active.iter().position(|active| *active) else {
        return order;
    };
    let mut visited = vec![false; active.len()];
    visited[current] = true;
    order.push(current);

    while order.len() < active_count {
        let current_center = &centers[current * dimensions..(current + 1) * dimensions];
        let next = (0..active.len())
            .filter(|candidate| active[*candidate] && !visited[*candidate])
            .map(|candidate| {
                let candidate_center =
                    &centers[candidate * dimensions..(candidate + 1) * dimensions];
                (squared_l2(current_center, candidate_center), candidate)
            })
            .min_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            })
            .expect("an unvisited active cluster must remain")
            .1;
        visited[next] = true;
        order.push(next);
        current = next;
    }
    order
}

fn greedy_query_order(
    data: &[f32],
    dimensions: usize,
    center: &[f32],
    members: &[usize],
) -> Vec<usize> {
    if members.is_empty() {
        return Vec::new();
    }
    let first = *members
        .iter()
        .min_by(|left, right| {
            let left_vector = &data[**left * dimensions..(**left + 1) * dimensions];
            let right_vector = &data[**right * dimensions..(**right + 1) * dimensions];
            squared_l2(left_vector, center)
                .total_cmp(&squared_l2(right_vector, center))
                .then_with(|| left.cmp(right))
        })
        .expect("members is non-empty");

    let mut order = Vec::with_capacity(members.len());
    let mut remaining = members.to_vec();
    remaining.retain(|query_id| *query_id != first);
    order.push(first);

    let mut current = first;
    while !remaining.is_empty() {
        let current_vector = &data[current * dimensions..(current + 1) * dimensions];
        let (position, _) = remaining
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                let left_vector = &data[**left * dimensions..(**left + 1) * dimensions];
                let right_vector = &data[**right * dimensions..(**right + 1) * dimensions];
                squared_l2(current_vector, left_vector)
                    .total_cmp(&squared_l2(current_vector, right_vector))
                    .then_with(|| left.cmp(right))
            })
            .expect("remaining is non-empty");
        current = remaining.swap_remove(position);
        order.push(current);
    }
    order
}

#[allow(clippy::too_many_arguments)]
fn build_kmeans_order(
    data: &[f32],
    nqueries: usize,
    dimensions: usize,
    clusters: usize,
    seed: u64,
    max_reps: usize,
    num_threads: usize,
) -> Result<(Vec<usize>, Vec<u32>)> {
    ensure!(nqueries > 0, "query file must contain at least one vector");
    ensure!(
        (1..=nqueries).contains(&clusters),
        "clusters must be in [1, {nqueries}]"
    );
    ensure!(max_reps > 0, "max_reps must be greater than zero");
    ensure!(num_threads > 0, "num_threads must be greater than zero");

    let pool = create_thread_pool(num_threads)?;
    let mut centers = vec![0.0f32; clusters * dimensions];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cancellation_token = false;
    let (members, assignments, _) = k_means_clustering(
        data,
        nqueries,
        dimensions,
        &mut centers,
        clusters,
        max_reps,
        &mut rng,
        &mut cancellation_token,
        pool.as_ref(),
    )?;

    let active: Vec<_> = members.iter().map(|cluster| !cluster.is_empty()).collect();
    let cluster_order = greedy_centroid_order(&centers, dimensions, &active);
    let mut order = Vec::with_capacity(nqueries);
    for cluster_id in cluster_order {
        let center = &centers[cluster_id * dimensions..(cluster_id + 1) * dimensions];
        order.extend(greedy_query_order(
            data,
            dimensions,
            center,
            &members[cluster_id],
        ));
    }
    ensure!(
        order.len() == nqueries,
        "k-means ordering produced {} ids for {} queries",
        order.len(),
        nqueries
    );
    Ok((order, assignments))
}

fn write_assignments(path: &Path, assignments: &[u32]) -> Result<()> {
    let mut writer = create_output(path)?;
    writeln!(writer, "query_id,cluster_id")?;
    for (query_id, cluster_id) in assignments.iter().enumerate() {
        writeln!(writer, "{query_id},{cluster_id}")?;
    }
    writer.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn kmeans_order_command(
    query_file: &Path,
    clusters: usize,
    seed: u64,
    max_reps: usize,
    num_threads: usize,
    order_output: &Path,
    assignments_output: Option<&Path>,
) -> Result<()> {
    let matrix = read_matrix_bytes(query_file, size_of::<f32>())?;
    let data = decode_f32(&matrix.payload);
    let (order, assignments) = build_kmeans_order(
        &data,
        matrix.npoints,
        matrix.dimensions,
        clusters,
        seed,
        max_reps,
        num_threads,
    )?;
    write_order(order_output, &order)?;
    if let Some(path) = assignments_output {
        write_assignments(path, &assignments)?;
    }
    println!(
        "wrote k-means order for {} queries and {} clusters to {}",
        matrix.npoints,
        clusters,
        order_output.display()
    );
    Ok(())
}

fn write_reordered_matrix(
    output: &Path,
    matrix: &MatrixBytes,
    element_size: usize,
    order: &[usize],
) -> Result<()> {
    let row_size = matrix
        .dimensions
        .checked_mul(element_size)
        .context("matrix row size overflow")?;
    let mut writer = create_output(output)?;
    Metadata::new(matrix.npoints, matrix.dimensions)?.write(&mut writer)?;
    for query_id in order {
        let start = query_id * row_size;
        writer.write_all(&matrix.payload[start..start + row_size])?;
    }
    writer.flush()?;
    Ok(())
}

fn reorder_ground_truth(input: &Path, output: &Path, order: &[usize]) -> Result<()> {
    let mut reader = BufReader::new(
        File::open(input).with_context(|| format!("failed to open {}", input.display()))?,
    );
    let metadata = Metadata::read(&mut reader)?;
    let (npoints, dimensions) = metadata.into_dims();
    ensure!(
        npoints == order.len(),
        "ground truth has {} rows, but order has {}",
        npoints,
        order.len()
    );
    let row_size = dimensions
        .checked_mul(size_of::<u32>())
        .context("ground-truth row size overflow")?;
    let ids_size = npoints
        .checked_mul(row_size)
        .context("ground-truth payload size overflow")?;
    let mut payload = Vec::new();
    reader.read_to_end(&mut payload)?;
    ensure!(
        payload.len() == ids_size || payload.len() == 2 * ids_size,
        "ground truth payload has {} bytes, expected {} ids-only bytes or {} ids+distances bytes",
        payload.len(),
        ids_size,
        2 * ids_size
    );

    let mut writer = create_output(output)?;
    metadata.write(&mut writer)?;
    for query_id in order {
        let start = query_id * row_size;
        writer.write_all(&payload[start..start + row_size])?;
    }
    if payload.len() == 2 * ids_size {
        for query_id in order {
            let start = ids_size + query_id * row_size;
            writer.write_all(&payload[start..start + row_size])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn apply_order(
    query_file: &Path,
    ground_truth_file: &Path,
    order_file: &Path,
    data_type: DataType,
    query_output: &Path,
    ground_truth_output: &Path,
) -> Result<()> {
    let element_size = element_size(data_type);
    let matrix = read_matrix_bytes(query_file, element_size)?;
    let order = read_order(order_file, matrix.npoints)?;
    write_reordered_matrix(query_output, &matrix, element_size, &order)?;
    reorder_ground_truth(ground_truth_file, ground_truth_output, &order)?;
    println!(
        "reordered {} queries into {} and ground truth into {}",
        matrix.npoints,
        query_output.display(),
        ground_truth_output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "diskann_query_workload_{}_{}_{}",
            std::process::id(),
            name,
            id
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_f32_matrix(path: &Path, rows: &[&[f32]]) {
        let mut writer = BufWriter::new(File::create(path).unwrap());
        Metadata::new(rows.len(), rows[0].len())
            .unwrap()
            .write(&mut writer)
            .unwrap();
        for row in rows {
            for value in *row {
                writer.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        writer.flush().unwrap();
    }

    fn write_truthset(path: &Path, ids: &[&[u32]], distances: Option<&[&[f32]]>) {
        let mut writer = BufWriter::new(File::create(path).unwrap());
        Metadata::new(ids.len(), ids[0].len())
            .unwrap()
            .write(&mut writer)
            .unwrap();
        for row in ids {
            for value in *row {
                writer.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        if let Some(distances) = distances {
            for row in distances {
                for value in *row {
                    writer.write_all(&value.to_le_bytes()).unwrap();
                }
            }
        }
        writer.flush().unwrap();
    }

    fn read_u32_payload(path: &Path) -> (usize, usize, Vec<u32>) {
        let matrix = read_matrix_bytes(path, size_of::<u32>()).unwrap();
        let values = matrix
            .payload
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        (matrix.npoints, matrix.dimensions, values)
    }

    #[test]
    fn order_parser_accepts_header_and_rejects_invalid_permutations() {
        assert_eq!(
            parse_order_csv_text("query_id\n2\n0\n1\n", 3).unwrap(),
            vec![2, 0, 1]
        );
        assert!(parse_order_csv_text("query_id\n0\n0\n2\n", 3).is_err());
        assert!(parse_order_csv_text("query_id\n0\n2\n", 3).is_err());
        assert!(parse_order_csv_text("query_id\n0\n1\n3\n", 3).is_err());
    }

    #[test]
    fn random_order_is_reproducible_and_seed_sensitive() {
        assert_eq!(random_permutation(100, 42), random_permutation(100, 42));
        assert_ne!(random_permutation(100, 42), random_permutation(100, 43));
    }

    #[test]
    fn apply_order_reorders_queries_and_truthset_with_distances() {
        let directory = temp_dir("apply_distances");
        let query = directory.join("query.fbin");
        let truth = directory.join("truth.bin");
        let order = directory.join("order.csv");
        let query_output = directory.join("query.out.fbin");
        let truth_output = directory.join("truth.out.bin");
        write_f32_matrix(&query, &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        write_truthset(
            &truth,
            &[&[10, 11], &[20, 21], &[30, 31]],
            Some(&[&[0.1, 0.2], &[0.3, 0.4], &[0.5, 0.6]]),
        );
        fs::write(&order, "query_id\n2\n0\n1\n").unwrap();

        apply_order(
            &query,
            &truth,
            &order,
            DataType::Float,
            &query_output,
            &truth_output,
        )
        .unwrap();

        let reordered = read_matrix_bytes(&query_output, size_of::<f32>()).unwrap();
        assert_eq!(
            decode_f32(&reordered.payload),
            vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]
        );
        let bytes = fs::read(&truth_output).unwrap();
        let ids: Vec<_> = bytes[8..32]
            .chunks_exact(4)
            .map(|value| u32::from_le_bytes(value.try_into().unwrap()))
            .collect();
        let distances: Vec<_> = bytes[32..]
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect();
        assert_eq!(ids, vec![30, 31, 10, 11, 20, 21]);
        assert_eq!(distances, vec![0.5, 0.6, 0.1, 0.2, 0.3, 0.4]);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn apply_order_reorders_ids_only_truthset() {
        let directory = temp_dir("apply_ids");
        let query = directory.join("query.fbin");
        let truth = directory.join("truth.bin");
        let order = directory.join("order.csv");
        let query_output = directory.join("query.out.fbin");
        let truth_output = directory.join("truth.out.bin");
        write_f32_matrix(&query, &[&[1.0], &[2.0], &[3.0]]);
        write_truthset(&truth, &[&[10, 11], &[20, 21], &[30, 31]], None);
        fs::write(&order, "query_id\n1\n2\n0\n").unwrap();

        apply_order(
            &query,
            &truth,
            &order,
            DataType::Float,
            &query_output,
            &truth_output,
        )
        .unwrap();
        let (_, _, ids) = read_u32_payload(&truth_output);
        assert_eq!(ids, vec![20, 21, 30, 31, 10, 11]);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn kmeans_order_keeps_synthetic_clusters_contiguous() {
        let data = vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
        ];
        let (order, assignments) = build_kmeans_order(&data, 6, 2, 2, 42, 20, 1).unwrap();
        let ordered_clusters: Vec<_> = order
            .iter()
            .map(|query_id| assignments[*query_id])
            .collect();
        assert_eq!(
            ordered_clusters
                .windows(2)
                .filter(|pair| pair[0] != pair[1])
                .count(),
            1
        );
        let first_group_is_low = order[0] < 3;
        assert!(order[..3]
            .iter()
            .all(|query_id| (*query_id < 3) == first_group_is_low));
        assert!(order[3..]
            .iter()
            .all(|query_id| (*query_id < 3) != first_group_is_low));
    }
}
