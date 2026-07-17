#!/usr/bin/env python3
"""Analyze cluster distributions and cross-split vector-pattern similarity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import MiniBatchKMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = REPO_ROOT / "experiments/cache_eval/datasets/catalog.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/cache_eval/results/vector_patterns"


@dataclass
class SplitResult:
    name: str
    count: int
    dimensions: int
    centers: np.ndarray
    labels: np.ndarray
    cluster_counts: np.ndarray
    cluster_fractions: np.ndarray
    point_distances: np.ndarray
    radius_mean: np.ndarray
    radius_p95: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name from catalog.json")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clusters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-init", type=int, default=3)
    return parser.parse_args()


def load_catalog(path: Path, dataset_name: str) -> dict[str, object]:
    catalog = json.loads(path.read_text())
    if dataset_name not in catalog:
        raise KeyError(f"unknown dataset {dataset_name!r}")
    return catalog[dataset_name]


def resolve_repo_path(value: str) -> Path:
    return (REPO_ROOT / value).resolve()


def preprocess(vectors: np.ndarray, mode: str) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if mode == "none":
        return vectors
    if mode == "l2_normalize":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.divide(vectors, norms, out=vectors, where=norms > 0)
        return vectors
    raise ValueError(f"unsupported cluster preprocessing mode {mode!r}")


def read_rows(dataset: h5py.Dataset, indices: np.ndarray, mode: str) -> np.ndarray:
    sorted_order = np.argsort(indices)
    sorted_indices = indices[sorted_order]
    sorted_rows = preprocess(dataset[sorted_indices], mode)
    rows = np.empty_like(sorted_rows)
    rows[sorted_order] = sorted_rows
    return rows


def iter_batches(
    dataset: h5py.Dataset,
    batch_size: int,
    mode: str,
    batch_order: np.ndarray | None = None,
):
    batch_count = math.ceil(len(dataset) / batch_size)
    order = np.arange(batch_count) if batch_order is None else batch_order
    for batch_id in order:
        start = int(batch_id) * batch_size
        end = min(start + batch_size, len(dataset))
        yield start, end, preprocess(dataset[start:end], mode)


def fit_split(
    split_name: str,
    dataset: h5py.Dataset,
    clusters: int,
    seed: int,
    batch_size: int,
    sample_size: int,
    epochs: int,
    n_init: int,
    mode: str,
) -> SplitResult:
    count, dimensions = dataset.shape
    if count < clusters:
        raise ValueError(
            f"split {split_name} has {count} vectors, fewer than {clusters} clusters"
        )

    rng = np.random.default_rng(seed)
    sample_count = min(count, max(clusters * 20, sample_size))
    sample_indices = rng.choice(count, size=sample_count, replace=False)
    sample = read_rows(dataset, sample_indices, mode)

    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        batch_size=max(batch_size, clusters),
        n_init=n_init,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    model.fit(sample)

    batch_count = math.ceil(count / batch_size)
    for _ in range(epochs):
        for _, _, batch in iter_batches(
            dataset,
            batch_size,
            mode,
            batch_order=rng.permutation(batch_count),
        ):
            model.partial_fit(batch)

    labels = np.empty(count, dtype=np.int32)
    distances = np.empty(count, dtype=np.float32)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    for start, end, batch in iter_batches(dataset, batch_size, mode):
        batch_labels = model.predict(batch).astype(np.int32)
        labels[start:end] = batch_labels
        residual = batch - centers[batch_labels]
        distances[start:end] = np.linalg.norm(residual, axis=1)

    cluster_counts = np.bincount(labels, minlength=clusters).astype(np.int64)
    fractions = cluster_counts / count
    radius_mean = np.zeros(clusters, dtype=np.float64)
    radius_p95 = np.zeros(clusters, dtype=np.float64)
    for cluster_id in range(clusters):
        cluster_distances = distances[labels == cluster_id]
        if len(cluster_distances) == 0:
            continue
        radius_mean[cluster_id] = float(cluster_distances.mean())
        radius_p95[cluster_id] = float(np.quantile(cluster_distances, 0.95))

    return SplitResult(
        name=split_name,
        count=count,
        dimensions=dimensions,
        centers=centers,
        labels=labels,
        cluster_counts=cluster_counts,
        cluster_fractions=fractions,
        point_distances=distances,
        radius_mean=radius_mean,
        radius_p95=radius_p95,
    )


def cluster_size_gini(counts: np.ndarray) -> float:
    values = np.sort(np.asarray(counts, dtype=np.float64))
    if values.sum() == 0:
        return 0.0
    n = len(values)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * np.dot(ranks, values) / (n * values.sum())) - (n + 1) / n)


def sequence_metrics(labels: np.ndarray, clusters: int, seed: int) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int32)
    if len(labels) < 2:
        return {}

    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    runs = np.diff(np.concatenate(([0], boundaries, [len(labels)])))
    transition_counts = np.zeros((clusters, clusters), dtype=np.int64)
    np.add.at(transition_counts, (labels[:-1], labels[1:]), 1)
    row_totals = transition_counts.sum(axis=1)
    total_transitions = transition_counts.sum()
    conditional_entropy = 0.0
    for cluster_id in range(clusters):
        if row_totals[cluster_id] == 0:
            continue
        probabilities = transition_counts[cluster_id] / row_totals[cluster_id]
        probabilities = probabilities[probabilities > 0]
        entropy = -float(np.sum(probabilities * np.log2(probabilities)))
        conditional_entropy += (row_totals[cluster_id] / total_transitions) * entropy

    rng = np.random.default_rng(seed)
    lags = [1, 2, 5, 10, 50, 100]
    lag_rows = []
    for lag in lags:
        if lag >= len(labels):
            continue
        observed = float(np.mean(labels[:-lag] == labels[lag:]))
        shuffled_values = []
        for _ in range(10):
            shuffled = rng.permutation(labels)
            shuffled_values.append(float(np.mean(shuffled[:-lag] == shuffled[lag:])))
        shuffled_mean = float(np.mean(shuffled_values))
        lag_rows.append(
            {
                "lag": lag,
                "same_cluster_probability": observed,
                "shuffled_probability": shuffled_mean,
                "locality_lift": observed / shuffled_mean if shuffled_mean > 0 else None,
            }
        )

    return {
        "self_transition_probability": float(np.mean(labels[:-1] == labels[1:])),
        "run_length_mean": float(runs.mean()),
        "run_length_p95": float(np.quantile(runs, 0.95)),
        "run_length_max": int(runs.max()),
        "transition_conditional_entropy_bits": conditional_entropy,
        "transition_entropy_normalized": conditional_entropy / math.log2(clusters)
        if clusters > 1
        else 0.0,
        "lag_same_cluster": lag_rows,
    }


def safe_correlation(function, left: np.ndarray, right: np.ndarray) -> float | None:
    if np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = function(left, right).statistic
    return float(value) if np.isfinite(value) else None


def compare_splits(reference: SplitResult, other: SplitResult) -> tuple[dict[str, object], np.ndarray]:
    distance_matrix = cdist(reference.centers, other.centers, metric="euclidean")
    reference_ids, other_ids = linear_sum_assignment(distance_matrix)
    matched_distances = distance_matrix[reference_ids, other_ids]
    reference_mass = reference.cluster_fractions[reference_ids]
    other_mass = other.cluster_fractions[other_ids]
    pooled_radius = 0.5 * (
        reference.radius_mean[reference_ids] + other.radius_mean[other_ids]
    )
    valid_radius = pooled_radius > 0
    normalized = matched_distances[valid_radius] / pooled_radius[valid_radius]

    epsilon = 1e-12
    js_divergence = float(
        jensenshannon(reference_mass + epsilon, other_mass + epsilon, base=2.0) ** 2
    )
    metrics = {
        "reference_split": reference.name,
        "other_split": other.name,
        "matched_center_distance_mean": float(matched_distances.mean()),
        "matched_center_distance_median": float(np.median(matched_distances)),
        "matched_center_distance_p95": float(np.quantile(matched_distances, 0.95)),
        "matched_center_distance_over_radius_mean": float(normalized.mean())
        if len(normalized)
        else None,
        "matched_cluster_mass_pearson": safe_correlation(
            pearsonr, reference_mass, other_mass
        ),
        "matched_cluster_mass_spearman": safe_correlation(
            spearmanr, reference_mass, other_mass
        ),
        "matched_cluster_mass_js_divergence_bits": js_divergence,
    }
    return metrics, distance_matrix


def assign_to_centers(
    dataset: h5py.Dataset,
    centers: np.ndarray,
    batch_size: int,
    mode: str,
) -> np.ndarray:
    counts = np.zeros(len(centers), dtype=np.int64)
    for _, _, batch in iter_batches(dataset, batch_size, mode):
        distances = cdist(batch, centers, metric="sqeuclidean")
        labels = np.argmin(distances, axis=1)
        counts += np.bincount(labels, minlength=len(centers))
    return counts


def shared_reference_metrics(
    reference: SplitResult,
    other_dataset: h5py.Dataset,
    other_name: str,
    batch_size: int,
    mode: str,
) -> dict[str, object]:
    other_counts = assign_to_centers(other_dataset, reference.centers, batch_size, mode)
    other_mass = other_counts / other_counts.sum()
    reference_mass = reference.cluster_fractions
    epsilon = 1e-12
    return {
        "reference_split": reference.name,
        "other_split": other_name,
        "occupancy_pearson": safe_correlation(pearsonr, reference_mass, other_mass),
        "occupancy_spearman": safe_correlation(spearmanr, reference_mass, other_mass),
        "occupancy_js_divergence_bits": float(
            jensenshannon(reference_mass + epsilon, other_mass + epsilon, base=2.0)
            ** 2
        ),
        "other_counts_on_reference_centers": other_counts.tolist(),
    }


def save_split_outputs(output_dir: Path, result: SplitResult) -> dict[str, object]:
    np.save(output_dir / f"{result.name}.centers.npy", result.centers)
    np.save(output_dir / f"{result.name}.assignments.npy", result.labels)

    frame = pd.DataFrame(
        {
            "cluster_id": np.arange(len(result.cluster_counts)),
            "count": result.cluster_counts,
            "fraction": result.cluster_fractions,
            "radius_mean": result.radius_mean,
            "radius_p95": result.radius_p95,
        }
    )
    frame.to_csv(output_dir / f"{result.name}.cluster_statistics.csv", index=False)

    sorted_counts = np.sort(result.cluster_counts)[::-1]
    cumulative = np.cumsum(sorted_counts) / sorted_counts.sum()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(np.arange(1, len(sorted_counts) + 1), sorted_counts, marker="o", markersize=2)
    axes[0].set_xlabel("Cluster rank by size")
    axes[0].set_ylabel("Vectors in cluster")
    axes[0].set_title(f"{result.name}: sorted cluster sizes")
    axes[0].grid(alpha=0.25)
    axes[1].plot(np.arange(1, len(cumulative) + 1) / len(cumulative), cumulative)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[1].set_xlabel("Cumulative fraction of clusters")
    axes[1].set_ylabel("Cumulative fraction of vectors")
    axes[1].set_title(f"{result.name}: cluster-mass concentration")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / f"{result.name}.cluster_distribution.png", dpi=160)
    plt.close(figure)

    return {
        "name": result.name,
        "count": result.count,
        "dimensions": result.dimensions,
        "non_empty_clusters": int(np.count_nonzero(result.cluster_counts)),
        "cluster_size_min": int(result.cluster_counts.min()),
        "cluster_size_median": float(np.median(result.cluster_counts)),
        "cluster_size_max": int(result.cluster_counts.max()),
        "cluster_size_cv": float(result.cluster_counts.std() / result.cluster_counts.mean()),
        "cluster_size_gini": cluster_size_gini(result.cluster_counts),
        "top_10_percent_clusters_mass": float(
            sorted_counts[: max(1, len(sorted_counts) // 10)].sum() / sorted_counts.sum()
        ),
        "mean_point_to_center_distance": float(result.point_distances.mean()),
        "p95_point_to_center_distance": float(np.quantile(result.point_distances, 0.95)),
    }


def save_heatmap(path: Path, matrix: np.ndarray, title: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_xlabel("Other split cluster")
    axis.set_ylabel("Reference split cluster")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Center distance")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_markdown_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        f"# {summary['dataset']} vector-pattern analysis",
        "",
        f"- clusters: {summary['clusters']}",
        f"- preprocessing: `{summary['cluster_preprocess']}`",
        f"- seed: {summary['seed']}",
        "",
        "## Split statistics",
        "",
        "| split | vectors | cluster CV | cluster Gini | top 10% cluster mass |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in summary["splits"]:
        lines.append(
            f"| {split['name']} | {split['count']} | {split['cluster_size_cv']:.4f} | "
            f"{split['cluster_size_gini']:.4f} | {split['top_10_percent_clusters_mass']:.4f} |"
        )

    lines.extend(["", "## Query sequence", ""])
    query_sequence = summary.get("query_sequence", {})
    if query_sequence:
        lines.extend(
            [
                f"- self-transition probability: {query_sequence['self_transition_probability']:.6f}",
                f"- mean run length: {query_sequence['run_length_mean']:.4f}",
                f"- normalized transition entropy: {query_sequence['transition_entropy_normalized']:.6f}",
                "",
                "| lag | observed same-cluster | shuffled | locality lift |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in query_sequence["lag_same_cluster"]:
            lines.append(
                f"| {row['lag']} | {row['same_cluster_probability']:.6f} | "
                f"{row['shuffled_probability']:.6f} | {row['locality_lift']:.4f} |"
            )
    else:
        lines.append("No query split was available.")

    lines.extend(["", "## Cross-split comparisons", ""])
    lines.append(
        "| reference | other | matched center mean | mass Spearman | mass JS divergence |"
    )
    lines.append("|---|---|---:|---:|---:|")
    for row in summary["independent_center_comparisons"]:
        spearman = row["matched_cluster_mass_spearman"]
        lines.append(
            f"| {row['reference_split']} | {row['other_split']} | "
            f"{row['matched_center_distance_mean']:.6f} | "
            f"{spearman if spearman is not None else 'n/a'} | "
            f"{row['matched_cluster_mass_js_divergence_bits']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    if args.clusters <= 0:
        raise ValueError("--clusters must be positive")
    spec = load_catalog(args.catalog, args.dataset)
    input_path = resolve_repo_path(str(spec["local_hdf5"]))
    output_dir = args.output_root / args.dataset / f"k{args.clusters}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = str(spec["cluster_preprocess"])

    split_keys = [("base", str(spec["base_key"]))]
    if spec.get("learn_key"):
        split_keys.append(("learn", str(spec["learn_key"])))
    split_keys.append(("query", str(spec["query_key"])))

    with h5py.File(input_path, "r") as hdf5:
        results: dict[str, SplitResult] = {}
        split_summaries = []
        for index, (split_name, key) in enumerate(split_keys):
            print(f"fitting {args.dataset}/{split_name} from HDF5 key {key}")
            result = fit_split(
                split_name,
                hdf5[key],
                args.clusters,
                args.seed + index,
                args.batch_size,
                args.sample_size,
                args.epochs,
                args.n_init,
                mode,
            )
            results[split_name] = result
            split_summaries.append(save_split_outputs(output_dir, result))

        independent_comparisons = []
        shared_comparisons = []
        reference = results["base"]
        for split_name, key in split_keys:
            if split_name == "base":
                continue
            comparison, matrix = compare_splits(reference, results[split_name])
            independent_comparisons.append(comparison)
            save_heatmap(
                output_dir / f"base_vs_{split_name}.center_distance_heatmap.png",
                matrix,
                f"{args.dataset}: base vs {split_name} centers",
            )
            shared_comparisons.append(
                shared_reference_metrics(
                    reference, hdf5[key], split_name, args.batch_size, mode
                )
            )

        summary = {
            "dataset": args.dataset,
            "display_name": spec["display_name"],
            "input_hdf5": str(input_path),
            "distance": spec["distance"],
            "cluster_preprocess": mode,
            "clusters": args.clusters,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "sample_size": args.sample_size,
            "epochs": args.epochs,
            "splits": split_summaries,
            "query_sequence": sequence_metrics(
                results["query"].labels, args.clusters, args.seed
            ),
            "independent_center_comparisons": independent_comparisons,
            "shared_base_center_comparisons": shared_comparisons,
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown_summary(output_dir / "summary.md", summary)
    print(f"wrote analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
