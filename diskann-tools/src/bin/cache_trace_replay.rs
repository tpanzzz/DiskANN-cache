/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use diskann_disk::data_model::{
    replay_belady_optimal, replay_online_policy, CachePolicyKind, CachePolicyStats,
};
use serde_json::Value;

#[derive(Debug, Parser)]
#[command(
    name = "cache_trace_replay",
    about = "Replay vertex-id traces against cache policies for DiskANN cache evaluation"
)]
struct Args {
    /// Input trace file. JSONL lines may contain vertex_id/id/vertex fields; text/CSV lines use --vertex-id-column.
    #[arg(long)]
    trace: PathBuf,

    /// Cache capacities to evaluate, as node counts.
    #[arg(long, value_delimiter = ',', required = true)]
    capacities: Vec<usize>,

    /// Policies to evaluate.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "no_cache,fifo,lru,lfu,tiny_lfu,random,belady_opt"
    )]
    policies: Vec<CachePolicyKind>,

    /// Zero-based vertex id column for non-JSONL trace lines.
    #[arg(long, default_value_t = 0)]
    vertex_id_column: usize,

    /// Skip the first non-empty non-JSONL line.
    #[arg(long, default_value_t = false)]
    has_header: bool,

    /// Optional CSV output path. Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let trace = read_trace(&args)?;

    let mut output: Box<dyn Write> = match &args.output {
        Some(path) => Box::new(BufWriter::new(
            File::create(path).with_context(|| format!("creating {}", path.display()))?,
        )),
        None => Box::new(BufWriter::new(std::io::stdout())),
    };

    writeln!(
        output,
        "policy,capacity,accesses,hits,misses,hit_rate,admissions,evictions,rejections"
    )?;

    for capacity in &args.capacities {
        for policy in &args.policies {
            let stats = replay_policy(*policy, *capacity, &trace)
                .with_context(|| format!("replaying policy {policy} capacity {capacity}"))?;
            write_stats(output.as_mut(), *policy, *capacity, stats)?;
        }
    }

    Ok(())
}

fn replay_policy(
    policy: CachePolicyKind,
    capacity: usize,
    trace: &[u32],
) -> Result<CachePolicyStats> {
    if policy == CachePolicyKind::BeladyOptimal {
        Ok(replay_belady_optimal(capacity, trace))
    } else {
        replay_online_policy(policy, capacity, trace).map_err(|err| anyhow!(err.to_string()))
    }
}

fn write_stats(
    output: &mut dyn Write,
    policy: CachePolicyKind,
    capacity: usize,
    stats: CachePolicyStats,
) -> Result<()> {
    writeln!(
        output,
        "{policy},{capacity},{},{},{},{:.8},{},{},{}",
        stats.accesses,
        stats.hits,
        stats.misses,
        stats.hit_rate(),
        stats.admissions,
        stats.evictions,
        stats.rejections
    )?;
    Ok(())
}

fn read_trace(args: &Args) -> Result<Vec<u32>> {
    let file =
        File::open(&args.trace).with_context(|| format!("opening {}", args.trace.display()))?;
    let reader = BufReader::new(file);
    let mut trace = Vec::new();
    let mut skipped_header = false;

    for (line_number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", line_number + 1))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with('{') {
            trace.push(parse_json_vertex_id(line, line_number + 1)?);
            continue;
        }

        if args.has_header && !skipped_header {
            skipped_header = true;
            continue;
        }

        trace.push(parse_text_vertex_id(
            line,
            args.vertex_id_column,
            line_number + 1,
        )?);
    }

    if trace.is_empty() {
        return Err(anyhow!(
            "trace {} did not contain any vertex ids",
            args.trace.display()
        ));
    }

    Ok(trace)
}

fn parse_json_vertex_id(line: &str, line_number: usize) -> Result<u32> {
    let value: Value =
        serde_json::from_str(line).with_context(|| format!("parsing JSON line {line_number}"))?;
    for field in ["vertex_id", "id", "vertex"] {
        if let Some(id) = value.get(field).and_then(Value::as_u64) {
            return u32::try_from(id)
                .with_context(|| format!("vertex id on line {line_number} exceeds u32"));
        }
    }

    Err(anyhow!(
        "JSON line {line_number} has no numeric vertex_id, id, or vertex field"
    ))
}

fn parse_text_vertex_id(line: &str, vertex_id_column: usize, line_number: usize) -> Result<u32> {
    let fields = line
        .split(|ch: char| ch == ',' || ch == '\t' || ch.is_ascii_whitespace())
        .filter(|field| !field.is_empty())
        .collect::<Vec<_>>();

    let field = fields.get(vertex_id_column).ok_or_else(|| {
        anyhow!(
            "line {line_number} has {} fields, missing vertex id column {}",
            fields.len(),
            vertex_id_column
        )
    })?;

    field
        .parse::<u32>()
        .with_context(|| format!("parsing vertex id on line {line_number}"))
}
