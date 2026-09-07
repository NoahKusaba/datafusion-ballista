// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Measures what the `shuffle-affinity` task distribution policy buys against
//! the `bias` and `round-robin` policies it competes with ([#2319]).
//!
//! Every `ShuffleReaderExec` counts the partitions it served from a node-local
//! file against the ones it fetched over Arrow Flight, and `EXPLAIN ANALYZE`
//! brings those counters back from the executors: a policy that works shows a
//! higher local share.
//!
//! Wall clock is reported too, but only means something on a cluster with a
//! real network between executors — on one machine a "remote" read is a
//! loopback gRPC hop. `--force-remote-read` makes every read remote, so the gap
//! against a normal run is the whole prize any locality policy competes for.
//!
//! # What can and cannot show a benefit
//!
//! A policy can only be judged where free capacity and data placement disagree,
//! and on a plain hash shuffle they agree: every partition ranks the executors
//! identically, so affinity reduces to "pack onto the biggest holder", which is
//! what bias already does. All three policies read exactly 1/4 of the input
//! locally on a uniform 4-executor cluster.
//!
//! The default `aggregate` workload is therefore a *control*, showing the
//! policy costs nothing. [`Workload::Collapse`] is the one that separates the
//! policies reproducibly: its single task lands on the executor holding most of
//! the stage every time, where bias gets there only by luck.
//!
//! # Usage
//!
//! ```sh
//! # 1. Generate input (once). `--skew` concentrates a share of the rows on
//! #    one hot key, which only the `join` workload carries into the shuffle.
//! cargo run --release --bin affinity_bench -- generate --path /tmp/affinity-data
//!
//! # 2. Start a scheduler with the policy under test under push-staged
//! #    scheduling, plus N executors, each with its OWN work dir: executors
//! #    sharing one make every read look local and measure nothing.
//!
//! # 3. Run.
//! cargo run --release --bin affinity_bench -- \
//!   run --path /tmp/affinity-data --workload union --runs 5
//! ```
//!
//! `benchmarks/affinity-bench.sh` does all three for every policy in turn.
//!
//! [#2319]: https://github.com/apache/datafusion-ballista/issues/2319

use ballista::datafusion::arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use ballista::datafusion::arrow::datatypes::{DataType, Field, Schema};
use ballista::datafusion::common::Result;
use ballista::datafusion::execution::SessionStateBuilder;
use ballista::datafusion::parquet::arrow::ArrowWriter;
use ballista::datafusion::parquet::basic::Compression;
use ballista::datafusion::parquet::file::properties::WriterProperties;
use ballista::datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use ballista::prelude::{SessionConfigExt, SessionContextExt};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(about = "Benchmark the shuffle-affinity task distribution policy")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write the synthetic input the benchmark reads: table `t`, plus the
    /// smaller `u` that the `join` and `union` workloads need.
    Generate {
        #[arg(long)]
        path: PathBuf,
        /// One file per producer task, so this sets the map-side width.
        #[arg(long, default_value_t = 8)]
        files: usize,
        #[arg(long, default_value_t = 2_000_000)]
        rows_per_file: usize,
        /// Distinct group keys. High cardinality keeps the shuffle wide and
        /// stops the partial aggregate from collapsing it away.
        #[arg(long, default_value_t = 400_000)]
        keys: usize,
        /// Concentrate a share of `t`'s rows on one hot key, so the shuffle
        /// partition it hashes to dwarfs the rest. Only `join` carries that
        /// skew into the shuffle — an aggregate collapses a hot key to one
        /// row per producer before the data ever moves.
        #[arg(long, default_value_t = 0.0)]
        skew: f64,
    },
    /// Run the query against a cluster and report locality and wall clock.
    Run(RunArgs),
}

#[derive(Args)]
struct RunArgs {
    #[arg(long, default_value = "df://localhost:50050")]
    scheduler: String,
    #[arg(long)]
    path: PathBuf,
    #[arg(long, default_value_t = 5)]
    runs: usize,
    #[arg(long, default_value_t = 16)]
    partitions: usize,
    /// Treat every shuffle read as remote, whatever the policy decided.
    /// The gap against a normal run is the ceiling on what any locality
    /// policy can recover.
    #[arg(long)]
    force_remote_read: bool,
    /// Label for the results line, e.g. the policy under test.
    #[arg(long, default_value = "unlabelled")]
    label: String,
    /// Print the annotated distributed plan before the results line.
    #[arg(long)]
    show_plan: bool,
    /// Which query to run. See [`Workload`].
    #[arg(long, value_enum, default_value_t = Workload::Aggregate)]
    workload: Workload,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    match Cli::parse().command {
        Command::Generate {
            path,
            files,
            rows_per_file,
            keys,
            skew,
        } => generate(&path, files, rows_per_file, keys, skew),
        Command::Run(args) => run(args).await,
    }
}

/// The query shape under test. Each one puts the policy in a different
/// position, because what a distribution policy can exploit is decided by the
/// plan, not by the cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Workload {
    /// High-cardinality grouped aggregate: the control.
    ///
    /// Every producer writes every output partition at roughly equal size, so
    /// each partition is spread evenly over the `E` producing executors and
    /// every placement reads `1/E` of it locally. Affinity cannot beat bias
    /// here and is not expected to — run it to confirm the policy costs
    /// nothing, not to show a win.
    Aggregate,
    /// `UNION ALL` of two tables.
    ///
    /// A union splits its output partition space across its children, so a
    /// partition belongs to *one* child and inherits only that child's
    /// producers. Partitions therefore have genuinely different homes, which
    /// is the shape the per-partition ranking exists for and the one a plain
    /// hash shuffle never produces.
    Union,
    /// Hash join of `t` against `u`.
    ///
    /// The shuffle carries raw rows rather than pre-aggregated groups, so a
    /// hot key stays hot all the way through it. With `generate --skew` the
    /// partitions differ by orders of magnitude in size, and a scarce vcore
    /// spent on the heaviest one is worth many spent on the rest.
    Join,
    /// Global aggregate: the consumer stage collapses to a single task.
    ///
    /// That one task reads *every* partition, so it is placed on the executor
    /// holding most of the stage rather than sliced up. Bias places it on the
    /// biggest budget instead, which is the same executor only by luck.
    Collapse,
}

impl Workload {
    /// The SQL to time. `t` is the main table, `u` the smaller one.
    fn query(self) -> &'static str {
        match self {
            // The partial aggregate runs on the map side, the shuffle carries
            // one row per (key, producer), and the final aggregate is the
            // consumer stage whose tasks the policy places.
            Workload::Aggregate => {
                "SELECT k, count(*) AS n, sum(v) AS total \
                 FROM t GROUP BY k ORDER BY total DESC LIMIT 20"
            }
            // Aggregated per branch so each union child is its own shuffle
            // producer, giving the two halves of the partition space
            // different homes.
            Workload::Union => {
                "SELECT k, sum(total) AS total FROM ( \
                   SELECT k, sum(v) AS total FROM t GROUP BY k \
                   UNION ALL \
                   SELECT k, sum(v) AS total FROM u GROUP BY k \
                 ) GROUP BY k ORDER BY total DESC LIMIT 20"
            }
            // No GROUP BY before the join, so the shuffle moves rows and
            // keeps whatever size skew the data has.
            Workload::Join => {
                "SELECT t.k, count(*) AS n, sum(t.v) AS total \
                 FROM t JOIN u ON t.k = u.k \
                 GROUP BY t.k ORDER BY total DESC LIMIT 20"
            }
            // No GROUP BY at all: one final task drains every partition.
            Workload::Collapse => "SELECT count(*) AS n, sum(v) AS total FROM t",
        }
    }

    /// Whether the workload reads `u`.
    fn needs_second_table(self) -> bool {
        matches!(self, Workload::Union | Workload::Join)
    }
}

fn generate(
    path: &Path,
    files: usize,
    rows_per_file: usize,
    keys: usize,
    skew: f64,
) -> Result<()> {
    // `u` is a tenth of `t` and one file wide, so the join and union
    // workloads have a small side whose producers are a different set of
    // executors from the large side's.
    write_table(&path.join("t"), files, rows_per_file, keys, skew)?;
    write_table(&path.join("u"), 1, rows_per_file / 10, keys, 0.0)?;
    Ok(())
}

/// Writes one table as `files` Parquet files, one row group each, so the scan
/// produces one map task per file.
///
/// `skew` is the share of rows forced onto a single hot key; the rest spread
/// deterministically over the key space, every file touching every key range.
fn write_table(
    path: &Path,
    files: usize,
    rows_per_file: usize,
    keys: usize,
    skew: f64,
) -> Result<()> {
    fs::create_dir_all(path)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
        Field::new("pad", DataType::Utf8, false),
    ]));

    let hot_rows = (rows_per_file as f64 * skew.clamp(0.0, 1.0)) as usize;

    for file in 0..files {
        let ks: Vec<String> = (0..rows_per_file)
            .map(|row| {
                if row < hot_rows {
                    "key-hot".to_string()
                } else {
                    format!("key-{:08}", (row * 7919 + file * 104_729) % keys)
                }
            })
            .collect();
        let vs: Vec<i64> = (0..rows_per_file).map(|row| (row % 1000) as i64).collect();
        let pad: Vec<String> = (0..rows_per_file)
            .map(|row| format!("{:0>48}", row % 97))
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ks)),
                Arc::new(Int64Array::from(vs)),
                Arc::new(StringArray::from(pad)),
            ],
        )?;

        let target = path.join(format!("part-{file:03}.parquet"));
        let out = fs::File::create(&target)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(out, schema.clone(), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        println!("wrote {}", target.display());
    }
    Ok(())
}

async fn run(args: RunArgs) -> Result<()> {
    let RunArgs {
        scheduler,
        path,
        runs,
        partitions,
        force_remote_read,
        label,
        show_plan,
        workload,
    } = args;
    let mut config = SessionConfig::new_with_ballista()
        .with_target_partitions(partitions)
        .with_ballista_job_name("affinity benchmark");
    if force_remote_read {
        config = config.with_ballista_shuffle_reader_force_remote_read(true);
    }
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .build();
    let ctx = SessionContext::remote_with_state(&scheduler, state).await?;
    register(&ctx, "t", &path.join("t")).await?;
    if workload.needs_second_table() {
        register(&ctx, "u", &path.join("u")).await?;
    }

    let query = workload.query();

    // One untimed run so file caches and client pools are warm for all
    // policies alike.
    ctx.sql(query).await?.collect().await?;

    let mut elapsed = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        ctx.sql(query).await?.collect().await?;
        elapsed.push(started.elapsed());
    }
    elapsed.sort();

    let analyze = ctx
        .sql(&format!("EXPLAIN ANALYZE {query}"))
        .await?
        .collect()
        .await?;
    if show_plan {
        println!(
            "{}",
            ballista::datafusion::arrow::util::pretty::pretty_format_batches(&analyze)?
        );
    }
    let locality = Locality::from_explain(&analyze);

    println!(
        "{label:<20} workload={workload:?} runs={runs} median={:>8.3}s min={:>8.3}s max={:>8.3}s  \
         local_partitions={} remote_partitions={} local_share={:.1}%  \
         remote_bytes={:.1}MB local_read_time={:?} fetch_time={:?} \
         permit_wait={:?}",
        elapsed[runs / 2].as_secs_f64(),
        elapsed[0].as_secs_f64(),
        elapsed[runs - 1].as_secs_f64(),
        locality.local_partitions,
        locality.remote_partitions,
        locality.local_share() * 100.0,
        locality.remote_bytes as f64 / 1e6,
        locality.local_read_time,
        locality.fetch_time,
        locality.permit_wait_time,
    );
    Ok(())
}

/// Registers one table from its directory under `--path`.
async fn register(ctx: &SessionContext, name: &str, dir: &Path) -> Result<()> {
    ctx.register_parquet(name, dir.to_str().unwrap(), ParquetReadOptions::default())
        .await
}

/// The reader-side counters, summed over every stage of the query.
#[derive(Debug, Default)]
struct Locality {
    local_partitions: u64,
    remote_partitions: u64,
    local_read_time: Duration,
    fetch_time: Duration,
    /// Time blocked on the reduce-side in-flight governor. Only remote fetches
    /// take its permits, so locality shows up here as well as in `fetch_time`.
    permit_wait_time: Duration,
    /// In-memory Arrow bytes pulled over Arrow Flight. This is the number that
    /// moves when a policy works: `local_partitions` counts *locations*, so it
    /// is blind to whether a local location holds 10% or 90% of a partition.
    remote_bytes: u64,
}

impl Locality {
    /// Share of shuffle partitions served from a node-local file.
    fn local_share(&self) -> f64 {
        let total = self.local_partitions + self.remote_partitions;
        if total == 0 {
            return 0.0;
        }
        self.local_partitions as f64 / total as f64
    }

    /// `EXPLAIN ANALYZE` returns the stage plans with their metrics rendered
    /// inline, so the counters are scraped back out of that text.
    fn from_explain(batches: &[RecordBatch]) -> Self {
        let mut locality = Self::default();
        for batch in batches {
            for column in batch.columns() {
                let Some(text) = column.as_any().downcast_ref::<StringArray>() else {
                    continue;
                };
                for row in 0..text.len() {
                    if text.is_null(row) {
                        continue;
                    }
                    locality.scrape(text.value(row));
                }
            }
        }
        locality
    }

    fn scrape(&mut self, text: &str) {
        self.local_partitions += sum_metric(text, "local_partitions=", Unit::Count);
        self.remote_partitions += sum_metric(text, "remote_partitions=", Unit::Count);
        self.local_read_time +=
            Duration::from_nanos(sum_metric(text, "local_read_time=", Unit::Nanos));
        self.fetch_time +=
            Duration::from_nanos(sum_metric(text, "fetch_time=", Unit::Nanos));
        self.permit_wait_time +=
            Duration::from_nanos(sum_metric(text, "permit_wait_time=", Unit::Nanos));
        self.remote_bytes += sum_metric(text, "decoded_bytes=", Unit::Bytes);
    }
}

/// How a metric's rendered value should be read back.
#[derive(Clone, Copy)]
enum Unit {
    /// A bare count, no suffix.
    Count,
    /// `ns` / `µs` / `ms` / `s`, no space — normalised to nanoseconds.
    Nanos,
    /// `B` / `KB` / `MB` / `GB`, or the bare magnitude, after a space —
    /// normalised to bytes.
    Bytes,
}

/// Sums every `name<value>` occurrence in `text`.
///
/// DataFusion renders each metric kind differently — `fetch_time=17.94s`,
/// `decoded_bytes=389.3 M`, `local_partitions=16` — and a suffix like `M` is
/// ambiguous, so the caller says which shape to expect.
fn sum_metric(text: &str, name: &str, unit: Unit) -> u64 {
    let mut total = 0u64;
    for (index, _) in text.match_indices(name) {
        let rest = &text[index + name.len()..];
        let digits: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        let Ok(number) = digits.parse::<f64>() else {
            continue;
        };
        let suffix = rest[digits.len()..].trim_start();
        let scale: f64 = match unit {
            Unit::Count => 1.0,
            Unit::Nanos => {
                if suffix.starts_with("ns") {
                    1.0
                } else if suffix.starts_with("µs") || suffix.starts_with("us") {
                    1e3
                } else if suffix.starts_with("ms") {
                    1e6
                } else if suffix.starts_with('s') {
                    1e9
                } else {
                    1.0
                }
            }
            Unit::Bytes => match suffix.chars().next() {
                Some('K') => 1e3,
                Some('M') => 1e6,
                Some('G') => 1e9,
                _ => 1.0,
            },
        };
        total += (number * scale) as u64;
    }
    total
}
