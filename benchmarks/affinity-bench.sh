#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Drives `affinity_bench` against a throwaway cluster, once per task
# distribution policy, and prints the locality each one achieved.
#
# Cluster shape matters as much as the query: a distribution policy can only
# be judged where free capacity and data placement disagree. Two scenarios:
#
#   uniform   every executor up before the query; capacity and data agree,
#             so bias and shuffle-affinity tie on a plain hash shuffle.
#   scale-up  an extra executor joins just before the measured query. This does
#             NOT isolate the cold-executor case — the late executor also picks
#             up map work, so it holds data by the time the consumer stage
#             binds — and measured, it does not separate the policies.
#
# Usage:
#   benchmarks/affinity-bench.sh generate
#   benchmarks/affinity-bench.sh run [workload] [scenario]
#
#   workload: aggregate | union | join | collapse   (default: aggregate)
#   scenario: uniform | scale-up                    (default: uniform)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/target/release"
DATA="${DATA:-/tmp/affinity-bench-data}"
RUNDIR="${RUNDIR:-/tmp/affinity-bench-run}"
EXECUTORS="${EXECUTORS:-4}"
VCORES="${VCORES:-4}"
RUNS="${RUNS:-5}"
PARTITIONS="${PARTITIONS:-16}"

start_executor() {
  local i=$1
  mkdir -p "$RUNDIR/work-$i"
  # Each executor needs its OWN work dir: a read counts as local when the
  # shuffle file exists locally, so a shared dir makes everything look local
  # and measures nothing.
  "$BIN/ballista-executor" \
    --scheduler-host localhost --scheduler-port 50050 \
    --bind-port $((50100 + i * 10)) \
    --bind-grpc-port $((50101 + i * 10)) \
    --bind-health-port $((50102 + i * 10)) \
    --external-host localhost \
    --work-dir "$RUNDIR/work-$i" \
    --vcores "$VCORES" \
    --task-scheduling-policy push-staged \
    > "$RUNDIR/executor-$i.log" 2>&1 &
}

teardown() { pkill -f 'ballista-scheduler|ballista-executor' 2>/dev/null; sleep 1; }

bench_one() {
  local policy=$1 workload=$2 scenario=$3
  teardown
  rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"

  # The policy only sees the whole cluster's budget under push-staged
  # scheduling; pull-staged offers one executor at a time, leaving nothing
  # to choose between.
  RUST_LOG="warn,ballista_scheduler::cluster::affinity=debug" \
  "$BIN/ballista-scheduler" \
    --bind-port 50050 \
    --scheduler-policy push-staged \
    --task-distribution "$policy" \
    > "$RUNDIR/scheduler.log" 2>&1 &
  sleep 3

  local up=$EXECUTORS
  [ "$scenario" = "scale-up" ] && up=$((EXECUTORS - 1))
  for i in $(seq 1 $up); do start_executor "$i"; done
  sleep 6

  if [ "$scenario" = "scale-up" ]; then
    # Joins with a full budget and no data, which is what bias reaches for.
    start_executor "$EXECUTORS"
    sleep 6
  fi

  "$BIN/affinity_bench" run \
    --path "$DATA" --label "$policy" --workload "$workload" \
    --runs "$RUNS" --partitions "$PARTITIONS"

  # The policy's own byte-weighted view, which the reader-side counters
  # cannot give: they count locations, not bytes.
  grep 'shuffle-affinity:' "$RUNDIR/scheduler.log" | tail -1 \
    | sed 's/.*shuffle-affinity: /    scheduler: /'
  teardown
}

case "${1:-run}" in
  generate)
    shift
    "$BIN/affinity_bench" generate --path "$DATA" "$@"
    ;;
  run)
    workload="${2:-aggregate}"
    scenario="${3:-uniform}"
    echo "# workload=$workload scenario=$scenario executors=$EXECUTORS x ${VCORES}vcores"
    for policy in bias round-robin shuffle-affinity; do
      bench_one "$policy" "$workload" "$scenario"
    done
    ;;
  *)
    echo "usage: $0 generate|run [workload] [scenario]" >&2
    exit 1
    ;;
esac
