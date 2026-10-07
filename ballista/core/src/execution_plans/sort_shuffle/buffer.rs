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

//! In-memory buffering for sort-based shuffle.
//!
//! Rows are buffered in one of two layouts:
//!
//! * **Deferred.** Whole input batches are kept along with a
//!   `(batch_idx, row_idx)` pair per row, and output batches are gathered
//!   with `interleave_record_batch` at spill or final-write time. Nothing is
//!   held per output partition except its indices, so memory does not grow
//!   with the partition count or with how few rows each partition gets. But
//!   the gather reads each row from a different place in the buffer, which
//!   misses the cache on almost every row once the buffer outgrows it.
//! * **Split.** Each output partition's rows are `take`n into that
//!   partition's [`BatchCoalescer`] while the input batch is still hot in
//!   cache, and the input batch is dropped. The coalescer emits batches of
//!   `batch_size` rows. This is several times cheaper per row on large
//!   shuffles, but every coalescer holds a partly filled batch with room for
//!   `batch_size` rows, so it costs `num_partitions × batch_size` rows of
//!   memory however few rows arrive, plus up to a 1 MiB block of string data
//!   per partition for each view column with long values.
//!
//! Every buffer starts deferred. Once it holds `num_partitions × batch_size`
//! rows, enough to fill one batch per partition on average, it switches to
//! split if that many rows, at the width measured so far, and the string
//! blocks fit in the
//! `split_budget_bytes` given to [`BufferedBatches::new`]; otherwise it stays
//! deferred. A split buffer that is drained while most of its bytes are in
//! partly filled batches, which is what a spill under memory pressure looks
//! like, goes back to deferred for good: otherwise every later spill would
//! write a small batch per partition.
//!
//! So small shuffles, and ones whose partly filled batches would crowd the
//! memory budget (many partitions, wide rows, a small memory pool), keep the
//! deferred layout.

use super::partitioned_batch_iterator::PartitionedBatchIterator;
use datafusion::arrow::array::{Array, ArrayRef, AsArray, UInt32Array};
use datafusion::arrow::compute::BatchCoalescer;
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::arrow::error::ArrowError;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;

/// Rows buffered for every output partition of one input partition.
#[derive(Debug)]
pub struct BufferedBatches {
    num_partitions: usize,
    schema: SchemaRef,
    batch_size: usize,
    split_budget_bytes: usize,
    mode: Mode,
    /// False once the buffer has decided on its layout for good.
    may_split: bool,
    /// True once the buffer has switched to the split layout.
    has_split: bool,
}

#[derive(Debug)]
enum Mode {
    Deferred(DeferredBuffer),
    Split(SplitBuffer),
}

impl BufferedBatches {
    /// Creates a buffer for `num_partitions` output partitions that emits
    /// batches of at most `batch_size` rows.
    ///
    /// The buffer starts in the deferred layout and switches to the split
    /// layout once `num_partitions × batch_size` rows, at the width measured
    /// so far, have arrived and fit in `split_budget_bytes`. A budget of 0
    /// keeps it deferred.
    pub fn new(
        num_partitions: usize,
        schema: SchemaRef,
        batch_size: usize,
        split_budget_bytes: usize,
    ) -> Self {
        Self {
            num_partitions,
            schema,
            batch_size,
            split_budget_bytes,
            mode: Mode::Deferred(DeferredBuffer::new(num_partitions)),
            may_split: split_budget_bytes > 0,
            has_split: false,
        }
    }

    /// Returns the configured number of output partitions.
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Returns true if rows are currently split into per-partition batches
    /// as they arrive, false if they are gathered at write time.
    pub fn splits_on_arrival(&self) -> bool {
        matches!(self.mode, Mode::Split(_))
    }

    /// Returns true if the buffer has switched to the split layout at any
    /// point, even if it has since gone back to deferred.
    pub fn has_split(&self) -> bool {
        self.has_split
    }

    /// Returns true if no rows are buffered.
    pub fn is_empty(&self) -> bool {
        match &self.mode {
            Mode::Deferred(b) => b.batches.is_empty(),
            Mode::Split(b) => b.is_empty(),
        }
    }

    /// Returns the bytes held by the buffered rows, counting allocated
    /// capacity.
    pub fn memory_size(&self) -> usize {
        match &self.mode {
            Mode::Deferred(b) => b.memory_size(),
            Mode::Split(b) => b.memory_size(),
        }
    }

    /// Buffers the rows of `batch` listed in `per_partition_rows[p]` for
    /// output partition `p`, for every partition.
    ///
    /// `per_partition_rows.len()` must equal `num_partitions()`.
    pub fn push_batch(
        &mut self,
        batch: RecordBatch,
        per_partition_rows: &[Vec<u32>],
    ) -> Result<()> {
        debug_assert_eq!(per_partition_rows.len(), self.num_partitions);
        if batch.num_rows() == 0 {
            return Ok(());
        }
        match &mut self.mode {
            Mode::Split(b) => b.push_batch(&batch, per_partition_rows)?,
            Mode::Deferred(b) => {
                b.push_batch(batch, per_partition_rows);
                self.maybe_switch_to_split()?;
            }
        }
        Ok(())
    }

    /// Switches a deferred buffer to the split layout once it holds enough
    /// rows to fill a batch per partition on average, if those rows fit in
    /// the split budget. The buffered rows are gathered into the coalescers.
    fn maybe_switch_to_split(&mut self) -> Result<()> {
        let Mode::Deferred(deferred) = &mut self.mode else {
            return Ok(());
        };
        let enough_rows = self.num_partitions.saturating_mul(self.batch_size);
        if !self.may_split || deferred.rows < enough_rows {
            return Ok(());
        }
        // Decided once: the row width the buffer has seen so far stands for
        // the rest of its input.
        self.may_split = false;
        let row_bytes = deferred.used_bytes.div_ceil(deferred.rows);
        // Each coalescer also allocates its view columns' string data in
        // blocks that grow to `VIEW_BLOCK_BYTES` and stay that size, however
        // little data each batch holds.
        let view_block_bytes = self
            .num_partitions
            .saturating_mul(view_columns_with_data(&deferred.batches))
            .saturating_mul(VIEW_BLOCK_BYTES);
        let split_bytes = enough_rows
            .saturating_mul(row_bytes)
            .saturating_add(view_block_bytes);
        if split_bytes > self.split_budget_bytes {
            return Ok(());
        }

        let mut split =
            SplitBuffer::new(self.num_partitions, &self.schema, self.batch_size);
        deferred.drain(self.batch_size, |partition, batch| {
            Ok(split.push_partition(partition, batch)?)
        })?;
        self.mode = Mode::Split(split);
        self.has_split = true;
        Ok(())
    }

    /// Drains every buffered row, passing each output batch of at most
    /// `batch_size` rows to `f` with its output partition, in partition
    /// order and, within a partition, in arrival order. After this call the
    /// buffer is empty.
    ///
    /// Deferred batches are gathered one at a time, so the buffer never
    /// holds its gathered output alongside its input. A split buffer whose
    /// bytes were mostly in partly filled batches switches to the deferred
    /// layout for good: draining it wrote a small batch per partition, and
    /// staying split would do so again at every later spill.
    pub fn drain(
        &mut self,
        mut f: impl FnMut(usize, RecordBatch) -> Result<()>,
    ) -> Result<()> {
        match &mut self.mode {
            Mode::Deferred(b) => b.drain(self.batch_size, f),
            Mode::Split(b) => {
                let mostly_partial = b.partial_bytes() > b.completed_bytes;
                for (partition, batches) in b.take()?.into_iter().enumerate() {
                    for batch in batches {
                        f(partition, batch)?;
                    }
                }
                if mostly_partial {
                    self.mode = Mode::Deferred(DeferredBuffer::new(self.num_partitions));
                    self.may_split = false;
                }
                Ok(())
            }
        }
    }
}

/// Largest block a [`BatchCoalescer`] allocates for a view column's string
/// data (`MAX_BLOCK_SIZE` in arrow-select's `coalesce/byte_view.rs`). Block
/// sizes double from 8 KiB up to this and never shrink, so once warmed up
/// every partly filled batch holds one such block per view column with
/// non-inline values.
const VIEW_BLOCK_BYTES: usize = 1024 * 1024;

/// Number of `Utf8View` / `BinaryView` columns that hold values too long to
/// inline in the view (over 12 bytes) in any of `batches`.
fn view_columns_with_data(batches: &[RecordBatch]) -> usize {
    let Some(first) = batches.first() else {
        return 0;
    };
    (0..first.num_columns())
        .filter(|&col| {
            batches.iter().any(|batch| {
                let array = batch.column(col);
                match array.data_type() {
                    DataType::Utf8View => {
                        array.as_string_view().total_buffer_bytes_used() > 0
                    }
                    DataType::BinaryView => {
                        array.as_binary_view().total_buffer_bytes_used() > 0
                    }
                    _ => false,
                }
            })
        })
        .count()
}

/// Bytes `array`'s values take: fixed-width values and offsets, the
/// referenced bytes of variable-width values, and the validity bitmap.
///
/// Unlike [`Array::get_array_memory_size`] this ignores unused capacity and,
/// for view arrays, data-buffer bytes no view references, so it is what the
/// rows amount to once written, and what a reader decodes them back into.
pub(crate) fn used_bytes(array: &ArrayRef) -> usize {
    fn view_bytes(views: usize, data: usize, nulls: usize) -> usize {
        views * size_of::<u128>() + data + nulls
    }
    let nulls = array.nulls().map_or(0, |n| n.len().div_ceil(8));
    match array.data_type() {
        DataType::Utf8View => {
            let a = array.as_string_view();
            view_bytes(a.len(), a.total_buffer_bytes_used(), nulls)
        }
        DataType::BinaryView => {
            let a = array.as_binary_view();
            view_bytes(a.len(), a.total_buffer_bytes_used(), nulls)
        }
        // Covers nested types through their children, but counts a
        // dictionary's whole values array for every batch.
        _ => array
            .to_data()
            .get_slice_memory_size()
            .unwrap_or_else(|_| array.get_array_memory_size()),
    }
}

/// [`used_bytes`] summed over `batch`'s columns.
pub(crate) fn batch_used_bytes(batch: &RecordBatch) -> usize {
    batch.columns().iter().map(used_bytes).sum()
}

/// Rows split by output partition as they arrive.
#[derive(Debug)]
struct SplitBuffer {
    /// One coalescer per output partition, holding that partition's rows
    /// copied out of the input batches, in arrival order, until they fill a
    /// batch.
    coalescers: Vec<BatchCoalescer>,
    /// Each output partition's filled batches, moved out of its coalescer.
    completed: Vec<Vec<RecordBatch>>,
    /// Total `get_array_memory_size` of the batches in `completed`.
    completed_bytes: usize,
}

impl SplitBuffer {
    fn new(num_partitions: usize, schema: &SchemaRef, batch_size: usize) -> Self {
        Self {
            coalescers: (0..num_partitions)
                .map(|_| BatchCoalescer::new(schema.clone(), batch_size))
                .collect(),
            completed: vec![Vec::new(); num_partitions],
            completed_bytes: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.completed.iter().all(Vec::is_empty)
            && self.coalescers.iter().all(BatchCoalescer::is_empty)
    }

    /// Bytes held by the coalescers' partly filled batches.
    fn partial_bytes(&self) -> usize {
        self.coalescers.iter().map(BatchCoalescer::size).sum()
    }

    /// Completed batches plus the coalescers' partly filled batches.
    fn memory_size(&self) -> usize {
        self.completed_bytes + self.partial_bytes()
    }

    fn push_batch(
        &mut self,
        batch: &RecordBatch,
        per_partition_rows: &[Vec<u32>],
    ) -> std::result::Result<(), ArrowError> {
        for (partition, rows) in per_partition_rows.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            if rows.len() == batch.num_rows() {
                // Every row goes to this partition, in order.
                self.push_partition(partition, batch.clone())?;
            } else {
                let indices = UInt32Array::from_iter_values(rows.iter().copied());
                self.coalescers[partition]
                    .push_batch_with_indices(batch.clone(), &indices)?;
                self.collect_completed(partition);
            }
        }
        Ok(())
    }

    /// Appends all of `batch`'s rows to output partition `partition`.
    fn push_partition(
        &mut self,
        partition: usize,
        batch: RecordBatch,
    ) -> std::result::Result<(), ArrowError> {
        self.coalescers[partition].push_batch(batch)?;
        self.collect_completed(partition);
        Ok(())
    }

    fn collect_completed(&mut self, partition: usize) {
        while let Some(done) = self.coalescers[partition].next_completed_batch() {
            self.completed_bytes += done.get_array_memory_size();
            self.completed[partition].push(done);
        }
    }

    /// Drains every partition's rows, one `Vec` of batches per output
    /// partition.
    fn take(&mut self) -> std::result::Result<Vec<Vec<RecordBatch>>, ArrowError> {
        self.completed_bytes = 0;
        self.coalescers
            .iter_mut()
            .zip(self.completed.iter_mut())
            .map(|(coalescer, completed)| {
                coalescer.finish_buffered_batch()?;
                let mut batches = std::mem::take(completed);
                batches.extend(std::iter::from_fn(|| coalescer.next_completed_batch()));
                Ok(batches)
            })
            .collect()
    }
}

/// Whole input batches plus, per output partition, the `(batch_idx,
/// row_idx)` of each of its rows. Rows are not copied until drained.
#[derive(Debug)]
struct DeferredBuffer {
    /// All input batches, in arrival order. Indexed by `batch_idx` in
    /// `indices`.
    batches: Vec<RecordBatch>,
    /// Total `get_array_memory_size` of `batches`.
    batches_bytes: usize,
    /// One entry per output partition.
    indices: Vec<Vec<(u32, u32)>>,
    /// Total rows in `batches`.
    rows: usize,
    /// Total [`batch_used_bytes`] of `batches`.
    used_bytes: usize,
}

impl DeferredBuffer {
    fn new(num_partitions: usize) -> Self {
        Self {
            batches: Vec::new(),
            batches_bytes: 0,
            indices: vec![Vec::new(); num_partitions],
            rows: 0,
            used_bytes: 0,
        }
    }

    /// Input batches plus the allocated capacity of the index lists.
    fn memory_size(&self) -> usize {
        self.batches_bytes
            + self
                .indices
                .iter()
                .map(|v| v.capacity() * size_of::<(u32, u32)>())
                .sum::<usize>()
    }

    fn push_batch(&mut self, batch: RecordBatch, per_partition_rows: &[Vec<u32>]) {
        let batch_idx = self.batches.len() as u32;
        for (dst, rows) in self.indices.iter_mut().zip(per_partition_rows) {
            dst.extend(rows.iter().map(|&r| (batch_idx, r)));
        }
        self.batches_bytes += batch.get_array_memory_size();
        self.rows += batch.num_rows();
        self.used_bytes += batch_used_bytes(&batch);
        self.batches.push(batch);
    }

    /// Gathers each partition's rows, in arrival order, into output batches
    /// of at most `batch_size` rows, passes them to `f`, and empties the
    /// buffer. Batches are gathered one at a time.
    fn drain(
        &mut self,
        batch_size: usize,
        mut f: impl FnMut(usize, RecordBatch) -> Result<()>,
    ) -> Result<()> {
        self.batches_bytes = 0;
        self.rows = 0;
        self.used_bytes = 0;
        let batches = std::mem::take(&mut self.batches);
        let indices: Vec<_> = self.indices.iter_mut().map(std::mem::take).collect();
        for (partition, partition_indices) in indices.iter().enumerate() {
            let iter =
                PartitionedBatchIterator::new(&batches, partition_indices, batch_size);
            for batch in iter {
                f(partition, batch?)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Int32Array, StringViewArray};
    use datafusion::arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    /// Bytes a row of `create_test_batch` takes: a 4-byte int, a 16-byte
    /// view and a 32-byte string.
    const ROW_BYTES: usize = 4 + 16 + 32;

    /// What `push_two_batches` needs to split: 2 partitions x 2 rows, plus
    /// a string block per partition for the one view column.
    const TWO_BATCH_SPLIT_BYTES: usize = 2 * 2 * ROW_BYTES + 2 * VIEW_BLOCK_BYTES;

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("s", DataType::Utf8View, false),
        ]))
    }

    fn create_test_batch(schema: &SchemaRef, values: Vec<i32>) -> RecordBatch {
        let strings: Vec<String> = values
            .iter()
            .map(|v| format!("a string longer than 12 bytes {v:<2}"))
            .collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(values)),
                Arc::new(StringViewArray::from_iter_values(strings)),
            ],
        )
        .unwrap()
    }

    fn drain(bb: &mut BufferedBatches) -> Vec<(usize, RecordBatch)> {
        let mut out = vec![];
        bb.drain(|p, b| {
            out.push((p, b));
            Ok(())
        })
        .unwrap();
        assert!(bb.is_empty());
        out
    }

    #[test]
    fn used_bytes_counts_referenced_view_bytes_only() {
        let schema = create_test_schema();
        let batch = create_test_batch(&schema, (0..10).collect());
        assert_eq!(batch_used_bytes(&batch), 10 * ROW_BYTES);
        // A slice still shares the whole data buffer, but only its own
        // strings are counted.
        assert_eq!(batch_used_bytes(&batch.slice(2, 3)), 3 * ROW_BYTES);
    }

    /// With 2 partitions and a batch size of 2, the buffer has enough rows
    /// to decide after 4. Pushes 4 rows, then 3, and returns the buffer's
    /// layout after each push, and what it drains.
    fn push_two_batches(
        split_budget_bytes: usize,
    ) -> (bool, bool, Vec<(usize, RecordBatch)>) {
        let schema = create_test_schema();
        let mut bb = BufferedBatches::new(2, schema.clone(), 2, split_budget_bytes);
        assert_eq!(bb.num_partitions(), 2);
        assert!(bb.is_empty());
        assert!(!bb.splits_on_arrival());

        let a = create_test_batch(&schema, vec![10, 20, 30, 40]);
        bb.push_batch(a, &[vec![0, 2], vec![3, 1]]).unwrap();
        let split_after_a = bb.splits_on_arrival();
        let b = create_test_batch(&schema, vec![50, 60, 70]);
        bb.push_batch(b, &[vec![0], vec![1, 2]]).unwrap();
        let split_after_b = bb.splits_on_arrival();
        assert!(!bb.is_empty());
        assert!(bb.memory_size() > 0);
        (split_after_a, split_after_b, drain(&mut bb))
    }

    fn expected_two_batches() -> Vec<(usize, RecordBatch)> {
        let schema = create_test_schema();
        vec![
            (0, create_test_batch(&schema, vec![10, 30])),
            (0, create_test_batch(&schema, vec![50])),
            (1, create_test_batch(&schema, vec![40, 20])),
            (1, create_test_batch(&schema, vec![60, 70])),
        ]
    }

    #[test]
    fn switches_to_split_once_a_batch_per_partition_fits() {
        // 2 partitions x 2 rows x 52 bytes, plus a string block per
        // partition, fits exactly.
        assert_eq!(
            push_two_batches(TWO_BATCH_SPLIT_BYTES),
            (true, true, expected_two_batches())
        );
    }

    #[test]
    fn stays_deferred_when_a_batch_per_partition_does_not_fit() {
        assert_eq!(
            push_two_batches(TWO_BATCH_SPLIT_BYTES - 1),
            (false, false, expected_two_batches())
        );
    }

    #[test]
    fn zero_budget_stays_deferred() {
        assert_eq!(push_two_batches(0), (false, false, expected_two_batches()));
    }

    /// A batch of `values` whose strings are short enough to be inlined in
    /// their views, so the view column has no data buffers.
    fn create_inline_batch(schema: &SchemaRef, values: Vec<i32>) -> RecordBatch {
        let strings: Vec<String> = values.iter().map(|v| format!("s{v}")).collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(values)),
                Arc::new(StringViewArray::from_iter_values(strings)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn inline_strings_need_no_string_block() {
        let schema = create_test_schema();
        // 4-byte int + 16-byte view, no string data: 2 partitions x 2 rows
        // x 20 bytes fits exactly.
        let mut bb = BufferedBatches::new(2, schema.clone(), 2, 2 * 2 * 20);
        let batch = create_inline_batch(&schema, vec![1, 2, 3, 4]);
        bb.push_batch(batch, &[vec![0, 1], vec![2, 3]]).unwrap();
        assert!(bb.splits_on_arrival());
        assert_eq!(
            drain(&mut bb),
            vec![
                (0, create_inline_batch(&schema, vec![1, 2])),
                (1, create_inline_batch(&schema, vec![3, 4])),
            ]
        );
    }

    #[test]
    fn view_columns_with_data_counts_columns_with_long_values() {
        let schema = create_test_schema();
        let long = create_test_batch(&schema, vec![1, 2]);
        let short = create_inline_batch(&schema, vec![1, 2]);
        assert_eq!(view_columns_with_data(&[]), 0);
        assert_eq!(view_columns_with_data(std::slice::from_ref(&short)), 0);
        assert_eq!(view_columns_with_data(std::slice::from_ref(&long)), 1);
        assert_eq!(view_columns_with_data(&[short, long]), 1);
    }

    #[test]
    fn stays_deferred_until_a_batch_per_partition_has_arrived() {
        let schema = create_test_schema();
        let mut bb = BufferedBatches::new(2, schema.clone(), 4, usize::MAX);
        let batch = create_test_batch(&schema, (0..7).collect());
        bb.push_batch(batch, &[vec![0, 1, 2], vec![3, 4, 5, 6]])
            .unwrap();
        // 7 rows of the 8 that would fill a batch per partition.
        assert!(!bb.splits_on_arrival());
        assert_eq!(
            drain(&mut bb),
            vec![
                (0, create_test_batch(&schema, vec![0, 1, 2])),
                (1, create_test_batch(&schema, vec![3, 4, 5, 6])),
            ]
        );
    }

    #[test]
    fn goes_back_to_deferred_after_draining_mostly_partial_batches() {
        let schema = create_test_schema();
        let mut bb = BufferedBatches::new(2, schema.clone(), 4, usize::MAX);
        // 8 rows switch the buffer to split, leaving 3 + 1 rows in partly
        // filled batches and one completed batch.
        let batch = create_test_batch(&schema, (0..8).collect());
        bb.push_batch(batch, &[vec![0, 1, 2], vec![3, 4, 5, 6, 7]])
            .unwrap();
        assert!(bb.splits_on_arrival());

        assert_eq!(
            drain(&mut bb),
            vec![
                (0, create_test_batch(&schema, vec![0, 1, 2])),
                (1, create_test_batch(&schema, vec![3, 4, 5, 6])),
                (1, create_test_batch(&schema, vec![7])),
            ]
        );
        assert!(!bb.splits_on_arrival());

        // It stays deferred, however many rows arrive.
        let batch = create_test_batch(&schema, (0..16).collect());
        bb.push_batch(batch, &[(0..8).collect(), (8..16).collect()])
            .unwrap();
        assert!(!bb.splits_on_arrival());
    }

    #[test]
    fn stays_split_after_draining_mostly_completed_batches() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let mut bb = BufferedBatches::new(1, schema.clone(), 2, usize::MAX);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from((0..9).collect::<Vec<i32>>()))],
        )
        .unwrap();
        bb.push_batch(batch, &[(0..9).collect()]).unwrap();
        assert!(bb.splits_on_arrival());

        let drained: Vec<usize> =
            drain(&mut bb).iter().map(|(_, b)| b.num_rows()).collect();
        assert_eq!(drained, vec![2, 2, 2, 2, 1]);
        assert!(bb.splits_on_arrival());
    }

    #[test]
    fn both_layouts_compact_sparse_view_columns() {
        for split_budget_bytes in [usize::MAX, 0] {
            let schema = create_test_schema();
            let mut bb = BufferedBatches::new(2, schema.clone(), 8, split_budget_bytes);
            let batch = create_test_batch(&schema, (0..50).collect());
            let rest: Vec<u32> = (0..50).filter(|&r| r != 7).collect();
            bb.push_batch(batch, &[vec![7], rest]).unwrap();
            assert_eq!(bb.splits_on_arrival(), split_budget_bytes > 0);

            let drained = drain(&mut bb);
            assert_eq!(drained[0], (0, create_test_batch(&schema, vec![7])));
            let s = drained[0].1.column(1).as_string_view();
            let bytes: usize = s.data_buffers().iter().map(|b| b.len()).sum();
            assert_eq!(bytes, 32);
        }
    }
}
