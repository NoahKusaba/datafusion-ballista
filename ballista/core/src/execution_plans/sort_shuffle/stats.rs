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

//! Byte counts for the sort-shuffle writer's partition stats.

use datafusion::arrow::array::{Array, ArrayRef, AsArray};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;

/// Bytes `array`'s values take, ignoring unused capacity and, for view
/// arrays, data-buffer bytes no view references: what the rows amount to
/// once written. A dictionary counts its whole values array.
pub(crate) fn used_bytes(array: &ArrayRef) -> usize {
    let view_data = match array.data_type() {
        DataType::Utf8View => array.as_string_view().total_buffer_bytes_used(),
        DataType::BinaryView => array.as_binary_view().total_buffer_bytes_used(),
        _ => {
            return array
                .to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| array.get_array_memory_size());
        }
    };
    let nulls = array.nulls().map_or(0, |n| n.len().div_ceil(8));
    array.len() * size_of::<u128>() + view_data + nulls
}

/// [`used_bytes`] summed over `batch`'s columns.
pub(crate) fn batch_used_bytes(batch: &RecordBatch) -> usize {
    batch.columns().iter().map(used_bytes).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Int64Builder, StringViewArray};
    use std::sync::Arc;

    #[test]
    fn primitive_ignores_spare_capacity() {
        let mut builder = Int64Builder::with_capacity(1024);
        builder.append_slice(&[1, 2, 3]);
        builder.append_null();
        let array: ArrayRef = Arc::new(builder.finish());

        // 4 values and a 1-byte validity bitmap, not 1024 values' worth.
        assert_eq!(used_bytes(&array), 4 * 8 + 1);
    }

    #[test]
    fn view_counts_referenced_data_only() {
        let strings = (0..10).map(|i| format!("a string longer than 12 bytes {i}"));
        let array: ArrayRef = Arc::new(StringViewArray::from_iter_values(strings));
        const ROW_BYTES: usize = 16 + 31;

        assert_eq!(used_bytes(&array), 10 * ROW_BYTES);
        // The slice shares the whole data buffer; only its strings count.
        assert_eq!(used_bytes(&array.slice(2, 3)), 3 * ROW_BYTES);
        // Short strings are stored inline in the view.
        let inline: ArrayRef = Arc::new(StringViewArray::from_iter_values(["a", "b"]));
        assert_eq!(used_bytes(&inline), 2 * 16);
    }
}
