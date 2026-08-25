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

use datafusion::error::DataFusionError;
use datafusion::physical_plan::metrics::{MetricValue, MetricsSet};
use std::convert::TryInto;

use crate::error::BallistaError;

use crate::serde::protobuf::{self, NamedPruningMetrics, NamedRatio};
use datafusion_proto::protobuf as datafusion_protobuf;

use crate::serde::scheduler::{
    Action, ExecutorData, ExecutorMetadata, ExecutorOperatingSystemSpecification,
    ExecutorSpecification, PartitionId, PartitionLocation, PartitionStats,
};
use datafusion::physical_plan::Partitioning;
use protobuf::{NamedCount, NamedGauge, NamedTime, action::ActionType, operator_metric};

impl TryInto<protobuf::Action> for Action {
    type Error = BallistaError;

    fn try_into(self) -> Result<protobuf::Action, Self::Error> {
        match self {
            Action::FetchPartition {
                job_id,
                stage_id,
                partition_id,
                file_id,
                host,
                port,
                is_sort_shuffle,
            } => Ok(protobuf::Action {
                action_type: Some(ActionType::FetchPartition(protobuf::FetchPartition {
                    job_id: job_id.into(),
                    stage_id: stage_id as u32,
                    partition_id: partition_id as u32,
                    host,
                    port: port as u32,
                    file_id,
                    is_sort_shuffle,
                })),
                settings: vec![],
            }),
        }
    }
}

#[allow(clippy::from_over_into)]
impl Into<protobuf::PartitionId> for PartitionId {
    fn into(self) -> protobuf::PartitionId {
        protobuf::PartitionId {
            job_id: self.job_id.into(),
            stage_id: self.stage_id as u32,
            partition_id: self.partition_id as u32,
        }
    }
}

/// Encodes only what a shuffle fetch addresses: which executor produced the
/// block (`id`) and where to ask for it (`host`, `port`, `grpc_port`).
///
/// Executor capacity (`specification`) and machine description (`os_info`) are
/// left off — no consumer of a decoded `PartitionLocation` reads them, and a
/// stage holds K x M locations that every task re-encodes.
fn shuffle_fetch_executor_meta(meta: ExecutorMetadata) -> protobuf::ExecutorMetadata {
    protobuf::ExecutorMetadata {
        id: meta.id,
        host: meta.host,
        port: meta.port as u32,
        grpc_port: meta.grpc_port as u32,
        specification: None,
        os_info: None,
    }
}

impl TryInto<protobuf::PartitionLocation> for PartitionLocation {
    type Error = BallistaError;

    fn try_into(self) -> Result<protobuf::PartitionLocation, Self::Error> {
        Ok(protobuf::PartitionLocation {
            map_partition_id: self.map_partition_id as u32,
            partition_id: Some(self.partition_id.into()),
            executor_meta: Some(shuffle_fetch_executor_meta(self.executor_meta)),
            partition_stats: Some(self.partition_stats.into()),
            file_id: self.file_id,
            is_sort_shuffle: self.is_sort_shuffle,
        })
    }
}

#[allow(clippy::from_over_into)]
impl Into<protobuf::PartitionStats> for PartitionStats {
    fn into(self) -> protobuf::PartitionStats {
        let none_value = -1_i64;
        protobuf::PartitionStats {
            num_rows: self.num_rows.map(|n| n as i64).unwrap_or(none_value),
            num_batches: self.num_batches.map(|n| n as i64).unwrap_or(none_value),
            num_bytes: self.num_bytes.map(|n| n as i64).unwrap_or(none_value),
            column_stats: vec![],
        }
    }
}

/// Converts a hash partitioning scheme to its protobuf representation.
///
/// Returns `None` if the partitioning is not hash-based or if no partitioning is specified.
pub fn hash_partitioning_to_proto(
    output_partitioning: Option<&Partitioning>,
) -> Result<Option<datafusion_protobuf::PhysicalHashRepartition>, BallistaError> {
    match output_partitioning {
        Some(Partitioning::Hash(exprs, partition_count)) => {
            let default_codec =
                datafusion_proto::physical_plan::DefaultPhysicalExtensionCodec {};
            Ok(Some(datafusion_protobuf::PhysicalHashRepartition {
                hash_expr: exprs
                    .iter()
                    .map(|expr|datafusion_proto::physical_plan::to_proto::serialize_physical_expr(&expr.clone(), &default_codec))
                    .collect::<Result<Vec<_>, DataFusionError>>()?,
                partition_count: *partition_count as u64,
            }))
        }
        None => Ok(None),
        other => Err(BallistaError::General(format!(
            "scheduler::to_proto() invalid partitioning for ExecutePartition: {other:?}"
        ))),
    }
}

impl TryInto<operator_metric::Metric> for &MetricValue {
    type Error = BallistaError;

    fn try_into(self) -> Result<operator_metric::Metric, Self::Error> {
        match self {
            MetricValue::OutputRows(count) => {
                Ok(operator_metric::Metric::OutputRows(count.value() as u64))
            }
            MetricValue::ElapsedCompute(time) => {
                Ok(operator_metric::Metric::ElapseTime(time.value() as u64))
            }
            MetricValue::SpillCount(count) => {
                Ok(operator_metric::Metric::SpillCount(count.value() as u64))
            }
            MetricValue::SpilledBytes(count) => {
                Ok(operator_metric::Metric::SpilledBytes(count.value() as u64))
            }
            MetricValue::SpilledRows(count) => {
                Ok(operator_metric::Metric::SpilledRows(count.value() as u64))
            }
            MetricValue::CurrentMemoryUsage(gauge) => Ok(
                operator_metric::Metric::CurrentMemoryUsage(gauge.value() as u64),
            ),
            MetricValue::Count { name, count } => {
                Ok(operator_metric::Metric::Count(NamedCount {
                    name: name.to_string(),
                    value: count.value() as u64,
                }))
            }
            MetricValue::Gauge { name, gauge } => {
                Ok(operator_metric::Metric::Gauge(NamedGauge {
                    name: name.to_string(),
                    value: gauge.value() as u64,
                }))
            }
            MetricValue::PeakMemoryUsage { name, gauge } => {
                Ok(operator_metric::Metric::Gauge(NamedGauge {
                    name: name.to_string(),
                    value: gauge.value() as u64,
                }))
            }
            MetricValue::Time { name, time } => {
                Ok(operator_metric::Metric::Time(NamedTime {
                    name: name.to_string(),
                    value: time.value() as u64,
                }))
            }
            MetricValue::StartTimestamp(timestamp) => {
                Ok(operator_metric::Metric::StartTimestamp(
                    timestamp
                        .value()
                        .and_then(|m| m.timestamp_nanos_opt())
                        .unwrap_or(0),
                ))
            }
            MetricValue::EndTimestamp(timestamp) => {
                Ok(operator_metric::Metric::EndTimestamp(
                    timestamp
                        .value()
                        .and_then(|m| m.timestamp_nanos_opt())
                        .unwrap_or(0),
                ))
            }
            MetricValue::OutputBytes(count) => {
                Ok(operator_metric::Metric::OutputBytes(count.value() as u64))
            }
            MetricValue::PruningMetrics {
                name,
                pruning_metrics,
            } => Ok(operator_metric::Metric::PruningMetrics(
                NamedPruningMetrics {
                    name: name.to_string(),
                    pruned: pruning_metrics.pruned() as u64,
                    matched: pruning_metrics.matched() as u64,
                },
            )),
            MetricValue::Ratio {
                name,
                ratio_metrics,
            } => Ok(operator_metric::Metric::Ratio(NamedRatio {
                name: name.to_string(),
                part: ratio_metrics.part() as u64,
                total: ratio_metrics.total() as u64,
            })),
            MetricValue::OutputBatches(count) => {
                Ok(operator_metric::Metric::OutputBatches(count.value() as u64))
            }
            // at the moment there there is no way to serialize custom metrics
            // thus at the moment we can't support it
            MetricValue::Custom { .. } => Err(BallistaError::General(String::from(
                "Custom metrics values are not supported",
            ))),
        }
    }
}

impl TryInto<protobuf::OperatorMetricsSet> for MetricsSet {
    type Error = BallistaError;

    fn try_into(self) -> Result<protobuf::OperatorMetricsSet, Self::Error> {
        let metrics = self
            .iter()
            .map(|m| {
                let metric: operator_metric::Metric = m.value().try_into()?;
                Ok(protobuf::OperatorMetric {
                    metric: Some(metric),
                    partition: m.partition().map(|p| p as u32),
                })
            })
            .collect::<Result<Vec<_>, BallistaError>>()?;
        Ok(protobuf::OperatorMetricsSet { metrics })
    }
}

#[allow(clippy::from_over_into)]
impl Into<protobuf::ExecutorMetadata> for ExecutorMetadata {
    fn into(self) -> protobuf::ExecutorMetadata {
        protobuf::ExecutorMetadata {
            id: self.id,
            host: self.host,
            port: self.port as u32,
            grpc_port: self.grpc_port as u32,
            specification: Some(self.specification.into()),
            os_info: Some(self.os_info.into()),
        }
    }
}

#[allow(clippy::from_over_into)]
impl Into<protobuf::ExecutorSpecification> for ExecutorSpecification {
    fn into(self) -> protobuf::ExecutorSpecification {
        protobuf::ExecutorSpecification {
            resources: vec![protobuf::executor_resource::Resource::Vcores(self.vcores)]
                .into_iter()
                .map(|r| protobuf::ExecutorResource { resource: Some(r) })
                .collect(),
        }
    }
}

#[allow(clippy::from_over_into)]
impl Into<protobuf::ExecutorOperatingSystemSpecification>
    for ExecutorOperatingSystemSpecification
{
    fn into(self) -> protobuf::ExecutorOperatingSystemSpecification {
        protobuf::ExecutorOperatingSystemSpecification {
            system_name: self.system_name,
            kernel_ver: self.kernel_ver,
            os_ver: self.os_ver,
            os_ver_long: self.os_ver_long,
            physical_cores: self.physical_cores,
            num_disks: self.num_disks,
            total_disk_space: self.total_disk_space,
            total_available_disk_space: self.total_available_disk_space,
            open_files_limit: self.open_files_limit,
        }
    }
}

struct ExecutorResourcePair {
    total: protobuf::executor_resource::Resource,
    available: protobuf::executor_resource::Resource,
}

#[allow(clippy::from_over_into)]
impl Into<protobuf::ExecutorData> for ExecutorData {
    fn into(self) -> protobuf::ExecutorData {
        protobuf::ExecutorData {
            executor_id: self.executor_id,
            resources: vec![ExecutorResourcePair {
                total: protobuf::executor_resource::Resource::Vcores(self.total_vcores),
                available: protobuf::executor_resource::Resource::Vcores(
                    self.available_vcores,
                ),
            }]
            .into_iter()
            .map(|r| protobuf::ExecutorResourcePair {
                total: Some(protobuf::ExecutorResource {
                    resource: Some(r.total),
                }),
                available: Some(protobuf::ExecutorResource {
                    resource: Some(r.available),
                }),
            })
            .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serde::scheduler::PartitionId;
    use prost::Message;

    /// A location as the scheduler holds it: full executor metadata.
    fn location() -> PartitionLocation {
        PartitionLocation {
            map_partition_id: 7,
            partition_id: PartitionId {
                job_id: "0b8b2a1e-1f6a-4c3d-9f2e-2b7c5d4e6f10".into(),
                stage_id: 3,
                partition_id: 42,
            },
            executor_meta: ExecutorMetadata {
                id: "4d1c9b7a-5e2f-4a3b-8c6d-7e9f0a1b2c3d".to_string(),
                host: "ballista-executor-17.default.svc".to_string(),
                port: 50051,
                grpc_port: 50052,
                specification: ExecutorSpecification { vcores: 8 },
                os_info: ExecutorOperatingSystemSpecification {
                    system_name: "Ubuntu".to_string(),
                    kernel_ver: "Linux 6.8.0-1021-aws".to_string(),
                    os_ver: "24.04".to_string(),
                    os_ver_long: "Linux (Ubuntu 24.04.1 LTS)".to_string(),
                    ..Default::default()
                },
            },
            partition_stats: PartitionStats::new(Some(1), Some(1), Some(1)),
            file_id: Some(11),
            is_sort_shuffle: true,
        }
    }

    /// The address survives; the executor's inventory does not.
    #[test]
    fn encodes_address_only() {
        let meta = TryInto::<protobuf::PartitionLocation>::try_into(location())
            .unwrap()
            .executor_meta
            .unwrap();

        assert_eq!(meta.id, "4d1c9b7a-5e2f-4a3b-8c6d-7e9f0a1b2c3d");
        assert_eq!(meta.host, "ballista-executor-17.default.svc");
        assert_eq!((meta.port, meta.grpc_port), (50051, 50052));
        assert!(meta.specification.is_none());
        assert!(meta.os_info.is_none());
    }

    /// Decoding survives the omission rather than panicking on it.
    #[test]
    fn omitted_inventory_decodes_to_defaults() {
        let encoded: protobuf::PartitionLocation = location().try_into().unwrap();
        let decoded: PartitionLocation = encoded.try_into().unwrap();

        assert_eq!(
            decoded.executor_meta.host,
            "ballista-executor-17.default.svc"
        );
        assert_eq!(
            decoded.executor_meta.specification,
            ExecutorSpecification::default()
        );
        assert_eq!(
            decoded.executor_meta.os_info,
            ExecutorOperatingSystemSpecification::default()
        );
    }

    /// The saving is what justifies the change. The fixture's OS strings are
    /// shorter than a real `kernel_long_version()`, so this understates it.
    #[test]
    fn dropping_inventory_shrinks_a_location() {
        let slim: protobuf::PartitionLocation = location().try_into().unwrap();
        let mut fat = slim.clone();
        fat.executor_meta = Some(location().executor_meta.into());

        let (slim_len, fat_len) = (slim.encoded_len(), fat.encoded_len());
        assert!(
            slim_len * 3 < fat_len * 2,
            "expected under two thirds of {fat_len} bytes, got {slim_len}"
        );
    }
}
