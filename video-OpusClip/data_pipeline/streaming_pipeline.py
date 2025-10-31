#!/usr/bin/env python3
"""
Real-Time Data Pipeline System

Advanced data pipeline with:
- Real-time data streaming
- Data transformation and ETL
- Data validation and quality
- Data warehousing integration
- Stream processing
- Data lineage tracking
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Type
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

logger = structlog.get_logger("data_pipeline")

# =============================================================================
# DATA PIPELINE MODELS
# =============================================================================

class DataType(Enum):
    """Data types for pipeline processing."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    JSON = "json"
    BINARY = "binary"

class ProcessingMode(Enum):
    """Data processing modes."""
    BATCH = "batch"
    STREAMING = "streaming"
    MICRO_BATCH = "micro_batch"
    REAL_TIME = "real_time"

class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class DataRecord:
    """Data record structure."""
    record_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    quality_score: float
    schema_version: str
    source: str
    
    def __post_init__(self):
        if not self.record_id:
            self.record_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "schema_version": self.schema_version,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataRecord:
        """Create from dictionary."""
        return cls(
            record_id=data["record_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            metadata=data["metadata"],
            quality_score=data["quality_score"],
            schema_version=data["schema_version"],
            source=data["source"]
        )

@dataclass
class DataSchema:
    """Data schema definition."""
    schema_id: str
    name: str
    version: str
    fields: Dict[str, Dict[str, Any]]
    constraints: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_id": self.schema_id,
            "name": self.name,
            "version": self.version,
            "fields": self.fields,
            "constraints": self.constraints,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class PipelineStage:
    """Pipeline stage definition."""
    stage_id: str
    name: str
    stage_type: str
    processor: Callable[[DataRecord], Awaitable[DataRecord]]
    input_schema: Optional[DataSchema]
    output_schema: Optional[DataSchema]
    parallel_workers: int = 1
    timeout: int = 30
    retry_count: int = 3
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "stage_type": self.stage_type,
            "input_schema": self.input_schema.to_dict() if self.input_schema else None,
            "output_schema": self.output_schema.to_dict() if self.output_schema else None,
            "parallel_workers": self.parallel_workers,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "enabled": self.enabled
        }

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    pipeline_id: str
    name: str
    description: str
    processing_mode: ProcessingMode
    stages: List[PipelineStage]
    input_sources: List[str]
    output_destinations: List[str]
    batch_size: int = 1000
    window_size: int = 60  # seconds
    checkpoint_interval: int = 300  # seconds
    max_parallelism: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "processing_mode": self.processing_mode.value,
            "stages": [stage.to_dict() for stage in self.stages],
            "input_sources": self.input_sources,
            "output_destinations": self.output_destinations,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "checkpoint_interval": self.checkpoint_interval,
            "max_parallelism": self.max_parallelism
        }

# =============================================================================
# DATA PROCESSORS
# =============================================================================

class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    async def process(self, record: DataRecord) -> DataRecord:
        """Process a data record."""
        pass
    
    @abstractmethod
    def validate_schema(self, record: DataRecord, schema: DataSchema) -> bool:
        """Validate record against schema."""
        pass

class DataValidator(DataProcessor):
    """Data validation processor."""
    
    def __init__(self, schema: DataSchema):
        self.schema = schema
    
    async def process(self, record: DataRecord) -> DataRecord:
        """Validate data record."""
        if not self.validate_schema(record, self.schema):
            record.metadata['validation_errors'] = self._get_validation_errors(record)
            record.quality_score = 0.0
        else:
            record.quality_score = 1.0
        
        return record
    
    def validate_schema(self, record: DataRecord, schema: DataSchema) -> bool:
        """Validate record against schema."""
        for field_name, field_config in schema.fields.items():
            if field_name not in record.data:
                if field_config.get('required', False):
                    return False
                continue
            
            field_value = record.data[field_name]
            field_type = field_config.get('type', 'string')
            
            if not self._validate_field_type(field_value, field_type):
                return False
            
            # Validate constraints
            if not self._validate_constraints(field_value, field_config.get('constraints', {})):
                return False
        
        return True
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type."""
        type_validators = {
            'string': lambda v: isinstance(v, str),
            'integer': lambda v: isinstance(v, int),
            'float': lambda v: isinstance(v, (int, float)),
            'boolean': lambda v: isinstance(v, bool),
            'datetime': lambda v: isinstance(v, (str, datetime)),
            'json': lambda v: isinstance(v, (dict, list)),
            'binary': lambda v: isinstance(v, bytes)
        }
        
        validator = type_validators.get(expected_type)
        return validator(value) if validator else True
    
    def _validate_constraints(self, value: Any, constraints: Dict[str, Any]) -> bool:
        """Validate field constraints."""
        # Min/Max length for strings
        if isinstance(value, str):
            if 'min_length' in constraints and len(value) < constraints['min_length']:
                return False
            if 'max_length' in constraints and len(value) > constraints['max_length']:
                return False
        
        # Min/Max values for numbers
        if isinstance(value, (int, float)):
            if 'min_value' in constraints and value < constraints['min_value']:
                return False
            if 'max_value' in constraints and value > constraints['max_value']:
                return False
        
        # Pattern matching for strings
        if isinstance(value, str) and 'pattern' in constraints:
            import re
            if not re.match(constraints['pattern'], value):
                return False
        
        return True
    
    def _get_validation_errors(self, record: DataRecord) -> List[str]:
        """Get validation errors for record."""
        errors = []
        
        for field_name, field_config in self.schema.fields.items():
            if field_name not in record.data:
                if field_config.get('required', False):
                    errors.append(f"Required field '{field_name}' is missing")
                continue
            
            field_value = record.data[field_name]
            field_type = field_config.get('type', 'string')
            
            if not self._validate_field_type(field_value, field_type):
                errors.append(f"Field '{field_name}' has invalid type. Expected {field_type}, got {type(field_value).__name__}")
            
            if not self._validate_constraints(field_value, field_config.get('constraints', {})):
                errors.append(f"Field '{field_name}' violates constraints")
        
        return errors

class DataTransformer(DataProcessor):
    """Data transformation processor."""
    
    def __init__(self, transformation_rules: Dict[str, Callable]):
        self.transformation_rules = transformation_rules
    
    async def process(self, record: DataRecord) -> DataRecord:
        """Transform data record."""
        transformed_data = {}
        
        for field_name, value in record.data.items():
            if field_name in self.transformation_rules:
                try:
                    transformed_value = await self.transformation_rules[field_name](value)
                    transformed_data[field_name] = transformed_value
                except Exception as e:
                    logger.error(f"Transformation error for field {field_name}", error=str(e))
                    transformed_data[field_name] = value
            else:
                transformed_data[field_name] = value
        
        record.data = transformed_data
        return record
    
    def validate_schema(self, record: DataRecord, schema: DataSchema) -> bool:
        """Validate record against schema."""
        return True  # Transformers don't validate schemas

class DataEnricher(DataProcessor):
    """Data enrichment processor."""
    
    def __init__(self, enrichment_sources: Dict[str, Callable]):
        self.enrichment_sources = enrichment_sources
    
    async def process(self, record: DataRecord) -> DataRecord:
        """Enrich data record."""
        for field_name, enrichment_func in self.enrichment_sources.items():
            try:
                enriched_value = await enrichment_func(record.data)
                record.data[field_name] = enriched_value
            except Exception as e:
                logger.error(f"Enrichment error for field {field_name}", error=str(e))
        
        return record
    
    def validate_schema(self, record: DataRecord, schema: DataSchema) -> bool:
        """Validate record against schema."""
        return True  # Enrichers don't validate schemas

class DataAggregator(DataProcessor):
    """Data aggregation processor."""
    
    def __init__(self, aggregation_config: Dict[str, Any]):
        self.aggregation_config = aggregation_config
        self.aggregation_buffer: Dict[str, List[DataRecord]] = defaultdict(list)
        self.last_aggregation = time.time()
    
    async def process(self, record: DataRecord) -> DataRecord:
        """Aggregate data records."""
        # Add to buffer
        buffer_key = self._get_buffer_key(record)
        self.aggregation_buffer[buffer_key].append(record)
        
        # Check if aggregation should be triggered
        if self._should_aggregate():
            aggregated_records = await self._perform_aggregation()
            # Return the first aggregated record (others will be handled separately)
            return aggregated_records[0] if aggregated_records else record
        
        return record
    
    def _get_buffer_key(self, record: DataRecord) -> str:
        """Get buffer key for record."""
        key_fields = self.aggregation_config.get('key_fields', [])
        if not key_fields:
            return 'default'
        
        key_values = [str(record.data.get(field, '')) for field in key_fields]
        return ':'.join(key_values)
    
    def _should_aggregate(self) -> bool:
        """Check if aggregation should be triggered."""
        current_time = time.time()
        window_size = self.aggregation_config.get('window_size', 60)
        
        return (current_time - self.last_aggregation) >= window_size
    
    async def _perform_aggregation(self) -> List[DataRecord]:
        """Perform data aggregation."""
        aggregated_records = []
        
        for buffer_key, records in self.aggregation_buffer.items():
            if not records:
                continue
            
            # Perform aggregation
            aggregated_data = {}
            aggregation_functions = self.aggregation_config.get('functions', {})
            
            for field_name, func_name in aggregation_functions.items():
                values = [record.data.get(field_name) for record in records if field_name in record.data]
                
                if values:
                    if func_name == 'sum':
                        aggregated_data[field_name] = sum(values)
                    elif func_name == 'avg':
                        aggregated_data[field_name] = sum(values) / len(values)
                    elif func_name == 'count':
                        aggregated_data[field_name] = len(values)
                    elif func_name == 'min':
                        aggregated_data[field_name] = min(values)
                    elif func_name == 'max':
                        aggregated_data[field_name] = max(values)
            
            # Create aggregated record
            if aggregated_data:
                aggregated_record = DataRecord(
                    record_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    data=aggregated_data,
                    metadata={'aggregation_key': buffer_key, 'record_count': len(records)},
                    quality_score=1.0,
                    schema_version='1.0',
                    source='aggregator'
                )
                aggregated_records.append(aggregated_record)
        
        # Clear buffer
        self.aggregation_buffer.clear()
        self.last_aggregation = time.time()
        
        return aggregated_records
    
    def validate_schema(self, record: DataRecord, schema: DataSchema) -> bool:
        """Validate record against schema."""
        return True  # Aggregators don't validate schemas

# =============================================================================
# STREAMING DATA PIPELINE
# =============================================================================

class StreamingDataPipeline:
    """Real-time streaming data pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.is_running = False
        self.processing_tasks: List[asyncio.Task] = []
        self.data_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.output_queues: Dict[str, asyncio.Queue] = {}
        
        # Statistics
        self.stats = {
            'records_processed': 0,
            'records_failed': 0,
            'processing_time': 0.0,
            'throughput': 0.0,
            'last_checkpoint': None
        }
        
        # Data lineage tracking
        self.data_lineage: Dict[str, List[str]] = defaultdict(list)
        
        # Quality metrics
        self.quality_metrics = {
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'average_quality_score': 0.0
        }
    
    async def start(self) -> None:
        """Start the data pipeline."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Create output queues
        for destination in self.config.output_destinations:
            self.output_queues[destination] = asyncio.Queue(maxsize=10000)
        
        # Start processing tasks
        for stage in self.config.stages:
            if stage.enabled:
                for worker_id in range(stage.parallel_workers):
                    task = asyncio.create_task(
                        self._process_stage(stage, worker_id)
                    )
                    self.processing_tasks.append(task)
        
        # Start checkpoint task
        checkpoint_task = asyncio.create_task(self._checkpoint_task())
        self.processing_tasks.append(checkpoint_task)
        
        logger.info(
            "Data pipeline started",
            pipeline_id=self.config.pipeline_id,
            stages=len(self.config.stages),
            workers=sum(stage.parallel_workers for stage in self.config.stages)
        )
    
    async def stop(self) -> None:
        """Stop the data pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        logger.info("Data pipeline stopped", pipeline_id=self.config.pipeline_id)
    
    async def ingest_data(self, record: DataRecord) -> None:
        """Ingest data into the pipeline."""
        try:
            await self.data_queue.put(record)
            logger.debug("Data ingested", record_id=record.record_id)
        except asyncio.QueueFull:
            logger.error("Data queue is full, dropping record", record_id=record.record_id)
            raise RuntimeError("Data queue is full")
    
    async def _process_stage(self, stage: PipelineStage, worker_id: int) -> None:
        """Process data through a pipeline stage."""
        logger.info(
            "Stage worker started",
            stage_id=stage.stage_id,
            worker_id=worker_id
        )
        
        while self.is_running:
            try:
                # Get data from input queue
                if stage.stage_id == self.config.stages[0].stage_id:
                    # First stage gets data from main queue
                    record = await self.data_queue.get()
                else:
                    # Other stages get data from previous stage output
                    # This is simplified - in practice, you'd have proper stage-to-stage communication
                    record = await self.data_queue.get()
                
                # Process record
                start_time = time.time()
                
                try:
                    processed_record = await asyncio.wait_for(
                        stage.processor(record),
                        timeout=stage.timeout
                    )
                    
                    # Track data lineage
                    self.data_lineage[processed_record.record_id].append(stage.stage_id)
                    
                    # Update quality metrics
                    self._update_quality_metrics(processed_record)
                    
                    # Send to output
                    await self._send_to_output(processed_record)
                    
                    # Update statistics
                    processing_time = time.time() - start_time
                    self.stats['records_processed'] += 1
                    self.stats['processing_time'] += processing_time
                    
                except asyncio.TimeoutError:
                    logger.error(
                        "Stage processing timeout",
                        stage_id=stage.stage_id,
                        record_id=record.record_id,
                        timeout=stage.timeout
                    )
                    self.stats['records_failed'] += 1
                
                except Exception as e:
                    logger.error(
                        "Stage processing error",
                        stage_id=stage.stage_id,
                        record_id=record.record_id,
                        error=str(e)
                    )
                    self.stats['records_failed'] += 1
                
                # Mark task as done
                self.data_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Stage worker error", error=str(e))
                await asyncio.sleep(1)
    
    async def _send_to_output(self, record: DataRecord) -> None:
        """Send processed record to output destinations."""
        for destination in self.config.output_destinations:
            if destination in self.output_queues:
                try:
                    await self.output_queues[destination].put(record)
                except asyncio.QueueFull:
                    logger.error("Output queue full", destination=destination)
    
    async def _checkpoint_task(self) -> None:
        """Periodic checkpoint task."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.checkpoint_interval)
                
                # Save checkpoint
                checkpoint = {
                    'pipeline_id': self.config.pipeline_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'stats': self.stats,
                    'quality_metrics': self.quality_metrics
                }
                
                # In a real implementation, you'd save this to persistent storage
                logger.info("Checkpoint saved", checkpoint=checkpoint)
                self.stats['last_checkpoint'] = datetime.utcnow()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Checkpoint error", error=str(e))
    
    def _update_quality_metrics(self, record: DataRecord) -> None:
        """Update data quality metrics."""
        self.quality_metrics['total_records'] += 1
        
        if record.quality_score >= 0.8:
            self.quality_metrics['valid_records'] += 1
        else:
            self.quality_metrics['invalid_records'] += 1
        
        # Update average quality score
        total_records = self.quality_metrics['total_records']
        current_avg = self.quality_metrics['average_quality_score']
        self.quality_metrics['average_quality_score'] = (
            (current_avg * (total_records - 1) + record.quality_score) / total_records
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total_processed = self.stats['records_processed'] + self.stats['records_failed']
        
        return {
            **self.stats,
            'quality_metrics': self.quality_metrics,
            'data_lineage_count': len(self.data_lineage),
            'queue_sizes': {
                'input_queue': self.data_queue.qsize(),
                'output_queues': {
                    dest: queue.qsize() for dest, queue in self.output_queues.items()
                }
            },
            'throughput': self.stats['records_processed'] / max(1, self.stats['processing_time']),
            'success_rate': self.stats['records_processed'] / max(1, total_processed)
        }
    
    def get_data_lineage(self, record_id: str) -> List[str]:
        """Get data lineage for a record."""
        return self.data_lineage.get(record_id, [])

# =============================================================================
# DATA WAREHOUSE INTEGRATION
# =============================================================================

class DataWarehouseConnector:
    """Data warehouse connector for analytics."""
    
    def __init__(self, connection_config: Dict[str, Any]):
        self.connection_config = connection_config
        self.connection = None
        self.is_connected = False
    
    async def connect(self) -> None:
        """Connect to data warehouse."""
        # In a real implementation, you'd connect to your data warehouse
        # (e.g., BigQuery, Snowflake, Redshift, etc.)
        self.is_connected = True
        logger.info("Connected to data warehouse")
    
    async def disconnect(self) -> None:
        """Disconnect from data warehouse."""
        self.is_connected = False
        logger.info("Disconnected from data warehouse")
    
    async def write_data(self, table_name: str, records: List[DataRecord]) -> None:
        """Write data to warehouse table."""
        if not self.is_connected:
            raise RuntimeError("Not connected to data warehouse")
        
        # Convert records to warehouse format
        warehouse_data = []
        for record in records:
            warehouse_record = {
                'record_id': record.record_id,
                'timestamp': record.timestamp.isoformat(),
                'data': json.dumps(record.data),
                'metadata': json.dumps(record.metadata),
                'quality_score': record.quality_score,
                'schema_version': record.schema_version,
                'source': record.source
            }
            warehouse_data.append(warehouse_record)
        
        # In a real implementation, you'd write to the warehouse
        logger.info(
            "Data written to warehouse",
            table_name=table_name,
            record_count=len(records)
        )
    
    async def query_data(self, query: str) -> List[Dict[str, Any]]:
        """Query data from warehouse."""
        if not self.is_connected:
            raise RuntimeError("Not connected to data warehouse")
        
        # In a real implementation, you'd execute the query
        logger.info("Query executed", query=query)
        return []

# =============================================================================
# GLOBAL DATA PIPELINE INSTANCES
# =============================================================================

# Global data pipeline components
data_warehouse_connector = DataWarehouseConnector({})

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DataType',
    'ProcessingMode',
    'DataQuality',
    'DataRecord',
    'DataSchema',
    'PipelineStage',
    'PipelineConfig',
    'DataProcessor',
    'DataValidator',
    'DataTransformer',
    'DataEnricher',
    'DataAggregator',
    'StreamingDataPipeline',
    'DataWarehouseConnector',
    'data_warehouse_connector'
]





























