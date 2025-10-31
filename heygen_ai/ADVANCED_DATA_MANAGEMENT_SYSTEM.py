#!/usr/bin/env python3
"""
📊 HeyGen AI - Advanced Data Management System
=============================================

This module implements a comprehensive data management system that provides
data ingestion, processing, storage, retrieval, and analytics capabilities
for the HeyGen AI system.
"""

import asyncio
import logging
import time
import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import queue
import hashlib
import secrets
import base64
import hmac
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import aiohttp
import asyncio
from aiohttp import web, WSMsgType
import ssl
import certifi
import pandas as pd
import sqlite3
import redis
import pickle
from collections import defaultdict
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import zlib
import gzip
import bz2
import lzma
from pathlib import Path
import os
import shutil
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(str, Enum):
    """Data types"""
    TEXT = "text"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    TIME_SERIES = "time_series"
    GEOSPATIAL = "geospatial"

class DataSource(str, Enum):
    """Data sources"""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    SENSOR = "sensor"
    USER_INPUT = "user_input"
    GENERATED = "generated"
    EXTERNAL = "external"

class DataStatus(str, Enum):
    """Data status"""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"
    DELETED = "deleted"

class CompressionType(str, Enum):
    """Compression types"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    ZLIB = "zlib"
    CUSTOM = "custom"

@dataclass
class DataRecord:
    """Data record representation"""
    record_id: str
    data_type: DataType
    source: DataSource
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: DataStatus = DataStatus.PENDING
    size_bytes: int = 0
    checksum: str = ""
    tags: List[str] = field(default_factory=list)
    version: int = 1

@dataclass
class DataSchema:
    """Data schema representation"""
    schema_id: str
    name: str
    description: str
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

@dataclass
class DataPipeline:
    """Data pipeline representation"""
    pipeline_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    input_schema: Optional[DataSchema] = None
    output_schema: Optional[DataSchema] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "inactive"
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataIngestionEngine:
    """Advanced data ingestion engine"""
    
    def __init__(self):
        self.ingestion_queue: queue.Queue = queue.Queue()
        self.processing_threads: List[threading.Thread] = []
        self.initialized = False
    
    async def initialize(self):
        """Initialize data ingestion engine"""
        self.initialized = True
        logger.info("✅ Data Ingestion Engine initialized")
    
    async def ingest_data(self, data: Any, data_type: DataType, 
                         source: DataSource, metadata: Dict[str, Any] = None) -> str:
        """Ingest data into the system"""
        if not self.initialized:
            raise RuntimeError("Data ingestion engine not initialized")
        
        try:
            # Create data record
            record_id = str(uuid.uuid4())
            
            # Calculate size and checksum
            if isinstance(data, str):
                size_bytes = len(data.encode('utf-8'))
                checksum = hashlib.md5(data.encode('utf-8')).hexdigest()
            elif isinstance(data, (bytes, bytearray)):
                size_bytes = len(data)
                checksum = hashlib.md5(data).hexdigest()
            else:
                # Serialize for size calculation
                serialized = pickle.dumps(data)
                size_bytes = len(serialized)
                checksum = hashlib.md5(serialized).hexdigest()
            
            record = DataRecord(
                record_id=record_id,
                data_type=data_type,
                source=source,
                content=data,
                metadata=metadata or {},
                size_bytes=size_bytes,
                checksum=checksum
            )
            
            # Add to ingestion queue
            self.ingestion_queue.put(record)
            
            logger.info(f"✅ Data ingested: {record_id} ({data_type.value})")
            return record_id
            
        except Exception as e:
            logger.error(f"❌ Data ingestion failed: {e}")
            raise
    
    async def ingest_from_file(self, file_path: str, data_type: DataType,
                              metadata: Dict[str, Any] = None) -> str:
        """Ingest data from file"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            return await self.ingest_data(data, data_type, DataSource.FILE, metadata)
            
        except Exception as e:
            logger.error(f"❌ File ingestion failed: {e}")
            raise
    
    async def ingest_from_api(self, api_url: str, data_type: DataType,
                             headers: Dict[str, str] = None,
                             metadata: Dict[str, Any] = None) -> str:
        """Ingest data from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers) as response:
                    data = await response.text()
            
            return await self.ingest_data(data, data_type, DataSource.API, metadata)
            
        except Exception as e:
            logger.error(f"❌ API ingestion failed: {e}")
            raise

class DataStorageEngine:
    """Advanced data storage engine"""
    
    def __init__(self):
        self.storage_backends: Dict[str, Any] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize data storage engine"""
        try:
            # Initialize SQLite for metadata
            self.storage_backends['sqlite'] = sqlite3.connect(':memory:')
            self._create_metadata_tables()
            
            # Initialize Redis for caching
            try:
                self.storage_backends['redis'] = redis.Redis(host='localhost', port=6379, db=0)
                self.storage_backends['redis'].ping()
            except:
                logger.warning("Redis not available, using memory storage")
                self.storage_backends['redis'] = {}
            
            self.initialized = True
            logger.info("✅ Data Storage Engine initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize storage engine: {e}")
            raise
    
    def _create_metadata_tables(self):
        """Create metadata tables"""
        cursor = self.storage_backends['sqlite'].cursor()
        
        # Data records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_records (
                record_id TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                size_bytes INTEGER,
                checksum TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Data schemas table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_schemas (
                schema_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                fields TEXT,
                constraints TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                version INTEGER
            )
        ''')
        
        # Data pipelines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_pipelines (
                pipeline_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                steps TEXT,
                input_schema_id TEXT,
                output_schema_id TEXT,
                status TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        self.storage_backends['sqlite'].commit()
    
    async def store_record(self, record: DataRecord) -> bool:
        """Store data record"""
        if not self.initialized:
            return False
        
        try:
            # Store metadata in SQLite
            cursor = self.storage_backends['sqlite'].cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO data_records 
                (record_id, data_type, source, status, size_bytes, checksum, 
                 created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.record_id,
                record.data_type.value,
                record.source.value,
                record.status.value,
                record.size_bytes,
                record.checksum,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                json.dumps(record.metadata)
            ))
            self.storage_backends['sqlite'].commit()
            
            # Store data in Redis cache
            if isinstance(self.storage_backends['redis'], redis.Redis):
                self.storage_backends['redis'].setex(
                    f"data:{record.record_id}",
                    3600,  # 1 hour TTL
                    pickle.dumps(record)
                )
            else:
                # Memory storage
                self.storage_backends['redis'][f"data:{record.record_id}"] = record
            
            logger.info(f"✅ Record stored: {record.record_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store record: {e}")
            return False
    
    async def retrieve_record(self, record_id: str) -> Optional[DataRecord]:
        """Retrieve data record"""
        if not self.initialized:
            return None
        
        try:
            # Try cache first
            if isinstance(self.storage_backends['redis'], redis.Redis):
                cached_data = self.storage_backends['redis'].get(f"data:{record_id}")
                if cached_data:
                    return pickle.loads(cached_data)
            else:
                # Memory storage
                if f"data:{record_id}" in self.storage_backends['redis']:
                    return self.storage_backends['redis'][f"data:{record_id}"]
            
            # Get from metadata
            cursor = self.storage_backends['sqlite'].cursor()
            cursor.execute('SELECT * FROM data_records WHERE record_id = ?', (record_id,))
            row = cursor.fetchone()
            
            if row:
                # Reconstruct record
                record = DataRecord(
                    record_id=row[0],
                    data_type=DataType(row[1]),
                    source=DataSource(row[2]),
                    content=None,  # Content not stored in metadata
                    status=DataStatus(row[3]),
                    size_bytes=row[4],
                    checksum=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    metadata=json.loads(row[8]) if row[8] else {}
                )
                return record
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve record: {e}")
            return None
    
    async def list_records(self, data_type: DataType = None, 
                          source: DataSource = None,
                          status: DataStatus = None,
                          limit: int = 100) -> List[DataRecord]:
        """List data records with filters"""
        if not self.initialized:
            return []
        
        try:
            cursor = self.storage_backends['sqlite'].cursor()
            
            query = "SELECT * FROM data_records WHERE 1=1"
            params = []
            
            if data_type:
                query += " AND data_type = ?"
                params.append(data_type.value)
            
            if source:
                query += " AND source = ?"
                params.append(source.value)
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                record = DataRecord(
                    record_id=row[0],
                    data_type=DataType(row[1]),
                    source=DataSource(row[2]),
                    content=None,
                    status=DataStatus(row[3]),
                    size_bytes=row[4],
                    checksum=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    metadata=json.loads(row[8]) if row[8] else {}
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"❌ Failed to list records: {e}")
            return []

class DataProcessingEngine:
    """Advanced data processing engine"""
    
    def __init__(self):
        self.processing_pipelines: Dict[str, DataPipeline] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize data processing engine"""
        self.initialized = True
        logger.info("✅ Data Processing Engine initialized")
    
    async def create_pipeline(self, pipeline: DataPipeline) -> bool:
        """Create a data processing pipeline"""
        if not self.initialized:
            return False
        
        try:
            self.processing_pipelines[pipeline.pipeline_id] = pipeline
            logger.info(f"✅ Pipeline created: {pipeline.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create pipeline: {e}")
            return False
    
    async def process_data(self, record: DataRecord, pipeline_id: str) -> Optional[DataRecord]:
        """Process data through a pipeline"""
        if not self.initialized or pipeline_id not in self.processing_pipelines:
            return None
        
        try:
            pipeline = self.processing_pipelines[pipeline_id]
            
            # Create processed record
            processed_record = DataRecord(
                record_id=str(uuid.uuid4()),
                data_type=record.data_type,
                source=record.source,
                content=record.content,
                metadata=record.metadata.copy(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=DataStatus.PROCESSING
            )
            
            # Apply pipeline steps
            for step in pipeline.steps:
                processed_record = await self._apply_processing_step(processed_record, step)
            
            processed_record.status = DataStatus.READY
            processed_record.updated_at = datetime.now()
            
            logger.info(f"✅ Data processed: {processed_record.record_id}")
            return processed_record
            
        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            return None
    
    async def _apply_processing_step(self, record: DataRecord, step: Dict[str, Any]) -> DataRecord:
        """Apply a processing step to a record"""
        step_type = step.get('type', 'unknown')
        
        if step_type == 'transform':
            # Apply transformation
            transform_func = step.get('function')
            if transform_func and callable(transform_func):
                record.content = transform_func(record.content)
        
        elif step_type == 'filter':
            # Apply filter
            filter_func = step.get('function')
            if filter_func and callable(filter_func):
                if not filter_func(record.content):
                    record.status = DataStatus.ERROR
                    record.metadata['error'] = 'Filtered out'
        
        elif step_type == 'validate':
            # Apply validation
            validation_func = step.get('function')
            if validation_func and callable(validation_func):
                if not validation_func(record.content):
                    record.status = DataStatus.ERROR
                    record.metadata['error'] = 'Validation failed'
        
        elif step_type == 'enrich':
            # Apply enrichment
            enrich_func = step.get('function')
            if enrich_func and callable(enrich_func):
                enrichment_data = enrich_func(record.content)
                record.metadata.update(enrichment_data)
        
        return record

class DataAnalyticsEngine:
    """Advanced data analytics engine"""
    
    def __init__(self):
        self.analytics_cache: Dict[str, Any] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize data analytics engine"""
        self.initialized = True
        logger.info("✅ Data Analytics Engine initialized")
    
    async def analyze_data(self, records: List[DataRecord]) -> Dict[str, Any]:
        """Analyze data records"""
        if not self.initialized:
            return {}
        
        try:
            analysis = {
                'total_records': len(records),
                'data_types': {},
                'sources': {},
                'statuses': {},
                'size_distribution': {},
                'temporal_distribution': {},
                'quality_metrics': {}
            }
            
            # Analyze data types
            for record in records:
                data_type = record.data_type.value
                analysis['data_types'][data_type] = analysis['data_types'].get(data_type, 0) + 1
            
            # Analyze sources
            for record in records:
                source = record.source.value
                analysis['sources'][source] = analysis['sources'].get(source, 0) + 1
            
            # Analyze statuses
            for record in records:
                status = record.status.value
                analysis['statuses'][status] = analysis['statuses'].get(status, 0) + 1
            
            # Analyze size distribution
            sizes = [record.size_bytes for record in records if record.size_bytes > 0]
            if sizes:
                analysis['size_distribution'] = {
                    'min': min(sizes),
                    'max': max(sizes),
                    'mean': sum(sizes) / len(sizes),
                    'median': sorted(sizes)[len(sizes) // 2]
                }
            
            # Analyze temporal distribution
            dates = [record.created_at.date() for record in records]
            if dates:
                date_counts = {}
                for date in dates:
                    date_counts[str(date)] = date_counts.get(str(date), 0) + 1
                analysis['temporal_distribution'] = date_counts
            
            # Quality metrics
            total_records = len(records)
            ready_records = len([r for r in records if r.status == DataStatus.READY])
            error_records = len([r for r in records if r.status == DataStatus.ERROR])
            
            analysis['quality_metrics'] = {
                'ready_rate': (ready_records / total_records * 100) if total_records > 0 else 0,
                'error_rate': (error_records / total_records * 100) if total_records > 0 else 0,
                'average_size': sum(sizes) / len(sizes) if sizes else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Data analysis failed: {e}")
            return {}
    
    async def generate_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from analysis"""
        if not self.initialized:
            return []
        
        insights = []
        
        try:
            # Data type insights
            data_types = analysis.get('data_types', {})
            if data_types:
                most_common_type = max(data_types, key=data_types.get)
                insights.append(f"Most common data type: {most_common_type} ({data_types[most_common_type]} records)")
            
            # Source insights
            sources = analysis.get('sources', {})
            if sources:
                most_common_source = max(sources, key=sources.get)
                insights.append(f"Primary data source: {most_common_source} ({sources[most_common_source]} records)")
            
            # Quality insights
            quality_metrics = analysis.get('quality_metrics', {})
            ready_rate = quality_metrics.get('ready_rate', 0)
            error_rate = quality_metrics.get('error_rate', 0)
            
            if ready_rate > 90:
                insights.append("Excellent data quality: >90% records are ready")
            elif ready_rate > 70:
                insights.append("Good data quality: >70% records are ready")
            else:
                insights.append("Data quality needs improvement: <70% records are ready")
            
            if error_rate > 10:
                insights.append(f"High error rate detected: {error_rate:.1f}% records have errors")
            
            # Size insights
            size_dist = analysis.get('size_distribution', {})
            if size_dist:
                avg_size = size_dist.get('mean', 0)
                if avg_size > 1024 * 1024:  # > 1MB
                    insights.append(f"Large average record size: {avg_size / 1024 / 1024:.1f}MB")
                elif avg_size < 1024:  # < 1KB
                    insights.append(f"Small average record size: {avg_size:.1f} bytes")
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Insight generation failed: {e}")
            return []

class AdvancedDataManagementSystem:
    """Main data management system"""
    
    def __init__(self):
        self.ingestion_engine = DataIngestionEngine()
        self.storage_engine = DataStorageEngine()
        self.processing_engine = DataProcessingEngine()
        self.analytics_engine = DataAnalyticsEngine()
        self.initialized = False
    
    async def initialize(self):
        """Initialize data management system"""
        try:
            logger.info("📊 Initializing Advanced Data Management System...")
            
            # Initialize components
            await self.ingestion_engine.initialize()
            await self.storage_engine.initialize()
            await self.processing_engine.initialize()
            await self.analytics_engine.initialize()
            
            # Create default pipelines
            await self._create_default_pipelines()
            
            self.initialized = True
            logger.info("✅ Advanced Data Management System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize data management system: {e}")
            raise
    
    async def _create_default_pipelines(self):
        """Create default data processing pipelines"""
        # Text processing pipeline
        text_pipeline = DataPipeline(
            pipeline_id="text_processing",
            name="Text Processing Pipeline",
            description="Processes text data with cleaning and validation",
            steps=[
                {
                    'type': 'transform',
                    'function': lambda x: x.strip().lower() if isinstance(x, str) else x
                },
                {
                    'type': 'validate',
                    'function': lambda x: len(x) > 0 if isinstance(x, str) else True
                }
            ]
        )
        
        await self.processing_engine.create_pipeline(text_pipeline)
        
        # Numerical processing pipeline
        numerical_pipeline = DataPipeline(
            pipeline_id="numerical_processing",
            name="Numerical Processing Pipeline",
            description="Processes numerical data with validation",
            steps=[
                {
                    'type': 'transform',
                    'function': lambda x: float(x) if isinstance(x, (int, str)) else x
                },
                {
                    'type': 'validate',
                    'function': lambda x: isinstance(x, (int, float)) and not np.isnan(x) if isinstance(x, (int, float)) else True
                }
            ]
        )
        
        await self.processing_engine.create_pipeline(numerical_pipeline)
    
    async def ingest_data(self, data: Any, data_type: DataType, 
                         source: DataSource, metadata: Dict[str, Any] = None) -> str:
        """Ingest data into the system"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        # Ingest data
        record_id = await self.ingestion_engine.ingest_data(data, data_type, source, metadata)
        
        # Store record
        record = DataRecord(
            record_id=record_id,
            data_type=data_type,
            source=source,
            content=data,
            metadata=metadata or {},
            status=DataStatus.PENDING
        )
        
        await self.storage_engine.store_record(record)
        
        return record_id
    
    async def process_data(self, record_id: str, pipeline_id: str) -> Optional[str]:
        """Process data through a pipeline"""
        if not self.initialized:
            return None
        
        # Retrieve record
        record = await self.storage_engine.retrieve_record(record_id)
        if not record:
            return None
        
        # Process data
        processed_record = await self.processing_engine.process_data(record, pipeline_id)
        if not processed_record:
            return None
        
        # Store processed record
        await self.storage_engine.store_record(processed_record)
        
        return processed_record.record_id
    
    async def analyze_data(self, data_type: DataType = None, 
                          source: DataSource = None,
                          limit: int = 1000) -> Dict[str, Any]:
        """Analyze data in the system"""
        if not self.initialized:
            return {}
        
        # Get records
        records = await self.storage_engine.list_records(data_type, source, limit=limit)
        
        # Analyze data
        analysis = await self.analytics_engine.analyze_data(records)
        
        # Generate insights
        insights = await self.analytics_engine.generate_insights(analysis)
        analysis['insights'] = insights
        
        return analysis
    
    async def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        if not self.initialized:
            return {}
        
        try:
            # Get all records
            all_records = await self.storage_engine.list_records(limit=10000)
            
            # Analyze data
            analysis = await self.analytics_engine.analyze_data(all_records)
            
            # Get pipeline information
            pipeline_count = len(self.processing_engine.processing_pipelines)
            
            return {
                'total_records': len(all_records),
                'data_analysis': analysis,
                'pipeline_count': pipeline_count,
                'system_status': 'operational',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get data summary: {e}")
            return {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'initialized': self.initialized,
            'ingestion_engine_ready': self.ingestion_engine.initialized,
            'storage_engine_ready': self.storage_engine.initialized,
            'processing_engine_ready': self.processing_engine.initialized,
            'analytics_engine_ready': self.analytics_engine.initialized,
            'timestamp': datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown data management system"""
        self.initialized = False
        logger.info("✅ Advanced Data Management System shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the advanced data management system"""
    print("📊 HeyGen AI - Advanced Data Management System Demo")
    print("=" * 70)
    
    # Initialize system
    data_system = AdvancedDataManagementSystem()
    
    try:
        # Initialize the system
        print("\n🚀 Initializing Advanced Data Management System...")
        await data_system.initialize()
        print("✅ Advanced Data Management System initialized successfully")
        
        # Get system status
        print("\n📊 System Status:")
        status = await data_system.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Ingest some sample data
        print("\n📥 Ingesting Sample Data...")
        
        # Text data
        text_record_id = await data_system.ingest_data(
            "Hello, this is a sample text for testing the data management system.",
            DataType.TEXT,
            DataSource.GENERATED,
            {"language": "en", "category": "sample"}
        )
        print(f"  Text data ingested: {text_record_id}")
        
        # Numerical data
        numerical_record_id = await data_system.ingest_data(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            DataType.NUMERICAL,
            DataSource.GENERATED,
            {"range": "1-10", "count": 10}
        )
        print(f"  Numerical data ingested: {numerical_record_id}")
        
        # Structured data
        structured_record_id = await data_system.ingest_data(
            {"name": "John Doe", "age": 30, "city": "New York"},
            DataType.STRUCTURED,
            DataSource.GENERATED,
            {"type": "person", "version": "1.0"}
        )
        print(f"  Structured data ingested: {structured_record_id}")
        
        # Process data through pipelines
        print("\n⚙️ Processing Data Through Pipelines...")
        
        # Process text data
        processed_text_id = await data_system.process_data(text_record_id, "text_processing")
        if processed_text_id:
            print(f"  Text data processed: {processed_text_id}")
        
        # Process numerical data
        processed_numerical_id = await data_system.process_data(numerical_record_id, "numerical_processing")
        if processed_numerical_id:
            print(f"  Numerical data processed: {processed_numerical_id}")
        
        # Analyze data
        print("\n📈 Analyzing Data...")
        
        analysis = await data_system.analyze_data(limit=100)
        
        print(f"  Total Records: {analysis.get('total_records', 0)}")
        print(f"  Data Types: {analysis.get('data_types', {})}")
        print(f"  Sources: {analysis.get('sources', {})}")
        print(f"  Statuses: {analysis.get('statuses', {})}")
        
        # Quality metrics
        quality_metrics = analysis.get('quality_metrics', {})
        print(f"  Ready Rate: {quality_metrics.get('ready_rate', 0):.1f}%")
        print(f"  Error Rate: {quality_metrics.get('error_rate', 0):.1f}%")
        print(f"  Average Size: {quality_metrics.get('average_size', 0):.1f} bytes")
        
        # Insights
        insights = analysis.get('insights', [])
        if insights:
            print(f"\n  Insights:")
            for insight in insights:
                print(f"    - {insight}")
        
        # Get comprehensive summary
        print("\n📋 Data Summary:")
        summary = await data_system.get_data_summary()
        
        print(f"  Total Records: {summary.get('total_records', 0)}")
        print(f"  Pipeline Count: {summary.get('pipeline_count', 0)}")
        print(f"  System Status: {summary.get('system_status', 'unknown')}")
        
        # Data analysis details
        data_analysis = summary.get('data_analysis', {})
        if data_analysis:
            print(f"\n  Data Analysis:")
            print(f"    Data Types: {data_analysis.get('data_types', {})}")
            print(f"    Sources: {data_analysis.get('sources', {})}")
            print(f"    Statuses: {data_analysis.get('statuses', {})}")
            
            size_dist = data_analysis.get('size_distribution', {})
            if size_dist:
                print(f"    Size Distribution:")
                print(f"      Min: {size_dist.get('min', 0)} bytes")
                print(f"      Max: {size_dist.get('max', 0)} bytes")
                print(f"      Mean: {size_dist.get('mean', 0):.1f} bytes")
                print(f"      Median: {size_dist.get('median', 0)} bytes")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Shutdown
        await data_system.shutdown()
        print("\n✅ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())


