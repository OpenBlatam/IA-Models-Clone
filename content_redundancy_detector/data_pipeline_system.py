"""
Data Pipeline System for Advanced Data Processing
Sistema de Pipeline de Datos para procesamiento avanzado de datos ultra-optimizado
"""

import asyncio
import logging
import time
import json
import threading
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import sqlite3
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Etapas del pipeline"""
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    CLEAN = "clean"
    ENRICH = "enrich"
    AGGREGATE = "aggregate"
    FILTER = "filter"


class DataSource(Enum):
    """Fuentes de datos"""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    MESSAGE_QUEUE = "message_queue"
    CLOUD_STORAGE = "cloud_storage"
    IOT_DEVICE = "iot_device"
    WEBHOOK = "webhook"


class DataFormat(Enum):
    """Formatos de datos"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PARQUET = "parquet"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    BINARY = "binary"
    TEXT = "text"


class PipelineStatus(Enum):
    """Estados del pipeline"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DataPipeline:
    """Pipeline de datos"""
    id: str
    name: str
    description: str
    stages: List[Dict[str, Any]]
    status: PipelineStatus
    source: DataSource
    destination: DataSource
    data_format: DataFormat
    created_at: float
    last_run: Optional[float]
    next_run: Optional[float]
    schedule: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class PipelineExecution:
    """Ejecución de pipeline"""
    id: str
    pipeline_id: str
    status: PipelineStatus
    started_at: float
    completed_at: Optional[float]
    records_processed: int
    records_failed: int
    execution_time: float
    error_message: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class DataTransformation:
    """Transformación de datos"""
    id: str
    name: str
    type: str
    parameters: Dict[str, Any]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    created_at: float
    metadata: Dict[str, Any]


class DataExtractor:
    """Extractor de datos"""
    
    def __init__(self):
        self.extractors: Dict[DataSource, Callable] = {
            DataSource.FILE: self._extract_from_file,
            DataSource.DATABASE: self._extract_from_database,
            DataSource.API: self._extract_from_api,
            DataSource.STREAM: self._extract_from_stream,
            DataSource.MESSAGE_QUEUE: self._extract_from_message_queue,
            DataSource.CLOUD_STORAGE: self._extract_from_cloud_storage,
            DataSource.IOT_DEVICE: self._extract_from_iot_device,
            DataSource.WEBHOOK: self._extract_from_webhook
        }
    
    async def extract_data(self, source: DataSource, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer datos de la fuente"""
        try:
            extractor = self.extractors.get(source)
            if not extractor:
                raise ValueError(f"Unsupported data source: {source}")
            
            return await extractor(config)
            
        except Exception as e:
            logger.error(f"Error extracting data from {source}: {e}")
            raise
    
    async def _extract_from_file(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer datos de archivo"""
        file_path = config["file_path"]
        data_format = DataFormat(config.get("format", "json"))
        
        if data_format == DataFormat.JSON:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        elif data_format == DataFormat.CSV:
            data = []
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                reader = csv.DictReader(content.splitlines())
                for row in reader:
                    data.append(dict(row))
            return data
        else:
            raise ValueError(f"Unsupported file format: {data_format}")
    
    async def _extract_from_database(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer datos de base de datos"""
        # Simular extracción de base de datos
        await asyncio.sleep(0.1)
        return [
            {"id": 1, "name": "Sample Data 1", "value": 100},
            {"id": 2, "name": "Sample Data 2", "value": 200},
            {"id": 3, "name": "Sample Data 3", "value": 300}
        ]
    
    async def _extract_from_api(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer datos de API"""
        url = config["url"]
        headers = config.get("headers", {})
        params = config.get("params", {})
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API request failed with status {response.status}")
    
    async def _extract_from_stream(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer datos de stream"""
        # Simular extracción de stream
        await asyncio.sleep(0.1)
        return [
            {"timestamp": time.time(), "data": "stream_data_1"},
            {"timestamp": time.time(), "data": "stream_data_2"}
        ]
    
    async def _extract_from_message_queue(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer datos de cola de mensajes"""
        # Simular extracción de cola de mensajes
        await asyncio.sleep(0.1)
        return [
            {"message_id": "msg_1", "content": "message_content_1"},
            {"message_id": "msg_2", "content": "message_content_2"}
        ]
    
    async def _extract_from_cloud_storage(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer datos de almacenamiento en la nube"""
        # Simular extracción de almacenamiento en la nube
        await asyncio.sleep(0.1)
        return [
            {"file_name": "cloud_file_1.txt", "content": "cloud_content_1"},
            {"file_name": "cloud_file_2.txt", "content": "cloud_content_2"}
        ]
    
    async def _extract_from_iot_device(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer datos de dispositivo IoT"""
        # Simular extracción de dispositivo IoT
        await asyncio.sleep(0.1)
        return [
            {"device_id": "iot_1", "sensor_data": {"temperature": 25.5, "humidity": 60}},
            {"device_id": "iot_2", "sensor_data": {"temperature": 26.0, "humidity": 65}}
        ]
    
    async def _extract_from_webhook(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer datos de webhook"""
        # Simular extracción de webhook
        await asyncio.sleep(0.1)
        return [
            {"webhook_id": "webhook_1", "event": "data_received", "payload": {"key": "value"}}
        ]


class DataTransformer:
    """Transformador de datos"""
    
    def __init__(self):
        self.transformations: Dict[str, Callable] = {
            "filter": self._filter_data,
            "map": self._map_data,
            "aggregate": self._aggregate_data,
            "join": self._join_data,
            "sort": self._sort_data,
            "deduplicate": self._deduplicate_data,
            "validate": self._validate_data,
            "enrich": self._enrich_data
        }
    
    async def transform_data(self, data: List[Dict[str, Any]], 
                           transformations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transformar datos"""
        try:
            result = data.copy()
            
            for transformation in transformations:
                transform_type = transformation["type"]
                parameters = transformation.get("parameters", {})
                
                if transform_type in self.transformations:
                    result = await self.transformations[transform_type](result, parameters)
                else:
                    raise ValueError(f"Unsupported transformation type: {transform_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            raise
    
    async def _filter_data(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filtrar datos"""
        filter_condition = parameters.get("condition", {})
        filtered_data = []
        
        for record in data:
            if self._evaluate_condition(record, filter_condition):
                filtered_data.append(record)
        
        return filtered_data
    
    async def _map_data(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mapear datos"""
        mapping = parameters.get("mapping", {})
        mapped_data = []
        
        for record in data:
            mapped_record = {}
            for old_key, new_key in mapping.items():
                if old_key in record:
                    mapped_record[new_key] = record[old_key]
            mapped_data.append(mapped_record)
        
        return mapped_data
    
    async def _aggregate_data(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Agregar datos"""
        group_by = parameters.get("group_by", [])
        aggregations = parameters.get("aggregations", {})
        
        if not group_by:
            return data
        
        # Agrupar datos
        groups = defaultdict(list)
        for record in data:
            group_key = tuple(record.get(key) for key in group_by)
            groups[group_key].append(record)
        
        # Agregar cada grupo
        aggregated_data = []
        for group_key, group_data in groups.items():
            aggregated_record = dict(zip(group_by, group_key))
            
            for field, operation in aggregations.items():
                values = [record.get(field) for record in group_data if field in record]
                if values:
                    if operation == "sum":
                        aggregated_record[f"{field}_sum"] = sum(values)
                    elif operation == "avg":
                        aggregated_record[f"{field}_avg"] = sum(values) / len(values)
                    elif operation == "count":
                        aggregated_record[f"{field}_count"] = len(values)
                    elif operation == "min":
                        aggregated_record[f"{field}_min"] = min(values)
                    elif operation == "max":
                        aggregated_record[f"{field}_max"] = max(values)
            
            aggregated_data.append(aggregated_record)
        
        return aggregated_data
    
    async def _join_data(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Unir datos"""
        join_key = parameters.get("join_key")
        join_data = parameters.get("join_data", [])
        join_type = parameters.get("join_type", "inner")
        
        if not join_key or not join_data:
            return data
        
        # Crear índice de datos de unión
        join_index = {}
        for record in join_data:
            key_value = record.get(join_key)
            if key_value is not None:
                if key_value not in join_index:
                    join_index[key_value] = []
                join_index[key_value].append(record)
        
        # Realizar unión
        joined_data = []
        for record in data:
            key_value = record.get(join_key)
            if key_value in join_index:
                for join_record in join_index[key_value]:
                    merged_record = {**record, **join_record}
                    joined_data.append(merged_record)
            elif join_type == "left":
                joined_data.append(record)
        
        return joined_data
    
    async def _sort_data(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ordenar datos"""
        sort_key = parameters.get("sort_key")
        sort_order = parameters.get("sort_order", "asc")
        
        if not sort_key:
            return data
        
        reverse = sort_order == "desc"
        return sorted(data, key=lambda x: x.get(sort_key, 0), reverse=reverse)
    
    async def _deduplicate_data(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Eliminar duplicados"""
        dedup_key = parameters.get("dedup_key")
        
        if not dedup_key:
            return data
        
        seen = set()
        deduplicated_data = []
        
        for record in data:
            key_value = record.get(dedup_key)
            if key_value not in seen:
                seen.add(key_value)
                deduplicated_data.append(record)
        
        return deduplicated_data
    
    async def _validate_data(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validar datos"""
        validation_rules = parameters.get("validation_rules", {})
        valid_data = []
        
        for record in data:
            is_valid = True
            
            for field, rules in validation_rules.items():
                value = record.get(field)
                
                if "required" in rules and rules["required"] and value is None:
                    is_valid = False
                    break
                
                if "type" in rules and value is not None:
                    expected_type = rules["type"]
                    if expected_type == "string" and not isinstance(value, str):
                        is_valid = False
                        break
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        is_valid = False
                        break
            
            if is_valid:
                valid_data.append(record)
        
        return valid_data
    
    async def _enrich_data(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enriquecer datos"""
        enrichment_data = parameters.get("enrichment_data", {})
        enriched_data = []
        
        for record in data:
            enriched_record = {**record, **enrichment_data}
            enriched_data.append(enriched_record)
        
        return enriched_data
    
    def _evaluate_condition(self, record: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Evaluar condición de filtro"""
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        
        if not field or not operator:
            return True
        
        record_value = record.get(field)
        
        if operator == "eq":
            return record_value == value
        elif operator == "ne":
            return record_value != value
        elif operator == "gt":
            return record_value > value
        elif operator == "lt":
            return record_value < value
        elif operator == "gte":
            return record_value >= value
        elif operator == "lte":
            return record_value <= value
        elif operator == "in":
            return record_value in value
        elif operator == "contains":
            return value in str(record_value)
        else:
            return True


class DataLoader:
    """Cargador de datos"""
    
    def __init__(self):
        self.loaders: Dict[DataSource, Callable] = {
            DataSource.FILE: self._load_to_file,
            DataSource.DATABASE: self._load_to_database,
            DataSource.API: self._load_to_api,
            DataSource.STREAM: self._load_to_stream,
            DataSource.MESSAGE_QUEUE: self._load_to_message_queue,
            DataSource.CLOUD_STORAGE: self._load_to_cloud_storage,
            DataSource.IOT_DEVICE: self._load_to_iot_device,
            DataSource.WEBHOOK: self._load_to_webhook
        }
    
    async def load_data(self, data: List[Dict[str, Any]], destination: DataSource, 
                       config: Dict[str, Any]) -> bool:
        """Cargar datos al destino"""
        try:
            loader = self.loaders.get(destination)
            if not loader:
                raise ValueError(f"Unsupported data destination: {destination}")
            
            return await loader(data, config)
            
        except Exception as e:
            logger.error(f"Error loading data to {destination}: {e}")
            raise
    
    async def _load_to_file(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Cargar datos a archivo"""
        file_path = config["file_path"]
        data_format = DataFormat(config.get("format", "json"))
        
        if data_format == DataFormat.JSON:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        elif data_format == DataFormat.CSV:
            if data:
                fieldnames = data[0].keys()
                async with aiofiles.open(file_path, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    await f.write(','.join(fieldnames) + '\n')
                    for record in data:
                        await f.write(','.join(str(record.get(field, '')) for field in fieldnames) + '\n')
        
        return True
    
    async def _load_to_database(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Cargar datos a base de datos"""
        # Simular carga a base de datos
        await asyncio.sleep(0.1)
        return True
    
    async def _load_to_api(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Cargar datos a API"""
        url = config["url"]
        headers = config.get("headers", {})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                return response.status == 200
    
    async def _load_to_stream(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Cargar datos a stream"""
        # Simular carga a stream
        await asyncio.sleep(0.1)
        return True
    
    async def _load_to_message_queue(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Cargar datos a cola de mensajes"""
        # Simular carga a cola de mensajes
        await asyncio.sleep(0.1)
        return True
    
    async def _load_to_cloud_storage(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Cargar datos a almacenamiento en la nube"""
        # Simular carga a almacenamiento en la nube
        await asyncio.sleep(0.1)
        return True
    
    async def _load_to_iot_device(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Cargar datos a dispositivo IoT"""
        # Simular carga a dispositivo IoT
        await asyncio.sleep(0.1)
        return True
    
    async def _load_to_webhook(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Cargar datos a webhook"""
        # Simular carga a webhook
        await asyncio.sleep(0.1)
        return True


class DataPipelineSystem:
    """Sistema principal de pipeline de datos"""
    
    def __init__(self):
        self.pipelines: Dict[str, DataPipeline] = {}
        self.executions: Dict[str, PipelineExecution] = {}
        self.transformations: Dict[str, DataTransformation] = {}
        self.extractor = DataExtractor()
        self.transformer = DataTransformer()
        self.loader = DataLoader()
        self.is_running = False
        self._execution_queue = queue.Queue()
        self._executor_thread = None
        self._lock = threading.Lock()
    
    async def start(self):
        """Iniciar sistema de pipeline de datos"""
        try:
            self.is_running = True
            
            # Iniciar hilo ejecutor
            self._executor_thread = threading.Thread(target=self._execution_worker)
            self._executor_thread.start()
            
            logger.info("Data pipeline system started")
            
        except Exception as e:
            logger.error(f"Error starting data pipeline system: {e}")
            raise
    
    async def stop(self):
        """Detener sistema de pipeline de datos"""
        try:
            self.is_running = False
            
            # Detener hilo ejecutor
            if self._executor_thread:
                self._executor_thread.join(timeout=5)
            
            logger.info("Data pipeline system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping data pipeline system: {e}")
    
    def _execution_worker(self):
        """Worker para ejecutar pipelines"""
        while self.is_running:
            try:
                execution_id = self._execution_queue.get(timeout=1)
                if execution_id:
                    asyncio.run(self._execute_pipeline(execution_id))
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in execution worker: {e}")
    
    async def create_pipeline(self, pipeline_info: Dict[str, Any]) -> str:
        """Crear pipeline de datos"""
        pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        
        pipeline = DataPipeline(
            id=pipeline_id,
            name=pipeline_info["name"],
            description=pipeline_info.get("description", ""),
            stages=pipeline_info["stages"],
            status=PipelineStatus.IDLE,
            source=DataSource(pipeline_info["source"]),
            destination=DataSource(pipeline_info["destination"]),
            data_format=DataFormat(pipeline_info.get("data_format", "json")),
            created_at=time.time(),
            last_run=None,
            next_run=None,
            schedule=pipeline_info.get("schedule"),
            metadata=pipeline_info.get("metadata", {})
        )
        
        async with self._lock:
            self.pipelines[pipeline_id] = pipeline
        
        logger.info(f"Data pipeline created: {pipeline_id} ({pipeline.name})")
        return pipeline_id
    
    async def execute_pipeline(self, pipeline_id: str) -> str:
        """Ejecutar pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        execution = PipelineExecution(
            id=execution_id,
            pipeline_id=pipeline_id,
            status=PipelineStatus.RUNNING,
            started_at=time.time(),
            completed_at=None,
            records_processed=0,
            records_failed=0,
            execution_time=0.0,
            error_message=None,
            metadata={}
        )
        
        async with self._lock:
            self.executions[execution_id] = execution
            self.pipelines[pipeline_id].status = PipelineStatus.RUNNING
        
        # Agregar a cola de ejecución
        self._execution_queue.put(execution_id)
        
        return execution_id
    
    async def _execute_pipeline(self, execution_id: str):
        """Ejecutar pipeline internamente"""
        try:
            execution = self.executions[execution_id]
            pipeline = self.pipelines[execution.pipeline_id]
            
            # Extraer datos
            extract_config = pipeline.stages[0].get("config", {})
            data = await self.extractor.extract_data(pipeline.source, extract_config)
            execution.records_processed = len(data)
            
            # Transformar datos
            for stage in pipeline.stages[1:-1]:  # Excluir extract y load
                if stage["type"] == "transform":
                    transform_config = stage.get("config", {})
                    transformations = transform_config.get("transformations", [])
                    data = await self.transformer.transform_data(data, transformations)
            
            # Cargar datos
            load_config = pipeline.stages[-1].get("config", {})
            success = await self.loader.load_data(data, pipeline.destination, load_config)
            
            if success:
                execution.status = PipelineStatus.COMPLETED
                execution.completed_at = time.time()
                execution.execution_time = execution.completed_at - execution.started_at
                
                async with self._lock:
                    pipeline.status = PipelineStatus.COMPLETED
                    pipeline.last_run = execution.completed_at
            else:
                execution.status = PipelineStatus.FAILED
                execution.error_message = "Failed to load data"
                execution.completed_at = time.time()
                execution.execution_time = execution.completed_at - execution.started_at
                
                async with self._lock:
                    pipeline.status = PipelineStatus.FAILED
            
        except Exception as e:
            logger.error(f"Error executing pipeline {execution_id}: {e}")
            
            execution.status = PipelineStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = time.time()
            execution.execution_time = execution.completed_at - execution.started_at
            
            async with self._lock:
                pipeline.status = PipelineStatus.FAILED
    
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado del pipeline"""
        if pipeline_id not in self.pipelines:
            return None
        
        pipeline = self.pipelines[pipeline_id]
        return {
            "id": pipeline.id,
            "name": pipeline.name,
            "status": pipeline.status.value,
            "last_run": pipeline.last_run,
            "next_run": pipeline.next_run,
            "created_at": pipeline.created_at
        }
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de ejecución"""
        if execution_id not in self.executions:
            return None
        
        execution = self.executions[execution_id]
        return {
            "id": execution.id,
            "pipeline_id": execution.pipeline_id,
            "status": execution.status.value,
            "started_at": execution.started_at,
            "completed_at": execution.completed_at,
            "records_processed": execution.records_processed,
            "records_failed": execution.records_failed,
            "execution_time": execution.execution_time,
            "error_message": execution.error_message
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "is_running": self.is_running,
            "pipelines": {
                "total": len(self.pipelines),
                "by_status": {
                    status.value: sum(1 for p in self.pipelines.values() if p.status == status)
                    for status in PipelineStatus
                }
            },
            "executions": {
                "total": len(self.executions),
                "by_status": {
                    status.value: sum(1 for e in self.executions.values() if e.status == status)
                    for status in PipelineStatus
                }
            },
            "transformations": len(self.transformations),
            "queue_size": self._execution_queue.qsize()
        }


# Instancia global del sistema de pipeline de datos
data_pipeline_system = DataPipelineSystem()


# Router para endpoints del sistema de pipeline de datos
data_pipeline_router = APIRouter()


@data_pipeline_router.post("/data-pipeline/create")
async def create_data_pipeline_endpoint(pipeline_data: dict):
    """Crear pipeline de datos"""
    try:
        pipeline_id = await data_pipeline_system.create_pipeline(pipeline_data)
        
        return {
            "message": "Data pipeline created successfully",
            "pipeline_id": pipeline_id,
            "name": pipeline_data["name"],
            "source": pipeline_data["source"],
            "destination": pipeline_data["destination"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data source or format: {e}")
    except Exception as e:
        logger.error(f"Error creating data pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create data pipeline: {str(e)}")


@data_pipeline_router.get("/data-pipeline/pipelines")
async def get_data_pipelines_endpoint():
    """Obtener pipelines de datos"""
    try:
        pipelines = data_pipeline_system.pipelines
        return {
            "pipelines": [
                {
                    "id": pipeline.id,
                    "name": pipeline.name,
                    "description": pipeline.description,
                    "status": pipeline.status.value,
                    "source": pipeline.source.value,
                    "destination": pipeline.destination.value,
                    "data_format": pipeline.data_format.value,
                    "created_at": pipeline.created_at,
                    "last_run": pipeline.last_run,
                    "next_run": pipeline.next_run
                }
                for pipeline in pipelines.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting data pipelines: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data pipelines: {str(e)}")


@data_pipeline_router.get("/data-pipeline/pipelines/{pipeline_id}")
async def get_data_pipeline_endpoint(pipeline_id: str):
    """Obtener pipeline de datos específico"""
    try:
        status = await data_pipeline_system.get_pipeline_status(pipeline_id)
        
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Data pipeline not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data pipeline: {str(e)}")


@data_pipeline_router.post("/data-pipeline/pipelines/{pipeline_id}/execute")
async def execute_data_pipeline_endpoint(pipeline_id: str):
    """Ejecutar pipeline de datos"""
    try:
        execution_id = await data_pipeline_system.execute_pipeline(pipeline_id)
        
        return {
            "message": "Data pipeline execution started successfully",
            "execution_id": execution_id,
            "pipeline_id": pipeline_id
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing data pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute data pipeline: {str(e)}")


@data_pipeline_router.get("/data-pipeline/executions/{execution_id}")
async def get_execution_status_endpoint(execution_id: str):
    """Obtener estado de ejecución"""
    try:
        status = await data_pipeline_system.get_execution_status(execution_id)
        
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Pipeline execution not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get execution status: {str(e)}")


@data_pipeline_router.get("/data-pipeline/stats")
async def get_data_pipeline_stats_endpoint():
    """Obtener estadísticas del sistema de pipeline de datos"""
    try:
        stats = await data_pipeline_system.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting data pipeline stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data pipeline stats: {str(e)}")


# Funciones de utilidad para integración
async def start_data_pipeline_system():
    """Iniciar sistema de pipeline de datos"""
    await data_pipeline_system.start()


async def stop_data_pipeline_system():
    """Detener sistema de pipeline de datos"""
    await data_pipeline_system.stop()


async def create_data_pipeline(pipeline_info: Dict[str, Any]) -> str:
    """Crear pipeline de datos"""
    return await data_pipeline_system.create_pipeline(pipeline_info)


async def execute_data_pipeline(pipeline_id: str) -> str:
    """Ejecutar pipeline de datos"""
    return await data_pipeline_system.execute_pipeline(pipeline_id)


async def get_data_pipeline_system_stats() -> Dict[str, Any]:
    """Obtener estadísticas del sistema de pipeline de datos"""
    return await data_pipeline_system.get_system_stats()


logger.info("Data pipeline system module loaded successfully")

