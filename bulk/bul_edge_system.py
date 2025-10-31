"""
BUL - Business Universal Language (Edge Computing System)
========================================================

Advanced Edge Computing system with distributed processing and edge intelligence.
"""

import asyncio
import logging
import json
import time
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import redis
from prometheus_client import Counter, Histogram, Gauge
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import tensorflow as tf
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import psutil
import platform
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///bul_edge.db', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Prometheus Metrics
EDGE_NODES = Gauge('bul_edge_nodes_total', 'Total edge nodes', ['status', 'location'])
EDGE_TASKS = Counter('bul_edge_tasks_total', 'Total edge tasks', ['task_type', 'status'])
EDGE_MODELS = Gauge('bul_edge_models_total', 'Total edge models', ['model_type', 'framework'])
EDGE_LATENCY = Histogram('bul_edge_latency_seconds', 'Edge processing latency')
EDGE_THROUGHPUT = Histogram('bul_edge_throughput_ops_per_second', 'Edge processing throughput')

class EdgeNodeType(str, Enum):
    """Edge node type enumeration."""
    GATEWAY = "gateway"
    COMPUTE = "compute"
    STORAGE = "storage"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    MOBILE = "mobile"
    EMBEDDED = "embedded"
    FOG = "fog"
    MICROCLOUD = "microcloud"
    EDGE_SERVER = "edge_server"

class TaskType(str, Enum):
    """Task type enumeration."""
    INFERENCE = "inference"
    TRAINING = "training"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    DATA_AGGREGATION = "data_aggregation"
    STREAM_PROCESSING = "stream_processing"
    IMAGE_PROCESSING = "image_processing"
    NLP = "nlp"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"

class ModelFramework(str, Enum):
    """Model framework enumeration."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"
    COREML = "coreml"
    TFLITE = "tflite"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUEUED = "queued"

class NodeStatus(str, Enum):
    """Node status enumeration."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"
    MAINTENANCE = "maintenance"
    ERROR = "error"

# Database Models
class EdgeNode(Base):
    __tablename__ = "edge_nodes"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    node_type = Column(String, nullable=False)
    location = Column(String, nullable=False)
    ip_address = Column(String, nullable=False)
    port = Column(Integer, default=8000)
    status = Column(String, default=NodeStatus.OFFLINE)
    cpu_cores = Column(Integer, default=1)
    memory_gb = Column(Float, default=1.0)
    storage_gb = Column(Float, default=10.0)
    gpu_available = Column(Boolean, default=False)
    gpu_memory_gb = Column(Float, default=0.0)
    network_bandwidth_mbps = Column(Float, default=100.0)
    latency_ms = Column(Float, default=0.0)
    last_heartbeat = Column(DateTime)
    capabilities = Column(Text, default="[]")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, default="{}")

class EdgeModel(Base):
    __tablename__ = "edge_models"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    version = Column(String, default="1.0.0")
    model_path = Column(String, nullable=False)
    model_size_mb = Column(Float, default=0.0)
    input_shape = Column(Text, default="[]")
    output_shape = Column(Text, default="[]")
    accuracy = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    throughput_ops_per_sec = Column(Float, default=0.0)
    memory_usage_mb = Column(Float, default=0.0)
    is_optimized = Column(Boolean, default=False)
    optimization_level = Column(Integer, default=0)
    compatible_nodes = Column(Text, default="[]")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, default="{}")

class EdgeTask(Base):
    __tablename__ = "edge_tasks"
    
    id = Column(String, primary_key=True)
    task_name = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    node_id = Column(String, ForeignKey("edge_nodes.id"))
    model_id = Column(String, ForeignKey("edge_models.id"))
    input_data = Column(Text, default="{}")
    output_data = Column(Text, default="{}")
    status = Column(String, default=TaskStatus.PENDING)
    priority = Column(Integer, default=1)
    estimated_duration = Column(Float, default=0.0)
    actual_duration = Column(Float, default=0.0)
    cpu_usage = Column(Float, default=0.0)
    memory_usage = Column(Float, default=0.0)
    gpu_usage = Column(Float, default=0.0)
    network_usage = Column(Float, default=0.0)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    node = relationship("EdgeNode")
    model = relationship("EdgeModel")

class EdgeDataStream(Base):
    __tablename__ = "edge_data_streams"
    
    id = Column(String, primary_key=True)
    stream_name = Column(String, nullable=False)
    source_node_id = Column(String, ForeignKey("edge_nodes.id"))
    target_node_id = Column(String, ForeignKey("edge_nodes.id"))
    data_type = Column(String, nullable=False)
    sampling_rate = Column(Float, default=1.0)
    data_size_bytes = Column(Integer, default=0)
    compression_ratio = Column(Float, default=1.0)
    encryption_enabled = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    source_node = relationship("EdgeNode", foreign_keys=[source_node_id])
    target_node = relationship("EdgeNode", foreign_keys=[target_node_id])

# Create tables
Base.metadata.create_all(bind=engine)

# Edge Computing Configuration
EDGE_CONFIG = {
    "max_nodes": 1000,
    "max_tasks_per_node": 10,
    "task_timeout": 300,  # 5 minutes
    "heartbeat_interval": 30,  # 30 seconds
    "load_balancing": "round_robin",
    "failover_enabled": True,
    "auto_scaling": True,
    "resource_thresholds": {
        "cpu": 80.0,
        "memory": 80.0,
        "storage": 90.0,
        "network": 80.0
    },
    "optimization_settings": {
        "model_quantization": True,
        "pruning_enabled": True,
        "distillation_enabled": True,
        "compression_enabled": True
    },
    "security_settings": {
        "encryption_enabled": True,
        "authentication_required": True,
        "access_control": True,
        "audit_logging": True
    }
}

class AdvancedEdgeSystem:
    """Advanced Edge Computing system with comprehensive features."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL Edge Computing System",
            description="Advanced Edge Computing system with distributed processing and edge intelligence",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Database session
        self.db = SessionLocal()
        
        # Edge components
        self.active_nodes: Dict[str, EdgeNode] = {}
        self.model_cache: Dict[str, Any] = {}
        self.task_queue: List[EdgeTask] = []
        self.data_streams: Dict[str, EdgeDataStream] = {}
        
        # Setup components
        self.setup_middleware()
        self.setup_routes()
        self.setup_default_data()
        self.initialize_edge_components()
        
        logger.info("Advanced Edge Computing System initialized")
    
    def setup_middleware(self):
        """Setup edge middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup edge API routes."""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with edge system information."""
            return {
                "message": "BUL Edge Computing System",
                "version": "1.0.0",
                "status": "operational",
                "features": [
                    "Distributed Edge Processing",
                    "Model Optimization & Deployment",
                    "Real-time Data Streaming",
                    "Load Balancing & Failover",
                    "Resource Management",
                    "Edge Intelligence",
                    "Federated Learning",
                    "Edge Analytics"
                ],
                "node_types": [node_type.value for node_type in EdgeNodeType],
                "task_types": [task_type.value for task_type in TaskType],
                "model_frameworks": [framework.value for framework in ModelFramework],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/nodes/register", tags=["Nodes"])
        async def register_edge_node(node_request: dict):
            """Register edge node."""
            try:
                # Validate request
                required_fields = ["name", "node_type", "location", "ip_address"]
                if not all(field in node_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                name = node_request["name"]
                node_type = node_request["node_type"]
                location = node_request["location"]
                ip_address = node_request["ip_address"]
                
                # Check if node already exists
                existing_node = self.db.query(EdgeNode).filter(EdgeNode.ip_address == ip_address).first()
                if existing_node:
                    raise HTTPException(status_code=400, detail="Node with this IP already registered")
                
                # Create edge node
                node = EdgeNode(
                    id=f"node_{int(time.time())}",
                    name=name,
                    node_type=node_type,
                    location=location,
                    ip_address=ip_address,
                    port=node_request.get("port", 8000),
                    status=NodeStatus.ONLINE,
                    cpu_cores=node_request.get("cpu_cores", 1),
                    memory_gb=node_request.get("memory_gb", 1.0),
                    storage_gb=node_request.get("storage_gb", 10.0),
                    gpu_available=node_request.get("gpu_available", False),
                    gpu_memory_gb=node_request.get("gpu_memory_gb", 0.0),
                    network_bandwidth_mbps=node_request.get("network_bandwidth_mbps", 100.0),
                    capabilities=json.dumps(node_request.get("capabilities", [])),
                    metadata=json.dumps(node_request.get("metadata", {}))
                )
                
                self.db.add(node)
                self.db.commit()
                
                # Add to active nodes
                self.active_nodes[node.id] = node
                
                EDGE_NODES.labels(status=node.status, location=node.location).inc()
                
                return {
                    "message": "Edge node registered successfully",
                    "node_id": node.id,
                    "name": node.name,
                    "node_type": node.node_type,
                    "location": node.location,
                    "status": node.status
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error registering edge node: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/nodes", tags=["Nodes"])
        async def get_edge_nodes():
            """Get all edge nodes."""
            try:
                nodes = self.db.query(EdgeNode).filter(EdgeNode.is_active == True).all()
                
                return {
                    "nodes": [
                        {
                            "id": node.id,
                            "name": node.name,
                            "node_type": node.node_type,
                            "location": node.location,
                            "ip_address": node.ip_address,
                            "port": node.port,
                            "status": node.status,
                            "cpu_cores": node.cpu_cores,
                            "memory_gb": node.memory_gb,
                            "storage_gb": node.storage_gb,
                            "gpu_available": node.gpu_available,
                            "gpu_memory_gb": node.gpu_memory_gb,
                            "network_bandwidth_mbps": node.network_bandwidth_mbps,
                            "latency_ms": node.latency_ms,
                            "last_heartbeat": node.last_heartbeat.isoformat() if node.last_heartbeat else None,
                            "capabilities": json.loads(node.capabilities),
                            "metadata": json.loads(node.metadata),
                            "created_at": node.created_at.isoformat()
                        }
                        for node in nodes
                    ],
                    "total": len(nodes)
                }
                
            except Exception as e:
                logger.error(f"Error getting edge nodes: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/deploy", tags=["Models"])
        async def deploy_edge_model(model_request: dict):
            """Deploy model to edge nodes."""
            try:
                # Validate request
                required_fields = ["name", "model_type", "framework", "model_path"]
                if not all(field in model_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                name = model_request["name"]
                model_type = model_request["model_type"]
                framework = model_request["framework"]
                model_path = model_request["model_path"]
                
                # Create edge model
                model = EdgeModel(
                    id=f"model_{int(time.time())}",
                    name=name,
                    model_type=model_type,
                    framework=framework,
                    version=model_request.get("version", "1.0.0"),
                    model_path=model_path,
                    model_size_mb=model_request.get("model_size_mb", 0.0),
                    input_shape=json.dumps(model_request.get("input_shape", [])),
                    output_shape=json.dumps(model_request.get("output_shape", [])),
                    accuracy=model_request.get("accuracy", 0.0),
                    latency_ms=model_request.get("latency_ms", 0.0),
                    throughput_ops_per_sec=model_request.get("throughput_ops_per_sec", 0.0),
                    memory_usage_mb=model_request.get("memory_usage_mb", 0.0),
                    is_optimized=model_request.get("is_optimized", False),
                    optimization_level=model_request.get("optimization_level", 0),
                    compatible_nodes=json.dumps(model_request.get("compatible_nodes", [])),
                    metadata=json.dumps(model_request.get("metadata", {}))
                )
                
                self.db.add(model)
                self.db.commit()
                
                # Deploy model to compatible nodes
                await self.deploy_model_to_nodes(model.id, model_request.get("target_nodes", []))
                
                EDGE_MODELS.labels(model_type=model_type, framework=framework).inc()
                
                return {
                    "message": "Edge model deployed successfully",
                    "model_id": model.id,
                    "name": model.name,
                    "model_type": model.model_type,
                    "framework": model.framework,
                    "version": model.version
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error deploying edge model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models", tags=["Models"])
        async def get_edge_models():
            """Get all edge models."""
            try:
                models = self.db.query(EdgeModel).filter(EdgeModel.is_active == True).all()
                
                return {
                    "models": [
                        {
                            "id": model.id,
                            "name": model.name,
                            "model_type": model.model_type,
                            "framework": model.framework,
                            "version": model.version,
                            "model_path": model.model_path,
                            "model_size_mb": model.model_size_mb,
                            "input_shape": json.loads(model.input_shape),
                            "output_shape": json.loads(model.output_shape),
                            "accuracy": model.accuracy,
                            "latency_ms": model.latency_ms,
                            "throughput_ops_per_sec": model.throughput_ops_per_sec,
                            "memory_usage_mb": model.memory_usage_mb,
                            "is_optimized": model.is_optimized,
                            "optimization_level": model.optimization_level,
                            "compatible_nodes": json.loads(model.compatible_nodes),
                            "metadata": json.loads(model.metadata),
                            "created_at": model.created_at.isoformat()
                        }
                        for model in models
                    ],
                    "total": len(models)
                }
                
            except Exception as e:
                logger.error(f"Error getting edge models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tasks/submit", tags=["Tasks"])
        async def submit_edge_task(task_request: dict, background_tasks: BackgroundTasks):
            """Submit task to edge nodes."""
            try:
                # Validate request
                required_fields = ["task_name", "task_type", "input_data"]
                if not all(field in task_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                task_name = task_request["task_name"]
                task_type = task_request["task_type"]
                input_data = task_request["input_data"]
                
                # Select optimal node for task
                optimal_node = await self.select_optimal_node(task_type, task_request.get("requirements", {}))
                
                if not optimal_node:
                    raise HTTPException(status_code=503, detail="No suitable edge node available")
                
                # Create edge task
                task = EdgeTask(
                    id=f"task_{int(time.time())}",
                    task_name=task_name,
                    task_type=task_type,
                    node_id=optimal_node.id,
                    model_id=task_request.get("model_id"),
                    input_data=json.dumps(input_data),
                    priority=task_request.get("priority", 1),
                    estimated_duration=task_request.get("estimated_duration", 0.0)
                )
                
                self.db.add(task)
                self.db.commit()
                
                # Execute task in background
                background_tasks.add_task(
                    self.execute_edge_task,
                    task.id,
                    optimal_node.id,
                    task_request
                )
                
                EDGE_TASKS.labels(task_type=task_type, status="pending").inc()
                
                return {
                    "message": "Edge task submitted successfully",
                    "task_id": task.id,
                    "task_name": task.task_name,
                    "task_type": task.task_type,
                    "node_id": task.node_id,
                    "node_name": optimal_node.name,
                    "status": task.status
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error submitting edge task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks", tags=["Tasks"])
        async def get_edge_tasks(limit: int = 100, status: str = None):
            """Get edge tasks."""
            try:
                query = self.db.query(EdgeTask)
                
                if status:
                    query = query.filter(EdgeTask.status == status)
                
                tasks = query.order_by(EdgeTask.created_at.desc()).limit(limit).all()
                
                return {
                    "tasks": [
                        {
                            "id": task.id,
                            "task_name": task.task_name,
                            "task_type": task.task_type,
                            "node_id": task.node_id,
                            "node_name": task.node.name if task.node else None,
                            "model_id": task.model_id,
                            "model_name": task.model.name if task.model else None,
                            "status": task.status,
                            "priority": task.priority,
                            "estimated_duration": task.estimated_duration,
                            "actual_duration": task.actual_duration,
                            "cpu_usage": task.cpu_usage,
                            "memory_usage": task.memory_usage,
                            "gpu_usage": task.gpu_usage,
                            "network_usage": task.network_usage,
                            "error_message": task.error_message,
                            "created_at": task.created_at.isoformat(),
                            "started_at": task.started_at.isoformat() if task.started_at else None,
                            "completed_at": task.completed_at.isoformat() if task.completed_at else None
                        }
                        for task in tasks
                    ],
                    "total": len(tasks)
                }
                
            except Exception as e:
                logger.error(f"Error getting edge tasks: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/streams/create", tags=["Streams"])
        async def create_data_stream(stream_request: dict):
            """Create data stream between edge nodes."""
            try:
                # Validate request
                required_fields = ["stream_name", "source_node_id", "target_node_id", "data_type"]
                if not all(field in stream_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                stream_name = stream_request["stream_name"]
                source_node_id = stream_request["source_node_id"]
                target_node_id = stream_request["target_node_id"]
                data_type = stream_request["data_type"]
                
                # Create data stream
                stream = EdgeDataStream(
                    id=f"stream_{int(time.time())}",
                    stream_name=stream_name,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    data_type=data_type,
                    sampling_rate=stream_request.get("sampling_rate", 1.0),
                    data_size_bytes=stream_request.get("data_size_bytes", 0),
                    compression_ratio=stream_request.get("compression_ratio", 1.0),
                    encryption_enabled=stream_request.get("encryption_enabled", False)
                )
                
                self.db.add(stream)
                self.db.commit()
                
                # Add to data streams
                self.data_streams[stream.id] = stream
                
                return {
                    "message": "Data stream created successfully",
                    "stream_id": stream.id,
                    "stream_name": stream.stream_name,
                    "source_node_id": stream.source_node_id,
                    "target_node_id": stream.target_node_id,
                    "data_type": stream.data_type
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error creating data stream: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/streams", tags=["Streams"])
        async def get_data_streams():
            """Get all data streams."""
            try:
                streams = self.db.query(EdgeDataStream).filter(EdgeDataStream.is_active == True).all()
                
                return {
                    "streams": [
                        {
                            "id": stream.id,
                            "stream_name": stream.stream_name,
                            "source_node_id": stream.source_node_id,
                            "source_node_name": stream.source_node.name if stream.source_node else None,
                            "target_node_id": stream.target_node_id,
                            "target_node_name": stream.target_node.name if stream.target_node else None,
                            "data_type": stream.data_type,
                            "sampling_rate": stream.sampling_rate,
                            "data_size_bytes": stream.data_size_bytes,
                            "compression_ratio": stream.compression_ratio,
                            "encryption_enabled": stream.encryption_enabled,
                            "created_at": stream.created_at.isoformat()
                        }
                        for stream in streams
                    ],
                    "total": len(streams)
                }
                
            except Exception as e:
                logger.error(f"Error getting data streams: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dashboard", tags=["Dashboard"])
        async def get_edge_dashboard():
            """Get edge system dashboard."""
            try:
                # Get statistics
                total_nodes = self.db.query(EdgeNode).count()
                online_nodes = self.db.query(EdgeNode).filter(EdgeNode.status == NodeStatus.ONLINE).count()
                total_models = self.db.query(EdgeModel).count()
                total_tasks = self.db.query(EdgeTask).count()
                completed_tasks = self.db.query(EdgeTask).filter(EdgeTask.status == TaskStatus.COMPLETED).count()
                total_streams = self.db.query(EdgeDataStream).count()
                
                # Get node type distribution
                node_types = {}
                for node_type in EdgeNodeType:
                    count = self.db.query(EdgeNode).filter(EdgeNode.node_type == node_type.value).count()
                    node_types[node_type.value] = count
                
                # Get task type distribution
                task_types = {}
                for task_type in TaskType:
                    count = self.db.query(EdgeTask).filter(EdgeTask.task_type == task_type.value).count()
                    task_types[task_type.value] = count
                
                # Get recent tasks
                recent_tasks = self.db.query(EdgeTask).order_by(EdgeTask.created_at.desc()).limit(10).all()
                
                return {
                    "summary": {
                        "total_nodes": total_nodes,
                        "online_nodes": online_nodes,
                        "total_models": total_models,
                        "total_tasks": total_tasks,
                        "completed_tasks": completed_tasks,
                        "total_streams": total_streams
                    },
                    "node_type_distribution": node_types,
                    "task_type_distribution": task_types,
                    "recent_tasks": [
                        {
                            "id": task.id,
                            "task_name": task.task_name,
                            "task_type": task.task_type,
                            "node_name": task.node.name if task.node else None,
                            "status": task.status,
                            "actual_duration": task.actual_duration,
                            "created_at": task.created_at.isoformat()
                        }
                        for task in recent_tasks
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def setup_default_data(self):
        """Setup default edge data."""
        try:
            # Create sample edge nodes
            sample_nodes = [
                {
                    "name": "Edge Gateway 1",
                    "node_type": EdgeNodeType.GATEWAY,
                    "location": "Data Center A",
                    "ip_address": "192.168.1.100",
                    "cpu_cores": 8,
                    "memory_gb": 16.0,
                    "storage_gb": 500.0,
                    "capabilities": ["inference", "preprocessing", "routing"]
                },
                {
                    "name": "Edge Compute Node 1",
                    "node_type": EdgeNodeType.COMPUTE,
                    "location": "Factory Floor",
                    "ip_address": "192.168.1.101",
                    "cpu_cores": 4,
                    "memory_gb": 8.0,
                    "storage_gb": 100.0,
                    "gpu_available": True,
                    "gpu_memory_gb": 4.0,
                    "capabilities": ["inference", "training", "image_processing"]
                },
                {
                    "name": "Edge Sensor Node 1",
                    "node_type": EdgeNodeType.SENSOR,
                    "location": "Production Line",
                    "ip_address": "192.168.1.102",
                    "cpu_cores": 2,
                    "memory_gb": 2.0,
                    "storage_gb": 10.0,
                    "capabilities": ["data_collection", "preprocessing"]
                }
            ]
            
            for node_data in sample_nodes:
                node = EdgeNode(
                    id=f"node_{node_data['name'].lower().replace(' ', '_')}",
                    name=node_data["name"],
                    node_type=node_data["node_type"],
                    location=node_data["location"],
                    ip_address=node_data["ip_address"],
                    status=NodeStatus.ONLINE,
                    cpu_cores=node_data["cpu_cores"],
                    memory_gb=node_data["memory_gb"],
                    storage_gb=node_data["storage_gb"],
                    gpu_available=node_data.get("gpu_available", False),
                    gpu_memory_gb=node_data.get("gpu_memory_gb", 0.0),
                    capabilities=json.dumps(node_data["capabilities"]),
                    is_active=True
                )
                
                self.db.add(node)
                self.active_nodes[node.id] = node
            
            self.db.commit()
            logger.info("Default edge data created")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating default edge data: {e}")
    
    def initialize_edge_components(self):
        """Initialize edge computing components."""
        try:
            # Initialize model frameworks
            self.model_cache = {
                "pytorch": {},
                "tensorflow": {},
                "onnx": {},
                "scikit_learn": {}
            }
            
            # Initialize task queue
            self.task_queue = []
            
            logger.info("Edge components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing edge components: {e}")
    
    async def select_optimal_node(self, task_type: str, requirements: dict) -> Optional[EdgeNode]:
        """Select optimal edge node for task."""
        try:
            # Get available nodes
            available_nodes = self.db.query(EdgeNode).filter(
                EdgeNode.status == NodeStatus.ONLINE,
                EdgeNode.is_active == True
            ).all()
            
            if not available_nodes:
                return None
            
            # Score nodes based on requirements
            best_node = None
            best_score = -1
            
            for node in available_nodes:
                score = await self.calculate_node_score(node, task_type, requirements)
                if score > best_score:
                    best_score = score
                    best_node = node
            
            return best_node
            
        except Exception as e:
            logger.error(f"Error selecting optimal node: {e}")
            return None
    
    async def calculate_node_score(self, node: EdgeNode, task_type: str, requirements: dict) -> float:
        """Calculate node score for task assignment."""
        try:
            score = 0.0
            
            # CPU requirements
            required_cpu = requirements.get("cpu_cores", 1)
            if node.cpu_cores >= required_cpu:
                score += (node.cpu_cores / required_cpu) * 10
            
            # Memory requirements
            required_memory = requirements.get("memory_gb", 1.0)
            if node.memory_gb >= required_memory:
                score += (node.memory_gb / required_memory) * 10
            
            # GPU requirements
            if requirements.get("gpu_required", False):
                if node.gpu_available:
                    score += 20
                else:
                    score -= 50
            
            # Network requirements
            required_bandwidth = requirements.get("network_bandwidth_mbps", 10.0)
            if node.network_bandwidth_mbps >= required_bandwidth:
                score += (node.network_bandwidth_mbps / required_bandwidth) * 5
            
            # Latency requirements
            max_latency = requirements.get("max_latency_ms", 100.0)
            if node.latency_ms <= max_latency:
                score += (max_latency - node.latency_ms) / max_latency * 10
            
            # Capabilities
            capabilities = json.loads(node.capabilities)
            if task_type in capabilities:
                score += 15
            
            # Location preference
            preferred_location = requirements.get("preferred_location")
            if preferred_location and node.location == preferred_location:
                score += 10
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating node score: {e}")
            return 0.0
    
    async def deploy_model_to_nodes(self, model_id: str, target_nodes: List[str]):
        """Deploy model to target edge nodes."""
        try:
            # Get model
            model = self.db.query(EdgeModel).filter(EdgeModel.id == model_id).first()
            if not model:
                return
            
            # If no target nodes specified, deploy to all compatible nodes
            if not target_nodes:
                compatible_nodes = json.loads(model.compatible_nodes)
                if compatible_nodes:
                    target_nodes = compatible_nodes
                else:
                    # Deploy to all online nodes
                    nodes = self.db.query(EdgeNode).filter(
                        EdgeNode.status == NodeStatus.ONLINE,
                        EdgeNode.is_active == True
                    ).all()
                    target_nodes = [node.id for node in nodes]
            
            # Deploy to each target node
            for node_id in target_nodes:
                await self.deploy_model_to_node(model_id, node_id)
            
        except Exception as e:
            logger.error(f"Error deploying model to nodes: {e}")
    
    async def deploy_model_to_node(self, model_id: str, node_id: str):
        """Deploy model to specific edge node."""
        try:
            # Get model and node
            model = self.db.query(EdgeModel).filter(EdgeModel.id == model_id).first()
            node = self.db.query(EdgeNode).filter(EdgeNode.id == node_id).first()
            
            if not model or not node:
                return
            
            # Simulate model deployment
            logger.info(f"Deploying model {model.name} to node {node.name}")
            
            # In a real implementation, this would:
            # 1. Transfer model files to the edge node
            # 2. Load the model on the node
            # 3. Verify model functionality
            # 4. Update node capabilities
            
            # Update node capabilities
            capabilities = json.loads(node.capabilities)
            if model.model_type not in capabilities:
                capabilities.append(model.model_type)
                node.capabilities = json.dumps(capabilities)
                self.db.commit()
            
        except Exception as e:
            logger.error(f"Error deploying model to node: {e}")
    
    async def execute_edge_task(self, task_id: str, node_id: str, task_request: dict):
        """Execute edge task on specified node."""
        try:
            start_time = time.time()
            
            # Get task and node
            task = self.db.query(EdgeTask).filter(EdgeTask.id == task_id).first()
            node = self.db.query(EdgeNode).filter(EdgeNode.id == node_id).first()
            
            if not task or not node:
                return
            
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            self.db.commit()
            
            # Execute task based on type
            result = await self.execute_task_by_type(task, node, task_request)
            
            # Update task completion
            task.status = TaskStatus.COMPLETED if result["success"] else TaskStatus.FAILED
            task.actual_duration = time.time() - start_time
            task.completed_at = datetime.utcnow()
            
            if result["success"]:
                task.output_data = json.dumps(result["output"])
                task.cpu_usage = result.get("cpu_usage", 0.0)
                task.memory_usage = result.get("memory_usage", 0.0)
                task.gpu_usage = result.get("gpu_usage", 0.0)
                task.network_usage = result.get("network_usage", 0.0)
            else:
                task.error_message = result.get("error", "Unknown error")
            
            self.db.commit()
            
            EDGE_TASKS.labels(task_type=task.task_type, status=task.status).inc()
            EDGE_LATENCY.observe(task.actual_duration)
            
            logger.info(f"Edge task {task_id} completed: {task.status}")
            
        except Exception as e:
            logger.error(f"Error executing edge task: {e}")
            
            # Update task status
            task = self.db.query(EdgeTask).filter(EdgeTask.id == task_id).first()
            if task:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                task.completed_at = datetime.utcnow()
                self.db.commit()
    
    async def execute_task_by_type(self, task: EdgeTask, node: EdgeNode, task_request: dict) -> dict:
        """Execute task based on its type."""
        try:
            task_type = task.task_type
            
            if task_type == TaskType.INFERENCE:
                return await self.execute_inference_task(task, node, task_request)
            elif task_type == TaskType.TRAINING:
                return await self.execute_training_task(task, node, task_request)
            elif task_type == TaskType.PREPROCESSING:
                return await self.execute_preprocessing_task(task, node, task_request)
            elif task_type == TaskType.POSTPROCESSING:
                return await self.execute_postprocessing_task(task, node, task_request)
            elif task_type == TaskType.IMAGE_PROCESSING:
                return await self.execute_image_processing_task(task, node, task_request)
            elif task_type == TaskType.NLP:
                return await self.execute_nlp_task(task, node, task_request)
            elif task_type == TaskType.TIME_SERIES:
                return await self.execute_time_series_task(task, node, task_request)
            elif task_type == TaskType.ANOMALY_DETECTION:
                return await self.execute_anomaly_detection_task(task, node, task_request)
            else:
                return await self.execute_generic_task(task, node, task_request)
                
        except Exception as e:
            logger.error(f"Error executing task by type: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_inference_task(self, task: EdgeTask, node: EdgeNode, task_request: dict) -> dict:
        """Execute inference task."""
        try:
            # Simulate inference execution
            input_data = json.loads(task.input_data)
            
            # Simulate processing time based on model complexity
            processing_time = np.random.uniform(0.1, 2.0)
            await asyncio.sleep(processing_time)
            
            # Simulate inference result
            output = {
                "predictions": np.random.rand(10).tolist(),
                "confidence": np.random.uniform(0.7, 0.95),
                "processing_time": processing_time
            }
            
            return {
                "success": True,
                "output": output,
                "cpu_usage": np.random.uniform(20.0, 80.0),
                "memory_usage": np.random.uniform(100.0, 500.0),
                "gpu_usage": np.random.uniform(0.0, 90.0) if node.gpu_available else 0.0,
                "network_usage": np.random.uniform(1.0, 10.0)
            }
            
        except Exception as e:
            logger.error(f"Error executing inference task: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_training_task(self, task: EdgeTask, node: EdgeNode, task_request: dict) -> dict:
        """Execute training task."""
        try:
            # Simulate training execution
            epochs = task_request.get("epochs", 10)
            
            # Simulate training progress
            for epoch in range(epochs):
                await asyncio.sleep(0.1)  # Simulate training step
            
            # Simulate training result
            output = {
                "accuracy": np.random.uniform(0.8, 0.95),
                "loss": np.random.uniform(0.1, 0.5),
                "epochs_completed": epochs,
                "training_time": epochs * 0.1
            }
            
            return {
                "success": True,
                "output": output,
                "cpu_usage": np.random.uniform(60.0, 95.0),
                "memory_usage": np.random.uniform(500.0, 1000.0),
                "gpu_usage": np.random.uniform(70.0, 95.0) if node.gpu_available else 0.0,
                "network_usage": np.random.uniform(5.0, 20.0)
            }
            
        except Exception as e:
            logger.error(f"Error executing training task: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_preprocessing_task(self, task: EdgeTask, node: EdgeNode, task_request: dict) -> dict:
        """Execute preprocessing task."""
        try:
            # Simulate preprocessing execution
            input_data = json.loads(task.input_data)
            
            # Simulate preprocessing operations
            await asyncio.sleep(np.random.uniform(0.05, 0.5))
            
            # Simulate preprocessing result
            output = {
                "processed_data": input_data,
                "preprocessing_steps": ["normalization", "scaling", "feature_extraction"],
                "processing_time": 0.2
            }
            
            return {
                "success": True,
                "output": output,
                "cpu_usage": np.random.uniform(10.0, 40.0),
                "memory_usage": np.random.uniform(50.0, 200.0),
                "gpu_usage": 0.0,
                "network_usage": np.random.uniform(0.5, 2.0)
            }
            
        except Exception as e:
            logger.error(f"Error executing preprocessing task: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_postprocessing_task(self, task: EdgeTask, node: EdgeNode, task_request: dict) -> dict:
        """Execute postprocessing task."""
        try:
            # Simulate postprocessing execution
            input_data = json.loads(task.input_data)
            
            # Simulate postprocessing operations
            await asyncio.sleep(np.random.uniform(0.05, 0.3))
            
            # Simulate postprocessing result
            output = {
                "postprocessed_data": input_data,
                "postprocessing_steps": ["formatting", "validation", "aggregation"],
                "processing_time": 0.15
            }
            
            return {
                "success": True,
                "output": output,
                "cpu_usage": np.random.uniform(5.0, 25.0),
                "memory_usage": np.random.uniform(30.0, 150.0),
                "gpu_usage": 0.0,
                "network_usage": np.random.uniform(0.2, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error executing postprocessing task: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_image_processing_task(self, task: EdgeTask, node: EdgeNode, task_request: dict) -> dict:
        """Execute image processing task."""
        try:
            # Simulate image processing execution
            input_data = json.loads(task.input_data)
            
            # Simulate image processing operations
            await asyncio.sleep(np.random.uniform(0.2, 1.0))
            
            # Simulate image processing result
            output = {
                "processed_image": "base64_encoded_image",
                "features": np.random.rand(128).tolist(),
                "processing_time": 0.5,
                "image_size": "1024x768"
            }
            
            return {
                "success": True,
                "output": output,
                "cpu_usage": np.random.uniform(30.0, 70.0),
                "memory_usage": np.random.uniform(200.0, 800.0),
                "gpu_usage": np.random.uniform(40.0, 80.0) if node.gpu_available else 0.0,
                "network_usage": np.random.uniform(2.0, 8.0)
            }
            
        except Exception as e:
            logger.error(f"Error executing image processing task: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_nlp_task(self, task: EdgeTask, node: EdgeNode, task_request: dict) -> dict:
        """Execute NLP task."""
        try:
            # Simulate NLP execution
            input_data = json.loads(task.input_data)
            
            # Simulate NLP operations
            await asyncio.sleep(np.random.uniform(0.1, 0.8))
            
            # Simulate NLP result
            output = {
                "tokens": ["token1", "token2", "token3"],
                "sentiment": np.random.uniform(-1.0, 1.0),
                "entities": ["entity1", "entity2"],
                "processing_time": 0.4
            }
            
            return {
                "success": True,
                "output": output,
                "cpu_usage": np.random.uniform(25.0, 60.0),
                "memory_usage": np.random.uniform(150.0, 400.0),
                "gpu_usage": np.random.uniform(20.0, 60.0) if node.gpu_available else 0.0,
                "network_usage": np.random.uniform(1.0, 5.0)
            }
            
        except Exception as e:
            logger.error(f"Error executing NLP task: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_time_series_task(self, task: EdgeTask, node: EdgeNode, task_request: dict) -> dict:
        """Execute time series analysis task."""
        try:
            # Simulate time series analysis
            input_data = json.loads(task.input_data)
            
            # Simulate time series operations
            await asyncio.sleep(np.random.uniform(0.1, 0.6))
            
            # Simulate time series result
            output = {
                "trend": "increasing",
                "seasonality": "weekly",
                "anomalies": [10, 25, 40],
                "forecast": np.random.rand(10).tolist(),
                "processing_time": 0.3
            }
            
            return {
                "success": True,
                "output": output,
                "cpu_usage": np.random.uniform(20.0, 50.0),
                "memory_usage": np.random.uniform(100.0, 300.0),
                "gpu_usage": 0.0,
                "network_usage": np.random.uniform(0.5, 3.0)
            }
            
        except Exception as e:
            logger.error(f"Error executing time series task: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_anomaly_detection_task(self, task: EdgeTask, node: EdgeNode, task_request: dict) -> dict:
        """Execute anomaly detection task."""
        try:
            # Simulate anomaly detection
            input_data = json.loads(task.input_data)
            
            # Simulate anomaly detection operations
            await asyncio.sleep(np.random.uniform(0.1, 0.5))
            
            # Simulate anomaly detection result
            output = {
                "anomalies_detected": np.random.randint(0, 5),
                "anomaly_scores": np.random.rand(10).tolist(),
                "threshold": 0.8,
                "processing_time": 0.25
            }
            
            return {
                "success": True,
                "output": output,
                "cpu_usage": np.random.uniform(15.0, 45.0),
                "memory_usage": np.random.uniform(80.0, 250.0),
                "gpu_usage": 0.0,
                "network_usage": np.random.uniform(0.3, 2.0)
            }
            
        except Exception as e:
            logger.error(f"Error executing anomaly detection task: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_generic_task(self, task: EdgeTask, node: EdgeNode, task_request: dict) -> dict:
        """Execute generic task."""
        try:
            # Simulate generic task execution
            await asyncio.sleep(np.random.uniform(0.1, 1.0))
            
            # Simulate generic result
            output = {
                "result": "Task completed successfully",
                "processing_time": 0.5,
                "node_id": node.id,
                "node_name": node.name
            }
            
            return {
                "success": True,
                "output": output,
                "cpu_usage": np.random.uniform(10.0, 50.0),
                "memory_usage": np.random.uniform(50.0, 200.0),
                "gpu_usage": 0.0,
                "network_usage": np.random.uniform(0.1, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error executing generic task: {e}")
            return {"success": False, "error": str(e)}
    
    def run(self, host: str = "0.0.0.0", port: int = 8013, debug: bool = False):
        """Run the edge system."""
        logger.info(f"Starting Edge Computing System on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Edge Computing System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8013, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run edge system
    system = AdvancedEdgeSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
