#!/usr/bin/env python3
"""
Advanced Model Serving System for Frontier Model Training
Provides comprehensive model deployment, serving, monitoring, and management capabilities.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import fastapi
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
import requests
import aiohttp
import asyncio
from pydantic import BaseModel, Field
import joblib
import pickle
import docker
import kubernetes
from kubernetes import client, config
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import redis
import celery
from celery import Celery
import grpc
import protobuf
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class ServingProtocol(Enum):
    """Serving protocols."""
    REST_API = "rest_api"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MESSAGE_QUEUE = "message_queue"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"
    EDGE_SERVING = "edge_serving"

class DeploymentStrategy(Enum):
    """Deployment strategies."""
    SINGLE_INSTANCE = "single_instance"
    LOAD_BALANCED = "load_balanced"
    AUTO_SCALING = "auto_scaling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING_UPDATE = "rolling_update"
    A_B_TESTING = "a_b_testing"

class MonitoringMetric(Enum):
    """Monitoring metrics."""
    REQUEST_COUNT = "request_count"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    MODEL_ACCURACY = "model_accuracy"
    DRIFT_DETECTION = "drift_detection"

class ScalingPolicy(Enum):
    """Scaling policies."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    CUSTOM_METRIC = "custom_metric"
    SCHEDULE_BASED = "schedule_based"
    PREDICTIVE = "predictive"

@dataclass
class ModelServingConfig:
    """Model serving configuration."""
    serving_protocol: ServingProtocol = ServingProtocol.REST_API
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.AUTO_SCALING
    monitoring_metrics: List[MonitoringMetric] = None
    scaling_policy: ScalingPolicy = ScalingPolicy.CPU_BASED
    max_instances: int = 10
    min_instances: int = 1
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    enable_health_checks: bool = True
    enable_load_balancing: bool = True
    enable_auto_scaling: bool = True
    enable_monitoring: bool = True
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_alerting: bool = True
    enable_caching: bool = True
    enable_batch_processing: bool = True
    enable_streaming: bool = True
    device: str = "auto"

@dataclass
class ModelEndpoint:
    """Model endpoint configuration."""
    endpoint_id: str
    model_path: str
    model_type: str
    version: str
    protocol: ServingProtocol
    port: int
    health_check_path: str
    metrics_path: str
    created_at: datetime

@dataclass
class ServingRequest:
    """Serving request."""
    request_id: str
    endpoint_id: str
    input_data: Dict[str, Any]
    timestamp: datetime
    client_id: str
    request_metadata: Dict[str, Any]

@dataclass
class ServingResponse:
    """Serving response."""
    response_id: str
    request_id: str
    prediction: Any
    confidence: float
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ServingMetrics:
    """Serving metrics."""
    endpoint_id: str
    request_count: int
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime

class ModelLoader:
    """Model loading and management."""
    
    def __init__(self, config: ModelServingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Model cache
        self.model_cache: Dict[str, Any] = {}
    
    def load_model(self, model_path: str, model_type: str = "pytorch") -> Any:
        """Load model from path."""
        console.print(f"[blue]Loading model from {model_path}...[/blue]")
        
        try:
            if model_type == "pytorch":
                model = torch.load(model_path, map_location=self.device)
                model.eval()
            elif model_type == "sklearn":
                model = joblib.load(model_path)
            elif model_type == "onnx":
                import onnxruntime as ort
                model = ort.InferenceSession(model_path)
            elif model_type == "tensorflow":
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
            else:
                model = joblib.load(model_path)
            
            console.print(f"[green]Model loaded successfully[/green]")
            return model
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return None
    
    def cache_model(self, model_id: str, model: Any):
        """Cache model in memory."""
        self.model_cache[model_id] = model
        console.print(f"[green]Model {model_id} cached[/green]")
    
    def get_cached_model(self, model_id: str) -> Optional[Any]:
        """Get cached model."""
        return self.model_cache.get(model_id)

class PredictionEngine:
    """Prediction engine for model serving."""
    
    def __init__(self, config: ModelServingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def predict(self, model: Any, input_data: Dict[str, Any], model_type: str = "pytorch") -> Dict[str, Any]:
        """Make prediction using model."""
        start_time = time.time()
        
        try:
            if model_type == "pytorch":
                prediction = self._pytorch_predict(model, input_data)
            elif model_type == "sklearn":
                prediction = self._sklearn_predict(model, input_data)
            elif model_type == "onnx":
                prediction = self._onnx_predict(model, input_data)
            elif model_type == "tensorflow":
                prediction = self._tensorflow_predict(model, input_data)
            else:
                prediction = self._sklearn_predict(model, input_data)
            
            processing_time = time.time() - start_time
            
            return {
                'prediction': prediction,
                'confidence': self._calculate_confidence(prediction),
                'processing_time': processing_time,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {
                'prediction': None,
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _pytorch_predict(self, model: nn.Module, input_data: Dict[str, Any]) -> Any:
        """PyTorch model prediction."""
        model.eval()
        
        with torch.no_grad():
            # Convert input to tensor
            if 'input' in input_data:
                input_tensor = torch.FloatTensor(input_data['input']).to(self.device)
            else:
                # Assume first value is input
                input_tensor = torch.FloatTensor(list(input_data.values())[0]).to(self.device)
            
            # Make prediction
            output = model(input_tensor)
            
            # Convert to numpy
            if isinstance(output, torch.Tensor):
                return output.cpu().numpy()
            else:
                return output
    
    def _sklearn_predict(self, model: Any, input_data: Dict[str, Any]) -> Any:
        """Scikit-learn model prediction."""
        if 'input' in input_data:
            input_array = np.array(input_data['input'])
        else:
            input_array = np.array(list(input_data.values())[0])
        
        # Reshape if needed
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
        
        return model.predict(input_array)
    
    def _onnx_predict(self, session: Any, input_data: Dict[str, Any]) -> Any:
        """ONNX model prediction."""
        if 'input' in input_data:
            input_array = np.array(input_data['input'])
        else:
            input_array = np.array(list(input_data.values())[0])
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        result = session.run(None, {input_name: input_array})
        
        return result[0]
    
    def _tensorflow_predict(self, model: Any, input_data: Dict[str, Any]) -> Any:
        """TensorFlow model prediction."""
        if 'input' in input_data:
            input_array = np.array(input_data['input'])
        else:
            input_array = np.array(list(input_data.values())[0])
        
        return model.predict(input_array)
    
    def _calculate_confidence(self, prediction: Any) -> float:
        """Calculate prediction confidence."""
        try:
            if isinstance(prediction, np.ndarray):
                if len(prediction.shape) > 1:
                    # Multi-class prediction
                    max_prob = np.max(prediction)
                    return float(max_prob)
                else:
                    # Single prediction
                    return 1.0
            else:
                return 1.0
        except Exception:
            return 0.0

class MetricsCollector:
    """Metrics collection and monitoring."""
    
    def __init__(self, config: ModelServingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Prometheus metrics
        self.request_count = Counter('model_requests_total', 'Total requests', ['endpoint_id'])
        self.response_time = Histogram('model_response_time_seconds', 'Response time', ['endpoint_id'])
        self.error_count = Counter('model_errors_total', 'Total errors', ['endpoint_id'])
        self.active_requests = Gauge('model_active_requests', 'Active requests', ['endpoint_id'])
        self.memory_usage = Gauge('model_memory_usage_bytes', 'Memory usage', ['endpoint_id'])
        self.cpu_usage = Gauge('model_cpu_usage_percent', 'CPU usage', ['endpoint_id'])
        
        # Metrics storage
        self.metrics_history: List[ServingMetrics] = []
    
    def record_request(self, endpoint_id: str, processing_time: float, success: bool):
        """Record request metrics."""
        self.request_count.labels(endpoint_id=endpoint_id).inc()
        self.response_time.labels(endpoint_id=endpoint_id).observe(processing_time)
        
        if not success:
            self.error_count.labels(endpoint_id=endpoint_id).inc()
    
    def record_resource_usage(self, endpoint_id: str, memory_usage: float, cpu_usage: float):
        """Record resource usage metrics."""
        self.memory_usage.labels(endpoint_id=endpoint_id).set(memory_usage)
        self.cpu_usage.labels(endpoint_id=endpoint_id).set(cpu_usage)
    
    def get_metrics_summary(self, endpoint_id: str) -> Dict[str, float]:
        """Get metrics summary for endpoint."""
        # Simplified metrics calculation
        return {
            'request_count': 100,  # Placeholder
            'response_time_avg': 0.1,
            'response_time_p95': 0.2,
            'response_time_p99': 0.5,
            'error_rate': 0.01,
            'throughput': 1000.0,
            'memory_usage': 512.0,
            'cpu_usage': 50.0
        }

class LoadBalancer:
    """Load balancer for model serving."""
    
    def __init__(self, config: ModelServingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Endpoint registry
        self.endpoints: Dict[str, List[str]] = {}
        self.endpoint_weights: Dict[str, List[float]] = {}
        self.endpoint_health: Dict[str, List[bool]] = {}
    
    def register_endpoint(self, endpoint_id: str, instances: List[str], weights: List[float] = None):
        """Register endpoint instances."""
        self.endpoints[endpoint_id] = instances
        self.endpoint_weights[endpoint_id] = weights or [1.0] * len(instances)
        self.endpoint_health[endpoint_id] = [True] * len(instances)
        
        console.print(f"[green]Registered {len(instances)} instances for endpoint {endpoint_id}[/green]")
    
    def select_instance(self, endpoint_id: str) -> Optional[str]:
        """Select instance for request."""
        if endpoint_id not in self.endpoints:
            return None
        
        instances = self.endpoints[endpoint_id]
        weights = self.endpoint_weights[endpoint_id]
        health = self.endpoint_health[endpoint_id]
        
        # Filter healthy instances
        healthy_instances = [inst for i, inst in enumerate(instances) if health[i]]
        healthy_weights = [weights[i] for i, inst in enumerate(instances) if health[i]]
        
        if not healthy_instances:
            return None
        
        # Weighted random selection
        total_weight = sum(healthy_weights)
        if total_weight == 0:
            return np.random.choice(healthy_instances)
        
        weights_normalized = [w / total_weight for w in healthy_weights]
        return np.random.choice(healthy_instances, p=weights_normalized)
    
    def update_health(self, endpoint_id: str, instance: str, healthy: bool):
        """Update instance health status."""
        if endpoint_id in self.endpoints:
            instances = self.endpoints[endpoint_id]
            if instance in instances:
                index = instances.index(instance)
                self.endpoint_health[endpoint_id][index] = healthy

class AutoScaler:
    """Auto-scaling for model serving."""
    
    def __init__(self, config: ModelServingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Scaling metrics
        self.scaling_metrics: Dict[str, Dict[str, float]] = {}
        self.scaling_history: List[Dict[str, Any]] = []
    
    def should_scale_up(self, endpoint_id: str) -> bool:
        """Check if should scale up."""
        if endpoint_id not in self.scaling_metrics:
            return False
        
        metrics = self.scaling_metrics[endpoint_id]
        
        if self.config.scaling_policy == ScalingPolicy.CPU_BASED:
            return metrics.get('cpu_usage', 0) > self.config.target_cpu_utilization
        elif self.config.scaling_policy == ScalingPolicy.MEMORY_BASED:
            return metrics.get('memory_usage', 0) > self.config.target_memory_utilization
        elif self.config.scaling_policy == ScalingPolicy.REQUEST_BASED:
            return metrics.get('request_count', 0) > 1000  # Threshold
        else:
            return False
    
    def should_scale_down(self, endpoint_id: str) -> bool:
        """Check if should scale down."""
        if endpoint_id not in self.scaling_metrics:
            return False
        
        metrics = self.scaling_metrics[endpoint_id]
        
        if self.config.scaling_policy == ScalingPolicy.CPU_BASED:
            return metrics.get('cpu_usage', 0) < self.config.target_cpu_utilization * 0.5
        elif self.config.scaling_policy == ScalingPolicy.MEMORY_BASED:
            return metrics.get('memory_usage', 0) < self.config.target_memory_utilization * 0.5
        elif self.config.scaling_policy == ScalingPolicy.REQUEST_BASED:
            return metrics.get('request_count', 0) < 100  # Threshold
        else:
            return False
    
    def update_metrics(self, endpoint_id: str, metrics: Dict[str, float]):
        """Update scaling metrics."""
        self.scaling_metrics[endpoint_id] = metrics

class ModelServingAPI:
    """FastAPI-based model serving API."""
    
    def __init__(self, config: ModelServingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_loader = ModelLoader(config)
        self.prediction_engine = PredictionEngine(config)
        self.metrics_collector = MetricsCollector(config)
        self.load_balancer = LoadBalancer(config)
        self.auto_scaler = AutoScaler(config)
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Model Serving API", version="1.0.0")
        self._setup_middleware()
        self._setup_routes()
        
        # Model endpoints
        self.model_endpoints: Dict[str, ModelEndpoint] = {}
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/predict/{endpoint_id}")
        async def predict_endpoint(endpoint_id: str, request: Request):
            """Prediction endpoint."""
            try:
                # Get input data
                input_data = await request.json()
                
                # Get model endpoint
                if endpoint_id not in self.model_endpoints:
                    raise HTTPException(status_code=404, detail="Endpoint not found")
                
                endpoint = self.model_endpoints[endpoint_id]
                
                # Load model if not cached
                model = self.model_loader.get_cached_model(endpoint_id)
                if model is None:
                    model = self.model_loader.load_model(endpoint.model_path, endpoint.model_type)
                    if model is None:
                        raise HTTPException(status_code=500, detail="Model loading failed")
                    self.model_loader.cache_model(endpoint_id, model)
                
                # Make prediction
                start_time = time.time()
                result = self.prediction_engine.predict(model, input_data, endpoint.model_type)
                processing_time = time.time() - start_time
                
                # Record metrics
                self.metrics_collector.record_request(endpoint_id, processing_time, result['success'])
                
                if not result['success']:
                    raise HTTPException(status_code=500, detail=result.get('error', 'Prediction failed'))
                
                # Create response
                response = ServingResponse(
                    response_id=f"resp_{int(time.time())}",
                    request_id=f"req_{int(time.time())}",
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    processing_time=processing_time,
                    timestamp=datetime.now(),
                    metadata={'endpoint_id': endpoint_id}
                )
                
                return {
                    'response_id': response.response_id,
                    'prediction': response.prediction,
                    'confidence': response.confidence,
                    'processing_time': response.processing_time,
                    'timestamp': response.timestamp.isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health/{endpoint_id}")
        async def health_check(endpoint_id: str):
            """Health check endpoint."""
            if endpoint_id not in self.model_endpoints:
                raise HTTPException(status_code=404, detail="Endpoint not found")
            
            return {"status": "healthy", "endpoint_id": endpoint_id}
        
        @self.app.get("/metrics/{endpoint_id}")
        async def get_metrics(endpoint_id: str):
            """Metrics endpoint."""
            if endpoint_id not in self.model_endpoints:
                raise HTTPException(status_code=404, detail="Endpoint not found")
            
            metrics = self.metrics_collector.get_metrics_summary(endpoint_id)
            return metrics
        
        @self.app.post("/deploy")
        async def deploy_model(request: Request):
            """Deploy new model endpoint."""
            try:
                data = await request.json()
                
                endpoint = ModelEndpoint(
                    endpoint_id=data['endpoint_id'],
                    model_path=data['model_path'],
                    model_type=data.get('model_type', 'pytorch'),
                    version=data.get('version', '1.0'),
                    protocol=ServingProtocol(data.get('protocol', 'rest_api')),
                    port=data.get('port', 8000),
                    health_check_path=data.get('health_check_path', '/health'),
                    metrics_path=data.get('metrics_path', '/metrics'),
                    created_at=datetime.now()
                )
                
                self.model_endpoints[endpoint.endpoint_id] = endpoint
                
                # Register with load balancer
                self.load_balancer.register_endpoint(endpoint.endpoint_id, [f"localhost:{endpoint.port}"])
                
                return {"status": "deployed", "endpoint_id": endpoint.endpoint_id}
                
            except Exception as e:
                self.logger.error(f"Model deployment failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/deploy/{endpoint_id}")
        async def undeploy_model(endpoint_id: str):
            """Undeploy model endpoint."""
            if endpoint_id not in self.model_endpoints:
                raise HTTPException(status_code=404, detail="Endpoint not found")
            
            del self.model_endpoints[endpoint_id]
            return {"status": "undeployed", "endpoint_id": endpoint_id}
    
    def register_model(self, endpoint_id: str, model_path: str, model_type: str = "pytorch"):
        """Register model for serving."""
        endpoint = ModelEndpoint(
            endpoint_id=endpoint_id,
            model_path=model_path,
            model_type=model_type,
            version="1.0",
            protocol=self.config.serving_protocol,
            port=8000,
            health_check_path="/health",
            metrics_path="/metrics",
            created_at=datetime.now()
        )
        
        self.model_endpoints[endpoint_id] = endpoint
        
        # Load and cache model
        model = self.model_loader.load_model(model_path, model_type)
        if model:
            self.model_loader.cache_model(endpoint_id, model)
        
        console.print(f"[green]Model {endpoint_id} registered for serving[/green]")
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the serving server."""
        console.print(f"[blue]Starting model serving server on {host}:{port}...[/blue]")
        
        uvicorn.run(self.app, host=host, port=port)

class ModelServingSystem:
    """Main model serving system."""
    
    def __init__(self, config: ModelServingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.serving_api = ModelServingAPI(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.serving_metrics: Dict[str, List[ServingMetrics]] = {}
    
    def _init_database(self) -> str:
        """Initialize model serving database."""
        db_path = Path("./model_serving.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_endpoints (
                    endpoint_id TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    health_check_path TEXT NOT NULL,
                    metrics_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS serving_requests (
                    request_id TEXT PRIMARY KEY,
                    endpoint_id TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    client_id TEXT,
                    request_metadata TEXT,
                    FOREIGN KEY (endpoint_id) REFERENCES model_endpoints (endpoint_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS serving_responses (
                    response_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (request_id) REFERENCES serving_requests (request_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS serving_metrics (
                    endpoint_id TEXT NOT NULL,
                    request_count INTEGER NOT NULL,
                    response_time_avg REAL NOT NULL,
                    response_time_p95 REAL NOT NULL,
                    response_time_p99 REAL NOT NULL,
                    error_rate REAL NOT NULL,
                    throughput REAL NOT NULL,
                    memory_usage REAL NOT NULL,
                    cpu_usage REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def deploy_model(self, endpoint_id: str, model_path: str, model_type: str = "pytorch") -> bool:
        """Deploy model for serving."""
        console.print(f"[blue]Deploying model {endpoint_id}...[/blue]")
        
        try:
            # Register model with serving API
            self.serving_api.register_model(endpoint_id, model_path, model_type)
            
            # Save to database
            self._save_model_endpoint(endpoint_id, model_path, model_type)
            
            console.print(f"[green]Model {endpoint_id} deployed successfully[/green]")
            return True
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return False
    
    def _save_model_endpoint(self, endpoint_id: str, model_path: str, model_type: str):
        """Save model endpoint to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_endpoints 
                (endpoint_id, model_path, model_type, version, protocol, port,
                 health_check_path, metrics_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                endpoint_id,
                model_path,
                model_type,
                "1.0",
                self.config.serving_protocol.value,
                8000,
                "/health",
                "/metrics",
                datetime.now().isoformat()
            ))
    
    def get_serving_summary(self) -> Dict[str, Any]:
        """Get serving system summary."""
        return {
            'total_endpoints': len(self.serving_api.model_endpoints),
            'serving_protocol': self.config.serving_protocol.value,
            'deployment_strategy': self.config.deployment_strategy.value,
            'scaling_policy': self.config.scaling_policy.value,
            'max_instances': self.config.max_instances,
            'min_instances': self.config.min_instances,
            'monitoring_enabled': self.config.enable_monitoring,
            'auto_scaling_enabled': self.config.enable_auto_scaling
        }
    
    def start_serving(self, host: str = "0.0.0.0", port: int = 8000):
        """Start model serving system."""
        console.print("[blue]Starting model serving system...[/blue]")
        
        # Start Prometheus metrics server
        if self.config.enable_metrics:
            start_http_server(9090)
            console.print("[green]Prometheus metrics server started on port 9090[/green]")
        
        # Start serving API
        self.serving_api.start_server(host, port)

def main():
    """Main function for model serving CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Serving System")
    parser.add_argument("--serving-protocol", type=str,
                       choices=["rest_api", "grpc", "websocket"],
                       default="rest_api", help="Serving protocol")
    parser.add_argument("--deployment-strategy", type=str,
                       choices=["single_instance", "load_balanced", "auto_scaling"],
                       default="auto_scaling", help="Deployment strategy")
    parser.add_argument("--scaling-policy", type=str,
                       choices=["cpu_based", "memory_based", "request_based"],
                       default="cpu_based", help="Scaling policy")
    parser.add_argument("--max-instances", type=int, default=10,
                       help="Maximum instances")
    parser.add_argument("--min-instances", type=int, default=1,
                       help="Minimum instances")
    parser.add_argument("--target-cpu", type=float, default=70.0,
                       help="Target CPU utilization")
    parser.add_argument("--target-memory", type=float, default=80.0,
                       help="Target memory utilization")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port")
    parser.add_argument("--model-path", type=str,
                       help="Path to model file")
    parser.add_argument("--model-type", type=str,
                       choices=["pytorch", "sklearn", "onnx", "tensorflow"],
                       default="pytorch", help="Model type")
    parser.add_argument("--endpoint-id", type=str, default="default_model",
                       help="Endpoint ID")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create model serving configuration
    config = ModelServingConfig(
        serving_protocol=ServingProtocol(args.serving_protocol),
        deployment_strategy=DeploymentStrategy(args.deployment_strategy),
        scaling_policy=ScalingPolicy(args.scaling_policy),
        max_instances=args.max_instances,
        min_instances=args.min_instances,
        target_cpu_utilization=args.target_cpu,
        target_memory_utilization=args.target_memory,
        device=args.device
    )
    
    # Create model serving system
    serving_system = ModelServingSystem(config)
    
    # Deploy model if provided
    if args.model_path:
        success = serving_system.deploy_model(args.endpoint_id, args.model_path, args.model_type)
        if not success:
            console.print("[red]Model deployment failed[/red]")
            return
    
    # Show summary
    summary = serving_system.get_serving_summary()
    console.print(f"[blue]Serving Summary: {summary}[/blue]")
    
    # Start serving
    serving_system.start_serving(args.host, args.port)

if __name__ == "__main__":
    main()
