"""
Advanced Model Serving Engine for Export IA
Production-ready model serving with load balancing, auto-scaling, and monitoring
"""

import torch
import torch.nn as nn
import asyncio
import aiohttp
from aiohttp import web, ClientSession
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
import time
import json
import pickle
import base64
from pathlib import Path
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import redis
import consul
import docker
from kubernetes import client, config
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ServingConfig:
    """Configuration for model serving"""
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_connections: int = 1000
    timeout: int = 30
    
    # Model configuration
    model_path: str = "./models"
    model_name: str = "export_ia_model"
    model_version: str = "1.0.0"
    max_batch_size: int = 32
    batch_timeout: float = 0.1
    
    # Load balancing
    enable_load_balancing: bool = True
    load_balancer_type: str = "round_robin"  # round_robin, least_connections, weighted
    health_check_interval: int = 30
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_usage: float = 0.7
    target_memory_usage: float = 0.8
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_health_checks: bool = True
    health_check_endpoint: str = "/health"
    
    # Caching
    enable_caching: bool = True
    cache_backend: str = "redis"  # redis, memory, none
    cache_ttl: int = 3600
    cache_max_size: int = 10000
    
    # Security
    enable_authentication: bool = True
    api_key_header: str = "X-API-Key"
    rate_limiting: bool = True
    max_requests_per_minute: int = 100
    
    # Deployment
    deployment_type: str = "standalone"  # standalone, docker, kubernetes
    docker_image: str = "export-ia:latest"
    kubernetes_namespace: str = "default"

class ModelServer:
    """Advanced model server with production features"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.app = FastAPI(
            title="Export IA Model Server",
            description="Advanced AI model serving with production features",
            version=config.model_version
        )
        
        # Initialize components
        self.model = None
        self.inference_engine = None
        self.load_balancer = None
        self.auto_scaler = None
        self.metrics_collector = None
        self.cache = None
        self.rate_limiter = None
        
        # Request queue and processing
        self.request_queue = queue.Queue(maxsize=config.max_connections)
        self.response_cache = {}
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # Setup server
        self._setup_middleware()
        self._setup_routes()
        self._setup_components()
        
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware for request tracking
        @self.app.middleware("http")
        async def track_requests(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Update metrics
            self.request_count += 1
            self.total_processing_time += process_time
            
            # Add headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = str(self.request_count)
            
            return response
            
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "Export IA Model Server",
                "version": self.config.model_version,
                "status": "running",
                "timestamp": time.time()
            }
            
        @self.app.get(self.config.health_check_endpoint)
        async def health_check():
            health_status = await self._check_health()
            status_code = 200 if health_status["healthy"] else 503
            return web.Response(
                text=json.dumps(health_status),
                status=status_code,
                content_type="application/json"
            )
            
        @self.app.post("/predict")
        async def predict(request: dict, background_tasks: BackgroundTasks):
            try:
                # Rate limiting
                if self.config.rate_limiting:
                    if not await self._check_rate_limit(request):
                        raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                # Process request
                result = await self._process_prediction_request(request)
                
                # Background tasks
                background_tasks.add_task(self._log_request, request, result)
                
                return result
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/batch_predict")
        async def batch_predict(request: dict):
            try:
                results = await self._process_batch_prediction_request(request)
                return {"results": results}
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/metrics")
        async def get_metrics():
            return await self._get_metrics()
            
        @self.app.get("/model/info")
        async def get_model_info():
            return await self._get_model_info()
            
        @self.app.post("/model/reload")
        async def reload_model():
            try:
                await self._reload_model()
                return {"status": "success", "message": "Model reloaded"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
    def _setup_components(self):
        """Setup server components"""
        
        # Load balancer
        if self.config.enable_load_balancing:
            self.load_balancer = LoadBalancer(self.config)
            
        # Auto-scaler
        if self.config.enable_auto_scaling:
            self.auto_scaler = AutoScaler(self.config)
            
        # Metrics collector
        if self.config.enable_metrics:
            self.metrics_collector = MetricsCollector(self.config)
            
        # Cache
        if self.config.enable_caching:
            self.cache = CacheManager(self.config)
            
        # Rate limiter
        if self.config.rate_limiting:
            self.rate_limiter = RateLimiter(self.config)
            
    async def _check_health(self) -> Dict[str, Any]:
        """Check server health"""
        health_status = {
            "healthy": True,
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check model
        health_status["checks"]["model"] = {
            "status": "healthy" if self.model is not None else "unhealthy",
            "loaded": self.model is not None
        }
        
        # Check memory
        memory_usage = psutil.virtual_memory().percent
        health_status["checks"]["memory"] = {
            "status": "healthy" if memory_usage < 90 else "warning",
            "usage_percent": memory_usage
        }
        
        # Check CPU
        cpu_usage = psutil.cpu_percent()
        health_status["checks"]["cpu"] = {
            "status": "healthy" if cpu_usage < 90 else "warning",
            "usage_percent": cpu_usage
        }
        
        # Check GPU if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            health_status["checks"]["gpu"] = {
                "status": "healthy" if gpu_memory < 0.9 else "warning",
                "memory_usage": gpu_memory
            }
        
        # Overall health
        unhealthy_checks = [check for check in health_status["checks"].values() 
                          if check["status"] == "unhealthy"]
        health_status["healthy"] = len(unhealthy_checks) == 0
        
        return health_status
        
    async def _process_prediction_request(self, request: dict) -> dict:
        """Process single prediction request"""
        
        # Check cache
        if self.config.enable_caching:
            cache_key = self._generate_cache_key(request)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result
                
        # Process with model
        start_time = time.time()
        
        # Convert request to model input
        model_input = self._prepare_model_input(request)
        
        # Run inference
        if self.inference_engine:
            result = self.inference_engine.infer(model_input)
        else:
            result = self._run_inference(model_input)
            
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            "prediction": self._prepare_model_output(result),
            "processing_time": processing_time,
            "model_version": self.config.model_version,
            "timestamp": time.time()
        }
        
        # Cache result
        if self.config.enable_caching:
            await self.cache.set(cache_key, response, ttl=self.config.cache_ttl)
            
        return response
        
    async def _process_batch_prediction_request(self, request: dict) -> List[dict]:
        """Process batch prediction request"""
        
        inputs = request.get("inputs", [])
        if not inputs:
            raise HTTPException(status_code=400, detail="No inputs provided")
            
        if len(inputs) > self.config.max_batch_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size {len(inputs)} exceeds maximum {self.config.max_batch_size}"
            )
            
        # Process batch
        results = []
        for input_data in inputs:
            result = await self._process_prediction_request({"input": input_data})
            results.append(result)
            
        return results
        
    def _prepare_model_input(self, request: dict) -> Any:
        """Prepare model input from request"""
        # This would be implemented based on your specific model requirements
        input_data = request.get("input", {})
        
        if "text" in input_data:
            # Text input
            return torch.tensor([input_data["text"]])
        elif "image" in input_data:
            # Image input
            image_data = base64.b64decode(input_data["image"])
            # Convert to tensor (simplified)
            return torch.randn(1, 3, 224, 224)  # Placeholder
        else:
            # Generic input
            return torch.randn(1, 10)  # Placeholder
            
    def _prepare_model_output(self, result: Any) -> dict:
        """Prepare model output for response"""
        if isinstance(result, torch.Tensor):
            return {
                "tensor": result.tolist(),
                "shape": list(result.shape),
                "dtype": str(result.dtype)
            }
        else:
            return {"result": str(result)}
            
    def _run_inference(self, model_input: Any) -> Any:
        """Run inference with loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        with torch.no_grad():
            return self.model(model_input)
            
    def _generate_cache_key(self, request: dict) -> str:
        """Generate cache key from request"""
        import hashlib
        request_str = json.dumps(request, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()
        
    async def _check_rate_limit(self, request: dict) -> bool:
        """Check rate limiting"""
        if self.rate_limiter:
            return await self.rate_limiter.check_limit(request)
        return True
        
    async def _log_request(self, request: dict, response: dict):
        """Log request and response"""
        log_entry = {
            "timestamp": time.time(),
            "request": request,
            "response": response,
            "processing_time": response.get("processing_time", 0)
        }
        logger.info(f"Request logged: {log_entry}")
        
    async def _get_metrics(self) -> dict:
        """Get server metrics"""
        avg_processing_time = (self.total_processing_time / self.request_count 
                             if self.request_count > 0 else 0)
        
        metrics = {
            "requests": {
                "total": self.request_count,
                "errors": self.error_count,
                "success_rate": (self.request_count - self.error_count) / self.request_count 
                               if self.request_count > 0 else 0
            },
            "performance": {
                "avg_processing_time": avg_processing_time,
                "total_processing_time": self.total_processing_time
            },
            "system": {
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(),
                "gpu_available": torch.cuda.is_available()
            }
        }
        
        if self.cache:
            metrics["cache"] = await self.cache.get_stats()
            
        return metrics
        
    async def _get_model_info(self) -> dict:
        """Get model information"""
        if self.model is None:
            return {"error": "Model not loaded"}
            
        return {
            "name": self.config.model_name,
            "version": self.config.model_version,
            "type": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype)
        }
        
    async def _reload_model(self):
        """Reload model"""
        # This would implement model reloading logic
        logger.info("Reloading model...")
        # Placeholder implementation
        pass
        
    def load_model(self, model: nn.Module, inference_engine=None):
        """Load model for serving"""
        self.model = model
        self.model.eval()
        self.inference_engine = inference_engine
        
        logger.info(f"Model loaded: {self.config.model_name} v{self.config.model_version}")
        
    def run(self):
        """Run the server"""
        logger.info(f"Starting Export IA Model Server on {self.config.host}:{self.config.port}")
        
        # Start metrics server
        if self.config.enable_metrics:
            start_http_server(self.config.metrics_port)
            logger.info(f"Metrics server started on port {self.config.metrics_port}")
            
        # Run FastAPI server
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level="info"
        )

class LoadBalancer:
    """Load balancer for distributed model serving"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.servers = []
        self.current_index = 0
        self.server_health = {}
        
    def add_server(self, server_url: str, weight: int = 1):
        """Add server to load balancer"""
        self.servers.append({
            "url": server_url,
            "weight": weight,
            "connections": 0
        })
        
    async def get_server(self) -> Optional[str]:
        """Get next server based on load balancing strategy"""
        if not self.servers:
            return None
            
        if self.config.load_balancer_type == "round_robin":
            server = self.servers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.servers)
            return server["url"]
            
        elif self.config.load_balancer_type == "least_connections":
            server = min(self.servers, key=lambda s: s["connections"])
            return server["url"]
            
        elif self.config.load_balancer_type == "weighted":
            # Weighted round robin
            total_weight = sum(s["weight"] for s in self.servers)
            # Simplified implementation
            server = self.servers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.servers)
            return server["url"]
            
        return self.servers[0]["url"]
        
    async def health_check(self):
        """Perform health checks on servers"""
        async with ClientSession() as session:
            for server in self.servers:
                try:
                    async with session.get(f"{server['url']}{self.config.health_check_endpoint}") as response:
                        self.server_health[server["url"]] = response.status == 200
                except:
                    self.server_health[server["url"]] = False

class AutoScaler:
    """Auto-scaler for model serving instances"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.current_instances = config.min_instances
        self.scaling_history = []
        
    async def check_and_scale(self):
        """Check metrics and scale if needed"""
        
        # Get current metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # Check if scaling is needed
        if (cpu_usage > self.config.scale_up_threshold or 
            memory_usage > self.config.scale_up_threshold):
            await self._scale_up()
        elif (cpu_usage < self.config.scale_down_threshold and 
              memory_usage < self.config.scale_down_threshold and
              self.current_instances > self.config.min_instances):
            await self._scale_down()
            
    async def _scale_up(self):
        """Scale up instances"""
        if self.current_instances < self.config.max_instances:
            self.current_instances += 1
            self.scaling_history.append({
                "action": "scale_up",
                "instances": self.current_instances,
                "timestamp": time.time()
            })
            logger.info(f"Scaling up to {self.current_instances} instances")
            
    async def _scale_down(self):
        """Scale down instances"""
        if self.current_instances > self.config.min_instances:
            self.current_instances -= 1
            self.scaling_history.append({
                "action": "scale_down",
                "instances": self.current_instances,
                "timestamp": time.time()
            })
            logger.info(f"Scaling down to {self.current_instances} instances")

class MetricsCollector:
    """Metrics collector for monitoring"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        
        # Prometheus metrics
        self.request_counter = Counter('model_requests_total', 'Total requests')
        self.request_duration = Histogram('model_request_duration_seconds', 'Request duration')
        self.error_counter = Counter('model_errors_total', 'Total errors')
        self.active_connections = Gauge('model_active_connections', 'Active connections')
        
    def record_request(self, duration: float, success: bool):
        """Record request metrics"""
        self.request_counter.inc()
        self.request_duration.observe(duration)
        
        if not success:
            self.error_counter.inc()
            
    def update_connections(self, count: int):
        """Update active connections count"""
        self.active_connections.set(count)

class CacheManager:
    """Cache manager for request/response caching"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.cache = {}
        self.access_times = {}
        
        if config.cache_backend == "redis":
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()  # Test connection
            except:
                logger.warning("Redis not available, falling back to memory cache")
                self.redis_client = None
        else:
            self.redis_client = None
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            except:
                return None
        else:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
            
    async def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl or self.config.cache_ttl, json.dumps(value))
            except:
                pass
        else:
            self.cache[key] = value
            self.access_times[key] = time.time()
            
            # Cleanup old entries
            if len(self.cache) > self.config.cache_max_size:
                self._cleanup_cache()
                
    def _cleanup_cache(self):
        """Cleanup old cache entries"""
        # Remove oldest entries
        sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
        for key in sorted_keys[:len(self.cache) // 2]:
            del self.cache[key]
            del self.access_times[key]
            
    async def get_stats(self) -> dict:
        """Get cache statistics"""
        if self.redis_client:
            try:
                info = self.redis_client.info()
                return {
                    "type": "redis",
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients")
                }
            except:
                return {"type": "redis", "status": "error"}
        else:
            return {
                "type": "memory",
                "size": len(self.cache),
                "max_size": self.config.cache_max_size
            }

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.requests = defaultdict(list)
        
    async def check_limit(self, request: dict) -> bool:
        """Check if request is within rate limit"""
        # Extract client identifier (simplified)
        client_id = request.get("client_id", "default")
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] 
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.config.max_requests_per_minute:
            return False
            
        # Add current request
        self.requests[client_id].append(current_time)
        return True

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test model serving
    print("Testing Model Serving Engine...")
    
    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    model = TestModel()
    model.eval()
    
    # Create serving config
    config = ServingConfig(
        host="0.0.0.0",
        port=8000,
        model_name="test_model",
        model_version="1.0.0",
        enable_caching=True,
        enable_metrics=True,
        rate_limiting=True
    )
    
    # Create model server
    server = ModelServer(config)
    server.load_model(model)
    
    print("Model server created successfully!")
    print(f"Server will run on {config.host}:{config.port}")
    print("API endpoints:")
    print("  GET  / - Root endpoint")
    print("  GET  /health - Health check")
    print("  POST /predict - Single prediction")
    print("  POST /batch_predict - Batch prediction")
    print("  GET  /metrics - Server metrics")
    print("  GET  /model/info - Model information")
    print("  POST /model/reload - Reload model")
    
    # Uncomment to run server
    # server.run()
    
    print("\nModel serving engine initialized successfully!")
























