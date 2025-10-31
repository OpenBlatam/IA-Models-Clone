#!/usr/bin/env python3
"""
Advanced Model Manager for Enhanced HeyGen AI
Manages multiple AI models with load balancing, automatic switching, and performance optimization.
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import psutil
import torch
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

logger = structlog.get_logger()

class ModelType(Enum):
    """Types of AI models."""
    STABLE_DIFFUSION = "stable_diffusion"
    STABLE_DIFFUSION_XL = "stable_diffusion_xl"
    COQUI_TTS = "coqui_tts"
    YOUR_TTS = "your_tts"
    WAV2LIP = "wav2lip"
    FACE_RECOGNITION = "face_recognition"
    EMOTION_DETECTION = "emotion_detection"

class ModelStatus(Enum):
    """Model status values."""
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNLOADING = "unloading"
    OFFLINE = "offline"

class ModelPriority(Enum):
    """Model priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str
    model_type: ModelType
    model_path: str
    device: str
    precision: str  # "fp16", "fp32", "int8"
    max_batch_size: int
    memory_limit_gb: float
    priority: ModelPriority
    auto_load: bool = True
    enable_quantization: bool = False
    enable_optimization: bool = True
    fallback_models: List[str] = None
    health_check_interval: int = 300  # seconds

@dataclass
class ModelInstance:
    """Represents a loaded model instance."""
    config: ModelConfig
    status: ModelStatus
    model: Any
    memory_usage_bytes: int
    last_used: float
    usage_count: int
    error_count: int
    last_error: Optional[str]
    performance_metrics: Dict[str, float]
    health_score: float
    is_primary: bool
    load_time: float
    last_health_check: float

@dataclass
class ModelRequest:
    """Represents a model inference request."""
    request_id: str
    model_type: ModelType
    priority: ModelPriority
    payload: Dict[str, Any]
    timeout: float
    created_at: float
    user_id: Optional[str]
    callback: Optional[Callable]

class AdvancedModelManager:
    """Advanced model manager with load balancing and optimization."""
    
    def __init__(
        self,
        models_dir: str = "./models",
        max_concurrent_models: int = 5,
        enable_auto_scaling: bool = True,
        enable_model_caching: bool = True,
        health_check_interval: int = 60
    ):
        self.models_dir = Path(models_dir)
        self.max_concurrent_models = max_concurrent_models
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_model_caching = enable_model_caching
        self.health_check_interval = health_check_interval
        
        # Model storage
        self.model_configs: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, ModelInstance] = {}
        self.model_queues: Dict[ModelType, asyncio.Queue] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.model_usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Load balancing
        self.load_balancer = ModelLoadBalancer()
        self.auto_scaler = ModelAutoScaler(self)
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Thread pool for model operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize model configurations
        self._initialize_default_models()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_default_models(self):
        """Initialize default model configurations."""
        default_models = [
            ModelConfig(
                model_id="sd-1.5",
                model_type=ModelType.STABLE_DIFFUSION,
                model_path="stable-diffusion-v1-5",
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision="fp16",
                max_batch_size=4,
                memory_limit_gb=4.0,
                priority=ModelPriority.HIGH
            ),
            ModelConfig(
                model_id="sd-xl",
                model_type=ModelType.STABLE_DIFFUSION_XL,
                model_path="stable-diffusion-xl-base-1.0",
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision="fp16",
                max_batch_size=2,
                memory_limit_gb=8.0,
                priority=ModelPriority.HIGH
            ),
            ModelConfig(
                model_id="coqui-tts",
                model_type=ModelType.COQUI_TTS,
                model_path="tts_models/en/ljspeech/tacotron2-DDC",
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision="fp32",
                max_batch_size=8,
                memory_limit_gb=2.0,
                priority=ModelPriority.NORMAL
            ),
            ModelConfig(
                model_id="wav2lip",
                model_type=ModelType.WAV2LIP,
                model_path="wav2lip_gan.pth",
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision="fp32",
                max_batch_size=1,
                memory_limit_gb=3.0,
                priority=ModelPriority.NORMAL
            )
        ]
        
        for config in default_models:
            self.model_configs[config.model_id] = config
            self.model_usage_stats[config.model_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0.0,
                "last_used": 0.0
            }
    
    def _start_background_tasks(self):
        """Start background tasks."""
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def register_model(self, config: ModelConfig) -> bool:
        """Register a new model configuration."""
        try:
            self.model_configs[config.model_id] = config
            self.model_usage_stats[config.model_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0.0,
                "last_used": 0.0
            }
            
            logger.info(f"Model registered", 
                       model_id=config.model_id,
                       type=config.model_type.value)
            
            # Auto-load if configured
            if config.auto_load:
                await self.load_model(config.model_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {config.model_id}: {e}")
            return False
    
    async def load_model(self, model_id: str) -> bool:
        """Load a model into memory."""
        if model_id not in self.model_configs:
            logger.error(f"Model {model_id} not found in configurations")
            return False
        
        if model_id in self.loaded_models:
            logger.warning(f"Model {model_id} is already loaded")
            return True
        
        config = self.model_configs[model_id]
        
        # Check memory constraints
        if not await self._check_memory_constraints(config):
            logger.warning(f"Insufficient memory to load model {model_id}")
            return False
        
        try:
            # Update status
            instance = ModelInstance(
                config=config,
                status=ModelStatus.LOADING,
                model=None,
                memory_usage_bytes=0,
                last_used=time.time(),
                usage_count=0,
                error_count=0,
                last_error=None,
                performance_metrics={},
                health_score=1.0,
                is_primary=False,
                load_time=time.time(),
                last_health_check=time.time()
            )
            
            self.loaded_models[model_id] = instance
            
            # Load model in thread pool
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.thread_pool,
                self._load_model_in_thread,
                config
            )
            
            if model:
                instance.model = model
                instance.status = ModelStatus.READY
                instance.memory_usage_bytes = await self._estimate_memory_usage(config)
                
                logger.info(f"Model loaded successfully", 
                           model_id=model_id,
                           memory_usage_mb=instance.memory_usage_bytes / (1024 * 1024))
                
                return True
            else:
                instance.status = ModelStatus.ERROR
                instance.last_error = "Failed to load model"
                logger.error(f"Failed to load model {model_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            if model_id in self.loaded_models:
                self.loaded_models[model_id].status = ModelStatus.ERROR
                self.loaded_models[model_id].last_error = str(e)
            return False
    
    def _load_model_in_thread(self, config: ModelConfig) -> Any:
        """Load model in a separate thread."""
        try:
            # This is a placeholder for actual model loading logic
            # In a real implementation, you would load the specific model type
            logger.info(f"Loading model {config.model_id} in thread")
            
            # Simulate model loading
            time.sleep(2)
            
            # Return a mock model object
            return {"model_id": config.model_id, "type": config.model_type.value}
            
        except Exception as e:
            logger.error(f"Thread model loading failed: {e}")
            return None
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        if model_id not in self.loaded_models:
            return True
        
        try:
            instance = self.loaded_models[model_id]
            instance.status = ModelStatus.UNLOADING
            
            # Clean up model resources
            if instance.model:
                # This would include proper cleanup for the specific model type
                pass
            
            # Remove from loaded models
            del self.loaded_models[model_id]
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Model unloaded", model_id=model_id)
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")
            return False
    
    async def get_model(self, model_type: ModelType, priority: ModelPriority = ModelPriority.NORMAL) -> Optional[ModelInstance]:
        """Get an available model instance."""
        available_models = [
            instance for instance in self.loaded_models.values()
            if (instance.config.model_type == model_type and 
                instance.status == ModelStatus.READY)
        ]
        
        if not available_models:
            # Try to load a model of this type
            configs = [c for c in self.model_configs.values() if c.model_type == model_type]
            if configs:
                # Load the highest priority config
                configs.sort(key=lambda c: c.priority.value, reverse=True)
                await self.load_model(configs[0].model_id)
                return await self.get_model(model_type, priority)
            return None
        
        # Use load balancer to select the best model
        return self.load_balancer.select_model(available_models, priority)
    
    async def execute_model_request(
        self,
        model_type: ModelType,
        payload: Dict[str, Any],
        priority: ModelPriority = ModelPriority.NORMAL,
        timeout: float = 30.0,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a model inference request."""
        request_id = str(hashlib.md5(f"{time.time()}:{payload}".encode()).hexdigest())
        
        # Get available model
        model_instance = await self.get_model(model_type, priority)
        if not model_instance:
            raise RuntimeError(f"No available model for type {model_type.value}")
        
        start_time = time.time()
        
        try:
            # Update usage stats
            model_instance.usage_count += 1
            model_instance.last_used = time.time()
            model_instance.status = ModelStatus.BUSY
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.thread_pool,
                    self._execute_model_inference,
                    model_instance,
                    payload
                ),
                timeout=timeout
            )
            
            # Update performance metrics
            duration = time.time() - start_time
            self._update_performance_metrics(model_instance, duration, True)
            
            # Update usage stats
            self.model_usage_stats[model_instance.config.model_id]["successful_requests"] += 1
            self.model_usage_stats[model_instance.config.model_id]["total_requests"] += 1
            
            model_instance.status = ModelStatus.READY
            
            return {
                "request_id": request_id,
                "result": result,
                "model_id": model_instance.config.model_id,
                "duration": duration,
                "status": "success"
            }
            
        except Exception as e:
            # Update error metrics
            model_instance.error_count += 1
            model_instance.last_error = str(e)
            self._update_performance_metrics(model_instance, time.time() - start_time, False)
            
            # Update usage stats
            self.model_usage_stats[model_instance.config.model_id]["failed_requests"] += 1
            self.model_usage_stats[model_instance.config.model_id]["total_requests"] += 1
            
            model_instance.status = ModelStatus.READY
            
            logger.error(f"Model inference failed", 
                        model_id=model_instance.config.model_id,
                        error=str(e))
            
            raise
    
    def _execute_model_inference(self, model_instance: ModelInstance, payload: Dict[str, Any]) -> Any:
        """Execute model inference in thread."""
        try:
            # This is a placeholder for actual model inference
            # In a real implementation, you would execute the specific model
            logger.info(f"Executing inference on model {model_instance.config.model_id}")
            
            # Simulate inference
            time.sleep(1)
            
            # Return mock result
            return {
                "output": f"Result from {model_instance.config.model_id}",
                "confidence": 0.95,
                "processing_time": 1.0
            }
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise
    
    def _update_performance_metrics(self, model_instance: ModelInstance, duration: float, success: bool):
        """Update performance metrics for a model."""
        model_id = model_instance.config.model_id
        
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        self.performance_history[model_id].append(duration)
        
        # Keep only last 100 measurements
        if len(self.performance_history[model_id]) > 100:
            self.performance_history[model_id] = self.performance_history[model_id][-100:]
        
        # Update average response time
        avg_time = sum(self.performance_history[model_id]) / len(self.performance_history[model_id])
        self.model_usage_stats[model_id]["average_response_time"] = avg_time
        
        # Update health score
        if success:
            model_instance.health_score = min(1.0, model_instance.health_score + 0.01)
        else:
            model_instance.health_score = max(0.0, model_instance.health_score - 0.1)
    
    async def _check_memory_constraints(self, config: ModelConfig) -> bool:
        """Check if there's enough memory to load the model."""
        try:
            available_memory = psutil.virtual_memory().available
            required_memory = config.memory_limit_gb * 1024 * 1024 * 1024
            
            # Check if we have enough memory
            if available_memory < required_memory:
                # Try to unload some models
                await self._unload_low_priority_models(required_memory - available_memory)
                
                # Check again
                available_memory = psutil.virtual_memory().available
                return available_memory >= required_memory
            
            return True
            
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False
    
    async def _unload_low_priority_models(self, required_memory: int):
        """Unload low priority models to free memory."""
        # Sort models by priority and usage
        models_to_unload = sorted(
            self.loaded_models.values(),
            key=lambda m: (m.config.priority.value, m.last_used)
        )
        
        freed_memory = 0
        for model in models_to_unload:
            if freed_memory >= required_memory:
                break
            
            if model.status == ModelStatus.READY:
                await self.unload_model(model.config.model_id)
                freed_memory += model.memory_usage_bytes
    
    async def _estimate_memory_usage(self, config: ModelConfig) -> int:
        """Estimate memory usage for a model."""
        # This is a simplified estimation
        base_memory = config.memory_limit_gb * 1024 * 1024 * 1024
        
        if config.precision == "fp16":
            base_memory *= 0.5
        elif config.precision == "int8":
            base_memory *= 0.25
        
        return int(base_memory)
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self):
        """Perform health checks on all loaded models."""
        current_time = time.time()
        
        for model_id, instance in self.loaded_models.items():
            if current_time - instance.last_health_check >= instance.config.health_check_interval:
                await self._check_model_health(instance)
                instance.last_health_check = current_time
    
    async def _check_model_health(self, instance: ModelInstance):
        """Check health of a specific model."""
        try:
            # This would include actual health checks for the specific model type
            # For now, we'll just update the health score based on recent performance
            
            if instance.error_count > 5:
                instance.health_score = max(0.0, instance.health_score - 0.2)
            
            # If health score is too low, consider reloading
            if instance.health_score < 0.3:
                logger.warning(f"Model {instance.config.model_id} has low health score, considering reload")
                
        except Exception as e:
            logger.error(f"Health check failed for {instance.config.model_id}: {e}")
    
    async def _optimization_loop(self):
        """Background optimization loop."""
        while True:
            try:
                await self._perform_optimizations()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_optimizations(self):
        """Perform system optimizations."""
        try:
            # Auto-scaling
            if self.enable_auto_scaling:
                await self.auto_scaler.optimize()
            
            # Memory optimization
            await self._optimize_memory_usage()
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage."""
        try:
            # Check if we need to unload unused models
            current_time = time.time()
            unused_threshold = 1800  # 30 minutes
            
            for model_id, instance in list(self.loaded_models.items()):
                if (current_time - instance.last_used > unused_threshold and
                    instance.config.priority != ModelPriority.CRITICAL):
                    logger.info(f"Unloading unused model {model_id}")
                    await self.unload_model(model_id)
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(600)  # Every 10 minutes
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_cleanup(self):
        """Perform system cleanup."""
        try:
            # Clean up old performance history
            current_time = time.time()
            cutoff_time = current_time - 86400  # 24 hours
            
            for model_id, history in list(self.performance_history.items()):
                # Remove old measurements
                self.performance_history[model_id] = [
                    t for t in history if current_time - t < cutoff_time
                ]
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {}
        
        for model_id, instance in self.loaded_models.items():
            status[model_id] = {
                "status": instance.status.value,
                "type": instance.config.model_type.value,
                "priority": instance.config.priority.value,
                "memory_usage_mb": instance.memory_usage_bytes / (1024 * 1024),
                "usage_count": instance.usage_count,
                "error_count": instance.error_count,
                "health_score": instance.health_score,
                "last_used": instance.last_used,
                "is_primary": instance.is_primary
            }
        
        return status
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        total_models = len(self.loaded_models)
        ready_models = sum(1 for m in self.loaded_models.values() if m.status == ModelStatus.READY)
        busy_models = sum(1 for m in self.loaded_models.values() if m.status == ModelStatus.BUSY)
        error_models = sum(1 for m in self.loaded_models.values() if m.status == ModelStatus.ERROR)
        
        total_memory = sum(m.memory_usage_bytes for m in self.loaded_models.values())
        
        return {
            "total_models": total_models,
            "ready_models": ready_models,
            "busy_models": busy_models,
            "error_models": error_models,
            "total_memory_usage_mb": total_memory / (1024 * 1024),
            "model_usage_stats": self.model_usage_stats,
            "performance_history": {
                model_id: {
                    "count": len(history),
                    "average": sum(history) / len(history) if history else 0
                }
                for model_id, history in self.performance_history.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown the model manager."""
        # Cancel background tasks
        for task in [self.health_check_task, self.optimization_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Unload all models
        for model_id in list(self.loaded_models.keys()):
            await self.unload_model(model_id)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Advanced model manager shutdown complete")


class ModelLoadBalancer:
    """Load balancer for model selection."""
    
    def select_model(self, available_models: List[ModelInstance], priority: ModelPriority) -> ModelInstance:
        """Select the best model based on load balancing strategy."""
        if not available_models:
            return None
        
        # Simple round-robin with priority consideration
        # In a real implementation, you might use more sophisticated algorithms
        
        # Sort by priority first, then by health score, then by current load
        sorted_models = sorted(
            available_models,
            key=lambda m: (
                m.config.priority.value,
                m.health_score,
                -m.usage_count  # Lower usage count = higher priority
            ),
            reverse=True
        )
        
        return sorted_models[0]


class ModelAutoScaler:
    """Auto-scaling for models based on demand."""
    
    def __init__(self, model_manager: AdvancedModelManager):
        self.model_manager = model_manager
        self.scaling_history: List[Dict[str, Any]] = []
    
    async def optimize(self):
        """Optimize model scaling based on current demand."""
        try:
            # Analyze current demand
            demand_analysis = await self._analyze_demand()
            
            # Make scaling decisions
            scaling_decisions = await self._make_scaling_decisions(demand_analysis)
            
            # Execute scaling decisions
            await self._execute_scaling_decisions(scaling_decisions)
            
        except Exception as e:
            logger.error(f"Auto-scaling optimization failed: {e}")
    
    async def _analyze_demand(self) -> Dict[str, Any]:
        """Analyze current demand patterns."""
        # This would analyze request patterns, queue lengths, etc.
        return {
            "high_demand_models": [],
            "low_demand_models": [],
            "scaling_recommendations": []
        }
    
    async def _make_scaling_decisions(self, demand_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make scaling decisions based on demand analysis."""
        # This would implement scaling logic
        return []
    
    async def _execute_scaling_decisions(self, decisions: List[Dict[str, Any]]):
        """Execute scaling decisions."""
        # This would implement the actual scaling actions
        pass


# Global model manager instance
model_manager: Optional[AdvancedModelManager] = None

def get_model_manager() -> AdvancedModelManager:
    """Get global model manager instance."""
    global model_manager
    if model_manager is None:
        model_manager = AdvancedModelManager()
    return model_manager

async def shutdown_model_manager():
    """Shutdown global model manager."""
    global model_manager
    if model_manager:
        await model_manager.shutdown()
        model_manager = None

