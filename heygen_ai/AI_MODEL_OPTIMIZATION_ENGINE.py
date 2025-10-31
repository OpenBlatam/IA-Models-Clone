#!/usr/bin/env python3
"""
🧠 HeyGen AI - Advanced AI Model Optimization Engine
===================================================

This module provides comprehensive AI model optimization capabilities including:
- Model compilation and quantization
- Performance optimization and benchmarking
- Memory management and optimization
- Model versioning and management
- Automated optimization pipelines
"""

import asyncio
import logging
import time
import json
import pickle
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import queue
import hashlib
import secrets

# AI/ML imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    transforms = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationLevel(str, Enum):
    """Model optimization levels"""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    ULTRA = "ultra"
    QUANTUM = "quantum"

class ModelType(str, Enum):
    """Model types"""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    VAE = "vae"
    GAN = "gan"
    DIFFUSION = "diffusion"
    CUSTOM = "custom"

class QuantizationType(str, Enum):
    """Quantization types"""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training
    INT8 = "int8"
    INT16 = "int16"
    FP16 = "fp16"
    BF16 = "bf16"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    inference_time: float = 0.0
    memory_usage: float = 0.0
    model_size: float = 0.0
    accuracy: float = 0.0
    throughput: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationConfig:
    """Model optimization configuration"""
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    enable_compilation: bool = True
    enable_quantization: bool = True
    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    enable_pruning: bool = True
    pruning_ratio: float = 0.1
    enable_distillation: bool = False
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    batch_size: int = 32
    sequence_length: int = 512
    max_memory_usage: float = 0.8
    target_device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

@dataclass
class ModelInfo:
    """Model information and metadata"""
    model_id: str
    name: str
    model_type: ModelType
    version: str
    file_path: str
    config: OptimizationConfig
    metrics: ModelMetrics
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    description: str = ""

class ModelCompilationEngine:
    """Advanced model compilation engine"""
    
    def __init__(self):
        self.compiled_models = {}
        self.compilation_cache = {}
    
    def compile_model(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Compile model for optimal performance"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping compilation")
            return model
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(model, config)
            
            if cache_key in self.compilation_cache:
                logger.info("Using cached compiled model")
                return self.compilation_cache[cache_key]
            
            compiled_model = model
            
            # PyTorch 2.0+ compilation
            if hasattr(torch, 'compile') and config.enable_compilation:
                try:
                    compiled_model = torch.compile(
                        model,
                        mode="max-autotune" if config.optimization_level == OptimizationLevel.QUANTUM else "default"
                    )
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            # JIT compilation for older PyTorch versions
            elif hasattr(torch, 'jit') and config.enable_compilation:
                try:
                    compiled_model = torch.jit.script(model)
                    logger.info("Model compiled with torch.jit.script")
                except Exception as e:
                    logger.warning(f"torch.jit.script failed: {e}")
            
            # Cache compiled model
            self.compilation_cache[cache_key] = compiled_model
            
            return compiled_model
            
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            return model
    
    def _generate_cache_key(self, model: nn.Module, config: OptimizationConfig) -> str:
        """Generate cache key for model and config"""
        model_str = str(model.state_dict())
        config_str = f"{config.optimization_level}_{config.enable_compilation}_{config.enable_quantization}"
        return hashlib.md5(f"{model_str}_{config_str}".encode()).hexdigest()[:16]

class ModelQuantizationEngine:
    """Advanced model quantization engine"""
    
    def __init__(self):
        self.quantized_models = {}
    
    def quantize_model(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Quantize model for reduced size and faster inference"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping quantization")
            return model
        
        try:
            quantized_model = model
            
            if config.enable_quantization:
                if config.quantization_type == QuantizationType.DYNAMIC:
                    quantized_model = torch.quantization.quantize_dynamic(
                        model, 
                        {nn.Linear, nn.Conv2d}, 
                        dtype=torch.qint8
                    )
                    logger.info("Model quantized with dynamic quantization")
                
                elif config.quantization_type == QuantizationType.STATIC:
                    # Prepare model for static quantization
                    model.eval()
                    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                    torch.quantization.prepare(model, inplace=True)
                    
                    # Calibrate model (would need calibration data in real implementation)
                    # model = torch.quantization.convert(model, inplace=True)
                    logger.info("Model prepared for static quantization")
                
                elif config.quantization_type in [QuantizationType.FP16, QuantizationType.BF16]:
                    if config.target_device == "cuda":
                        dtype = torch.float16 if config.quantization_type == QuantizationType.FP16 else torch.bfloat16
                        quantized_model = model.half() if dtype == torch.float16 else model.to(dtype)
                        logger.info(f"Model converted to {config.quantization_type}")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return model

class ModelPruningEngine:
    """Advanced model pruning engine"""
    
    def __init__(self):
        self.pruned_models = {}
    
    def prune_model(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Prune model to reduce size and improve efficiency"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping pruning")
            return model
        
        try:
            if config.enable_pruning and config.pruning_ratio > 0:
                # Simple magnitude-based pruning
                parameters_to_prune = []
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        parameters_to_prune.append((module, 'weight'))
                
                if parameters_to_prune:
                    torch.nn.utils.prune.global_unstructured(
                        parameters_to_prune,
                        pruning_method=torch.nn.utils.prune.L1Unstructured,
                        amount=config.pruning_ratio,
                    )
                    logger.info(f"Model pruned with ratio {config.pruning_ratio}")
            
            return model
            
        except Exception as e:
            logger.error(f"Model pruning failed: {e}")
            return model

class ModelBenchmarkingEngine:
    """Advanced model benchmarking engine"""
    
    def __init__(self):
        self.benchmark_results = {}
    
    async def benchmark_model(self, 
                            model: nn.Module, 
                            config: OptimizationConfig,
                            test_data: Optional[Any] = None,
                            num_iterations: int = 100) -> ModelMetrics:
        """Benchmark model performance"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning dummy metrics")
            return ModelMetrics()
        
        try:
            device = torch.device(config.target_device)
            model = model.to(device)
            model.eval()
            
            # Generate dummy data if not provided
            if test_data is None:
                if config.model_type == ModelType.TRANSFORMER:
                    test_data = torch.randint(0, 1000, (config.batch_size, config.sequence_length)).to(device)
                elif config.model_type in [ModelType.CNN, ModelType.CUSTOM]:
                    test_data = torch.randn(config.batch_size, 3, 224, 224).to(device)
                else:
                    test_data = torch.randn(config.batch_size, config.sequence_length, 512).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_data)
            
            # Benchmark inference time
            inference_times = []
            memory_usage = []
            
            with torch.no_grad():
                for i in range(num_iterations):
                    start_time = time.time()
                    
                    if config.target_device == "cuda":
                        torch.cuda.synchronize()
                    
                    _ = model(test_data)
                    
                    if config.target_device == "cuda":
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    inference_times.append(end_time - start_time)
                    
                    # Memory usage
                    if config.target_device == "cuda":
                        memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
            
            # Calculate metrics
            inference_times = np.array(inference_times) if NUMPY_AVAILABLE else inference_times
            memory_usage = np.array(memory_usage) if NUMPY_AVAILABLE else memory_usage
            
            metrics = ModelMetrics(
                inference_time=np.mean(inference_times) if NUMPY_AVAILABLE else sum(inference_times) / len(inference_times),
                memory_usage=np.mean(memory_usage) if NUMPY_AVAILABLE else sum(memory_usage) / len(memory_usage),
                model_size=self._calculate_model_size(model),
                throughput=config.batch_size / np.mean(inference_times) if NUMPY_AVAILABLE else config.batch_size / (sum(inference_times) / len(inference_times)),
                latency_p50=np.percentile(inference_times, 50) if NUMPY_AVAILABLE else inference_times[len(inference_times)//2],
                latency_p95=np.percentile(inference_times, 95) if NUMPY_AVAILABLE else inference_times[int(len(inference_times)*0.95)],
                latency_p99=np.percentile(inference_times, 99) if NUMPY_AVAILABLE else inference_times[int(len(inference_times)*0.99)]
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model benchmarking failed: {e}")
            return ModelMetrics()
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            return (param_size + buffer_size) / 1024 / 1024  # MB
        except Exception:
            return 0.0

class ModelVersionManager:
    """Advanced model version management system"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.model_registry = {}
        self._load_model_registry()
    
    def _load_model_registry(self):
        """Load model registry from disk"""
        registry_file = self.models_dir / "model_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self.model_registry = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model registry: {e}")
    
    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_file = self.models_dir / "model_registry.json"
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def register_model(self, model_info: ModelInfo) -> str:
        """Register a new model"""
        model_id = model_info.model_id or secrets.token_urlsafe(16)
        model_info.model_id = model_id
        
        self.model_registry[model_id] = {
            'name': model_info.name,
            'model_type': model_info.model_type.value,
            'version': model_info.version,
            'file_path': model_info.file_path,
            'created_at': model_info.created_at.isoformat(),
            'updated_at': model_info.updated_at.isoformat(),
            'tags': model_info.tags,
            'description': model_info.description,
            'config': {
                'optimization_level': model_info.config.optimization_level.value,
                'enable_compilation': model_info.config.enable_compilation,
                'enable_quantization': model_info.config.enable_quantization,
                'quantization_type': model_info.config.quantization_type.value,
                'enable_pruning': model_info.config.enable_pruning,
                'pruning_ratio': model_info.config.pruning_ratio
            },
            'metrics': {
                'inference_time': model_info.metrics.inference_time,
                'memory_usage': model_info.metrics.memory_usage,
                'model_size': model_info.metrics.model_size,
                'accuracy': model_info.metrics.accuracy,
                'throughput': model_info.metrics.throughput
            }
        }
        
        self._save_model_registry()
        return model_id
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID"""
        if model_id not in self.model_registry:
            return None
        
        registry_data = self.model_registry[model_id]
        
        # Reconstruct ModelInfo
        config = OptimizationConfig(
            optimization_level=OptimizationLevel(registry_data['config']['optimization_level']),
            enable_compilation=registry_data['config']['enable_compilation'],
            enable_quantization=registry_data['config']['enable_quantization'],
            quantization_type=QuantizationType(registry_data['config']['quantization_type']),
            enable_pruning=registry_data['config']['enable_pruning'],
            pruning_ratio=registry_data['config']['pruning_ratio']
        )
        
        metrics = ModelMetrics(
            inference_time=registry_data['metrics']['inference_time'],
            memory_usage=registry_data['metrics']['memory_usage'],
            model_size=registry_data['metrics']['model_size'],
            accuracy=registry_data['metrics']['accuracy'],
            throughput=registry_data['metrics']['throughput']
        )
        
        return ModelInfo(
            model_id=model_id,
            name=registry_data['name'],
            model_type=ModelType(registry_data['model_type']),
            version=registry_data['version'],
            file_path=registry_data['file_path'],
            config=config,
            metrics=metrics,
            created_at=datetime.fromisoformat(registry_data['created_at']),
            updated_at=datetime.fromisoformat(registry_data['updated_at']),
            tags=registry_data['tags'],
            description=registry_data['description']
        )
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelInfo]:
        """List all registered models"""
        models = []
        for model_id in self.model_registry:
            model_info = self.get_model(model_id)
            if model_info and (model_type is None or model_info.model_type == model_type):
                models.append(model_info)
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model from registry"""
        if model_id not in self.model_registry:
            return False
        
        del self.model_registry[model_id]
        self._save_model_registry()
        return True

class AIModelOptimizationEngine:
    """Main AI model optimization engine"""
    
    def __init__(self, models_dir: str = "models"):
        self.compilation_engine = ModelCompilationEngine()
        self.quantization_engine = ModelQuantizationEngine()
        self.pruning_engine = ModelPruningEngine()
        self.benchmarking_engine = ModelBenchmarkingEngine()
        self.version_manager = ModelVersionManager(models_dir)
        self.initialized = False
    
    async def initialize(self):
        """Initialize the optimization engine"""
        try:
            logger.info("🧠 Initializing AI Model Optimization Engine...")
            
            # Check dependencies
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available - some features will be limited")
            
            if not NUMPY_AVAILABLE:
                logger.warning("NumPy not available - some features will be limited")
            
            self.initialized = True
            logger.info("✅ AI Model Optimization Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI Model Optimization Engine: {e}")
            raise
    
    async def optimize_model(self, 
                           model: nn.Module,
                           model_name: str,
                           model_type: ModelType,
                           config: OptimizationConfig,
                           save_model: bool = True) -> Tuple[nn.Module, ModelInfo]:
        """Comprehensive model optimization"""
        if not self.initialized:
            raise RuntimeError("Optimization engine not initialized")
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning original model")
            return model, None
        
        try:
            logger.info(f"🔧 Optimizing model: {model_name}")
            
            # Start with original model
            optimized_model = model
            
            # Apply optimizations based on level
            if config.optimization_level in [OptimizationLevel.STANDARD, OptimizationLevel.ADVANCED, OptimizationLevel.ULTRA, OptimizationLevel.QUANTUM]:
                # Compilation
                optimized_model = self.compilation_engine.compile_model(optimized_model, config)
                
                # Quantization
                optimized_model = self.quantization_engine.quantize_model(optimized_model, config)
                
                # Pruning
                optimized_model = self.pruning_engine.prune_model(optimized_model, config)
            
            # Benchmark optimized model
            logger.info("📊 Benchmarking optimized model...")
            metrics = await self.benchmarking_engine.benchmark_model(optimized_model, config)
            
            # Create model info
            model_id = secrets.token_urlsafe(16)
            model_info = ModelInfo(
                model_id=model_id,
                name=model_name,
                model_type=model_type,
                version="1.0.0",
                file_path=f"{self.version_manager.models_dir}/{model_name}_{model_id}.pt",
                config=config,
                metrics=metrics,
                tags=[config.optimization_level.value, model_type.value],
                description=f"Optimized {model_type.value} model with {config.optimization_level.value} optimization"
            )
            
            # Save model if requested
            if save_model:
                await self._save_model(optimized_model, model_info)
                self.version_manager.register_model(model_info)
            
            logger.info(f"✅ Model optimization completed: {model_name}")
            logger.info(f"   Inference time: {metrics.inference_time:.4f}s")
            logger.info(f"   Memory usage: {metrics.memory_usage:.2f} MB")
            logger.info(f"   Model size: {metrics.model_size:.2f} MB")
            logger.info(f"   Throughput: {metrics.throughput:.2f} samples/s")
            
            return optimized_model, model_info
            
        except Exception as e:
            logger.error(f"❌ Model optimization failed: {e}")
            raise
    
    async def _save_model(self, model: nn.Module, model_info: ModelInfo):
        """Save model to disk"""
        try:
            if TORCH_AVAILABLE:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_info': model_info,
                    'timestamp': datetime.now().isoformat()
                }, model_info.file_path)
                logger.info(f"Model saved to: {model_info.file_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    async def load_model(self, model_id: str) -> Optional[nn.Module]:
        """Load model from disk"""
        try:
            model_info = self.version_manager.get_model(model_id)
            if not model_info:
                return None
            
            if TORCH_AVAILABLE and Path(model_info.file_path).exists():
                checkpoint = torch.load(model_info.file_path, map_location='cpu')
                return checkpoint['model_state_dict']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    async def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models"""
        comparison = {
            'models': [],
            'best_performance': None,
            'best_memory': None,
            'best_size': None
        }
        
        for model_id in model_ids:
            model_info = self.version_manager.get_model(model_id)
            if model_info:
                comparison['models'].append({
                    'model_id': model_id,
                    'name': model_info.name,
                    'metrics': model_info.metrics.__dict__
                })
        
        if comparison['models']:
            # Find best performers
            models = comparison['models']
            comparison['best_performance'] = min(models, key=lambda m: m['metrics']['inference_time'])
            comparison['best_memory'] = min(models, key=lambda m: m['metrics']['memory_usage'])
            comparison['best_size'] = min(models, key=lambda m: m['metrics']['model_size'])
        
        return comparison
    
    async def get_optimization_recommendations(self, model_info: ModelInfo) -> List[str]:
        """Get optimization recommendations for a model"""
        recommendations = []
        
        # Performance recommendations
        if model_info.metrics.inference_time > 0.1:  # > 100ms
            recommendations.append("Consider enabling model compilation for faster inference")
        
        if model_info.metrics.memory_usage > 1000:  # > 1GB
            recommendations.append("Consider quantization to reduce memory usage")
        
        if model_info.metrics.model_size > 100:  # > 100MB
            recommendations.append("Consider pruning to reduce model size")
        
        # Configuration recommendations
        if not model_info.config.enable_compilation:
            recommendations.append("Enable model compilation for better performance")
        
        if not model_info.config.enable_quantization:
            recommendations.append("Enable quantization to reduce memory usage and improve speed")
        
        if not model_info.config.enable_pruning:
            recommendations.append("Enable pruning to reduce model size")
        
        return recommendations
    
    async def shutdown(self):
        """Shutdown the optimization engine"""
        self.initialized = False
        logger.info("✅ AI Model Optimization Engine shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the AI model optimization engine"""
    print("🧠 HeyGen AI - AI Model Optimization Engine Demo")
    print("=" * 60)
    
    # Initialize optimization engine
    engine = AIModelOptimizationEngine()
    
    try:
        # Initialize the engine
        print("\n🚀 Initializing Optimization Engine...")
        await engine.initialize()
        print("✅ Optimization Engine initialized successfully")
        
        if TORCH_AVAILABLE:
            # Create a simple model for demonstration
            print("\n🏗️ Creating Demo Model...")
            class SimpleTransformer(nn.Module):
                def __init__(self, vocab_size=1000, d_model=512, nhead=8, num_layers=6):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, d_model)
                    self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
                    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                    self.output_projection = nn.Linear(d_model, vocab_size)
                
                def forward(self, x):
                    x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
                    x = self.transformer(x)
                    return self.output_projection(x)
            
            model = SimpleTransformer()
            print(f"✅ Demo model created: {sum(p.numel() for p in model.parameters())} parameters")
            
            # Test different optimization levels
            optimization_levels = [
                OptimizationLevel.BASIC,
                OptimizationLevel.STANDARD,
                OptimizationLevel.ADVANCED,
                OptimizationLevel.ULTRA
            ]
            
            optimized_models = []
            
            for level in optimization_levels:
                print(f"\n🔧 Optimizing with {level.value} level...")
                
                config = OptimizationConfig(
                    optimization_level=level,
                    enable_compilation=True,
                    enable_quantization=True,
                    enable_pruning=True,
                    pruning_ratio=0.1,
                    batch_size=16,
                    sequence_length=128
                )
                
                optimized_model, model_info = await engine.optimize_model(
                    model=model,
                    model_name=f"demo_model_{level.value}",
                    model_type=ModelType.TRANSFORMER,
                    config=config,
                    save_model=True
                )
                
                if model_info:
                    optimized_models.append(model_info.model_id)
                    print(f"  ✅ Optimized: {model_info.name}")
                    print(f"  📊 Inference time: {model_info.metrics.inference_time:.4f}s")
                    print(f"  💾 Memory usage: {model_info.metrics.memory_usage:.2f} MB")
                    print(f"  📦 Model size: {model_info.metrics.model_size:.2f} MB")
            
            # Compare models
            if optimized_models:
                print(f"\n📊 Comparing {len(optimized_models)} models...")
                comparison = await engine.compare_models(optimized_models)
                
                print(f"  🏆 Best Performance: {comparison['best_performance']['name']}")
                print(f"  💾 Best Memory: {comparison['best_memory']['name']}")
                print(f"  📦 Best Size: {comparison['best_size']['name']}")
            
            # Get recommendations
            if optimized_models:
                print(f"\n💡 Optimization Recommendations:")
                model_info = engine.version_manager.get_model(optimized_models[0])
                if model_info:
                    recommendations = await engine.get_optimization_recommendations(model_info)
                    for i, rec in enumerate(recommendations, 1):
                        print(f"  {i}. {rec}")
        
        else:
            print("\n⚠️ PyTorch not available - demo limited to basic functionality")
        
        # List registered models
        print(f"\n📋 Registered Models:")
        models = engine.version_manager.list_models()
        for model in models:
            print(f"  - {model.name} ({model.model_type.value}) - {model.version}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
    
    finally:
        # Shutdown
        await engine.shutdown()
        print("\n✅ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())


