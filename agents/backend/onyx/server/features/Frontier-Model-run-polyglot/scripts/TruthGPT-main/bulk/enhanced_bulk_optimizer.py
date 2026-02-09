#!/usr/bin/env python3
"""
Enhanced Bulk Optimizer - Advanced bulk optimization with AI-powered features
Enhanced with machine learning, adaptive optimization, and intelligent resource management
"""

import torch
import torch.nn as nn
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import time
import json
import logging
import threading
import queue
import concurrent.futures
from pathlib import Path
import psutil
import gc
from collections import defaultdict, deque
import pickle
import hashlib
from datetime import datetime, timezone
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Import enhanced components
from enhanced_production_config import EnhancedProductionConfig, create_enhanced_production_config
from production_logging import create_production_logger
from production_monitoring import create_production_monitor

@dataclass
class ModelProfile:
    """Model profile for optimization."""
    model_id: str
    model_type: str
    parameters: int
    memory_usage: float
    complexity_score: float
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_optimized: Optional[datetime] = None

@dataclass
class OptimizationStrategy:
    """Optimization strategy configuration."""
    name: str
    priority: int
    conditions: Dict[str, Any]
    parameters: Dict[str, Any]
    success_rate: float = 0.0
    avg_improvement: float = 0.0
    execution_time: float = 0.0

@dataclass
class IntelligentConfig:
    """Intelligent optimization configuration."""
    enable_ml_optimization: bool = True
    enable_adaptive_scheduling: bool = True
    enable_resource_prediction: bool = True
    enable_performance_learning: bool = True
    ml_model_path: str = "models/optimization_predictor.pkl"
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    memory_threshold: float = 0.8
    cpu_threshold: float = 0.8
    batch_size_adaptation: bool = True
    strategy_evolution: bool = True

class ModelAnalyzer:
    """Advanced model analysis and profiling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.feature_names = [
            'parameters', 'memory_usage', 'complexity_score', 'model_type_encoded',
            'input_size', 'output_size', 'num_layers', 'activation_type'
        ]
    
    def analyze_model(self, model: nn.Module, model_name: str) -> ModelProfile:
        """Analyze model and create profile."""
        try:
            # Calculate basic metrics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate memory usage
            memory_usage = self._estimate_memory_usage(model)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(model)
            
            # Determine model type
            model_type = self._classify_model_type(model)
            
            # Create profile
            profile = ModelProfile(
                model_id=str(uuid.uuid4()),
                model_type=model_type,
                parameters=total_params,
                memory_usage=memory_usage,
                complexity_score=complexity_score,
                performance_metrics={
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'memory_usage_mb': memory_usage,
                    'complexity_score': complexity_score
                }
            )
            
            self.logger.info(f"Analyzed model {model_name}: {total_params} params, {memory_usage:.2f}MB")
            return profile
            
        except Exception as e:
            self.logger.error(f"Error analyzing model {model_name}: {e}")
            return ModelProfile(
                model_id=str(uuid.uuid4()),
                model_type="unknown",
                parameters=0,
                memory_usage=0.0,
                complexity_score=0.0
            )
    
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate model memory usage in MB."""
        try:
            # Calculate parameter memory
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            
            # Calculate buffer memory
            buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
            
            # Add overhead (typically 20-30%)
            total_memory = (param_memory + buffer_memory) * 1.25
            
            return total_memory / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def _calculate_complexity_score(self, model: nn.Module) -> float:
        """Calculate model complexity score."""
        try:
            # Count layers
            num_layers = len(list(model.modules())) - 1  # Exclude the model itself
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            # Calculate depth
            max_depth = self._get_model_depth(model)
            
            # Complexity score (normalized)
            complexity = (num_params / 1e6) * (num_layers / 10) * (max_depth / 5)
            return min(complexity, 10.0)  # Cap at 10
        except:
            return 0.0
    
    def _get_model_depth(self, model: nn.Module) -> int:
        """Get maximum depth of the model."""
        def get_depth(module, current_depth=0):
            if not list(module.children()):
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in module.children())
        
        return get_depth(model)
    
    def _classify_model_type(self, model: nn.Module) -> str:
        """Classify model type."""
        model_str = str(model).lower()
        
        if 'conv' in model_str:
            return 'cnn'
        elif 'lstm' in model_str or 'gru' in model_str:
            return 'rnn'
        elif 'transformer' in model_str or 'attention' in model_str:
            return 'transformer'
        elif 'linear' in model_str:
            return 'mlp'
        else:
            return 'custom'

class MLOptimizationPredictor:
    """Machine learning-based optimization predictor."""
    
    def __init__(self, config: IntelligentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.training_data = []
        self.is_trained = False
        
        # Load existing model if available
        if Path(config.ml_model_path).exists():
            self._load_model()
    
    def _load_model(self):
        """Load pre-trained model."""
        try:
            model_data = joblib.load(self.config.ml_model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = True
            self.logger.info("Loaded pre-trained optimization model")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def _save_model(self):
        """Save trained model."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'training_data': self.training_data
            }
            joblib.dump(model_data, self.config.ml_model_path)
            self.logger.info("Saved optimization model")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def predict_optimization_strategy(self, model_profile: ModelProfile, 
                                    system_resources: Dict[str, float]) -> OptimizationStrategy:
        """Predict optimal optimization strategy."""
        if not self.is_trained:
            return self._get_default_strategy()
        
        try:
            # Prepare features
            features = self._extract_features(model_profile, system_resources)
            features_scaled = self.scaler.transform([features])
            
            # Predict strategy
            strategy_id = self.model.predict(features_scaled)[0]
            
            return self._get_strategy_by_id(strategy_id)
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self._get_default_strategy()
    
    def _extract_features(self, model_profile: ModelProfile, 
                         system_resources: Dict[str, float]) -> List[float]:
        """Extract features for ML model."""
        features = [
            model_profile.parameters / 1e6,  # Normalize parameters
            model_profile.memory_usage,
            model_profile.complexity_score,
            hash(model_profile.model_type) % 1000,  # Encode model type
            system_resources.get('cpu_usage', 0.0),
            system_resources.get('memory_usage', 0.0),
            system_resources.get('gpu_usage', 0.0),
            system_resources.get('available_memory', 0.0)
        ]
        return features
    
    def _get_default_strategy(self) -> OptimizationStrategy:
        """Get default optimization strategy."""
        return OptimizationStrategy(
            name="default",
            priority=1,
            conditions={},
            parameters={
                'enable_quantization': True,
                'enable_pruning': True,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        )
    
    def _get_strategy_by_id(self, strategy_id: int) -> OptimizationStrategy:
        """Get strategy by ID."""
        strategies = {
            0: OptimizationStrategy(
                name="memory_optimized",
                priority=1,
                conditions={'memory_usage': 'high'},
                parameters={'enable_quantization': True, 'batch_size': 16}
            ),
            1: OptimizationStrategy(
                name="speed_optimized",
                priority=2,
                conditions={'complexity_score': 'high'},
                parameters={'enable_pruning': True, 'batch_size': 64}
            ),
            2: OptimizationStrategy(
                name="balanced",
                priority=3,
                conditions={},
                parameters={'enable_quantization': True, 'enable_pruning': True, 'batch_size': 32}
            )
        }
        return strategies.get(strategy_id, self._get_default_strategy())
    
    def learn_from_optimization(self, model_profile: ModelProfile, 
                               strategy: OptimizationStrategy,
                               results: Dict[str, Any]):
        """Learn from optimization results."""
        if not self.config.enable_performance_learning:
            return
        
        try:
            # Extract features and outcome
            features = self._extract_features(model_profile, results.get('system_resources', {}))
            outcome = results.get('success', False)
            improvement = results.get('improvement', 0.0)
            
            # Add to training data
            self.training_data.append({
                'features': features,
                'strategy_id': hash(strategy.name) % 1000,
                'outcome': outcome,
                'improvement': improvement
            })
            
            # Retrain if enough data
            if len(self.training_data) >= 100:
                self._retrain_model()
                
        except Exception as e:
            self.logger.error(f"Learning error: {e}")
    
    def _retrain_model(self):
        """Retrain the ML model."""
        try:
            if len(self.training_data) < 10:
                return
            
            # Prepare training data
            X = np.array([data['features'] for data in self.training_data])
            y = np.array([data['strategy_id'] for data in self.training_data])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            self._save_model()
            
            self.logger.info("Retrained optimization model")
        except Exception as e:
            self.logger.error(f"Retraining error: {e}")

class AdaptiveResourceManager:
    """Adaptive resource management system."""
    
    def __init__(self, config: IntelligentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.resource_history = deque(maxlen=1000)
        self.prediction_model = None
        self.current_allocation = {}
    
    def get_optimal_allocation(self, models: List[ModelProfile], 
                             system_resources: Dict[str, float]) -> Dict[str, Any]:
        """Get optimal resource allocation."""
        try:
            # Analyze current system state
            system_state = self._analyze_system_state(system_resources)
            
            # Predict resource needs
            predicted_needs = self._predict_resource_needs(models, system_state)
            
            # Optimize allocation
            allocation = self._optimize_allocation(predicted_needs, system_state)
            
            return allocation
        except Exception as e:
            self.logger.error(f"Resource allocation error: {e}")
            return self._get_default_allocation()
    
    def _analyze_system_state(self, system_resources: Dict[str, float]) -> Dict[str, Any]:
        """Analyze current system state."""
        return {
            'cpu_usage': system_resources.get('cpu_usage', 0.0),
            'memory_usage': system_resources.get('memory_usage', 0.0),
            'gpu_usage': system_resources.get('gpu_usage', 0.0),
            'available_memory': system_resources.get('available_memory', 0.0),
            'load_average': system_resources.get('load_average', 0.0)
        }
    
    def _predict_resource_needs(self, models: List[ModelProfile], 
                              system_state: Dict[str, Any]) -> Dict[str, float]:
        """Predict resource needs for models."""
        total_memory = sum(model.memory_usage for model in models)
        total_complexity = sum(model.complexity_score for model in models)
        
        # Predict based on model characteristics
        predicted_cpu = min(total_complexity * 0.1, 0.8)  # Cap at 80%
        predicted_memory = min(total_memory * 1.2, system_state['available_memory'] * 0.8)
        predicted_gpu = min(total_complexity * 0.05, 0.6)  # Cap at 60%
        
        return {
            'cpu_usage': predicted_cpu,
            'memory_usage': predicted_memory,
            'gpu_usage': predicted_gpu,
            'batch_size': self._calculate_optimal_batch_size(models, system_state)
        }
    
    def _calculate_optimal_batch_size(self, models: List[ModelProfile], 
                                     system_state: Dict[str, Any]) -> int:
        """Calculate optimal batch size."""
        if not self.config.batch_size_adaptation:
            return 32
        
        # Base batch size on available memory
        available_memory = system_state['available_memory']
        avg_model_memory = np.mean([model.memory_usage for model in models])
        
        if avg_model_memory > 0:
            optimal_batch = int(available_memory * 0.8 / avg_model_memory)
            return max(1, min(optimal_batch, 128))  # Between 1 and 128
        
        return 32
    
    def _optimize_allocation(self, predicted_needs: Dict[str, float], 
                           system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation."""
        allocation = {
            'cpu_limit': min(predicted_needs['cpu_usage'] * 1.2, 0.9),
            'memory_limit': min(predicted_needs['memory_usage'] * 1.1, system_state['available_memory']),
            'gpu_limit': min(predicted_needs['gpu_usage'] * 1.1, 0.8),
            'batch_size': predicted_needs['batch_size'],
            'workers': self._calculate_optimal_workers(predicted_needs, system_state)
        }
        
        return allocation
    
    def _calculate_optimal_workers(self, predicted_needs: Dict[str, float], 
                                 system_state: Dict[str, Any]) -> int:
        """Calculate optimal number of workers."""
        cpu_cores = psutil.cpu_count()
        cpu_usage = system_state['cpu_usage']
        
        # Calculate based on available CPU
        available_cpu = max(0, 1.0 - cpu_usage)
        optimal_workers = int(cpu_cores * available_cpu * 0.8)
        
        return max(1, min(optimal_workers, 8))  # Between 1 and 8
    
    def _get_default_allocation(self) -> Dict[str, Any]:
        """Get default resource allocation."""
        return {
            'cpu_limit': 0.8,
            'memory_limit': 4096,  # 4GB
            'gpu_limit': 0.6,
            'batch_size': 32,
            'workers': 4
        }

class EnhancedBulkOptimizer:
    """Enhanced bulk optimizer with AI-powered features."""
    
    def __init__(self, config: EnhancedProductionConfig):
        self.config = config
        self.intelligent_config = IntelligentConfig()
        self.logger = create_production_logger("enhanced_bulk_optimizer")
        self.monitor = create_production_monitor()
        
        # Initialize components
        self.model_analyzer = ModelAnalyzer()
        self.ml_predictor = MLOptimizationPredictor(self.intelligent_config)
        self.resource_manager = AdaptiveResourceManager(self.intelligent_config)
        
        # Optimization strategies
        self.strategies = self._initialize_strategies()
        
        # Performance tracking
        self.optimization_history = []
        self.performance_metrics = defaultdict(list)
        
        # Start monitoring
        self.monitor.start()
    
    def _initialize_strategies(self) -> List[OptimizationStrategy]:
        """Initialize optimization strategies."""
        return [
            OptimizationStrategy(
                name="memory_optimized",
                priority=1,
                conditions={'memory_usage': 'high'},
                parameters={
                    'enable_quantization': True,
                    'enable_pruning': True,
                    'batch_size': 16,
                    'mixed_precision': True
                }
            ),
            OptimizationStrategy(
                name="speed_optimized",
                priority=2,
                conditions={'complexity_score': 'high'},
                parameters={
                    'enable_kernel_fusion': True,
                    'enable_attention_fusion': True,
                    'batch_size': 64,
                    'parallel_processing': True
                }
            ),
            OptimizationStrategy(
                name="balanced",
                priority=3,
                conditions={},
                parameters={
                    'enable_quantization': True,
                    'enable_pruning': True,
                    'batch_size': 32,
                    'mixed_precision': True
                }
            )
        ]
    
    async def optimize_models_intelligent(self, models: List[Tuple[str, nn.Module]], 
                                        system_resources: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Intelligent bulk optimization with AI-powered features."""
        self.logger.info(f"Starting intelligent optimization of {len(models)} models")
        
        try:
            # Analyze models
            model_profiles = []
            for model_name, model in models:
                profile = self.model_analyzer.analyze_model(model, model_name)
                model_profiles.append(profile)
            
            # Get system resources
            if system_resources is None:
                system_resources = self._get_system_resources()
            
            # Get optimal resource allocation
            allocation = self.resource_manager.get_optimal_allocation(model_profiles, system_resources)
            
            # Optimize each model with intelligent strategy selection
            results = []
            for i, (model_name, model) in enumerate(models):
                profile = model_profiles[i]
                
                # Predict optimal strategy
                strategy = self.ml_predictor.predict_optimization_strategy(profile, system_resources)
                
                # Apply optimization
                result = await self._optimize_single_model_intelligent(
                    model_name, model, profile, strategy, allocation
                )
                
                results.append(result)
                
                # Learn from results
                self.ml_predictor.learn_from_optimization(profile, strategy, result)
            
            # Update performance metrics
            self._update_performance_metrics(results)
            
            self.logger.info(f"Completed intelligent optimization: {len(results)} models processed")
            return results
            
        except Exception as e:
            self.logger.error(f"Intelligent optimization failed: {e}")
            return []
    
    async def _optimize_single_model_intelligent(self, model_name: str, model: nn.Module, 
                                               profile: ModelProfile, strategy: OptimizationStrategy,
                                               allocation: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize single model with intelligent strategy."""
        start_time = time.time()
        
        try:
            # Apply strategy-specific optimizations
            optimized_model = await self._apply_strategy_optimizations(model, strategy)
            
            # Measure performance
            optimization_time = time.time() - start_time
            memory_usage = self._measure_memory_usage(optimized_model)
            performance_improvement = self._calculate_performance_improvement(model, optimized_model)
            
            result = {
                'model_name': model_name,
                'strategy': strategy.name,
                'success': True,
                'optimization_time': optimization_time,
                'memory_usage': memory_usage,
                'performance_improvement': performance_improvement,
                'parameters_before': sum(p.numel() for p in model.parameters()),
                'parameters_after': sum(p.numel() for p in optimized_model.parameters()),
                'system_resources': allocation
            }
            
            # Update strategy performance
            self._update_strategy_performance(strategy, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed for {model_name}: {e}")
            return {
                'model_name': model_name,
                'strategy': strategy.name,
                'success': False,
                'error': str(e),
                'optimization_time': time.time() - start_time
            }
    
    async def _apply_strategy_optimizations(self, model: nn.Module, 
                                         strategy: OptimizationStrategy) -> nn.Module:
        """Apply strategy-specific optimizations."""
        optimized_model = model
        
        # Apply quantization if enabled
        if strategy.parameters.get('enable_quantization', False):
            optimized_model = self._apply_quantization(optimized_model)
        
        # Apply pruning if enabled
        if strategy.parameters.get('enable_pruning', False):
            optimized_model = self._apply_pruning(optimized_model)
        
        # Apply mixed precision if enabled
        if strategy.parameters.get('mixed_precision', False):
            optimized_model = self._apply_mixed_precision(optimized_model)
        
        return optimized_model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization optimization."""
        try:
            # Simple quantization - convert to half precision
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = module.weight.data.half()
                    if module.bias is not None:
                        module.bias.data = module.bias.data.half()
            return model
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
            return model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning optimization."""
        try:
            # Simple pruning - remove small weights
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    threshold = torch.quantile(torch.abs(module.weight.data), 0.1)
                    module.weight.data[torch.abs(module.weight.data) < threshold] = 0
            return model
        except Exception as e:
            self.logger.warning(f"Pruning failed: {e}")
            return model
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Apply mixed precision optimization."""
        try:
            # Convert to half precision
            model = model.half()
            return model
        except Exception as e:
            self.logger.warning(f"Mixed precision failed: {e}")
            return model
    
    def _measure_memory_usage(self, model: nn.Module) -> float:
        """Measure model memory usage."""
        try:
            total_memory = 0
            for param in model.parameters():
                total_memory += param.numel() * param.element_size()
            return total_memory / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def _calculate_performance_improvement(self, original_model: nn.Module, 
                                         optimized_model: nn.Module) -> float:
        """Calculate performance improvement."""
        try:
            # Simple improvement calculation based on parameter reduction
            original_params = sum(p.numel() for p in original_model.parameters())
            optimized_params = sum(p.numel() for p in optimized_model.parameters())
            
            if original_params > 0:
                reduction = (original_params - optimized_params) / original_params
                return max(0, reduction)
            return 0.0
        except:
            return 0.0
    
    def _get_system_resources(self) -> Dict[str, float]:
        """Get current system resources."""
        try:
            return {
                'cpu_usage': psutil.cpu_percent() / 100.0,
                'memory_usage': psutil.virtual_memory().percent / 100.0,
                'available_memory': psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            }
        except:
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'available_memory': 8.0,
                'load_average': 0.0
            }
    
    def _update_strategy_performance(self, strategy: OptimizationStrategy, result: Dict[str, Any]):
        """Update strategy performance metrics."""
        if result['success']:
            strategy.success_rate = (strategy.success_rate + 1) / 2
            strategy.avg_improvement = (strategy.avg_improvement + result.get('performance_improvement', 0)) / 2
            strategy.execution_time = (strategy.execution_time + result['optimization_time']) / 2
    
    def _update_performance_metrics(self, results: List[Dict[str, Any]]):
        """Update performance metrics."""
        successful_results = [r for r in results if r.get('success', False)]
        
        if successful_results:
            avg_improvement = np.mean([r.get('performance_improvement', 0) for r in successful_results])
            avg_time = np.mean([r.get('optimization_time', 0) for r in successful_results])
            
            self.performance_metrics['avg_improvement'].append(avg_improvement)
            self.performance_metrics['avg_time'].append(avg_time)
            self.performance_metrics['success_rate'].append(len(successful_results) / len(results))
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'total_optimizations': len(self.optimization_history),
            'success_rate': np.mean(self.performance_metrics.get('success_rate', [0])),
            'avg_improvement': np.mean(self.performance_metrics.get('avg_improvement', [0])),
            'avg_time': np.mean(self.performance_metrics.get('avg_time', [0])),
            'strategy_performance': {
                strategy.name: {
                    'success_rate': strategy.success_rate,
                    'avg_improvement': strategy.avg_improvement,
                    'execution_time': strategy.execution_time
                }
                for strategy in self.strategies
            }
        }
    
    def save_optimization_model(self, filepath: str):
        """Save the ML optimization model."""
        try:
            self.ml_predictor._save_model()
            self.logger.info(f"Optimization model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def load_optimization_model(self, filepath: str):
        """Load the ML optimization model."""
        try:
            self.ml_predictor.config.ml_model_path = filepath
            self.ml_predictor._load_model()
            self.logger.info(f"Optimization model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

def create_enhanced_bulk_optimizer(config: Optional[Dict[str, Any]] = None) -> EnhancedBulkOptimizer:
    """Create enhanced bulk optimizer."""
    if config is None:
        config = {}
    
    from enhanced_production_config import EnhancedProductionConfig
    enhanced_config = EnhancedProductionConfig(**config)
    return EnhancedBulkOptimizer(enhanced_config)

async def optimize_models_enhanced(models: List[Tuple[str, nn.Module]], 
                                 config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Enhanced bulk optimization function."""
    optimizer = create_enhanced_bulk_optimizer(config)
    return await optimizer.optimize_models_intelligent(models)

if __name__ == "__main__":
    # Example usage
    import torch
    import torch.nn as nn
    
    # Create test models
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    models = [
        ("model_1", SimpleModel()),
        ("model_2", SimpleModel()),
        ("model_3", SimpleModel())
    ]
    
    # Run enhanced optimization
    async def main():
        results = await optimize_models_enhanced(models)
        
        print("🚀 Enhanced Bulk Optimization Results")
        print("=" * 50)
        
        for result in results:
            if result['success']:
                print(f"✅ {result['model_name']}: {result['performance_improvement']:.2%} improvement")
            else:
                print(f"❌ {result['model_name']}: {result.get('error', 'Unknown error')}")
    
    asyncio.run(main())

