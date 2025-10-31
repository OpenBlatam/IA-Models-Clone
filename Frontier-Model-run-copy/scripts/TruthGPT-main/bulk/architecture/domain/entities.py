#!/usr/bin/env python3
"""
Domain Entities - Core business entities with rich behavior
Implements Domain-Driven Design entities with business logic
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
import enum
from enum import Enum

class Entity(ABC):
    """Base entity class with identity and equality."""
    
    def __init__(self, entity_id: str = None):
        self._id = entity_id or str(uuid.uuid4())
        self._created_at = datetime.now(timezone.utc)
        self._updated_at = datetime.now(timezone.utc)
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def created_at(self) -> datetime:
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        return self._updated_at
    
    def touch(self):
        """Update the last modified timestamp."""
        self._updated_at = datetime.now(timezone.utc)
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self._id == other._id
    
    def __hash__(self):
        return hash(self._id)

class OptimizationType(Enum):
    """Types of optimization strategies."""
    SPEED = "speed"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    BALANCED = "balanced"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    TRANSFORMER = "transformer"
    LLM = "llm"
    DIFFUSION = "diffusion"
    QUANTUM = "quantum"

class OptimizationStatus(Enum):
    """Status of optimization task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PerformanceScore:
    """Value object for performance scores."""
    speed_score: float
    memory_score: float
    accuracy_score: float
    efficiency_score: float
    
    def __post_init__(self):
        # Validate scores are between 0 and 1
        for score_name, score in [
            ('speed_score', self.speed_score),
            ('memory_score', self.memory_score),
            ('accuracy_score', self.accuracy_score),
            ('efficiency_score', self.efficiency_score)
        ]:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{score_name} must be between 0.0 and 1.0, got {score}")
    
    @property
    def overall_score(self) -> float:
        """Calculate overall performance score."""
        return (self.speed_score + self.memory_score + self.accuracy_score + self.efficiency_score) / 4.0
    
    def is_improved(self, other: 'PerformanceScore') -> bool:
        """Check if this score is improved over another."""
        return self.overall_score > other.overall_score

@dataclass
class ResourceUsage:
    """Value object for resource usage."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    execution_time: float
    
    def __post_init__(self):
        # Validate usage values
        if self.cpu_usage < 0 or self.cpu_usage > 100:
            raise ValueError(f"CPU usage must be between 0 and 100, got {self.cpu_usage}")
        if self.memory_usage < 0:
            raise ValueError(f"Memory usage must be non-negative, got {self.memory_usage}")
        if self.gpu_usage < 0 or self.gpu_usage > 100:
            raise ValueError(f"GPU usage must be between 0 and 100, got {self.gpu_usage}")
        if self.execution_time < 0:
            raise ValueError(f"Execution time must be non-negative, got {self.execution_time}")
    
    def is_efficient(self, threshold: float = 0.8) -> bool:
        """Check if resource usage is efficient."""
        return (self.cpu_usage + self.gpu_usage) / 2.0 <= threshold * 100

class ModelProfile(Entity):
    """Domain entity representing a model profile."""
    
    def __init__(self, model: nn.Module, model_name: str, entity_id: str = None):
        super().__init__(entity_id)
        self._model_name = model_name
        self._model = model
        self._total_parameters = sum(p.numel() for p in model.parameters())
        self._trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._memory_usage = self._calculate_memory_usage()
        self._complexity_score = self._calculate_complexity_score()
        self._architecture_info = self._analyze_architecture()
        self._performance_metrics = self._calculate_performance_metrics()
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def model(self) -> nn.Module:
        return self._model
    
    @property
    def total_parameters(self) -> int:
        return self._total_parameters
    
    @property
    def trainable_parameters(self) -> int:
        return self._trainable_parameters
    
    @property
    def memory_usage(self) -> float:
        return self._memory_usage
    
    @property
    def complexity_score(self) -> float:
        return self._complexity_score
    
    @property
    def architecture_info(self) -> Dict[str, Any]:
        return self._architecture_info
    
    @property
    def performance_metrics(self) -> Dict[str, float]:
        return self._performance_metrics
    
    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage in MB."""
        try:
            return sum(p.numel() * p.element_size() for p in self._model.parameters()) / (1024**2)
        except Exception:
            return 0.0
    
    def _calculate_complexity_score(self) -> float:
        """Calculate model complexity score."""
        try:
            # Normalize complexity based on parameters and layers
            param_complexity = min(1.0, self._total_parameters / 10000000)  # 10M params = 1.0
            layer_complexity = min(1.0, len(list(self._model.modules())) / 100)  # 100 layers = 1.0
            return (param_complexity + layer_complexity) / 2.0
        except Exception:
            return 0.0
    
    def _analyze_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture."""
        try:
            architecture_info = {
                'num_layers': len(list(self._model.modules())),
                'has_conv_layers': any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in self._model.modules()),
                'has_linear_layers': any(isinstance(m, nn.Linear) for m in self._model.modules()),
                'has_attention': any('attention' in str(type(m)).lower() for m in self._model.modules()),
                'has_dropout': any(isinstance(m, nn.Dropout) for m in self._model.modules()),
                'has_batch_norm': any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in self._model.modules()),
                'has_activation': any(isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU, nn.Swish)) for m in self._model.modules())
            }
            return architecture_info
        except Exception:
            return {}
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        try:
            # Simple performance estimation
            total_params = self._total_parameters
            
            # Estimate inference speed (simplified)
            estimated_speed = max(0.1, 1.0 / (total_params / 1000000))
            
            # Estimate memory efficiency
            memory_efficiency = min(1.0, 1000000 / max(total_params, 1))
            
            return {
                'estimated_speed': estimated_speed,
                'memory_efficiency': memory_efficiency,
                'parameter_efficiency': min(1.0, 1000000 / max(total_params, 1))
            }
        except Exception:
            return {}
    
    def is_suitable_for_optimization(self, optimization_type: OptimizationType) -> bool:
        """Check if model is suitable for specific optimization type."""
        try:
            if optimization_type == OptimizationType.SPEED:
                return self._complexity_score > 0.3
            elif optimization_type == OptimizationType.MEMORY:
                return self._memory_usage > 100  # MB
            elif optimization_type == OptimizationType.ACCURACY:
                return self._total_parameters > 100000  # 100K params
            elif optimization_type == OptimizationType.QUANTIZATION:
                return self._total_parameters > 1000000  # 1M params
            elif optimization_type == OptimizationType.PRUNING:
                return self._total_parameters > 500000  # 500K params
            else:
                return True
        except Exception:
            return False
    
    def get_optimization_recommendations(self) -> List[OptimizationType]:
        """Get optimization recommendations based on model characteristics."""
        recommendations = []
        
        try:
            # Speed optimization
            if self._complexity_score > 0.5:
                recommendations.append(OptimizationType.SPEED)
            
            # Memory optimization
            if self._memory_usage > 500:  # MB
                recommendations.append(OptimizationType.MEMORY)
            
            # Quantization
            if self._total_parameters > 1000000:
                recommendations.append(OptimizationType.QUANTIZATION)
            
            # Pruning
            if self._total_parameters > 500000:
                recommendations.append(OptimizationType.PRUNING)
            
            # Always recommend balanced optimization
            recommendations.append(OptimizationType.BALANCED)
            
        except Exception:
            recommendations = [OptimizationType.BALANCED]
        
        return recommendations

class OptimizationTask(Entity):
    """Domain entity representing an optimization task."""
    
    def __init__(self, model_profile: ModelProfile, optimization_type: OptimizationType,
                 target_improvement: float = 0.5, priority: int = 1, entity_id: str = None):
        super().__init__(entity_id)
        self._model_profile = model_profile
        self._optimization_type = optimization_type
        self._target_improvement = target_improvement
        self._priority = priority
        self._status = OptimizationStatus.PENDING
        self._applied_strategies = []
        self._performance_scores = None
        self._resource_usage = None
        self._error_message = None
        self._started_at = None
        self._completed_at = None
    
    @property
    def model_profile(self) -> ModelProfile:
        return self._model_profile
    
    @property
    def optimization_type(self) -> OptimizationType:
        return self._optimization_type
    
    @property
    def target_improvement(self) -> float:
        return self._target_improvement
    
    @property
    def priority(self) -> int:
        return self._priority
    
    @property
    def status(self) -> OptimizationStatus:
        return self._status
    
    @property
    def applied_strategies(self) -> List[str]:
        return self._applied_strategies.copy()
    
    @property
    def performance_scores(self) -> Optional[PerformanceScore]:
        return self._performance_scores
    
    @property
    def resource_usage(self) -> Optional[ResourceUsage]:
        return self._resource_usage
    
    @property
    def error_message(self) -> Optional[str]:
        return self._error_message
    
    @property
    def started_at(self) -> Optional[datetime]:
        return self._started_at
    
    @property
    def completed_at(self) -> Optional[datetime]:
        return self._completed_at
    
    def start(self):
        """Start the optimization task."""
        if self._status != OptimizationStatus.PENDING:
            raise ValueError(f"Cannot start task in status {self._status}")
        
        self._status = OptimizationStatus.RUNNING
        self._started_at = datetime.now(timezone.utc)
        self.touch()
    
    def add_strategy(self, strategy_name: str):
        """Add an applied strategy."""
        if self._status != OptimizationStatus.RUNNING:
            raise ValueError(f"Cannot add strategy to task in status {self._status}")
        
        self._applied_strategies.append(strategy_name)
        self.touch()
    
    def complete(self, performance_scores: PerformanceScore, resource_usage: ResourceUsage):
        """Complete the optimization task successfully."""
        if self._status != OptimizationStatus.RUNNING:
            raise ValueError(f"Cannot complete task in status {self._status}")
        
        self._status = OptimizationStatus.COMPLETED
        self._performance_scores = performance_scores
        self._resource_usage = resource_usage
        self._completed_at = datetime.now(timezone.utc)
        self.touch()
    
    def fail(self, error_message: str):
        """Mark the optimization task as failed."""
        if self._status != OptimizationStatus.RUNNING:
            raise ValueError(f"Cannot fail task in status {self._status}")
        
        self._status = OptimizationStatus.FAILED
        self._error_message = error_message
        self._completed_at = datetime.now(timezone.utc)
        self.touch()
    
    def cancel(self):
        """Cancel the optimization task."""
        if self._status not in [OptimizationStatus.PENDING, OptimizationStatus.RUNNING]:
            raise ValueError(f"Cannot cancel task in status {self._status}")
        
        self._status = OptimizationStatus.CANCELLED
        self._completed_at = datetime.now(timezone.utc)
        self.touch()
    
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self._status == OptimizationStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if task is failed."""
        return self._status == OptimizationStatus.FAILED
    
    def is_running(self) -> bool:
        """Check if task is running."""
        return self._status == OptimizationStatus.RUNNING
    
    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self._started_at and self._completed_at:
            return (self._completed_at - self._started_at).total_seconds()
        return None
    
    def meets_target_improvement(self) -> bool:
        """Check if task meets target improvement."""
        if not self._performance_scores:
            return False
        
        return self._performance_scores.overall_score >= self._target_improvement

class OptimizationResult(Entity):
    """Domain entity representing an optimization result."""
    
    def __init__(self, task: OptimizationTask, success: bool, 
                 improvement_score: float, execution_time: float,
                 resource_usage: ResourceUsage, entity_id: str = None):
        super().__init__(entity_id)
        self._task = task
        self._success = success
        self._improvement_score = improvement_score
        self._execution_time = execution_time
        self._resource_usage = resource_usage
        self._metadata = {}
    
    @property
    def task(self) -> OptimizationTask:
        return self._task
    
    @property
    def success(self) -> bool:
        return self._success
    
    @property
    def improvement_score(self) -> float:
        return self._improvement_score
    
    @property
    def execution_time(self) -> float:
        return self._execution_time
    
    @property
    def resource_usage(self) -> ResourceUsage:
        return self._resource_usage
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the result."""
        self._metadata[key] = value
        self.touch()
    
    def is_successful(self) -> bool:
        """Check if optimization was successful."""
        return self._success and self._improvement_score > 0
    
    def get_efficiency_ratio(self) -> float:
        """Calculate efficiency ratio (improvement per time)."""
        if self._execution_time <= 0:
            return 0.0
        return self._improvement_score / self._execution_time
