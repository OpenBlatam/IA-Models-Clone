from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import Union, List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
import statistics
from collections import defaultdict
import numpy as np
from enum import Enum
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib
from ..core.math_service import MathService, MathOperation, MathResult, OperationType, CalculationMethod
from ..analytics.analytics_engine import MathAnalyticsEngine, AnalyticsMetric
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Math Optimization Engine
Automatic optimization of mathematical operations using ML and analytics with production features.
"""



logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    MEMORY = "memory"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"


class OptimizationLevel(Enum):
    """Optimization levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class OptimizationStatus(Enum):
    """Optimization status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class OptimizationRule:
    """Rule for automatic optimization with enhanced features."""
    rule_id: str
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[MathOperation], MathOperation]
    priority: int = 1
    enabled: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[datetime] = None
    performance_impact: float = 0.0
    
    def __post_init__(self) -> Any:
        if not self.rule_id:
            self.rule_id = hashlib.md5(f"{self.name}_{time.time()}".encode()).hexdigest()[:8]


@dataclass
class OptimizationResult:
    """Result of an optimization operation with enhanced metrics."""
    optimization_id: str = field(default_factory=lambda: hashlib.md5(f"{time.time()}_{random.random()}".encode()).hexdigest()[:12])
    original_operation: Optional[MathOperation] = None
    optimized_operation: Optional[MathOperation] = None
    optimization_applied: str = ""
    performance_improvement: float = 0.0
    accuracy_impact: float = 0.0
    memory_impact: float = 0.0
    optimization_time: float = 0.0
    status: OptimizationStatus = OptimizationStatus.PENDING
    success: bool = True
    error_message: Optional[str] = None
    applied_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationProfile:
    """Optimization profile for specific scenarios with enhanced configuration."""
    profile_id: str
    name: str
    strategy: OptimizationStrategy
    level: OptimizationLevel
    rules: List[OptimizationRule] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    average_improvement: float = 0.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformancePrediction:
    """Performance prediction for optimization decisions."""
    operation_type: str
    method: str
    predicted_time: float
    confidence: float
    factors: Dict[str, float]
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationMetrics:
    """Optimization performance metrics."""
    total_optimizations: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    average_improvement: float = 0.0
    total_time_saved: float = 0.0
    cache_hit_rate: float = 0.0
    rule_effectiveness: Dict[str, float] = field(default_factory=dict)
    profile_usage: Dict[str, int] = field(default_factory=dict)


class MathOptimizationEngine:
    """Advanced optimization engine for mathematical operations with production features."""
    
    def __init__(self, math_service: Optional[MathService] = None, 
                 analytics_engine: Optional[MathAnalyticsEngine] = None,
                 max_optimization_history: int = 10000,
                 enable_ml: bool = True):
        
    """__init__ function."""
self.math_service = math_service
        self.analytics_engine = analytics_engine
        self.max_optimization_history = max_optimization_history
        
        # Core components
        self.optimization_rules: List[OptimizationRule] = []
        self.optimization_profiles: Dict[str, OptimizationProfile] = {}
        self.optimization_history: List[OptimizationResult] = []
        self.optimization_cache: Dict[str, OptimizationResult] = {}
        
        # Performance tracking
        self.performance_baseline: Dict[str, float] = {}
        self.optimization_metrics = OptimizationMetrics()
        
        # Configuration
        self.adaptive_learning: bool = True
        self.optimization_enabled: bool = True
        self.enable_ml = enable_ml
        
        # ML-based optimization
        self.performance_predictor = None
        self.method_recommender = None
        self.optimization_models = {}
        
        # Threading and concurrency
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._training_task: Optional[asyncio.Task] = None
        
        # Initialize default rules and profiles
        self._initialize_default_rules()
        self._initialize_default_profiles()
        
        logger.info("MathOptimizationEngine initialized")
    
    async def initialize(self) -> Any:
        """Initialize the optimization engine."""
        logger.info("Initializing optimization engine...")
        
        # Initialize ML models if enabled
        if self.enable_ml:
            await self._initialize_ml_models()
        
        # Start background training task
        if self.adaptive_learning:
            self._training_task = asyncio.create_task(self._adaptive_learning_loop())
        
        logger.info("Optimization engine initialized")
    
    async def shutdown(self) -> Any:
        """Shutdown the optimization engine gracefully."""
        logger.info("Shutting down optimization engine...")
        
        # Cancel background tasks
        if self._training_task:
            self._training_task.cancel()
        
        # Save models
        await self._save_optimization_models()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        logger.info("Optimization engine shutdown completed")
    
    def _initialize_default_rules(self) -> Any:
        """Initialize default optimization rules."""
        logger.info("Initializing default optimization rules...")
        
        # Large array optimization
        def large_array_condition(context: Dict[str, Any]) -> bool:
            operands = context.get("operands", [])
            return len(operands) > 1000
        
        def large_array_action(operation: MathOperation) -> MathOperation:
            return MathOperation(
                operation_type=operation.operation_type,
                operands=operation.operands,
                method=CalculationMethod.NUMPY,
                precision=operation.precision,
                metadata=operation.metadata
            )
        
        large_array_rule = OptimizationRule(
            rule_id="large_array_optimization",
            name="Large Array Optimization",
            condition=large_array_condition,
            action=large_array_action,
            priority=1,
            description="Use NumPy for large arrays"
        )
        self.optimization_rules.append(large_array_rule)
        
        # Simple operation optimization
        def simple_operation_condition(context: Dict[str, Any]) -> bool:
            operands = context.get("operands", [])
            return len(operands) <= 5 and all(isinstance(x, (int, float)) for x in operands)
        
        def simple_operation_action(operation: MathOperation) -> MathOperation:
            return MathOperation(
                operation_type=operation.operation_type,
                operands=operation.operands,
                method=CalculationMethod.BASIC,
                precision=operation.precision,
                metadata=operation.metadata
            )
        
        simple_rule = OptimizationRule(
            rule_id="simple_operation_optimization",
            name="Simple Operation Optimization",
            condition=simple_operation_condition,
            action=simple_operation_action,
            priority=2,
            description="Use basic method for simple operations"
        )
        self.optimization_rules.append(simple_rule)
        
        # High precision optimization
        def high_precision_condition(context: Dict[str, Any]) -> bool:
            precision = context.get("precision", 10)
            return precision > 15
        
        def high_precision_action(operation: MathOperation) -> MathOperation:
            return MathOperation(
                operation_type=operation.operation_type,
                operands=operation.operands,
                method=CalculationMethod.DECIMAL,
                precision=operation.precision,
                metadata=operation.metadata
            )
        
        precision_rule = OptimizationRule(
            rule_id="high_precision_optimization",
            name="High Precision Optimization",
            condition=high_precision_condition,
            action=high_precision_action,
            priority=1,
            description="Use Decimal for high precision"
        )
        self.optimization_rules.append(precision_rule)
        
        logger.info(f"Initialized {len(self.optimization_rules)} default rules")
    
    def _initialize_default_profiles(self) -> Any:
        """Initialize default optimization profiles."""
        logger.info("Initializing default optimization profiles...")
        
        # Performance profile
        performance_profile = self._create_performance_profile()
        self.optimization_profiles[performance_profile.profile_id] = performance_profile
        
        # Accuracy profile
        accuracy_profile = self._create_accuracy_profile()
        self.optimization_profiles[accuracy_profile.profile_id] = accuracy_profile
        
        # Balanced profile
        balanced_profile = self._create_balanced_profile()
        self.optimization_profiles[balanced_profile.profile_id] = balanced_profile
        
        logger.info(f"Initialized {len(self.optimization_profiles)} default profiles")
    
    def _create_performance_profile(self) -> OptimizationProfile:
        """Create a performance-focused optimization profile."""
        def performance_condition(context: Dict[str, Any]) -> bool:
            return context.get("priority", "balanced") == "performance"
        
        def performance_action(operation: MathOperation) -> MathOperation:
            # Always prefer numpy for performance
            return MathOperation(
                operation_type=operation.operation_type,
                operands=operation.operands,
                method=CalculationMethod.NUMPY,
                precision=operation.precision,
                metadata=operation.metadata
            )
        
        performance_rule = OptimizationRule(
            rule_id="performance_optimization",
            name="Performance Optimization",
            condition=performance_condition,
            action=performance_action,
            priority=1,
            description="Optimize for maximum performance"
        )
        
        return OptimizationProfile(
            profile_id="performance_profile",
            name="Performance Profile",
            strategy=OptimizationStrategy.PERFORMANCE,
            level=OptimizationLevel.HIGH,
            rules=[performance_rule],
            conditions={"priority": "performance"},
            performance_targets={"execution_time": 0.001}
        )
    
    def _create_accuracy_profile(self) -> OptimizationProfile:
        """Create an accuracy-focused optimization profile."""
        def accuracy_condition(context: Dict[str, Any]) -> bool:
            return context.get("priority", "balanced") == "accuracy"
        
        def accuracy_action(operation: MathOperation) -> MathOperation:
            # Use decimal for high accuracy
            return MathOperation(
                operation_type=operation.operation_type,
                operands=operation.operands,
                method=CalculationMethod.DECIMAL,
                precision=operation.precision,
                metadata=operation.metadata
            )
        
        accuracy_rule = OptimizationRule(
            rule_id="accuracy_optimization",
            name="Accuracy Optimization",
            condition=accuracy_condition,
            action=accuracy_action,
            priority=1,
            description="Optimize for maximum accuracy"
        )
        
        return OptimizationProfile(
            profile_id="accuracy_profile",
            name="Accuracy Profile",
            strategy=OptimizationStrategy.ACCURACY,
            level=OptimizationLevel.HIGH,
            rules=[accuracy_rule],
            conditions={"priority": "accuracy"},
            performance_targets={"accuracy": 0.999999}
        )
    
    def _create_balanced_profile(self) -> OptimizationProfile:
        """Create a balanced optimization profile."""
        return OptimizationProfile(
            profile_id="balanced_profile",
            name="Balanced Profile",
            strategy=OptimizationStrategy.BALANCED,
            level=OptimizationLevel.MEDIUM,
            rules=[],
            conditions={"priority": "balanced"},
            performance_targets={"execution_time": 0.01, "accuracy": 0.999}
        )
    
    async def _initialize_ml_models(self) -> Any:
        """Initialize ML models for optimization."""
        try:
            # Simple performance predictor (could be enhanced with actual ML models)
            self.performance_predictor = self._create_performance_predictor()
            self.method_recommender = self._create_method_recommender()
            
            logger.info("ML models initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ML models: {e}")
            self.enable_ml = False
    
    def _create_performance_predictor(self) -> Any:
        """Create a simple performance predictor."""
        # This is a simplified predictor - in production, you'd use actual ML models
        def predict_performance(operation_type: str, method: str, operands: List) -> float:
            base_times = {
                "add": {"basic": 0.001, "numpy": 0.0001, "decimal": 0.01},
                "multiply": {"basic": 0.001, "numpy": 0.0001, "decimal": 0.01},
                "divide": {"basic": 0.001, "numpy": 0.0001, "decimal": 0.01}
            }
            
            base_time = base_times.get(operation_type, {}).get(method, 0.001)
            size_factor = len(operands) / 1000.0
            
            return base_time * (1 + size_factor)
        
        return predict_performance
    
    def _create_method_recommender(self) -> Any:
        """Create a method recommendation system."""
        def recommend_method(operation_type: str, operands: List, context: Dict[str, Any]) -> str:
            size = len(operands)
            priority = context.get("priority", "balanced")
            
            if priority == "performance":
                if size > 100:
                    return "numpy"
                else:
                    return "basic"
            elif priority == "accuracy":
                return "decimal"
            else:  # balanced
                if size > 1000:
                    return "numpy"
                elif size < 10:
                    return "basic"
                else:
                    return "math"
        
        return recommend_method
    
    async def _adaptive_learning_loop(self) -> Any:
        """Background adaptive learning loop."""
        while True:
            try:
                # Retrain models based on recent data
                if len(self.optimization_history) > 100:
                    await self._retrain_models()
                
                # Update rule effectiveness
                self._update_rule_effectiveness()
                
                # Clean up old data
                self._cleanup_old_data()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Adaptive learning error: {e}")
                await asyncio.sleep(3600)
    
    async def _retrain_models(self) -> Any:
        """Retrain ML models with recent data."""
        try:
            # Extract training data from optimization history
            training_data = []
            for result in self.optimization_history[-1000:]:
                if result.success and result.original_operation:
                    training_data.append({
                        "operation_type": result.original_operation.operation_type.value,
                        "method": result.original_operation.method.value,
                        "operand_count": len(result.original_operation.operands),
                        "performance_improvement": result.performance_improvement,
                        "optimization_time": result.optimization_time
                    })
            
            if training_data:
                # Update performance predictor (simplified)
                logger.info(f"Retrained models with {len(training_data)} samples")
                
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
    
    def _update_rule_effectiveness(self) -> Any:
        """Update rule effectiveness based on recent usage."""
        with self._lock:
            for rule in self.optimization_rules:
                total_usage = rule.success_count + rule.failure_count
                if total_usage > 0:
                    rule.performance_impact = rule.success_count / total_usage
                    self.optimization_metrics.rule_effectiveness[rule.rule_id] = rule.performance_impact
    
    def _cleanup_old_data(self) -> Any:
        """Clean up old optimization data."""
        with self._lock:
            if len(self.optimization_history) > self.max_optimization_history:
                # Remove oldest entries
                excess = len(self.optimization_history) - self.max_optimization_history
                self.optimization_history = self.optimization_history[excess:]
                
                logger.info(f"Cleaned up {excess} old optimization records")
    
    async def optimize_operation(self, operation: MathOperation, 
                                context: Dict[str, Any] = None) -> OptimizationResult:
        """Optimize a mathematical operation with enhanced features."""
        if not self.optimization_enabled:
            return OptimizationResult(
                original_operation=operation,
                optimized_operation=operation,
                optimization_applied="disabled",
                status=OptimizationStatus.SKIPPED
            )
        
        start_time = time.time()
        context = context or {}
        
        # Check cache first
        cache_key = self._generate_cache_key(operation, context)
        if cache_key in self.optimization_cache:
            cached_result = self.optimization_cache[cache_key]
            cached_result.timestamp = datetime.now()
            return cached_result
        
        # Create optimization result
        result = OptimizationResult(
            original_operation=operation,
            status=OptimizationStatus.RUNNING
        )
        
        try:
            # Apply optimization rules
            optimized_operation = await self._apply_optimization_rules(operation, context)
            
            # Apply profile optimization
            profile = self._select_optimization_profile(context)
            if profile:
                optimized_operation = await self._apply_profile_optimization(
                    optimized_operation, profile, context
                )
            
            # Apply ML optimization if enabled
            if self.enable_ml:
                optimized_operation = await self._apply_ml_optimization(
                    optimized_operation, context
                )
            
            # Calculate improvements
            performance_improvement = self._calculate_performance_improvement(
                operation, optimized_operation
            )
            accuracy_impact = self._calculate_accuracy_impact(operation, optimized_operation)
            memory_impact = self._calculate_memory_impact(operation, optimized_operation)
            
            # Update result
            result.optimized_operation = optimized_operation
            result.performance_improvement = performance_improvement
            result.accuracy_impact = accuracy_impact
            result.memory_impact = memory_impact
            result.optimization_time = time.time() - start_time
            result.status = OptimizationStatus.COMPLETED
            result.success = True
            
            # Cache result
            self.optimization_cache[cache_key] = result
            
            # Update metrics
            self._update_optimization_metrics(result)
            
            logger.info(f"Optimization completed: {performance_improvement:.2%} improvement")
            
        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.success = False
            result.error_message = str(e)
            logger.error(f"Optimization failed: {e}")
        
        # Add to history
        with self._lock:
            self.optimization_history.append(result)
        
        return result
    
    def _generate_cache_key(self, operation: MathOperation, context: Dict[str, Any]) -> str:
        """Generate cache key for optimization result."""
        key_data = {
            "operation_type": operation.operation_type.value,
            "method": operation.method.value,
            "operands_hash": hashlib.md5(str(operation.operands).encode()).hexdigest()[:8],
            "context_hash": hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:8]
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    async def _apply_optimization_rules(self, operation: MathOperation, 
                                     context: Dict[str, Any]) -> MathOperation:
        """Apply optimization rules to an operation."""
        optimized_operation = operation
        
        # Sort rules by priority
        sorted_rules = sorted(self.optimization_rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            try:
                if rule.condition(context):
                    optimized_operation = rule.action(optimized_operation)
                    rule.success_count += 1
                    rule.last_used = datetime.now()
                    
                    logger.debug(f"Applied rule: {rule.name}")
                else:
                    rule.failure_count += 1
                    
            except Exception as e:
                rule.failure_count += 1
                logger.error(f"Rule application error: {e}")
        
        return optimized_operation
    
    def _select_optimization_profile(self, context: Dict[str, Any]) -> Optional[OptimizationProfile]:
        """Select appropriate optimization profile based on context."""
        for profile in self.optimization_profiles.values():
            if not profile.enabled:
                continue
            
            # Check if profile conditions match
            if self._profile_conditions_match(profile, context):
                profile.usage_count += 1
                return profile
        
        return None
    
    def _profile_conditions_match(self, profile: OptimizationProfile, 
                                context: Dict[str, Any]) -> bool:
        """Check if profile conditions match the context."""
        for key, value in profile.conditions.items():
            if context.get(key) != value:
                return False
        return True
    
    async def _apply_profile_optimization(self, operation: MathOperation, 
                                       profile: OptimizationProfile, 
                                       context: Dict[str, Any]) -> MathOperation:
        """Apply profile-specific optimization."""
        optimized_operation = operation
        
        for rule in profile.rules:
            if not rule.enabled:
                continue
            
            try:
                if rule.condition(context):
                    optimized_operation = rule.action(optimized_operation)
                    logger.debug(f"Applied profile rule: {rule.name}")
            except Exception as e:
                logger.error(f"Profile rule application error: {e}")
        
        return optimized_operation
    
    async def _apply_ml_optimization(self, operation: MathOperation, 
                                  context: Dict[str, Any]) -> MathOperation:
        """Apply ML-based optimization."""
        if not self.method_recommender:
            return operation
        
        try:
            # Get method recommendation
            recommended_method = self.method_recommender(
                operation.operation_type.value,
                operation.operands,
                context
            )
            
            # Create optimized operation with recommended method
            optimized_operation = MathOperation(
                operation_type=operation.operation_type,
                operands=operation.operands,
                method=CalculationMethod(recommended_method),
                precision=operation.precision,
                metadata=operation.metadata
            )
            
            logger.debug(f"ML optimization: recommended {recommended_method}")
            return optimized_operation
            
        except Exception as e:
            logger.error(f"ML optimization error: {e}")
            return operation
    
    def _calculate_performance_improvement(self, original: MathOperation, 
                                        optimized: MathOperation) -> float:
        """Calculate performance improvement percentage."""
        if not self.performance_predictor:
            return 0.0
        
        try:
            original_time = self.performance_predictor(
                original.operation_type.value,
                original.method.value,
                original.operands
            )
            
            optimized_time = self.performance_predictor(
                optimized.operation_type.value,
                optimized.method.value,
                optimized.operands
            )
            
            if original_time > 0:
                return (original_time - optimized_time) / original_time
            
        except Exception as e:
            logger.error(f"Performance calculation error: {e}")
        
        return 0.0
    
    def _calculate_accuracy_impact(self, original: MathOperation, 
                                optimized: MathOperation) -> float:
        """Calculate accuracy impact of optimization."""
        # Simplified accuracy impact calculation
        accuracy_impacts = {
            CalculationMethod.BASIC: 1.0,
            CalculationMethod.NUMPY: 0.9999,
            CalculationMethod.MATH: 0.9999,
            CalculationMethod.DECIMAL: 1.0
        }
        
        original_accuracy = accuracy_impacts.get(original.method, 1.0)
        optimized_accuracy = accuracy_impacts.get(optimized.method, 1.0)
        
        return optimized_accuracy - original_accuracy
    
    def _calculate_memory_impact(self, original: MathOperation, 
                              optimized: MathOperation) -> float:
        """Calculate memory impact of optimization."""
        # Simplified memory impact calculation
        memory_impacts = {
            CalculationMethod.BASIC: 1.0,
            CalculationMethod.NUMPY: 1.5,  # NumPy arrays use more memory
            CalculationMethod.MATH: 1.0,
            CalculationMethod.DECIMAL: 2.0  # Decimal objects use more memory
        }
        
        original_memory = memory_impacts.get(original.method, 1.0)
        optimized_memory = memory_impacts.get(optimized.method, 1.0)
        
        return (optimized_memory - original_memory) / original_memory
    
    def _update_optimization_metrics(self, result: OptimizationResult):
        """Update optimization metrics."""
        with self._lock:
            self.optimization_metrics.total_optimizations += 1
            
            if result.success:
                self.optimization_metrics.successful_optimizations += 1
                self.optimization_metrics.total_time_saved += result.performance_improvement
                
                # Update average improvement
                total_successful = self.optimization_metrics.successful_optimizations
                current_avg = self.optimization_metrics.average_improvement
                new_avg = ((current_avg * (total_successful - 1)) + result.performance_improvement) / total_successful
                self.optimization_metrics.average_improvement = new_avg
            else:
                self.optimization_metrics.failed_optimizations += 1
    
    def add_optimization_rule(self, rule: OptimizationRule):
        """Add a new optimization rule."""
        with self._lock:
            self.optimization_rules.append(rule)
            logger.info(f"Added optimization rule: {rule.name}")
    
    def remove_optimization_rule(self, rule_id: str) -> bool:
        """Remove an optimization rule."""
        with self._lock:
            for i, rule in enumerate(self.optimization_rules):
                if rule.rule_id == rule_id:
                    removed_rule = self.optimization_rules.pop(i)
                    logger.info(f"Removed optimization rule: {removed_rule.name}")
                    return True
        return False
    
    def create_optimization_profile(self, profile: OptimizationProfile):
        """Create a new optimization profile."""
        with self._lock:
            self.optimization_profiles[profile.profile_id] = profile
            logger.info(f"Created optimization profile: {profile.name}")
    
    def get_optimization_profile(self, profile_id: str) -> Optional[OptimizationProfile]:
        """Get optimization profile by ID."""
        return self.optimization_profiles.get(profile_id)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        with self._lock:
            return {
                "metrics": {
                    "total_optimizations": self.optimization_metrics.total_optimizations,
                    "successful_optimizations": self.optimization_metrics.successful_optimizations,
                    "failed_optimizations": self.optimization_metrics.failed_optimizations,
                    "success_rate": (
                        self.optimization_metrics.successful_optimizations / 
                        self.optimization_metrics.total_optimizations
                        if self.optimization_metrics.total_optimizations > 0 else 0.0
                    ),
                    "average_improvement": self.optimization_metrics.average_improvement,
                    "total_time_saved": self.optimization_metrics.total_time_saved
                },
                "rules": [
                    {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "enabled": rule.enabled,
                        "success_count": rule.success_count,
                        "failure_count": rule.failure_count,
                        "effectiveness": rule.performance_impact,
                        "last_used": rule.last_used.isoformat() if rule.last_used else None
                    }
                    for rule in self.optimization_rules
                ],
                "profiles": [
                    {
                        "profile_id": profile.profile_id,
                        "name": profile.name,
                        "strategy": profile.strategy.value,
                        "level": profile.level.value,
                        "enabled": profile.enabled,
                        "usage_count": profile.usage_count,
                        "success_rate": profile.success_rate
                    }
                    for profile in self.optimization_profiles.values()
                ],
                "configuration": {
                    "optimization_enabled": self.optimization_enabled,
                    "adaptive_learning": self.adaptive_learning,
                    "enable_ml": self.enable_ml,
                    "max_history": self.max_optimization_history
                }
            }
    
    def enable_optimization(self, enabled: bool = True):
        """Enable or disable optimization."""
        self.optimization_enabled = enabled
        logger.info(f"Optimization {'enabled' if enabled else 'disabled'}")
    
    def enable_adaptive_learning(self, enabled: bool = True):
        """Enable or disable adaptive learning."""
        self.adaptive_learning = enabled
        logger.info(f"Adaptive learning {'enabled' if enabled else 'disabled'}")
    
    async def _save_optimization_models(self) -> Any:
        """Save optimization models to disk."""
        try:
            models_data = {
                "performance_predictor": self.performance_predictor,
                "method_recommender": self.method_recommender,
                "optimization_models": self.optimization_models
            }
            
            with open("optimization_models.pkl", "wb") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                pickle.dump(models_data, f)
            
            logger.info("Optimization models saved")
        except Exception as e:
            logger.error(f"Failed to save optimization models: {e}")
    
    def export_optimization_config(self) -> str:
        """Export optimization configuration."""
        config = {
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "priority": rule.priority,
                    "enabled": rule.enabled,
                    "metadata": rule.metadata
                }
                for rule in self.optimization_rules
            ],
            "profiles": [
                {
                    "profile_id": profile.profile_id,
                    "name": profile.name,
                    "strategy": profile.strategy.value,
                    "level": profile.level.value,
                    "conditions": profile.conditions,
                    "performance_targets": profile.performance_targets,
                    "enabled": profile.enabled
                }
                for profile in self.optimization_profiles.values()
            ],
            "configuration": {
                "optimization_enabled": self.optimization_enabled,
                "adaptive_learning": self.adaptive_learning,
                "enable_ml": self.enable_ml,
                "max_optimization_history": self.max_optimization_history
            }
        }
        
        return json.dumps(config, indent=2)


async def main():
    """Main function for testing."""
    # Create optimization engine
    engine = MathOptimizationEngine()
    await engine.initialize()
    
    try:
        # Test optimization
        operation = MathOperation(
            operation_type=OperationType.ADD,
            operands=[1, 2, 3, 4, 5],
            method=CalculationMethod.BASIC
        )
        
        result = await engine.optimize_operation(operation, {"priority": "performance"})
        
        print(f"Optimization result: {result}")
        print(f"Performance improvement: {result.performance_improvement:.2%}")
        
        # Get statistics
        stats = engine.get_optimization_statistics()
        print(f"Optimization statistics: {stats}")
        
    finally:
        await engine.shutdown()


match __name__:
    case "__main__":
    asyncio.run(main()) 