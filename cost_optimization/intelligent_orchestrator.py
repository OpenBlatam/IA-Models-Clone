# cost_optimization/intelligent_orchestrator.py
"""
Intelligent API Cost Optimization Orchestrator
Combines all optimization strategies into a unified system
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
from datetime import datetime, timedelta
from enterprise_api_cost_optimizer import EnterpriseAPICostOptimizer, RequestContext
from intelligent_cost_predictor import IntelligentCostPredictor, CostPrediction

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    COST_FIRST = "cost_first"
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"
    SPEED_FIRST = "speed_first"

@dataclass
class OptimizationResult:
    response: str
    model_used: str
    total_cost: float
    predicted_cost: float
    actual_savings: float
    response_time: float
    strategy_used: str
    quality_score: float
    cache_hit: bool

class IntelligentOrchestrator:
    """
    Master orchestrator that coordinates all cost optimization strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.optimizer = EnterpriseAPICostOptimizer()
        self.predictor = IntelligentCostPredictor()
        
        # Performance tracking
        self.total_requests = 0
        self.total_cost = 0.0
        self.total_savings = 0.0
        self.avg_response_time = 0.0
        
        # Circuit breaker for API failures
        self.circuit_breaker = {
            'failures': 0,
            'success_count': 0,
            'last_failure_time': None,
            'state': 'closed'  # closed, open, half_open
        }
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_retries': 3,
            'circuit_breaker_threshold': 5,
            'circuit_breaker_timeout': 60,
            'auto_retrain_threshold': 100,
            'cost_budget_per_hour': 10.0,
            'quality_threshold': 75.0,
            'cache_ttl_hours': 24
        }
    
    async def optimize_request(self, 
                             prompt: str,
                             max_tokens: int = 100,
                             temperature: float = 0.7,
                             priority: str = "medium",
                             strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                             budget_limit: Optional[float] = None) -> OptimizationResult:
        """
        Main entry point for request optimization
        """
        start_time = time.time()
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise Exception("Circuit breaker is open - API temporarily unavailable")
        
        try:
            # Create request context
            context = RequestContext(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                priority=priority,
                quality_threshold=self.config['quality_threshold'],
                budget_limit=budget_limit,
                cache_eligible=True
            )
            
            # Get cost prediction
            prediction = self.predictor.predict_cost(prompt, asdict(context))
            
            # Adjust strategy based on prediction
            adjusted_strategy = self._adjust_strategy(strategy, prediction, context)
            
            # Execute optimization
            result = await self.optimizer.optimize_request(context)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(result, prediction)
            
            # Update metrics
            self._update_metrics(result, prediction, time.time() - start_time)
            
            # Record actual cost for learning
            self.predictor.record_actual_cost(
                prompt, asdict(context), result['cost'], result['model_used']
            )
            
            # Auto-retrain if needed
            await self._auto_retrain_check()
            
            # Update circuit breaker on success
            self._update_circuit_breaker(True)
            
            return OptimizationResult(
                response=result['response'],
                model_used=result['model_used'],
                total_cost=result['cost'],
                predicted_cost=prediction.predicted_cost,
                actual_savings=result.get('cost_saved', 0),
                response_time=time.time() - start_time,
                strategy_used=adjusted_strategy,
                quality_score=quality_score,
                cache_hit=result.get('cached', False)
            )
            
        except Exception as e:
            self._update_circuit_breaker(False)
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests"""
        cb = self.circuit_breaker
        
        if cb['state'] == 'open':
            if (datetime.now() - cb['last_failure_time']).seconds > self.config['circuit_breaker_timeout']:
                cb['state'] = 'half_open'
                return True
            return False
        
        return True
    
    def _update_circuit_breaker(self, success: bool):
        """Update circuit breaker state"""
        cb = self.circuit_breaker
        
        if success:
            cb['success_count'] += 1
            cb['failures'] = 0
            if cb['state'] == 'half_open' and cb['success_count'] >= 3:
                cb['state'] = 'closed'
        else:
            cb['failures'] += 1
            cb['last_failure_time'] = datetime.now()
            if cb['failures'] >= self.config['circuit_breaker_threshold']:
                cb['state'] = 'open'
    
    def _adjust_strategy(self, 
                       base_strategy: OptimizationStrategy, 
                       prediction: CostPrediction, 
                       context: RequestContext) -> str:
        """Dynamically adjust optimization strategy"""
        if prediction.predicted_cost > 0.01 and base_strategy != OptimizationStrategy.COST_FIRST:
            return "cost_optimized_" + prediction.optimization_strategy
        elif context.priority == "critical":
            return "quality_first_" + prediction.optimization_strategy
        else:
            return prediction.optimization_strategy
    
    def _calculate_quality_score(self, result: Dict, prediction: CostPrediction) -> float:
        """Calculate quality score for the response"""
        base_score = 80.0  # Default score
        
        # Adjust based on model quality
        model_quality = {
            'gpt-4o-mini': 85.0,
            'claude-3-haiku': 80.0,
            'gpt-3.5-turbo': 75.0,
            'claude-3-sonnet': 92.0,
            'cache': 85.0  # Cached responses maintain quality
        }.get(result.get('model_used', 'unknown'), base_score)
        
        # Adjust based on prediction confidence
        confidence_bonus = prediction.confidence * 10
        
        return min(100.0, model_quality + confidence_bonus)
    
    def _update_metrics(self, result: Dict, prediction: CostPrediction, response_time: float):
        """Update performance metrics"""
        self.total_requests += 1
        self.total_cost += result['cost']
        self.total_savings += result.get('cost_saved', 0) + prediction.estimated_savings
        
        # Update moving average of response time
        alpha = 0.1  # Smoothing factor
        self.avg_response_time = (alpha * response_time + 
                                (1 - alpha) * self.avg_response_time)
    
    async def _auto_retrain_check(self):
        """Check and perform auto-retraining"""
        if self.total_requests % self.config['auto_retrain_threshold'] == 0:
            logger.info("Triggering auto-retrain")
            self.predictor.auto_retrain()
    
    async def batch_optimize(self, requests: List[Dict[str, Any]]) -> List[OptimizationResult]:
        """Optimize multiple requests in batch"""
        contexts = []
        for req in requests:
            contexts.append(RequestContext(
                prompt=req['prompt'],
                max_tokens=req.get('max_tokens', 100),
                temperature=req.get('temperature', 0.7),
                priority=req.get('priority', 'medium'),
                quality_threshold=self.config['quality_threshold'],
                budget_limit=req.get('budget_limit'),
                cache_eligible=req.get('cache_eligible', True)
            ))
        
        # Use batch optimization from the base optimizer
        batch_results = await self.optimizer.batch_optimize(contexts)
        
        # Convert to OptimizationResults
        results = []
        for i, result in enumerate(batch_results):
            prediction = self.predictor.predict_cost(requests[i]['prompt'], asdict(contexts[i]))
            quality_score = self._calculate_quality_score(result, prediction)
            
            results.append(OptimizationResult(
                response=result['response'],
                model_used=result['model_used'],
                total_cost=result['cost'],
                predicted_cost=prediction.predicted_cost,
                actual_savings=result.get('cost_saved', 0),
                response_time=result['response_time'],
                strategy_used=prediction.optimization_strategy,
                quality_score=quality_score,
                cache_hit=result.get('cached', False)
            ))
        
        return results
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        optimizer_analytics = self.optimizer.get_cost_analytics()
        prediction_analytics = self.predictor.get_prediction_analytics()
        
        roi = ((self.total_savings / max(self.total_cost, 0.001)) * 100) if self.total_cost > 0 else 0
        
        return {
            'system_metrics': {
                'total_requests': self.total_requests,
                'total_cost_usd': self.total_cost,
                'total_savings_usd': self.total_savings,
                'roi_percentage': roi,
                'avg_response_time_ms': self.avg_response_time * 1000,
                'circuit_breaker_state': self.circuit_breaker['state']
            },
            'optimization_metrics': optimizer_analytics,
            'prediction_metrics': prediction_analytics,
            'cost_efficiency': {
                'cost_per_request': self.total_cost / max(self.total_requests, 1),
                'savings_per_request': self.total_savings / max(self.total_requests, 1),
                'efficiency_score': min(100, (self.total_savings / max(self.total_cost, 0.001)) * 100)
            }
        }
    
    async def set_budget_alert(self, hourly_budget: float, callback: callable = None):
        """Set up budget monitoring and alerts"""
        while True:
            current_hour_cost = self._get_current_hour_cost()
            if current_hour_cost > hourly_budget:
                alert_msg = f"Budget alert: ${current_hour_cost:.4f} exceeds ${hourly_budget:.4f}"
                logger.warning(alert_msg)
                if callback:
                    await callback(alert_msg)
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    def _get_current_hour_cost(self) -> float:
        """Get cost for current hour"""
        # This is a simplified implementation
        # In practice, you'd query the database for recent costs
        return self.total_cost * 0.1  # Rough estimate
    
    async def optimize_with_constraints(self,
                                      prompt: str,
                                      cost_limit: float,
                                      min_quality: float,
                                      max_latency_ms: int) -> OptimizationResult:
        """Optimize with multiple constraints"""
        context = RequestContext(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            priority="medium",
            quality_threshold=min_quality,
            budget_limit=cost_limit,
            cache_eligible=True
        )
        
        prediction = self.predictor.predict_cost(prompt, asdict(context))
        
        # Check if constraints can be met
        if prediction.predicted_cost > cost_limit:
            # Try to find alternative strategy
            context.priority = "low"  # Lower priority for cost savings
            context.quality_threshold = min_quality * 0.9  # Slight quality reduction
        
        start_time = time.time()
        result = await self.optimizer.optimize_request(context)
        actual_latency = (time.time() - start_time) * 1000
        
        if actual_latency > max_latency_ms:
            logger.warning(f"Latency constraint violated: {actual_latency}ms > {max_latency_ms}ms")
        
        return OptimizationResult(
            response=result['response'],
            model_used=result['model_used'],
            total_cost=result['cost'],
            predicted_cost=prediction.predicted_cost,
            actual_savings=result.get('cost_saved', 0),
            response_time=actual_latency / 1000,
            strategy_used="constraint_optimized",
            quality_score=self._calculate_quality_score(result, prediction),
            cache_hit=result.get('cached', False)
        )
