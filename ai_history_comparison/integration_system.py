"""
Integration System for AI History Analysis and Workflow Chain Engine
==================================================================

This module provides seamless integration between the AI history analyzer
and the workflow chain engine, enabling comprehensive performance tracking
and optimization across all AI operations.

Features:
- Automatic performance tracking
- Model selection optimization
- Performance-based routing
- Historical analysis integration
- Predictive model selection
- Performance degradation detection
- Automated optimization recommendations
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

# Import our components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import (
    AIHistoryConfig, ModelProvider, ModelCategory,
    get_ai_history_config
)

# Import workflow chain components (assuming they exist)
try:
    from ..document_workflow_chain.workflow_chain_engine import WorkflowChainEngine
    from ..document_workflow_chain.advanced_analysis import AdvancedWorkflowChainEngine
    from ..document_workflow_chain.config import get_workflow_config, ModelPriority
except ImportError:
    # Fallback for when workflow chain is not available
    WorkflowChainEngine = None
    AdvancedWorkflowChainEngine = None
    get_workflow_config = None
    ModelPriority = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceInsight:
    """Represents a performance insight or recommendation"""
    type: str  # "optimization", "warning", "recommendation", "alert"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    model_name: str
    metric: str
    current_value: float
    recommended_value: Optional[float] = None
    confidence: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ModelSelectionRecommendation:
    """Represents a model selection recommendation"""
    task_type: str
    content_size: int
    priority: str  # "speed", "quality", "cost", "balanced"
    recommended_model: str
    alternative_models: List[str]
    confidence: float
    reasoning: str
    expected_performance: Dict[str, float]
    cost_estimate: float


class AIHistoryIntegrationSystem:
    """Integration system for AI history analysis and workflow management"""
    
    def __init__(self):
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        self.workflow_config = get_workflow_config() if get_workflow_config else None
        
        # Performance tracking
        self.performance_insights: List[PerformanceInsight] = []
        self.model_recommendations: List[ModelSelectionRecommendation] = []
        
        # Integration settings
        self.auto_tracking_enabled = True
        self.auto_optimization_enabled = True
        self.performance_thresholds = {
            "quality_score": 0.7,
            "response_time": 5.0,
            "cost_efficiency": 0.5,
            "token_efficiency": 0.6
        }
        
        # Model performance cache
        self.model_performance_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def track_workflow_performance(self, 
                                       workflow_id: str,
                                       model_name: str,
                                       task_type: str,
                                       performance_data: Dict[str, Any]) -> bool:
        """Track performance data from workflow operations"""
        try:
            if not self.auto_tracking_enabled:
                return True
            
            # Map workflow metrics to analyzer metrics
            metric_mapping = {
                "quality_score": PerformanceMetric.QUALITY_SCORE,
                "response_time": PerformanceMetric.RESPONSE_TIME,
                "token_efficiency": PerformanceMetric.TOKEN_EFFICIENCY,
                "cost_efficiency": PerformanceMetric.COST_EFFICIENCY,
                "coherence": PerformanceMetric.COHERENCE,
                "relevance": PerformanceMetric.RELEVANCE,
                "creativity": PerformanceMetric.CREATIVITY
            }
            
            # Record performance for each metric
            for metric_name, metric_enum in metric_mapping.items():
                if metric_name in performance_data:
                    value = performance_data[metric_name]
                    
                    # Record in analyzer
                    self.analyzer.record_performance(
                        model_name=model_name,
                        model_type=ModelType.TEXT_GENERATION,  # Default for workflow chains
                        metric=metric_enum,
                        value=value,
                        context={
                            "workflow_id": workflow_id,
                            "task_type": task_type,
                            "timestamp": datetime.now().isoformat()
                        },
                        metadata={
                            "source": "workflow_chain",
                            "integration": True
                        }
                    )
            
            # Generate insights
            await self._generate_performance_insights(model_name)
            
            logger.debug(f"Tracked performance for workflow {workflow_id} with model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking workflow performance: {str(e)}")
            return False
    
    async def get_model_recommendation(self, 
                                     task_type: str,
                                     content_size: int,
                                     priority: str = "balanced",
                                     budget_constraint: Optional[float] = None) -> ModelSelectionRecommendation:
        """Get model recommendation based on historical performance"""
        try:
            # Check cache first
            cache_key = f"{task_type}_{content_size}_{priority}_{budget_constraint}"
            if cache_key in self.model_performance_cache:
                cached_data = self.model_performance_cache[cache_key]
                if datetime.now() - cached_data["timestamp"] < timedelta(seconds=self.cache_ttl):
                    return cached_data["recommendation"]
            
            # Get available models
            available_models = self._get_available_models(content_size, budget_constraint)
            
            if not available_models:
                raise ValueError("No suitable models found for the given constraints")
            
            # Analyze performance for each model
            model_scores = {}
            for model_name in available_models:
                score = await self._calculate_model_score(model_name, priority, task_type)
                model_scores[model_name] = score
            
            # Sort models by score
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get best model
            recommended_model = sorted_models[0][0]
            alternative_models = [model for model, _ in sorted_models[1:4]]  # Top 3 alternatives
            
            # Calculate confidence
            confidence = self._calculate_recommendation_confidence(sorted_models)
            
            # Generate reasoning
            reasoning = self._generate_recommendation_reasoning(
                recommended_model, sorted_models, priority, task_type
            )
            
            # Get expected performance
            expected_performance = await self._get_expected_performance(recommended_model, task_type)
            
            # Calculate cost estimate
            cost_estimate = self._calculate_cost_estimate(recommended_model, content_size)
            
            # Create recommendation
            recommendation = ModelSelectionRecommendation(
                task_type=task_type,
                content_size=content_size,
                priority=priority,
                recommended_model=recommended_model,
                alternative_models=alternative_models,
                confidence=confidence,
                reasoning=reasoning,
                expected_performance=expected_performance,
                cost_estimate=cost_estimate
            )
            
            # Cache the recommendation
            self.model_performance_cache[cache_key] = {
                "recommendation": recommendation,
                "timestamp": datetime.now()
            }
            
            # Store recommendation
            self.model_recommendations.append(recommendation)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error getting model recommendation: {str(e)}")
            raise
    
    async def _calculate_model_score(self, model_name: str, priority: str, task_type: str) -> float:
        """Calculate overall score for a model based on priority and historical performance"""
        try:
            # Get performance summary
            summary = self.analyzer.get_performance_summary(model_name, days=30)
            
            if not summary or "metrics" not in summary:
                return 0.0
            
            # Define priority weights
            priority_weights = {
                "speed": {"response_time": 0.5, "quality_score": 0.3, "cost_efficiency": 0.2},
                "quality": {"quality_score": 0.5, "coherence": 0.2, "relevance": 0.2, "response_time": 0.1},
                "cost": {"cost_efficiency": 0.5, "token_efficiency": 0.3, "quality_score": 0.2},
                "balanced": {"quality_score": 0.3, "response_time": 0.2, "cost_efficiency": 0.2, "token_efficiency": 0.2, "coherence": 0.1}
            }
            
            weights = priority_weights.get(priority, priority_weights["balanced"])
            
            # Calculate weighted score
            total_score = 0.0
            total_weight = 0.0
            
            for metric_name, weight in weights.items():
                if metric_name in summary["metrics"]:
                    metric_data = summary["metrics"][metric_name]
                    metric_value = metric_data["mean"]
                    
                    # Normalize metric value (0-1 scale)
                    normalized_value = self._normalize_metric_value(metric_name, metric_value)
                    
                    total_score += normalized_value * weight
                    total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating model score: {str(e)}")
            return 0.0
    
    def _normalize_metric_value(self, metric_name: str, value: float) -> float:
        """Normalize metric value to 0-1 scale"""
        try:
            # Get metric configuration
            metric_config = self.config.get_metric(metric_name)
            if not metric_config:
                return 0.0
            
            # Normalize based on min/max values
            min_val = metric_config.min_value
            max_val = metric_config.max_value
            
            if max_val == min_val:
                return 0.0
            
            normalized = (value - min_val) / (max_val - min_val)
            
            # Invert if higher is not better
            if not metric_config.higher_is_better:
                normalized = 1.0 - normalized
            
            return max(0.0, min(1.0, normalized))
            
        except Exception as e:
            logger.error(f"Error normalizing metric value: {str(e)}")
            return 0.0
    
    def _get_available_models(self, content_size: int, budget_constraint: Optional[float]) -> List[str]:
        """Get list of available models that meet constraints"""
        try:
            available_models = []
            
            for model_name, model_def in self.config.models.items():
                if not model_def.is_active:
                    continue
                
                # Check context length constraint
                if content_size > model_def.context_length:
                    continue
                
                # Check budget constraint
                if budget_constraint is not None:
                    estimated_cost = (content_size / 1000) * model_def.cost_per_1k_tokens
                    if estimated_cost > budget_constraint:
                        continue
                
                available_models.append(model_name)
            
            return available_models
            
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []
    
    def _calculate_recommendation_confidence(self, sorted_models: List[Tuple[str, float]]) -> float:
        """Calculate confidence in model recommendation"""
        try:
            if len(sorted_models) < 2:
                return 0.5
            
            best_score = sorted_models[0][1]
            second_best_score = sorted_models[1][1]
            
            # Higher gap between best and second best = higher confidence
            if best_score == 0:
                return 0.0
            
            gap = (best_score - second_best_score) / best_score
            confidence = min(1.0, gap * 2)  # Scale gap to 0-1
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating recommendation confidence: {str(e)}")
            return 0.5
    
    def _generate_recommendation_reasoning(self, 
                                         recommended_model: str,
                                         sorted_models: List[Tuple[str, float]],
                                         priority: str,
                                         task_type: str) -> str:
        """Generate human-readable reasoning for model recommendation"""
        try:
            reasoning_parts = []
            
            # Add priority-based reasoning
            if priority == "speed":
                reasoning_parts.append(f"Selected {recommended_model} for optimal speed performance")
            elif priority == "quality":
                reasoning_parts.append(f"Selected {recommended_model} for highest quality output")
            elif priority == "cost":
                reasoning_parts.append(f"Selected {recommended_model} for best cost efficiency")
            else:
                reasoning_parts.append(f"Selected {recommended_model} for balanced performance")
            
            # Add score comparison
            if len(sorted_models) > 1:
                best_score = sorted_models[0][1]
                second_best_score = sorted_models[1][1]
                gap = ((best_score - second_best_score) / best_score) * 100
                
                if gap > 10:
                    reasoning_parts.append(f"Significantly outperforms alternatives ({gap:.1f}% better)")
                elif gap > 5:
                    reasoning_parts.append(f"Moderately outperforms alternatives ({gap:.1f}% better)")
                else:
                    reasoning_parts.append("Closely matched with alternatives")
            
            # Add task-specific reasoning
            if task_type == "document_generation":
                reasoning_parts.append("Optimized for long-form content generation")
            elif task_type == "analysis":
                reasoning_parts.append("Optimized for analytical tasks")
            elif task_type == "summarization":
                reasoning_parts.append("Optimized for summarization tasks")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating recommendation reasoning: {str(e)}")
            return f"Selected {recommended_model} based on historical performance analysis."
    
    async def _get_expected_performance(self, model_name: str, task_type: str) -> Dict[str, float]:
        """Get expected performance metrics for a model"""
        try:
            summary = self.analyzer.get_performance_summary(model_name, days=30)
            
            if not summary or "metrics" not in summary:
                return {}
            
            expected_performance = {}
            for metric_name, metric_data in summary["metrics"].items():
                expected_performance[metric_name] = metric_data["mean"]
            
            return expected_performance
            
        except Exception as e:
            logger.error(f"Error getting expected performance: {str(e)}")
            return {}
    
    def _calculate_cost_estimate(self, model_name: str, content_size: int) -> float:
        """Calculate cost estimate for using a model"""
        try:
            model_def = self.config.get_model(model_name)
            if not model_def:
                return 0.0
            
            # Estimate tokens (rough approximation: 4 characters per token)
            estimated_tokens = content_size / 4
            
            # Calculate cost
            cost = (estimated_tokens / 1000) * model_def.cost_per_1k_tokens
            
            return cost
            
        except Exception as e:
            logger.error(f"Error calculating cost estimate: {str(e)}")
            return 0.0
    
    async def _generate_performance_insights(self, model_name: str):
        """Generate performance insights for a model"""
        try:
            if not self.auto_optimization_enabled:
                return
            
            # Get recent performance summary
            summary = self.analyzer.get_performance_summary(model_name, days=7)
            
            if not summary or "metrics" not in summary:
                return
            
            # Check each metric against thresholds
            for metric_name, metric_data in summary["metrics"].items():
                current_value = metric_data["mean"]
                threshold = self.performance_thresholds.get(metric_name)
                
                if threshold is None:
                    continue
                
                # Determine if performance is below threshold
                metric_config = self.config.get_metric(metric_name)
                if not metric_config:
                    continue
                
                is_below_threshold = False
                if metric_config.higher_is_better:
                    is_below_threshold = current_value < threshold
                else:
                    is_below_threshold = current_value > threshold
                
                if is_below_threshold:
                    # Generate insight
                    severity = "high" if abs(current_value - threshold) / threshold > 0.2 else "medium"
                    
                    insight = PerformanceInsight(
                        type="warning",
                        severity=severity,
                        message=f"{metric_name} is below optimal threshold ({current_value:.3f} < {threshold:.3f})",
                        model_name=model_name,
                        metric=metric_name,
                        current_value=current_value,
                        recommended_value=threshold,
                        confidence=0.8
                    )
                    
                    self.performance_insights.append(insight)
                    logger.warning(f"Performance insight generated: {insight.message}")
            
            # Check for trends
            for metric_name in summary["metrics"].keys():
                trend_analysis = self.analyzer.analyze_trends(
                    model_name, 
                    PerformanceMetric(metric_name), 
                    days=30
                )
                
                if trend_analysis and trend_analysis.trend_direction == "declining":
                    insight = PerformanceInsight(
                        type="alert",
                        severity="medium",
                        message=f"{metric_name} shows declining trend",
                        model_name=model_name,
                        metric=metric_name,
                        current_value=0.0,  # Will be filled from trend data
                        confidence=trend_analysis.confidence
                    )
                    
                    self.performance_insights.append(insight)
                    logger.warning(f"Trend insight generated: {insight.message}")
            
        except Exception as e:
            logger.error(f"Error generating performance insights: {str(e)}")
    
    def get_performance_insights(self, model_name: Optional[str] = None, 
                               severity: Optional[str] = None) -> List[PerformanceInsight]:
        """Get performance insights"""
        try:
            insights = self.performance_insights
            
            # Filter by model name
            if model_name:
                insights = [i for i in insights if i.model_name == model_name]
            
            # Filter by severity
            if severity:
                insights = [i for i in insights if i.severity == severity]
            
            # Sort by timestamp (newest first)
            insights.sort(key=lambda x: x.timestamp, reverse=True)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting performance insights: {str(e)}")
            return []
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of integration system"""
        try:
            return {
                "auto_tracking_enabled": self.auto_tracking_enabled,
                "auto_optimization_enabled": self.auto_optimization_enabled,
                "performance_insights_count": len(self.performance_insights),
                "model_recommendations_count": len(self.model_recommendations),
                "performance_thresholds": self.performance_thresholds,
                "analyzer_stats": self.analyzer.performance_stats,
                "config_summary": self.config.get_configuration_summary()
            }
            
        except Exception as e:
            logger.error(f"Error getting integration summary: {str(e)}")
            return {"error": str(e)}
    
    async def optimize_model_selection(self, 
                                     workflow_engine: Optional[WorkflowChainEngine] = None) -> Dict[str, Any]:
        """Optimize model selection for workflow engine"""
        try:
            if not workflow_engine:
                return {"error": "Workflow engine not provided"}
            
            # Get current active chains
            active_chains = workflow_engine.get_all_active_chains()
            
            optimization_results = {
                "chains_analyzed": len(active_chains),
                "recommendations": [],
                "optimizations_applied": 0
            }
            
            # Analyze each chain
            for chain in active_chains:
                # Get chain analytics
                if hasattr(workflow_engine, 'get_chain_analytics'):
                    analytics = workflow_engine.get_chain_analytics(chain.id)
                    
                    # Generate recommendations for this chain
                    recommendation = await self.get_model_recommendation(
                        task_type="document_generation",
                        content_size=analytics.get("total_tokens", 1000),
                        priority="balanced"
                    )
                    
                    optimization_results["recommendations"].append({
                        "chain_id": chain.id,
                        "current_model": "unknown",  # Would need to track this
                        "recommended_model": recommendation.recommended_model,
                        "confidence": recommendation.confidence,
                        "expected_improvement": "TBD"  # Would calculate this
                    })
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing model selection: {str(e)}")
            return {"error": str(e)}


# Global integration system instance
_integration_system: Optional[AIHistoryIntegrationSystem] = None


def get_integration_system() -> AIHistoryIntegrationSystem:
    """Get or create global integration system"""
    global _integration_system
    if _integration_system is None:
        _integration_system = AIHistoryIntegrationSystem()
    return _integration_system


# Example usage and testing
async def main():
    """Example usage of the integration system"""
    integration = get_integration_system()
    
    # Simulate tracking workflow performance
    await integration.track_workflow_performance(
        workflow_id="test_workflow_1",
        model_name="gpt-4",
        task_type="document_generation",
        performance_data={
            "quality_score": 0.85,
            "response_time": 2.5,
            "token_efficiency": 0.78,
            "cost_efficiency": 0.65
        }
    )
    
    # Get model recommendation
    recommendation = await integration.get_model_recommendation(
        task_type="document_generation",
        content_size=5000,
        priority="balanced"
    )
    
    print(f"Recommended model: {recommendation.recommended_model}")
    print(f"Confidence: {recommendation.confidence:.2f}")
    print(f"Reasoning: {recommendation.reasoning}")
    
    # Get performance insights
    insights = integration.get_performance_insights()
    print(f"Performance insights: {len(insights)}")
    
    # Get integration summary
    summary = integration.get_integration_summary()
    print(f"Integration summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())



























