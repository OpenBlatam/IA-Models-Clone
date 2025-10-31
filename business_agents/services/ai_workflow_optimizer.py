"""
AI Workflow Optimizer Service
=============================

Advanced AI-powered workflow optimization service that analyzes, improves,
and automates workflow performance using machine learning and AI.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from collections import defaultdict, Counter
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import aiohttp

from ..models import Workflow, WorkflowExecution, AgentExecution, BusinessAgent
from ..services.database_service import DatabaseService
from ..services.ai_service import AIService, AIProvider

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Types of workflow optimizations."""
    PERFORMANCE = "performance"
    COST = "cost"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    RESOURCE = "resource"
    TIME = "time"
    COMPREHENSIVE = "comprehensive"

class OptimizationStrategy(Enum):
    """Optimization strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    ML_BASED = "ml_based"

@dataclass
class WorkflowMetrics:
    """Workflow performance metrics."""
    execution_time: float
    success_rate: float
    resource_usage: float
    cost: float
    quality_score: float
    error_rate: float
    throughput: float
    latency: float
    cpu_usage: float
    memory_usage: float
    network_usage: float
    user_satisfaction: float
    business_value: float

@dataclass
class OptimizationSuggestion:
    """Workflow optimization suggestion."""
    suggestion_id: str
    workflow_id: str
    optimization_type: OptimizationType
    strategy: OptimizationStrategy
    description: str
    expected_improvement: float
    confidence_score: float
    implementation_effort: str
    risk_level: str
    prerequisites: List[str]
    steps: List[Dict[str, Any]]
    estimated_impact: Dict[str, float]
    cost_benefit_analysis: Dict[str, Any]
    created_at: datetime

@dataclass
class WorkflowPattern:
    """Identified workflow pattern."""
    pattern_id: str
    pattern_type: str
    frequency: int
    success_rate: float
    avg_execution_time: float
    common_issues: List[str]
    optimization_opportunities: List[str]
    similar_workflows: List[str]

@dataclass
class PerformancePrediction:
    """Workflow performance prediction."""
    workflow_id: str
    predicted_execution_time: float
    predicted_success_rate: float
    predicted_resource_usage: float
    confidence_interval: Tuple[float, float]
    factors: Dict[str, float]
    recommendations: List[str]

class AIWorkflowOptimizer:
    """
    AI-powered workflow optimization service.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_service = DatabaseService(config)
        self.ai_service = AIService(config)
        self.ml_models = {}
        self.optimization_history = []
        self.pattern_cache = {}
        self.performance_cache = {}
        
        # ML model configurations
        self.model_configs = {
            "execution_time": {
                "model": RandomForestRegressor(n_estimators=100, random_state=42),
                "features": ["step_count", "agent_count", "complexity", "resource_requirements"]
            },
            "success_rate": {
                "model": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "features": ["error_history", "agent_reliability", "workflow_complexity"]
            },
            "resource_usage": {
                "model": RandomForestRegressor(n_estimators=100, random_state=42),
                "features": ["step_count", "parallel_steps", "agent_capabilities"]
            }
        }
        
    async def initialize(self):
        """Initialize the optimizer."""
        try:
            await self.db_service.initialize()
            await self.ai_service.initialize()
            await self._load_ml_models()
            await self._analyze_historical_data()
            logger.info("AI Workflow Optimizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI Workflow Optimizer: {str(e)}")
            raise
            
    async def _load_ml_models(self):
        """Load or train ML models."""
        try:
            # Load historical data for training
            historical_data = await self._get_historical_workflow_data()
            
            if len(historical_data) > 100:  # Minimum data for training
                for model_name, config in self.model_configs.items():
                    model = config["model"]
                    features = config["features"]
                    
                    # Prepare training data
                    X, y = self._prepare_training_data(historical_data, features, model_name)
                    
                    if len(X) > 0:
                        # Train model
                        model.fit(X, y)
                        self.ml_models[model_name] = model
                        logger.info(f"Trained {model_name} model with {len(X)} samples")
                        
        except Exception as e:
            logger.error(f"Failed to load ML models: {str(e)}")
            
    async def _get_historical_workflow_data(self) -> List[Dict[str, Any]]:
        """Get historical workflow execution data."""
        try:
            # Get workflow executions from last 6 months
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=180)
            
            # This would query the database for historical data
            # For now, return sample data
            return [
                {
                    "workflow_id": "wf_001",
                    "execution_time": 120.5,
                    "success_rate": 0.95,
                    "resource_usage": 0.75,
                    "step_count": 5,
                    "agent_count": 3,
                    "complexity": 0.6,
                    "error_rate": 0.05
                },
                # Add more historical data...
            ]
        except Exception as e:
            logger.error(f"Failed to get historical data: {str(e)}")
            return []
            
    def _prepare_training_data(self, data: List[Dict[str, Any]], features: List[str], target: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        try:
            df = pd.DataFrame(data)
            
            # Select features and target
            X = df[features].fillna(0).values
            y = df[target].fillna(0).values
            
            return X, y
        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            return np.array([]), np.array([])
            
    async def _analyze_historical_data(self):
        """Analyze historical data for patterns and insights."""
        try:
            historical_data = await self._get_historical_workflow_data()
            
            if len(historical_data) > 0:
                # Identify patterns
                patterns = await self._identify_workflow_patterns(historical_data)
                self.pattern_cache = {pattern.pattern_id: pattern for pattern in patterns}
                
                # Analyze performance trends
                performance_trends = await self._analyze_performance_trends(historical_data)
                self.performance_cache = performance_trends
                
                logger.info(f"Analyzed {len(historical_data)} historical records")
                
        except Exception as e:
            logger.error(f"Failed to analyze historical data: {str(e)}")
            
    async def _identify_workflow_patterns(self, data: List[Dict[str, Any]]) -> List[WorkflowPattern]:
        """Identify common workflow patterns."""
        try:
            patterns = []
            
            # Group workflows by characteristics
            workflow_groups = defaultdict(list)
            for record in data:
                key = f"{record.get('step_count', 0)}_{record.get('agent_count', 0)}"
                workflow_groups[key].append(record)
                
            # Analyze each group
            for group_key, group_data in workflow_groups.items():
                if len(group_data) >= 5:  # Minimum group size
                    pattern = WorkflowPattern(
                        pattern_id=f"pattern_{group_key}",
                        pattern_type=group_key,
                        frequency=len(group_data),
                        success_rate=np.mean([r.get('success_rate', 0) for r in group_data]),
                        avg_execution_time=np.mean([r.get('execution_time', 0) for r in group_data]),
                        common_issues=self._identify_common_issues(group_data),
                        optimization_opportunities=self._identify_optimization_opportunities(group_data),
                        similar_workflows=[r.get('workflow_id', '') for r in group_data]
                    )
                    patterns.append(pattern)
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to identify workflow patterns: {str(e)}")
            return []
            
    def _identify_common_issues(self, data: List[Dict[str, Any]]) -> List[str]:
        """Identify common issues in workflow data."""
        issues = []
        
        # Analyze error rates
        error_rates = [r.get('error_rate', 0) for r in data]
        if np.mean(error_rates) > 0.1:
            issues.append("High error rate")
            
        # Analyze execution times
        execution_times = [r.get('execution_time', 0) for r in data]
        if np.std(execution_times) > np.mean(execution_times) * 0.5:
            issues.append("Inconsistent execution times")
            
        # Analyze resource usage
        resource_usage = [r.get('resource_usage', 0) for r in data]
        if np.mean(resource_usage) > 0.8:
            issues.append("High resource usage")
            
        return issues
        
    def _identify_optimization_opportunities(self, data: List[Dict[str, Any]]) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Check for parallelization opportunities
        step_counts = [r.get('step_count', 0) for r in data]
        if np.mean(step_counts) > 5:
            opportunities.append("Parallel execution opportunities")
            
        # Check for resource optimization
        resource_usage = [r.get('resource_usage', 0) for r in data]
        if np.mean(resource_usage) > 0.6:
            opportunities.append("Resource optimization potential")
            
        # Check for automation opportunities
        agent_counts = [r.get('agent_count', 0) for r in data]
        if np.mean(agent_counts) > 3:
            opportunities.append("Agent consolidation opportunities")
            
        return opportunities
        
    async def _analyze_performance_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            if not data:
                return {}
                
            df = pd.DataFrame(data)
            
            trends = {
                "execution_time_trend": self._calculate_trend(df, 'execution_time'),
                "success_rate_trend": self._calculate_trend(df, 'success_rate'),
                "resource_usage_trend": self._calculate_trend(df, 'resource_usage'),
                "error_rate_trend": self._calculate_trend(df, 'error_rate')
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {str(e)}")
            return {}
            
    def _calculate_trend(self, df: pd.DataFrame, column: str) -> Dict[str, float]:
        """Calculate trend for a specific column."""
        try:
            if column not in df.columns:
                return {"slope": 0, "r_squared": 0, "trend": "stable"}
                
            values = df[column].dropna()
            if len(values) < 2:
                return {"slope": 0, "r_squared": 0, "trend": "stable"}
                
            x = np.arange(len(values))
            y = values.values
            
            # Calculate linear regression
            slope = np.polyfit(x, y, 1)[0]
            r_squared = np.corrcoef(x, y)[0, 1] ** 2
            
            # Determine trend
            if abs(slope) < 0.01:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
                
            return {
                "slope": slope,
                "r_squared": r_squared,
                "trend": trend
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate trend for {column}: {str(e)}")
            return {"slope": 0, "r_squared": 0, "trend": "stable"}
            
    async def optimize_workflow(self, workflow_id: str, optimization_type: OptimizationType = OptimizationType.COMPREHENSIVE) -> List[OptimizationSuggestion]:
        """Optimize a specific workflow."""
        try:
            # Get workflow data
            workflow = await self.db_service.get_workflow_by_id(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
                
            # Get workflow execution history
            execution_history = await self._get_workflow_execution_history(workflow_id)
            
            # Analyze current performance
            current_metrics = await self._analyze_workflow_performance(workflow, execution_history)
            
            # Generate optimization suggestions
            suggestions = []
            
            if optimization_type in [OptimizationType.PERFORMANCE, OptimizationType.COMPREHENSIVE]:
                suggestions.extend(await self._generate_performance_optimizations(workflow, current_metrics))
                
            if optimization_type in [OptimizationType.COST, OptimizationType.COMPREHENSIVE]:
                suggestions.extend(await self._generate_cost_optimizations(workflow, current_metrics))
                
            if optimization_type in [OptimizationType.QUALITY, OptimizationType.COMPREHENSIVE]:
                suggestions.extend(await self._generate_quality_optimizations(workflow, current_metrics))
                
            if optimization_type in [OptimizationType.EFFICIENCY, OptimizationType.COMPREHENSIVE]:
                suggestions.extend(await self._generate_efficiency_optimizations(workflow, current_metrics))
                
            # Sort suggestions by expected improvement
            suggestions.sort(key=lambda x: x.expected_improvement, reverse=True)
            
            # Store optimization history
            self.optimization_history.append({
                "workflow_id": workflow_id,
                "optimization_type": optimization_type.value,
                "suggestions_count": len(suggestions),
                "timestamp": datetime.utcnow()
            })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to optimize workflow {workflow_id}: {str(e)}")
            raise
            
    async def _get_workflow_execution_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        try:
            # This would query the database for execution history
            # For now, return sample data
            return [
                {
                    "execution_id": "exec_001",
                    "started_at": datetime.utcnow() - timedelta(hours=1),
                    "completed_at": datetime.utcnow() - timedelta(minutes=30),
                    "status": "completed",
                    "duration": 1800,
                    "success": True,
                    "error_message": None,
                    "resource_usage": 0.75
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get execution history for workflow {workflow_id}: {str(e)}")
            return []
            
    async def _analyze_workflow_performance(self, workflow: Workflow, execution_history: List[Dict[str, Any]]) -> WorkflowMetrics:
        """Analyze workflow performance metrics."""
        try:
            if not execution_history:
                # Return default metrics for new workflows
                return WorkflowMetrics(
                    execution_time=0,
                    success_rate=0,
                    resource_usage=0,
                    cost=0,
                    quality_score=0,
                    error_rate=0,
                    throughput=0,
                    latency=0,
                    cpu_usage=0,
                    memory_usage=0,
                    network_usage=0,
                    user_satisfaction=0,
                    business_value=0
                )
                
            # Calculate metrics from execution history
            durations = [exec.get('duration', 0) for exec in execution_history]
            successes = [exec.get('success', False) for exec in execution_history]
            resource_usage = [exec.get('resource_usage', 0) for exec in execution_history]
            
            return WorkflowMetrics(
                execution_time=np.mean(durations) if durations else 0,
                success_rate=np.mean(successes) if successes else 0,
                resource_usage=np.mean(resource_usage) if resource_usage else 0,
                cost=self._calculate_cost(workflow, execution_history),
                quality_score=self._calculate_quality_score(workflow, execution_history),
                error_rate=1 - np.mean(successes) if successes else 0,
                throughput=len(execution_history) / max(durations) if durations else 0,
                latency=np.mean(durations) if durations else 0,
                cpu_usage=np.mean(resource_usage) if resource_usage else 0,
                memory_usage=np.mean(resource_usage) * 0.8 if resource_usage else 0,
                network_usage=np.mean(resource_usage) * 0.2 if resource_usage else 0,
                user_satisfaction=self._calculate_user_satisfaction(workflow, execution_history),
                business_value=self._calculate_business_value(workflow, execution_history)
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze workflow performance: {str(e)}")
            return WorkflowMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
    def _calculate_cost(self, workflow: Workflow, execution_history: List[Dict[str, Any]]) -> float:
        """Calculate workflow cost."""
        # Simple cost calculation based on execution time and resource usage
        total_cost = 0
        for exec in execution_history:
            duration = exec.get('duration', 0)
            resource_usage = exec.get('resource_usage', 0)
            cost = (duration / 3600) * resource_usage * 0.1  # $0.1 per hour per resource unit
            total_cost += cost
        return total_cost
        
    def _calculate_quality_score(self, workflow: Workflow, execution_history: List[Dict[str, Any]]) -> float:
        """Calculate workflow quality score."""
        if not execution_history:
            return 0
            
        # Quality score based on success rate, consistency, and error handling
        success_rate = np.mean([exec.get('success', False) for exec in execution_history])
        consistency = 1 - np.std([exec.get('duration', 0) for exec in execution_history]) / np.mean([exec.get('duration', 0) for exec in execution_history]) if execution_history else 0
        error_handling = 1 - len([exec for exec in execution_history if exec.get('error_message')]) / len(execution_history)
        
        return (success_rate + consistency + error_handling) / 3
        
    def _calculate_user_satisfaction(self, workflow: Workflow, execution_history: List[Dict[str, Any]]) -> float:
        """Calculate user satisfaction score."""
        # Simple satisfaction calculation based on success rate and execution time
        if not execution_history:
            return 0
            
        success_rate = np.mean([exec.get('success', False) for exec in execution_history])
        avg_duration = np.mean([exec.get('duration', 0) for exec in execution_history])
        
        # Satisfaction decreases with longer execution times
        time_satisfaction = max(0, 1 - (avg_duration / 3600))  # 1 hour = 0 satisfaction
        
        return (success_rate + time_satisfaction) / 2
        
    def _calculate_business_value(self, workflow: Workflow, execution_history: List[Dict[str, Any]]) -> float:
        """Calculate business value score."""
        # Business value based on frequency of execution and success rate
        if not execution_history:
            return 0
            
        frequency = len(execution_history)
        success_rate = np.mean([exec.get('success', False) for exec in execution_history])
        
        # Higher frequency and success rate = higher business value
        return (frequency / 100) * success_rate  # Normalize frequency
        
    async def _generate_performance_optimizations(self, workflow: Workflow, metrics: WorkflowMetrics) -> List[OptimizationSuggestion]:
        """Generate performance optimization suggestions."""
        suggestions = []
        
        # Parallel execution optimization
        if metrics.execution_time > 300:  # 5 minutes
            suggestion = OptimizationSuggestion(
                suggestion_id=f"perf_parallel_{workflow.id}",
                workflow_id=workflow.id,
                optimization_type=OptimizationType.PERFORMANCE,
                strategy=OptimizationStrategy.PARALLEL,
                description="Implement parallel execution for independent steps",
                expected_improvement=0.4,  # 40% improvement
                confidence_score=0.85,
                implementation_effort="medium",
                risk_level="low",
                prerequisites=["Step dependency analysis", "Resource availability check"],
                steps=[
                    {"action": "Analyze step dependencies", "duration": "2 hours"},
                    {"action": "Identify parallelizable steps", "duration": "1 hour"},
                    {"action": "Implement parallel execution", "duration": "4 hours"},
                    {"action": "Test and validate", "duration": "2 hours"}
                ],
                estimated_impact={"execution_time": -0.4, "resource_usage": 0.2},
                cost_benefit_analysis={"cost": 1000, "benefit": 5000, "roi": 5.0},
                created_at=datetime.utcnow()
            )
            suggestions.append(suggestion)
            
        # Resource optimization
        if metrics.resource_usage > 0.8:
            suggestion = OptimizationSuggestion(
                suggestion_id=f"perf_resource_{workflow.id}",
                workflow_id=workflow.id,
                optimization_type=OptimizationType.PERFORMANCE,
                strategy=OptimizationStrategy.ML_BASED,
                description="Optimize resource allocation using ML predictions",
                expected_improvement=0.25,  # 25% improvement
                confidence_score=0.75,
                implementation_effort="high",
                risk_level="medium",
                prerequisites=["ML model training", "Resource monitoring setup"],
                steps=[
                    {"action": "Train resource prediction model", "duration": "8 hours"},
                    {"action": "Implement dynamic resource allocation", "duration": "6 hours"},
                    {"action": "Deploy and monitor", "duration": "4 hours"}
                ],
                estimated_impact={"resource_usage": -0.25, "cost": -0.15},
                cost_benefit_analysis={"cost": 2000, "benefit": 8000, "roi": 4.0},
                created_at=datetime.utcnow()
            )
            suggestions.append(suggestion)
            
        return suggestions
        
    async def _generate_cost_optimizations(self, workflow: Workflow, metrics: WorkflowMetrics) -> List[OptimizationSuggestion]:
        """Generate cost optimization suggestions."""
        suggestions = []
        
        # Agent consolidation
        if len(workflow.steps) > 5:
            suggestion = OptimizationSuggestion(
                suggestion_id=f"cost_consolidation_{workflow.id}",
                workflow_id=workflow.id,
                optimization_type=OptimizationType.COST,
                strategy=OptimizationStrategy.HYBRID,
                description="Consolidate similar agents to reduce costs",
                expected_improvement=0.3,  # 30% cost reduction
                confidence_score=0.8,
                implementation_effort="medium",
                risk_level="low",
                prerequisites=["Agent capability analysis", "Workflow testing"],
                steps=[
                    {"action": "Analyze agent capabilities", "duration": "3 hours"},
                    {"action": "Identify consolidation opportunities", "duration": "2 hours"},
                    {"action": "Implement consolidated agents", "duration": "5 hours"},
                    {"action": "Test and validate", "duration": "3 hours"}
                ],
                estimated_impact={"cost": -0.3, "execution_time": 0.1},
                cost_benefit_analysis={"cost": 1500, "benefit": 6000, "roi": 4.0},
                created_at=datetime.utcnow()
            )
            suggestions.append(suggestion)
            
        return suggestions
        
    async def _generate_quality_optimizations(self, workflow: Workflow, metrics: WorkflowMetrics) -> List[OptimizationSuggestion]:
        """Generate quality optimization suggestions."""
        suggestions = []
        
        # Error handling improvement
        if metrics.error_rate > 0.1:
            suggestion = OptimizationSuggestion(
                suggestion_id=f"quality_error_handling_{workflow.id}",
                workflow_id=workflow.id,
                optimization_type=OptimizationType.QUALITY,
                strategy=OptimizationStrategy.ADAPTIVE,
                description="Implement advanced error handling and recovery",
                expected_improvement=0.5,  # 50% error reduction
                confidence_score=0.9,
                implementation_effort="medium",
                risk_level="low",
                prerequisites=["Error analysis", "Recovery strategy design"],
                steps=[
                    {"action": "Analyze error patterns", "duration": "2 hours"},
                    {"action": "Design recovery strategies", "duration": "3 hours"},
                    {"action": "Implement error handling", "duration": "4 hours"},
                    {"action": "Test error scenarios", "duration": "3 hours"}
                ],
                estimated_impact={"error_rate": -0.5, "success_rate": 0.3},
                cost_benefit_analysis={"cost": 1200, "benefit": 4000, "roi": 3.3},
                created_at=datetime.utcnow()
            )
            suggestions.append(suggestion)
            
        return suggestions
        
    async def _generate_efficiency_optimizations(self, workflow: Workflow, metrics: WorkflowMetrics) -> List[OptimizationSuggestion]:
        """Generate efficiency optimization suggestions."""
        suggestions = []
        
        # Step optimization
        if len(workflow.steps) > 8:
            suggestion = OptimizationSuggestion(
                suggestion_id=f"efficiency_steps_{workflow.id}",
                workflow_id=workflow.id,
                optimization_type=OptimizationType.EFFICIENCY,
                strategy=OptimizationStrategy.ML_BASED,
                description="Optimize workflow steps using AI analysis",
                expected_improvement=0.35,  # 35% efficiency improvement
                confidence_score=0.8,
                implementation_effort="high",
                risk_level="medium",
                prerequisites=["Step analysis", "AI model training"],
                steps=[
                    {"action": "Analyze step efficiency", "duration": "4 hours"},
                    {"action": "Train optimization model", "duration": "6 hours"},
                    {"action": "Implement optimized steps", "duration": "5 hours"},
                    {"action": "Validate improvements", "duration": "3 hours"}
                ],
                estimated_impact={"execution_time": -0.35, "efficiency": 0.35},
                cost_benefit_analysis={"cost": 1800, "benefit": 7000, "roi": 3.9},
                created_at=datetime.utcnow()
            )
            suggestions.append(suggestion)
            
        return suggestions
        
    async def predict_workflow_performance(self, workflow_id: str, input_data: Dict[str, Any]) -> PerformancePrediction:
        """Predict workflow performance using ML models."""
        try:
            # Get workflow data
            workflow = await self.db_service.get_workflow_by_id(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
                
            # Prepare features for prediction
            features = self._prepare_prediction_features(workflow, input_data)
            
            # Make predictions using trained models
            predictions = {}
            confidence_intervals = {}
            
            for model_name, model in self.ml_models.items():
                if model_name in features:
                    try:
                        prediction = model.predict([features[model_name]])[0]
                        predictions[model_name] = prediction
                        
                        # Calculate confidence interval (simplified)
                        confidence_intervals[model_name] = (prediction * 0.9, prediction * 1.1)
                    except Exception as e:
                        logger.error(f"Failed to make prediction with {model_name}: {str(e)}")
                        
            # Generate recommendations
            recommendations = await self._generate_performance_recommendations(workflow, predictions)
            
            return PerformancePrediction(
                workflow_id=workflow_id,
                predicted_execution_time=predictions.get("execution_time", 0),
                predicted_success_rate=predictions.get("success_rate", 0),
                predicted_resource_usage=predictions.get("resource_usage", 0),
                confidence_interval=confidence_intervals.get("execution_time", (0, 0)),
                factors=features,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to predict workflow performance: {str(e)}")
            raise
            
    def _prepare_prediction_features(self, workflow: Workflow, input_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Prepare features for ML prediction."""
        features = {}
        
        # Execution time features
        features["execution_time"] = [
            len(workflow.steps),  # step_count
            len(workflow.agents) if hasattr(workflow, 'agents') else 0,  # agent_count
            self._calculate_complexity(workflow),  # complexity
            input_data.get("resource_requirements", 0.5)  # resource_requirements
        ]
        
        # Success rate features
        features["success_rate"] = [
            input_data.get("error_history", 0.1),  # error_history
            input_data.get("agent_reliability", 0.9),  # agent_reliability
            self._calculate_complexity(workflow)  # workflow_complexity
        ]
        
        # Resource usage features
        features["resource_usage"] = [
            len(workflow.steps),  # step_count
            self._count_parallel_steps(workflow),  # parallel_steps
            input_data.get("agent_capabilities", 0.7)  # agent_capabilities
        ]
        
        return features
        
    def _calculate_complexity(self, workflow: Workflow) -> float:
        """Calculate workflow complexity score."""
        # Simple complexity calculation based on steps and configuration
        base_complexity = len(workflow.steps) / 10.0  # Normalize to 0-1
        config_complexity = len(workflow.configuration) / 20.0 if workflow.configuration else 0
        
        return min(1.0, base_complexity + config_complexity)
        
    def _count_parallel_steps(self, workflow: Workflow) -> int:
        """Count potentially parallel steps."""
        # Simple parallel step counting
        return max(1, len(workflow.steps) // 2)
        
    async def _generate_performance_recommendations(self, workflow: Workflow, predictions: Dict[str, float]) -> List[str]:
        """Generate performance recommendations based on predictions."""
        recommendations = []
        
        # Execution time recommendations
        if predictions.get("execution_time", 0) > 300:  # 5 minutes
            recommendations.append("Consider implementing parallel execution for long-running workflows")
            
        # Success rate recommendations
        if predictions.get("success_rate", 0) < 0.9:
            recommendations.append("Implement additional error handling and validation steps")
            
        # Resource usage recommendations
        if predictions.get("resource_usage", 0) > 0.8:
            recommendations.append("Optimize resource allocation and consider scaling")
            
        return recommendations
        
    async def get_optimization_insights(self) -> Dict[str, Any]:
        """Get overall optimization insights."""
        try:
            insights = {
                "total_optimizations": len(self.optimization_history),
                "patterns_identified": len(self.pattern_cache),
                "ml_models_trained": len(self.ml_models),
                "performance_trends": self.performance_cache,
                "top_optimization_types": self._get_top_optimization_types(),
                "average_improvement": self._calculate_average_improvement(),
                "optimization_recommendations": await self._get_global_recommendations()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get optimization insights: {str(e)}")
            return {}
            
    def _get_top_optimization_types(self) -> List[Dict[str, Any]]:
        """Get top optimization types by frequency."""
        type_counts = Counter([opt["optimization_type"] for opt in self.optimization_history])
        return [{"type": opt_type, "count": count} for opt_type, count in type_counts.most_common(5)]
        
    def _calculate_average_improvement(self) -> float:
        """Calculate average improvement across all optimizations."""
        # This would be calculated from actual optimization results
        return 0.25  # 25% average improvement
        
    async def _get_global_recommendations(self) -> List[str]:
        """Get global optimization recommendations."""
        recommendations = [
            "Implement parallel execution for workflows with >5 steps",
            "Use ML-based resource allocation for high-usage workflows",
            "Add error handling for workflows with >10% error rate",
            "Consider agent consolidation for cost optimization",
            "Implement performance monitoring for all critical workflows"
        ]
        
        return recommendations
        
    async def apply_optimization(self, suggestion_id: str, workflow_id: str) -> Dict[str, Any]:
        """Apply an optimization suggestion to a workflow."""
        try:
            # Find the suggestion
            suggestion = None
            for opt in self.optimization_history:
                if opt.get("suggestion_id") == suggestion_id:
                    suggestion = opt
                    break
                    
            if not suggestion:
                raise ValueError(f"Optimization suggestion {suggestion_id} not found")
                
            # Apply the optimization
            result = await self._apply_optimization_steps(suggestion, workflow_id)
            
            # Update optimization history
            self.optimization_history.append({
                "suggestion_id": suggestion_id,
                "workflow_id": workflow_id,
                "applied_at": datetime.utcnow(),
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply optimization {suggestion_id}: {str(e)}")
            raise
            
    async def _apply_optimization_steps(self, suggestion: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        """Apply optimization steps to a workflow."""
        try:
            # This would implement the actual optimization steps
            # For now, return a success response
            
            return {
                "status": "success",
                "applied_steps": len(suggestion.get("steps", [])),
                "estimated_improvement": suggestion.get("expected_improvement", 0),
                "applied_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to apply optimization steps: {str(e)}")
            return {"status": "error", "error": str(e)}




























