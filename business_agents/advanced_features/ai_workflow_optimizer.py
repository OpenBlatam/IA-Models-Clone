"""
AI Workflow Optimizer
=====================

Advanced AI-powered workflow optimization service that analyzes, improves,
and automates business workflows using machine learning and AI techniques.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import networkx as nx
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    PERFORMANCE = "performance"
    COST = "cost"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    USER_EXPERIENCE = "user_experience"
    AUTOMATION = "automation"

class OptimizationStrategy(Enum):
    PARALLEL_EXECUTION = "parallel_execution"
    CACHING = "caching"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    WORKFLOW_RESTRUCTURING = "workflow_restructuring"
    AUTOMATION_ENHANCEMENT = "automation_enhancement"
    ERROR_HANDLING = "error_handling"
    MONITORING_IMPROVEMENT = "monitoring_improvement"

@dataclass
class WorkflowMetrics:
    execution_time: float
    success_rate: float
    resource_usage: Dict[str, float]
    cost: float
    error_count: int
    throughput: float
    latency: float
    scalability_score: float
    maintainability_score: float

@dataclass
class OptimizationRecommendation:
    strategy: OptimizationStrategy
    description: str
    expected_improvement: float
    implementation_effort: str
    risk_level: str
    priority: int
    details: Dict[str, Any]

@dataclass
class WorkflowAnalysis:
    workflow_id: str
    current_metrics: WorkflowMetrics
    bottlenecks: List[str]
    inefficiencies: List[str]
    optimization_opportunities: List[OptimizationRecommendation]
    complexity_score: float
    automation_potential: float
    performance_trend: str

class AIWorkflowOptimizer:
    """
    Advanced AI-powered workflow optimization service.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_history: List[Dict[str, Any]] = []
        self.optimization_cache: Dict[str, Any] = {}
        self.ml_models: Dict[str, Any] = {}
        
        # Initialize ML models
        self._initialize_ml_models()
        
    def _initialize_ml_models(self):
        """Initialize machine learning models for optimization."""
        
        # Performance prediction model
        self.ml_models['performance_predictor'] = {
            'model': None,  # Would be a trained model in production
            'features': ['workflow_complexity', 'resource_usage', 'historical_performance'],
            'target': 'execution_time'
        }
        
        # Bottleneck detection model
        self.ml_models['bottleneck_detector'] = {
            'model': None,  # Would be a trained model in production
            'features': ['step_duration', 'resource_utilization', 'error_rate'],
            'target': 'bottleneck_probability'
        }
        
        # Optimization recommendation model
        self.ml_models['recommendation_engine'] = {
            'model': None,  # Would be a trained model in production
            'features': ['workflow_characteristics', 'performance_metrics', 'business_context'],
            'target': 'optimization_strategy'
        }
        
    async def analyze_workflow(self, workflow_data: Dict[str, Any]) -> WorkflowAnalysis:
        """Analyze workflow for optimization opportunities."""
        
        try:
            # Extract workflow metrics
            metrics = self._extract_workflow_metrics(workflow_data)
            
            # Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks(workflow_data, metrics)
            
            # Identify inefficiencies
            inefficiencies = await self._identify_inefficiencies(workflow_data, metrics)
            
            # Generate optimization recommendations
            recommendations = await self._generate_optimization_recommendations(
                workflow_data, metrics, bottlenecks, inefficiencies
            )
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(workflow_data)
            
            # Calculate automation potential
            automation_potential = self._calculate_automation_potential(workflow_data)
            
            # Analyze performance trend
            performance_trend = await self._analyze_performance_trend(workflow_data['id'])
            
            return WorkflowAnalysis(
                workflow_id=workflow_data['id'],
                current_metrics=metrics,
                bottlenecks=bottlenecks,
                inefficiencies=inefficiencies,
                optimization_opportunities=recommendations,
                complexity_score=complexity_score,
                automation_potential=automation_potential,
                performance_trend=performance_trend
            )
            
        except Exception as e:
            logger.error(f"Workflow analysis failed: {str(e)}")
            raise
            
    def _extract_workflow_metrics(self, workflow_data: Dict[str, Any]) -> WorkflowMetrics:
        """Extract metrics from workflow data."""
        
        # Calculate execution time
        execution_time = self._calculate_execution_time(workflow_data)
        
        # Calculate success rate
        success_rate = self._calculate_success_rate(workflow_data)
        
        # Calculate resource usage
        resource_usage = self._calculate_resource_usage(workflow_data)
        
        # Calculate cost
        cost = self._calculate_cost(workflow_data, resource_usage)
        
        # Calculate error count
        error_count = self._calculate_error_count(workflow_data)
        
        # Calculate throughput
        throughput = self._calculate_throughput(workflow_data)
        
        # Calculate latency
        latency = self._calculate_latency(workflow_data)
        
        # Calculate scalability score
        scalability_score = self._calculate_scalability_score(workflow_data)
        
        # Calculate maintainability score
        maintainability_score = self._calculate_maintainability_score(workflow_data)
        
        return WorkflowMetrics(
            execution_time=execution_time,
            success_rate=success_rate,
            resource_usage=resource_usage,
            cost=cost,
            error_count=error_count,
            throughput=throughput,
            latency=latency,
            scalability_score=scalability_score,
            maintainability_score=maintainability_score
        )
        
    def _calculate_execution_time(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate average execution time."""
        
        executions = workflow_data.get('executions', [])
        if not executions:
            return 0.0
            
        times = [exec.get('duration', 0) for exec in executions]
        return statistics.mean(times) if times else 0.0
        
    def _calculate_success_rate(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate success rate."""
        
        executions = workflow_data.get('executions', [])
        if not executions:
            return 0.0
            
        successful = sum(1 for exec in executions if exec.get('status') == 'completed')
        return successful / len(executions) if executions else 0.0
        
    def _calculate_resource_usage(self, workflow_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate resource usage metrics."""
        
        executions = workflow_data.get('executions', [])
        if not executions:
            return {}
            
        resource_usage = defaultdict(list)
        
        for execution in executions:
            resources = execution.get('resource_usage', {})
            for resource, usage in resources.items():
                resource_usage[resource].append(usage)
                
        # Calculate averages
        avg_usage = {}
        for resource, usages in resource_usage.items():
            avg_usage[resource] = statistics.mean(usages) if usages else 0.0
            
        return avg_usage
        
    def _calculate_cost(self, workflow_data: Dict[str, Any], resource_usage: Dict[str, float]) -> float:
        """Calculate workflow cost."""
        
        # Simple cost calculation based on resource usage
        cost_per_resource = {
            'cpu': 0.1,
            'memory': 0.05,
            'storage': 0.02,
            'network': 0.01
        }
        
        total_cost = 0.0
        for resource, usage in resource_usage.items():
            cost_rate = cost_per_resource.get(resource, 0.01)
            total_cost += usage * cost_rate
            
        return total_cost
        
    def _calculate_error_count(self, workflow_data: Dict[str, Any]) -> int:
        """Calculate total error count."""
        
        executions = workflow_data.get('executions', [])
        return sum(exec.get('error_count', 0) for exec in executions)
        
    def _calculate_throughput(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate workflow throughput."""
        
        executions = workflow_data.get('executions', [])
        if not executions:
            return 0.0
            
        # Calculate throughput as executions per hour
        time_span = self._get_workflow_time_span(executions)
        if time_span == 0:
            return 0.0
            
        return len(executions) / (time_span / 3600)  # executions per hour
        
    def _calculate_latency(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate average latency."""
        
        executions = workflow_data.get('executions', [])
        if not executions:
            return 0.0
            
        latencies = [exec.get('latency', 0) for exec in executions]
        return statistics.mean(latencies) if latencies else 0.0
        
    def _calculate_scalability_score(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate scalability score (0-1)."""
        
        # Factors affecting scalability
        factors = {
            'parallel_steps': len([step for step in workflow_data.get('steps', []) if step.get('parallel', False)]),
            'resource_dependencies': len(workflow_data.get('resource_dependencies', [])),
            'external_dependencies': len(workflow_data.get('external_dependencies', [])),
            'state_management': 1 if workflow_data.get('state_management') else 0
        }
        
        # Calculate score (higher is better)
        score = 0.0
        score += min(factors['parallel_steps'] * 0.2, 0.4)  # Parallel steps are good
        score += max(0, 0.3 - factors['resource_dependencies'] * 0.1)  # Fewer dependencies are better
        score += max(0, 0.2 - factors['external_dependencies'] * 0.05)  # Fewer external deps are better
        score += factors['state_management'] * 0.1  # State management is good
        
        return min(1.0, score)
        
    def _calculate_maintainability_score(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate maintainability score (0-1)."""
        
        # Factors affecting maintainability
        factors = {
            'step_count': len(workflow_data.get('steps', [])),
            'complexity': self._calculate_step_complexity(workflow_data.get('steps', [])),
            'documentation': 1 if workflow_data.get('documentation') else 0,
            'error_handling': len(workflow_data.get('error_handlers', [])),
            'modularity': self._calculate_modularity_score(workflow_data)
        }
        
        # Calculate score (higher is better)
        score = 0.0
        score += max(0, 0.3 - factors['step_count'] * 0.01)  # Fewer steps are better
        score += max(0, 0.2 - factors['complexity'] * 0.1)  # Lower complexity is better
        score += factors['documentation'] * 0.2  # Documentation is good
        score += min(factors['error_handling'] * 0.1, 0.2)  # Error handling is good
        score += factors['modularity'] * 0.1  # Modularity is good
        
        return min(1.0, score)
        
    def _calculate_step_complexity(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate average step complexity."""
        
        if not steps:
            return 0.0
            
        complexities = []
        for step in steps:
            complexity = 0.0
            complexity += len(step.get('conditions', [])) * 0.1
            complexity += len(step.get('actions', [])) * 0.05
            complexity += 1 if step.get('parallel', False) else 0.1
            complexity += len(step.get('dependencies', [])) * 0.05
            complexities.append(complexity)
            
        return statistics.mean(complexities) if complexities else 0.0
        
    def _calculate_modularity_score(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate modularity score."""
        
        # Check for reusable components
        reusable_components = workflow_data.get('reusable_components', [])
        total_steps = len(workflow_data.get('steps', []))
        
        if total_steps == 0:
            return 0.0
            
        return min(1.0, len(reusable_components) / total_steps)
        
    def _get_workflow_time_span(self, executions: List[Dict[str, Any]]) -> float:
        """Get time span of workflow executions in seconds."""
        
        if not executions:
            return 0.0
            
        timestamps = [exec.get('start_time', 0) for exec in executions]
        if not timestamps:
            return 0.0
            
        return max(timestamps) - min(timestamps)
        
    async def _identify_bottlenecks(self, workflow_data: Dict[str, Any], metrics: WorkflowMetrics) -> List[str]:
        """Identify workflow bottlenecks."""
        
        bottlenecks = []
        
        # Analyze step durations
        steps = workflow_data.get('steps', [])
        if steps:
            step_durations = [step.get('avg_duration', 0) for step in steps]
            if step_durations:
                avg_duration = statistics.mean(step_durations)
                for i, step in enumerate(steps):
                    if step.get('avg_duration', 0) > avg_duration * 2:
                        bottlenecks.append(f"Step '{step.get('name', f'Step {i}')}' is taking too long")
        
        # Analyze resource usage
        for resource, usage in metrics.resource_usage.items():
            if usage > 0.8:  # 80% threshold
                bottlenecks.append(f"High {resource} usage ({usage:.1%})")
        
        # Analyze error rates
        if metrics.error_count > 0:
            executions = workflow_data.get('executions', [])
            if executions:
                error_rate = metrics.error_count / len(executions)
                if error_rate > 0.1:  # 10% error rate threshold
                    bottlenecks.append(f"High error rate ({error_rate:.1%})")
        
        # Analyze dependencies
        dependencies = workflow_data.get('dependencies', [])
        if len(dependencies) > 5:
            bottlenecks.append("Too many dependencies causing delays")
        
        return bottlenecks
        
    async def _identify_inefficiencies(self, workflow_data: Dict[str, Any], metrics: WorkflowMetrics) -> List[str]:
        """Identify workflow inefficiencies."""
        
        inefficiencies = []
        
        # Check for sequential steps that could be parallel
        steps = workflow_data.get('steps', [])
        sequential_steps = [step for step in steps if not step.get('parallel', False)]
        if len(sequential_steps) > 3:
            inefficiencies.append("Multiple sequential steps could be parallelized")
        
        # Check for redundant steps
        step_names = [step.get('name', '') for step in steps]
        duplicate_names = [name for name, count in Counter(step_names).items() if count > 1]
        if duplicate_names:
            inefficiencies.append(f"Duplicate steps found: {', '.join(duplicate_names)}")
        
        # Check for unused resources
        for resource, usage in metrics.resource_usage.items():
            if usage < 0.1:  # Less than 10% usage
                inefficiencies.append(f"Low {resource} utilization ({usage:.1%})")
        
        # Check for long execution times
        if metrics.execution_time > 300:  # 5 minutes
            inefficiencies.append("Long execution time indicates potential optimization opportunities")
        
        # Check for low success rate
        if metrics.success_rate < 0.9:  # Less than 90% success rate
            inefficiencies.append(f"Low success rate ({metrics.success_rate:.1%}) indicates reliability issues")
        
        return inefficiencies
        
    async def _generate_optimization_recommendations(
        self,
        workflow_data: Dict[str, Any],
        metrics: WorkflowMetrics,
        bottlenecks: List[str],
        inefficiencies: List[str]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        
        recommendations = []
        
        # Performance optimization recommendations
        if metrics.execution_time > 60:  # More than 1 minute
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.PARALLEL_EXECUTION,
                description="Implement parallel execution for independent steps",
                expected_improvement=0.3,  # 30% improvement
                implementation_effort="medium",
                risk_level="low",
                priority=1,
                details={
                    "current_execution_time": metrics.execution_time,
                    "potential_parallel_steps": self._identify_parallel_candidates(workflow_data)
                }
            ))
        
        # Resource optimization recommendations
        if any(usage > 0.8 for usage in metrics.resource_usage.values()):
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.RESOURCE_OPTIMIZATION,
                description="Optimize resource allocation and usage",
                expected_improvement=0.2,  # 20% improvement
                implementation_effort="medium",
                risk_level="medium",
                priority=2,
                details={
                    "resource_usage": metrics.resource_usage,
                    "optimization_targets": [r for r, u in metrics.resource_usage.items() if u > 0.8]
                }
            ))
        
        # Caching recommendations
        if self._has_cacheable_operations(workflow_data):
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.CACHING,
                description="Implement caching for frequently accessed data",
                expected_improvement=0.4,  # 40% improvement
                implementation_effort="low",
                risk_level="low",
                priority=1,
                details={
                    "cacheable_operations": self._identify_cacheable_operations(workflow_data)
                }
            ))
        
        # Error handling recommendations
        if metrics.error_count > 0:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.ERROR_HANDLING,
                description="Improve error handling and recovery mechanisms",
                expected_improvement=0.15,  # 15% improvement
                implementation_effort="medium",
                risk_level="low",
                priority=2,
                details={
                    "current_error_count": metrics.error_count,
                    "error_types": self._analyze_error_types(workflow_data)
                }
            ))
        
        # Automation recommendations
        if metrics.automation_potential > 0.7:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.AUTOMATION_ENHANCEMENT,
                description="Enhance automation for manual steps",
                expected_improvement=0.5,  # 50% improvement
                implementation_effort="high",
                risk_level="medium",
                priority=3,
                details={
                    "automation_potential": metrics.automation_potential,
                    "manual_steps": self._identify_manual_steps(workflow_data)
                }
            ))
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority)
        
        return recommendations
        
    def _identify_parallel_candidates(self, workflow_data: Dict[str, Any]) -> List[str]:
        """Identify steps that could be executed in parallel."""
        
        steps = workflow_data.get('steps', [])
        parallel_candidates = []
        
        for step in steps:
            if not step.get('parallel', False) and not step.get('dependencies'):
                parallel_candidates.append(step.get('name', 'Unknown'))
        
        return parallel_candidates
        
    def _has_cacheable_operations(self, workflow_data: Dict[str, Any]) -> bool:
        """Check if workflow has cacheable operations."""
        
        steps = workflow_data.get('steps', [])
        for step in steps:
            if step.get('cacheable', False):
                return True
            # Check for data retrieval operations
            actions = step.get('actions', [])
            for action in actions:
                if action.get('type') in ['data_retrieval', 'api_call', 'database_query']:
                    return True
        
        return False
        
    def _identify_cacheable_operations(self, workflow_data: Dict[str, Any]) -> List[str]:
        """Identify operations that could benefit from caching."""
        
        cacheable_ops = []
        steps = workflow_data.get('steps', [])
        
        for step in steps:
            actions = step.get('actions', [])
            for action in actions:
                if action.get('type') in ['data_retrieval', 'api_call', 'database_query']:
                    cacheable_ops.append(f"{step.get('name', 'Unknown')}: {action.get('type')}")
        
        return cacheable_ops
        
    def _analyze_error_types(self, workflow_data: Dict[str, Any]) -> Dict[str, int]:
        """Analyze error types in workflow executions."""
        
        executions = workflow_data.get('executions', [])
        error_types = defaultdict(int)
        
        for execution in executions:
            errors = execution.get('errors', [])
            for error in errors:
                error_type = error.get('type', 'unknown')
                error_types[error_type] += 1
        
        return dict(error_types)
        
    def _identify_manual_steps(self, workflow_data: Dict[str, Any]) -> List[str]:
        """Identify manual steps in workflow."""
        
        steps = workflow_data.get('steps', [])
        manual_steps = []
        
        for step in steps:
            if step.get('manual', False):
                manual_steps.append(step.get('name', 'Unknown'))
        
        return manual_steps
        
    def _calculate_complexity_score(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate workflow complexity score (0-1, higher is more complex)."""
        
        factors = {
            'step_count': len(workflow_data.get('steps', [])),
            'dependency_count': len(workflow_data.get('dependencies', [])),
            'condition_count': sum(len(step.get('conditions', [])) for step in workflow_data.get('steps', [])),
            'parallel_count': len([step for step in workflow_data.get('steps', []) if step.get('parallel', False)])
        }
        
        # Calculate complexity score
        complexity = 0.0
        complexity += min(factors['step_count'] * 0.05, 0.3)  # Step count factor
        complexity += min(factors['dependency_count'] * 0.1, 0.3)  # Dependency factor
        complexity += min(factors['condition_count'] * 0.05, 0.2)  # Condition factor
        complexity += min(factors['parallel_count'] * 0.1, 0.2)  # Parallel factor
        
        return min(1.0, complexity)
        
    def _calculate_automation_potential(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate automation potential score (0-1, higher is more automatable)."""
        
        steps = workflow_data.get('steps', [])
        if not steps:
            return 0.0
        
        automatable_steps = 0
        for step in steps:
            if not step.get('manual', False) and step.get('automation_support', True):
                automatable_steps += 1
        
        return automatable_steps / len(steps)
        
    async def _analyze_performance_trend(self, workflow_id: str) -> str:
        """Analyze performance trend over time."""
        
        # Get historical data for this workflow
        historical_data = self._get_historical_data(workflow_id)
        
        if len(historical_data) < 3:
            return "insufficient_data"
        
        # Analyze trend
        execution_times = [data.get('execution_time', 0) for data in historical_data]
        success_rates = [data.get('success_rate', 0) for data in historical_data]
        
        # Calculate trends
        time_trend = self._calculate_trend(execution_times)
        success_trend = self._calculate_trend(success_rates)
        
        # Determine overall trend
        if time_trend < -0.1 and success_trend > 0.05:
            return "improving"
        elif time_trend > 0.1 and success_trend < -0.05:
            return "degrading"
        else:
            return "stable"
        
    def _get_historical_data(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get historical performance data for workflow."""
        
        # In a real implementation, this would query a database
        # For now, return mock data
        return [
            {"execution_time": 100, "success_rate": 0.95, "timestamp": "2024-01-01"},
            {"execution_time": 95, "success_rate": 0.96, "timestamp": "2024-01-02"},
            {"execution_time": 90, "success_rate": 0.97, "timestamp": "2024-01-03"},
        ]
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope for a series of values."""
        
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
        
    async def optimize_workflow(
        self,
        workflow_data: Dict[str, Any],
        optimization_type: OptimizationType,
        target_improvement: float = 0.2
    ) -> Dict[str, Any]:
        """Optimize workflow based on specified type and target improvement."""
        
        try:
            # Analyze current workflow
            analysis = await self.analyze_workflow(workflow_data)
            
            # Generate optimization plan
            optimization_plan = self._generate_optimization_plan(
                analysis, optimization_type, target_improvement
            )
            
            # Apply optimizations
            optimized_workflow = await self._apply_optimizations(
                workflow_data, optimization_plan
            )
            
            # Validate optimization
            validation_result = await self._validate_optimization(
                workflow_data, optimized_workflow, optimization_plan
            )
            
            return {
                "original_workflow": workflow_data,
                "optimized_workflow": optimized_workflow,
                "optimization_plan": optimization_plan,
                "validation_result": validation_result,
                "expected_improvement": target_improvement,
                "optimization_type": optimization_type.value
            }
            
        except Exception as e:
            logger.error(f"Workflow optimization failed: {str(e)}")
            raise
            
    def _generate_optimization_plan(
        self,
        analysis: WorkflowAnalysis,
        optimization_type: OptimizationType,
        target_improvement: float
    ) -> Dict[str, Any]:
        """Generate optimization plan based on analysis and requirements."""
        
        plan = {
            "optimization_type": optimization_type.value,
            "target_improvement": target_improvement,
            "strategies": [],
            "implementation_steps": [],
            "expected_outcomes": {},
            "risk_assessment": {}
        }
        
        # Select relevant recommendations
        relevant_recommendations = [
            rec for rec in analysis.optimization_opportunities
            if self._is_relevant_for_optimization_type(rec.strategy, optimization_type)
        ]
        
        # Sort by expected improvement
        relevant_recommendations.sort(key=lambda x: x.expected_improvement, reverse=True)
        
        # Build implementation plan
        cumulative_improvement = 0.0
        for recommendation in relevant_recommendations:
            if cumulative_improvement >= target_improvement:
                break
                
            plan["strategies"].append({
                "strategy": recommendation.strategy.value,
                "description": recommendation.description,
                "expected_improvement": recommendation.expected_improvement,
                "implementation_effort": recommendation.implementation_effort,
                "risk_level": recommendation.risk_level,
                "details": recommendation.details
            })
            
            cumulative_improvement += recommendation.expected_improvement
        
        return plan
        
    def _is_relevant_for_optimization_type(
        self,
        strategy: OptimizationStrategy,
        optimization_type: OptimizationType
    ) -> bool:
        """Check if strategy is relevant for optimization type."""
        
        relevance_map = {
            OptimizationType.PERFORMANCE: [
                OptimizationStrategy.PARALLEL_EXECUTION,
                OptimizationStrategy.CACHING,
                OptimizationStrategy.RESOURCE_OPTIMIZATION
            ],
            OptimizationType.COST: [
                OptimizationStrategy.RESOURCE_OPTIMIZATION,
                OptimizationStrategy.CACHING
            ],
            OptimizationType.EFFICIENCY: [
                OptimizationStrategy.PARALLEL_EXECUTION,
                OptimizationStrategy.WORKFLOW_RESTRUCTURING,
                OptimizationStrategy.AUTOMATION_ENHANCEMENT
            ],
            OptimizationType.RELIABILITY: [
                OptimizationStrategy.ERROR_HANDLING,
                OptimizationStrategy.MONITORING_IMPROVEMENT
            ],
            OptimizationType.SCALABILITY: [
                OptimizationStrategy.PARALLEL_EXECUTION,
                OptimizationStrategy.RESOURCE_OPTIMIZATION,
                OptimizationStrategy.WORKFLOW_RESTRUCTURING
            ],
            OptimizationType.USER_EXPERIENCE: [
                OptimizationStrategy.MONITORING_IMPROVEMENT,
                OptimizationStrategy.ERROR_HANDLING
            ],
            OptimizationType.AUTOMATION: [
                OptimizationStrategy.AUTOMATION_ENHANCEMENT,
                OptimizationStrategy.WORKFLOW_RESTRUCTURING
            ]
        }
        
        return strategy in relevance_map.get(optimization_type, [])
        
    async def _apply_optimizations(
        self,
        workflow_data: Dict[str, Any],
        optimization_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply optimizations to workflow."""
        
        optimized_workflow = workflow_data.copy()
        
        for strategy_info in optimization_plan["strategies"]:
            strategy = OptimizationStrategy(strategy_info["strategy"])
            
            if strategy == OptimizationStrategy.PARALLEL_EXECUTION:
                optimized_workflow = self._apply_parallel_execution(optimized_workflow)
            elif strategy == OptimizationStrategy.CACHING:
                optimized_workflow = self._apply_caching(optimized_workflow)
            elif strategy == OptimizationStrategy.RESOURCE_OPTIMIZATION:
                optimized_workflow = self._apply_resource_optimization(optimized_workflow)
            elif strategy == OptimizationStrategy.ERROR_HANDLING:
                optimized_workflow = self._apply_error_handling(optimized_workflow)
            elif strategy == OptimizationStrategy.AUTOMATION_ENHANCEMENT:
                optimized_workflow = self._apply_automation_enhancement(optimized_workflow)
        
        return optimized_workflow
        
    def _apply_parallel_execution(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parallel execution optimizations."""
        
        steps = workflow_data.get('steps', [])
        optimized_steps = []
        
        # Group independent steps for parallel execution
        independent_steps = []
        dependent_steps = []
        
        for step in steps:
            if not step.get('dependencies'):
                independent_steps.append(step)
            else:
                dependent_steps.append(step)
        
        # Create parallel groups
        if len(independent_steps) > 1:
            parallel_group = {
                "name": "Parallel Execution Group",
                "type": "parallel",
                "steps": independent_steps,
                "parallel": True
            }
            optimized_steps.append(parallel_group)
        else:
            optimized_steps.extend(independent_steps)
        
        optimized_steps.extend(dependent_steps)
        
        workflow_data['steps'] = optimized_steps
        return workflow_data
        
    def _apply_caching(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply caching optimizations."""
        
        steps = workflow_data.get('steps', [])
        
        for step in steps:
            actions = step.get('actions', [])
            for action in actions:
                if action.get('type') in ['data_retrieval', 'api_call', 'database_query']:
                    action['cacheable'] = True
                    action['cache_ttl'] = 3600  # 1 hour default
        
        return workflow_data
        
    def _apply_resource_optimization(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resource optimization."""
        
        # Add resource optimization configuration
        workflow_data['resource_optimization'] = {
            'cpu_limit': '80%',
            'memory_limit': '80%',
            'auto_scaling': True,
            'resource_monitoring': True
        }
        
        return workflow_data
        
    def _apply_error_handling(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply error handling improvements."""
        
        # Add comprehensive error handling
        workflow_data['error_handling'] = {
            'retry_policy': {
                'max_retries': 3,
                'retry_delay': 5,
                'exponential_backoff': True
            },
            'fallback_actions': [],
            'error_notifications': True
        }
        
        return workflow_data
        
    def _apply_automation_enhancement(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply automation enhancements."""
        
        steps = workflow_data.get('steps', [])
        
        for step in steps:
            if step.get('manual', False):
                step['automation_support'] = True
                step['automation_priority'] = 'high'
        
        return workflow_data
        
    async def _validate_optimization(
        self,
        original_workflow: Dict[str, Any],
        optimized_workflow: Dict[str, Any],
        optimization_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate optimization results."""
        
        validation_result = {
            "valid": True,
            "issues": [],
            "improvements": [],
            "recommendations": []
        }
        
        # Validate workflow structure
        if not self._validate_workflow_structure(optimized_workflow):
            validation_result["valid"] = False
            validation_result["issues"].append("Invalid workflow structure after optimization")
        
        # Check for breaking changes
        breaking_changes = self._detect_breaking_changes(original_workflow, optimized_workflow)
        if breaking_changes:
            validation_result["issues"].extend(breaking_changes)
        
        # Estimate improvements
        estimated_improvements = self._estimate_improvements(original_workflow, optimized_workflow)
        validation_result["improvements"] = estimated_improvements
        
        return validation_result
        
    def _validate_workflow_structure(self, workflow_data: Dict[str, Any]) -> bool:
        """Validate workflow structure."""
        
        # Basic validation checks
        if not workflow_data.get('steps'):
            return False
        
        # Check for circular dependencies
        if self._has_circular_dependencies(workflow_data):
            return False
        
        return True
        
    def _has_circular_dependencies(self, workflow_data: Dict[str, Any]) -> bool:
        """Check for circular dependencies in workflow."""
        
        steps = workflow_data.get('steps', [])
        dependencies = workflow_data.get('dependencies', [])
        
        # Build dependency graph
        graph = nx.DiGraph()
        
        for step in steps:
            graph.add_node(step.get('id', step.get('name')))
        
        for dep in dependencies:
            graph.add_edge(dep.get('from'), dep.get('to'))
        
        # Check for cycles
        try:
            nx.find_cycle(graph)
            return True
        except nx.NetworkXNoCycle:
            return False
        
    def _detect_breaking_changes(
        self,
        original_workflow: Dict[str, Any],
        optimized_workflow: Dict[str, Any]
    ) -> List[str]:
        """Detect breaking changes in optimization."""
        
        breaking_changes = []
        
        # Check for removed steps
        original_steps = {step.get('id', step.get('name')) for step in original_workflow.get('steps', [])}
        optimized_steps = {step.get('id', step.get('name')) for step in optimized_workflow.get('steps', [])}
        
        removed_steps = original_steps - optimized_steps
        if removed_steps:
            breaking_changes.append(f"Removed steps: {', '.join(removed_steps)}")
        
        # Check for changed step interfaces
        for original_step in original_workflow.get('steps', []):
            step_id = original_step.get('id', original_step.get('name'))
            optimized_step = next(
                (s for s in optimized_workflow.get('steps', []) 
                 if s.get('id', s.get('name')) == step_id), None
            )
            
            if optimized_step:
                # Check for interface changes
                if original_step.get('inputs') != optimized_step.get('inputs'):
                    breaking_changes.append(f"Step {step_id} input interface changed")
                
                if original_step.get('outputs') != optimized_step.get('outputs'):
                    breaking_changes.append(f"Step {step_id} output interface changed")
        
        return breaking_changes
        
    def _estimate_improvements(
        self,
        original_workflow: Dict[str, Any],
        optimized_workflow: Dict[str, Any]
    ) -> List[str]:
        """Estimate improvements from optimization."""
        
        improvements = []
        
        # Count parallel steps
        original_parallel = len([s for s in original_workflow.get('steps', []) if s.get('parallel', False)])
        optimized_parallel = len([s for s in optimized_workflow.get('steps', []) if s.get('parallel', False)])
        
        if optimized_parallel > original_parallel:
            improvements.append(f"Added {optimized_parallel - original_parallel} parallel execution groups")
        
        # Check for caching
        original_cacheable = sum(1 for step in original_workflow.get('steps', [])
                               for action in step.get('actions', [])
                               if action.get('cacheable', False))
        optimized_cacheable = sum(1 for step in optimized_workflow.get('steps', [])
                                for action in step.get('actions', [])
                                if action.get('cacheable', False))
        
        if optimized_cacheable > original_cacheable:
            improvements.append(f"Added caching to {optimized_cacheable - original_cacheable} operations")
        
        # Check for error handling
        if optimized_workflow.get('error_handling') and not original_workflow.get('error_handling'):
            improvements.append("Added comprehensive error handling")
        
        return improvements





























