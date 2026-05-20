"""
Lewis Hammond Framework
=======================

Paper: "Lewis Hammond"

Key concepts:
- Agent analysis and evaluation
- Performance metrics
- Behavioral analysis
- Agent assessment frameworks
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class AnalysisType(Enum):
    """Types of analysis."""
    PERFORMANCE = "performance"
    BEHAVIORAL = "behavioral"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"
    COMPREHENSIVE = "comprehensive"


class MetricCategory(Enum):
    """Metric categories."""
    SPEED = "speed"
    ACCURACY = "accuracy"
    RESOURCE = "resource"
    QUALITY = "quality"
    CONSISTENCY = "consistency"


@dataclass
class AnalysisResult:
    """Analysis result."""
    result_id: str
    analysis_type: AnalysisType
    metrics: Dict[str, float] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetric:
    """Performance metric."""
    metric_id: str
    category: MetricCategory
    value: float
    unit: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class LewisHammondAgent(BaseAgent):
    """
    Agent for comprehensive analysis and evaluation.
    
    Performs various types of analysis on agent behavior,
    performance, and efficiency.
    """
    
    def __init__(
        self,
        name: str,
        analysis_depth: str = "standard",  # basic, standard, comprehensive
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Lewis Hammond agent.
        
        Args:
            name: Agent name
            analysis_depth: Depth of analysis to perform
            config: Additional configuration
        """
        super().__init__(name, config)
        self.analysis_depth = analysis_depth
        
        # Analysis tracking
        self.analysis_results: List[AnalysisResult] = []
        self.performance_metrics: List[PerformanceMetric] = []
        self.behavioral_patterns: List[Dict[str, Any]] = []
        
        # Metrics
        self.analyses_performed = 0
        self.metrics_collected = 0
        self.patterns_identified = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Performance tracking
        self.performance_history: Dict[MetricCategory, List[float]] = {
            cat: [] for cat in MetricCategory
        }
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about analysis task.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with analysis plan
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Determine analysis type
        analysis_type = self._determine_analysis_type(task)
        
        # Plan analysis
        analysis_plan = self._plan_analysis(analysis_type, context)
        
        result = {
            "task": task,
            "analysis_type": analysis_type.value,
            "analysis_plan": analysis_plan,
            "analysis_depth": self.analysis_depth
        }
        
        self.state.add_step("think", result)
        return result
    
    def _determine_analysis_type(self, task: str) -> AnalysisType:
        """Determine type of analysis needed."""
        task_lower = task.lower()
        
        if "performance" in task_lower:
            return AnalysisType.PERFORMANCE
        elif "behavior" in task_lower or "behavioral" in task_lower:
            return AnalysisType.BEHAVIORAL
        elif "efficiency" in task_lower:
            return AnalysisType.EFFICIENCY
        elif "reliability" in task_lower:
            return AnalysisType.RELIABILITY
        else:
            return AnalysisType.COMPREHENSIVE
    
    def _plan_analysis(
        self,
        analysis_type: AnalysisType,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Plan analysis execution."""
        plan = {
            "analysis_type": analysis_type.value,
            "depth": self.analysis_depth,
            "metrics_to_collect": [],
            "steps": []
        }
        
        if analysis_type == AnalysisType.PERFORMANCE:
            plan["metrics_to_collect"] = [cat.value for cat in MetricCategory]
            plan["steps"] = ["Collect metrics", "Calculate averages", "Identify trends"]
        elif analysis_type == AnalysisType.BEHAVIORAL:
            plan["steps"] = ["Observe behavior", "Identify patterns", "Analyze consistency"]
        elif analysis_type == AnalysisType.COMPREHENSIVE:
            plan["metrics_to_collect"] = [cat.value for cat in MetricCategory]
            plan["steps"] = [
                "Performance analysis",
                "Behavioral analysis",
                "Efficiency analysis",
                "Reliability analysis",
                "Synthesize findings"
            ]
        
        return plan
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analysis action.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        action_type = action.get("type", "analyze")
        
        if action_type == "analyze":
            result = self._perform_analysis(action)
        elif action_type == "collect_metric":
            result = self._collect_metric(action)
        elif action_type == "identify_pattern":
            result = self._identify_pattern(action)
        else:
            result = self._execute_generic_action(action)
        
        self.state.add_step("act", {
            "action": action,
            "result": result
        })
        
        return result
    
    def _perform_analysis(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis."""
        analysis_type = AnalysisType(action.get("analysis_type", AnalysisType.COMPREHENSIVE.value))
        
        # Collect metrics
        metrics = {}
        for category in MetricCategory:
            if category in self.performance_history and self.performance_history[category]:
                avg_value = sum(self.performance_history[category]) / len(self.performance_history[category])
                metrics[category.value] = avg_value
            else:
                metrics[category.value] = 0.0
        
        # Generate findings
        findings = []
        if metrics.get("accuracy", 0) < 0.7:
            findings.append("Accuracy below threshold")
        if metrics.get("speed", 0) < 0.5:
            findings.append("Speed could be improved")
        
        # Generate recommendations
        recommendations = []
        if metrics.get("accuracy", 0) < 0.7:
            recommendations.append("Improve accuracy through better training")
        if metrics.get("efficiency", 0) < 0.6:
            recommendations.append("Optimize resource usage")
        
        result = AnalysisResult(
            result_id=f"analysis_{datetime.now().timestamp()}",
            analysis_type=analysis_type,
            metrics=metrics,
            findings=findings,
            recommendations=recommendations
        )
        
        self.analysis_results.append(result)
        self.analyses_performed += 1
        
        return {
            "status": "completed",
            "result_id": result.result_id,
            "analysis_type": analysis_type.value,
            "metrics": metrics,
            "findings_count": len(findings),
            "recommendations_count": len(recommendations)
        }
    
    def _collect_metric(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Collect performance metric."""
        category = MetricCategory(action.get("category", MetricCategory.SPEED.value))
        value = action.get("value", 0.0)
        unit = action.get("unit")
        
        metric = PerformanceMetric(
            metric_id=f"metric_{datetime.now().timestamp()}",
            category=category,
            value=value,
            unit=unit
        )
        
        self.performance_metrics.append(metric)
        self.metrics_collected += 1
        
        # Add to history
        self.performance_history[category].append(value)
        
        return {
            "status": "completed",
            "metric_id": metric.metric_id,
            "category": category.value,
            "value": value
        }
    
    def _identify_pattern(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Identify behavioral pattern."""
        pattern_type = action.get("pattern_type", "unknown")
        pattern_data = action.get("pattern_data", {})
        
        pattern = {
            "pattern_id": f"pattern_{datetime.now().timestamp()}",
            "pattern_type": pattern_type,
            "data": pattern_data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.behavioral_patterns.append(pattern)
        self.patterns_identified += 1
        
        return {
            "status": "completed",
            "pattern_id": pattern["pattern_id"],
            "pattern_type": pattern_type
        }
    
    def _execute_generic_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic action."""
        return {
            "status": "executed",
            "action": action,
            "timestamp": datetime.now().isoformat()
        }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update analysis state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Update metrics if provided
        if isinstance(observation, dict):
            if observation.get("performance_data"):
                self._update_performance_data(observation["performance_data"])
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "analyses_performed": self.analyses_performed,
                "metrics_collected": self.metrics_collected,
                "patterns_identified": self.patterns_identified
            }
        )
    
    def _update_performance_data(self, data: Dict[str, Any]):
        """Update performance data."""
        for category_str, values in data.items():
            try:
                category = MetricCategory(category_str)
                if isinstance(values, list):
                    self.performance_history[category].extend(values)
                else:
                    self.performance_history[category].append(values)
            except (ValueError, KeyError):
                pass
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run analysis task.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Prepare context
        if context is None:
            context = {}
        
        context["analysis_depth"] = self.analysis_depth
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add analysis-specific information
        result["analysis_summary"] = {
            "analyses_performed": self.analyses_performed,
            "metrics_collected": self.metrics_collected,
            "patterns_identified": self.patterns_identified,
            "analysis_depth": self.analysis_depth,
            "average_metrics": {
                cat.value: sum(values) / len(values) if values else 0.0
                for cat, values in self.performance_history.items()
            }
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "analyses_performed": self.analyses_performed,
            "metrics_collected": self.metrics_collected,
            "patterns_identified": self.patterns_identified,
            "analysis_depth": self.analysis_depth
        })
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """Get comprehensive analysis report."""
        latest_analysis = self.analysis_results[-1] if self.analysis_results else None
        
        return {
            "total_analyses": len(self.analysis_results),
            "total_metrics": len(self.performance_metrics),
            "total_patterns": len(self.behavioral_patterns),
            "latest_analysis": {
                "type": latest_analysis.analysis_type.value,
                "metrics": latest_analysis.metrics,
                "findings": latest_analysis.findings,
                "recommendations": latest_analysis.recommendations
            } if latest_analysis else None,
            "performance_summary": {
                cat.value: {
                    "count": len(values),
                    "average": sum(values) / len(values) if values else 0.0,
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0
                }
                for cat, values in self.performance_history.items()
            }
        }
