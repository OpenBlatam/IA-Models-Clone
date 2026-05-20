"""
Self-Reflection Engine
Implements concepts from EvoAgent paper: self-planning, self-control, self-reflection
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ReflectionType(str, Enum):
    """Types of self-reflection"""
    PERFORMANCE = "performance"
    STRATEGY = "strategy"
    CAPABILITY = "capability"
    ERROR = "error"
    SUCCESS = "success"
    PERIODIC = "periodic"


@dataclass
class Reflection:
    """Self-reflection record"""
    timestamp: datetime
    reflection_type: ReflectionType
    insights: Dict[str, Any]
    actions_taken: List[str] = field(default_factory=list)
    confidence: float = 0.0


class SelfReflectionEngine:
    """
    Self-reflection engine for autonomous agents
    Implements EvoAgent concepts: self-planning, self-control, self-reflection
    """
    
    def __init__(self):
        self._reflections: List[Reflection] = []
        self._reflection_lock = asyncio.Lock()
        self._last_periodic_reflection: Optional[datetime] = None
        self._reflection_interval = 300  # 5 minutes
    
    async def reflect_on_performance(
        self,
        metrics: Dict[str, Any],
        recent_tasks: List[Dict[str, Any]]
    ) -> Reflection:
        """Reflect on agent performance"""
        insights = {
            "tasks_completed": metrics.get("tasks_completed", 0),
            "tasks_failed": metrics.get("tasks_failed", 0),
            "success_rate": self._calculate_success_rate(metrics),
            "average_task_time": self._calculate_avg_task_time(recent_tasks),
            "improvement_areas": self._identify_improvement_areas(metrics, recent_tasks),
            "strengths": self._identify_strengths(metrics, recent_tasks)
        }
        
        reflection = Reflection(
            timestamp=datetime.utcnow(),
            reflection_type=ReflectionType.PERFORMANCE,
            insights=insights,
            confidence=self._calculate_confidence(insights)
        )
        
        # Determine actions based on reflection
        reflection.actions_taken = await self._determine_actions(reflection)
        
        async with self._reflection_lock:
            self._reflections.append(reflection)
            if len(self._reflections) > 100:
                self._reflections = self._reflections[-100:]
        
        logger.info(f"Self-reflection on performance: {insights}")
        return reflection
    
    async def reflect_on_strategy(
        self,
        current_strategy: Dict[str, Any],
        outcomes: List[Dict[str, Any]]
    ) -> Reflection:
        """Reflect on current strategy and effectiveness"""
        insights = {
            "strategy_effectiveness": self._evaluate_strategy_effectiveness(outcomes),
            "strategy_adaptations_needed": self._identify_strategy_adaptations(outcomes),
            "alternative_strategies": self._suggest_alternatives(outcomes),
            "strategy_confidence": self._calculate_strategy_confidence(outcomes)
        }
        
        reflection = Reflection(
            timestamp=datetime.utcnow(),
            reflection_type=ReflectionType.STRATEGY,
            insights=insights,
            confidence=insights.get("strategy_confidence", 0.0)
        )
        
        reflection.actions_taken = await self._determine_strategy_actions(reflection)
        
        async with self._reflection_lock:
            self._reflections.append(reflection)
            if len(self._reflections) > 100:
                self._reflections = self._reflections[-100:]
        
        return reflection
    
    async def reflect_on_capabilities(
        self,
        capabilities: Dict[str, Any],
        task_requirements: List[Dict[str, Any]]
    ) -> Reflection:
        """Reflect on agent capabilities vs requirements"""
        insights = {
            "capability_gaps": self._identify_capability_gaps(capabilities, task_requirements),
            "underutilized_capabilities": self._identify_underutilized(capabilities),
            "capability_evolution": self._track_capability_evolution(capabilities),
            "learning_priorities": self._determine_learning_priorities(capabilities, task_requirements)
        }
        
        reflection = Reflection(
            timestamp=datetime.utcnow(),
            reflection_type=ReflectionType.CAPABILITY,
            insights=insights,
            confidence=0.7
        )
        
        async with self._reflection_lock:
            self._reflections.append(reflection)
            if len(self._reflections) > 100:
                self._reflections = self._reflections[-100:]
        
        return reflection
    
    async def periodic_reflection(self) -> Optional[Reflection]:
        """Perform periodic self-reflection"""
        now = datetime.utcnow()
        
        if self._last_periodic_reflection:
            elapsed = (now - self._last_periodic_reflection).total_seconds()
            if elapsed < self._reflection_interval:
                return None
        
        self._last_periodic_reflection = now
        
        insights = {
            "reflection_trigger": "periodic",
            "time_since_last": elapsed if self._last_periodic_reflection else 0,
            "overall_health": "good",  # Would be calculated from health checks
            "recommendations": []
        }
        
        reflection = Reflection(
            timestamp=now,
            reflection_type=ReflectionType.PERIODIC,
            insights=insights,
            confidence=0.5
        )
        
        async with self._reflection_lock:
            self._reflections.append(reflection)
            if len(self._reflections) > 100:
                self._reflections = self._reflections[-100:]
        
        logger.info("Periodic self-reflection completed")
        return reflection
    
    def _calculate_success_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate success rate from metrics"""
        completed = metrics.get("tasks_completed", 0)
        failed = metrics.get("tasks_failed", 0)
        total = completed + failed
        return completed / total if total > 0 else 0.0
    
    def _calculate_avg_task_time(self, tasks: List[Dict[str, Any]]) -> float:
        """Calculate average task completion time"""
        if not tasks:
            return 0.0
        
        times = []
        for task in tasks:
            if "started_at" in task and "completed_at" in task:
                try:
                    start = datetime.fromisoformat(task["started_at"])
                    end = datetime.fromisoformat(task["completed_at"])
                    times.append((end - start).total_seconds())
                except Exception:
                    pass
        
        return sum(times) / len(times) if times else 0.0
    
    def _identify_improvement_areas(self, metrics: Dict[str, Any], tasks: List[Dict[str, Any]]) -> List[str]:
        """Identify areas for improvement"""
        areas = []
        
        if metrics.get("tasks_failed", 0) > metrics.get("tasks_completed", 0) * 0.2:
            areas.append("error_handling")
        
        if self._calculate_avg_task_time(tasks) > 60:
            areas.append("task_efficiency")
        
        return areas
    
    def _identify_strengths(self, metrics: Dict[str, Any], tasks: List[Dict[str, Any]]) -> List[str]:
        """Identify agent strengths"""
        strengths = []
        
        if metrics.get("tasks_completed", 0) > 10:
            strengths.append("task_completion")
        
        if self._calculate_success_rate(metrics) > 0.8:
            strengths.append("reliability")
        
        return strengths
    
    def _calculate_confidence(self, insights: Dict[str, Any]) -> float:
        """Calculate confidence in reflection"""
        success_rate = insights.get("success_rate", 0.0)
        return min(success_rate, 1.0)
    
    async def _determine_actions(self, reflection: Reflection) -> List[str]:
        """Determine actions based on reflection"""
        actions = []
        
        if "error_handling" in reflection.insights.get("improvement_areas", []):
            actions.append("increase_retry_attempts")
            actions.append("improve_error_recovery")
        
        if "task_efficiency" in reflection.insights.get("improvement_areas", []):
            actions.append("optimize_task_processing")
        
        return actions
    
    def _evaluate_strategy_effectiveness(self, outcomes: List[Dict[str, Any]]) -> float:
        """Evaluate how effective current strategy is"""
        if not outcomes:
            return 0.5
        
        success_count = sum(1 for o in outcomes if o.get("outcome") == "success")
        return success_count / len(outcomes)
    
    def _identify_strategy_adaptations(self, outcomes: List[Dict[str, Any]]) -> List[str]:
        """Identify needed strategy adaptations"""
        adaptations = []
        
        failure_rate = sum(1 for o in outcomes if o.get("outcome") == "failure") / len(outcomes) if outcomes else 0
        
        if failure_rate > 0.3:
            adaptations.append("more_conservative_approach")
        elif failure_rate < 0.1:
            adaptations.append("more_aggressive_approach")
        
        return adaptations
    
    def _suggest_alternatives(self, outcomes: List[Dict[str, Any]]) -> List[str]:
        """Suggest alternative strategies"""
        return ["exploration_focused", "exploitation_focused", "balanced"]
    
    def _calculate_strategy_confidence(self, outcomes: List[Dict[str, Any]]) -> float:
        """Calculate confidence in current strategy"""
        return self._evaluate_strategy_effectiveness(outcomes)
    
    async def _determine_strategy_actions(self, reflection: Reflection) -> List[str]:
        """Determine strategy-related actions"""
        actions = []
        
        adaptations = reflection.insights.get("strategy_adaptations_needed", [])
        for adaptation in adaptations:
            if "conservative" in adaptation:
                actions.append("reduce_risk_taking")
            elif "aggressive" in adaptation:
                actions.append("increase_exploration")
        
        return actions
    
    def _identify_capability_gaps(self, capabilities: Dict[str, Any], requirements: List[Dict[str, Any]]) -> List[str]:
        """Identify gaps between capabilities and requirements"""
        gaps = []
        # Simplified: would compare capabilities with task requirements
        return gaps
    
    def _identify_underutilized(self, capabilities: Dict[str, Any]) -> List[str]:
        """Identify underutilized capabilities"""
        return []
    
    def _track_capability_evolution(self, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Track how capabilities have evolved"""
        return {"evolution_rate": 0.1, "new_capabilities": []}
    
    def _determine_learning_priorities(self, capabilities: Dict[str, Any], requirements: List[Dict[str, Any]]) -> List[str]:
        """Determine what to learn next"""
        return ["error_handling", "efficiency"]
    
    async def get_reflection_history(self, limit: int = 10) -> List[Reflection]:
        """Get recent reflection history"""
        async with self._reflection_lock:
            return self._reflections[-limit:]
    
    async def get_reflection_stats(self) -> Dict[str, Any]:
        """Get reflection statistics"""
        async with self._reflection_lock:
            return {
                "total_reflections": len(self._reflections),
                "reflection_types": {
                    rt.value: sum(1 for r in self._reflections if r.reflection_type == rt)
                    for rt in ReflectionType
                },
                "average_confidence": sum(r.confidence for r in self._reflections) / len(self._reflections) if self._reflections else 0.0,
                "last_reflection": self._reflections[-1].timestamp.isoformat() if self._reflections else None
            }
