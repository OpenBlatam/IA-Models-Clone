"""
Learning Engine for Self-Initiated Learning
Implements concepts from "AI Autonomy: Self-initiated Open-world Continual Learning"
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LearningEvent:
    """Represents a learning event"""
    timestamp: datetime
    event_type: str
    context: Dict[str, Any]
    outcome: Optional[str] = None
    adaptation_applied: bool = False


class LearningEngine:
    """
    Self-initiated learning engine
    Implements SOLA (Self-initiated Open-world Learning and Adaptation) concepts
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.1,
        learning_enabled: bool = True
    ):
        self.adaptation_rate = adaptation_rate
        self.learning_enabled = learning_enabled
        self._learning_history: List[LearningEvent] = []
        self._adaptation_params: Dict[str, Any] = {}
        self._performance_metrics: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def record_event(
        self,
        event_type: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None
    ) -> None:
        """Record a learning event"""
        if not self.learning_enabled:
            return
        
        event = LearningEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            context=context,
            outcome=outcome
        )
        
        async with self._lock:
            self._learning_history.append(event)
            # Keep only last 1000 events
            if len(self._learning_history) > 1000:
                self._learning_history = self._learning_history[-1000:]
        
        # Trigger self-initiated learning if conditions are met
        await self._check_learning_opportunity(event)
    
    async def _check_learning_opportunity(self, event: LearningEvent) -> None:
        """Check if conditions are met for self-initiated learning"""
        # Simple heuristic: learn from failures or significant events
        if event.outcome == "failure" or event.event_type == "significant_change":
            await self._initiate_learning(event)
    
    async def _initiate_learning(self, trigger_event: LearningEvent) -> None:
        """Self-initiate learning based on event"""
        logger.info(f"Self-initiating learning from event: {trigger_event.event_type}")
        
        # Analyze recent events
        recent_events = [
            e for e in self._learning_history[-100:]
            if (datetime.utcnow() - e.timestamp).total_seconds() < 3600
        ]
        
        # Calculate adaptation
        if recent_events:
            failure_rate = sum(1 for e in recent_events if e.outcome == "failure") / len(recent_events)
            
            if failure_rate > 0.2:  # If more than 20% failures
                await self._adapt_parameters(failure_rate)
    
    async def _adapt_parameters(self, failure_rate: float) -> None:
        """Adapt agent parameters based on performance"""
        async with self._lock:
            # Adjust adaptation parameters
            current_adaptation = self._adaptation_params.get("adaptation_rate", self.adaptation_rate)
            new_adaptation = current_adaptation * (1 + failure_rate * self.adaptation_rate)
            
            self._adaptation_params["adaptation_rate"] = min(new_adaptation, 1.0)
            self._adaptation_params["last_adaptation"] = datetime.utcnow().isoformat()
            self._adaptation_params["failure_rate"] = failure_rate
            
            logger.info(f"Adapted parameters: adaptation_rate={new_adaptation:.3f}")
    
    async def get_adaptation_params(self) -> Dict[str, Any]:
        """Get current adaptation parameters"""
        async with self._lock:
            return self._adaptation_params.copy()
    
    async def update_performance_metric(self, metric_name: str, value: float) -> None:
        """Update a performance metric"""
        async with self._lock:
            self._performance_metrics[metric_name] = value
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get all performance metrics"""
        async with self._lock:
            return self._performance_metrics.copy()
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        async with self._lock:
            recent_events = [
                e for e in self._learning_history
                if (datetime.utcnow() - e.timestamp).total_seconds() < 3600
            ]
            
            return {
                "total_events": len(self._learning_history),
                "recent_events_1h": len(recent_events),
                "adaptation_params": self._adaptation_params.copy(),
                "performance_metrics": self._performance_metrics.copy(),
                "learning_enabled": self.learning_enabled
            }
