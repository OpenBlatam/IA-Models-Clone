"""
Chaos Engineer
==============

Chaos engineering utilities.
"""

import logging
import asyncio
import random
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ChaosType(Enum):
    """Chaos experiment types."""
    LATENCY = "latency"
    ERROR = "error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    SERVICE_DOWN = "service_down"


@dataclass
class ChaosExperiment:
    """Chaos experiment."""
    id: str
    name: str
    chaos_type: ChaosType
    target: str
    probability: float = 0.1
    duration: float = 60.0
    enabled: bool = True
    handler: Optional[Callable] = None


class ChaosEngineer:
    """Chaos engineering manager."""
    
    def __init__(self):
        self._experiments: Dict[str, ChaosExperiment] = {}
        self._active_experiments: Dict[str, asyncio.Task] = {}
    
    def register_experiment(
        self,
        experiment_id: str,
        name: str,
        chaos_type: ChaosType,
        target: str,
        probability: float = 0.1,
        duration: float = 60.0,
        handler: Optional[Callable] = None
    ) -> ChaosExperiment:
        """Register chaos experiment."""
        experiment = ChaosExperiment(
            id=experiment_id,
            name=name,
            chaos_type=chaos_type,
            target=target,
            probability=probability,
            duration=duration,
            handler=handler
        )
        
        self._experiments[experiment_id] = experiment
        logger.info(f"Registered chaos experiment: {experiment_id}")
        return experiment
    
    def start_experiment(self, experiment_id: str):
        """Start chaos experiment."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if experiment_id in self._active_experiments:
            logger.warning(f"Experiment {experiment_id} already running")
            return
        
        experiment = self._experiments[experiment_id]
        
        async def run_experiment():
            end_time = asyncio.get_event_loop().time() + experiment.duration
            
            while asyncio.get_event_loop().time() < end_time:
                if random.random() < experiment.probability:
                    await self._apply_chaos(experiment)
                
                await asyncio.sleep(1)
            
            del self._active_experiments[experiment_id]
            logger.info(f"Experiment {experiment_id} completed")
        
        task = asyncio.create_task(run_experiment())
        self._active_experiments[experiment_id] = task
        logger.info(f"Started chaos experiment: {experiment_id}")
    
    async def _apply_chaos(self, experiment: ChaosExperiment):
        """Apply chaos effect."""
        if experiment.handler:
            try:
                if asyncio.iscoroutinefunction(experiment.handler):
                    await experiment.handler()
                else:
                    experiment.handler()
            except Exception as e:
                logger.error(f"Chaos handler failed: {e}")
        else:
            # Default chaos effects
            if experiment.chaos_type == ChaosType.LATENCY:
                await asyncio.sleep(random.uniform(0.1, 1.0))
            elif experiment.chaos_type == ChaosType.ERROR:
                raise Exception(f"Chaos error injected for {experiment.target}")
    
    def stop_experiment(self, experiment_id: str):
        """Stop chaos experiment."""
        if experiment_id in self._active_experiments:
            self._active_experiments[experiment_id].cancel()
            del self._active_experiments[experiment_id]
            logger.info(f"Stopped chaos experiment: {experiment_id}")
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get experiment statistics."""
        return {
            "total_experiments": len(self._experiments),
            "active_experiments": len(self._active_experiments),
            "by_type": {
                chaos_type.value: sum(
                    1 for e in self._experiments.values()
                    if e.chaos_type == chaos_type
                )
                for chaos_type in ChaosType
            }
        }















