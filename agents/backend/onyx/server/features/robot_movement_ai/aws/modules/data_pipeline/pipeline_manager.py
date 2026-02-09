"""
Pipeline Manager
================

Data pipeline management.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"


@dataclass
class PipelineStage:
    """Pipeline stage definition."""
    name: str
    processor: Callable
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class PipelineManager:
    """Data pipeline manager."""
    
    def __init__(self):
        self._pipelines: Dict[str, List[PipelineStage]] = {}
        self._status: Dict[str, PipelineStatus] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}
    
    def create_pipeline(self, pipeline_id: str):
        """Create pipeline."""
        self._pipelines[pipeline_id] = []
        self._status[pipeline_id] = PipelineStatus.IDLE
        self._stats[pipeline_id] = {
            "runs": 0,
            "successes": 0,
            "failures": 0,
            "total_items": 0
        }
        logger.info(f"Created pipeline: {pipeline_id}")
    
    def add_stage(
        self,
        pipeline_id: str,
        stage_name: str,
        processor: Callable,
        config: Optional[Dict[str, Any]] = None
    ):
        """Add stage to pipeline."""
        if pipeline_id not in self._pipelines:
            self.create_pipeline(pipeline_id)
        
        stage = PipelineStage(
            name=stage_name,
            processor=processor,
            config=config or {}
        )
        
        self._pipelines[pipeline_id].append(stage)
        logger.info(f"Added stage {stage_name} to pipeline {pipeline_id}")
    
    async def process(self, pipeline_id: str, data: Any) -> Any:
        """Process data through pipeline."""
        if pipeline_id not in self._pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        self._status[pipeline_id] = PipelineStatus.RUNNING
        stages = self._pipelines[pipeline_id]
        
        try:
            result = data
            for stage in stages:
                if asyncio.iscoroutinefunction(stage.processor):
                    result = await stage.processor(result, **stage.config)
                else:
                    result = stage.processor(result, **stage.config)
            
            self._status[pipeline_id] = PipelineStatus.IDLE
            self._stats[pipeline_id]["runs"] += 1
            self._stats[pipeline_id]["successes"] += 1
            
            return result
        
        except Exception as e:
            self._status[pipeline_id] = PipelineStatus.FAILED
            self._stats[pipeline_id]["runs"] += 1
            self._stats[pipeline_id]["failures"] += 1
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            raise
    
    async def process_batch(self, pipeline_id: str, items: List[Any]) -> List[Any]:
        """Process batch of items."""
        results = []
        
        for item in items:
            try:
                result = await self.process(pipeline_id, item)
                results.append(result)
                self._stats[pipeline_id]["total_items"] += 1
            except Exception as e:
                logger.error(f"Failed to process item in {pipeline_id}: {e}")
                results.append(None)
        
        return results
    
    def get_pipeline(self, pipeline_id: str) -> Optional[List[PipelineStage]]:
        """Get pipeline stages."""
        return self._pipelines.get(pipeline_id)
    
    def get_status(self, pipeline_id: str) -> Optional[PipelineStatus]:
        """Get pipeline status."""
        return self._status.get(pipeline_id)
    
    def get_stats(self, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if pipeline_id:
            return self._stats.get(pipeline_id, {})
        return self._stats.copy()















