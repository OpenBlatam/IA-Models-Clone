"""
Real-Time Analyzer
==================

Real-time analysis and monitoring during upscaling.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class RealtimeMetrics:
    """Real-time metrics."""
    timestamp: float
    stage: str
    progress: float  # 0.0-1.0
    quality_estimate: float
    time_remaining: Optional[float] = None
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0


class RealtimeAnalyzer:
    """
    Real-time analyzer for upscaling operations.
    
    Features:
    - Progress tracking
    - Quality estimation
    - Time prediction
    - Resource monitoring
    - Stage detection
    """
    
    def __init__(
        self,
        update_interval: float = 0.1,  # Update every 100ms
        history_size: int = 100
    ):
        """
        Initialize real-time analyzer.
        
        Args:
            update_interval: Update interval in seconds
            history_size: Size of metrics history
        """
        self.update_interval = update_interval
        self.history_size = history_size
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_metrics: Optional[RealtimeMetrics] = None
        
        # Callbacks
        self.progress_callbacks: List[Callable[[RealtimeMetrics], None]] = []
        self.quality_callbacks: List[Callable[[float], None]] = []
        
        # Operation tracking
        self.operation_start: Optional[float] = None
        self.current_stage: str = "idle"
        self.stage_start: Optional[float] = None
        
        logger.info("RealtimeAnalyzer initialized")
    
    def start_operation(self) -> None:
        """Start tracking an operation."""
        self.operation_start = time.time()
        self.current_stage = "preprocessing"
        self.stage_start = time.time()
        self.metrics_history.clear()
    
    def update_stage(self, stage: str) -> None:
        """
        Update current stage.
        
        Args:
            stage: Stage name (preprocessing, upscaling, postprocessing, etc.)
        """
        if self.stage_start:
            stage_duration = time.time() - self.stage_start
            logger.debug(f"Stage '{self.current_stage}' took {stage_duration:.2f}s")
        
        self.current_stage = stage
        self.stage_start = time.time()
    
    def update_progress(
        self,
        progress: float,
        quality_estimate: Optional[float] = None,
        memory_usage_mb: float = 0.0,
        gpu_utilization: float = 0.0
    ) -> None:
        """
        Update progress.
        
        Args:
            progress: Progress (0.0-1.0)
            quality_estimate: Estimated quality (0.0-1.0)
            memory_usage_mb: Memory usage in MB
            gpu_utilization: GPU utilization (0.0-1.0)
        """
        if self.operation_start is None:
            return
        
        elapsed = time.time() - self.operation_start
        
        # Estimate time remaining
        time_remaining = None
        if progress > 0.1:  # Need some progress to estimate
            time_remaining = (elapsed / progress) - elapsed
        
        # Create metrics
        metrics = RealtimeMetrics(
            timestamp=time.time(),
            stage=self.current_stage,
            progress=progress,
            quality_estimate=quality_estimate or 0.0,
            time_remaining=time_remaining,
            memory_usage_mb=memory_usage_mb,
            gpu_utilization=gpu_utilization
        )
        
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        # Call callbacks
        for callback in self.progress_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
        
        if quality_estimate is not None:
            for callback in self.quality_callbacks:
                try:
                    callback(quality_estimate)
                except Exception as e:
                    logger.error(f"Error in quality callback: {e}")
    
    def add_progress_callback(
        self,
        callback: Callable[[RealtimeMetrics], None]
    ) -> None:
        """Add progress callback."""
        self.progress_callbacks.append(callback)
    
    def add_quality_callback(
        self,
        callback: Callable[[float], None]
    ) -> None:
        """Add quality callback."""
        self.quality_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[RealtimeMetrics]:
        """Get current metrics."""
        return self.current_metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from history."""
        if not self.metrics_history:
            return {
                "total_updates": 0,
                "avg_progress": 0.0,
                "avg_quality": 0.0,
            }
        
        return {
            "total_updates": len(self.metrics_history),
            "avg_progress": statistics.mean([m.progress for m in self.metrics_history]),
            "avg_quality": statistics.mean([m.quality_estimate for m in self.metrics_history]),
            "avg_memory_mb": statistics.mean([m.memory_usage_mb for m in self.metrics_history]),
            "avg_gpu_util": statistics.mean([m.gpu_utilization for m in self.metrics_history]),
            "current_stage": self.current_stage,
            "elapsed_time": (
                time.time() - self.operation_start
                if self.operation_start else 0.0
            ),
        }
    
    def estimate_completion_time(self) -> Optional[float]:
        """Estimate time to completion."""
        if not self.current_metrics or not self.current_metrics.time_remaining:
            return None
        
        return self.current_metrics.time_remaining
    
    def finish_operation(self) -> Dict[str, Any]:
        """Finish operation and return summary."""
        if self.operation_start is None:
            return {}
        
        total_time = time.time() - self.operation_start
        
        summary = {
            "total_time": total_time,
            "stages": {},
            "final_quality": (
                self.current_metrics.quality_estimate
                if self.current_metrics else 0.0
            ),
            "total_updates": len(self.metrics_history),
        }
        
        # Calculate stage times
        if self.stage_start:
            stage_duration = time.time() - self.stage_start
            summary["stages"][self.current_stage] = stage_duration
        
        # Reset
        self.operation_start = None
        self.current_stage = "idle"
        self.stage_start = None
        
        return summary


