"""
Progress Tracker for Upscaling
================================

Real-time progress tracking for upscaling operations.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProgressStage(Enum):
    """Progress stages."""
    INITIALIZING = "initializing"
    PREPROCESSING = "preprocessing"
    UPSCALING = "upscaling"
    POST_PROCESSING = "post_processing"
    AI_ENHANCEMENT = "ai_enhancement"
    OPTIMIZATION = "optimization"
    QUALITY_CHECK = "quality_check"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressInfo:
    """Progress information."""
    stage: ProgressStage
    progress: float  # 0.0 to 1.0
    message: str
    elapsed_time: float
    estimated_time_remaining: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """
    Track progress of upscaling operations.
    
    Features:
    - Stage tracking
    - Progress percentage
    - Time estimation
    - Callback support
    """
    
    def __init__(
        self,
        callback: Optional[Callable[[ProgressInfo], None]] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            callback: Optional callback function for progress updates
        """
        self.callback = callback
        self.start_time = None
        self.current_stage = None
        self.stage_times = {}
        self.history = []
    
    def start(self) -> None:
        """Start tracking."""
        self.start_time = time.time()
        self.current_stage = ProgressStage.INITIALIZING
        self.stage_times = {}
        self.history = []
        self._update(ProgressStage.INITIALIZING, 0.0, "Initializing...")
    
    def update(
        self,
        stage: ProgressStage,
        progress: float,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update progress.
        
        Args:
            stage: Current stage
            progress: Progress (0.0 to 1.0)
            message: Progress message
            details: Optional additional details
        """
        self.current_stage = stage
        
        # Track stage timing
        if stage not in self.stage_times:
            self.stage_times[stage] = time.time()
        
        self._update(stage, progress, message, details)
    
    def _update(
        self,
        stage: ProgressStage,
        progress: float,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Internal update method."""
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        
        # Estimate remaining time
        eta = None
        if progress > 0 and progress < 1.0:
            eta = (elapsed / progress) * (1.0 - progress)
        
        progress_info = ProgressInfo(
            stage=stage,
            progress=progress,
            message=message,
            elapsed_time=elapsed,
            estimated_time_remaining=eta,
            details=details or {}
        )
        
        self.history.append(progress_info)
        
        # Call callback if provided
        if self.callback:
            try:
                self.callback(progress_info)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")
        
        logger.debug(f"Progress: {stage.value} - {progress*100:.1f}% - {message}")
    
    def complete(self, message: str = "Completed") -> None:
        """Mark as completed."""
        self._update(ProgressStage.COMPLETED, 1.0, message)
    
    def error(self, error_message: str) -> None:
        """Mark as error."""
        self._update(ProgressStage.ERROR, 0.0, f"Error: {error_message}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        
        return {
            "current_stage": self.current_stage.value if self.current_stage else None,
            "elapsed_time": elapsed,
            "stages_completed": len(self.stage_times),
            "history_count": len(self.history),
            "stage_times": {
                stage.value: time.time() - start_time
                for stage, start_time in self.stage_times.items()
            }
        }


