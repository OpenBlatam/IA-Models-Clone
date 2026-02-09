from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import logging
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.optimized_config import settings
from onyx.server.features.ads.training_logger import TrainingLogger, TrainingPhase, AsyncTrainingLogger
from onyx.server.features.ads.pytorch_debug_utils import PyTorchDebugger, TrainingDebugger, DiffusionModelDebugger, DebugLevel
from typing import Any, List, Dict, Optional
"""
Training logs API for accessing training progress, errors, and statistics.
Provides endpoints for monitoring training progress and retrieving detailed logs.
"""


logger = setup_logger()

router = APIRouter(prefix="/training-logs", tags=["training-logs"])

class TrainingLogRequest(BaseModel):
    """Request model for training logs."""
    user_id: int
    model_name: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    phase: Optional[TrainingPhase] = None
    log_level: Optional[str] = None

class TrainingStatsResponse(BaseModel):
    """Response model for training statistics."""
    user_id: int
    model_name: str
    total_training_sessions: int
    successful_sessions: int
    failed_sessions: int
    total_training_time: float
    avg_training_time: float
    total_errors: int
    recent_metrics: Dict[str, Any]
    last_training_date: Optional[datetime] = None

class TrainingLogEntry(BaseModel):
    """Model for training log entries."""
    timestamp: datetime
    level: str
    message: str
    phase: Optional[str] = None
    user_id: Optional[int] = None
    model_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ErrorLogEntry(BaseModel):
    """Model for error log entries."""
    timestamp: datetime
    error_type: str
    error_message: str
    phase: str
    user_id: Optional[int] = None
    model_name: Optional[str] = None
    traceback: str
    context: Optional[Dict[str, Any]] = None

class TrainingProgressResponse(BaseModel):
    """Response model for training progress."""
    user_id: int
    model_name: str
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    progress_percentage: float
    elapsed_time: float
    estimated_completion: Optional[datetime] = None
    status: str
    recent_loss: Optional[float] = None
    recent_accuracy: Optional[float] = None
    recent_learning_rate: Optional[float] = None

class TrainingMetricsResponse(BaseModel):
    """Response model for training metrics."""
    user_id: int
    model_name: str
    metrics: List[Dict[str, Any]]
    summary: Dict[str, Any]
    plots: Optional[Dict[str, str]] = None

class LogFilter(BaseModel):
    """Filter for log queries."""
    user_id: Optional[int] = None
    model_name: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    phase: Optional[TrainingPhase] = None
    log_level: Optional[str] = None
    limit: int = 100
    offset: int = 0

class TrainingLogsService:
    """Service for managing training logs and statistics."""
    
    def __init__(self) -> Any:
        """Initialize the training logs service."""
        self._redis_client = None
        self.log_dirs = {
            "training": Path("logs/training"),
            "diffusion": Path("logs/diffusion"),
            "tokenization": Path("logs/tokenization")
        }
        
        # Create log directories
        for log_dir in self.log_dirs.values():
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize debuggers
        self.training_debugger = TrainingDebugger(DebugLevel.BASIC)
        self.diffusion_debugger = DiffusionModelDebugger(DebugLevel.BASIC)
        self.general_debugger = PyTorchDebugger(DebugLevel.BASIC)
    
    @property
    async def redis_client(self) -> Any:
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client
    
    async def get_training_stats(self, user_id: int, model_name: Optional[str] = None) -> TrainingStatsResponse:
        """Get comprehensive training statistics for a user."""
        try:
            redis = await self.redis_client
            
            # Get all training sessions for user
            pattern = f"training_logs:stats:{user_id}:*"
            if model_name:
                pattern = f"training_logs:stats:{user_id}:{model_name}"
            
            keys = await redis.keys(pattern)
            
            total_sessions = 0
            successful_sessions = 0
            failed_sessions = 0
            total_training_time = 0.0
            total_errors = 0
            last_training_date = None
            recent_metrics = {}
            
            for key in keys:
                try:
                    stats_data = await redis.get(key)
                    if stats_data:
                        stats = json.loads(stats_data)
                        total_sessions += 1
                        
                        if stats.get("status") == "completed":
                            successful_sessions += 1
                        elif stats.get("status") == "failed":
                            failed_sessions += 1
                        
                        total_training_time += stats.get("total_time", 0)
                        total_errors += stats.get("errors_count", 0)
                        
                        # Track last training date
                        if "last_training_date" in stats:
                            training_date = datetime.fromisoformat(stats["last_training_date"])
                            if not last_training_date or training_date > last_training_date:
                                last_training_date = training_date
                        
                        # Collect recent metrics
                        if "recent_metrics" in stats:
                            recent_metrics.update(stats["recent_metrics"])
                
                except Exception as e:
                    logger.warning(f"Error parsing stats from key {key}: {e}")
                    continue
            
            avg_training_time = total_training_time / total_sessions if total_sessions > 0 else 0
            
            return TrainingStatsResponse(
                user_id=user_id,
                model_name=model_name or "all",
                total_training_sessions=total_sessions,
                successful_sessions=successful_sessions,
                failed_sessions=failed_sessions,
                total_training_time=total_training_time,
                avg_training_time=avg_training_time,
                total_errors=total_errors,
                recent_metrics=recent_metrics,
                last_training_date=last_training_date
            )
        
        except Exception as e:
            logger.error(f"Error getting training stats: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving training statistics: {str(e)}")
    
    async def get_training_progress(self, user_id: int, model_name: str) -> TrainingProgressResponse:
        """Get current training progress for a user."""
        try:
            redis = await self.redis_client
            
            # Get current training progress
            progress_key = f"training_logs:progress:{user_id}:{model_name}"
            progress_data = await redis.get(progress_key)
            
            if not progress_data:
                raise HTTPException(status_code=404, detail="No active training session found")
            
            progress = json.loads(progress_data)
            
            return TrainingProgressResponse(
                user_id=user_id,
                model_name=model_name,
                current_epoch=progress.get("current_epoch", 0),
                total_epochs=progress.get("total_epochs", 0),
                current_step=progress.get("current_step", 0),
                total_steps=progress.get("total_steps", 0),
                progress_percentage=progress.get("progress_percentage", 0),
                elapsed_time=progress.get("elapsed_time", 0),
                estimated_completion=datetime.fromisoformat(progress["estimated_completion"]) if progress.get("estimated_completion") else None,
                status=progress.get("status", "unknown"),
                recent_loss=progress.get("recent_loss"),
                recent_accuracy=progress.get("recent_accuracy"),
                recent_learning_rate=progress.get("recent_learning_rate")
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting training progress: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving training progress: {str(e)}")
    
    async def get_training_metrics(self, user_id: int, model_name: str, limit: int = 100) -> TrainingMetricsResponse:
        """Get training metrics for a user."""
        try:
            redis = await self.redis_client
            
            # Get metrics from Redis
            metrics_key = f"training_logs:metrics:{user_id}:{model_name}"
            metrics_data = await redis.get(metrics_key)
            
            if not metrics_data:
                return TrainingMetricsResponse(
                    user_id=user_id,
                    model_name=model_name,
                    metrics=[],
                    summary={}
                )
            
            metrics = json.loads(metrics_data)
            
            # Limit metrics
            recent_metrics = metrics[-limit:] if len(metrics) > limit else metrics
            
            # Calculate summary
            if recent_metrics:
                losses = [m.get("loss", 0) for m in recent_metrics if m.get("loss")]
                accuracies = [m.get("accuracy", 0) for m in recent_metrics if m.get("accuracy")]
                learning_rates = [m.get("learning_rate", 0) for m in recent_metrics if m.get("learning_rate")]
                
                summary = {
                    "total_metrics": len(recent_metrics),
                    "avg_loss": sum(losses) / len(losses) if losses else 0,
                    "min_loss": min(losses) if losses else 0,
                    "max_loss": max(losses) if losses else 0,
                    "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                    "avg_learning_rate": sum(learning_rates) / len(learning_rates) if learning_rates else 0,
                    "last_metric_time": recent_metrics[-1].get("timestamp") if recent_metrics else None
                }
            else:
                summary = {}
            
            return TrainingMetricsResponse(
                user_id=user_id,
                model_name=model_name,
                metrics=recent_metrics,
                summary=summary
            )
        
        except Exception as e:
            logger.error(f"Error getting training metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving training metrics: {str(e)}")
    
    async def get_error_logs(self, user_id: int, model_name: Optional[str] = None, limit: int = 50) -> List[ErrorLogEntry]:
        """Get error logs for a user."""
        try:
            redis = await self.redis_client
            
            # Get error logs from Redis
            pattern = f"training_logs:errors:{user_id}:*"
            if model_name:
                pattern = f"training_logs:errors:{user_id}:{model_name}"
            
            keys = await redis.keys(pattern)
            
            error_logs = []
            for key in keys:
                try:
                    errors_data = await redis.get(key)
                    if errors_data:
                        errors = json.loads(errors_data)
                        for error in errors[-limit:]:  # Get recent errors
                            error_logs.append(ErrorLogEntry(
                                timestamp=datetime.fromisoformat(error["timestamp"]),
                                error_type=error["error_type"],
                                error_message=error["error_message"],
                                phase=error["phase"],
                                user_id=error.get("user_id"),
                                model_name=error.get("model_name"),
                                traceback=error["traceback"],
                                context=error.get("context")
                            ))
                except Exception as e:
                    logger.warning(f"Error parsing error logs from key {key}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            error_logs.sort(key=lambda x: x.timestamp, reverse=True)
            
            return error_logs[:limit]
        
        except Exception as e:
            logger.error(f"Error getting error logs: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving error logs: {str(e)}")
    
    async def get_file_logs(self, user_id: int, model_name: Optional[str] = None, 
                           start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None,
                           limit: int = 100) -> List[TrainingLogEntry]:
        """Get logs from log files."""
        try:
            log_entries = []
            
            # Search in all log directories
            for log_type, log_dir in self.log_dirs.items():
                if not log_dir.exists():
                    continue
                
                # Find log files for user
                pattern = f"*user_{user_id}*"
                if model_name:
                    pattern = f"*user_{user_id}*{model_name}*"
                
                log_files = list(log_dir.glob(pattern))
                
                for log_file in log_files:
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            for line in f:
                                try:
                                    # Parse log line
                                    # Expected format: timestamp - level - message
                                    parts = line.strip().split(' - ', 2)
                                    if len(parts) >= 3:
                                        timestamp_str, level, message = parts
                                        timestamp = datetime.fromisoformat(timestamp_str.replace(',', '.'))
                                        
                                        # Apply date filters
                                        if start_date and timestamp < start_date:
                                            continue
                                        if end_date and timestamp > end_date:
                                            continue
                                        
                                        # Extract phase from message if present
                                        phase = None
                                        if "[TRAINING]" in message:
                                            phase = "training"
                                        elif "[DATA_PREPARATION]" in message:
                                            phase = "data_preparation"
                                        elif "[MODEL_LOADING]" in message:
                                            phase = "model_loading"
                                        elif "[INFERENCE]" in message:
                                            phase = "inference"
                                        
                                        log_entries.append(TrainingLogEntry(
                                            timestamp=timestamp,
                                            level=level,
                                            message=message,
                                            phase=phase,
                                            user_id=user_id,
                                            model_name=model_name
                                        ))
                                        
                                        if len(log_entries) >= limit:
                                            break
                                
                                except Exception as e:
                                    logger.warning(f"Error parsing log line: {e}")
                                    continue
                        
                        if len(log_entries) >= limit:
                            break
                    
                    except Exception as e:
                        logger.warning(f"Error reading log file {log_file}: {e}")
                        continue
            
            # Sort by timestamp (newest first)
            log_entries.sort(key=lambda x: x.timestamp, reverse=True)
            
            return log_entries[:limit]
        
        except Exception as e:
            logger.error(f"Error getting file logs: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving file logs: {str(e)}")
    
    async def cleanup_old_logs(self, days: int = 7) -> Dict[str, int]:
        """Clean up old log files and Redis entries."""
        try:
            deleted_files = 0
            deleted_redis_keys = 0
            
            # Clean up log files
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for log_type, log_dir in self.log_dirs.items():
                if not log_dir.exists():
                    continue
                
                for log_file in log_dir.glob("*.log"):
                    try:
                        if log_file.stat().st_mtime < cutoff_date.timestamp():
                            log_file.unlink()
                            deleted_files += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete log file {log_file}: {e}")
            
            # Clean up Redis entries
            redis = await self.redis_client
            cutoff_timestamp = cutoff_date.timestamp()
            
            # Get all training log keys
            pattern = "training_logs:*"
            keys = await redis.keys(pattern)
            
            for key in keys:
                try:
                    # Check if key has timestamp data
                    data = await redis.get(key)
                    if data:
                        try:
                            parsed_data = json.loads(data)
                            if "timestamp" in parsed_data:
                                key_timestamp = datetime.fromisoformat(parsed_data["timestamp"]).timestamp()
                                if key_timestamp < cutoff_timestamp:
                                    await redis.delete(key)
                                    deleted_redis_keys += 1
                        except (json.JSONDecodeError, KeyError, ValueError):
                            # If we can't parse the data, skip it
                            continue
                except Exception as e:
                    logger.warning(f"Failed to process Redis key {key}: {e}")
            
            return {
                "deleted_files": deleted_files,
                "deleted_redis_keys": deleted_redis_keys
            }
        
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
            raise HTTPException(status_code=500, detail=f"Error cleaning up old logs: {str(e)}")

# Initialize service
training_logs_service = TrainingLogsService()

@router.get("/stats/{user_id}", response_model=TrainingStatsResponse)
async def get_training_stats(
    user_id: int,
    model_name: Optional[str] = Query(None, description="Filter by model name")
):
    """Get training statistics for a user."""
    return await training_logs_service.get_training_stats(user_id, model_name)

@router.get("/progress/{user_id}/{model_name}", response_model=TrainingProgressResponse)
async def get_training_progress(user_id: int, model_name: str):
    """Get current training progress for a user."""
    return await training_logs_service.get_training_progress(user_id, model_name)

@router.get("/metrics/{user_id}/{model_name}", response_model=TrainingMetricsResponse)
async def get_training_metrics(
    user_id: int,
    model_name: str,
    limit: int = Query(100, description="Number of metrics to return")
):
    """Get training metrics for a user."""
    return await training_logs_service.get_training_metrics(user_id, model_name, limit)

@router.get("/errors/{user_id}", response_model=List[ErrorLogEntry])
async def get_error_logs(
    user_id: int,
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    limit: int = Query(50, description="Number of errors to return")
):
    """Get error logs for a user."""
    return await training_logs_service.get_error_logs(user_id, model_name, limit)

@router.get("/logs/{user_id}", response_model=List[TrainingLogEntry])
async def get_training_logs(
    user_id: int,
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    limit: int = Query(100, description="Number of log entries to return")
):
    """Get training logs for a user."""
    return await training_logs_service.get_file_logs(
        user_id, model_name, start_date, end_date, limit
    )

@router.post("/logs/filter", response_model=List[TrainingLogEntry])
async def filter_training_logs(filter_params: LogFilter):
    """Filter training logs with advanced parameters."""
    return await training_logs_service.get_file_logs(
        filter_params.user_id,
        filter_params.model_name,
        filter_params.start_date,
        filter_params.end_date,
        filter_params.limit
    )

@router.delete("/cleanup")
async def cleanup_old_logs(days: int = Query(7, description="Number of days to keep logs")):
    """Clean up old log files and Redis entries."""
    result = await training_logs_service.cleanup_old_logs(days)
    return {
        "message": f"Cleaned up {result['deleted_files']} log files and {result['deleted_redis_keys']} Redis keys",
        "deleted_files": result["deleted_files"],
        "deleted_redis_keys": result["deleted_redis_keys"]
    }

@router.get("/health")
async def health_check():
    """Health check for training logs service."""
    try:
        redis = await training_logs_service.redis_client
        await redis.ping()
        
        return {
            "status": "healthy",
            "redis_connected": True,
            "log_directories": {
                name: str(path) for name, path in training_logs_service.log_dirs.items()
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@router.get("/debug/summary")
async def get_debug_summary():
    """Get debug summary from all debuggers."""
    try:
        return {
            "training_debugger": training_logs_service.training_debugger.get_debug_summary(),
            "diffusion_debugger": training_logs_service.diffusion_debugger.get_debug_summary(),
            "general_debugger": training_logs_service.general_debugger.get_debug_summary()
        }
    except Exception as e:
        logger.error(f"Error getting debug summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving debug summary: {str(e)}")

@router.get("/debug/training/summary")
async def get_training_debug_summary():
    """Get training-specific debug summary."""
    try:
        return training_logs_service.training_debugger.get_training_summary()
    except Exception as e:
        logger.error(f"Error getting training debug summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving training debug summary: {str(e)}")

@router.post("/debug/clear")
async def clear_debug_history():
    """Clear debug history from all debuggers."""
    try:
        training_logs_service.training_debugger.clear_debug_history()
        training_logs_service.diffusion_debugger.clear_debug_history()
        training_logs_service.general_debugger.clear_debug_history()
        
        return {"message": "Debug history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing debug history: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing debug history: {str(e)}")

@router.post("/debug/level/{debug_level}")
async def set_debug_level(debug_level: str):
    """Set debug level for all debuggers."""
    try:
        level = DebugLevel(debug_level.lower())
        
        # Update debug levels
        training_logs_service.training_debugger.debug_level = level
        training_logs_service.diffusion_debugger.debug_level = level
        training_logs_service.general_debugger.debug_level = level
        
        return {"message": f"Debug level set to {debug_level}"}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid debug level: {debug_level}")
    except Exception as e:
        logger.error(f"Error setting debug level: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting debug level: {str(e)}") 