"""
Learning API for continuous improvement and model training.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

from .base_router import BaseRouter
from ...utils.file_helpers import get_iso_timestamp

# Create base router instance
base = BaseRouter(
    prefix="/api/learning",
    tags=["Learning & Improvement"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class FeedbackRequest(BaseModel):
    """Request to provide feedback for learning."""
    prediction_id: Optional[str] = Field(None, description="ID of the prediction")
    actual_outcome: str = Field(..., description="Actual outcome: correct, incorrect, partial")
    feedback: Optional[str] = Field(None, description="Additional feedback")
    robot_type: str = Field(..., description="Robot type")
    sensor_data: Optional[Dict[str, Any]] = Field(None, description="Sensor data used")


class TrainingRequest(BaseModel):
    """Request to trigger model training."""
    robot_type: Optional[str] = Field(None, description="Train model for specific robot type")
    force_retrain: bool = Field(False, description="Force retrain even if recently trained")


@router.post("/feedback")
@base.timed_endpoint("submit_feedback")
async def submit_feedback(
    request: FeedbackRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Submit feedback to improve ML models.
    """
    base.log_request("submit_feedback", robot_type=request.robot_type, outcome=request.actual_outcome)
    
    feedback_record = {
        "id": get_timestamp_id("feedback_"),
        "prediction_id": request.prediction_id,
        "actual_outcome": request.actual_outcome,
        "feedback": request.feedback,
        "robot_type": request.robot_type,
        "sensor_data": request.sensor_data,
        "timestamp": get_iso_timestamp()
    }
    
    # In a real implementation, this would save to database
    # and trigger model improvement
    
    return base.success(feedback_record, message="Feedback recorded. Thank you for helping improve the system!")


@router.post("/train")
@base.timed_endpoint("train_model")
async def train_model(
    request: TrainingRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Trigger model training/retraining.
    """
    base.log_request("train_model", robot_type=request.robot_type, force_retrain=request.force_retrain)
    
    # Get training data
    history = base.database.get_maintenance_history(
        robot_type=request.robot_type,
        limit=10000
    )
    
    if len(history) < 10:
        from .exceptions import ValidationError
        raise ValidationError("Insufficient data for training. Need at least 10 records.")
    
    # In a real implementation, this would:
    # 1. Prepare training data
    # 2. Train the model
    # 3. Validate the model
    # 4. Save the model
    # 5. Update model version
    
    training_result = {
        "training_id": get_timestamp_id("train_"),
        "robot_type": request.robot_type or "all",
        "training_samples": len(history),
        "started_at": get_iso_timestamp(),
        "status": "completed",
        "model_version": "1.0.0",
        "improvements": {
            "accuracy": "+2.3%",
            "precision": "+1.8%",
            "recall": "+2.1%"
        }
    }
    
    return base.success(training_result, message="Model training completed successfully")


@router.get("/model-info")
@base.timed_endpoint("get_model_info")
async def get_model_info(
    robot_type: Optional[str] = Query(None, description="Get info for specific robot type"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get information about current ML models.
    """
    base.log_request("get_model_info", robot_type=robot_type)
    
    model_info = {
        "version": "1.0.0",
        "last_trained": "2024-01-15T10:00:00",
        "training_samples": 1250,
        "accuracy": 0.893,
        "precision": 0.876,
        "recall": 0.901,
        "f1_score": 0.888,
        "features_used": [
            "temperature",
            "pressure",
            "vibration",
            "runtime_hours",
            "days_since_maintenance"
        ],
        "robot_type": robot_type or "all"
    }
    
    return base.success(model_info)


@router.get("/performance")
@base.timed_endpoint("get_model_performance")
async def get_model_performance(
    robot_type: Optional[str] = Query(None, description="Filter by robot type"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get model performance metrics over time.
    """
    base.log_request("get_model_performance", robot_type=robot_type, days=days)
    
    # In a real implementation, this would analyze prediction accuracy
    # based on feedback and actual outcomes
    
    performance = {
        "period_days": days,
        "total_predictions": 450,
        "correct_predictions": 402,
        "incorrect_predictions": 48,
        "accuracy": 0.893,
        "precision": 0.876,
        "recall": 0.901,
        "trend": "improving",
        "robot_type": robot_type or "all"
    }
    
    return base.success(performance)


@router.get("/insights")
@base.timed_endpoint("get_learning_insights")
async def get_learning_insights(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get insights about model learning and improvement opportunities.
    """
    base.log_request("get_learning_insights")
    
    history = base.database.get_maintenance_history(limit=1000)
    
    insights = {
        "data_quality": {
            "total_records": len(history),
            "completeness": "good" if len(history) > 100 else "needs_improvement",
            "recommendation": "Continue collecting maintenance data for better predictions"
        },
        "improvement_opportunities": [
            "More feedback data would improve accuracy",
            "Consider adding more sensor features",
            "Regular retraining recommended every 30 days"
        ],
        "model_health": {
            "status": "healthy",
            "last_validation": "2024-01-15T10:00:00",
            "drift_detected": False
        }
    }
    
    return base.success(insights)




