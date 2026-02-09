"""
Pydantic schemas for API request/response models.
Centralized location for all API data models.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

from ..config.maintenance_config import MaintenanceConfig
from ..utils.validators import (
    validate_sensor_data_strict,
    validate_robot_type,
    validate_maintenance_type,
    validate_difficulty_level
)


class MaintenanceQuestionRequest(BaseModel):
    """Request model for asking maintenance questions."""
    question: str = Field(..., description="The maintenance question", min_length=10, max_length=2000)
    robot_type: Optional[str] = Field(None, description="Type of robot/machine")
    maintenance_type: Optional[str] = Field(None, description="Maintenance category")
    difficulty: Optional[str] = Field(None, description="Difficulty level")
    sensor_data: Optional[Dict[str, Any]] = Field(None, description="Sensor data for ML analysis")
    context: Optional[str] = Field(None, description="Additional context")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    
    @field_validator('sensor_data')
    @classmethod
    def validate_sensor_data(cls, v):
        if v is not None:
            is_valid, error_msg = validate_sensor_data_strict(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v
    
    @field_validator('robot_type')
    @classmethod
    def validate_robot_type(cls, v):
        if v is not None:
            config = MaintenanceConfig()
            if not validate_robot_type(v, config.robot_types):
                raise ValueError(f"Invalid robot_type. Allowed: {', '.join(config.robot_types)}")
        return v
    
    @field_validator('maintenance_type')
    @classmethod
    def validate_maintenance_type(cls, v):
        if v is not None:
            config = MaintenanceConfig()
            if not validate_maintenance_type(v, config.maintenance_categories):
                raise ValueError(f"Invalid maintenance_type. Allowed: {', '.join(config.maintenance_categories)}")
        return v
    
    @field_validator('difficulty')
    @classmethod
    def validate_difficulty(cls, v):
        if v is not None:
            config = MaintenanceConfig()
            if not validate_difficulty_level(v, config.difficulty_levels):
                raise ValueError(f"Invalid difficulty. Allowed: {', '.join(config.difficulty_levels)}")
        return v


class ProcedureRequest(BaseModel):
    """Request model for explaining procedures."""
    procedure: str = Field(..., description="Procedure to explain", min_length=3, max_length=500)
    robot_type: str = Field(..., description="Type of robot")
    difficulty: str = Field("intermedio", description="Difficulty level")
    
    @field_validator('robot_type')
    @classmethod
    def validate_robot_type(cls, v):
        config = MaintenanceConfig()
        if not validate_robot_type(v, config.robot_types):
            raise ValueError(f"Invalid robot_type. Allowed: {', '.join(config.robot_types)}")
        return v
    
    @field_validator('difficulty')
    @classmethod
    def validate_difficulty(cls, v):
        config = MaintenanceConfig()
        if not validate_difficulty_level(v, config.difficulty_levels):
            raise ValueError(f"Invalid difficulty. Allowed: {', '.join(config.difficulty_levels)}")
        return v


class DiagnosisRequest(BaseModel):
    """Request model for diagnosis."""
    symptoms: str = Field(..., description="Description of symptoms", min_length=10, max_length=2000)
    robot_type: str = Field(..., description="Type of robot")
    sensor_data: Optional[Dict[str, float]] = Field(None, description="Sensor readings")
    
    @field_validator('robot_type')
    @classmethod
    def validate_robot_type(cls, v):
        config = MaintenanceConfig()
        if not validate_robot_type(v, config.robot_types):
            raise ValueError(f"Invalid robot_type. Allowed: {', '.join(config.robot_types)}")
        return v
    
    @field_validator('sensor_data')
    @classmethod
    def validate_sensor_data(cls, v):
        if v is not None:
            is_valid, error_msg = validate_sensor_data_strict(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v


class PredictionRequest(BaseModel):
    """Request model for maintenance prediction."""
    robot_type: str = Field(..., description="Type of robot")
    sensor_data: Dict[str, float] = Field(..., description="Current sensor readings")
    historical_data: Optional[List[Dict[str, Any]]] = Field(None, description="Historical data")
    
    @field_validator('sensor_data')
    @classmethod
    def validate_sensor_data(cls, v):
        is_valid, error_msg = validate_sensor_data_strict(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v
    
    @field_validator('robot_type')
    @classmethod
    def validate_robot_type(cls, v):
        config = MaintenanceConfig()
        if not validate_robot_type(v, config.robot_types):
            raise ValueError(f"Invalid robot_type. Allowed: {', '.join(config.robot_types)}")
        return v


class ChecklistRequest(BaseModel):
    """Request model for generating checklists."""
    robot_type: str = Field(..., description="Type of robot")
    maintenance_type: str = Field("preventivo", description="Type of maintenance")
    
    @field_validator('robot_type')
    @classmethod
    def validate_robot_type(cls, v):
        config = MaintenanceConfig()
        if not validate_robot_type(v, config.robot_types):
            raise ValueError(f"Invalid robot_type. Allowed: {', '.join(config.robot_types)}")
        return v
    
    @field_validator('maintenance_type')
    @classmethod
    def validate_maintenance_type(cls, v):
        config = MaintenanceConfig()
        if not validate_maintenance_type(v, config.maintenance_categories):
            raise ValueError(f"Invalid maintenance_type. Allowed: {', '.join(config.maintenance_categories)}")
        return v


class ScheduleRequest(BaseModel):
    """Request model for generating maintenance schedules."""
    robot_type: str = Field(..., description="Type of robot/machine")
    usage_hours: int = Field(..., description="Hours of operation")
    operating_conditions: Optional[str] = Field(None, description="Operating conditions")






