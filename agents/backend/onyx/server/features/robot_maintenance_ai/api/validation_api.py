"""
Advanced Validation API for data validation and quality checks.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import re

from .base_router import BaseRouter
from ..utils.data_helpers import count_matching

# Create base router instance
base = BaseRouter(
    prefix="/api/validation",
    tags=["Validation"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class ValidationRequest(BaseModel):
    """Request for data validation."""
    data: Dict[str, Any] = Field(..., description="Data to validate")
    validation_rules: Optional[Dict[str, Any]] = Field(None, description="Custom validation rules")
    data_type: str = Field("sensor_data", description="Type of data: sensor_data, maintenance_record, conversation")


class ValidationRule(BaseModel):
    """Validation rule definition."""
    field: str = Field(..., description="Field to validate")
    rule_type: str = Field(..., description="Type of rule: required, type, range, pattern, custom")
    value: Optional[Any] = Field(None, description="Rule value/parameters")
    message: Optional[str] = Field(None, description="Custom error message")


@router.post("/validate")
@base.timed_endpoint("validate_data")
async def validate_data(
    request: ValidationRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Validate data against rules.
    """
    base.log_request("validate_data", data_type=request.data_type)
    
    errors = []
    warnings = []
    
    data = request.data
    
    # Default validation rules based on data type
    if request.data_type == "sensor_data":
        rules = {
            "temperature": {"type": "float", "range": [-50, 200]},
            "pressure": {"type": "float", "min": 0},
            "vibration": {"type": "float", "min": 0},
            "runtime_hours": {"type": "float", "min": 0}
        }
    elif request.data_type == "maintenance_record":
        rules = {
            "robot_type": {"type": "string", "required": True},
            "maintenance_type": {"type": "string", "required": True},
            "description": {"type": "string", "min_length": 10}
        }
    else:
        rules = {}
    
    # Merge with custom rules
    if request.validation_rules:
        rules.update(request.validation_rules)
    
    # Validate each field
    for field, rule in rules.items():
        value = data.get(field)
        
        # Required check
        if rule.get("required") and value is None:
            errors.append({
                "field": field,
                "rule": "required",
                "message": f"{field} is required"
            })
            continue
        
        if value is None:
            continue
        
        # Type check
        if "type" in rule:
            expected_type = rule["type"]
            if expected_type == "float" and not isinstance(value, (int, float)):
                errors.append({
                    "field": field,
                    "rule": "type",
                    "message": f"{field} must be a number"
                })
            elif expected_type == "string" and not isinstance(value, str):
                errors.append({
                    "field": field,
                    "rule": "type",
                    "message": f"{field} must be a string"
                })
        
        # Range check
        if "range" in rule and isinstance(value, (int, float)):
            min_val, max_val = rule["range"]
            if not (min_val <= value <= max_val):
                errors.append({
                    "field": field,
                    "rule": "range",
                    "message": f"{field} must be between {min_val} and {max_val}"
                })
        
        # Min/Max check
        if "min" in rule and isinstance(value, (int, float)):
            if value < rule["min"]:
                errors.append({
                    "field": field,
                    "rule": "min",
                    "message": f"{field} must be at least {rule['min']}"
                })
        
        if "max" in rule and isinstance(value, (int, float)):
            if value > rule["max"]:
                errors.append({
                    "field": field,
                    "rule": "max",
                    "message": f"{field} must be at most {rule['max']}"
                })
        
        # Pattern check
        if "pattern" in rule and isinstance(value, str):
            if not re.match(rule["pattern"], value):
                errors.append({
                    "field": field,
                    "rule": "pattern",
                    "message": f"{field} does not match required pattern"
                })
        
        # Length check
        if "min_length" in rule and isinstance(value, str):
            if len(value) < rule["min_length"]:
                errors.append({
                    "field": field,
                    "rule": "min_length",
                    "message": f"{field} must be at least {rule['min_length']} characters"
                })
    
    return base.success({
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "validated_fields": len(rules),
        "data_type": request.data_type
    })


@router.post("/batch")
@base.timed_endpoint("validate_batch")
async def validate_batch(
    items: List[Dict[str, Any]] = Field(..., description="List of items to validate"),
    data_type: str = Field("sensor_data", description="Type of data"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Validate multiple items in batch.
    """
    base.log_request("validate_batch", items_count=len(items), data_type=data_type)
    
    results = []
    total_errors = 0
    
    for i, item in enumerate(items):
        request = ValidationRequest(
            data=item,
            data_type=data_type
        )
        # Reuse validation logic
        validation_result = await validate_data(request, _)
        
        results.append({
            "index": i,
            "valid": validation_result["data"]["valid"],
            "errors": validation_result["data"]["errors"],
            "errors_count": len(validation_result["data"]["errors"])
        })
        
        if not validation_result["data"]["valid"]:
            total_errors += len(validation_result["data"]["errors"])
    
    valid_count = count_matching(results, lambda r: r["valid"])
    
    return base.success({
        "total_items": len(items),
        "valid_items": valid_count,
        "invalid_items": len(items) - valid_count,
        "total_errors": total_errors,
        "results": results
    })


@router.get("/rules")
@base.timed_endpoint("get_validation_rules")
async def get_validation_rules(
    data_type: str = Query("sensor_data", description="Type of data"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get available validation rules for a data type.
    """
    base.log_request("get_validation_rules", data_type=data_type)
    
    rules_map = {
        "sensor_data": {
            "temperature": {
                "type": "float",
                "range": [-50, 200],
                "description": "Temperature in Celsius"
            },
            "pressure": {
                "type": "float",
                "min": 0,
                "description": "Pressure in PSI"
            },
            "vibration": {
                "type": "float",
                "min": 0,
                "description": "Vibration level"
            },
            "runtime_hours": {
                "type": "float",
                "min": 0,
                "description": "Runtime in hours"
            }
        },
        "maintenance_record": {
            "robot_type": {
                "type": "string",
                "required": True,
                "description": "Type of robot"
            },
            "maintenance_type": {
                "type": "string",
                "required": True,
                "description": "Type of maintenance"
            },
            "description": {
                "type": "string",
                "min_length": 10,
                "description": "Maintenance description"
            }
        }
    }
    
    rules = rules_map.get(data_type, {})
    
    return base.success({
        "data_type": data_type,
        "rules": rules,
        "total_rules": len(rules)
    })




