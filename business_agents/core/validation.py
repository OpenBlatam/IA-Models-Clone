"""
Validation Layer
================

Input validation and sanitization utilities.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, validator, Field
import re
import logging

logger = logging.getLogger(__name__)

class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
    
    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_agent_id(agent_id: str) -> ValidationResult:
        """Validate agent ID format."""
        result = ValidationResult(True)
        
        if not agent_id:
            result.add_error("Agent ID cannot be empty")
        elif not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
            result.add_error("Agent ID must contain only alphanumeric characters, underscores, and hyphens")
        elif len(agent_id) > 100:
            result.add_error("Agent ID must be 100 characters or less")
        
        return result
    
    @staticmethod
    def validate_workflow_id(workflow_id: str) -> ValidationResult:
        """Validate workflow ID format."""
        result = ValidationResult(True)
        
        if not workflow_id:
            result.add_error("Workflow ID cannot be empty")
        elif not re.match(r'^[a-zA-Z0-9_-]+$', workflow_id):
            result.add_error("Workflow ID must contain only alphanumeric characters, underscores, and hyphens")
        elif len(workflow_id) > 100:
            result.add_error("Workflow ID must be 100 characters or less")
        
        return result
    
    @staticmethod
    def validate_document_id(document_id: str) -> ValidationResult:
        """Validate document ID format."""
        result = ValidationResult(True)
        
        if not document_id:
            result.add_error("Document ID cannot be empty")
        elif not re.match(r'^[a-zA-Z0-9_-]+$', document_id):
            result.add_error("Document ID must contain only alphanumeric characters, underscores, and hyphens")
        elif len(document_id) > 100:
            result.add_error("Document ID must be 100 characters or less")
        
        return result
    
    @staticmethod
    def validate_business_area(business_area: str) -> ValidationResult:
        """Validate business area."""
        result = ValidationResult(True)
        
        valid_areas = [
            "marketing", "sales", "operations", "hr", "finance",
            "legal", "technical", "content", "customer_service",
            "product_development", "strategy", "compliance"
        ]
        
        if not business_area:
            result.add_error("Business area cannot be empty")
        elif business_area.lower() not in valid_areas:
            result.add_error(f"Business area must be one of: {', '.join(valid_areas)}")
        
        return result
    
    @staticmethod
    def validate_capability_inputs(inputs: Dict[str, Any]) -> ValidationResult:
        """Validate capability inputs."""
        result = ValidationResult(True)
        
        if not isinstance(inputs, dict):
            result.add_error("Inputs must be a dictionary")
            return result
        
        # Check for required fields based on capability type
        # This is a simplified validation - in practice, this would be more sophisticated
        
        for key, value in inputs.items():
            if not isinstance(key, str):
                result.add_error(f"Input key '{key}' must be a string")
            
            if value is None:
                result.add_error(f"Input value for '{key}' cannot be None")
        
        return result
    
    @staticmethod
    def validate_workflow_steps(steps: List[Dict[str, Any]]) -> ValidationResult:
        """Validate workflow steps."""
        result = ValidationResult(True)
        
        if not isinstance(steps, list):
            result.add_error("Steps must be a list")
            return result
        
        if not steps:
            result.add_error("Workflow must have at least one step")
            return result
        
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                result.add_error(f"Step {i} must be a dictionary")
                continue
            
            # Validate required fields
            required_fields = ["name", "step_type", "description", "agent_type"]
            for field in required_fields:
                if field not in step:
                    result.add_error(f"Step {i} missing required field: {field}")
            
            # Validate step type
            valid_step_types = [
                "task", "condition", "parallel", "sequence", "loop",
                "api_call", "document_generation", "notification"
            ]
            if "step_type" in step and step["step_type"] not in valid_step_types:
                result.add_error(f"Step {i} has invalid step_type: {step['step_type']}")
        
        return result
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', value)
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary input."""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = InputValidator.sanitize_string(str(key), 100)
            
            # Sanitize value
            if isinstance(value, str):
                clean_value = InputValidator.sanitize_string(value)
            elif isinstance(value, dict):
                clean_value = InputValidator.sanitize_dict(value)
            elif isinstance(value, list):
                clean_value = [
                    InputValidator.sanitize_string(str(item)) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                clean_value = value
            
            sanitized[clean_key] = clean_value
        
        return sanitized

class ValidationMiddleware:
    """Middleware for request validation."""
    
    def __init__(self):
        self.validator = InputValidator()
    
    def validate_request(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate incoming request data."""
        result = ValidationResult(True)
        
        # Basic validation
        if not isinstance(data, dict):
            result.add_error("Request data must be a dictionary")
            return result
        
        # Sanitize input
        sanitized_data = self.validator.sanitize_dict(data)
        
        # Additional validation can be added here
        
        return result
