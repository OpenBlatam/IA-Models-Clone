"""
Advanced validators for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import re
import time
from typing import Any, Dict, List, Optional, Union, Callable
from marshmallow import Schema, fields, validates, ValidationError, post_load
from flask import request

class BaseValidator(Schema):
    """Base validator with common validation patterns."""
    
    def validate_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> None:
        """Validate required fields with early returns."""
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        if missing_fields:
            raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
    
    def validate_field_length(self, value: str, min_length: int = 1, max_length: int = 1000) -> None:
        """Validate field length with early returns."""
        if not isinstance(value, str):
            raise ValidationError("Field must be a string")
        
        if len(value) < min_length:
            raise ValidationError(f"Field must be at least {min_length} characters long")
        
        if len(value) > max_length:
            raise ValidationError(f"Field must be no more than {max_length} characters long")

class QueryValidator(BaseValidator):
    """Validator for query parameters with early returns."""
    
    query = fields.Str(required=True, validate=lambda x: len(x.strip()) > 0)
    max_documents = fields.Int(missing=None, validate=lambda x: x is None or (x > 0 and x <= 1000000))
    optimization_level = fields.Str(missing=None)
    
    @validates('query')
    def validate_query(self, value: str) -> None:
        """Validate query with early returns."""
        if not value or not value.strip():
            raise ValidationError("Query cannot be empty")
        
        if len(value) > 10000:
            raise ValidationError("Query is too long (max 10000 characters)")
        
        # Check for potentially harmful content
        harmful_patterns = ['<script', 'javascript:', 'data:', 'vbscript:']
        if any(pattern in value.lower() for pattern in harmful_patterns):
            raise ValidationError("Query contains potentially harmful content")
    
    @validates('optimization_level')
    def validate_optimization_level(self, value: Optional[str]) -> None:
        """Validate optimization level with early returns."""
        if value is None:
            return
        
        valid_levels = [
            'supreme_basic', 'supreme_advanced', 'supreme_expert', 'supreme_master',
            'supreme_legendary', 'supreme_transcendent', 'supreme_divine', 'supreme_omnipotent',
            'lightning', 'blazing', 'turbo', 'hyper', 'ultra', 'mega', 'giga', 'tera',
            'peta', 'exa', 'zetta', 'yotta', 'infinite', 'ultimate', 'absolute', 'perfect', 'infinity',
            'basic_hybrid', 'advanced_hybrid', 'expert_hybrid', 'master_hybrid',
            'legendary_hybrid', 'transcendent_hybrid', 'divine_hybrid', 'ultimate_hybrid'
        ]
        
        if value not in valid_levels:
            raise ValidationError(f"Invalid optimization level. Must be one of: {', '.join(valid_levels)}")

class ConfigValidator(BaseValidator):
    """Validator for configuration updates with early returns."""
    
    supreme_optimization_level = fields.Str(missing=None)
    ultra_fast_level = fields.Str(missing=None)
    refactored_ultimate_hybrid_level = fields.Str(missing=None)
    cuda_kernel_level = fields.Str(missing=None)
    gpu_utilization_level = fields.Str(missing=None)
    memory_optimization_level = fields.Str(missing=None)
    reward_function_level = fields.Str(missing=None)
    truthgpt_adapter_level = fields.Str(missing=None)
    microservices_level = fields.Str(missing=None)
    max_concurrent_generations = fields.Int(missing=None, validate=lambda x: x is None or (x > 0 and x <= 100000))
    max_documents_per_query = fields.Int(missing=None, validate=lambda x: x is None or (x > 0 and x <= 10000000))
    max_continuous_documents = fields.Int(missing=None, validate=lambda x: x is None or (x > 0 and x <= 100000000))
    
    @validates('supreme_optimization_level')
    def validate_supreme_level(self, value: Optional[str]) -> None:
        """Validate Supreme optimization level with early returns."""
        if value is None:
            return
        
        valid_levels = ['supreme_basic', 'supreme_advanced', 'supreme_expert', 'supreme_master',
                       'supreme_legendary', 'supreme_transcendent', 'supreme_divine', 'supreme_omnipotent']
        
        if value not in valid_levels:
            raise ValidationError(f"Invalid Supreme level. Must be one of: {', '.join(valid_levels)}")
    
    @validates('ultra_fast_level')
    def validate_ultra_fast_level(self, value: Optional[str]) -> None:
        """Validate Ultra-Fast level with early returns."""
        if value is None:
            return
        
        valid_levels = ['lightning', 'blazing', 'turbo', 'hyper', 'ultra', 'mega', 'giga', 'tera',
                       'peta', 'exa', 'zetta', 'yotta', 'infinite', 'ultimate', 'absolute', 'perfect', 'infinity']
        
        if value not in valid_levels:
            raise ValidationError(f"Invalid Ultra-Fast level. Must be one of: {', '.join(valid_levels)}")

class OptimizationRequestValidator(BaseValidator):
    """Validator for optimization requests with early returns."""
    
    optimization_type = fields.Str(required=True)
    level = fields.Str(required=True)
    model_data = fields.Dict(missing={})
    optimization_options = fields.Dict(missing={})
    
    @validates('optimization_type')
    def validate_optimization_type(self, value: str) -> None:
        """Validate optimization type with early returns."""
        valid_types = ['supreme', 'ultra_fast', 'refactored_ultimate_hybrid', 'cuda_kernel',
                      'gpu_utils', 'memory_utils', 'reward_function', 'truthgpt_adapter', 'microservices']
        
        if value not in valid_types:
            raise ValidationError(f"Invalid optimization type. Must be one of: {', '.join(valid_types)}")
    
    @validates('level')
    def validate_level(self, value: str) -> None:
        """Validate optimization level with early returns."""
        if not value or not value.strip():
            raise ValidationError("Level cannot be empty")
        
        if len(value) > 100:
            raise ValidationError("Level is too long (max 100 characters)")

class MonitoringQueryValidator(BaseValidator):
    """Validator for monitoring queries with early returns."""
    
    start_time = fields.DateTime(missing=None)
    end_time = fields.DateTime(missing=None)
    metric_types = fields.List(fields.Str(), missing=None)
    aggregation_level = fields.Str(missing='minute')
    include_details = fields.Bool(missing=False)
    
    @validates('aggregation_level')
    def validate_aggregation_level(self, value: str) -> None:
        """Validate aggregation level with early returns."""
        valid_levels = ['second', 'minute', 'hour', 'day']
        
        if value not in valid_levels:
            raise ValidationError(f"Invalid aggregation level. Must be one of: {', '.join(valid_levels)}")
    
    @validates('metric_types')
    def validate_metric_types(self, value: Optional[List[str]]) -> None:
        """Validate metric types with early returns."""
        if value is None:
            return
        
        valid_types = ['cpu', 'memory', 'gpu', 'disk', 'network', 'response_time', 'throughput', 'error_rate']
        invalid_types = [t for t in value if t not in valid_types]
        
        if invalid_types:
            raise ValidationError(f"Invalid metric types: {', '.join(invalid_types)}. Must be one of: {', '.join(valid_types)}")

class AnalyticsQueryValidator(BaseValidator):
    """Validator for analytics queries with early returns."""
    
    start_time = fields.DateTime(missing=None)
    end_time = fields.DateTime(missing=None)
    metric_types = fields.List(fields.Str(), missing=None)
    aggregation_level = fields.Str(missing='hour')
    include_trends = fields.Bool(missing=True)
    include_predictions = fields.Bool(missing=False)
    
    @validates('aggregation_level')
    def validate_aggregation_level(self, value: str) -> None:
        """Validate aggregation level with early returns."""
        valid_levels = ['minute', 'hour', 'day', 'week', 'month']
        
        if value not in valid_levels:
            raise ValidationError(f"Invalid aggregation level. Must be one of: {', '.join(valid_levels)}")

class AlertConfigValidator(BaseValidator):
    """Validator for alert configurations with early returns."""
    
    metric_name = fields.Str(required=True)
    threshold_value = fields.Float(required=True)
    threshold_type = fields.Str(required=True)
    alert_level = fields.Str(required=True)
    enabled = fields.Bool(missing=True)
    notification_channels = fields.List(fields.Str(), missing=[])
    
    @validates('threshold_type')
    def validate_threshold_type(self, value: str) -> None:
        """Validate threshold type with early returns."""
        valid_types = ['greater_than', 'less_than', 'equals']
        
        if value not in valid_types:
            raise ValidationError(f"Invalid threshold type. Must be one of: {', '.join(valid_types)}")
    
    @validates('alert_level')
    def validate_alert_level(self, value: str) -> None:
        """Validate alert level with early returns."""
        valid_levels = ['info', 'warning', 'critical']
        
        if value not in valid_levels:
            raise ValidationError(f"Invalid alert level. Must be one of: {', '.join(valid_levels)}")
    
    @validates('threshold_value')
    def validate_threshold_value(self, value: float) -> None:
        """Validate threshold value with early returns."""
        if not isinstance(value, (int, float)):
            raise ValidationError("Threshold value must be a number")
        
        if value < 0:
            raise ValidationError("Threshold value must be non-negative")

# Utility validation functions following RORO pattern
def validate_email(email: str) -> bool:
    """Validate email format with early returns."""
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url: str) -> bool:
    """Validate URL format with early returns."""
    if not url or not isinstance(url, str):
        return False
    
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))

def validate_phone(phone: str) -> bool:
    """Validate phone number format with early returns."""
    if not phone or not isinstance(phone, str):
        return False
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    return len(digits) >= 10 and len(digits) <= 15

def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength with early returns."""
    if not password or not isinstance(password, str):
        return {'valid': False, 'message': 'Password must be a string'}
    
    if len(password) < 8:
        return {'valid': False, 'message': 'Password must be at least 8 characters long'}
    
    if len(password) > 128:
        return {'valid': False, 'message': 'Password must be no more than 128 characters long'}
    
    # Check for required character types
    has_lower = bool(re.search(r'[a-z]', password))
    has_upper = bool(re.search(r'[A-Z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    
    strength_score = sum([has_lower, has_upper, has_digit, has_special])
    
    if strength_score < 3:
        return {'valid': False, 'message': 'Password must contain at least 3 of: lowercase, uppercase, digits, special characters'}
    
    return {'valid': True, 'strength_score': strength_score}

def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data against JSON schema with early returns."""
    if not isinstance(data, dict):
        return {'valid': False, 'message': 'Data must be a dictionary'}
    
    if not isinstance(schema, dict):
        return {'valid': False, 'message': 'Schema must be a dictionary'}
    
    # Basic schema validation (simplified implementation)
    required_fields = schema.get('required', [])
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return {'valid': False, 'message': f'Missing required fields: {", ".join(missing_fields)}'}
    
    return {'valid': True, 'message': 'Schema validation passed'}

def sanitize_input(input_string: str) -> str:
    """Sanitize input string with early returns."""
    if not input_string or not isinstance(input_string, str):
        return ""
    
    # Remove potentially harmful characters
    sanitized = re.sub(r'[<>"\']', '', input_string)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'on\w+=', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()

def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension with early returns."""
    if not filename or not isinstance(filename, str):
        return False
    
    if not allowed_extensions:
        return False
    
    # Get file extension
    extension = filename.split('.')[-1].lower() if '.' in filename else ''
    
    return extension in [ext.lower() for ext in allowed_extensions]

def validate_file_size(file_size: int, max_size: int) -> bool:
    """Validate file size with early returns."""
    if not isinstance(file_size, int) or file_size < 0:
        return False
    
    if not isinstance(max_size, int) or max_size <= 0:
        return False
    
    return file_size <= max_size

# Validation decorators
def validate_request_data(validator_class):
    """Decorator for request data validation with early returns."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                return {'success': False, 'message': 'Request must be JSON'}
            
            try:
                data = request.get_json()
                if not data:
                    return {'success': False, 'message': 'Request body is empty'}
                
                validator = validator_class()
                validated_data = validator.load(data)
                kwargs['validated_data'] = validated_data
                
                return func(*args, **kwargs)
            except ValidationError as e:
                return {'success': False, 'message': str(e)}
        
        return wrapper
    return decorator









