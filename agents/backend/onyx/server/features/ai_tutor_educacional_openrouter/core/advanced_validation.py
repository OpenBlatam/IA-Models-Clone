"""
Advanced validation system for inputs and data.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, validator, Field
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class AdvancedValidator:
    """
    Advanced validation system with custom rules and validators.
    """
    
    def __init__(self):
        self.custom_validators: Dict[str, List[Callable]] = {}
    
    def validate_question(self, question: str) -> bool:
        """Validate a question input."""
        if not question or not question.strip():
            raise ValidationError("Question cannot be empty")
        
        if len(question) < 3:
            raise ValidationError("Question must be at least 3 characters")
        
        if len(question) > 2000:
            raise ValidationError("Question must be less than 2000 characters")
        
        return True
    
    def validate_subject(self, subject: str, allowed_subjects: List[str]) -> bool:
        """Validate subject name."""
        if not subject:
            raise ValidationError("Subject is required")
        
        if subject.lower() not in [s.lower() for s in allowed_subjects]:
            raise ValidationError(f"Subject must be one of: {', '.join(allowed_subjects)}")
        
        return True
    
    def validate_difficulty(self, difficulty: str) -> bool:
        """Validate difficulty level."""
        allowed_levels = ["basico", "intermedio", "avanzado"]
        
        if difficulty.lower() not in allowed_levels:
            raise ValidationError(f"Difficulty must be one of: {', '.join(allowed_levels)}")
        
        return True
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            raise ValidationError("Invalid email format")
        
        return True
    
    def validate_student_id(self, student_id: str) -> bool:
        """Validate student ID format."""
        if not student_id or not student_id.strip():
            raise ValidationError("Student ID cannot be empty")
        
        if len(student_id) < 3 or len(student_id) > 50:
            raise ValidationError("Student ID must be between 3 and 50 characters")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', student_id):
            raise ValidationError("Student ID can only contain letters, numbers, underscores, and hyphens")
        
        return True
    
    def validate_conversation_id(self, conversation_id: str) -> bool:
        """Validate conversation ID format."""
        if not conversation_id or not conversation_id.strip():
            raise ValidationError("Conversation ID cannot be empty")
        
        if len(conversation_id) > 100:
            raise ValidationError("Conversation ID must be less than 100 characters")
        
        return True
    
    def validate_quiz_data(self, quiz_data: Dict[str, Any]) -> bool:
        """Validate quiz data structure."""
        required_fields = ["topic", "subject", "difficulty", "num_questions"]
        
        for field in required_fields:
            if field not in quiz_data:
                raise ValidationError(f"Missing required field: {field}")
        
        if not isinstance(quiz_data["num_questions"], int):
            raise ValidationError("num_questions must be an integer")
        
        if quiz_data["num_questions"] < 1 or quiz_data["num_questions"] > 50:
            raise ValidationError("num_questions must be between 1 and 50")
        
        return True
    
    def validate_pagination(self, page: int, page_size: int) -> bool:
        """Validate pagination parameters."""
        if page < 1:
            raise ValidationError("Page must be greater than 0")
        
        if page_size < 1 or page_size > 100:
            raise ValidationError("Page size must be between 1 and 100")
        
        return True
    
    def register_custom_validator(self, field_name: str, validator_func: Callable):
        """Register a custom validator for a field."""
        if field_name not in self.custom_validators:
            self.custom_validators[field_name] = []
        
        self.custom_validators[field_name].append(validator_func)
    
    def validate_with_custom(self, field_name: str, value: Any) -> bool:
        """Validate using custom validators."""
        if field_name in self.custom_validators:
            for validator_func in self.custom_validators[field_name]:
                if not validator_func(value):
                    raise ValidationError(f"Custom validation failed for {field_name}")
        
        return True






