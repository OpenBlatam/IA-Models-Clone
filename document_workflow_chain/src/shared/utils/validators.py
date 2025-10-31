"""
Validation Utilities
====================

Advanced validation utilities for the application.
"""

from __future__ import annotations
import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, date
from email.utils import parseaddr
from urllib.parse import urlparse
import uuid

from pydantic import BaseModel, ValidationError, validator
from pydantic.validators import str_validator

from ..exceptions.application_exceptions import ValidationException


logger = logging.getLogger(__name__)


class AdvancedValidator:
    """Advanced validation utilities"""
    
    # Common regex patterns
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    PHONE_PATTERN = re.compile(
        r'^\+?1?-?\.?\s?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$'
    )
    URL_PATTERN = re.compile(
        r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    )
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    )
    SLUG_PATTERN = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9]+$')
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address"""
        if not email or not isinstance(email, str):
            return False
        
        # Basic format check
        if not AdvancedValidator.EMAIL_PATTERN.match(email):
            return False
        
        # Additional checks
        local, domain = parseaddr(email)
        if not local or not domain:
            return False
        
        # Check for common issues
        if '..' in local or local.startswith('.') or local.endswith('.'):
            return False
        
        if domain.startswith('.') or domain.endswith('.'):
            return False
        
        return True
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number"""
        if not phone or not isinstance(phone, str):
            return False
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Check if it's a valid length (7-15 digits)
        if len(digits) < 7 or len(digits) > 15:
            return False
        
        return True
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL"""
        if not url or not isinstance(url, str):
            return False
        
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def validate_uuid(uuid_string: str) -> bool:
        """Validate UUID string"""
        if not uuid_string or not isinstance(uuid_string, str):
            return False
        
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_slug(slug: str) -> bool:
        """Validate slug (URL-friendly string)"""
        if not slug or not isinstance(slug, str):
            return False
        
        return AdvancedValidator.SLUG_PATTERN.match(slug) is not None
    
    @staticmethod
    def validate_alphanumeric(text: str) -> bool:
        """Validate alphanumeric string"""
        if not text or not isinstance(text, str):
            return False
        
        return AdvancedValidator.ALPHANUMERIC_PATTERN.match(text) is not None
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength"""
        if not password or not isinstance(password, str):
            return {
                "is_valid": False,
                "score": 0,
                "issues": ["Password is required"]
            }
        
        issues = []
        score = 0
        
        # Length check
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        else:
            score += 1
        
        # Character variety checks
        if not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        else:
            score += 1
        
        if not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        else:
            score += 1
        
        if not re.search(r'\d', password):
            issues.append("Password must contain at least one digit")
        else:
            score += 1
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character")
        else:
            score += 1
        
        # Common password check
        common_passwords = [
            "password", "123456", "123456789", "qwerty", "abc123",
            "password123", "admin", "letmein", "welcome", "monkey"
        ]
        
        if password.lower() in common_passwords:
            issues.append("Password is too common")
            score = max(0, score - 2)
        
        # Sequential characters check
        if re.search(r'(.)\1{2,}', password):
            issues.append("Password contains repeated characters")
            score = max(0, score - 1)
        
        return {
            "is_valid": len(issues) == 0,
            "score": min(5, score),
            "issues": issues,
            "strength": "weak" if score < 3 else "medium" if score < 5 else "strong"
        }
    
    @staticmethod
    def validate_json_schema(data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against JSON schema"""
        try:
            from jsonschema import validate, ValidationError as JsonSchemaValidationError
            
            validate(instance=data, schema=schema)
            return {
                "is_valid": True,
                "errors": []
            }
        except JsonSchemaValidationError as e:
            return {
                "is_valid": False,
                "errors": [str(error) for error in e.errors]
            }
        except ImportError:
            logger.warning("jsonschema library not available for validation")
            return {
                "is_valid": False,
                "errors": ["JSON schema validation not available"]
            }
    
    @staticmethod
    def validate_date_range(start_date: Union[str, datetime, date], 
                          end_date: Union[str, datetime, date]) -> bool:
        """Validate date range"""
        try:
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            if isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_date = datetime.combine(start_date, datetime.min.time())
            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_date = datetime.combine(end_date, datetime.min.time())
            
            return start_date <= end_date
        except Exception:
            return False
    
    @staticmethod
    def validate_file_size(file_size: int, max_size: int) -> bool:
        """Validate file size"""
        return 0 <= file_size <= max_size
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
        """Validate file extension"""
        if not filename:
            return False
        
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        return extension in [ext.lower() for ext in allowed_extensions]


class PydanticValidators:
    """Pydantic validators for common use cases"""
    
    @staticmethod
    def validate_email_field(cls, v):
        """Pydantic validator for email fields"""
        if not AdvancedValidator.validate_email(v):
            raise ValueError('Invalid email format')
        return v
    
    @staticmethod
    def validate_phone_field(cls, v):
        """Pydantic validator for phone fields"""
        if not AdvancedValidator.validate_phone(v):
            raise ValueError('Invalid phone number format')
        return v
    
    @staticmethod
    def validate_url_field(cls, v):
        """Pydantic validator for URL fields"""
        if not AdvancedValidator.validate_url(v):
            raise ValueError('Invalid URL format')
        return v
    
    @staticmethod
    def validate_uuid_field(cls, v):
        """Pydantic validator for UUID fields"""
        if not AdvancedValidator.validate_uuid(v):
            raise ValueError('Invalid UUID format')
        return v
    
    @staticmethod
    def validate_slug_field(cls, v):
        """Pydantic validator for slug fields"""
        if not AdvancedValidator.validate_slug(v):
            raise ValueError('Invalid slug format')
        return v
    
    @staticmethod
    def validate_password_field(cls, v):
        """Pydantic validator for password fields"""
        result = AdvancedValidator.validate_password_strength(v)
        if not result["is_valid"]:
            raise ValueError(f'Password validation failed: {", ".join(result["issues"])}')
        return v
    
    @staticmethod
    def validate_not_empty(cls, v):
        """Pydantic validator for non-empty fields"""
        if not v or (isinstance(v, str) and not v.strip()):
            raise ValueError('Field cannot be empty')
        return v
    
    @staticmethod
    def validate_positive_number(cls, v):
        """Pydantic validator for positive numbers"""
        if v <= 0:
            raise ValueError('Value must be positive')
        return v
    
    @staticmethod
    def validate_non_negative_number(cls, v):
        """Pydantic validator for non-negative numbers"""
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v


class CustomValidators:
    """Custom validation functions"""
    
    @staticmethod
    def validate_workflow_name(name: str) -> None:
        """Validate workflow name"""
        if not name or not name.strip():
            raise ValidationException("Workflow name cannot be empty")
        
        if len(name) > 255:
            raise ValidationException("Workflow name cannot exceed 255 characters")
        
        # Check for invalid characters
        if re.search(r'[<>:"/\\|?*]', name):
            raise ValidationException("Workflow name contains invalid characters")
    
    @staticmethod
    def validate_node_title(title: str) -> None:
        """Validate node title"""
        if not title or not title.strip():
            raise ValidationException("Node title cannot be empty")
        
        if len(title) > 255:
            raise ValidationException("Node title cannot exceed 255 characters")
    
    @staticmethod
    def validate_node_content(content: str) -> None:
        """Validate node content"""
        if not content or not content.strip():
            raise ValidationException("Node content cannot be empty")
        
        if len(content) > 100000:
            raise ValidationException("Node content cannot exceed 100,000 characters")
    
    @staticmethod
    def validate_priority(priority: int) -> None:
        """Validate priority value"""
        if not isinstance(priority, int):
            raise ValidationException("Priority must be an integer")
        
        if not (1 <= priority <= 5):
            raise ValidationException("Priority must be between 1 and 5")
    
    @staticmethod
    def validate_tags(tags: List[str]) -> None:
        """Validate tags list"""
        if not isinstance(tags, list):
            raise ValidationException("Tags must be a list")
        
        if len(tags) > 20:
            raise ValidationException("Cannot have more than 20 tags")
        
        for tag in tags:
            if not isinstance(tag, str):
                raise ValidationException("All tags must be strings")
            
            if not tag.strip():
                raise ValidationException("Tags cannot be empty")
            
            if len(tag) > 50:
                raise ValidationException("Tag cannot exceed 50 characters")
            
            # Check for invalid characters
            if re.search(r'[<>:"/\\|?*]', tag):
                raise ValidationException(f"Tag '{tag}' contains invalid characters")


class ValidationResult:
    """Validation result container"""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings
        }


class ValidationService:
    """Service for complex validation operations"""
    
    def __init__(self):
        self.validators = AdvancedValidator()
        self.custom_validators = CustomValidators()
    
    def validate_workflow_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate workflow data"""
        result = ValidationResult(is_valid=True)
        
        try:
            # Validate name
            if "name" in data:
                self.custom_validators.validate_workflow_name(data["name"])
            
            # Validate description length
            if "description" in data and data["description"]:
                if len(data["description"]) > 1000:
                    result.add_error("Description cannot exceed 1000 characters")
            
            # Validate settings
            if "settings" in data and data["settings"]:
                if not isinstance(data["settings"], dict):
                    result.add_error("Settings must be a dictionary")
            
        except ValidationException as e:
            result.add_error(str(e))
        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
        
        return result
    
    def validate_node_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate node data"""
        result = ValidationResult(is_valid=True)
        
        try:
            # Validate title
            if "title" in data:
                self.custom_validators.validate_node_title(data["title"])
            
            # Validate content
            if "content" in data:
                self.custom_validators.validate_node_content(data["content"])
            
            # Validate priority
            if "priority" in data:
                self.custom_validators.validate_priority(data["priority"])
            
            # Validate tags
            if "tags" in data:
                self.custom_validators.validate_tags(data["tags"])
            
        except ValidationException as e:
            result.add_error(str(e))
        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
        
        return result
    
    def validate_bulk_operation(self, items: List[Dict[str, Any]], 
                              validator_func: Callable) -> Dict[str, ValidationResult]:
        """Validate bulk operation"""
        results = {}
        
        for i, item in enumerate(items):
            try:
                results[f"item_{i}"] = validator_func(item)
            except Exception as e:
                results[f"item_{i}"] = ValidationResult(
                    is_valid=False,
                    errors=[f"Validation error: {str(e)}"]
                )
        
        return results




