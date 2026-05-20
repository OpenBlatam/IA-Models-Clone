"""
Validation Engine for Document Analyzer
========================================

Advanced validation system for documents, inputs, and outputs.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"

@dataclass
class ValidationRule:
    """Validation rule"""
    name: str
    validator: Callable
    error_message: str
    level: ValidationLevel = ValidationLevel.MODERATE

@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

class ValidationEngine:
    """Advanced validation engine"""
    
    def __init__(self, default_level: ValidationLevel = ValidationLevel.MODERATE):
        self.default_level = default_level
        self.rules: Dict[str, ValidationRule] = {}
        self._register_default_rules()
        logger.info(f"ValidationEngine initialized with level: {default_level.value}")
    
    def _register_default_rules(self):
        """Register default validation rules"""
        
        # Document size validation
        def validate_size(content: str, max_size: int = 10 * 1024 * 1024) -> bool:
            return len(content.encode('utf-8')) <= max_size
        
        self.register_rule(
            "document_size",
            validate_size,
            "Document size exceeds maximum allowed (10MB)",
            ValidationLevel.STRICT
        )
        
        # Content validation
        def validate_not_empty(content: str) -> bool:
            return len(content.strip()) > 0
        
        self.register_rule(
            "not_empty",
            validate_not_empty,
            "Document content cannot be empty",
            ValidationLevel.STRICT
        )
        
        # Encoding validation
        def validate_encoding(content: str) -> bool:
            try:
                content.encode('utf-8')
                return True
            except UnicodeEncodeError:
                return False
        
        self.register_rule(
            "encoding",
            validate_encoding,
            "Document contains invalid UTF-8 characters",
            ValidationLevel.STRICT
        )
    
    def register_rule(
        self,
        name: str,
        validator: Callable,
        error_message: str,
        level: ValidationLevel = None
    ):
        """Register a validation rule"""
        rule = ValidationRule(
            name=name,
            validator=validator,
            error_message=error_message,
            level=level or self.default_level
        )
        self.rules[name] = rule
        logger.info(f"Registered validation rule: {name}")
    
    def validate(
        self,
        data: Any,
        rules: Optional[List[str]] = None,
        level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """Validate data against rules"""
        result = ValidationResult(is_valid=True)
        level = level or self.default_level
        
        rules_to_check = rules or list(self.rules.keys())
        
        for rule_name in rules_to_check:
            if rule_name not in self.rules:
                result.warnings.append(f"Rule '{rule_name}' not found")
                continue
            
            rule = self.rules[rule_name]
            
            # Check if rule level matches
            if level == ValidationLevel.STRICT and rule.level != ValidationLevel.STRICT:
                continue
            elif level == ValidationLevel.LENIENT and rule.level == ValidationLevel.STRICT:
                continue
            
            try:
                is_valid = rule.validator(data)
                if not is_valid:
                    result.is_valid = False
                    if rule.level == ValidationLevel.STRICT:
                        result.errors.append(rule.error_message)
                    else:
                        result.warnings.append(rule.error_message)
            except Exception as e:
                result.is_valid = False
                result.errors.append(f"Validation error in rule '{rule_name}': {str(e)}")
        
        return result
    
    def validate_document(
        self,
        content: str,
        document_type: Optional[str] = None
    ) -> ValidationResult:
        """Validate document content"""
        result = ValidationResult(is_valid=True)
        
        # Basic validations
        basic_result = self.validate(content, rules=["not_empty", "encoding", "document_size"])
        if not basic_result.is_valid:
            result.is_valid = False
            result.errors.extend(basic_result.errors)
            result.warnings.extend(basic_result.warnings)
        
        # Type-specific validations
        if document_type:
            type_result = self._validate_by_type(content, document_type)
            if not type_result.is_valid:
                result.is_valid = False
                result.errors.extend(type_result.errors)
                result.warnings.extend(type_result.warnings)
        
        return result
    
    def _validate_by_type(self, content: str, document_type: str) -> ValidationResult:
        """Validate based on document type"""
        result = ValidationResult(is_valid=True)
        
        if document_type == "json":
            try:
                import json
                json.loads(content)
            except json.JSONDecodeError as e:
                result.is_valid = False
                result.errors.append(f"Invalid JSON: {str(e)}")
        
        elif document_type == "xml":
            try:
                import xml.etree.ElementTree as ET
                ET.fromstring(content)
            except ET.ParseError as e:
                result.is_valid = False
                result.errors.append(f"Invalid XML: {str(e)}")
        
        return result

# Global instance
validation_engine = ValidationEngine()
















