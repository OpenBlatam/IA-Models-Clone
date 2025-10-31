"""
Input validation system for Export IA.
"""

import re
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class Severity(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    field: str
    message: str
    severity: Severity
    value: Any = None
    suggestion: Optional[str] = None


class ContentValidator:
    """Validates document content structure and quality."""
    
    def __init__(self):
        self.rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules."""
        return {
            "title": {
                "required": True,
                "min_length": 3,
                "max_length": 200,
                "pattern": r"^[a-zA-Z0-9\s\-_.,!?()]+$",
                "error_message": "Title must be 3-200 characters and contain only letters, numbers, spaces, and basic punctuation"
            },
            "sections": {
                "required": True,
                "min_count": 1,
                "max_count": 50,
                "error_message": "Document must have 1-50 sections"
            },
            "section_heading": {
                "required": True,
                "min_length": 2,
                "max_length": 100,
                "pattern": r"^[a-zA-Z0-9\s\-_.,!?()]+$",
                "error_message": "Section heading must be 2-100 characters and contain only letters, numbers, spaces, and basic punctuation"
            },
            "section_content": {
                "required": True,
                "min_length": 10,
                "max_length": 10000,
                "error_message": "Section content must be 10-10000 characters"
            }
        }
    
    def validate_content(self, content: Dict[str, Any]) -> List[ValidationResult]:
        """Validate document content."""
        results = []
        
        # Validate title
        results.extend(self._validate_title(content.get("title")))
        
        # Validate sections
        results.extend(self._validate_sections(content.get("sections", [])))
        
        # Validate overall structure
        results.extend(self._validate_structure(content))
        
        return results
    
    def _validate_title(self, title: Any) -> List[ValidationResult]:
        """Validate document title."""
        results = []
        rule = self.rules["title"]
        
        if title is None:
            results.append(ValidationResult(
                field="title",
                message="Title is required",
                severity=Severity.ERROR,
                suggestion="Add a descriptive title for your document"
            ))
            return results
        
        if not isinstance(title, str):
            results.append(ValidationResult(
                field="title",
                message="Title must be a string",
                severity=Severity.ERROR,
                value=title
            ))
            return results
        
        # Length validation
        if len(title) < rule["min_length"]:
            results.append(ValidationResult(
                field="title",
                message=f"Title too short (minimum {rule['min_length']} characters)",
                severity=Severity.ERROR,
                value=title,
                suggestion="Make the title more descriptive"
            ))
        
        if len(title) > rule["max_length"]:
            results.append(ValidationResult(
                field="title",
                message=f"Title too long (maximum {rule['max_length']} characters)",
                severity=Severity.WARNING,
                value=title,
                suggestion="Consider shortening the title"
            ))
        
        # Pattern validation
        if not re.match(rule["pattern"], title):
            results.append(ValidationResult(
                field="title",
                message="Title contains invalid characters",
                severity=Severity.WARNING,
                value=title,
                suggestion="Use only letters, numbers, spaces, and basic punctuation"
            ))
        
        return results
    
    def _validate_sections(self, sections: Any) -> List[ValidationResult]:
        """Validate document sections."""
        results = []
        rule = self.rules["sections"]
        
        if sections is None:
            results.append(ValidationResult(
                field="sections",
                message="Sections are required",
                severity=Severity.ERROR,
                suggestion="Add at least one section to your document"
            ))
            return results
        
        if not isinstance(sections, list):
            results.append(ValidationResult(
                field="sections",
                message="Sections must be a list",
                severity=Severity.ERROR,
                value=sections
            ))
            return results
        
        # Count validation
        if len(sections) < rule["min_count"]:
            results.append(ValidationResult(
                field="sections",
                message=f"Too few sections (minimum {rule['min_count']})",
                severity=Severity.ERROR,
                value=len(sections),
                suggestion="Add more sections to your document"
            ))
        
        if len(sections) > rule["max_count"]:
            results.append(ValidationResult(
                field="sections",
                message=f"Too many sections (maximum {rule['max_count']})",
                severity=Severity.WARNING,
                value=len(sections),
                suggestion="Consider combining or removing some sections"
            ))
        
        # Validate individual sections
        for i, section in enumerate(sections):
            if not isinstance(section, dict):
                results.append(ValidationResult(
                    field=f"sections[{i}]",
                    message="Section must be a dictionary",
                    severity=Severity.ERROR,
                    value=section
                ))
                continue
            
            # Validate section heading
            results.extend(self._validate_section_heading(section.get("heading"), i))
            
            # Validate section content
            results.extend(self._validate_section_content(section.get("content"), i))
        
        return results
    
    def _validate_section_heading(self, heading: Any, section_index: int) -> List[ValidationResult]:
        """Validate section heading."""
        results = []
        rule = self.rules["section_heading"]
        field = f"sections[{section_index}].heading"
        
        if heading is None:
            results.append(ValidationResult(
                field=field,
                message="Section heading is required",
                severity=Severity.ERROR,
                suggestion="Add a descriptive heading for this section"
            ))
            return results
        
        if not isinstance(heading, str):
            results.append(ValidationResult(
                field=field,
                message="Section heading must be a string",
                severity=Severity.ERROR,
                value=heading
            ))
            return results
        
        # Length validation
        if len(heading) < rule["min_length"]:
            results.append(ValidationResult(
                field=field,
                message=f"Section heading too short (minimum {rule['min_length']} characters)",
                severity=Severity.ERROR,
                value=heading,
                suggestion="Make the heading more descriptive"
            ))
        
        if len(heading) > rule["max_length"]:
            results.append(ValidationResult(
                field=field,
                message=f"Section heading too long (maximum {rule['max_length']} characters)",
                severity=Severity.WARNING,
                value=heading,
                suggestion="Consider shortening the heading"
            ))
        
        # Pattern validation
        if not re.match(rule["pattern"], heading):
            results.append(ValidationResult(
                field=field,
                message="Section heading contains invalid characters",
                severity=Severity.WARNING,
                value=heading,
                suggestion="Use only letters, numbers, spaces, and basic punctuation"
            ))
        
        return results
    
    def _validate_section_content(self, content: Any, section_index: int) -> List[ValidationResult]:
        """Validate section content."""
        results = []
        rule = self.rules["section_content"]
        field = f"sections[{section_index}].content"
        
        if content is None:
            results.append(ValidationResult(
                field=field,
                message="Section content is required",
                severity=Severity.ERROR,
                suggestion="Add content to this section"
            ))
            return results
        
        if not isinstance(content, str):
            results.append(ValidationResult(
                field=field,
                message="Section content must be a string",
                severity=Severity.ERROR,
                value=content
            ))
            return results
        
        # Length validation
        if len(content) < rule["min_length"]:
            results.append(ValidationResult(
                field=field,
                message=f"Section content too short (minimum {rule['min_length']} characters)",
                severity=Severity.ERROR,
                value=content,
                suggestion="Add more detailed content to this section"
            ))
        
        if len(content) > rule["max_length"]:
            results.append(ValidationResult(
                field=field,
                message=f"Section content too long (maximum {rule['max_length']} characters)",
                severity=Severity.WARNING,
                value=content,
                suggestion="Consider breaking this section into smaller parts"
            ))
        
        return results
    
    def _validate_structure(self, content: Dict[str, Any]) -> List[ValidationResult]:
        """Validate overall document structure."""
        results = []
        
        # Check for required top-level fields
        required_fields = ["title", "sections"]
        for field in required_fields:
            if field not in content:
                results.append(ValidationResult(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    severity=Severity.ERROR,
                    suggestion=f"Add the '{field}' field to your document"
                ))
        
        # Check for duplicate section headings
        if "sections" in content and isinstance(content["sections"], list):
            headings = []
            for i, section in enumerate(content["sections"]):
                if isinstance(section, dict) and "heading" in section:
                    heading = section["heading"]
                    if heading in headings:
                        results.append(ValidationResult(
                            field=f"sections[{i}].heading",
                            message=f"Duplicate section heading: '{heading}'",
                            severity=Severity.WARNING,
                            value=heading,
                            suggestion="Use unique headings for each section"
                        ))
                    headings.append(heading)
        
        return results


class ConfigValidator:
    """Validates export configuration."""
    
    def validate_config(self, config: Any) -> List[ValidationResult]:
        """Validate export configuration."""
        results = []
        
        if config is None:
            results.append(ValidationResult(
                field="config",
                message="Configuration is required",
                severity=Severity.ERROR
            ))
            return results
        
        # Validate format
        if not hasattr(config, 'format'):
            results.append(ValidationResult(
                field="config.format",
                message="Export format is required",
                severity=Severity.ERROR
            ))
        elif config.format is None:
            results.append(ValidationResult(
                field="config.format",
                message="Export format cannot be None",
                severity=Severity.ERROR
            ))
        
        # Validate document type
        if not hasattr(config, 'document_type'):
            results.append(ValidationResult(
                field="config.document_type",
                message="Document type is required",
                severity=Severity.ERROR
            ))
        elif config.document_type is None:
            results.append(ValidationResult(
                field="config.document_type",
                message="Document type cannot be None",
                severity=Severity.ERROR
            ))
        
        # Validate quality level
        if not hasattr(config, 'quality_level'):
            results.append(ValidationResult(
                field="config.quality_level",
                message="Quality level is required",
                severity=Severity.ERROR
            ))
        elif config.quality_level is None:
            results.append(ValidationResult(
                field="config.quality_level",
                message="Quality level cannot be None",
                severity=Severity.ERROR
            ))
        
        return results


class FileValidator:
    """Validates file paths and permissions."""
    
    def validate_output_path(self, path: str) -> List[ValidationResult]:
        """Validate output file path."""
        results = []
        
        if not path:
            results.append(ValidationResult(
                field="output_path",
                message="Output path is required",
                severity=Severity.ERROR
            ))
            return results
        
        # Check if directory exists and is writable
        directory = os.path.dirname(os.path.abspath(path))
        if not os.path.exists(directory):
            results.append(ValidationResult(
                field="output_path",
                message=f"Output directory does not exist: {directory}",
                severity=Severity.ERROR,
                value=path,
                suggestion="Create the directory or choose a different path"
            ))
        elif not os.access(directory, os.W_OK):
            results.append(ValidationResult(
                field="output_path",
                message=f"Output directory is not writable: {directory}",
                severity=Severity.ERROR,
                value=path,
                suggestion="Check directory permissions or choose a different path"
            ))
        
        # Check file extension
        if not os.path.splitext(path)[1]:
            results.append(ValidationResult(
                field="output_path",
                message="Output path must have a file extension",
                severity=Severity.WARNING,
                value=path,
                suggestion="Add an appropriate file extension"
            ))
        
        return results


class ValidationManager:
    """Centralized validation management."""
    
    def __init__(self):
        self.content_validator = ContentValidator()
        self.config_validator = ConfigValidator()
        self.file_validator = FileValidator()
    
    def validate_export_request(
        self, 
        content: Dict[str, Any], 
        config: Any, 
        output_path: Optional[str] = None
    ) -> List[ValidationResult]:
        """Validate a complete export request."""
        results = []
        
        # Validate content
        results.extend(self.content_validator.validate_content(content))
        
        # Validate configuration
        results.extend(self.config_validator.validate_config(config))
        
        # Validate output path if provided
        if output_path:
            results.extend(self.file_validator.validate_output_path(output_path))
        
        return results
    
    def has_errors(self, results: List[ValidationResult]) -> bool:
        """Check if validation results contain any errors."""
        return any(result.severity == Severity.ERROR for result in results)
    
    def get_errors(self, results: List[ValidationResult]) -> List[ValidationResult]:
        """Get only error-level validation results."""
        return [result for result in results if result.severity == Severity.ERROR]
    
    def get_warnings(self, results: List[ValidationResult]) -> List[ValidationResult]:
        """Get only warning-level validation results."""
        return [result for result in results if result.severity == Severity.WARNING]
    
    def format_results(self, results: List[ValidationResult]) -> str:
        """Format validation results as a readable string."""
        if not results:
            return "Validation passed with no issues."
        
        output = []
        errors = self.get_errors(results)
        warnings = self.get_warnings(results)
        
        if errors:
            output.append("Validation Errors:")
            for error in errors:
                output.append(f"  ❌ {error.field}: {error.message}")
                if error.suggestion:
                    output.append(f"     💡 {error.suggestion}")
        
        if warnings:
            output.append("\nValidation Warnings:")
            for warning in warnings:
                output.append(f"  ⚠️  {warning.field}: {warning.message}")
                if warning.suggestion:
                    output.append(f"     💡 {warning.suggestion}")
        
        return "\n".join(output)


# Global validation manager instance
_validation_manager: Optional[ValidationManager] = None


def get_validation_manager() -> ValidationManager:
    """Get the global validation manager instance."""
    global _validation_manager
    if _validation_manager is None:
        _validation_manager = ValidationManager()
    return _validation_manager




