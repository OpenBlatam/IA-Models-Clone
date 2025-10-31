"""
Export domain value objects - Immutable objects that represent concepts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional
import re


class ExportFormat(Enum):
    """Export format enumeration."""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    RTF = "rtf"
    TXT = "txt"
    JSON = "json"
    XML = "xml"


class DocumentType(Enum):
    """Document type enumeration."""
    REPORT = "report"
    PRESENTATION = "presentation"
    MANUAL = "manual"
    PROPOSAL = "proposal"
    CONTRACT = "contract"
    INVOICE = "invoice"
    LETTER = "letter"
    MEMO = "memo"
    NEWSLETTER = "newsletter"
    CATALOG = "catalog"


class QualityLevel(Enum):
    """Quality level enumeration."""
    DRAFT = "draft"
    STANDARD = "standard"
    PROFESSIONAL = "professional"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass(frozen=True)
class ExportConfig:
    """Export configuration value object."""
    format: ExportFormat
    document_type: DocumentType
    quality_level: QualityLevel = QualityLevel.PROFESSIONAL
    template_id: Optional[str] = None
    custom_styles: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.custom_styles is None:
            object.__setattr__(self, 'custom_styles', {})
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    def with_format(self, format: ExportFormat) -> 'ExportConfig':
        """Create new config with different format."""
        return ExportConfig(
            format=format,
            document_type=self.document_type,
            quality_level=self.quality_level,
            template_id=self.template_id,
            custom_styles=self.custom_styles.copy(),
            metadata=self.metadata.copy()
        )
    
    def with_quality_level(self, quality_level: QualityLevel) -> 'ExportConfig':
        """Create new config with different quality level."""
        return ExportConfig(
            format=self.format,
            document_type=self.document_type,
            quality_level=quality_level,
            template_id=self.template_id,
            custom_styles=self.custom_styles.copy(),
            metadata=self.metadata.copy()
        )
    
    def add_style(self, key: str, value: Any) -> 'ExportConfig':
        """Add custom style to config."""
        new_styles = self.custom_styles.copy()
        new_styles[key] = value
        return ExportConfig(
            format=self.format,
            document_type=self.document_type,
            quality_level=self.quality_level,
            template_id=self.template_id,
            custom_styles=new_styles,
            metadata=self.metadata.copy()
        )
    
    def add_metadata(self, key: str, value: Any) -> 'ExportConfig':
        """Add metadata to config."""
        new_metadata = self.metadata.copy()
        new_metadata[key] = value
        return ExportConfig(
            format=self.format,
            document_type=self.document_type,
            quality_level=self.quality_level,
            template_id=self.template_id,
            custom_styles=self.custom_styles.copy(),
            metadata=new_metadata
        )


@dataclass(frozen=True)
class FileInfo:
    """File information value object."""
    name: str
    size: int
    mime_type: str
    extension: str
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Validate file info after initialization."""
        if self.size < 0:
            raise ValueError("File size cannot be negative")
        
        if not self.name:
            raise ValueError("File name cannot be empty")
        
        if not self.mime_type:
            raise ValueError("MIME type cannot be empty")
    
    def is_valid_size(self, max_size: int) -> bool:
        """Check if file size is within limits."""
        return self.size <= max_size
    
    def get_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.size / (1024 * 1024)
    
    def get_size_kb(self) -> float:
        """Get file size in kilobytes."""
        return self.size / 1024


@dataclass(frozen=True)
class QualityMetrics:
    """Quality metrics value object."""
    overall_score: float
    formatting_score: float
    content_score: float
    accessibility_score: float
    professional_score: float
    issues: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        """Validate quality metrics after initialization."""
        if self.issues is None:
            object.__setattr__(self, 'issues', [])
        if self.suggestions is None:
            object.__setattr__(self, 'suggestions', [])
        
        # Validate scores are between 0 and 100
        for score_name, score_value in [
            ("overall_score", self.overall_score),
            ("formatting_score", self.formatting_score),
            ("content_score", self.content_score),
            ("accessibility_score", self.accessibility_score),
            ("professional_score", self.professional_score)
        ]:
            if not 0 <= score_value <= 100:
                raise ValueError(f"{score_name} must be between 0 and 100")
    
    def is_high_quality(self, threshold: float = 80.0) -> bool:
        """Check if quality meets threshold."""
        return self.overall_score >= threshold
    
    def get_quality_grade(self) -> str:
        """Get quality grade based on overall score."""
        if self.overall_score >= 90:
            return "A"
        elif self.overall_score >= 80:
            return "B"
        elif self.overall_score >= 70:
            return "C"
        elif self.overall_score >= 60:
            return "D"
        else:
            return "F"
    
    def add_issue(self, issue: str) -> 'QualityMetrics':
        """Add issue to metrics."""
        new_issues = self.issues.copy()
        if issue not in new_issues:
            new_issues.append(issue)
        
        return QualityMetrics(
            overall_score=self.overall_score,
            formatting_score=self.formatting_score,
            content_score=self.content_score,
            accessibility_score=self.accessibility_score,
            professional_score=self.professional_score,
            issues=new_issues,
            suggestions=self.suggestions.copy()
        )
    
    def add_suggestion(self, suggestion: str) -> 'QualityMetrics':
        """Add suggestion to metrics."""
        new_suggestions = self.suggestions.copy()
        if suggestion not in new_suggestions:
            new_suggestions.append(suggestion)
        
        return QualityMetrics(
            overall_score=self.overall_score,
            formatting_score=self.formatting_score,
            content_score=self.content_score,
            accessibility_score=self.accessibility_score,
            professional_score=self.professional_score,
            issues=self.issues.copy(),
            suggestions=new_suggestions
        )


@dataclass(frozen=True)
class ProcessingTime:
    """Processing time value object."""
    total_time: float
    validation_time: float
    enhancement_time: float
    export_time: float
    quality_check_time: float
    
    def __post_init__(self):
        """Validate processing times after initialization."""
        for time_name, time_value in [
            ("total_time", self.total_time),
            ("validation_time", self.validation_time),
            ("enhancement_time", self.enhancement_time),
            ("export_time", self.export_time),
            ("quality_check_time", self.quality_check_time)
        ]:
            if time_value < 0:
                raise ValueError(f"{time_name} cannot be negative")
    
    def get_total_seconds(self) -> float:
        """Get total processing time in seconds."""
        return self.total_time
    
    def get_total_minutes(self) -> float:
        """Get total processing time in minutes."""
        return self.total_time / 60
    
    def is_fast(self, threshold: float = 30.0) -> bool:
        """Check if processing is fast."""
        return self.total_time <= threshold
    
    def get_breakdown_percentage(self) -> Dict[str, float]:
        """Get processing time breakdown as percentages."""
        if self.total_time == 0:
            return {
                "validation": 0.0,
                "enhancement": 0.0,
                "export": 0.0,
                "quality_check": 0.0
            }
        
        return {
            "validation": (self.validation_time / self.total_time) * 100,
            "enhancement": (self.enhancement_time / self.total_time) * 100,
            "export": (self.export_time / self.total_time) * 100,
            "quality_check": (self.quality_check_time / self.total_time) * 100
        }


@dataclass(frozen=True)
class ExportSpecification:
    """Export specification value object."""
    format: ExportFormat
    document_type: DocumentType
    quality_level: QualityLevel
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_formats: List[ExportFormat] = None
    required_fields: List[str] = None
    optional_fields: List[str] = None
    
    def __post_init__(self):
        """Validate specification after initialization."""
        if self.allowed_formats is None:
            object.__setattr__(self, 'allowed_formats', list(ExportFormat))
        if self.required_fields is None:
            object.__setattr__(self, 'required_fields', [])
        if self.optional_fields is None:
            object.__setattr__(self, 'optional_fields', [])
    
    def is_format_allowed(self, format: ExportFormat) -> bool:
        """Check if format is allowed."""
        return format in self.allowed_formats
    
    def is_field_required(self, field: str) -> bool:
        """Check if field is required."""
        return field in self.required_fields
    
    def is_field_optional(self, field: str) -> bool:
        """Check if field is optional."""
        return field in self.optional_fields
    
    def validate_content(self, content: Dict[str, Any]) -> List[str]:
        """Validate content against specification."""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in content:
                errors.append(f"Required field missing: {field}")
        
        # Check file size if applicable
        if "file_size" in content:
            if content["file_size"] > self.max_file_size:
                errors.append(f"File size exceeds maximum: {self.max_file_size}")
        
        return errors


@dataclass(frozen=True)
class UserContext:
    """User context value object."""
    user_id: str
    username: str
    email: str
    role: str
    permissions: List[str] = None
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate user context after initialization."""
        if self.permissions is None:
            object.__setattr__(self, 'permissions', [])
        if self.preferences is None:
            object.__setattr__(self, 'preferences', {})
        
        if not self.user_id:
            raise ValueError("User ID cannot be empty")
        
        if not self.username:
            raise ValueError("Username cannot be empty")
        
        if not self.email or not re.match(r'^[^@]+@[^@]+\.[^@]+$', self.email):
            raise ValueError("Valid email is required")
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return self.role == role
    
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return self.role == "admin"
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference."""
        return self.preferences.get(key, default)




