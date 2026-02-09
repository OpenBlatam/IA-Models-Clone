"""
Enums
Enumeration types for API models
"""

from enum import Enum

class ContentType(str, Enum):
    """Content types"""
    PRESENTATION = "presentation"
    DOCUMENT = "document"
    WEB_PAGE = "web_page"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    REPORT = "report"
    PROPOSAL = "proposal"

class OutputFormat(str, Enum):
    """Output formats"""
    PDF = "pdf"
    PPTX = "pptx"
    HTML = "html"
    DOCX = "docx"
    MD = "markdown"
    JSON = "json"
    PNG = "png"
    JPG = "jpg"

class DesignStyle(str, Enum):
    """Design styles"""
    MODERN = "modern"
    MINIMALIST = "minimalist"
    CORPORATE = "corporate"
    CREATIVE = "creative"
    ACADEMIC = "academic"
    CASUAL = "casual"
    PROFESSIONAL = "professional"

class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    COLLABORATOR = "collaborator"
    VIEWER = "viewer"

class SessionStatus(str, Enum):
    """Collaboration session status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"







