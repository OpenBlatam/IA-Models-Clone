"""
Type definitions for professional documents module.

Centralized type aliases and TypedDict definitions for better type safety.
"""

from typing import TypedDict, List, Dict, Any, Literal

# Type aliases for common patterns
DocumentMetadata = Dict[str, Any]
SectionMetadata = Dict[str, Any]

# TypedDict for section data structure
class SectionData(TypedDict, total=False):
    """Type definition for section data from AI responses."""
    title: str
    content: str
    level: int
    metadata: SectionMetadata


# Type aliases for document tones and lengths
DocumentTone = Literal["formal", "professional", "casual", "academic", "technical"]
DocumentLength = Literal["short", "medium", "long", "comprehensive"]

# Type alias for document status
DocumentStatus = Literal["draft", "generating", "completed", "failed", "archived"]

# Type alias for language codes
LanguageCode = Literal["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"]

# Type alias for export statistics
ExportStats = Dict[str, int]

# TypedDict for template usage statistics
class TemplateUsage(TypedDict):
    """Type definition for template usage statistics."""
    template_id: str
    name: str
    usage_count: int

