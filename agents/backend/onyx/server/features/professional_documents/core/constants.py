"""
Constants for the professional documents module.

Centralized constants and configuration values.
"""

from typing import Dict, List

MIN_QUERY_LENGTH = 10
WORDS_PER_PAGE = 275
DEFAULT_EXPORT_DIR = "exports"

# AI Configuration
DEFAULT_AI_MODEL = "gpt-4"
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.7
AI_SIMULATION_DELAY_SECONDS = 1.0

SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"]

DOCUMENT_TONES = ["formal", "professional", "casual", "academic", "technical"]

DOCUMENT_LENGTHS = ["short", "medium", "long", "comprehensive"]

LENGTH_GUIDELINES: Dict[str, str] = {
    "short": "2-5 pages, concise and to the point",
    "medium": "5-15 pages, comprehensive but focused",
    "long": "15-30 pages, detailed and thorough",
    "comprehensive": "30+ pages, exhaustive coverage"
}

TONE_GUIDELINES: Dict[str, str] = {
    "formal": "Use formal language, avoid contractions, maintain professional distance",
    "professional": "Use clear, professional language suitable for business contexts",
    "casual": "Use conversational tone, contractions are acceptable",
    "academic": "Use scholarly language, include citations and references",
    "technical": "Use precise technical terminology, include technical details"
}

EXPORT_FORMAT_EXTENSIONS: Dict[str, str] = {
    "pdf": "pdf",
    "md": "md",
    "markdown": "md",
    "docx": "docx",
    "word": "docx",
    "html": "html"
}

DOCUMENT_TYPE_SUFFIXES: Dict[str, str] = {
    "report": "Report",
    "proposal": "Proposal",
    "manual": "Manual",
    "guide": "Guide",
    "whitepaper": "Whitepaper",
    "business_plan": "Business Plan",
    "technical_document": "Technical Documentation",
    "academic_paper": "Research Paper",
    "newsletter": "Newsletter",
    "brochure": "Brochure",
    "catalog": "Catalog",
    "presentation": "Presentation"
}

