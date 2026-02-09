"""
Document Generation Package
===========================

Modular document generation system with templates, formatters, and generators.
"""

from .generators import DocumentGenerator
from .templates import TemplateManager, DocumentTemplate
from .formatters import (
    PDFFormatter, DOCXFormatter, PPTXFormatter, 
    HTMLFormatter, MarkdownFormatter, JSONFormatter
)
from .types import DocumentType, DocumentFormat, TemplateType

__all__ = [
    "DocumentGenerator",
    "TemplateManager", 
    "DocumentTemplate",
    "PDFFormatter",
    "DOCXFormatter", 
    "PPTXFormatter",
    "HTMLFormatter",
    "MarkdownFormatter",
    "JSONFormatter",
    "DocumentType",
    "DocumentFormat", 
    "TemplateType"
]
