"""
AI Document Classifier Package
==============================

An AI-powered system that can identify document types from text queries
and export appropriate template designs.

This package provides:
- Document type classification (novel, contract, design, etc.)
- Template export in multiple formats
- RESTful API endpoints
- Docker deployment support

Example usage:
    from ai_document_classifier import DocumentClassifierEngine
    
    classifier = DocumentClassifierEngine()
    result = classifier.classify_document("I want to write a novel")
    print(f"Document type: {result.document_type}")
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__description__ = "AI-powered document type classification and template export"

from .document_classifier_engine import (
    DocumentClassifierEngine,
    DocumentType,
    ClassificationResult,
    TemplateDesign
)

__all__ = [
    "DocumentClassifierEngine",
    "DocumentType", 
    "ClassificationResult",
    "TemplateDesign"
]



























