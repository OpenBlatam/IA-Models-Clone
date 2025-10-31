"""
Use Cases Package
=================

Comprehensive use cases for the AI Document Classifier system
covering various industries and document types.
"""

from .document_use_cases import (
    DocumentUseCaseManager,
    UseCase,
    IndustryType,
    DocumentComplexity
)

from .industry_specific_cases import (
    IndustrySpecificManager,
    IndustryUseCase,
    IndustryRequirement,
    ComplianceStandard,
    DocumentCategory
)

__all__ = [
    "DocumentUseCaseManager",
    "UseCase",
    "IndustryType", 
    "DocumentComplexity",
    "IndustrySpecificManager",
    "IndustryUseCase",
    "IndustryRequirement",
    "ComplianceStandard",
    "DocumentCategory"
]

