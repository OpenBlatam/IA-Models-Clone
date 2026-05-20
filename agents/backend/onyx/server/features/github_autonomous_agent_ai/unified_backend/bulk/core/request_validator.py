"""
Request Validator - Validation utilities for bulk requests
========================================================

Provides validation functions for bulk document generation requests.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from .constants import (
    MAX_DOCUMENTS_LIMIT, MIN_DOCUMENTS, MAX_DOCUMENT_TYPES, MAX_BUSINESS_AREAS,
    MIN_QUERY_LENGTH, MAX_QUERY_LENGTH, MAX_DOCUMENT_TYPE_LENGTH,
    MAX_BUSINESS_AREA_LENGTH, MIN_PRIORITY, MAX_PRIORITY
)

logger = logging.getLogger(__name__)


class RequestValidator:
    """Validates bulk document generation requests."""
    
    @staticmethod
    def validate_request_params(
        query: str,
        document_types: List[str],
        business_areas: List[str],
        max_documents: int = 100,
        priority: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate request parameters.
        
        Args:
            query: Query string
            document_types: List of document types
            business_areas: List of business areas
            max_documents: Maximum documents to generate
            priority: Priority level
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        is_valid, error = RequestValidator.validate_query(query)
        if not is_valid:
            return False, error
        
        is_valid, error = RequestValidator.validate_document_types(document_types)
        if not is_valid:
            return False, error
        
        is_valid, error = RequestValidator.validate_business_areas(business_areas)
        if not is_valid:
            return False, error
        
        if max_documents < MIN_DOCUMENTS:
            return False, f"max_documents must be at least {MIN_DOCUMENTS}"
        
        if max_documents > MAX_DOCUMENTS_LIMIT:
            return False, f"max_documents cannot exceed {MAX_DOCUMENTS_LIMIT}"
        
        if priority < MIN_PRIORITY or priority > MAX_PRIORITY:
            return False, f"Priority must be between {MIN_PRIORITY} and {MAX_PRIORITY}"
        
        return True, None
    
    @staticmethod
    def validate_document_types(document_types: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate document types list.
        
        Args:
            document_types: List of document types
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not document_types:
            return False, "Document types list cannot be empty"
        
        if len(document_types) > MAX_DOCUMENT_TYPES:
            return False, f"Cannot specify more than {MAX_DOCUMENT_TYPES} document types"
        
        for doc_type in document_types:
            if not doc_type or not doc_type.strip():
                return False, "Document types cannot contain empty strings"
            if len(doc_type) > MAX_DOCUMENT_TYPE_LENGTH:
                return False, f"Document type '{doc_type}' exceeds maximum length of {MAX_DOCUMENT_TYPE_LENGTH} characters"
        
        return True, None
    
    @staticmethod
    def validate_business_areas(business_areas: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate business areas list.
        
        Args:
            business_areas: List of business areas
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not business_areas:
            return False, "Business areas list cannot be empty"
        
        if len(business_areas) > MAX_BUSINESS_AREAS:
            return False, f"Cannot specify more than {MAX_BUSINESS_AREAS} business areas"
        
        for area in business_areas:
            if not area or not area.strip():
                return False, "Business areas cannot contain empty strings"
            if len(area) > MAX_BUSINESS_AREA_LENGTH:
                return False, f"Business area '{area}' exceeds maximum length of {MAX_BUSINESS_AREA_LENGTH} characters"
        
        return True, None
    
    @staticmethod
    def validate_query(query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate query string.
        
        Args:
            query: Query string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query:
            return False, "Query cannot be empty"
        
        if not query.strip():
            return False, "Query cannot be only whitespace"
        
        if len(query) < MIN_QUERY_LENGTH:
            return False, f"Query must be at least {MIN_QUERY_LENGTH} characters long"
        
        if len(query) > MAX_QUERY_LENGTH:
            return False, f"Query cannot exceed {MAX_QUERY_LENGTH} characters"
        
        return True, None

