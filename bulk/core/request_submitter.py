"""
Request Submitter - Common request submission utilities
======================================================

Provides common utilities for submitting bulk document generation requests.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable, Type, Tuple

from .request_validator import RequestValidator

logger = logging.getLogger(__name__)


class RequestSubmitter:
    """Common utilities for submitting bulk requests."""
    
    @staticmethod
    def validate_and_prepare_request(
        query: str,
        document_types: List[str],
        business_areas: List[str],
        max_documents: int = 100,
        priority: int = 1
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate request parameters and generate request ID.
        
        Args:
            query: Query string
            document_types: List of document types
            business_areas: List of business areas
            max_documents: Maximum documents to generate
            priority: Priority level
            
        Returns:
            Tuple of (is_valid, error_message, request_id)
            If valid, request_id will be a UUID string, otherwise None
        """
        is_valid, error_msg = RequestValidator.validate_request_params(
            query=query,
            document_types=document_types,
            business_areas=business_areas,
            max_documents=max_documents,
            priority=priority
        )
        
        if not is_valid:
            return False, error_msg, None
        
        request_id = str(uuid.uuid4())
        return True, None, request_id
    
    @staticmethod
    async def register_request_and_start_processing(
        request: Any,
        active_requests: Dict[str, Any],
        stats_updater: Optional[Callable] = None,
        task_creator: Optional[Callable] = None,
        processor: Any = None,
        start_processing_func: Optional[Callable] = None
    ) -> None:
        """
        Register a request and start processing if needed.
        
        Args:
            request: The request object to register
            active_requests: Dictionary to store active requests
            stats_updater: Optional function to update statistics
            task_creator: Optional async function to create initial tasks
            processor: Optional processor instance (for checking is_running)
            start_processing_func: Optional async function to start processing
        """
        active_requests[request.id] = request
        
        if stats_updater:
            stats_updater()
        
        if task_creator:
            await task_creator(request)
        
        if processor and start_processing_func:
            if not processor.is_running:
                asyncio.create_task(start_processing_func())
    
    @staticmethod
    def create_request_object(
        request_class: Type,
        request_id: str,
        query: str,
        document_types: List[str],
        business_areas: List[str],
        max_documents: int = 100,
        continuous_mode: bool = True,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        **extra_kwargs
    ) -> Any:
        """
        Create a request object with common parameters.
        
        Args:
            request_class: The request dataclass class
            request_id: Unique request identifier
            query: Query string
            document_types: List of document types
            business_areas: List of business areas
            max_documents: Maximum documents to generate
            continuous_mode: Whether to continue generating
            priority: Priority level
            metadata: Optional metadata dictionary
            **extra_kwargs: Additional keyword arguments for enhanced requests
            
        Returns:
            Instance of request_class
        """
        return request_class(
            id=request_id,
            query=query,
            document_types=document_types,
            business_areas=business_areas,
            max_documents=max_documents,
            continuous_mode=continuous_mode,
            priority=priority,
            metadata=metadata or {},
            **extra_kwargs
        )

