"""
Quote service for generating freight quotes

This service provides business logic for quote management including:
- Quote creation and validation
- Quote retrieval and expiration handling
- Quote option management
"""

from typing import Optional, List
from datetime import datetime, timedelta
import logging

from models.schemas import QuoteRequest, QuoteResponse
from repositories.quote_repository import QuoteRepository
from business_logic.quote_logic import (
    generate_quote_id,
    generate_request_id,
    create_quote_options,
)
from validators.quote_validators import validate_quote_request
from utils.exceptions import NotFoundError, ValidationError, BusinessLogicError

logger = logging.getLogger(__name__)


class QuoteService:
    """
    Service for managing freight quotes
    
    Provides business logic for quote operations including
    creation, validation, retrieval, and expiration management.
    """
    
    def __init__(self, repository: QuoteRepository):
        """
        Initialize quote service
        
        Args:
            repository: Quote repository for data access
        """
        self.repository = repository
        logger.info("QuoteService initialized")
    
    async def create_quote(self, request: QuoteRequest) -> QuoteResponse:
        """
        Create a new freight quote
        
        Args:
            request: Quote creation request
            
        Returns:
            QuoteResponse: Created quote
            
        Raises:
            ValidationError: If request data is invalid
            BusinessLogicError: If quote cannot be created
        """
        # Validate request
        try:
            validate_quote_request(request)
        except ValueError as e:
            raise ValidationError(str(e))
        
        # Generate IDs
        quote_id = generate_quote_id()
        request_id = generate_request_id()
        
        # Create quote options
        try:
            options = create_quote_options(request)
        except Exception as e:
            logger.error(f"Error creating quote options: {e}", exc_info=True)
            raise BusinessLogicError(f"Failed to create quote options: {str(e)}")
        
        if not options:
            raise BusinessLogicError("No quote options available for the given request")
        
        # Create quote response
        now = datetime.now()
        quote_response = QuoteResponse(
            quote_id=quote_id,
            request_id=request_id,
            origin=request.origin,
            destination=request.destination,
            cargo=request.cargo,
            options=options,
            valid_until=now + timedelta(days=7),
            created_at=now
        )
        
        try:
            await self.repository.save(quote_response)
            logger.info(
                f"Quote created: {quote_id} "
                f"(request_id: {request_id}, options: {len(options)})"
            )
        except Exception as e:
            logger.error(f"Error saving quote: {e}", exc_info=True)
            raise BusinessLogicError(f"Failed to save quote: {str(e)}")
        
        return quote_response
    
    async def get_quote(self, quote_id: str) -> Optional[QuoteResponse]:
        """
        Get quote by ID
        
        Args:
            quote_id: Quote identifier
            
        Returns:
            QuoteResponse if found, None otherwise
        """
        if not quote_id:
            raise ValidationError("Quote ID is required", field="quote_id")
        
        quote = await self.repository.find_by_id(quote_id)
        
        if not quote:
            logger.debug(f"Quote not found: {quote_id}")
        elif self._is_quote_expired(quote):
            logger.warning(f"Quote {quote_id} has expired")
        
        return quote
    
    async def get_quote_or_raise(self, quote_id: str) -> QuoteResponse:
        """
        Get quote by ID or raise NotFoundError
        
        Args:
            quote_id: Quote identifier
            
        Returns:
            QuoteResponse: Found quote
            
        Raises:
            NotFoundError: If quote not found
            BusinessLogicError: If quote is expired
        """
        quote = await self.get_quote(quote_id)
        if not quote:
            raise NotFoundError("Quote", quote_id)
        
        if self._is_quote_expired(quote):
            raise BusinessLogicError(
                f"Quote {quote_id} has expired",
                error_code="QUOTE_EXPIRED"
            )
        
        return quote
    
    async def get_quotes_by_request_id(
        self,
        request_id: str
    ) -> List[QuoteResponse]:
        """
        Get quotes by request ID
        
        Args:
            request_id: Request identifier
            
        Returns:
            List of quotes for the request
        """
        if not request_id:
            raise ValidationError("Request ID is required", field="request_id")
        
        quotes = await self.repository.find_by_request_id(request_id)
        logger.debug(f"Found {len(quotes)} quotes for request {request_id}")
        
        return quotes
    
    def _is_quote_expired(self, quote: QuoteResponse) -> bool:
        """
        Check if quote has expired
        
        Args:
            quote: Quote to check
            
        Returns:
            True if expired, False otherwise
        """
        if not quote.valid_until:
            return False
        
        return datetime.now() > quote.valid_until

