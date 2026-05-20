"""Service functions - functional approach"""

from typing import Optional
from datetime import datetime, timedelta

from models.schemas import QuoteRequest, QuoteResponse
from repositories.quote_repository import QuoteRepository
from business_logic.quote_logic import (
    generate_quote_id,
    generate_request_id,
    create_quote_options,
)
from validators.quote_validators import validate_quote_request
from utils.logger import logger


async def create_quote(
    request: QuoteRequest,
    repository: QuoteRepository
) -> QuoteResponse:
    """Create a new freight quote - pure function"""
    validate_quote_request(request)
    
    quote_id = generate_quote_id()
    request_id = generate_request_id()
    options = create_quote_options(request)
    
    quote_response = QuoteResponse(
        quote_id=quote_id,
        request_id=request_id,
        origin=request.origin,
        destination=request.destination,
        cargo=request.cargo,
        options=options,
        valid_until=datetime.now() + timedelta(days=7),
        created_at=datetime.now()
    )
    
    await repository.save(quote_response)
    logger.info(f"Quote created: {quote_id}")
    
    return quote_response


async def get_quote(
    quote_id: str,
    repository: QuoteRepository
) -> Optional[QuoteResponse]:
    """Get quote by ID - pure function"""
    return await repository.find_by_id(quote_id)













