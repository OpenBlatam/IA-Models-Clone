"""Quote domain logic - pure functions"""

from typing import Optional

from models.schemas import QuoteRequest, QuoteResponse
from repositories.quote_repository import QuoteRepository
from validators.quote_validators import validate_quote_request
from factories.quote_factory import build_quote_response
from utils.logger import logger


async def create_quote_domain(
    request: QuoteRequest,
    repository: QuoteRepository
) -> QuoteResponse:
    """Create quote - pure domain function"""
    validate_quote_request(request)
    
    quote = build_quote_response(request)
    await repository.save(quote)
    
    logger.info(f"Quote created: {quote.quote_id}")
    return quote


async def get_quote_domain(
    quote_id: str,
    repository: QuoteRepository
) -> Optional[QuoteResponse]:
    """Get quote - pure domain function"""
    return await repository.find_by_id(quote_id)

