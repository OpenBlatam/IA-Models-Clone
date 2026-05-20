"""Quote request handlers"""

from models.schemas import QuoteRequest, QuoteResponse
from repositories.quote_repository import QuoteRepository
from domain.quotes import create_quote_domain, get_quote_domain
from utils.handler_helpers import (
    get_entity_or_raise,
    create_entity_with_cache
)
from utils.constants import DEFAULT_CACHE_TTL
from utils.metrics import get_metrics_collector


async def handle_create_quote(
    request: QuoteRequest,
    repository: QuoteRepository
) -> QuoteResponse:
    """Handle quote creation request"""
    quote = await create_quote_domain(request, repository)
    result = await create_entity_with_cache(
        quote,
        f"quote:{quote.quote_id}",
        cache_ttl=DEFAULT_CACHE_TTL
    )
    # Record business metric
    get_metrics_collector().record_quote_created()
    return result


async def handle_get_quote(
    quote_id: str,
    repository: QuoteRepository
) -> QuoteResponse:
    """Handle get quote request"""
    return await get_entity_or_raise(
        quote_id,
        lambda: get_quote_domain(quote_id, repository),
        "Quote",
        cache_key=f"quote:{quote_id}",
        cache_ttl=DEFAULT_CACHE_TTL,
        model_class=QuoteResponse
    )


