"""Factory functions for quote objects"""

from datetime import datetime, timedelta

from models.schemas import QuoteRequest, QuoteResponse
from business_logic.quote_logic import (
    generate_quote_id,
    generate_request_id,
    create_quote_options,
)


def build_quote_response(request: QuoteRequest) -> QuoteResponse:
    """Build quote response from request - pure factory function"""
    quote_id = generate_quote_id()
    request_id = generate_request_id()
    options = create_quote_options(request)
    
    return QuoteResponse(
        quote_id=quote_id,
        request_id=request_id,
        origin=request.origin,
        destination=request.destination,
        cargo=request.cargo,
        options=options,
        valid_until=datetime.now() + timedelta(days=7),
        created_at=datetime.now()
    )









