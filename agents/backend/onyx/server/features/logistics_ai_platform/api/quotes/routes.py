"""
Quote routes

This module provides API endpoints for quote management including:
- Quote creation
- Quote retrieval
- Quote validation
"""

from typing import Annotated
from fastapi import APIRouter, Depends, status

from models.schemas import QuoteRequest, QuoteResponse
from handlers.quote_handlers import handle_create_quote, handle_get_quote
from utils.dependencies import get_quote_repository
from utils.exceptions import ValidationError
from repositories.quote_repository import QuoteRepository

router = APIRouter(
    prefix="/quotes",
    tags=["Quotes"],
    responses={
        404: {"description": "Quote not found"},
        422: {"description": "Validation error"},
        400: {"description": "Business logic error"}
    }
)


QuoteRepositoryDep = Annotated[QuoteRepository, Depends(get_quote_repository)]


@router.post(
    "",
    response_model=QuoteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new freight quote",
    description="Creates a new freight quote based on origin, destination, and cargo details",
    response_description="Created quote with multiple transportation options",
    responses={
        201: {
            "description": "Quote created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "quote_id": "Q12345678",
                        "request_id": "REQ12345678",
                        "origin": {
                            "country": "Mexico",
                            "city": "Veracruz",
                            "port_code": "MXVER"
                        },
                        "destination": {
                            "country": "Honduras",
                            "city": "Comayagua",
                            "port_code": "HNCMY"
                        },
                        "cargo": {
                            "description": "Electronics",
                            "weight_kg": 1000,
                            "volume_m3": 5.0,
                            "quantity": 10,
                            "unit_type": "CTN",
                            "value_usd": 50000
                        },
                        "transportation_mode": "maritime",
                        "options": [
                            {
                                "quote_id": "Q12345678",
                                "transportation_mode": "maritime",
                                "carrier": "Maersk",
                                "estimated_departure": "2024-01-15T10:00:00Z",
                                "estimated_arrival": "2024-01-25T14:00:00Z",
                                "transit_days": 10,
                                "price_usd": 2500.00,
                                "currency": "USD",
                                "service_level": "standard"
                            }
                        ],
                        "created_at": "2024-01-10T12:00:00Z",
                        "expires_at": "2024-01-17T12:00:00Z"
                    }
                }
            }
        }
    }
)
async def create_quote(
    request: QuoteRequest,
    repository: QuoteRepositoryDep
) -> QuoteResponse:
    """
    Create a new freight quote
    
    Args:
        request: Quote creation request with origin, destination, and cargo details
        repository: Injected quote repository
        
    Returns:
        QuoteResponse: Created quote with options and pricing
        
    Raises:
        ValidationError: If request data is invalid
        BusinessLogicError: If quote cannot be created
    """
    return await handle_create_quote(request, repository)


@router.get(
    "/{quote_id}",
    response_model=QuoteResponse,
    summary="Get quote by ID",
    description="Retrieves a quote by its unique identifier"
)
async def get_quote(
    quote_id: str,
    repository: QuoteRepositoryDep
) -> QuoteResponse:
    """
    Get quote by ID
    
    Args:
        quote_id: Unique quote identifier
        repository: Injected quote repository
        
    Returns:
        QuoteResponse: Quote details
        
    Raises:
        NotFoundError: If quote not found
        ValidationError: If quote_id is invalid
    """
    if not quote_id or not quote_id.strip():
        raise ValidationError("Quote ID is required", field="quote_id")
    
    return await handle_get_quote(quote_id, repository)

