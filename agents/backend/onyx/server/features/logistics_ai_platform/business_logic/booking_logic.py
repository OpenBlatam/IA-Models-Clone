"""Pure functions for booking business logic"""

from datetime import datetime
import uuid

from models.schemas import QuoteResponse, QuoteOption


def generate_booking_id() -> str:
    """Generate a unique booking ID"""
    return f"BKG{str(uuid.uuid4())[:8].upper()}"


def generate_booking_reference(booking_id: str) -> str:
    """Generate booking reference"""
    date_str = datetime.now().strftime('%Y%m%d')
    return f"BK-{date_str}-{booking_id[-6:]}"


def validate_quote_option(
    quote: QuoteResponse,
    option_id: str
) -> QuoteOption:
    """Validate and return quote option"""
    for option in quote.options:
        if option.quote_id == option_id:
            return option
    
    raise ValueError(f"Quote option {option_id} not found in quote {quote.quote_id}")













