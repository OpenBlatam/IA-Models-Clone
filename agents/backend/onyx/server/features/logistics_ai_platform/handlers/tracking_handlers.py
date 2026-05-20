"""Tracking request handlers"""

from typing import Dict, Any

from repositories.shipment_repository import ShipmentRepository
from repositories.container_repository import ContainerRepository
from domain.tracking import get_public_tracking_info
from utils.exceptions import NotFoundError
from utils.cache.helpers import get_cached_or_fetch
from utils.logger import logger
from utils.constants import TRACKING_CACHE_TTL


async def handle_public_tracking(
    identifier: str,
    shipment_repo: ShipmentRepository,
    container_repo: ContainerRepository
) -> Dict[str, Any]:
    """
    Handle public tracking request - no authentication required
    
    This handler allows anyone to track shipments or containers using
    various identifiers without requiring an account.
    """
    if not identifier or not identifier.strip():
        raise NotFoundError("Tracking identifier", identifier)
    
    identifier = identifier.strip().upper()
    cache_key = f"public_tracking:{identifier}"
    
    tracking_info = await get_cached_or_fetch(
        cache_key,
        lambda: get_public_tracking_info(
            identifier=identifier,
            shipment_repo=shipment_repo,
            container_repo=container_repo
        ),
        ttl=TRACKING_CACHE_TTL
    )
    
    if not tracking_info.get("found"):
        raise NotFoundError("Shipment or Container", identifier)
    
    logger.info(f"Public tracking request for {identifier} - Found: {tracking_info.get('type')}")
    return tracking_info

