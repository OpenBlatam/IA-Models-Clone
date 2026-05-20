"""
Dependency injection for services

This module provides FastAPI dependency injection functions for all
services and repositories in the application. Uses lru_cache for
singleton pattern and FastAPI's Depends for dependency resolution.
"""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from repositories import (
    QuoteRepository,
    BookingRepository,
    ShipmentRepository,
    ContainerRepository,
)
from core.quote_service import QuoteService
from core.booking_service import BookingService
from core.shipment_service import ShipmentService
from core.container_service import ContainerService
from core.tracking_service import TrackingService
from services.invoice_service import InvoiceService
from services.document_service import DocumentService
from services.alert_service import AlertService
from services.insurance_service import InsuranceService

logger = logging.getLogger(__name__)


# Repository singletons
@lru_cache()
def get_quote_repository() -> QuoteRepository:
    """
    Get quote repository instance (singleton)
    
    Returns:
        QuoteRepository: Singleton instance of quote repository
    """
    logger.debug("Creating QuoteRepository instance")
    return QuoteRepository()


@lru_cache()
def get_booking_repository() -> BookingRepository:
    """
    Get booking repository instance (singleton)
    
    Returns:
        BookingRepository: Singleton instance of booking repository
    """
    logger.debug("Creating BookingRepository instance")
    return BookingRepository()


@lru_cache()
def get_shipment_repository() -> ShipmentRepository:
    """
    Get shipment repository instance (singleton)
    
    Returns:
        ShipmentRepository: Singleton instance of shipment repository
    """
    logger.debug("Creating ShipmentRepository instance")
    return ShipmentRepository()


@lru_cache()
def get_container_repository() -> ContainerRepository:
    """
    Get container repository instance (singleton)
    
    Returns:
        ContainerRepository: Singleton instance of container repository
    """
    logger.debug("Creating ContainerRepository instance")
    return ContainerRepository()


# Service singletons - kept for backward compatibility
# New code should use domain functions and handlers directly
@lru_cache()
def get_quote_service(
    repository: Annotated[QuoteRepository, Depends(get_quote_repository)]
) -> QuoteService:
    """
    Get quote service instance (singleton)
    
    Args:
        repository: Quote repository dependency
        
    Returns:
        QuoteService: Singleton instance of quote service
        
    Note:
        Legacy function - new code should use domain functions instead
    """
    logger.debug("Creating QuoteService instance")
    return QuoteService(repository)


@lru_cache()
def get_shipment_service(
    repository: Annotated[ShipmentRepository, Depends(get_shipment_repository)]
) -> ShipmentService:
    """
    Get shipment service instance (singleton)
    
    Args:
        repository: Shipment repository dependency
        
    Returns:
        ShipmentService: Singleton instance of shipment service
        
    Note:
        Legacy function - new code should use domain functions instead
    """
    logger.debug("Creating ShipmentService instance")
    return ShipmentService(repository)


@lru_cache()
def get_container_service(
    repository: Annotated[ContainerRepository, Depends(get_container_repository)]
) -> ContainerService:
    """
    Get container service instance (singleton)
    
    Args:
        repository: Container repository dependency
        
    Returns:
        ContainerService: Singleton instance of container service
    """
    logger.debug("Creating ContainerService instance")
    return ContainerService(repository)


# Services with dependencies - FastAPI handles singleton behavior
def get_booking_service(
    repository: Annotated[BookingRepository, Depends(get_booking_repository)],
    quote_service: Annotated[QuoteService, Depends(get_quote_service)],
    shipment_service: Annotated[ShipmentService, Depends(get_shipment_service)]
) -> BookingService:
    """
    Get booking service instance
    
    Args:
        repository: Booking repository dependency
        quote_service: Quote service dependency
        shipment_service: Shipment service dependency
        
    Returns:
        BookingService: Instance of booking service
        
    Note:
        Legacy function - new code should use domain functions instead.
        FastAPI handles singleton behavior through dependency injection.
    """
    logger.debug("Creating BookingService instance")
    return BookingService(repository, quote_service, shipment_service)


def get_tracking_service(
    shipment_service: Annotated[ShipmentService, Depends(get_shipment_service)],
    container_service: Annotated[ContainerService, Depends(get_container_service)]
) -> TrackingService:
    """
    Get tracking service instance
    
    Args:
        shipment_service: Shipment service dependency
        container_service: Container service dependency
        
    Returns:
        TrackingService: Instance of tracking service
        
    Note:
        FastAPI handles singleton behavior through dependency injection.
    """
    logger.debug("Creating TrackingService instance")
    return TrackingService(shipment_service, container_service)


@lru_cache()
def get_invoice_service() -> InvoiceService:
    """
    Get invoice service instance (singleton)
    
    Returns:
        InvoiceService: Singleton instance of invoice service
    """
    logger.debug("Creating InvoiceService instance")
    return InvoiceService()


@lru_cache()
def get_document_service() -> DocumentService:
    """
    Get document service instance (singleton)
    
    Returns:
        DocumentService: Singleton instance of document service
    """
    logger.debug("Creating DocumentService instance")
    return DocumentService()


@lru_cache()
def get_alert_service() -> AlertService:
    """
    Get alert service instance (singleton)
    
    Returns:
        AlertService: Singleton instance of alert service
    """
    logger.debug("Creating AlertService instance")
    return AlertService()


@lru_cache()
def get_insurance_service() -> InsuranceService:
    """
    Get insurance service instance (singleton)
    
    Returns:
        InsuranceService: Singleton instance of insurance service
    """
    logger.debug("Creating InsuranceService instance")
    return InsuranceService()


# Type aliases for dependency injection
# These can be used as type hints in route handlers for automatic dependency injection
QuoteServiceDep = Annotated[QuoteService, Depends(get_quote_service)]
ShipmentServiceDep = Annotated[ShipmentService, Depends(get_shipment_service)]
ContainerServiceDep = Annotated[ContainerService, Depends(get_container_service)]
BookingServiceDep = Annotated[BookingService, Depends(get_booking_service)]
TrackingServiceDep = Annotated[TrackingService, Depends(get_tracking_service)]
InvoiceServiceDep = Annotated[InvoiceService, Depends(get_invoice_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
AlertServiceDep = Annotated[AlertService, Depends(get_alert_service)]
InsuranceServiceDep = Annotated[InsuranceService, Depends(get_insurance_service)]

