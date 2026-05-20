"""Invoice routes"""

from fastapi import APIRouter
from typing import List

from models.schemas import (
    InvoiceRequest,
    InvoiceResponse,
)
from utils.dependencies import InvoiceServiceDep
from utils.exceptions import NotFoundError

router = APIRouter(prefix="/invoices", tags=["Invoices"])


@router.post("", response_model=InvoiceResponse, status_code=201)
async def create_invoice(
    request: InvoiceRequest,
    invoice_service: InvoiceServiceDep
) -> InvoiceResponse:
    """Create a new invoice"""
    invoice = await invoice_service.create_invoice(request)
    return invoice


@router.get("", response_model=List[InvoiceResponse])
async def get_invoices(
    limit: int = 100,
    offset: int = 0,
    invoice_service: InvoiceServiceDep = None
) -> List[InvoiceResponse]:
    """Get all invoices"""
    if limit > 1000:
        limit = 1000  # Enforce max limit
    
    invoices = await invoice_service.get_all_invoices(limit=limit, offset=offset)
    return invoices


@router.get("/{invoice_id}", response_model=InvoiceResponse)
async def get_invoice(
    invoice_id: str,
    invoice_service: InvoiceServiceDep
) -> InvoiceResponse:
    """Get invoice by ID"""
    invoice = await invoice_service.get_invoice(invoice_id)
    if not invoice:
        raise NotFoundError("Invoice", invoice_id)
    return invoice


@router.get("/shipment/{shipment_id}", response_model=List[InvoiceResponse])
async def get_invoices_by_shipment(
    shipment_id: str,
    invoice_service: InvoiceServiceDep
) -> List[InvoiceResponse]:
    """Get invoices for a shipment"""
    invoices = await invoice_service.get_invoices_by_shipment(shipment_id)
    return invoices

