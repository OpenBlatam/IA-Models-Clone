"""Invoice service for managing invoices"""

from typing import List, Optional
from datetime import datetime, timedelta
import uuid
import logging

from models.schemas import (
    InvoiceRequest,
    InvoiceResponse,
    InvoiceLineItem,
)

logger = logging.getLogger(__name__)


class InvoiceService:
    """Service for managing invoices"""
    
    def __init__(self):
        """Initialize invoice service"""
        self.invoices = {}  # In-memory storage
    
    async def create_invoice(self, request: InvoiceRequest) -> InvoiceResponse:
        """Create a new invoice"""
        try:
            invoice_id = f"INV{str(uuid.uuid4())[:8].upper()}"
            invoice_number = request.invoice_number or f"INV-{datetime.now().strftime('%Y%m%d')}-{invoice_id[-6:]}"
            
            # Generate line items (simplified - in production, calculate from shipment)
            line_items = [
                InvoiceLineItem(
                    description="Freight charges",
                    quantity=1.0,
                    unit_price=1000.0,
                    total=1000.0,
                    tax_rate=0.0
                ),
                InvoiceLineItem(
                    description="Insurance",
                    quantity=1.0,
                    unit_price=50.0,
                    total=50.0,
                    tax_rate=0.0
                )
            ]
            
            subtotal = sum(item.total for item in line_items)
            tax = subtotal * 0.16  # 16% tax (example)
            total = subtotal + tax
            
            invoice = InvoiceResponse(
                invoice_id=invoice_id,
                invoice_number=invoice_number,
                shipment_id=request.shipment_id,
                issue_date=datetime.now(),
                due_date=request.due_date or datetime.now() + timedelta(days=30),
                subtotal=subtotal,
                tax=tax,
                total=total,
                currency="USD",
                line_items=line_items,
                status="pending",
                created_at=datetime.now()
            )
            
            self.invoices[invoice_id] = invoice
            
            logger.info(f"Invoice created: {invoice_id}")
            return invoice
            
        except Exception as e:
            logger.error(f"Error creating invoice: {str(e)}")
            raise
    
    async def get_invoice(self, invoice_id: str) -> Optional[InvoiceResponse]:
        """Get invoice by ID"""
        return self.invoices.get(invoice_id)
    
    async def get_invoices_by_shipment(
        self,
        shipment_id: str
    ) -> List[InvoiceResponse]:
        """Get invoices for a shipment"""
        return [
            invoice for invoice in self.invoices.values()
            if invoice.shipment_id == shipment_id
        ]
    
    async def get_all_invoices(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[InvoiceResponse]:
        """Get all invoices"""
        invoices = list(self.invoices.values())
        invoices.sort(key=lambda x: x.created_at, reverse=True)
        return invoices[offset:offset + limit]













