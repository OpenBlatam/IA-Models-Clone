"""Insurance service for managing cargo and container insurance"""

from typing import Optional
from datetime import datetime, timedelta
import uuid
import logging

from models.schemas import (
    InsuranceRequest,
    InsuranceResponse,
)

logger = logging.getLogger(__name__)


class InsuranceService:
    """Service for managing insurance"""
    
    def __init__(self):
        """Initialize insurance service"""
        self.policies = {}  # In-memory storage
    
    async def create_insurance(self, request: InsuranceRequest) -> InsuranceResponse:
        """Create a new insurance policy"""
        try:
            insurance_id = f"INS{str(uuid.uuid4())[:8].upper()}"
            policy_number = f"POL-{datetime.now().strftime('%Y%m%d')}-{insurance_id[-6:]}"
            
            # Calculate premium (simplified - 2% of coverage amount)
            premium = request.coverage_amount_usd * 0.02
            
            insurance = InsuranceResponse(
                insurance_id=insurance_id,
                shipment_id=request.shipment_id,
                coverage_type=request.coverage_type,
                coverage_amount_usd=request.coverage_amount_usd,
                premium_usd=premium,
                deductible_usd=request.deductible_usd or 0.0,
                policy_number=policy_number,
                status="active",
                valid_from=datetime.now(),
                valid_until=datetime.now() + timedelta(days=365),
                created_at=datetime.now()
            )
            
            self.policies[insurance_id] = insurance
            
            logger.info(f"Insurance created: {insurance_id}")
            return insurance
            
        except Exception as e:
            logger.error(f"Error creating insurance: {str(e)}")
            raise
    
    async def get_insurance(self, insurance_id: str) -> Optional[InsuranceResponse]:
        """Get insurance by ID"""
        return self.policies.get(insurance_id)
    
    async def get_insurance_by_shipment(
        self,
        shipment_id: str
    ) -> Optional[InsuranceResponse]:
        """Get insurance for a shipment"""
        for policy in self.policies.values():
            if policy.shipment_id == shipment_id:
                return policy
        return None













