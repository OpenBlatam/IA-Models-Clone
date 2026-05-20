"""Insurance routes"""

from fastapi import APIRouter

from models.schemas import (
    InsuranceRequest,
    InsuranceResponse,
)
from utils.dependencies import InsuranceServiceDep
from utils.exceptions import NotFoundError

router = APIRouter(prefix="/insurance", tags=["Insurance"])


@router.post("", response_model=InsuranceResponse, status_code=201)
async def create_insurance(
    request: InsuranceRequest,
    insurance_service: InsuranceServiceDep
) -> InsuranceResponse:
    """Create a new insurance policy"""
    insurance = await insurance_service.create_insurance(request)
    return insurance


@router.get("/{insurance_id}", response_model=InsuranceResponse)
async def get_insurance(
    insurance_id: str,
    insurance_service: InsuranceServiceDep
) -> InsuranceResponse:
    """Get insurance by ID"""
    insurance = await insurance_service.get_insurance(insurance_id)
    if not insurance:
        raise NotFoundError("Insurance", insurance_id)
    return insurance


@router.get("/shipment/{shipment_id}", response_model=InsuranceResponse)
async def get_insurance_by_shipment(
    shipment_id: str,
    insurance_service: InsuranceServiceDep
) -> InsuranceResponse:
    """Get insurance for a shipment"""
    insurance = await insurance_service.get_insurance_by_shipment(shipment_id)
    if not insurance:
        raise NotFoundError("Insurance", shipment_id)
    return insurance

