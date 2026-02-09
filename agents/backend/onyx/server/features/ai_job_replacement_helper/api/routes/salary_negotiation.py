"""
Salary Negotiation endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.salary_negotiation import SalaryNegotiationService, SalaryOffer

router = APIRouter()
negotiation_service = SalaryNegotiationService()


@router.post("/start/{user_id}")
async def start_negotiation(
    user_id: str,
    job_title: str,
    company: str,
    initial_offer: Dict[str, Any],
    target_salary: float
) -> Dict[str, Any]:
    """Iniciar sesión de negociación"""
    try:
        offer = SalaryOffer(
            base_salary=initial_offer.get("base_salary", 0),
            bonus=initial_offer.get("bonus"),
            equity=initial_offer.get("equity"),
            benefits=initial_offer.get("benefits", []),
        )
        
        session = negotiation_service.start_negotiation(
            user_id, job_title, company, offer, target_salary
        )
        
        return {
            "session_id": session.id,
            "initial_offer": {
                "base_salary": session.initial_offer.base_salary,
                "total_compensation": session.initial_offer.total_compensation,
            },
            "target_salary": session.target_salary,
            "current_stage": session.current_stage.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategy/{session_id}")
async def get_negotiation_strategy(session_id: str) -> Dict[str, Any]:
    """Obtener estrategia de negociación"""
    try:
        strategy = negotiation_service.get_negotiation_strategy(session_id)
        return strategy
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate/{session_id}")
async def simulate_counter_offer(
    session_id: str,
    counter_amount: float
) -> Dict[str, Any]:
    """Simular contraoferta"""
    try:
        result = negotiation_service.simulate_counter_offer(session_id, counter_amount)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tips")
async def get_negotiation_tips(
    job_title: str,
    location: str
) -> Dict[str, Any]:
    """Obtener tips de negociación"""
    try:
        tips = negotiation_service.get_negotiation_tips(job_title, location)
        return {"tips": tips}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




