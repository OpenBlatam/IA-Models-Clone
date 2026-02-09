"""
Salary Benchmarking endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.salary_benchmarking import SalaryBenchmarkingService

router = APIRouter()
benchmarking_service = SalaryBenchmarkingService()


@router.post("/benchmark")
async def benchmark_salary(
    role: str,
    location: str,
    experience_years: int,
    current_salary: float,
    company_size: Optional[str] = None,
    industry: Optional[str] = None
) -> Dict[str, Any]:
    """Comparar salario con mercado"""
    try:
        comparison = benchmarking_service.benchmark_salary(
            role, location, experience_years, current_salary, company_size, industry
        )
        return {
            "user_salary": comparison.user_salary,
            "market_data": {
                "median": comparison.market_data.salary_median,
                "min": comparison.market_data.salary_min,
                "max": comparison.market_data.salary_max,
            },
            "percentile": comparison.percentile,
            "recommendation": comparison.recommendation,
            "negotiation_power": comparison.negotiation_power,
            "factors": comparison.factors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-roles")
async def compare_multiple_roles(
    roles: List[str],
    location: str,
    experience_years: int
) -> Dict[str, Any]:
    """Comparar múltiples roles"""
    try:
        comparison = benchmarking_service.compare_multiple_roles(roles, location, experience_years)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




