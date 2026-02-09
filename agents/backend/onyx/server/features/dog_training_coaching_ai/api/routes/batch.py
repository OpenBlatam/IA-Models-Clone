"""
Batch Processing Endpoints
==========================
Endpoints para procesamiento por lotes.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from pydantic import BaseModel

from ...api.dependencies import get_coaching_service
from ...schemas import CoachingRequest
from ...utils.batch_processing import BatchProcessor
from ...utils.logger import get_logger
from ...utils.rate_limiter import limiter

router = APIRouter(prefix="/api/v1/batch", tags=["batch"])
logger = get_logger(__name__)


class BatchCoachingRequest(BaseModel):
    """Request para procesamiento por lotes."""
    requests: List[CoachingRequest]
    batch_size: int = 10
    max_concurrent: int = 5


@router.post("/coach")
@limiter.limit("5/minute")
async def batch_coaching(
    request: BatchCoachingRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
) -> Dict[str, Any]:
    """
    Procesar múltiples requests de coaching en batch.
    
    Args:
        request: Request con lista de coaching requests
        service: Servicio de coaching
        
    Returns:
        Resultados del procesamiento batch
    """
    try:
        processor = BatchProcessor(
            batch_size=request.batch_size,
            max_concurrent=request.max_concurrent
        )
        
        async def process_coaching(coaching_req: CoachingRequest):
            return await service.get_coaching_advice(
                question=coaching_req.question,
                dog_breed=coaching_req.dog_breed,
                dog_age=coaching_req.dog_age,
                dog_size=coaching_req.dog_size,
                training_goal=coaching_req.training_goal,
                experience_level=coaching_req.experience_level,
                previous_context=coaching_req.previous_context,
                specific_issues=coaching_req.specific_issues
            )
        
        result = await processor.process(
            request.requests,
            process_coaching
        )
        
        return {
            "success": True,
            "total": len(request.requests),
            "processed": result["stats"]["processed"],
            "failed": result["stats"]["failed"],
            "results": result["results"],
            "errors": result["errors"],
            "stats": result["stats"]
        }
        
    except Exception as e:
        logger.error(f"Error in batch coaching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

