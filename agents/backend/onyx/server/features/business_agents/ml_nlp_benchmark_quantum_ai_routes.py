"""
ML NLP Benchmark Quantum AI Routes
API routes for quantum AI system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ml_nlp_benchmark_quantum_ai import (
    get_quantum_ai,
    create_quantum_ai,
    train_quantum_ai,
    predict_quantum_ai,
    quantum_reasoning,
    quantum_learning,
    quantum_creativity,
    quantum_consciousness,
    get_quantum_ai_summary,
    clear_quantum_ai_data
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/quantum-ai", tags=["Quantum AI"])

@router.post("/ais")
async def create_quantum_ai_endpoint(
    name: str,
    ai_type: str,
    quantum_circuit: Dict[str, Any],
    ai_capabilities: List[str],
    parameters: Optional[Dict[str, Any]] = None
):
    """Create a quantum AI"""
    try:
        ai_id = create_quantum_ai(name, ai_type, quantum_circuit, ai_capabilities, parameters)
        return {"ai_id": ai_id, "message": "Quantum AI created successfully"}
    except Exception as e:
        logger.error(f"Error creating quantum AI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ais/{ai_id}/train")
async def train_quantum_ai_endpoint(
    ai_id: str,
    training_data: List[Dict[str, Any]],
    validation_data: Optional[List[Dict[str, Any]]] = None
):
    """Train a quantum AI"""
    try:
        result = train_quantum_ai(ai_id, training_data, validation_data)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error training quantum AI {ai_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ais/{ai_id}/predict")
async def predict_quantum_ai_endpoint(
    ai_id: str,
    input_data: Any
):
    """Make predictions with quantum AI"""
    try:
        result = predict_quantum_ai(ai_id, input_data)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_ai_results": result.quantum_ai_results,
            "ai_intelligence": result.ai_intelligence,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error predicting with quantum AI {ai_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reasoning")
async def quantum_reasoning_endpoint(
    reasoning_data: Dict[str, Any],
    reasoning_type: str = "logical"
):
    """Perform quantum reasoning"""
    try:
        result = quantum_reasoning(reasoning_data, reasoning_type)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_ai_results": result.quantum_ai_results,
            "ai_intelligence": result.ai_intelligence,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum reasoning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/learning")
async def quantum_learning_endpoint(
    learning_data: List[Dict[str, Any]],
    learning_type: str = "supervised"
):
    """Perform quantum learning"""
    try:
        result = quantum_learning(learning_data, learning_type)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_ai_results": result.quantum_ai_results,
            "ai_intelligence": result.ai_intelligence,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/creativity")
async def quantum_creativity_endpoint(
    creativity_data: Dict[str, Any],
    creativity_type: str = "artistic"
):
    """Perform quantum creativity"""
    try:
        result = quantum_creativity(creativity_data, creativity_type)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_ai_results": result.quantum_ai_results,
            "ai_intelligence": result.ai_intelligence,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum creativity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/consciousness")
async def quantum_consciousness_endpoint(
    consciousness_data: Dict[str, Any],
    consciousness_level: str = "basic"
):
    """Perform quantum consciousness"""
    try:
        result = quantum_consciousness(consciousness_data, consciousness_level)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_ai_results": result.quantum_ai_results,
            "ai_intelligence": result.ai_intelligence,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum consciousness: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ais")
async def list_quantum_ais(
    ai_type: Optional[str] = None,
    active_only: bool = False
):
    """List quantum AIs"""
    try:
        quantum_ai_system = get_quantum_ai()
        ais = quantum_ai_system.list_quantum_ais(ai_type, active_only)
        return {
            "quantum_ais": [
                {
                    "ai_id": ai.ai_id,
                    "name": ai.name,
                    "ai_type": ai.ai_type,
                    "ai_capabilities": ai.ai_capabilities,
                    "is_active": ai.is_active,
                    "created_at": ai.created_at.isoformat(),
                    "last_updated": ai.last_updated.isoformat()
                }
                for ai in ais
            ]
        }
    except Exception as e:
        logger.error(f"Error listing quantum AIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ais/{ai_id}")
async def get_quantum_ai_endpoint(ai_id: str):
    """Get quantum AI information"""
    try:
        ai = get_quantum_ai().get_quantum_ai(ai_id)
        if not ai:
            raise HTTPException(status_code=404, detail="Quantum AI not found")
        
        return {
            "ai_id": ai.ai_id,
            "name": ai.name,
            "ai_type": ai.ai_type,
            "quantum_circuit": ai.quantum_circuit,
            "ai_capabilities": ai.ai_capabilities,
            "parameters": ai.parameters,
            "is_active": ai.is_active,
            "created_at": ai.created_at.isoformat(),
            "last_updated": ai.last_updated.isoformat(),
            "metadata": ai.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quantum AI {ai_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results")
async def get_quantum_ai_results(ai_id: Optional[str] = None):
    """Get quantum AI results"""
    try:
        quantum_ai_system = get_quantum_ai()
        results = quantum_ai_system.get_quantum_ai_results(ai_id)
        return {
            "results": [
                {
                    "result_id": result.result_id,
                    "ai_id": result.ai_id,
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "quantum_advantage": result.quantum_advantage,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error getting quantum AI results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_quantum_ai_summary_endpoint():
    """Get quantum AI system summary"""
    try:
        summary = get_quantum_ai_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting quantum AI summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_quantum_ai_capabilities():
    """Get quantum AI capabilities"""
    try:
        quantum_ai_system = get_quantum_ai()
        return {
            "quantum_ai_capabilities": quantum_ai_system.quantum_ai_capabilities,
            "quantum_ai_types": list(quantum_ai_system.quantum_ai_types.keys()),
            "quantum_ai_architectures": list(quantum_ai_system.quantum_ai_architectures.keys()),
            "quantum_ai_algorithms": list(quantum_ai_system.quantum_ai_algorithms.keys()),
            "quantum_ai_metrics": list(quantum_ai_system.quantum_ai_metrics.keys())
        }
    except Exception as e:
        logger.error(f"Error getting quantum AI capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def quantum_ai_health():
    """Get quantum AI system health"""
    try:
        quantum_ai_system = get_quantum_ai()
        summary = quantum_ai_system.get_quantum_ai_summary()
        
        return {
            "status": "healthy",
            "total_ais": summary["total_ais"],
            "active_ais": summary["active_ais"],
            "total_results": summary["total_results"],
            "recent_ais": summary["recent_ais"],
            "recent_results": summary["recent_results"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting quantum AI health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.delete("/clear")
async def clear_quantum_ai_data_endpoint():
    """Clear all quantum AI data"""
    try:
        clear_quantum_ai_data()
        return {"message": "Quantum AI data cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing quantum AI data: {e}")
        raise HTTPException(status_code=500, detail=str(e))











