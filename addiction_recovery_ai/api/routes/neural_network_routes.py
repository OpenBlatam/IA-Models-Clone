"""
Neural network analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.neural_network_analysis_service import NeuralNetworkAnalysisService
except ImportError:
    from ...services.neural_network_analysis_service import NeuralNetworkAnalysisService

router = APIRouter()

neural_network = NeuralNetworkAnalysisService()


@router.post("/neural-network/analyze")
async def analyze_with_neural_network(
    user_id: str = Body(...),
    input_data: Dict = Body(...),
    model_type: str = Body("default")
):
    """Analiza datos con red neuronal"""
    try:
        analysis = neural_network.analyze_with_neural_network(user_id, input_data, model_type)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando con red neuronal: {str(e)}")



