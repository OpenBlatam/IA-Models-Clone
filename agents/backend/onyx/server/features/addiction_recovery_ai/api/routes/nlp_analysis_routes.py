"""
NLP analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.nlp_analysis_service import NLPAnalysisService
except ImportError:
    from ...services.nlp_analysis_service import NLPAnalysisService

router = APIRouter()

nlp_analysis = NLPAnalysisService()


@router.post("/nlp/analyze-text")
async def analyze_text(
    text: str = Body(...),
    analysis_type: str = Body("comprehensive")
):
    """Analiza texto con NLP"""
    try:
        analysis = nlp_analysis.analyze_text(text, analysis_type)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando texto: {str(e)}")


@router.post("/nlp/extract-insights")
async def extract_insights(
    text: str = Body(...),
    context: Optional[Dict] = Body(None)
):
    """Extrae insights del texto"""
    try:
        insights = nlp_analysis.extract_insights(text, context)
        return JSONResponse(content=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extrayendo insights: {str(e)}")



