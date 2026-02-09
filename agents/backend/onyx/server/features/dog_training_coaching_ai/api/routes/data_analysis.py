"""
Data Analysis Routes
====================
Endpoints para análisis de datos.
"""

from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Any, Optional

from ...utils.data_analysis import DataAnalyzer, analyze_trends, detect_anomalies, calculate_correlation
from ...utils.logger import get_logger

router = APIRouter(prefix="/api/v1/analysis", tags=["Data Analysis"])
logger = get_logger(__name__)


@router.post("/describe")
async def describe_data(data: List[Any] = Body(...)):
    """
    Obtener descripción estadística de datos.
    
    Args:
        data: Datos a analizar
        
    Returns:
        Descripción estadística
    """
    try:
        analyzer = DataAnalyzer(data)
        return {
            "status": "success",
            "description": analyzer.describe()
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/frequency")
async def frequency_distribution(data: List[Any] = Body(...)):
    """
    Obtener distribución de frecuencias.
    
    Args:
        data: Datos a analizar
        
    Returns:
        Distribución de frecuencias
    """
    try:
        analyzer = DataAnalyzer(data)
        return {
            "status": "success",
            "frequency": analyzer.frequency_distribution()
        }
    except Exception as e:
        logger.error(f"Frequency analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trends")
async def analyze_data_trends(
    data: List[Dict[str, Any]] = Body(...),
    value_key: str = Body(...),
    time_key: str = Body(...)
):
    """
    Analizar tendencias en datos temporales.
    
    Args:
        data: Datos con timestamps
        value_key: Clave del valor a analizar
        time_key: Clave del timestamp
        
    Returns:
        Análisis de tendencias
    """
    try:
        trends = analyze_trends(data, value_key, time_key)
        return {
            "status": "success",
            "trends": trends
        }
    except Exception as e:
        logger.error(f"Trend analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/anomalies")
async def detect_data_anomalies(
    data: List[float] = Body(...),
    threshold: float = Body(2.0)
):
    """
    Detectar anomalías en datos.
    
    Args:
        data: Datos numéricos
        threshold: Threshold de desviación estándar
        
    Returns:
        Anomalías detectadas
    """
    try:
        anomalies = detect_anomalies(data, threshold)
        return {
            "status": "success",
            "anomalies": anomalies,
            "count": len(anomalies)
        }
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/correlation")
async def calculate_data_correlation(
    x: List[float] = Body(...),
    y: List[float] = Body(...)
):
    """
    Calcular correlación entre dos variables.
    
    Args:
        x: Primera variable
        y: Segunda variable
        
    Returns:
        Estadísticas de correlación
    """
    try:
        correlation = calculate_correlation(x, y)
        return {
            "status": "success",
            "correlation": correlation
        }
    except Exception as e:
        logger.error(f"Correlation calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



