"""
Optimized REST API for Recovery AI
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import torch
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Addiction Recovery AI API",
    description="Advanced AI-powered recovery support system",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ProgressRequest(BaseModel):
    days_sober: float = Field(..., ge=0, le=3650)
    cravings_level: float = Field(..., ge=0, le=10)
    stress_level: float = Field(..., ge=0, le=10)
    support_level: float = Field(..., ge=0, le=10)
    mood_score: float = Field(..., ge=0, le=10)
    sleep_quality: Optional[float] = Field(None, ge=0, le=10)
    exercise_frequency: Optional[float] = Field(None, ge=0, le=7)
    therapy_sessions: Optional[float] = Field(None, ge=0, le=10)
    medication_compliance: Optional[float] = Field(None, ge=0, le=1)
    social_activity: Optional[float] = Field(None, ge=0, le=7)


class ProgressResponse(BaseModel):
    progress: float = Field(..., ge=0, le=1)
    percentage: str
    inference_time_ms: float


class RelapseRiskRequest(BaseModel):
    sequence: List[Dict[str, float]]


class RelapseRiskResponse(BaseModel):
    risk: float = Field(..., ge=0, le=1)
    risk_level: str
    inference_time_ms: float


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class SentimentResponse(BaseModel):
    label: str
    score: float = Field(..., ge=0, le=1)
    inference_time_ms: float


# Global engine (lazy loading)
_engine = None


@lru_cache(maxsize=1)
def get_engine():
    """Get or create engine (cached)"""
    global _engine
    if _engine is None:
        from addiction_recovery_ai import create_ultra_fast_engine
        _engine = create_ultra_fast_engine(use_gpu=torch.cuda.is_available())
    return _engine


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Addiction Recovery AI",
        "version": "2.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        engine = get_engine()
        return {
            "status": "healthy",
            "gpu_available": torch.cuda.is_available(),
            "engine_loaded": engine is not None
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")


@app.post("/predict/progress", response_model=ProgressResponse)
async def predict_progress(
    request: ProgressRequest,
    background_tasks: BackgroundTasks
):
    """Predict recovery progress"""
    import time
    start = time.time()
    
    try:
        engine = get_engine()
        
        features = request.dict(exclude_none=True)
        progress = engine.predict_progress(features)
        
        elapsed = (time.time() - start) * 1000
        
        return ProgressResponse(
            progress=progress,
            percentage=f"{progress * 100:.1f}%",
            inference_time_ms=elapsed
        )
    except Exception as e:
        logger.error(f"Progress prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/relapse-risk", response_model=RelapseRiskResponse)
async def predict_relapse_risk(
    request: RelapseRiskRequest,
    background_tasks: BackgroundTasks
):
    """Predict relapse risk"""
    import time
    start = time.time()
    
    try:
        engine = get_engine()
        
        risk = engine.predict_relapse_risk(request.sequence)
        elapsed = (time.time() - start) * 1000
        
        risk_level = "High" if risk > 0.7 else "Medium" if risk > 0.4 else "Low"
        
        return RelapseRiskResponse(
            risk=risk,
            risk_level=risk_level,
            inference_time_ms=elapsed
        )
    except Exception as e:
        logger.error(f"Relapse risk prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks
):
    """Analyze sentiment"""
    import time
    start = time.time()
    
    try:
        engine = get_engine()
        
        sentiment = engine.analyze_sentiment(request.text)
        elapsed = (time.time() - start) * 1000
        
        return SentimentResponse(
            label=sentiment.get("label", "NEUTRAL"),
            score=sentiment.get("score", 0.5),
            inference_time_ms=elapsed
        )
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(
    requests: List[ProgressRequest],
    background_tasks: BackgroundTasks
):
    """Batch prediction"""
    import time
    start = time.time()
    
    try:
        engine = get_engine()
        
        results = []
        for request in requests:
            features = request.dict(exclude_none=True)
            progress = engine.predict_progress(features)
            results.append({
                "progress": progress,
                "percentage": f"{progress * 100:.1f}%"
            })
        
        elapsed = (time.time() - start) * 1000
        
        return {
            "results": results,
            "total_time_ms": elapsed,
            "avg_time_ms": elapsed / len(requests) if requests else 0,
            "count": len(requests)
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8018)

