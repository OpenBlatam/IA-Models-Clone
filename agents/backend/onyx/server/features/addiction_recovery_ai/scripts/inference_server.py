#!/usr/bin/env python3
"""
Inference Server
Fast inference server with all optimizations
"""

import argparse
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from addiction_recovery_ai import (
    get_config,
    ModelFactory,
    create_ultra_fast_inference,
    create_embedding_cache,
    create_inference_pipeline
)


app = FastAPI(title="Addiction Recovery AI Inference Server")


class PredictionRequest(BaseModel):
    features: List[float]
    model_type: Optional[str] = "progress_predictor"


class BatchPredictionRequest(BaseModel):
    features_list: List[List[float]]
    model_type: Optional[str] = "progress_predictor"


# Global models cache
_models_cache = {}
_inference_engines = {}
_embedding_cache = None


def load_model(model_type: str):
    """Load and cache model"""
    if model_type in _models_cache:
        return _models_cache[model_type], _inference_engines[model_type]
    
    config = get_config()
    model_config = config.get_model_config(model_type)
    model = ModelFactory.create(model_type, model_config)
    
    # Create ultra-fast inference engine
    engine = create_ultra_fast_inference(model)
    
    _models_cache[model_type] = model
    _inference_engines[model_type] = engine
    
    return model, engine


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    global _embedding_cache
    _embedding_cache = create_embedding_cache(max_size=10000)
    print("Inference server started")


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Single prediction"""
    try:
        model, engine = load_model(request.model_type)
        
        # Convert to tensor
        features_tensor = torch.tensor([request.features], dtype=torch.float32)
        
        # Predict
        output = engine.predict(features_tensor)
        
        return {
            "prediction": output.item() if output.numel() == 1 else output.tolist(),
            "model_type": request.model_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction"""
    try:
        model, engine = load_model(request.model_type)
        
        # Convert to tensors
        features_tensors = [torch.tensor(f, dtype=torch.float32) for f in request.features_list]
        
        # Batch predict
        outputs = engine.predict_batch_optimized(features_tensors, batch_size=64)
        
        predictions = [o.item() if o.numel() == 1 else o.tolist() for o in outputs]
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "model_type": request.model_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": list(_models_cache.keys()),
        "cached": len(_models_cache)
    }


def main():
    parser = argparse.ArgumentParser(description="Inference server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()













