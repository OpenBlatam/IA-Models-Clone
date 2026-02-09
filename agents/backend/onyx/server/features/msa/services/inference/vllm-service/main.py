"""
vLLM Inference Service

High-performance LLM inference service using vLLM with PagedAttention.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="vLLM Inference Service",
    version="1.0.0",
    description="High-performance LLM inference with PagedAttention"
)

# Service metadata
SERVICE_NAME = "vllm-inference-service"
SERVICE_VERSION = "1.0.0"


class GenerationRequest(BaseModel):
    """Request for text generation"""
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95
    stop: Optional[List[str]] = None


class GenerationResponse(BaseModel):
    """Response from text generation"""
    text: str
    tokens_generated: int
    latency_ms: float
    model: str


class BatchGenerationRequest(BaseModel):
    """Request for batch generation"""
    prompts: List[str]
    max_tokens: int = 100
    temperature: float = 0.7


# Global vLLM engine (lazy loaded)
_engine = None


def get_engine():
    """Get or initialize vLLM engine"""
    import os
    global _engine
    if _engine is None:
        try:
            from vllm import LLM, SamplingParams
            # Initialize with default model (can be configured via env)
            model_name = os.getenv("MODEL_NAME", "gpt2")
            _engine = LLM(model=model_name, tensor_parallel_size=1)
            logger.info(f"Initialized vLLM engine with model: {model_name}")
        except ImportError:
            logger.warning("vLLM not available, using mock")
            _engine = "mock"
    return _engine


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/v1/inference/health")
async def inference_health():
    """Inference service health check"""
    engine = get_engine()
    return {
        "status": "ready" if engine != "mock" else "mock",
        "engine_loaded": engine is not None
    }


@app.post("/v1/inference/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from prompt"""
    import time
    import os
    
    start_time = time.perf_counter()
    engine = get_engine()
    
    try:
        if engine == "mock":
            # Mock response for testing
            return GenerationResponse(
                text=f"Mock response for: {request.prompt[:50]}...",
                tokens_generated=10,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                model="mock"
            )
        
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop
        )
        
        outputs = engine.generate([request.prompt], sampling_params)
        
        generated_text = outputs[0].outputs[0].text
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return GenerationResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            model=os.getenv("MODEL_NAME", "unknown")
        )
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/inference/batch")
async def generate_batch(request: BatchGenerationRequest):
    """Generate text for multiple prompts"""
    import time
    
    start_time = time.perf_counter()
    engine = get_engine()
    
    try:
        if engine == "mock":
            return {
                "results": [
                    {"text": f"Mock: {p[:50]}", "tokens": 10}
                    for p in request.prompts
                ],
                "total_latency_ms": (time.perf_counter() - start_time) * 1000
            }
        
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        outputs = engine.generate(request.prompts, sampling_params)
        
        results = []
        for output in outputs:
            results.append({
                "text": output.outputs[0].text,
                "tokens": len(output.outputs[0].token_ids)
            })
        
        return {
            "results": results,
            "total_latency_ms": (time.perf_counter() - start_time) * 1000,
            "prompts_processed": len(request.prompts)
        }
    
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

