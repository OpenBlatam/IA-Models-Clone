"""
FastAPI optimization utilities for webhooks
"""

import orjson
from fastapi.responses import ORJSONResponse
from fastapi import FastAPI
import uvicorn

def setup_fastapi_optimization(app: FastAPI):
    """Setup FastAPI with orjson and optimizations"""
    
    # Use orjson for JSON responses
    app.default_response_class = ORJSONResponse
    
    # Custom JSON encoder for Pydantic models
    def custom_json_encoder(obj):
        if hasattr(obj, 'dict'):
            return obj.dict()
        return obj
    
    # Override default JSON encoder
    app.json_encoder = custom_json_encoder

def run_optimized_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8000):
    """Run server with optimizations"""
    uvicorn.run(
        app,
        host=host,
        port=port,
        loop="uvloop",  # Use uvloop for better performance
        http="httptools",  # Use httptools for better HTTP parsing
        workers=1,  # Single worker for webhooks (async is better)
        access_log=False,  # Disable access logs for performance
        log_level="warning"
    )