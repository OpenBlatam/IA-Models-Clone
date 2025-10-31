from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import Response, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE
from starlette.exceptions import HTTPException as StarletteHTTPException
import sentry_sdk
import os
from .auth import check_auth, require_scope, login_for_access_token, refresh_token_endpoint
from .model_loader import maybe_reload_model, load_model, device, model, tokenizer, last_loaded, background_reloader, startup_event
from .logging_utils import logger
from .metrics import REQUESTS, ERRORS, LATENCY, instrumentator
from .schemas import GenerationRequest, BatchGenerationRequest, GenerationResponse, BatchGenerationResponse, TokenResponse, RefreshTokenRequest
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    import time, uuid
        import torch
    import time, uuid, torch
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
API de inferencia LLM enterprise, modular, instrumentada y lista para producción.
- FastAPI + Transformers + structlog + Sentry + Prometheus + orjson
- Seguridad, observabilidad, recarga automática, middlewares y mejores prácticas
"""

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    default_response_class=ORJSONResponse,
    title="LLM Inference API",
    description="API de inferencia LLM lista para producción, modular y segura.",
    version="1.0.0"
)

# --- Middlewares producción ---
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
if os.getenv("TRUSTED_HOSTS"):
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=os.getenv("TRUSTED_HOSTS").split(","))
if os.getenv("FORCE_HTTPS") == "1":
    app.add_middleware(HTTPSRedirectMiddleware)

# --- Instrumentación Prometheus ---
@app.on_event("startup")
def on_startup():
    
    """on_startup function."""
startup_event()
    instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# --- Error handling global ---
@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    
    """global_exception_handler function."""
logger.error({"event": "unhandled_exception", "error": str(exc)})
    sentry_sdk.capture_exception(exc)
    return ORJSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.exception_handler(StarletteHTTPException)
def http_exception_handler(request: Request, exc: StarletteHTTPException):
    
    """http_exception_handler function."""
logger.warning({"event": "http_exception", "status_code": exc.status_code, "detail": exc.detail})
    return ORJSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# --- Endpoints auth ---
app.post("/token", response_model=TokenResponse)(login_for_access_token)
app.post("/token/refresh", response_model=TokenResponse)(refresh_token_endpoint)

# --- Endpoints health/readiness para K8s ---
@app.get("/health", tags=["infra"])
def health():
    
    """health function."""
REQUESTS.labels(endpoint="health").inc()
    return {"status": "ok"}

@app.get("/readiness", tags=["infra"])
def readiness():
    
    """readiness function."""
# Puedes agregar lógica para readiness real (ej: modelo cargado, DB, etc)
    if model is None or tokenizer is None:
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    return {"ready": True}

@app.get("/version", tags=["infra"])
def version():
    
    """version function."""
REQUESTS.labels(endpoint="version").inc()
    return {"model_path": getattr(model, 'model_path', None), "last_loaded": last_loaded}

@app.get("/metrics", include_in_schema=False)
def metrics():
    
    """metrics function."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/docs")
def custom_docs(auth=Depends(check_auth)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="LLM Inference API Docs")

@app.get("/openapi.json")
def openapi(auth=Depends(check_auth)):
    return app.openapi()

# --- Endpoints de inferencia ---
@app.post("/predict", response_model=GenerationResponse)
async def predict(
    req: GenerationRequest,
    request: Request,
    auth=Depends(require_scope("llm:predict"))
):
    endpoint = "predict"
    REQUESTS.labels(endpoint=endpoint).inc()
    start = time.time()
    request_id = str(uuid.uuid4())
    user = auth.get("user")
    try:
        maybe_reload_model()
        inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=True
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        LATENCY.labels(endpoint=endpoint).observe(time.time() - start)
        logger.info({"event": "predict", "request_id": request_id, "user": user})
        return {"result": result}
    except Exception as e:
        ERRORS.labels(endpoint=endpoint).inc()
        LATENCY.labels(endpoint=endpoint).observe(time.time() - start)
        logger.error({"event": "predict_error", "request_id": request_id, "user": user, "error": str(e)})
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/batch_predict", response_model=BatchGenerationResponse)
async def batch_predict(
    req: BatchGenerationRequest,
    request: Request,
    auth=Depends(require_scope("llm:predict"))
):
    endpoint = "batch_predict"
    REQUESTS.labels(endpoint=endpoint).inc()
    start = time.time()
    request_id = str(uuid.uuid4())
    user = auth.get("user")
    try:
        maybe_reload_model()
        results = []
        for prompt in req.prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    do_sample=True
                )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(result)
        LATENCY.labels(endpoint=endpoint).observe(time.time() - start)
        logger.info({"event": "batch_predict", "request_id": request_id, "user": user})
        return {"results": results}
    except Exception as e:
        ERRORS.labels(endpoint=endpoint).inc()
        LATENCY.labels(endpoint=endpoint).observe(time.time() - start)
        logger.error({"event": "batch_predict_error", "request_id": request_id, "user": user, "error": str(e)})
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

"""
# Ejemplo de healthcheck para K8s:
# livenessProbe:
#   httpGet:
#     path: /health
#     port: 8000
# readinessProbe:
#   httpGet:
#     path: /readiness
#     port: 8000
""" 