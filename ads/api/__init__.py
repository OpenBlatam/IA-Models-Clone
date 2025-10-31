"""
Unified API package for the ads feature.

This package consolidates all API functionality from the scattered implementations:
- api.py (basic ads generation)
- advanced_api.py (advanced AI features)
- optimized_api.py (production-ready features)
- ai_api.py (AI operations)
- integrated_api.py (Onyx integration)
- tokenization_api.py (tokenization operations)
- gradio demos

The new structure follows Clean Architecture principles with clear separation of concerns.
"""

from .core import router as core_router
from .ai_api import router as ai_router
from .advanced import router as advanced_router
from .integrated_api import router as integrated_router
from .optimized import router as optimized_router

# Optional imports to keep test environment lightweight
try:
    from .tokenization import router as tokenization_router
except Exception:  # pragma: no cover
    tokenization_router = None  # type: ignore

try:
    from .gradio_integration import router as gradio_router
except Exception:  # pragma: no cover
    gradio_router = None  # type: ignore

try:
    from .fastapi_integration import router as fastapi_integration_router
except Exception:  # pragma: no cover
    fastapi_integration_router = None  # type: ignore
# Infra-heavy imports are optional to allow lightweight usage during tests
try:
    from ..infrastructure.database import DatabaseManager
    from ..infrastructure.cache import CacheManager, CacheConfig, CacheType
    from ..config import get_optimized_settings
    from ..infrastructure.external_services import ExternalServiceManager
except Exception:  # pragma: no cover - fallback for minimal test envs
    DatabaseManager = None  # type: ignore[assignment]
    CacheManager = None  # type: ignore[assignment]
    CacheConfig = None  # type: ignore[assignment]
    CacheType = None  # type: ignore[assignment]

    def get_optimized_settings():  # type: ignore[no-redef]
        class _S:
            redis_url: str | None = None
        return _S()

    class ExternalServiceManager:  # type: ignore[no-redef]
        def get_health_snapshot(self) -> dict:
            return {"services": {}, "total": 0}

# Main router that includes all sub-routers
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
try:
    from fastapi.responses import ORJSONResponse  # type: ignore
    _DefaultResponse = ORJSONResponse
except Exception:
    _DefaultResponse = None

main_router = APIRouter(
    prefix="/ads",
    tags=["ads"],
    responses={404: {"description": "Not found"}},
    default_response_class=_DefaultResponse if _DefaultResponse is not None else None,  # type: ignore[arg-type]
)

# Include all sub-routers
main_router.include_router(core_router, prefix="/core")
main_router.include_router(ai_router, prefix="/ai")
main_router.include_router(advanced_router, prefix="/advanced")
main_router.include_router(integrated_router, prefix="/integrated")
main_router.include_router(optimized_router, prefix="/optimized")
if tokenization_router is not None:
    main_router.include_router(tokenization_router, prefix="/tokenization")
if gradio_router is not None:
    main_router.include_router(gradio_router, prefix="/gradio")
if fastapi_integration_router is not None:
    main_router.include_router(fastapi_integration_router, prefix="/fastapi")

# Convenience alias for compatibility with code expecting `router`
router = main_router

# Aggregated health endpoint (lightweight)
class HealthResponse(BaseModel):
    status: str
    routers: List[str]


@main_router.get("/health", response_model=HealthResponse)
async def ads_health() -> HealthResponse:
    return {
        "status": "ok",
        "routers": [
            "/core",
            "/ai",
            "/advanced",
            "/integrated",
            "/optimized",
            "/tokenization" if tokenization_router is not None else "",
            "/gradio" if gradio_router is not None else "",
            "/fastapi" if fastapi_integration_router is not None else "",
        ],
    }

# Capabilities summary endpoint
class CapabilitiesResponse(BaseModel):
    feature: str
    capabilities: Dict[str, List[str]]


@main_router.get("/capabilities", response_model=CapabilitiesResponse)
async def ads_capabilities() -> CapabilitiesResponse:
    return {
        "feature": "ads",
        "capabilities": {
            "domain": ["entities", "value_objects", "services"],
            "application": ["use_cases", "dto"],
            "infrastructure": ["database", "storage", "cache", "external_services"],
            "optimization": ["performance", "profiling", "gpu"],
            "training": ["pytorch", "diffusion", "multi_gpu", "experiment_tracking"],
            "api": ["core", "ai", "advanced", "integrated", "optimized", "tokenization", "gradio", "fastapi"],
        },
    }

# Simple micro-benchmark and system snapshot
class BenchmarksResponse(BaseModel):
    json_encode_ms: float
    cpu_percent: Optional[float] = None
    memory: Optional[Dict[str, Any]] = None


@main_router.get("/benchmarks", response_model=BenchmarksResponse)
async def ads_benchmarks() -> BenchmarksResponse:
    import time
    start = time.perf_counter()
    # minimal JSON encode work
    payload = {"n": 1000, "items": list(range(10))}
    try:
        if _DefaultResponse is not None:
            import orjson as _oj  # type: ignore
            _ = _oj.dumps(payload)
        else:
            import json as _json
            _ = _json.dumps(payload)
    except Exception:
        pass
    end = time.perf_counter()

    # Optional system metrics
    cpu_percent = None
    mem = None
    try:
        import psutil  # type: ignore
        cpu_percent = psutil.cpu_percent(interval=0.05)
        vm = psutil.virtual_memory()
        mem = {"total": vm.total, "available": vm.available, "percent": vm.percent}
    except Exception:
        pass

    return {
        "json_encode_ms": round((end - start) * 1000, 3),
        "cpu_percent": cpu_percent,
        "memory": mem,
    }

# Deep benchmark for infra roundtrips (DB and Cache)
class BenchmarksDeepResponse(BaseModel):
    timings: Dict[str, float]
    errors: Dict[str, str]


@main_router.get("/benchmarks/deep", response_model=BenchmarksDeepResponse)
async def ads_benchmarks_deep() -> BenchmarksDeepResponse:
    import time
    timings = {}
    errors = {}

    # DB roundtrip (open session and execute trivial SELECT 1 if supported)
    try:
        dbm = DatabaseManager()
        t0 = time.perf_counter()
        async with dbm.get_session() as session:
            # Attempt a trivial SELECT 1
            try:
                from sqlalchemy import text as _sql_text  # type: ignore
                await session.execute(_sql_text("SELECT 1"))
            except Exception:
                # Fallback: just opening the session
                pass
        timings["db_roundtrip_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    except Exception as e:
        errors["db"] = str(e)

    # Cache roundtrip
    try:
        cache = CacheManager(CacheConfig(cache_type=CacheType.MEMORY))
        key = cache.generate_key("bench", time.time())
        t1 = time.perf_counter()
        await cache.set(key, {"ok": True}, ttl=5)
        _ = await cache.get(key)
        await cache.strategy.delete(key)
        timings["cache_roundtrip_ms"] = round((time.perf_counter() - t1) * 1000, 3)
    except Exception as e:
        errors["cache"] = str(e)

    # Redis roundtrip if configured
    try:
        settings = get_optimized_settings()
        redis_url = getattr(settings, "redis_url", None)
        if isinstance(redis_url, str) and redis_url:
            rcache = CacheManager(CacheConfig(cache_type=CacheType.REDIS, redis_url=redis_url))
            rkey = rcache.generate_key("bench-redis", time.time())
            t2 = time.perf_counter()
            await rcache.set(rkey, {"ok": True, "ts": time.time()}, ttl=5)
            _ = await rcache.get(rkey)
            await rcache.strategy.delete(rkey)
            timings["redis_roundtrip_ms"] = round((time.perf_counter() - t2) * 1000, 3)
    except Exception as e:
        errors["redis"] = str(e)

    return {"timings": timings, "errors": errors}

# External services health snapshot
class ServicesHealthResponse(BaseModel):
    services: Dict[str, Any]
    total: int


@main_router.get("/health/services", response_model=ServicesHealthResponse)
async def ads_services_health() -> ServicesHealthResponse:
    mgr = ExternalServiceManager()
    # In real use, services would be registered during startup; snapshot still returns structure
    return mgr.get_health_snapshot()

__all__ = [
    "main_router",
    "router",
    "core_router",
    "ai_router", 
    "advanced_router",
    "integrated_router",
    "optimized_router",
    "tokenization_router",
    "gradio_router",
    "fastapi_integration_router",
]
