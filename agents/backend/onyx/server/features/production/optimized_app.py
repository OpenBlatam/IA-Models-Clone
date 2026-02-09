from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from fastapi import FastAPI, APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import ORJSONResponse
from cachetools import LRUCache, cached
import time
import logging
import asyncio
import httpx

from typing import Any, List, Dict, Optional
app = FastAPI(default_response_class=ORJSONResponse)
router = APIRouter()

# --- MODELOS ---
class Product(BaseModel):
    id: int
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)

# --- CACHE ---
product_cache = LRUCache(maxsize=128)

@cached(product_cache)
def get_product_from_cache(product_id: int) -> dict:
    time.sleep(0.01)
    return {"id": product_id, "name": f"Product {product_id}", "price": 10.0}

# --- DB ---
async async def fetch_product_from_db(product_id: int) -> dict:
    await asyncio.sleep(0.01)
    return {"id": product_id, "name": f"DB Product {product_id}", "price": 20.0}

# --- RUTAS ---
@router.post("/products/", response_model=Product, status_code=status.HTTP_201_CREATED)
async def create_product(product: Product) -> Product:
    return product

@router.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: int) -> Product:
    data = get_product_from_cache(product_id)
    return Product(**data)

@router.get("/db/products/{product_id}", response_model=Product)
async def get_product_db(product_id: int) -> Product:
    data = await fetch_product_from_db(product_id)
    return Product(**data)

@router.get("/external")
async def external_call() -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.github.com")
        return {"status": r.status_code, "url": r.json().get("current_user_url")}

@router.get("/ping")
async def ping() -> dict:
    return {"message": "pong"}

app.include_router(router)

# --- MIDDLEWARE PERFORMANCE Y LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("middleware")

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next) -> Any:
        start = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - start) * 1000
        logger.info(f"{request.method} {request.url.path} - {duration:.2f}ms")
        response.headers["X-Process-Time-ms"] = f"{duration:.2f}"
        return response

app.add_middleware(TimingMiddleware) 