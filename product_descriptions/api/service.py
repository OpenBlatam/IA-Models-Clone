from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from contextlib import asynccontextmanager
from typing import Annotated, Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends, Query, Path, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
import asyncio
import logging
import time
from datetime import datetime
import uvicorn
import redis.asyncio as redis
from functools import wraps
import hashlib
from ..core.generator import ProductDescriptionGenerator
from ..core.config import ProductDescriptionConfig
from typing import Any, List, Dict, Optional
"""
Enhanced Product API - Production Ready
======================================

High-performance, scalable API following FastAPI best practices:
- Async/await optimization
- Dependency injection
- Error handling with early returns  
- Performance monitoring
- Comprehensive caching
- Rate limiting
- Type safety
"""



# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for metrics
app_state = {
    "startup_time": None,
    "request_count": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "error_count": 0
}


# Pydantic models for request/response
class ProductRequest(BaseModel):
    """Enhanced request model for enterprise product management."""
    
    # Basic Information
    product_name: str = Field(..., min_length=1, max_length=200, description="Product name")
    description: str = Field("", max_length=5000, description="Product description")
    short_description: str = Field("", max_length=500, description="Short product description")
    sku: str = Field(..., min_length=1, max_length=100, description="Product SKU")
    product_type: str = Field("physical", description="Product type")
    brand_id: Optional[str] = Field(None, description="Brand ID")
    category_id: Optional[str] = Field(None, description="Category ID")
    
    # Pricing
    base_price_amount: Optional[float] = Field(None, gt=0, description="Base price amount")
    base_price_currency: str = Field("USD", description="Price currency")
    sale_price_amount: Optional[float] = Field(None, gt=0, description="Sale price amount")
    cost_price_amount: Optional[float] = Field(None, gt=0, description="Cost price amount")
    
    # Inventory
    inventory_quantity: int = Field(0, ge=0, description="Initial inventory quantity")
    low_stock_threshold: int = Field(10, ge=0, description="Low stock threshold")
    inventory_tracking: str = Field("track", description="Inventory tracking mode")
    allow_backorder: bool = Field(False, description="Allow backorder")
    
    # Physical Properties
    length: Optional[float] = Field(None, gt=0, description="Product length")
    width: Optional[float] = Field(None, gt=0, description="Product width")
    height: Optional[float] = Field(None, gt=0, description="Product height")
    weight: Optional[float] = Field(None, gt=0, description="Product weight")
    requires_shipping: bool = Field(True, description="Requires shipping")
    
    # Digital Properties
    download_url: Optional[str] = Field(None, description="Download URL for digital products")
    download_limit: Optional[int] = Field(None, gt=0, description="Download limit")
    
    # SEO and Marketing
    seo_title: Optional[str] = Field(None, max_length=100, description="SEO title")
    seo_description: Optional[str] = Field(None, max_length=300, description="SEO description")
    seo_keywords: List[str] = Field(default_factory=list, description="SEO keywords")
    slug: Optional[str] = Field(None, description="URL slug")
    tags: List[str] = Field(default_factory=list, description="Product tags")
    featured: bool = Field(False, description="Featured product")
    
    # Media
    images: List[str] = Field(default_factory=list, description="Product images")
    videos: List[str] = Field(default_factory=list, description="Product videos")
    
    # AI Generation (Legacy)
    features: List[str] = Field(default_factory=list, min_items=0, max_items=20, description="Product features")
    style: str = Field("professional", description="Writing style")
    tone: str = Field("friendly", description="Writing tone")
    max_length: int = Field(300, ge=50, le=1000, description="Maximum description length")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Generation temperature")
    num_variations: int = Field(1, ge=1, le=5, description="Number of variations")
    use_cache: bool = Field(True, description="Whether to use caching")
    auto_generate_description: bool = Field(False, description="Auto-generate AI description")
    
    @validator('price_range')
    def validate_price_range(cls, v) -> bool:
        allowed = ["low", "medium", "high", "luxury"]
        if v not in allowed:
            raise ValueError(f"price_range must be one of {allowed}")
        return v
    
    @validator('style')
    def validate_style(cls, v) -> bool:
        allowed = ["professional", "casual", "luxury", "technical", "creative"]
        if v not in allowed:
            raise ValueError(f"style must be one of {allowed}")
        return v
    
    @validator('tone')
    def validate_tone(cls, v) -> bool:
        allowed = ["friendly", "formal", "enthusiastic", "informative", "persuasive"]
        if v not in allowed:
            raise ValueError(f"tone must be one of {allowed}")
        return v


class BatchProductRequest(BaseModel):
    """Request model for batch product description generation."""
    
    products: List[ProductRequest] = Field(..., min_items=1, max_items=50)
    max_concurrent: int = Field(4, ge=1, le=10, description="Maximum concurrent generations")


class PresetRequest(BaseModel):
    """Request model for preset-based generation."""
    
    product_name: str = Field(..., min_length=1, max_length=200)
    features: List[str] = Field(..., min_items=1, max_items=20)
    preset: str = Field("ecommerce", description="Preset configuration")
    override_params: Optional[Dict[str, Any]] = Field(None, description="Parameter overrides")
    
    @validator('preset')
    def validate_preset(cls, v) -> bool:
        allowed = ["ecommerce", "luxury", "technical"]
        if v not in allowed:
            raise ValueError(f"preset must be one of {allowed}")
        return v


class GenerationResponse(BaseModel):
    """Response model for generated descriptions."""
    
    success: bool
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    uptime: float
    stats: Dict[str, Any]
    timestamp: str


class ProductDescriptionService:
    """
    FastAPI-based Product Description Generation Service
    
    Features:
    - RESTful API endpoints
    - Async request handling
    - Input validation
    - Error handling
    - Rate limiting
    - Health monitoring
    - Background tasks
    """
    
    def __init__(self, config: Optional[ProductDescriptionConfig] = None):
        
    """__init__ function."""
self.config = config or ProductDescriptionConfig()
        self.generator: Optional[ProductDescriptionGenerator] = None
        self.app = FastAPI(
            title="Product Description Generator API",
            description="AI-powered product description generation service",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Service state
        self.start_time = time.time()
        self.is_ready = False
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_middleware(self) -> Any:
        """Setup FastAPI middleware."""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Custom middleware for logging
        @self.app.middleware("http")
        async async def log_requests(request, call_next) -> Any:
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
    
    def _setup_routes(self) -> Any:
        """Setup API routes."""
        
        @self.app.on_event("startup")
        async def startup():
            """Initialize the service on startup."""
            await self.initialize()
        
        @self.app.on_event("shutdown")
        async def shutdown():
            """Cleanup on shutdown."""
            logger.info("Shutting down Product Description Service")
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with service information."""
            return {
                "service": "Product Description Generator API",
                "version": "1.0.0",
                "status": "operational" if self.is_ready else "initializing",
                "docs": "/docs"
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Service not ready")
            
            uptime = time.time() - self.start_time
            stats = self.generator.get_stats() if self.generator else {}
            
            return HealthResponse(
                status="healthy",
                uptime=uptime,
                stats=stats,
                timestamp=datetime.utcnow().isoformat()
            )
        
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate_description(request: ProductRequest):
            """Generate single product description."""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Service not ready")
            
            try:
                start_time = time.time()
                
                results = await self.generator.generate_async(
                    product_name=request.product_name,
                    features=request.features,
                    category=request.category,
                    brand=request.brand,
                    price_range=request.price_range,
                    style=request.style,
                    tone=request.tone,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    num_variations=request.num_variations,
                    use_cache=request.use_cache
                )
                
                generation_time = time.time() - start_time
                
                return GenerationResponse(
                    success=True,
                    data=results,
                    metadata={
                        "generation_time_ms": generation_time * 1000,
                        "num_results": len(results),
                        "request_params": request.dict()
                    },
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error generating description: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/generate/batch", response_model=GenerationResponse)
        async def generate_batch_descriptions(request: BatchProductRequest):
            """Generate descriptions for multiple products."""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Service not ready")
            
            try:
                start_time = time.time()
                
                # Convert request to list of dicts
                products = [product.dict() for product in request.products]
                
                results = await self.generator.generate_batch_async(
                    products=products,
                    max_concurrent=request.max_concurrent
                )
                
                generation_time = time.time() - start_time
                
                return GenerationResponse(
                    success=True,
                    data=results,
                    metadata={
                        "generation_time_ms": generation_time * 1000,
                        "num_products": len(products),
                        "max_concurrent": request.max_concurrent
                    },
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error generating batch descriptions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/generate/preset", response_model=GenerationResponse)
        async def generate_with_preset(request: PresetRequest):
            """Generate description using predefined preset."""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Service not ready")
            
            try:
                start_time = time.time()
                
                results = self.generator.generate_with_preset(
                    product_name=request.product_name,
                    features=request.features,
                    preset=request.preset,
                    **(request.override_params or {})
                )
                
                generation_time = time.time() - start_time
                
                return GenerationResponse(
                    success=True,
                    data=results,
                    metadata={
                        "generation_time_ms": generation_time * 1000,
                        "preset_used": request.preset,
                        "override_params": request.override_params
                    },
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error generating with preset: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats", response_model=Dict[str, Any])
        async def get_statistics():
            """Get service statistics."""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Service not ready")
            
            stats = self.generator.get_stats()
            stats["service_uptime"] = time.time() - self.start_time
            stats["timestamp"] = datetime.utcnow().isoformat()
            
            return stats
        
        @self.app.post("/cache/clear")
        async def clear_cache(background_tasks: BackgroundTasks):
            """Clear generation cache."""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Service not ready")
            
            background_tasks.add_task(self.generator.clear_cache)
            return {"message": "Cache clearing initiated"}
        
        @self.app.get("/presets")
        async def get_available_presets():
            """Get available generation presets."""
            return {
                "presets": [
                    {
                        "name": "ecommerce",
                        "description": "Standard e-commerce product descriptions",
                        "style": "professional",
                        "tone": "friendly"
                    },
                    {
                        "name": "luxury",
                        "description": "High-end luxury product descriptions",
                        "style": "luxury",
                        "tone": "sophisticated"
                    },
                    {
                        "name": "technical",
                        "description": "Technical and detailed descriptions",
                        "style": "technical",
                        "tone": "informative"
                    }
                ]
            }
    
    async def initialize(self) -> Any:
        """Initialize the service."""
        try:
            logger.info("Initializing Product Description Service...")
            
            # Initialize generator
            self.generator = ProductDescriptionGenerator(self.config)
            await self.generator.initialize()
            
            self.is_ready = True
            logger.info("Product Description Service initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the service."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            **kwargs
        ) 