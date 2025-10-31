from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Union, Any, Tuple
import asyncio
import logging
import time
from datetime import datetime
from decimal import Decimal
import uvicorn
from ..core.model import (
from typing import Any, List, Dict, Optional
"""
Enhanced Product Service - Enterprise API
==========================================

Servicio API empresarial mejorado para gestión completa de productos
con funcionalidades avanzadas, validaciones y Clean Architecture.
"""


# Importar modelos mejorados
    EnhancedProductEntity, ProductStatus, ProductType, PriceType, 
    InventoryTracking, Money, Dimensions, SEOData
)

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS - API Layer
# ============================================================================

class EnhancedProductRequest(BaseModel):
    """Request mejorado para productos empresariales"""
    
    # Información básica
    name: str = Field(..., min_length=2, max_length=200, description="Nombre del producto")
    description: str = Field("", max_length=5000, description="Descripción del producto")
    short_description: str = Field("", max_length=500, description="Descripción corta")
    sku: str = Field(..., min_length=1, max_length=100, description="SKU único del producto")
    product_type: ProductType = Field(ProductType.PHYSICAL, description="Tipo de producto")
    brand_id: Optional[str] = Field(None, description="ID de la marca")
    category_id: Optional[str] = Field(None, description="ID de la categoría")
    
    # Precios
    base_price_amount: Optional[float] = Field(None, gt=0, description="Precio base")
    base_price_currency: str = Field("USD", min_length=3, max_length=3, description="Moneda")
    sale_price_amount: Optional[float] = Field(None, gt=0, description="Precio de oferta")
    cost_price_amount: Optional[float] = Field(None, gt=0, description="Precio de costo")
    price_type: PriceType = Field(PriceType.FIXED, description="Tipo de precio")
    
    # Inventario
    inventory_quantity: int = Field(0, ge=0, description="Cantidad en inventario")
    low_stock_threshold: int = Field(10, ge=0, description="Umbral de stock bajo")
    inventory_tracking: InventoryTracking = Field(InventoryTracking.TRACK, description="Seguimiento de inventario")
    allow_backorder: bool = Field(False, description="Permitir pedidos pendientes")
    
    # Propiedades físicas
    length: Optional[float] = Field(None, gt=0, description="Longitud (cm)")
    width: Optional[float] = Field(None, gt=0, description="Ancho (cm)")
    height: Optional[float] = Field(None, gt=0, description="Alto (cm)")
    weight: Optional[float] = Field(None, gt=0, description="Peso (kg)")
    requires_shipping: bool = Field(True, description="Requiere envío")
    
    # Propiedades digitales
    download_url: Optional[str] = Field(None, description="URL de descarga")
    download_limit: Optional[int] = Field(None, gt=0, description="Límite de descargas")
    
    # SEO y Marketing
    seo_title: Optional[str] = Field(None, max_length=100, description="Título SEO")
    seo_description: Optional[str] = Field(None, max_length=300, description="Descripción SEO")
    seo_keywords: List[str] = Field(default_factory=list, description="Palabras clave SEO")
    slug: Optional[str] = Field(None, description="Slug URL")
    tags: List[str] = Field(default_factory=list, description="Etiquetas del producto")
    featured: bool = Field(False, description="Producto destacado")
    
    # Atributos y campos personalizados
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Atributos del producto")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Campos personalizados")
    
    # Media
    images: List[str] = Field(default_factory=list, description="URLs de imágenes")
    videos: List[str] = Field(default_factory=list, description="URLs de videos")
    documents: List[str] = Field(default_factory=list, description="URLs de documentos")
    
    # IA
    auto_generate_description: bool = Field(False, description="Generar descripción con IA")
    
    @validator('sku')
    def validate_sku(cls, v) -> bool:
        if not v or not v.strip():
            raise ValueError('SKU no puede estar vacío')
        return v.strip().upper()
    
    @validator('base_price_currency')
    def validate_currency(cls, v) -> bool:
        if len(v) != 3 or not v.isupper():
            raise ValueError('Código de moneda debe tener 3 caracteres en mayúsculas')
        return v
    
    @validator('tags')
    def validate_tags(cls, v) -> bool:
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    @validator('seo_keywords')
    def validate_keywords(cls, v) -> bool:
        return [kw.strip().lower() for kw in v if kw.strip()]
    
    @root_validator
    def validate_physical_product(cls, values) -> bool:
        product_type = values.get('product_type')
        if product_type == ProductType.PHYSICAL:
            requires_shipping = values.get('requires_shipping', True)
            if requires_shipping:
                dimensions = [
                    values.get('length'), values.get('width'), 
                    values.get('height'), values.get('weight')
                ]
                if not all(d is not None and d > 0 for d in dimensions):
                    raise ValueError('Productos físicos con envío requieren dimensiones válidas')
        return values
    
    @root_validator
    def validate_digital_product(cls, values) -> bool:
        product_type = values.get('product_type')
        if product_type == ProductType.DIGITAL:
            download_url = values.get('download_url')
            description = values.get('description')
            if not download_url and not description:
                raise ValueError('Productos digitales requieren URL de descarga o descripción')
        return values
    
    @root_validator
    def validate_prices(cls, values) -> bool:
        base_price = values.get('base_price_amount')
        sale_price = values.get('sale_price_amount')
        cost_price = values.get('cost_price_amount')
        
        if sale_price and base_price and sale_price >= base_price:
            raise ValueError('El precio de oferta debe ser menor al precio base')
        
        if cost_price and base_price and cost_price >= base_price:
            raise ValueError('El precio de costo debe ser menor al precio base')
        
        return values


class ProductUpdateRequest(BaseModel):
    """Request para actualizar productos"""
    
    name: Optional[str] = Field(None, min_length=2, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    short_description: Optional[str] = Field(None, max_length=500)
    status: Optional[ProductStatus] = None
    
    # Precios
    base_price_amount: Optional[float] = Field(None, gt=0)
    sale_price_amount: Optional[float] = Field(None, gt=0)
    cost_price_amount: Optional[float] = Field(None, gt=0)
    
    # Inventario
    inventory_quantity: Optional[int] = Field(None, ge=0)
    low_stock_threshold: Optional[int] = Field(None, ge=0)
    
    # Marketing
    featured: Optional[bool] = None
    tags: Optional[List[str]] = None
    
    # SEO
    seo_title: Optional[str] = Field(None, max_length=100)
    seo_description: Optional[str] = Field(None, max_length=300)
    seo_keywords: Optional[List[str]] = None
    
    # Atributos
    attributes: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class ProductSearchRequest(BaseModel):
    """Request para búsqueda avanzada de productos"""
    
    # Filtros de búsqueda
    query: Optional[str] = Field(None, description="Búsqueda por texto")
    sku: Optional[str] = Field(None, description="Búsqueda por SKU")
    category_id: Optional[str] = Field(None, description="ID de categoría")
    brand_id: Optional[str] = Field(None, description="ID de marca")
    status: Optional[ProductStatus] = Field(None, description="Estado del producto")
    product_type: Optional[ProductType] = Field(None, description="Tipo de producto")
    
    # Filtros de precio
    min_price: Optional[float] = Field(None, ge=0, description="Precio mínimo")
    max_price: Optional[float] = Field(None, ge=0, description="Precio máximo")
    on_sale: Optional[bool] = Field(None, description="En oferta")
    
    # Filtros de inventario
    in_stock: Optional[bool] = Field(None, description="En stock")
    low_stock: Optional[bool] = Field(None, description="Stock bajo")
    min_quantity: Optional[int] = Field(None, ge=0, description="Cantidad mínima")
    max_quantity: Optional[int] = Field(None, ge=0, description="Cantidad máxima")
    
    # Filtros de marketing
    featured: Optional[bool] = Field(None, description="Productos destacados")
    tags: Optional[List[str]] = Field(None, description="Etiquetas")
    
    # Filtros de fecha
    created_after: Optional[str] = Field(None, description="Creado después de (ISO)")
    created_before: Optional[str] = Field(None, description="Creado antes de (ISO)")
    updated_after: Optional[str] = Field(None, description="Actualizado después de (ISO)")
    updated_before: Optional[str] = Field(None, description="Actualizado antes de (ISO)")
    
    # Paginación
    page: int = Field(1, ge=1, description="Número de página")
    per_page: int = Field(20, ge=1, le=100, description="Elementos por página")
    
    # Ordenamiento
    sort_by: str = Field("updated_at", description="Campo de ordenamiento")
    sort_order: str = Field("desc", description="Orden (asc/desc)")
    
    @validator('sort_by')
    def validate_sort_by(cls, v) -> bool:
        allowed = [
            "name", "sku", "created_at", "updated_at", "published_at",
            "base_price", "inventory_quantity", "status"
        ]
        if v not in allowed:
            raise ValueError(f"sort_by debe ser uno de: {', '.join(allowed)}")
        return v
    
    @validator('sort_order')
    def validate_sort_order(cls, v) -> bool:
        if v not in ["asc", "desc"]:
            raise ValueError("sort_order debe ser 'asc' o 'desc'")
        return v


class EnhancedProductResponse(BaseModel):
    """Response mejorado para productos"""
    
    # Información básica
    id: str
    name: str
    description: str
    short_description: str
    sku: str
    product_type: str
    status: str
    brand_id: Optional[str]
    category_id: Optional[str]
    
    # Precios
    base_price: Optional[Dict[str, Any]]
    sale_price: Optional[Dict[str, Any]]
    cost_price: Optional[Dict[str, Any]]
    effective_price: Optional[Dict[str, Any]]
    price_type: str
    is_on_sale: bool
    discount_percentage: float
    profit_margin: Optional[float]
    
    # Inventario
    inventory_quantity: int
    low_stock_threshold: int
    inventory_tracking: str
    allow_backorder: bool
    is_low_stock: bool
    is_in_stock: bool
    
    # Propiedades físicas
    dimensions: Optional[Dict[str, Any]]
    requires_shipping: bool
    
    # Propiedades digitales
    download_url: Optional[str]
    download_limit: Optional[int]
    
    # SEO y Marketing
    seo_data: Dict[str, Any]
    tags: List[str]
    featured: bool
    
    # Atributos
    attributes: Dict[str, Any]
    custom_fields: Dict[str, Any]
    
    # Media
    images: List[str]
    videos: List[str]
    documents: List[str]
    
    # IA
    ai_generated_description: Optional[str]
    ai_confidence_score: Optional[float]
    ai_last_updated: Optional[str]
    
    # Timestamps
    created_at: str
    updated_at: str
    published_at: Optional[str]


class ProductListResponse(BaseModel):
    """Response para lista de productos"""
    
    products: List[EnhancedProductResponse]
    pagination: Dict[str, Any]
    filters_applied: Dict[str, Any]
    total: int
    page: int
    per_page: int
    total_pages: int
    has_next: bool
    has_prev: bool


class ProductStatsResponse(BaseModel):
    """Response para estadísticas de productos"""
    
    total_products: int
    by_status: Dict[str, int]
    by_type: Dict[str, int]
    by_category: Dict[str, int]
    inventory_stats: Dict[str, Any]
    pricing_stats: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    timestamp: str


# ============================================================================
# ENHANCED PRODUCT SERVICE
# ============================================================================

class EnhancedProductService:
    """
    Servicio empresarial mejorado para gestión de productos
    
    Funcionalidades:
    - CRUD completo de productos
    - Búsqueda avanzada con filtros
    - Gestión de inventario
    - Análisis y estadísticas
    - Integración con IA
    - Validaciones empresariales
    - Auditoría y logging
    """
    
    def __init__(self) -> Any:
        self.app = FastAPI(
            title="Enhanced Product Management API",
            description="API empresarial para gestión completa de productos",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Estado del servicio
        self.start_time = time.time()
        self.is_ready = False
        
        # Almacenamiento en memoria (en producción usar base de datos)
        self.products: Dict[str, EnhancedProductEntity] = {}
        self.sku_index: Dict[str, str] = {}  # sku -> product_id
        
        # Configurar middleware
        self._setup_middleware()
        
        # Configurar rutas
        self._setup_routes()
    
    def _setup_middleware(self) -> Any:
        """Configurar middleware de FastAPI"""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Logging personalizado
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
        """Configurar rutas de la API"""
        
        @self.app.on_event("startup")
        async def startup():
            """Inicializar el servicio"""
            logger.info("Iniciando Enhanced Product Service...")
            self.is_ready = True
            logger.info("Enhanced Product Service listo!")
        
        @self.app.get("/", response_model=Dict[str, Any])
        async def root():
            """Información del servicio"""
            return {
                "service": "Enhanced Product Management API",
                "version": "2.0.0",
                "status": "operational" if self.is_ready else "initializing",
                "uptime": time.time() - self.start_time,
                "docs": "/docs",
                "features": [
                    "CRUD completo de productos",
                    "Gestión de inventario",
                    "Precios dinámicos",
                    "Búsqueda avanzada",
                    "Integración IA",
                    "Análisis y estadísticas"
                ]
            }
        
        @self.app.post("/products", response_model=EnhancedProductResponse, status_code=201)
        async def create_product(request: EnhancedProductRequest):
            """Crear nuevo producto"""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Servicio no disponible")
            
            # Verificar SKU único
            if request.sku in self.sku_index:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Ya existe un producto con SKU: {request.sku}"
                )
            
            try:
                # Crear entidad
                product = self._create_product_from_request(request)
                
                # Validar
                errors = product.validate()
                if errors:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Errores de validación: {', '.join(errors)}"
                    )
                
                # Guardar
                self.products[product.id] = product
                self.sku_index[product.sku] = product.id
                
                # Generar descripción con IA si se solicita
                if request.auto_generate_description:
                    await self._generate_ai_description(product)
                
                logger.info(f"Producto creado: {product.id} - {product.name}")
                
                return self._product_to_response(product)
                
            except Exception as e:
                logger.error(f"Error creando producto: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/products/{product_id}", response_model=EnhancedProductResponse)
        async def get_product(product_id: str = Path(..., description="ID del producto")):
            """Obtener producto por ID"""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Servicio no disponible")
            
            product = self.products.get(product_id)
            if not product:
                raise HTTPException(status_code=404, detail="Producto no encontrado")
            
            return self._product_to_response(product)
        
        @self.app.put("/products/{product_id}", response_model=EnhancedProductResponse)
        async def update_product(
            product_id: str = Path(..., description="ID del producto"),
            request: ProductUpdateRequest = None
        ):
            """Actualizar producto"""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Servicio no disponible")
            
            product = self.products.get(product_id)
            if not product:
                raise HTTPException(status_code=404, detail="Producto no encontrado")
            
            try:
                # Actualizar campos
                self._update_product_from_request(product, request)
                
                # Validar
                errors = product.validate()
                if errors:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Errores de validación: {', '.join(errors)}"
                    )
                
                logger.info(f"Producto actualizado: {product.id}")
                
                return self._product_to_response(product)
                
            except Exception as e:
                logger.error(f"Error actualizando producto: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/products/{product_id}", status_code=204)
        async def delete_product(product_id: str = Path(..., description="ID del producto")):
            """Eliminar producto"""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Servicio no disponible")
            
            product = self.products.get(product_id)
            if not product:
                raise HTTPException(status_code=404, detail="Producto no encontrado")
            
            # Eliminar de índices
            self.sku_index.pop(product.sku, None)
            del self.products[product_id]
            
            logger.info(f"Producto eliminado: {product_id}")
        
        @self.app.post("/products/search", response_model=ProductListResponse)
        async def search_products(request: ProductSearchRequest):
            """Búsqueda avanzada de productos"""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Servicio no disponible")
            
            try:
                # Filtrar productos
                filtered_products = self._filter_products(request)
                
                # Ordenar
                sorted_products = self._sort_products(filtered_products, request.sort_by, request.sort_order)
                
                # Paginar
                total = len(sorted_products)
                start = (request.page - 1) * request.per_page
                end = start + request.per_page
                page_products = sorted_products[start:end]
                
                # Convertir a response
                product_responses = [self._product_to_response(p) for p in page_products]
                
                return ProductListResponse(
                    products=product_responses,
                    pagination={
                        "page": request.page,
                        "per_page": request.per_page,
                        "total": total,
                        "total_pages": (total + request.per_page - 1) // request.per_page,
                        "has_next": end < total,
                        "has_prev": request.page > 1
                    },
                    filters_applied=request.dict(exclude_unset=True),
                    total=total,
                    page=request.page,
                    per_page=request.per_page,
                    total_pages=(total + request.per_page - 1) // request.per_page,
                    has_next=end < total,
                    has_prev=request.page > 1
                )
                
            except Exception as e:
                logger.error(f"Error en búsqueda: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/products/stats", response_model=ProductStatsResponse)
        async def get_product_stats():
            """Obtener estadísticas de productos"""
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="Servicio no disponible")
            
            products = list(self.products.values())
            
            # Estadísticas por estado
            by_status = {}
            for status in ProductStatus:
                by_status[status.value] = len([p for p in products if p.status == status])
            
            # Estadísticas por tipo
            by_type = {}
            for ptype in ProductType:
                by_type[ptype.value] = len([p for p in products if p.product_type == ptype])
            
            # Estadísticas de inventario
            total_inventory = sum(p.inventory_quantity for p in products)
            low_stock_count = len([p for p in products if p.is_low_stock()])
            out_of_stock_count = len([p for p in products if not p.is_in_stock()])
            
            # Estadísticas de precios
            prices = [p.get_effective_price().amount for p in products if p.get_effective_price()]
            avg_price = float(sum(prices) / len(prices)) if prices else 0
            min_price = float(min(prices)) if prices else 0
            max_price = float(max(prices)) if prices else 0
            
            return ProductStatsResponse(
                total_products=len(products),
                by_status=by_status,
                by_type=by_type,
                by_category={},  # TODO: Implementar cuando se agreguen categorías
                inventory_stats={
                    "total_inventory": total_inventory,
                    "low_stock_count": low_stock_count,
                    "out_of_stock_count": out_of_stock_count,
                    "avg_inventory_per_product": total_inventory / len(products) if products else 0
                },
                pricing_stats={
                    "avg_price": avg_price,
                    "min_price": min_price,
                    "max_price": max_price,
                    "products_on_sale": len([p for p in products if p.is_on_sale()])
                },
                performance_metrics={
                    "active_products": by_status.get("active", 0),
                    "featured_products": len([p for p in products if p.featured]),
                    "products_with_ai_descriptions": len([p for p in products if p.ai_generated_description])
                },
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _create_product_from_request(self, request: EnhancedProductRequest) -> EnhancedProductEntity:
        """Crear entidad de producto desde request"""
        
        product = EnhancedProductEntity(
            name=request.name,
            description=request.description,
            short_description=request.short_description,
            sku=request.sku,
            product_type=request.product_type,
            brand_id=request.brand_id,
            category_id=request.category_id
        )
        
        # Configurar precios
        if request.base_price_amount:
            product.set_price(
                Money(amount=Decimal(str(request.base_price_amount)), currency=request.base_price_currency),
                request.price_type
            )
        
        if request.sale_price_amount:
            product.set_sale_price(
                Money(amount=Decimal(str(request.sale_price_amount)), currency=request.base_price_currency)
            )
        
        if request.cost_price_amount:
            product.cost_price = Money(
                amount=Decimal(str(request.cost_price_amount)), 
                currency=request.base_price_currency
            )
        
        # Configurar inventario
        product.inventory_quantity = request.inventory_quantity
        product.low_stock_threshold = request.low_stock_threshold
        product.inventory_tracking = request.inventory_tracking
        product.allow_backorder = request.allow_backorder
        
        # Configurar dimensiones
        if all(x is not None for x in [request.length, request.width, request.height, request.weight]):
            product.dimensions = Dimensions(
                length=request.length,
                width=request.width,
                height=request.height,
                weight=request.weight
            )
        
        # Configurar propiedades
        product.requires_shipping = request.requires_shipping
        product.download_url = request.download_url
        product.download_limit = request.download_limit
        
        # Configurar SEO
        product.seo_data = SEOData(
            title=request.seo_title,
            description=request.seo_description,
            keywords=request.seo_keywords,
            slug=request.slug
        )
        
        # Configurar marketing
        product.tags = set(request.tags)
        product.featured = request.featured
        
        # Configurar atributos
        product.attributes = request.attributes
        product.custom_fields = request.custom_fields
        
        # Configurar media
        product.images = request.images
        product.videos = request.videos
        product.documents = request.documents
        
        return product
    
    def _update_product_from_request(self, product: EnhancedProductEntity, request: ProductUpdateRequest):
        """Actualizar producto desde request"""
        
        if request.name is not None:
            product.name = request.name
        if request.description is not None:
            product.description = request.description
        if request.short_description is not None:
            product.short_description = request.short_description
        if request.status is not None:
            product.status = request.status
        
        # Actualizar precios
        if request.base_price_amount is not None:
            currency = product.base_price.currency if product.base_price else "USD"
            product.set_price(Money(amount=Decimal(str(request.base_price_amount)), currency=currency))
        
        if request.sale_price_amount is not None:
            currency = product.base_price.currency if product.base_price else "USD"
            product.set_sale_price(Money(amount=Decimal(str(request.sale_price_amount)), currency=currency))
        
        if request.cost_price_amount is not None:
            currency = product.base_price.currency if product.base_price else "USD"
            product.cost_price = Money(amount=Decimal(str(request.cost_price_amount)), currency=currency)
        
        # Actualizar inventario
        if request.inventory_quantity is not None:
            product.update_inventory(request.inventory_quantity, "set")
        if request.low_stock_threshold is not None:
            product.low_stock_threshold = request.low_stock_threshold
        
        # Actualizar marketing
        if request.featured is not None:
            product.featured = request.featured
        if request.tags is not None:
            product.tags = set(request.tags)
        
        # Actualizar SEO
        if request.seo_title is not None:
            product.seo_data.title = request.seo_title
        if request.seo_description is not None:
            product.seo_data.description = request.seo_description
        if request.seo_keywords is not None:
            product.seo_data.keywords = request.seo_keywords
        
        # Actualizar atributos
        if request.attributes is not None:
            product.attributes.update(request.attributes)
        if request.custom_fields is not None:
            product.custom_fields.update(request.custom_fields)
        
        product.updated_at = datetime.utcnow()
    
    def _filter_products(self, request: ProductSearchRequest) -> List[EnhancedProductEntity]:
        """Filtrar productos según criterios de búsqueda"""
        
        products = list(self.products.values())
        
        # Filtro por texto
        if request.query:
            query_lower = request.query.lower()
            products = [
                p for p in products 
                if (query_lower in p.name.lower() or 
                    query_lower in p.description.lower() or
                    query_lower in p.sku.lower() or
                    any(query_lower in tag for tag in p.tags))
            ]
        
        # Filtro por SKU
        if request.sku:
            products = [p for p in products if request.sku.upper() in p.sku.upper()]
        
        # Filtro por estado
        if request.status:
            products = [p for p in products if p.status == request.status]
        
        # Filtro por tipo
        if request.product_type:
            products = [p for p in products if p.product_type == request.product_type]
        
        # Filtro por precio
        if request.min_price is not None:
            products = [
                p for p in products 
                if p.get_effective_price() and p.get_effective_price().amount >= request.min_price
            ]
        
        if request.max_price is not None:
            products = [
                p for p in products 
                if p.get_effective_price() and p.get_effective_price().amount <= request.max_price
            ]
        
        # Filtro por oferta
        if request.on_sale is not None:
            products = [p for p in products if p.is_on_sale() == request.on_sale]
        
        # Filtro por stock
        if request.in_stock is not None:
            products = [p for p in products if p.is_in_stock() == request.in_stock]
        
        if request.low_stock is not None:
            products = [p for p in products if p.is_low_stock() == request.low_stock]
        
        # Filtro por cantidad
        if request.min_quantity is not None:
            products = [p for p in products if p.inventory_quantity >= request.min_quantity]
        
        if request.max_quantity is not None:
            products = [p for p in products if p.inventory_quantity <= request.max_quantity]
        
        # Filtro por destacado
        if request.featured is not None:
            products = [p for p in products if p.featured == request.featured]
        
        # Filtro por etiquetas
        if request.tags:
            tag_set = set(tag.lower() for tag in request.tags)
            products = [p for p in products if tag_set.intersection(p.tags)]
        
        return products
    
    def _sort_products(self, products: List[EnhancedProductEntity], sort_by: str, sort_order: str) -> List[EnhancedProductEntity]:
        """Ordenar productos"""
        
        reverse = sort_order == "desc"
        
        if sort_by == "name":
            return sorted(products, key=lambda p: p.name.lower(), reverse=reverse)
        elif sort_by == "sku":
            return sorted(products, key=lambda p: p.sku, reverse=reverse)
        elif sort_by == "created_at":
            return sorted(products, key=lambda p: p.created_at, reverse=reverse)
        elif sort_by == "updated_at":
            return sorted(products, key=lambda p: p.updated_at, reverse=reverse)
        elif sort_by == "published_at":
            return sorted(products, key=lambda p: p.published_at or datetime.min, reverse=reverse)
        elif sort_by == "base_price":
            return sorted(
                products, 
                key=lambda p: p.base_price.amount if p.base_price else 0, 
                reverse=reverse
            )
        elif sort_by == "inventory_quantity":
            return sorted(products, key=lambda p: p.inventory_quantity, reverse=reverse)
        elif sort_by == "status":
            return sorted(products, key=lambda p: p.status.value, reverse=reverse)
        else:
            return products
    
    def _product_to_response(self, product: EnhancedProductEntity) -> EnhancedProductResponse:
        """Convertir entidad a response"""
        
        data = product.to_dict()
        
        return EnhancedProductResponse(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            short_description=data["short_description"],
            sku=data["sku"],
            product_type=data["product_type"],
            status=data["status"],
            brand_id=data["brand_id"],
            category_id=data["category_id"],
            base_price=data["base_price"],
            sale_price=data["sale_price"],
            cost_price=data["cost_price"],
            effective_price=data["effective_price"],
            price_type=data["price_type"],
            is_on_sale=data["is_on_sale"],
            discount_percentage=data["discount_percentage"],
            profit_margin=data["profit_margin"],
            inventory_quantity=data["inventory_quantity"],
            low_stock_threshold=data["low_stock_threshold"],
            inventory_tracking=data["inventory_tracking"],
            allow_backorder=data["allow_backorder"],
            is_low_stock=data["is_low_stock"],
            is_in_stock=data["is_in_stock"],
            dimensions=data["dimensions"],
            requires_shipping=data["requires_shipping"],
            download_url=data["download_url"],
            download_limit=data["download_limit"],
            seo_data=data["seo_data"],
            tags=data["tags"],
            featured=data["featured"],
            attributes=data["attributes"],
            custom_fields=data["custom_fields"],
            images=data["images"],
            videos=data["videos"],
            documents=data.get("documents", []),
            ai_generated_description=data["ai_generated_description"],
            ai_confidence_score=data["ai_confidence_score"],
            ai_last_updated=data["ai_last_updated"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            published_at=data["published_at"]
        )
    
    async def _generate_ai_description(self, product: EnhancedProductEntity):
        """Generar descripción con IA (placeholder)"""
        
        # TODO: Integrar con servicio de IA
        ai_description = f"Descripción automática para {product.name} - Producto de alta calidad con características excepcionales."
        product.set_ai_description(ai_description, confidence=0.85)
        
        logger.info(f"Descripción IA generada para producto: {product.id}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8001, **kwargs):
        """Ejecutar el servicio"""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            **kwargs
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Crear y ejecutar servicio
    service = EnhancedProductService()
    service.run(port=8001) 