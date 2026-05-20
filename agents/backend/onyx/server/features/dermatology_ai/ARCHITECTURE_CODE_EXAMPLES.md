# Ejemplos de Código - Arquitectura V8.0

## 📋 Tabla de Contenidos

1. [Estructura de Feature Modules](#estructura-de-feature-modules)
2. [Implementación de Servicios](#implementación-de-servicios)
3. [API Layer Consolidado](#api-layer-consolidado)
4. [Composition Root Mejorado](#composition-root-mejorado)
5. [Testing](#testing)

---

## 🏗️ Estructura de Feature Modules

### Ejemplo Completo: Feature `recommendations`

```
features/recommendations/
├── __init__.py
├── services/
│   ├── __init__.py
│   ├── skincare_recommender.py
│   ├── intelligent_recommender.py
│   └── ml_recommender.py
├── controllers/
│   ├── __init__.py
│   └── recommendation_controller.py
└── schemas/
    ├── __init__.py
    └── recommendation_schemas.py
```

### `features/recommendations/__init__.py`

```python
"""
Recommendations Feature Module

Este módulo contiene todos los servicios, controllers y schemas
relacionados con recomendaciones de skincare.
"""

from .services.skincare_recommender import SkincareRecommender
from .services.intelligent_recommender import IntelligentRecommender
from .services.ml_recommender import MLRecommender

__all__ = [
    "SkincareRecommender",
    "IntelligentRecommender",
    "MLRecommender",
]

def register_recommendation_services(service_factory):
    """
    Registra todos los servicios de recomendaciones.
    
    Args:
        service_factory: Instancia del ServiceFactory
    """
    from core.service_factory import ServiceScope
    
    # Registrar SkincareRecommender
    service_factory.register(
        name="recommendations.skincare",
        factory=lambda: SkincareRecommender(
            product_repository=service_factory.get("product_repository"),
            analysis_service=service_factory.get("analysis_service")
        ),
        dependencies=["product_repository", "analysis_service"],
        scope=ServiceScope.SINGLETON
    )
    
    # Registrar IntelligentRecommender
    service_factory.register(
        name="recommendations.intelligent",
        factory=lambda: IntelligentRecommender(
            skincare_recommender=service_factory.get("recommendations.skincare"),
            ml_model=service_factory.get("ml_model")
        ),
        dependencies=["recommendations.skincare", "ml_model"],
        scope=ServiceScope.SINGLETON
    )
    
    # Registrar MLRecommender
    service_factory.register(
        name="recommendations.ml",
        factory=lambda: MLRecommender(
            ml_model=service_factory.get("ml_model"),
            training_data=service_factory.get("training_data")
        ),
        dependencies=["ml_model", "training_data"],
        scope=ServiceScope.SINGLETON
    )
```

### `features/recommendations/services/__init__.py`

```python
"""Services for recommendations feature"""

from .skincare_recommender import SkincareRecommender
from .intelligent_recommender import IntelligentRecommender
from .ml_recommender import MLRecommender

__all__ = [
    "SkincareRecommender",
    "IntelligentRecommender",
    "MLRecommender",
]
```

---

## 🔧 Implementación de Servicios

### Ejemplo: `SkincareRecommender` Refactorizado

```python
# features/recommendations/services/skincare_recommender.py
"""
Servicio de recomendaciones de skincare.

Implementa IRecommendationService siguiendo arquitectura hexagonal.
"""

from typing import Dict, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum
import logging

from core.domain.interfaces import IRecommendationService, IProductRepository, IAnalysisService
from core.domain.entities import Analysis

logger = logging.getLogger(__name__)


class ProductCategory(str, Enum):
    """Categorías de productos"""
    CLEANSER = "cleanser"
    MOISTURIZER = "moisturizer"
    SERUM = "serum"
    SUNSCREEN = "sunscreen"
    TONER = "toner"
    EXFOLIANT = "exfoliant"
    MASK = "mask"
    EYE_CREAM = "eye_cream"


@dataclass
class SkincareProduct:
    """Producto de skincare recomendado"""
    name: str
    category: ProductCategory
    description: str
    key_ingredients: List[str]
    usage_frequency: str
    priority: int  # 1-5, donde 1 es más prioritario


@dataclass
class SkincareRoutine:
    """Rutina de skincare recomendada"""
    morning_routine: List[SkincareProduct]
    evening_routine: List[SkincareProduct]
    weekly_treatments: List[SkincareProduct]
    tips: List[str]


class SkincareRecommender(IRecommendationService):
    """
    Genera recomendaciones personalizadas de skincare.
    
    Implementa IRecommendationService para seguir arquitectura hexagonal.
    """
    
    def __init__(
        self,
        product_repository: IProductRepository,
        analysis_service: Optional[IAnalysisService] = None
    ):
        """
        Inicializa el recomendador con dependencias inyectadas.
        
        Args:
            product_repository: Repositorio para acceder a productos
            analysis_service: Servicio opcional para análisis avanzado
        """
        if not product_repository:
            raise ValueError("product_repository is required")
        
        self.product_repository = product_repository
        self.analysis_service = analysis_service
        self._product_cache: Optional[Dict] = None
        logger.info("SkincareRecommender initialized")
    
    async def generate_recommendations(
        self,
        analysis_result: Dict,
        user_id: Optional[str] = None
    ) -> Dict:
        """
        Genera recomendaciones basadas en análisis de piel.
        
        Args:
            analysis_result: Resultado del análisis de piel
            user_id: ID del usuario (opcional, para personalización)
            
        Returns:
            Diccionario con recomendaciones completas
            
        Raises:
            ValueError: Si analysis_result es inválido
        """
        # Validar entrada
        if not analysis_result:
            raise ValueError("analysis_result is required")
        
        if not isinstance(analysis_result, dict):
            raise ValueError("analysis_result must be a dictionary")
        
        logger.info(f"Generating recommendations for user: {user_id}")
        
        try:
            # Obtener productos del repositorio
            products = await self.product_repository.get_all()
            
            # Extraer información del análisis
            quality_scores = analysis_result.get("quality_scores", {})
            conditions = analysis_result.get("conditions", [])
            skin_type = analysis_result.get("skin_type", "normal")
            priorities = analysis_result.get("recommendations_priority", [])
            
            # Generar rutina personalizada
            routine = await self._create_routine(
                skin_type=skin_type,
                quality_scores=quality_scores,
                conditions=conditions,
                priorities=priorities,
                products=products,
                user_id=user_id
            )
            
            # Generar recomendaciones específicas
            specific_recommendations = await self._get_specific_recommendations(
                quality_scores=quality_scores,
                conditions=conditions,
                priorities=priorities
            )
            
            # Generar tips generales
            tips = await self._generate_tips(
                skin_type=skin_type,
                conditions=conditions,
                quality_scores=quality_scores
            )
            
            return {
                "routine": {
                    "morning": [self._product_to_dict(p) for p in routine.morning_routine],
                    "evening": [self._product_to_dict(p) for p in routine.evening_routine],
                    "weekly": [self._product_to_dict(p) for p in routine.weekly_treatments]
                },
                "specific_recommendations": specific_recommendations,
                "tips": tips,
                "skin_type": skin_type,
                "priorities": priorities
            }
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            raise
    
    async def _create_routine(
        self,
        skin_type: str,
        quality_scores: Dict,
        conditions: List[Dict],
        priorities: List[str],
        products: List[Dict],
        user_id: Optional[str] = None
    ) -> SkincareRoutine:
        """
        Crea rutina personalizada basada en análisis.
        
        Args:
            skin_type: Tipo de piel
            quality_scores: Puntuaciones de calidad
            conditions: Condiciones detectadas
            priorities: Prioridades de mejora
            products: Lista de productos disponibles
            user_id: ID del usuario
            
        Returns:
            SkincareRoutine con rutinas personalizadas
        """
        # Filtrar productos por tipo de piel
        filtered_products = self._filter_products_by_skin_type(products, skin_type)
        
        # Crear rutina de mañana
        morning_routine = self._create_morning_routine(
            filtered_products,
            quality_scores,
            conditions
        )
        
        # Crear rutina de tarde
        evening_routine = self._create_evening_routine(
            filtered_products,
            quality_scores,
            conditions
        )
        
        # Crear tratamientos semanales
        weekly_treatments = self._create_weekly_treatments(
            filtered_products,
            quality_scores,
            conditions
        )
        
        return SkincareRoutine(
            morning_routine=morning_routine,
            evening_routine=evening_routine,
            weekly_treatments=weekly_treatments,
            tips=[]
        )
    
    def _filter_products_by_skin_type(
        self,
        products: List[Dict],
        skin_type: str
    ) -> List[Dict]:
        """Filtra productos por tipo de piel"""
        # Implementación...
        return products
    
    def _create_morning_routine(
        self,
        products: List[Dict],
        quality_scores: Dict,
        conditions: List[Dict]
    ) -> List[SkincareProduct]:
        """Crea rutina de mañana"""
        # Implementación...
        return []
    
    def _create_evening_routine(
        self,
        products: List[Dict],
        quality_scores: Dict,
        conditions: List[Dict]
    ) -> List[SkincareProduct]:
        """Crea rutina de tarde"""
        # Implementación...
        return []
    
    def _create_weekly_treatments(
        self,
        products: List[Dict],
        quality_scores: Dict,
        conditions: List[Dict]
    ) -> List[SkincareProduct]:
        """Crea tratamientos semanales"""
        # Implementación...
        return []
    
    async def _get_specific_recommendations(
        self,
        quality_scores: Dict,
        conditions: List[Dict],
        priorities: List[str]
    ) -> List[Dict]:
        """Genera recomendaciones específicas"""
        # Implementación...
        return []
    
    async def _generate_tips(
        self,
        skin_type: str,
        conditions: List[Dict],
        quality_scores: Dict
    ) -> List[str]:
        """Genera tips personalizados"""
        # Implementación...
        return []
    
    def _product_to_dict(self, product: SkincareProduct) -> Dict:
        """Convierte producto a diccionario"""
        return {
            "name": product.name,
            "category": product.category.value,
            "description": product.description,
            "key_ingredients": product.key_ingredients,
            "usage_frequency": product.usage_frequency,
            "priority": product.priority
        }
```

---

## 🌐 API Layer Consolidado

### Estructura de API V1

```
api/
├── v1/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── analysis.py
│   │   ├── recommendations.py
│   │   ├── tracking.py
│   │   └── products.py
│   └── schemas/
│       ├── __init__.py
│       ├── analysis_schemas.py
│       └── recommendation_schemas.py
```

### `api/v1/routes/recommendations.py`

```python
"""
API Routes for Recommendations Feature
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Optional
from pydantic import BaseModel, Field
import logging

from core.service_factory import get_service_factory
from features.recommendations.services.skincare_recommender import SkincareRecommender

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/recommendations",
    tags=["recommendations"]
)


class AnalysisResultRequest(BaseModel):
    """Request schema for analysis result"""
    quality_scores: Dict = Field(..., description="Quality scores from analysis")
    conditions: list = Field(default_factory=list, description="Detected conditions")
    skin_type: str = Field(..., description="Skin type")
    recommendations_priority: list = Field(default_factory=list, description="Priority areas")


class RecommendationResponse(BaseModel):
    """Response schema for recommendations"""
    routine: Dict
    specific_recommendations: list
    tips: list
    skin_type: str
    priorities: list


def get_recommendation_service() -> SkincareRecommender:
    """
    Dependency injection for recommendation service.
    
    Returns:
        SkincareRecommender instance
    """
    factory = get_service_factory()
    try:
        return factory.get("recommendations.skincare")
    except Exception as e:
        logger.error(f"Failed to get recommendation service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service unavailable"
        )


@router.post(
    "/generate",
    response_model=RecommendationResponse,
    summary="Generate skincare recommendations",
    description="Generates personalized skincare recommendations based on skin analysis"
)
async def generate_recommendations(
    request: AnalysisResultRequest,
    user_id: Optional[str] = None,
    recommender: SkincareRecommender = Depends(get_recommendation_service)
):
    """
    Generate skincare recommendations.
    
    Args:
        request: Analysis result data
        user_id: Optional user ID for personalization
        recommender: Injected recommendation service
        
    Returns:
        RecommendationResponse with personalized recommendations
    """
    try:
        analysis_result = request.dict()
        
        recommendations = await recommender.generate_recommendations(
            analysis_result=analysis_result,
            user_id=user_id
        )
        
        return RecommendationResponse(**recommendations)
    
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )


@router.get(
    "/health",
    summary="Recommendations service health check"
)
async def health_check(
    recommender: SkincareRecommender = Depends(get_recommendation_service)
):
    """Check if recommendations service is healthy"""
    return {
        "status": "healthy",
        "service": "recommendations"
    }
```

### `api/v1/__init__.py`

```python
"""
API Version 1

Consolidated API layer with organized routes and schemas.
"""

from fastapi import FastAPI
from .routes import analysis, recommendations, tracking, products

def register_v1_routes(app: FastAPI):
    """
    Register all v1 API routes.
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(analysis.router)
    app.include_router(recommendations.router)
    app.include_router(tracking.router)
    app.include_router(products.router)
```

---

## 🔌 Composition Root Mejorado

### `core/composition_root.py` (Mejoras)

```python
"""
Composition Root - Dependency Injection Container (Improved V8.0)

Wires up all dependencies following Dependency Inversion Principle
and Clean Architecture patterns with health checks and lifecycle management.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from .service_factory import get_service_factory, ServiceScope
from .domain.interfaces import (
    IAnalysisService,
    IRecommendationService,
    IAnalysisRepository,
    IProductRepository,
)
from .infrastructure.adapters import IDatabaseAdapter
from .adapter_factory import AdapterFactory
from .repository_factory import RepositoryFactory
from .service_registration import ServiceRegistration
from .domain_service_factory import DomainServiceFactory
from .use_case_factory import UseCaseFactory

logger = logging.getLogger(__name__)


class LifecycleStage(str, Enum):
    """Lifecycle stages of composition root"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class CompositionRoot:
    """
    Composition root for dependency injection (Improved V8.0).
    
    Features:
    - Health checks for all dependencies
    - Robust lifecycle management
    - Dependency graph visualization
    - Better error handling with cleanup
    """
    
    def __init__(self) -> None:
        """Initialize composition root"""
        self.service_factory = get_service_factory()
        self._lifecycle_stage = LifecycleStage.UNINITIALIZED
        self._initialization_lock = asyncio.Lock()
        self._database_adapter: Optional[IDatabaseAdapter] = None
        self._config: Optional[Dict[str, Any]] = None
        self._use_case_cache: Dict[str, Any] = {}
        self._dependency_graph: Dict[str, List[str]] = {}
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize composition root with configuration.
        
        Uses lock to prevent concurrent initialization.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            RuntimeError: If already initialized
            Exception: If initialization fails (resources are cleaned up)
        """
        async with self._initialization_lock:
            if self._lifecycle_stage != LifecycleStage.UNINITIALIZED:
                raise RuntimeError(f"Composition root already {self._lifecycle_stage.value}")
            
            self._lifecycle_stage = LifecycleStage.INITIALIZING
            self._config = config
            
            try:
                # Initialize adapters
                await self._initialize_adapters(config)
                
                # Initialize repositories
                await self._initialize_repositories()
                
                # Initialize domain services
                await self._initialize_domain_services()
                
                # Register feature services
                await self._register_feature_services()
                
                # Build dependency graph
                self._build_dependency_graph()
                
                self._lifecycle_stage = LifecycleStage.READY
                logger.info("✅ Composition root initialized successfully")
                
            except Exception as e:
                self._lifecycle_stage = LifecycleStage.FAILED
                logger.error(f"Failed to initialize composition root: {e}", exc_info=True)
                await self._cleanup()
                raise
    
    async def _initialize_adapters(self, config: Dict[str, Any]) -> None:
        """Initialize all adapters"""
        database_adapter = await AdapterFactory.create_database_adapter(config)
        self._database_adapter = database_adapter
        
        adapters = await asyncio.gather(
            AdapterFactory.create_cache_adapter(config),
            AdapterFactory.create_image_processor_adapter(config),
            AdapterFactory.create_event_publisher_adapter(config),
            return_exceptions=True
        )
        
        cache_adapter, image_processor_adapter, event_publisher_adapter = adapters
        
        ServiceRegistration.register_adapters(
            self.service_factory,
            {
                "image_processor": image_processor_adapter,
                "cache": cache_adapter,
                "event_publisher": event_publisher_adapter,
            }
        )
    
    async def _initialize_repositories(self) -> None:
        """Initialize all repositories"""
        if not self._database_adapter:
            raise RuntimeError("Database adapter not initialized")
        
        repositories = RepositoryFactory.create_repositories(self._database_adapter)
        ServiceRegistration.register_repositories(self.service_factory, repositories)
    
    async def _initialize_domain_services(self) -> None:
        """Initialize domain services"""
        domain_services = await asyncio.gather(
            DomainServiceFactory.create_analysis_service(self.service_factory),
            DomainServiceFactory.create_recommendation_service(self.service_factory),
            return_exceptions=True
        )
        
        analysis_service, recommendation_service = domain_services
        
        ServiceRegistration.register_domain_services(
            self.service_factory,
            {
                "analysis_service": analysis_service,
                "recommendation_service": recommendation_service,
            }
        )
    
    async def _register_feature_services(self) -> None:
        """Register all feature services"""
        from features.recommendations import register_recommendation_services
        from features.analysis import register_analysis_services
        from features.tracking import register_tracking_services
        from features.products import register_product_services
        # ... más features
        
        register_recommendation_services(self.service_factory)
        register_analysis_services(self.service_factory)
        register_tracking_services(self.service_factory)
        register_product_services(self.service_factory)
        # ...
    
    def _build_dependency_graph(self) -> None:
        """Build dependency graph for visualization"""
        self._dependency_graph = {
            "AnalyzeImageUseCase": [
                "IAnalysisRepository",
                "IImageProcessor",
                "IAnalysisService",
                "IEventPublisher"
            ],
            "IAnalysisService": [
                "IImageProcessor",
                "MLModelManager"
            ],
            "SkincareRecommender": [
                "IProductRepository",
                "IAnalysisService"
            ],
            # ... más dependencias
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of all dependencies.
        
        Returns:
            Dictionary with health status of all components
        """
        if self._lifecycle_stage != LifecycleStage.READY:
            return {
                "status": "unhealthy",
                "stage": self._lifecycle_stage.value,
                "message": "Composition root not ready"
            }
        
        health = {
            "status": "healthy",
            "stage": self._lifecycle_stage.value,
            "components": {}
        }
        
        # Check database
        health["components"]["database"] = await self._check_database()
        
        # Check cache
        health["components"]["cache"] = await self._check_cache()
        
        # Check image processor
        health["components"]["image_processor"] = await self._check_image_processor()
        
        # Determine overall status
        component_statuses = [
            comp.get("status") for comp in health["components"].values()
        ]
        if "unhealthy" in component_statuses:
            health["status"] = "degraded"
        if all(s == "unhealthy" for s in component_statuses):
            health["status"] = "unhealthy"
        
        return health
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            if self._database_adapter:
                if hasattr(self._database_adapter, "ping"):
                    await self._database_adapter.ping()
                return {
                    "status": "healthy",
                    "adapter": type(self._database_adapter).__name__
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
        return {"status": "not_initialized"}
    
    async def _check_cache(self) -> Dict[str, Any]:
        """Check cache connectivity"""
        try:
            cache_adapter = self.service_factory.get("cache")
            if cache_adapter and hasattr(cache_adapter, "ping"):
                await cache_adapter.ping()
                return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
        return {"status": "not_initialized"}
    
    async def _check_image_processor(self) -> Dict[str, Any]:
        """Check image processor availability"""
        try:
            processor = self.service_factory.get("image_processor")
            if processor:
                return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
        return {"status": "not_initialized"}
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get dependency graph for debugging.
        
        Returns:
            Dictionary mapping services to their dependencies
        """
        return self._dependency_graph.copy()
    
    async def _cleanup(self) -> None:
        """Cleanup resources on initialization failure"""
        if self._database_adapter:
            try:
                if hasattr(self._database_adapter, "close"):
                    await self._database_adapter.close()
            except Exception as e:
                logger.error(f"Error closing database adapter: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown and cleanup resources"""
        if self._lifecycle_stage == LifecycleStage.SHUTDOWN:
            return
        
        self._lifecycle_stage = LifecycleStage.SHUTTING_DOWN
        
        if self._database_adapter:
            try:
                if hasattr(self._database_adapter, "close"):
                    await self._database_adapter.close()
            except Exception as e:
                logger.error(f"Error closing database: {e}")
        
        self.service_factory.clear_request_scope()
        self._use_case_cache.clear()
        self._lifecycle_stage = LifecycleStage.SHUTDOWN
        logger.info("✅ Composition root shutdown")
```

---

## 🧪 Testing

### Ejemplo: Test de `SkincareRecommender`

```python
# tests/features/recommendations/test_skincare_recommender.py
import pytest
from unittest.mock import Mock, AsyncMock
from features.recommendations.services.skincare_recommender import (
    SkincareRecommender,
    SkincareProduct,
    ProductCategory
)

@pytest.fixture
def mock_product_repository():
    """Mock product repository"""
    repo = Mock()
    repo.get_all = AsyncMock(return_value=[
        {
            "name": "Cleanser",
            "category": "cleanser",
            "suitable_for": ["normal", "oily"]
        }
    ])
    return repo

@pytest.fixture
def recommender(mock_product_repository):
    """Create recommender instance"""
    return SkincareRecommender(
        product_repository=mock_product_repository
    )

@pytest.mark.asyncio
async def test_generate_recommendations_success(recommender, mock_product_repository):
    """Test successful recommendation generation"""
    analysis_result = {
        "quality_scores": {"overall_score": 75.0},
        "conditions": [],
        "skin_type": "normal",
        "recommendations_priority": []
    }
    
    result = await recommender.generate_recommendations(analysis_result)
    
    assert "routine" in result
    assert "specific_recommendations" in result
    assert "tips" in result
    mock_product_repository.get_all.assert_called_once()

@pytest.mark.asyncio
async def test_generate_recommendations_invalid_input(recommender):
    """Test recommendation generation with invalid input"""
    with pytest.raises(ValueError, match="analysis_result is required"):
        await recommender.generate_recommendations(None)
    
    with pytest.raises(ValueError, match="must be a dictionary"):
        await recommender.generate_recommendations("invalid")
```

---

**Versión:** 8.0.0  
**Fecha:** 2024




