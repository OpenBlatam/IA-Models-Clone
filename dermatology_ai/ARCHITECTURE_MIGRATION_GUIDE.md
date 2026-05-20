# Guía de Migración Arquitectónica V8.0 - Dermatology AI

## 📋 Tabla de Contenidos

1. [Mapeo de Servicios](#mapeo-de-servicios)
2. [Ejemplos de Refactorización](#ejemplos-de-refactorización)
3. [Patrones de Diseño](#patrones-de-diseño)
4. [Guía Paso a Paso](#guía-paso-a-paso)
5. [Checklist de Migración](#checklist-de-migración)

---

## 🗺️ Mapeo de Servicios

### Feature: Analysis (`features/analysis/`)

**Servicios a migrar:**
- `image_analysis_advanced.py` → `features/analysis/services/image_analysis.py`
- `video_analysis_advanced.py` → `features/analysis/services/video_analysis.py`
- `advanced_ml_analysis.py` → `features/analysis/services/ml_analysis.py`
- `advanced_texture_analysis.py` → `features/analysis/services/texture_analysis.py`
- `advanced_texture_ml.py` → `features/analysis/services/texture_ml.py`
- `multi_angle_analysis.py` → `features/analysis/services/multi_angle_analysis.py`
- `body_area_analyzer.py` → `features/analysis/services/body_area_analyzer.py`
- `format_analysis.py` → `features/analysis/services/format_analysis.py`
- `resolution_analysis.py` → `features/analysis/services/resolution_analysis.py`
- `lighting_analysis.py` → `features/analysis/services/lighting_analysis.py`
- `natural_lighting_analysis.py` → `features/analysis/services/natural_lighting_analysis.py`
- `device_analysis.py` → `features/analysis/services/device_analysis.py`
- `distance_analysis.py` → `features/analysis/services/distance_analysis.py`
- `age_analysis.py` → `features/analysis/services/age_analysis.py`
- `ai_photo_analysis.py` → `features/analysis/services/ai_photo_analysis.py`
- `image_processor.py` → `features/analysis/services/image_processor.py`
- `video_processor.py` → `features/analysis/services/video_processor.py`

### Feature: Recommendations (`features/recommendations/`)

**Servicios a migrar:**
- `skincare_recommender.py` → `features/recommendations/services/skincare_recommender.py`
- `intelligent_recommender.py` → `features/recommendations/services/intelligent_recommender.py`
- `ml_recommender.py` → `features/recommendations/services/ml_recommender.py`
- `smart_recommender.py` → `features/recommendations/services/smart_recommender.py`
- `age_based_recommendations.py` → `features/recommendations/services/age_based_recommendations.py`
- `budget_based_recommendations.py` → `features/recommendations/services/budget_based_recommendations.py`
- `budget_recommendations.py` → `features/recommendations/services/budget_recommendations.py`
- `monthly_budget_recommendations.py` → `features/recommendations/services/monthly_budget_recommendations.py`
- `ethnic_skin_recommendations.py` → `features/recommendations/services/ethnic_skin_recommendations.py`
- `fitness_based_recommendations.py` → `features/recommendations/services/fitness_based_recommendations.py`
- `genetic_recommendations.py` → `features/recommendations/services/genetic_recommendations.py`
- `lifestyle_recommendations.py` → `features/recommendations/services/lifestyle_recommendations.py`
- `medication_recommendations.py` → `features/recommendations/services/medication_recommendations.py`
- `occupation_recommendations.py` → `features/recommendations/services/occupation_recommendations.py`
- `seasonal_recommendations.py` → `features/recommendations/services/seasonal_recommendations.py`
- `time_based_recommendations.py` → `features/recommendations/services/time_based_recommendations.py`
- `water_type_recommendations.py` → `features/recommendations/services/water_type_recommendations.py`
- `local_weather_recommendations.py` → `features/recommendations/services/weather_recommendations.py`

### Feature: Tracking (`features/tracking/`)

**Servicios a migrar:**
- `progress_analyzer.py` → `features/tracking/services/progress_analyzer.py`
- `history_tracker.py` → `features/tracking/services/history_tracker.py`
- `before_after_analysis.py` → `features/tracking/services/before_after_analysis.py`
- `temporal_comparison.py` → `features/tracking/services/temporal_comparison.py`
- `visual_progress_tracker.py` → `features/tracking/services/visual_progress_tracker.py`
- `progress_visualization.py` → `features/tracking/services/progress_visualization.py`
- `ai_progress_analysis.py` → `features/tracking/services/ai_progress_analysis.py`
- `historical_photo_analysis.py` → `features/tracking/services/historical_photo_analysis.py`
- `comparative_analysis.py` → `features/tracking/services/comparative_analysis.py`
- `advanced_comparison.py` → `features/tracking/services/advanced_comparison.py`
- `anonymous_comparison.py` → `features/tracking/services/anonymous_comparison.py`
- `benchmark_analysis.py` → `features/tracking/services/benchmark_analysis.py`
- `plateau_detection.py` → `features/tracking/services/plateau_detection.py`
- `habit_analyzer.py` → `features/tracking/services/habit_analyzer.py`
- `skin_journal.py` → `features/tracking/services/skin_journal.py`
- `skin_goals.py` → `features/tracking/services/skin_goals.py`
- `skin_concern_tracker.py` → `features/tracking/services/skin_concern_tracker.py`
- `skin_state_analysis.py` → `features/tracking/services/skin_state_analysis.py`
- `custom_routine_tracker.py` → `features/tracking/services/routine_tracker.py`
- `routine_comparator.py` → `features/tracking/services/routine_comparator.py`
- `routine_optimizer.py` → `features/tracking/services/routine_optimizer.py`
- `successful_routines.py` → `features/tracking/services/successful_routines.py`

### Feature: Products (`features/products/`)

**Servicios a migrar:**
- `product_database.py` → `features/products/services/product_database.py`
- `product_comparison.py` → `features/products/services/product_comparison.py`
- `product_tracker.py` → `features/products/services/product_tracker.py`
- `product_compatibility.py` → `features/products/services/product_compatibility.py`
- `product_effectiveness_tracker.py` → `features/products/services/product_effectiveness_tracker.py`
- `product_needs_predictor.py` → `features/products/services/product_needs_predictor.py`
- `product_reminder_system.py` → `features/products/services/product_reminder.py`
- `product_trend_analyzer.py` → `features/products/services/product_trend_analyzer.py`
- `ingredient_analyzer.py` → `features/products/services/ingredient_analyzer.py`
- `ingredient_conflict_checker.py` → `features/products/services/ingredient_conflict_checker.py`
- `custom_recipes.py` → `features/products/services/custom_recipes.py`
- `reviews_ratings.py` → `features/products/services/reviews_ratings.py`

### Feature: Notifications (`features/notifications/`)

**Servicios a migrar:**
- `notification_service.py` → `features/notifications/services/notification_service.py`
- `push_notifications.py` → `features/notifications/services/push_notifications.py`
- `smart_reminders.py` → `features/notifications/services/smart_reminders.py`
- `enhanced_notifications.py` → `features/notifications/services/enhanced_notifications.py`
- `intelligent_alerts.py` → `features/notifications/services/intelligent_alerts.py`
- `alert_system.py` → `features/notifications/services/alert_system.py`

### Feature: Analytics (`features/analytics/`)

**Servicios a migrar:**
- `analytics.py` → `features/analytics/services/analytics.py`
- `business_metrics.py` → `features/analytics/services/business_metrics.py`
- `predictive_analytics.py` → `features/analytics/services/predictive_analytics.py`
- `trend_prediction.py` → `features/analytics/services/trend_prediction.py`
- `trend_predictor.py` → `features/analytics/services/trend_predictor.py`
- `future_prediction.py` → `features/analytics/services/future_prediction.py`
- `metrics_dashboard.py` → `features/analytics/services/metrics_dashboard.py`
- `realtime_metrics.py` → `features/analytics/services/realtime_metrics.py`
- `advanced_monitoring.py` → `features/analytics/services/advanced_monitoring.py`

### Feature: Integrations (`features/integrations/`)

**Servicios a migrar:**
- `iot_integration.py` → `features/integrations/services/iot_integration.py`
- `wearable_integration.py` → `features/integrations/services/wearable_integration.py`
- `pharmacy_integration.py` → `features/integrations/services/pharmacy_integration.py`
- `medical_device_integration.py` → `features/integrations/services/medical_device_integration.py`
- `integration_service.py` → `features/integrations/services/integration_service.py`
- `webhook_manager.py` → `features/integrations/services/webhook_manager.py`

### Shared Services (`shared/services/`)

**Servicios compartidos:**
- `database.py` → `shared/services/database_service.py`
- `cache_service.py` (nuevo) → `shared/services/cache_service.py`
- `event_system.py` → `shared/services/event_service.py`
- `async_queue.py` → `shared/services/async_queue.py`
- `batch_processor.py` → `shared/services/batch_processor.py`

### Otros Servicios (mantener en `services/` temporalmente)

**Servicios que requieren análisis adicional:**
- `admin_api.py` → Evaluar si va a `api/admin/` o `features/admin/`
- `auth_manager.py` → Evaluar si va a `core/auth/` o `features/auth/`
- `backup_manager.py` → Evaluar ubicación
- `challenge_system.py` → Evaluar si va a `features/gamification/`
- `collaboration_service.py` → Evaluar ubicación
- `community_features.py` → Evaluar si va a `features/community/`
- `condition_predictor.py` → Evaluar si va a `features/analysis/` o `features/ml/`
- `custom_routine_tracker.py` → Ya mapeado a tracking
- `diet_tracker.py` → Evaluar si va a `features/tracking/` o nuevo módulo
- `enhanced_export.py` → Evaluar ubicación
- `enhanced_ml.py` → Evaluar si va a `ml/` o `features/ml/`
- `environmental_tracker.py` → Evaluar ubicación
- `expert_consultation.py` → Evaluar si va a `features/consultation/`
- `export_manager.py` → Evaluar ubicación
- `feedback_system.py` → Evaluar ubicación
- `gamification.py` → Evaluar si va a `features/gamification/`
- `health_monitor.py` → Evaluar si va a `features/health/` o `api/health/`
- `hormonal_tracker.py` → Evaluar si va a `features/tracking/`
- `learning_system.py` → Evaluar ubicación
- `market_trends.py` → Evaluar si va a `features/analytics/`
- `medical_treatment_tracker.py` → Evaluar si va a `features/tracking/`
- `model_versioning.py` → Evaluar si va a `ml/` o `core/ml/`
- `performance_optimizer.py` → Evaluar si va a `utils/` o `shared/`
- `personalization_engine.py` → Evaluar ubicación
- `personalized_coaching.py` → Evaluar si va a `features/coaching/`
- `professional_treatment_tracker.py` → Evaluar si va a `features/tracking/`
- `report_generator.py` → Evaluar si va a `features/reports/`
- `report_templates.py` → Evaluar si va a `features/reports/`
- `advanced_reporting.py` → Evaluar si va a `features/reports/`
- `seasonal_changes_tracker.py` → Evaluar si va a `features/tracking/`
- `security_enhancer.py` → Evaluar si va a `core/security/`
- `side_effect_tracker.py` → Evaluar si va a `features/tracking/`
- `sleep_habit_tracker.py` → Evaluar si va a `features/tracking/`
- `social_features.py` → Evaluar si va a `features/social/`
- `stress_tracker.py` → Evaluar si va a `features/tracking/`
- `supplement_tracker.py` → Evaluar si va a `features/tracking/`
- `tagging_system.py` → Evaluar ubicación
- `visualization.py` → Evaluar si va a `features/visualization/` o `utils/`
- `weather_climate_analysis.py` → Evaluar si va a `features/analysis/` o `features/environment/`
- `climate_condition_analysis.py` → Evaluar si va a `features/analysis/` o `features/environment/`
- `ab_testing.py` → Evaluar si va a `features/analytics/`
- `advanced_search.py` → Evaluar ubicación
- `api_documentation.py` → Evaluar si va a `docs/` o `api/docs/`

---

## 🔄 Ejemplos de Refactorización

### Ejemplo 1: Migrar `skincare_recommender.py` a Feature Module

#### Antes (en `services/skincare_recommender.py`):

```python
# services/skincare_recommender.py
class SkincareRecommender:
    def __init__(self):
        self.product_database = self._initialize_product_database()
        # ...
```

#### Después (en `features/recommendations/services/skincare_recommender.py`):

```python
# features/recommendations/services/skincare_recommender.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from core.domain.interfaces import IRecommendationService, IProductRepository

@dataclass
class SkincareProduct:
    """Producto de skincare recomendado"""
    name: str
    category: ProductCategory
    description: str
    key_ingredients: List[str]
    usage_frequency: str
    priority: int

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
        self.product_repository = product_repository
        self.analysis_service = analysis_service
        self._product_cache: Optional[Dict] = None
    
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
        """
        # Validar entrada
        if not analysis_result:
            raise ValueError("analysis_result is required")
        
        # Obtener productos del repositorio
        products = await self.product_repository.get_all()
        
        # Generar rutina personalizada
        routine = await self._create_routine(
            analysis_result=analysis_result,
            products=products,
            user_id=user_id
        )
        
        return {
            "routine": routine,
            "specific_recommendations": await self._get_specific_recommendations(analysis_result),
            "tips": await self._generate_tips(analysis_result)
        }
    
    async def _create_routine(
        self,
        analysis_result: Dict,
        products: List[Dict],
        user_id: Optional[str] = None
    ) -> Dict:
        """Crea rutina personalizada"""
        # Implementación...
        pass
```

**Cambios clave:**
1. ✅ Implementa `IRecommendationService` (interfaz del dominio)
2. ✅ Dependencias inyectadas en constructor
3. ✅ Métodos async para mejor performance
4. ✅ Validación de entrada
5. ✅ Uso de repositorio en lugar de acceso directo a base de datos

### Ejemplo 2: Migrar `image_analysis_advanced.py` a Feature Module

#### Antes:

```python
# services/image_analysis_advanced.py
class AdvancedImageAnalysis:
    def __init__(self):
        pass
    
    def analyze_texture_features(self, image: np.ndarray) -> Dict:
        # Implementación...
        pass
```

#### Después:

```python
# features/analysis/services/image_analysis.py
from typing import Dict, Optional
import numpy as np
from core.domain.interfaces import IImageProcessor, IAnalysisService
from core.domain.entities import Analysis, SkinMetrics

class ImageAnalysisService(IAnalysisService):
    """
    Servicio de análisis de imágenes.
    
    Implementa IAnalysisService siguiendo arquitectura hexagonal.
    """
    
    def __init__(
        self,
        image_processor: IImageProcessor,
        texture_analyzer: Optional['TextureAnalyzer'] = None,
        color_analyzer: Optional['ColorAnalyzer'] = None
    ):
        """
        Inicializa el servicio de análisis.
        
        Args:
            image_processor: Procesador de imágenes
            texture_analyzer: Analizador de textura (opcional)
            color_analyzer: Analizador de color (opcional)
        """
        self.image_processor = image_processor
        self.texture_analyzer = texture_analyzer
        self.color_analyzer = color_analyzer
    
    async def analyze_image(
        self,
        image_data: bytes,
        metadata: Optional[Dict] = None
    ) -> Analysis:
        """
        Analiza imagen y retorna entidad Analysis.
        
        Args:
            image_data: Datos de la imagen en bytes
            metadata: Metadatos opcionales
            
        Returns:
            Entidad Analysis con métricas y condiciones
        """
        # Validar entrada
        if not image_data:
            raise ValueError("image_data is required")
        
        # Procesar imagen
        processed = await self.image_processor.process(image_data)
        
        # Analizar textura
        texture_features = await self._analyze_texture(processed)
        
        # Analizar color
        color_features = await self._analyze_color(processed)
        
        # Crear métricas
        metrics = self._create_metrics(texture_features, color_features)
        
        # Detectar condiciones
        conditions = await self._detect_conditions(processed, metrics)
        
        # Crear entidad Analysis
        return Analysis(
            metrics=metrics,
            conditions=conditions,
            metadata=metadata or {}
        )
    
    async def _analyze_texture(self, processed_image: np.ndarray) -> Dict:
        """Analiza características de textura"""
        if self.texture_analyzer:
            return await self.texture_analyzer.analyze(processed_image)
        return {}
    
    async def _analyze_color(self, processed_image: np.ndarray) -> Dict:
        """Analiza características de color"""
        if self.color_analyzer:
            return await self.color_analyzer.analyze(processed_image)
        return {}
    
    def _create_metrics(
        self,
        texture_features: Dict,
        color_features: Dict
    ) -> SkinMetrics:
        """Crea objeto SkinMetrics desde features"""
        # Implementación...
        pass
```

### Ejemplo 3: Crear Feature Module `__init__.py`

```python
# features/recommendations/__init__.py
"""
Recommendations Feature Module

Este módulo contiene todos los servicios relacionados con recomendaciones
de skincare.
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
    Registra todos los servicios de recomendaciones en el service factory.
    
    Args:
        service_factory: Instancia del service factory
    """
    from core.service_factory import get_service_factory
    
    factory = service_factory or get_service_factory()
    
    # Registrar servicios
    factory.register(
        "recommendations.skincare",
        lambda: SkincareRecommender(
            product_repository=factory.get("product_repository"),
            analysis_service=factory.get("analysis_service")
        ),
        dependencies=["product_repository", "analysis_service"],
        scope=ServiceScope.SINGLETON
    )
    
    factory.register(
        "recommendations.intelligent",
        lambda: IntelligentRecommender(
            skincare_recommender=factory.get("recommendations.skincare"),
            ml_model=factory.get("ml_model")
        ),
        dependencies=["recommendations.skincare", "ml_model"],
        scope=ServiceScope.SINGLETON
    )
```

---

## 🎨 Patrones de Diseño

### 1. Dependency Injection

**Antes:**
```python
class SkincareRecommender:
    def __init__(self):
        self.product_database = ProductDatabase()  # ❌ Acoplamiento fuerte
```

**Después:**
```python
class SkincareRecommender:
    def __init__(self, product_repository: IProductRepository):
        self.product_repository = product_repository  # ✅ Inyección de dependencia
```

### 2. Interface Segregation

**Crear interfaces específicas:**
```python
# core/domain/interfaces.py
class IRecommendationService(Protocol):
    """Interfaz para servicios de recomendación"""
    async def generate_recommendations(
        self,
        analysis_result: Dict,
        user_id: Optional[str] = None
    ) -> Dict: ...

class IProductRepository(Protocol):
    """Interfaz para repositorio de productos"""
    async def get_all(self) -> List[Product]: ...
    async def get_by_category(self, category: str) -> List[Product]: ...
```

### 3. Factory Pattern

**Service Factory mejorado:**
```python
# core/service_factory.py
class ServiceFactory:
    def register_feature_services(self):
        """Registra todos los servicios de features"""
        from features.recommendations import register_recommendation_services
        from features.analysis import register_analysis_services
        # ... más features
        
        register_recommendation_services(self)
        register_analysis_services(self)
        # ...
```

### 4. Repository Pattern

**Implementar repositorios:**
```python
# core/infrastructure/repositories/product_repository.py
class ProductRepository(IProductRepository):
    """Implementación de repositorio de productos"""
    
    def __init__(self, database_adapter: IDatabaseAdapter):
        self.db = database_adapter
    
    async def get_all(self) -> List[Product]:
        """Obtiene todos los productos"""
        # Implementación...
        pass
```

---

## 📝 Guía Paso a Paso

### Paso 1: Crear Estructura de Directorios

```bash
# Crear estructura de features
mkdir -p features/{analysis,recommendations,tracking,products,notifications,analytics,integrations}/services
mkdir -p features/{analysis,recommendations,tracking,products,notifications,analytics,integrations}/controllers
mkdir -p shared/services
mkdir -p api/v1/{routes,schemas}
```

### Paso 2: Migrar un Servicio (Ejemplo: skincare_recommender.py)

1. **Crear nuevo archivo:**
   ```bash
   cp services/skincare_recommender.py features/recommendations/services/skincare_recommender.py
   ```

2. **Refactorizar para usar interfaces:**
   - Agregar implementación de `IRecommendationService`
   - Inyectar dependencias en constructor
   - Convertir métodos a async si es necesario
   - Agregar validación de entrada

3. **Actualizar imports:**
   ```python
   # Cambiar imports relativos
   # Antes: from services.product_database import ProductDatabase
   # Después: from features.products.services.product_database import ProductDatabase
   ```

4. **Crear `__init__.py` del módulo:**
   ```python
   # features/recommendations/__init__.py
   from .services.skincare_recommender import SkincareRecommender
   ```

5. **Registrar en Service Factory:**
   ```python
   # En features/recommendations/__init__.py
   def register_recommendation_services(service_factory):
       service_factory.register(
           "recommendations.skincare",
           SkincareRecommender,
           dependencies=["product_repository"]
       )
   ```

6. **Actualizar código que usa el servicio:**
   ```python
   # Antes
   from services.skincare_recommender import SkincareRecommender
   recommender = SkincareRecommender()
   
   # Después
   from core.service_factory import get_service_factory
   factory = get_service_factory()
   recommender = await factory.get("recommendations.skincare")
   ```

### Paso 3: Migrar API Endpoints

1. **Crear nuevo router:**
   ```python
   # api/v1/routes/recommendations.py
   from fastapi import APIRouter, Depends
   from core.service_factory import get_service_factory
   
   router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"])
   
   @router.post("/generate")
   async def generate_recommendations(
       analysis_result: dict,
       recommender = Depends(get_recommendation_service)
   ):
       return await recommender.generate_recommendations(analysis_result)
   ```

2. **Registrar router:**
   ```python
   # api/v1/__init__.py
   from fastapi import FastAPI
   from .routes import recommendations, analysis, tracking
   
   def register_v1_routes(app: FastAPI):
       app.include_router(recommendations.router)
       app.include_router(analysis.router)
       app.include_router(tracking.router)
   ```

### Paso 4: Actualizar Composition Root

```python
# core/composition_root.py
class CompositionRoot:
    async def initialize(self, config: Dict[str, Any]) -> None:
        # ... código existente ...
        
        # Registrar servicios de features
        from features.recommendations import register_recommendation_services
        from features.analysis import register_analysis_services
        # ... más features
        
        register_recommendation_services(self.service_factory)
        register_analysis_services(self.service_factory)
        # ...
```

---

## ✅ Checklist de Migración

### Por Servicio

- [ ] Crear archivo en nuevo directorio
- [ ] Refactorizar para usar interfaces del dominio
- [ ] Inyectar dependencias en constructor
- [ ] Convertir a async si es necesario
- [ ] Agregar validación de entrada
- [ ] Actualizar imports
- [ ] Crear/actualizar `__init__.py` del módulo
- [ ] Registrar en Service Factory
- [ ] Actualizar código que usa el servicio
- [ ] Ejecutar tests
- [ ] Verificar que no hay regresiones

### Por Feature Module

- [ ] Crear estructura de directorios
- [ ] Migrar todos los servicios del módulo
- [ ] Crear `__init__.py` con exports
- [ ] Crear función de registro de servicios
- [ ] Crear controllers si es necesario
- [ ] Crear routers si es necesario
- [ ] Actualizar Composition Root
- [ ] Ejecutar tests del módulo
- [ ] Documentar el módulo

### Global

- [ ] Migrar todos los servicios
- [ ] Consolidar API layer
- [ ] Actualizar Composition Root
- [ ] Actualizar main.py
- [ ] Ejecutar suite completa de tests
- [ ] Validar performance
- [ ] Actualizar documentación
- [ ] Crear guía de migración para desarrolladores
- [ ] Deprecar APIs legacy (con período de transición)

---

## 🔗 Referencias

- Ver `ARCHITECTURE_IMPROVEMENTS_V8.md` para arquitectura completa
- Ver `QUICK_ARCHITECTURE_IMPROVEMENTS.md` para resumen rápido

---

**Versión:** 8.0.0  
**Fecha:** 2024




