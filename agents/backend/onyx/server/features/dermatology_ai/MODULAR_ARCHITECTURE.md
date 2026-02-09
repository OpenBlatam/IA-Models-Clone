# Arquitectura Modular - Dermatology AI

## Versión 5.3.0 - Refactorización Modular

Este documento describe la nueva arquitectura modular implementada en la versión 5.3.0 del sistema Dermatology AI.

## Estructura Modular

### Componentes Principales

1. **Service Locator** (`api/services_locator.py`)
   - Patrón de inyección de dependencias centralizado
   - Registra y proporciona acceso a todas las instancias de servicios
   - Evita dependencias circulares entre módulos

2. **Router Manager** (`api/routers/router_manager.py`)
   - Gestiona el registro y organización de todos los routers modulares
   - Proporciona metadatos y resúmenes de endpoints
   - Facilita la extensión y mantenimiento

3. **Routers Modulares** (`api/routers/`)
   - `analysis_router.py` - Endpoints de análisis (imagen, video, textura)
   - `recommendations_router.py` - Endpoints de recomendaciones
   - `tracking_router.py` - Endpoints de tracking (progreso, hábitos, efectos secundarios)
   - `products_router.py` - Endpoints de productos
   - `ml_router.py` - Endpoints de ML/AI avanzado
   - `integrations_router.py` - Endpoints de integraciones (IoT, wearable, pharmacy)
   - `reports_router.py` - Endpoints de reportes y visualización
   - `social_router.py` - Endpoints de social/gamificación
   - `health_router.py` - Endpoints de salud y monitoreo

4. **Base Router** (`api/routers/base_router.py`)
   - Utilidades comunes para todos los routers
   - Manejo de errores estandarizado
   - Funciones helper compartidas

## Beneficios de la Arquitectura Modular

1. **Mantenibilidad**: Cada router es independiente y fácil de mantener
2. **Escalabilidad**: Fácil agregar nuevos routers sin afectar los existentes
3. **Testabilidad**: Cada router puede ser probado de forma aislada
4. **Organización**: Endpoints agrupados por dominio funcional
5. **Reutilización**: Servicios compartidos a través del Service Locator

## Migración

El archivo `dermatology_api.py` original (5939 líneas) se mantiene para compatibilidad hacia atrás, pero se recomienda usar la nueva arquitectura modular (`dermatology_api_modular.py`).

## Uso

```python
from api.dermatology_api_modular import modular_router

app.include_router(modular_router)
```

## Estructura de Routers

### 1. Analysis Router
- `/analyze-image` - Análisis de imagen
- `/analyze-video` - Análisis de video
- `/texture-ml/analyze` - Análisis de textura con ML
- `/image-analysis/advanced` - Análisis avanzado de imagen

### 2. Recommendations Router
- `/get-recommendations` - Recomendaciones básicas
- `/recommendations/intelligent` - Recomendaciones inteligentes
- `/smart-recommendations/generate` - Recomendaciones smart
- `/ml-recommendations/generate` - Recomendaciones basadas en ML

### 3. Tracking Router
- `/progress/add-data` - Agregar datos de progreso
- `/progress/report/{user_id}` - Reporte de progreso
- `/side-effect/add` - Registrar efecto secundario
- `/side-effect/analyze/{user_id}` - Analizar efectos secundarios
- `/habits/record` - Registrar hábito
- `/habits/analyze/{user_id}` - Analizar hábitos
- `/sleep/add-record` - Registrar sueño
- `/sleep/analyze/{user_id}` - Analizar patrones de sueño

### 4. Products Router
- `/products/search` - Buscar productos
- `/products/{product_id}` - Obtener producto
- `/products/track` - Trackear uso de producto
- `/products/insights/{user_id}/{product_id}` - Insights de producto
- `/products/compare` - Comparar productos
- `/product-compatibility/register` - Registrar producto
- `/product-compatibility/check` - Verificar compatibilidad

### 5. ML/AI Router
- `/ml/predict` - Predicción ML
- `/ml/stats` - Estadísticas ML
- `/ml-advanced/add-data` - Agregar datos ML
- `/ml-advanced/analyze/{user_id}` - Análisis ML avanzado
- `/conditions/predict` - Predecir condiciones
- `/future-prediction/add-data` - Datos para predicción
- `/future-prediction/generate/{user_id}` - Generar predicción

### 6. Integrations Router
- `/iot/register` - Registrar dispositivo IoT
- `/iot/devices/{user_id}` - Dispositivos IoT
- `/iot/data` - Sincronizar datos IoT
- `/wearable/register` - Registrar wearable
- `/wearable/sync` - Sincronizar wearable
- `/wearable/insights/{user_id}` - Insights wearable
- `/pharmacy/register` - Registrar farmacia
- `/pharmacy/nearby` - Farmacias cercanas
- `/pharmacy/product-availability` - Disponibilidad de productos

### 7. Reports Router
- `/report/json` - Reporte JSON
- `/report/pdf` - Reporte PDF
- `/report/html` - Reporte HTML
- `/reports/advanced` - Reporte avanzado
- `/reports/formats` - Formatos disponibles
- `/visualization/radar` - Visualización radar
- `/visualization/timeline` - Visualización timeline
- `/visualization/comparison` - Visualización comparación

### 8. Social Router
- `/gamification/stats/{user_id}` - Estadísticas gamificación
- `/gamification/achievements/{user_id}` - Logros
- `/gamification/leaderboard` - Leaderboard
- `/challenges/available/{user_id}` - Desafíos disponibles
- `/challenges/start` - Iniciar desafío
- `/challenges/user/{user_id}` - Desafíos del usuario
- `/social/follow` - Seguir usuario
- `/social/post` - Crear post
- `/social/feed/{user_id}` - Feed social
- `/community/post` - Post en comunidad
- `/community/posts` - Posts de comunidad

### 9. Health Router
- `/health` - Health check básico
- `/health/detailed` - Health check detallado
- `/health/check/{check_name}` - Health check específico
- `/monitoring/health` - Estado de monitoreo
- `/monitoring/metrics/{metric_name}` - Métricas
- `/monitoring/alerts` - Alertas
- `/metrics/realtime` - Métricas en tiempo real

## Próximos Pasos

1. ✅ Completar la migración de todos los endpoints a routers modulares
2. Crear tests unitarios para cada router
3. Documentar cada router con OpenAPI/Swagger
4. Implementar versionado de API por router
5. Crear factory pattern para servicios
6. Mejorar dependency injection

