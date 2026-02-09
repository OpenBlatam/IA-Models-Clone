# Physical Store Designer AI

Sistema de IA completo para diseñar locales físicos, incluyendo diseño visual, plan de marketing y estrategia de decoración.

## 🏪 Características

- **Chat Interactivo**: Conversación con IA para recopilar información del cliente
- **Diseño Visual**: Generación de visualizaciones del local (exterior, interior, layout)
- **Plan de Marketing**: Estrategias completas de marketing y ventas
- **Plan de Decoración**: Recomendaciones detalladas de decoración, muebles y materiales
- **Múltiples Estilos**: Soporte para diferentes estilos de diseño (moderno, clásico, minimalista, industrial, etc.)
- **API REST**: API completa con FastAPI para integración

## 📋 Requisitos

- Python 3.10+
- API key de OpenAI (opcional, para generación avanzada con LLM)

## 🚀 Instalación

### Instalación Completa (Recomendado)
```bash
pip install -r requirements.txt
```

### Instalación Mínima (Solo Core)
```bash
pip install -r requirements-minimal.txt
```

### Instalación para Desarrollo
```bash
pip install -r requirements-dev.txt
```

Para más detalles sobre dependencias, ver [DEPENDENCIES.md](DEPENDENCIES.md).

2. **Configurar variables de entorno** (opcional):
```bash
# Crear archivo .env
OPENAI_API_KEY=tu_api_key
```

## 🎯 Uso

### Iniciar el servidor API

```bash
python main.py
# O usando uvicorn directamente
uvicorn physical_store_designer_ai.api.main:app --host 0.0.0.0 --port 8030
```

### Usar el Chat Interactivo

```python
import requests

# Crear sesión de chat
response = requests.post("http://localhost:8030/api/v1/chat/session")
session = response.json()
session_id = session["session_id"]

# Enviar mensaje
message = {
    "role": "user",
    "content": "Quiero abrir una cafetería moderna en el centro"
}
response = requests.post(
    f"http://localhost:8030/api/v1/chat/{session_id}/message",
    json=message
)
print(response.json())
```

### Generar Diseño Completo

```python
from physical_store_designer_ai.core.models import (
    StoreDesignRequest,
    StoreType,
    DesignStyle
)

# Crear request
request = StoreDesignRequest(
    store_name="Café Moderno",
    store_type=StoreType.CAFE,
    style_preference=DesignStyle.MODERN,
    budget_range="medio",
    location="Centro de la ciudad",
    target_audience="Jóvenes profesionales y estudiantes",
    dimensions={"width": 8.0, "length": 12.0, "height": 3.0}
)

# Enviar request a la API
import requests
response = requests.post(
    "http://localhost:8030/api/v1/design/generate",
    json=request.dict()
)
design = response.json()
```

### Generar Diseño desde Chat

```python
# Después de conversar en el chat, generar diseño
response = requests.post(
    f"http://localhost:8030/api/v1/design/from-chat/{session_id}"
)
design = response.json()
```

## 📊 Estructura del Diseño

El diseño generado incluye:

### 1. Layout del Local
- Dimensiones
- Zonas del local
- Ubicación de muebles
- Flujo de tráfico
- Accesibilidad

### 2. Visualizaciones
- Vista exterior
- Vista interior
- Plano de distribución

### 3. Plan de Marketing
- Audiencia objetivo
- Estrategias de marketing
- Tácticas de ventas
- Estrategia de precios
- Ideas de promoción
- Plan de redes sociales
- Estrategia de apertura

### 4. Plan de Decoración
- Esquema de colores
- Plan de iluminación
- Recomendaciones de muebles
- Elementos decorativos
- Materiales recomendados
- Estimación de presupuesto

## 🎨 Estilos Disponibles

- **Moderno**: Líneas limpias, minimalista
- **Clásico**: Elegante, tradicional
- **Minimalista**: Espacios abiertos, pocos elementos
- **Industrial**: Materiales rústicos, acabados metálicos
- **Rústico**: Materiales naturales, ambiente acogedor
- **Lujo**: Materiales premium, acabados refinados
- **Ecológico**: Materiales sostenibles, plantas
- **Vintage**: Elementos retro, nostalgia

## 🏪 Tipos de Tienda

- Restaurante
- Café
- Boutique
- Retail
- Supermercado
- Farmacia
- Electrónica
- Ropa
- Muebles
- Otros

## 📡 Endpoints de la API

### Chat
- `POST /api/v1/chat/session` - Crear sesión de chat
- `POST /api/v1/chat/{session_id}/message` - Enviar mensaje
- `GET /api/v1/chat/{session_id}` - Obtener sesión

### Diseños
- `POST /api/v1/design/generate` - Generar diseño
- `GET /api/v1/design/{store_id}` - Obtener diseño
- `GET /api/v1/designs` - Listar diseños
- `POST /api/v1/design/from-chat/{session_id}` - Generar desde chat
- `GET /api/v1/design/{store_id}/export?format=json|markdown|html` - Exportar diseño
- `DELETE /api/v1/design/{store_id}` - Eliminar diseño

### Análisis
- `GET /api/v1/analysis/competitor/{store_id}` - Análisis de competencia
- `GET /api/v1/analysis/financial/{store_id}` - Análisis financiero
- `GET /api/v1/analysis/inventory/{store_id}` - Recomendaciones de inventario
- `GET /api/v1/analysis/kpis/{store_id}` - KPIs y métricas
- `GET /api/v1/analysis/full/{store_id}` - Análisis completo

### Funcionalidades Avanzadas
- `POST /api/v1/compare/designs` - Comparar múltiples diseños
- `GET /api/v1/technical-plans/{store_id}` - Planos técnicos detallados
- `POST /api/v1/feedback/{store_id}` - Agregar feedback
- `GET /api/v1/feedback/{store_id}` - Obtener feedback y sugerencias
- `GET /api/v1/recommendations/{store_id}` - Recomendaciones inteligentes
- `GET /api/v1/location/analyze` - Analizar ubicación
- `POST /api/v1/versions/{store_id}` - Crear versión de diseño
- `GET /api/v1/versions/{store_id}` - Historial de versiones
- `GET /api/v1/versions/{store_id}/compare` - Comparar versiones
- `POST /api/v1/versions/{store_id}/approve` - Aprobar versión

### Funcionalidades Premium
- `GET /api/v1/reports/{store_id}` - Reporte completo
- `GET /api/v1/reports/{store_id}/pdf` - Reporte en PDF
- `GET /api/v1/reports/{store_id}/excel` - Reporte en Excel
- `POST /api/v1/share/{store_id}` - Compartir diseño
- `GET /api/v1/share/{share_id}` - Obtener diseño compartido
- `POST /api/v1/share/{share_id}/revoke` - Revocar compartir
- `POST /api/v1/comments/{store_id}` - Agregar comentario
- `GET /api/v1/comments/{store_id}` - Obtener comentarios
- `GET /api/v1/dashboard` - Dashboard completo
- `GET /api/v1/templates` - Listar templates
- `GET /api/v1/templates/{template_id}` - Obtener template
- `POST /api/v1/templates/{template_id}/apply` - Aplicar template
- `GET /api/v1/trends` - Análisis de tendencias
- `GET /api/v1/notifications/{user_id}` - Obtener notificaciones
- `POST /api/v1/notifications/{user_id}/read/{notification_id}` - Marcar como leída

### Funcionalidades Enterprise
- `POST /api/v1/auth/register` - Registro de usuario
- `POST /api/v1/auth/login` - Iniciar sesión
- `GET /api/v1/auth/me` - Usuario actual
- `POST /api/v1/optimize/budget/{store_id}` - Optimizar presupuesto
- `GET /api/v1/optimize/layout/{store_id}` - Optimizar layout
- `POST /api/v1/optimize/marketing/{store_id}` - Optimizar marketing
- `GET /api/v1/predict/success/{store_id}` - Predecir éxito
- `GET /api/v1/predict/revenue/{store_id}` - Predecir ingresos
- `GET /api/v1/predict/traffic/{store_id}` - Predecir tráfico
- `GET /api/v1/monitoring/health/{store_id}` - Salud del diseño
- `GET /api/v1/monitoring/alerts/{store_id}` - Obtener alertas
- `POST /api/v1/monitoring/alerts/{store_id}/acknowledge/{alert_id}` - Reconocer alerta
- `GET /api/v1/monitoring/metrics/{store_id}` - Obtener métricas

### Integraciones y Exportación
- `GET /api/v1/external/location/{location}` - Detalles de ubicación (Google Maps)
- `GET /api/v1/external/weather/{location}` - Información del clima
- `GET /api/v1/external/nearby/{location}` - Lugares cercanos
- `POST /api/v1/backup/create` - Crear backup
- `GET /api/v1/backup/list` - Listar backups
- `POST /api/v1/backup/restore/{backup_id}` - Restaurar backup
- `GET /api/v1/export/cad/{store_id}` - Exportar a CAD/DXF
- `GET /api/v1/export/3d/{store_id}` - Exportar a 3D (OBJ/STL)
- `GET /api/v1/export/svg/{store_id}` - Exportar a SVG
- `GET /api/v1/export/pdf-advanced/{store_id}` - Exportar PDF avanzado
- `POST /api/v1/webhooks/register` - Registar webhook
- `GET /api/v1/webhooks/{user_id}` - Obtener webhooks
- `GET /api/v1/cache/stats` - Estadísticas de caché
- `POST /api/v1/cache/clear` - Limpiar caché

### Funcionalidades de Negocio
- `POST /api/v1/vendors/register` - Registrar proveedor
- `GET /api/v1/vendors` - Obtener proveedores
- `POST /api/v1/vendors/quotations/{store_id}` - Solicitar cotización
- `GET /api/v1/vendors/recommend/{store_type}` - Recomendar proveedores
- `POST /api/v1/billing/subscribe` - Crear suscripción
- `GET /api/v1/billing/subscription/{user_id}` - Obtener suscripción
- `GET /api/v1/billing/invoices/{user_id}` - Obtener facturas
- `POST /api/v1/billing/pay/{invoice_id}` - Procesar pago
- `POST /api/v1/sentiment/analyze` - Analizar sentimiento
- `GET /api/v1/ml/recommendations/{user_id}` - Recomendaciones ML
- `POST /api/v1/social/generate-content/{store_id}` - Generar contenido social
- `GET /api/v1/social/calendar/{store_id}` - Calendario de contenido
- `POST /api/v1/ab-testing/create` - Crear test A/B
- `GET /api/v1/ab-testing/{test_id}/results` - Resultados de test

### Engagement y Funcionalidades Avanzadas
- `POST /api/v1/analytics/track` - Rastrear evento
- `GET /api/v1/analytics` - Obtener analytics
- `GET /api/v1/analytics/user/{user_id}/journey` - Journey de usuario
- `GET /api/v1/gamification/profile/{user_id}` - Perfil de gamificación
- `POST /api/v1/gamification/award-points/{user_id}` - Otorgar puntos
- `GET /api/v1/gamification/leaderboard` - Leaderboard
- `POST /api/v1/marketplace/listings` - Crear listing
- `GET /api/v1/marketplace/listings` - Obtener listings
- `POST /api/v1/marketplace/listings/{listing_id}/purchase` - Comprar listing
- `POST /api/v1/crm/contacts` - Crear contacto
- `GET /api/v1/crm/pipeline` - Pipeline de ventas
- `POST /api/v1/loyalty/enroll/{user_id}` - Inscribir en lealtad
- `GET /api/v1/loyalty/{user_id}` - Estadísticas de lealtad
- `POST /api/v1/loyalty/claim/{user_id}` - Reclamar recompensa
- `POST /api/v1/competitor/monitoring/start` - Iniciar monitoreo de competencia
- `GET /api/v1/competitor/monitoring/{store_id}` - Análisis en tiempo real
- `GET /api/v1/competitor/compare/{store_id}` - Comparar con competidores

### Funcionalidades Avanzadas
- `POST /api/v1/generative-ai/image/{store_id}` - Generar imagen del local
- `POST /api/v1/generative-ai/video/{store_id}` - Generar video promocional
- `POST /api/v1/generative-ai/marketing-copy/{store_id}` - Generar copy de marketing
- `POST /api/v1/workflows/create` - Crear workflow
- `POST /api/v1/workflows/{workflow_id}/execute` - Ejecutar workflow
- `POST /api/v1/bookings/create` - Crear reserva
- `GET /api/v1/bookings/availability/{store_id}/{date}` - Slots disponibles
- `POST /api/v1/roi/calculate` - Calcular ROI
- `POST /api/v1/roi/report/{store_id}` - Reporte completo de ROI
- `POST /api/v1/documentation/generate/{store_id}` - Generar documentación
- `POST /api/v1/payments/create` - Crear intención de pago
- `POST /api/v1/payments/{payment_id}/process` - Procesar pago

### IoT, ERP y Tecnologías Avanzadas
- `POST /api/v1/ar-vr/experience/{store_id}` - Crear experiencia AR
- `POST /api/v1/ar-vr/vr-tour/{store_id}` - Crear tour VR
- `POST /api/v1/iot/devices/{store_id}` - Registrar dispositivo IoT
- `POST /api/v1/iot/readings/{sensor_id}` - Registrar lectura de sensor
- `GET /api/v1/iot/analytics/{store_id}` - Analytics de IoT
- `POST /api/v1/inventory/products/{store_id}` - Agregar producto
- `GET /api/v1/inventory/predict/{product_id}` - Predecir demanda
- `POST /api/v1/analytics/metrics/{store_id}` - Registrar métrica
- `GET /api/v1/analytics/dashboard/{store_id}` - Dashboard en tiempo real
- `POST /api/v1/erp/register/{store_id}` - Registrar integración ERP
- `POST /api/v1/compliance/assess/{store_id}` - Evaluar compliance

### Tecnologías Futuras
- `POST /api/v1/blockchain/contract/{store_id}` - Desplegar contrato inteligente
- `POST /api/v1/blockchain/nft/{store_id}` - Crear NFT del diseño
- `POST /api/v1/sustainability/footprint/{store_id}` - Calcular huella de carbono
- `POST /api/v1/sustainability/assess/{store_id}` - Evaluar sostenibilidad
- `POST /api/v1/customer-behavior/interaction/{store_id}` - Registrar interacción
- `GET /api/v1/customer-behavior/journey/{customer_id}` - Analizar journey
- `POST /api/v1/security/events/{system_id}` - Registrar evento de seguridad
- `GET /api/v1/security/status/{store_id}` - Estado de seguridad
- `GET /api/v1/maintenance/predict/{equipment_id}` - Predecir mantenimiento
- `POST /api/v1/sentiment/stream/{store_id}` - Procesar sentimiento en tiempo real

### Tecnologías de Vanguardia
- `POST /api/v1/ml/train` - Entrenar modelo ML personalizado
- `POST /api/v1/ml/predict/{model_id}` - Hacer predicción con modelo
- `POST /api/v1/voice/skills/{store_id}` - Crear skill de voz
- `POST /api/v1/voice/command/{skill_id}` - Procesar comando de voz
- `POST /api/v1/mr/experience/{store_id}` - Crear experiencia de realidad mixta
- `GET /api/v1/mr/showroom/{store_id}` - Crear showroom virtual
- `POST /api/v1/market/data/{store_id}` - Registar dato de mercado
- `GET /api/v1/market/trends/{store_id}` - Analizar tendencias
- `POST /api/v1/collaborative/preference` - Registrar preferencia
- `GET /api/v1/collaborative/recommend/{user_id}` - Recomendaciones colaborativas
- `POST /api/v1/energy/devices/{store_id}` - Registrar dispositivo de energía
- `GET /api/v1/energy/optimize/{store_id}` - Optimización de energía

### Tecnologías de Próxima Generación
- `POST /api/v1/quantum/circuit` - Crear circuito cuántico
- `POST /api/v1/quantum/execute/{circuit_id}` - Ejecutar circuito cuántico
- `POST /api/v1/quantum/optimize` - Optimizar usando quantum
- `POST /api/v1/edge/devices/{store_id}` - Registrar dispositivo edge
- `POST /api/v1/edge/deploy/{device_id}` - Desplegar a edge
- `POST /api/v1/federated/models` - Crear modelo federado
- `POST /api/v1/federated/round/{model_id}` - Ejecutar ronda federada
- `POST /api/v1/graph/create` - Crear grafo
- `POST /api/v1/graph/analyze/{graph_id}` - Analizar grafo
- `POST /api/v1/simulation/create` - Crear simulación
- `POST /api/v1/simulation/run/{simulation_id}` - Ejecutar simulación
- `POST /api/v1/logistics/shipments/{store_id}` - Crear envío
- `GET /api/v1/logistics/track/{shipment_id}` - Rastrear envío
- `POST /api/v1/logistics/optimize-route` - Optimizar ruta

### Tecnologías Avanzadas Adicionales
- `POST /api/v1/biometrics/enroll/{user_id}` - Registrar biometría
- `POST /api/v1/biometrics/verify/{user_id}` - Verificar biometría
- `POST /api/v1/xr/experience/{store_id}` - Crear experiencia XR
- `GET /api/v1/xr/showroom/{store_id}` - Crear showroom XR
- `POST /api/v1/big-data/datasets` - Crear dataset de big data
- `POST /api/v1/big-data/query/{dataset_id}` - Ejecutar query de big data
- `POST /api/v1/robotics/register/{store_id}` - Registrar robot
- `POST /api/v1/robotics/tasks/{robot_id}` - Asignar tarea a robot
- `POST /api/v1/video/analyze/{video_id}` - Analizar video
- `POST /api/v1/video/detect-objects` - Detectar objetos en imagen
- `POST /api/v1/supply-chain/orders/{store_id}` - Crear orden de compra
- `GET /api/v1/supply-chain/track/{order_id}` - Rastrear orden

### Tecnologías Finales Avanzadas
- `POST /api/v1/multimodal/generate/{store_id}` - Generar contenido multimodal
- `POST /api/v1/multimodal/analyze` - Analizar entrada multimodal
- `POST /api/v1/behavior/record/{store_id}` - Registrar comportamiento
- `GET /api/v1/behavior/predict/{customer_id}` - Predecir comportamiento
- `POST /api/v1/waste/bins/{store_id}` - Registrar contenedor de residuos
- `GET /api/v1/waste/analytics/{store_id}` - Analytics de residuos
- `POST /api/v1/traffic/record/{store_id}` - Registrar punto de tráfico
- `GET /api/v1/traffic/analyze/{store_id}` - Analizar flujo de tráfico
- `POST /api/v1/renewable/install/{store_id}` - Instalar sistema renovable
- `GET /api/v1/renewable/savings/{store_id}` - Calcular ahorros
- `GET /api/v1/recommendations/hybrid/{user_id}` - Recomendaciones híbridas

### Deep Learning y Modelos Avanzados
- `POST /api/v1/deep-learning/models` - Crear modelo de deep learning
- `POST /api/v1/deep-learning/models/{model_id}/train` - Entrenar modelo
- `POST /api/v1/deep-learning/models/{model_id}/predict` - Hacer predicción
- `POST /api/v1/deep-learning/models/{model_id}/checkpoints` - Guardar checkpoint
- `POST /api/v1/diffusion/pipelines` - Crear pipeline de difusión
- `POST /api/v1/diffusion/pipelines/{pipeline_id}/generate` - Generar imagen de tienda
- `POST /api/v1/diffusion/pipelines/{pipeline_id}/variations` - Generar variaciones
- `POST /api/v1/llm-finetuning/datasets` - Preparar dataset para fine-tuning
- `POST /api/v1/llm-finetuning/jobs` - Crear job de fine-tuning
- `POST /api/v1/llm-finetuning/jobs/{job_id}/start` - Iniciar fine-tuning
- `GET /api/v1/llm-finetuning/jobs/{job_id}/model` - Obtener modelo fine-tuneado
- `POST /api/v1/embeddings/generate` - Generar embedding
- `POST /api/v1/embeddings/search` - Búsqueda semántica
- `POST /api/v1/embeddings/similar-designs` - Encontrar diseños similares
- `POST /api/v1/experiments` - Crear experimento
- `POST /api/v1/experiments/{experiment_id}/runs` - Crear run
- `POST /api/v1/experiments/runs/{run_id}/metrics` - Registrar métrica
- `POST /api/v1/experiments/runs/{run_id}/complete` - Completar run
- `GET /api/v1/experiments/{experiment_id}/results` - Obtener resultados
- `POST /api/v1/gradio/apps` - Crear aplicación demo
- `POST /api/v1/gradio/store-designer-demo` - Crear demo de diseñador
- `POST /api/v1/gradio/ml-model-demo` - Crear demo de modelo ML
- `POST /api/v1/gradio/apps/{app_id}/launch` - Lanzar aplicación Gradio

### Optimización de Rendimiento y Técnicas Avanzadas
- `GET /api/v1/performance/devices` - Detectar dispositivos disponibles
- `POST /api/v1/performance/data-parallel` - Configurar DataParallel
- `POST /api/v1/performance/distributed` - Configurar DistributedDataParallel
- `POST /api/v1/performance/mixed-precision` - Configurar mixed precision
- `POST /api/v1/performance/gradient-accumulation` - Configurar gradient accumulation
- `POST /api/v1/performance/profile` - Perfilar modelo
- `POST /api/v1/performance/optimize-batch-size` - Optimizar batch size
- `POST /api/v1/data/datasets` - Crear dataset
- `POST /api/v1/data/datasets/{dataset_id}/loaders` - Crear DataLoader
- `POST /api/v1/data/datasets/{dataset_id}/split` - Dividir dataset
- `GET /api/v1/data/datasets/{dataset_id}/stats` - Estadísticas del dataset
- `POST /api/v1/config/create` - Crear configuración
- `POST /api/v1/config/load-yaml` - Cargar configuración desde YAML
- `POST /api/v1/config/training` - Crear configuración de entrenamiento
- `POST /api/v1/config/merge` - Fusionar configuraciones
- `GET /api/v1/config/{config_id}/export` - Exportar configuración
- `POST /api/v1/logging/initialize` - Inicializar logging (Tensorboard/WandB)
- `POST /api/v1/logging/metrics` - Registrar métricas
- `POST /api/v1/logging/finish` - Finalizar logging
- `POST /api/v1/serving/export` - Exportar modelo (TorchScript/ONNX/TensorRT)
- `POST /api/v1/serving/quantize` - Cuantizar modelo
- `POST /api/v1/serving/deploy` - Crear deployment
- `GET /api/v1/serving/deployments/{deployment_id}/status` - Estado del deployment
- `POST /api/v1/serving/deployments/{deployment_id}/scale` - Escalar deployment
- `POST /api/v1/training/scheduler` - Crear scheduler de learning rate
- `POST /api/v1/training/early-stopping` - Configurar early stopping
- `POST /api/v1/training/gradient-clipping` - Configurar gradient clipping
- `POST /api/v1/training/checkpointing` - Configurar checkpointing
- `POST /api/v1/training/loop` - Crear loop de entrenamiento avanzado

### ML Ops y Gestión Avanzada de Modelos
- `POST /api/v1/evaluation/evaluate` - Evaluar modelo con métricas
- `POST /api/v1/evaluation/cross-validate` - Cross-validation
- `POST /api/v1/evaluation/compare` - Comparar modelos
- `POST /api/v1/tuning/studies` - Crear estudio de optimización
- `POST /api/v1/tuning/studies/{study_id}/optimize` - Ejecutar optimización
- `GET /api/v1/tuning/studies/{study_id}/best-params` - Mejores parámetros
- `POST /api/v1/compression/prune` - Aplicar pruning
- `POST /api/v1/compression/quantize` - Aplicar cuantización
- `POST /api/v1/compression/distill` - Knowledge distillation
- `POST /api/v1/interpretability/explain` - Explicar predicción
- `POST /api/v1/interpretability/attention` - Visualizar atención
- `GET /api/v1/interpretability/importance/{model_id}` - Importancia de características
- `POST /api/v1/registry/models` - Registar modelo
- `POST /api/v1/registry/models/{model_id}/versions` - Crear versión
- `POST /api/v1/registry/versions/{version_id}/promote` - Promover versión
- `GET /api/v1/registry/models/{model_id}/latest` - Última versión
- `POST /api/v1/monitoring/register` - Registrar para monitoreo
- `POST /api/v1/monitoring/predictions` - Registrar predicción
- `GET /api/v1/monitoring/health/{model_id}` - Verificar salud
- `POST /api/v1/monitoring/drift` - Detectar drift
- `GET /api/v1/monitoring/dashboard` - Dashboard de monitoreo

### Técnicas Avanzadas de Machine Learning
- `POST /api/v1/transfer-learning/create` - Crear modelo de transfer learning
- `POST /api/v1/multitask/create` - Crear modelo multi-task
- `POST /api/v1/continual-learning/create` - Crear modelo de continual learning
- `POST /api/v1/nas/search` - Búsqueda de arquitectura neuronal
- `POST /api/v1/automl/experiment` - Crear experimento AutoML
- `POST /api/v1/ensembling/create` - Crear ensemble de modelos

### Funcionalidades Expertas de ML
- `POST /api/v1/transformers/gpt` - Crear modelo GPT
- `POST /api/v1/transformers/bert` - Crear modelo BERT
- `POST /api/v1/transformers/t5` - Crear modelo T5
- `POST /api/v1/diffusion/controlnet` - Crear pipeline ControlNet
- `POST /api/v1/diffusion/lora` - Aplicar LoRA a diffusion
- `POST /api/v1/prompts/template` - Crear template de prompt
- `POST /api/v1/prompts/optimize` - Optimizar prompt
- `POST /api/v1/prompts/rag` - Crear pipeline RAG
- `POST /api/v1/optimization/onnx` - Convertir a ONNX
- `POST /api/v1/optimization/tensorrt` - Optimizar con TensorRT
- `POST /api/v1/distributed/horovod` - Configurar Horovod
- `POST /api/v1/distributed/deepspeed` - Configurar DeepSpeed
- `POST /api/v1/distributed/fsdp` - Configurar FSDP
- `POST /api/v1/quantization/qat` - Aplicar QAT
- `POST /api/v1/quantization/dynamic` - Cuantización dinámica
- `POST /api/v1/quantization/static` - Cuantización estática

### Herramientas Avanzadas de Entrenamiento
- `POST /api/v1/validation/validate` - Validar modelo
- `POST /api/v1/augmentation/pipeline` - Crear pipeline de augmentation
- `POST /api/v1/loss/custom` - Crear loss function personalizada
- `POST /api/v1/optimizers/create` - Crear optimizer avanzado
- `POST /api/v1/lr-finder/search` - Encontrar learning rate óptimo
- `POST /api/v1/debug/model` - Depurar modelo

### Utilidades y Herramientas Adicionales
- `POST /api/v1/visualization/architecture` - Visualizar arquitectura
- `POST /api/v1/visualization/training-curves` - Visualizar curvas de entrenamiento
- `POST /api/v1/comparison/models` - Comparar modelos
- `POST /api/v1/batch/process` - Procesar batch optimizado
- `POST /api/v1/memory/optimize` - Optimizar memoria
- `POST /api/v1/conversion/convert` - Convertir formato de modelo
- `POST /api/v1/metrics/track` - Trackear métricas
- `POST /api/v1/metrics/compute` - Calcular métricas avanzadas

## 🔧 Configuración

El servicio se puede configurar mediante variables de entorno (con prefijo `PSD_`):

### Variables Principales

- `PSD_OPENAI_API_KEY`: API key de OpenAI (opcional)
- `PSD_HOST`: Host del servidor (default: 0.0.0.0)
- `PSD_PORT`: Puerto del servidor (default: 8030)

### Seguridad y Performance

- `PSD_CORS_ORIGINS`: Orígenes permitidos para CORS (default: "*", separados por coma)
- `PSD_CORS_ALLOW_CREDENTIALS`: Permitir credenciales en CORS (default: true)
- `PSD_RATE_LIMIT_PER_MINUTE`: Límite de requests por minuto (default: 60)
- `PSD_API_KEY_HEADER`: Header para API key opcional

### Logging

- `PSD_LOG_LEVEL`: Nivel de logging - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
- `PSD_LOG_FORMAT`: Formato de logs - "json" o "text" (default: json)

### Almacenamiento

- `PSD_STORAGE_PATH`: Ruta base para almacenamiento (default: storage)
- `PSD_DESIGNS_PATH`: Ruta para almacenar diseños (default: storage/designs)

### Performance

- `PSD_MAX_WORKERS`: Número máximo de workers (default: 4)
- `PSD_REQUEST_TIMEOUT`: Timeout de requests en segundos (default: 300)

### Feature Flags

- `PSD_ENABLE_ML_FEATURES`: Habilitar características ML avanzadas (default: false)
- `PSD_ENABLE_DEEP_LEARNING`: Habilitar deep learning (default: false)

### Ejemplo de archivo .env

```bash
PSD_OPENAI_API_KEY=tu_api_key_aqui
PSD_HOST=0.0.0.0
PSD_PORT=8030
PSD_LOG_LEVEL=INFO
PSD_LOG_FORMAT=json
PSD_RATE_LIMIT_PER_MINUTE=60
PSD_CORS_ORIGINS=http://localhost:3000,https://app.example.com
```

## 📝 Ejemplo Completo

```python
import requests

# 1. Crear sesión de chat
session_resp = requests.post("http://localhost:8030/api/v1/chat/session")
session_id = session_resp.json()["session_id"]

# 2. Conversar con la IA
messages = [
    "Quiero abrir una cafetería",
    "Se llamará 'Café del Centro'",
    "Estilo moderno y minimalista",
    "Presupuesto medio",
    "Para jóvenes profesionales"
]

for msg in messages:
    resp = requests.post(
        f"http://localhost:8030/api/v1/chat/{session_id}/message",
        json={"role": "user", "content": msg}
    )
    print(resp.json()["response"])

# 3. Generar diseño
design_resp = requests.post(
    f"http://localhost:8030/api/v1/design/from-chat/{session_id}"
)
design = design_resp.json()

print(f"Diseño generado: {design['store_name']}")
print(f"Marketing: {len(design['marketing_plan']['marketing_strategy'])} estrategias")
print(f"Decoración: {len(design['decoration_plan']['decoration_elements'])} elementos")
```

## ✨ Mejoras Implementadas

### v1.45.0 - Timeout y Circuit Breaker Pattern

- ✅ **Módulo de Timeout**: Nuevo `core/timeout.py` con decorador `@timeout()` y context manager
- ✅ **TimeoutMiddleware**: Middleware automático para timeout de requests HTTP (default: 30s)
- ✅ **Circuit Breaker Pattern**: Nuevo `core/circuit_breaker.py` con implementación completa
- ✅ **CircuitBreaker Class**: Estados CLOSED, OPEN, HALF_OPEN con transiciones automáticas
- ✅ **Decorador @circuit_breaker()**: Aplicar circuit breaker a funciones con configuración flexible
- ✅ **Registry de Circuit Breakers**: Gestión centralizada de múltiples circuit breakers
- ✅ **Endpoint de Estado**: `/circuit-breakers` para consultar estado de todos los circuit breakers
- ✅ **Resiliencia Mejorada**: Protección automática contra fallos en cascada
- ✅ **Recovery Automático**: Transición automática de OPEN a HALF_OPEN después del timeout

### v1.44.0 - Compresión y Rate Limiting Avanzado

- ✅ **Módulo de Compresión**: Nuevo `core/compression.py` con utilidades para comprimir/descomprimir datos (gzip, deflate)
- ✅ **CompressionMiddleware**: Middleware automático para comprimir respuestas HTTP con gzip
- ✅ **Compresión Inteligente**: Solo comprime si reduce el tamaño y el cliente lo acepta
- ✅ **Rate Limiting por Endpoint**: Nuevo `core/rate_limiting.py` con `EndpointRateLimiter` y decorador `@rate_limit_endpoint`
- ✅ **Límites Múltiples**: Soporte para límites por minuto, hora y día por endpoint
- ✅ **Decorador de Rate Limit**: `@rate_limit_endpoint()` para aplicar límites específicos a endpoints
- ✅ **Utilidades de Compresión**: Funciones para calcular ratios de compresión y determinar si comprimir
- ✅ **Middleware Integrado**: CompressionMiddleware agregado al stack de middleware de FastAPI

### v1.43.0 - Serialización y Procesamiento de Tareas en Background

- ✅ **Módulo de Serialización**: Nuevo `core/serializers.py` con utilidades para serialización JSON, base64, y transformación de datos
- ✅ **JSONEncoder Personalizado**: Soporte para datetime, Decimal, Enum, Path, y objetos complejos
- ✅ **Transformación de Datos**: Funciones para snake_case/camelCase, flatten/unflatten, merge de diccionarios
- ✅ **Sanitización**: Función `sanitize_for_json()` para limpiar objetos antes de serialización
- ✅ **Sistema de Background Tasks**: Nuevo `core/background_tasks.py` con `TaskQueue` y `AsyncTaskExecutor`
- ✅ **TaskQueue Thread-Safe**: Cola de tareas con workers, retry logic, y tracking de estado
- ✅ **AsyncTaskExecutor**: Ejecutor asíncrono para tareas en background
- ✅ **Endpoints de Tareas**: `/tasks/{task_id}` para consultar estado de tareas en background
- ✅ **Utilidades de Nested Values**: Funciones para extraer y establecer valores anidados en diccionarios

### v1.42.0 - Sistema de Caché Centralizado y Mejoras de Manejo de Errores

- ✅ **Sistema de Caché Centralizado**: Nuevo módulo `core/cache.py` con `CacheManager`, `LRUCache`, y decorador `@cached`
- ✅ **LRU Cache con TTL**: Implementación thread-safe de LRU cache con soporte para TTL por entrada
- ✅ **Múltiples Caches**: Soporte para múltiples instancias de caché con nombres
- ✅ **Estadísticas de Caché**: Tracking de hits, misses, hit rate, y tamaño
- ✅ **Endpoints de Caché**: `/cache/stats`, `/cache/cleanup`, `/cache/clear` para gestión de caché
- ✅ **Mejoras de Manejo de Errores**: Reemplazo de bloques `except Exception` genéricos con excepciones específicas
- ✅ **Decorador @cached**: Decorador mejorado para cachear resultados de funciones con TTL configurable
- ✅ **Thread-Safe**: Todas las operaciones de caché son thread-safe usando locks

### v1.41.0 - Mejoras en Route Utils y Validadores

- ✅ **Route Utils Extendidos**: Agregadas 7+ funciones nuevas (get_query_params, get_path_params, get_request_body_size, is_json_request, get_request_id, extract_user_agent_info, build_pagination_response)
- ✅ **Validadores Adicionales**: Agregados validadores para UUID, ISO date, list not empty
- ✅ **Funciones de Request**: Mejoras en extracción de información de requests
- ✅ **Paginación Mejorada**: Helper para construir respuestas paginadas con metadata completa
- ✅ **Exportaciones Actualizadas**: Nuevas funciones exportadas en core/__init__.py

### v1.40.0 - Mejoras en Utilidades y Helpers

- ✅ **Utilidades Extendidas**: Agregadas 15+ funciones utilitarias nuevas (deep_merge, get_nested_value, flatten_dict, slugify, etc.)
- ✅ **Decoradores Mejorados**: Corregido bug en log_execution_time para funciones síncronas
- ✅ **Validaciones**: Agregadas funciones de validación (email, URL, domain extraction)
- ✅ **Manipulación de Datos**: Funciones para manipular diccionarios anidados
- ✅ **Formateo**: Funciones adicionales para formateo de datos y texto

### v1.39.0 - Documentación de Mejoras y Resumen de Refactorización

- ✅ **Documentación Completa**: Creado IMPROVEMENTS.md con resumen detallado de todas las mejoras
- ✅ **Estadísticas**: Documentadas 200+ rutas refactorizadas, 30+ servicios mejorados
- ✅ **Arquitectura**: Documentada arquitectura mejorada y beneficios obtenidos
- ✅ **Roadmap**: Sugerencias para próximas mejoras

### v1.38.0 - Refactorización Completa de Advanced DL Routes

- ✅ **Advanced DL Routes Completas**: Todas las rutas restantes refactorizadas (data, config, serving, training)
- ✅ **Eliminación Final**: Reducción de ~11 bloques try-except adicionales
- ✅ **Cobertura Total**: Todas las rutas en advanced_dl_routes.py ahora usan decoradores
- ✅ **Consistencia Total**: Todas las rutas principales del proyecto refactorizadas

### v1.37.0 - Refactorización Final de Rutas y Servicios de Training

- ✅ **Rutas Principales Completas**: Todas las rutas en routes.py refactorizadas (list, export, delete)
- ✅ **Servicios de Training Mejorados**: AdvancedValidation, CustomLoss, AdvancedOptimizers, LRFinder, AdvancedAugmentation ahora usan BaseService
- ✅ **Logging Consolidado**: Todos los servicios mejorados usan logging centralizado
- ✅ **Cobertura Total**: Todas las rutas principales ahora usan decoradores consistentes
- ✅ **Consistencia Total**: Más de 30 servicios ahora heredan de BaseService

### v1.36.0 - Mejora Masiva de Servicios y Rutas Principales

- ✅ **Servicios ML Mejorados**: ContinualLearning, MultiTaskLearning, TransferLearning, NAS, AutoML, Ensembling ahora usan BaseService
- ✅ **Servicios de Infraestructura**: DataPipelineService y ConfigService ahora usan BaseService
- ✅ **Rutas Principales Refactorizadas**: Rutas de chat refactorizadas con decoradores
- ✅ **Logging Consolidado**: Todos los servicios mejorados usan logging centralizado
- ✅ **Consistencia Total**: Más de 10 servicios adicionales ahora heredan de BaseService

### v1.35.0 - Refactorización de Advanced Routes y Mejora de Servicios

- ✅ **Advanced Routes Refactorizadas**: Todas las rutas refactorizadas (compare, technical-plans, feedback, recommendations, location, versions)
- ✅ **Servicios Mejorados**: `DistributedTrainingService` y `MemoryOptimizationService` ahora usan `BaseService`
- ✅ **Logging Consolidado**: Servicios mejorados usan el sistema de logging centralizado
- ✅ **Eliminación Final**: Reducción de ~10 bloques try-except adicionales
- ✅ **Consistencia Total**: Todas las rutas principales ahora usan decoradores consistentes

### v1.34.0 - Refactorización Completa de Todas las Rutas

- ✅ **Expert ML Routes**: Todas las rutas refactorizadas (transformers, diffusion, prompts, optimization, distributed, quantization)
- ✅ **Deep Learning Routes Completas**: Todas las rutas restantes refactorizadas (embeddings, experiments, gradio)
- ✅ **ML Ops Routes Completas**: Todas las rutas restantes refactorizadas (registry, monitoring)
- ✅ **Eliminación Masiva**: Reducción de ~60+ bloques try-except adicionales
- ✅ **Cobertura Total**: Todas las rutas principales ahora usan decoradores consistentes

### v1.33.0 - Refactorización de ML Ops y Deep Learning Routes

- ✅ **ML Ops Routes Refactorizadas**: Rutas principales de evaluación, tuning refactorizadas
- ✅ **Deep Learning Routes Refactorizadas**: Rutas principales de modelos, entrenamiento, predicción refactorizadas
- ✅ **Rutas de Diffusion**: Rutas de pipelines de difusión migradas a decoradores
- ✅ **Eliminación Masiva**: Reducción de ~40+ bloques try-except en ml_ops y deep_learning routes
- ✅ **Consistencia Total**: Todas las rutas principales ahora usan el mismo patrón

### v1.32.0 - Refactorización de Advanced DL Routes y Service Registry

- ✅ **Advanced DL Routes Refactorizadas**: Rutas principales migradas a decoradores
- ✅ **Service Registry**: Nuevo módulo para gestión centralizada de instancias de servicios
- ✅ **Rutas de Performance**: Todas las rutas de optimización de rendimiento refactorizadas
- ✅ **Rutas de Config**: Rutas de configuración migradas a decoradores
- ✅ **Rutas de Logging**: Rutas de experiment logging refactorizadas
- ✅ **Reducción Masiva**: Eliminación de ~30+ bloques try-except en advanced_dl_routes

### v1.31.0 - Refactorización Completa y Utilidades de Rutas

- ✅ **Utility Routes Refactorizadas**: Todas las rutas de utilidades migradas a decoradores
- ✅ **Advanced ML Routes Refactorizadas**: Todas las rutas de ML avanzado migradas
- ✅ **ServiceFactory Extendido**: Método genérico `get_service()` para cualquier servicio
- ✅ **Route Utils**: Nuevo módulo con utilidades para rutas (client info, logging, metrics)
- ✅ **Estadísticas de Factory**: Método `get_stats()` para monitorear instancias de servicios
- ✅ **Mejor Tracking**: Funciones helper para tracking de requests y logging contextual

### v1.30.0 - Refactorización Masiva de Rutas

- ✅ **Rutas Principales Refactorizadas**: `routes.py` ahora usa decoradores `@handle_route_errors` y `@track_route_metrics`
- ✅ **Production Routes**: Todas las rutas de producción migradas a decoradores
- ✅ **Training Tools Routes**: Inicio de migración a decoradores
- ✅ **Eliminación de Try-Except Duplicados**: Reducción masiva de código repetitivo en manejo de errores
- ✅ **Métricas Automáticas**: Todas las rutas ahora trackean métricas automáticamente
- ✅ **Código Más Limpio**: Rutas más legibles y mantenibles sin bloques try-except repetitivos

### v1.29.0 - Consolidación de Servicios y Logging

- ✅ **Consolidación de Clases Base**: Unificación de `BaseService` y `TimestampedService` en `core.service_base`
- ✅ **Migración de Logging**: Servicios migrados de `logging.getLogger` a logging centralizado
- ✅ **Refactorización de Servicios**: `HyperparameterTuningService` y `ExperimentLoggingService` refactorizados
- ✅ **Compatibilidad Hacia Atrás**: `services.base.py` ahora re-exporta clases de `core.service_base`
- ✅ **Mejor Consistencia**: Todos los servicios usan el mismo sistema de logging y estructura base
- ✅ **Reducción de Duplicación**: Eliminación de código duplicado en inicialización de servicios

### v1.28.0 - Refactorización y Mejoras de Arquitectura

- ✅ **Clases Base para Servicios**: `BaseService` y `TimestampedService` para reducir duplicación
- ✅ **Decoradores de Rutas**: `@handle_route_errors` y `@track_route_metrics` para manejo consistente
- ✅ **Utilidades de Servicios**: Métodos comunes para generar IDs, crear respuestas, logging
- ✅ **Refactorización de Servicios**: Servicios migrados a usar clases base y logging centralizado
- ✅ **Reducción de Código Duplicado**: Eliminación de patrones repetitivos de manejo de errores
- ✅ **Mejor Consistencia**: Estructuras de respuesta estandarizadas en todos los servicios

### v1.27.0 - Sistema de Métricas y Monitoreo

- ✅ **Sistema de Métricas**: `MetricsCollector` para recopilar métricas de aplicación (counters, gauges, timers)
- ✅ **Métricas HTTP Automáticas**: Tracking automático de requests, duración y códigos de estado
- ✅ **Context Manager para Timing**: `MetricsContext` y `time_operation()` para medir operaciones
- ✅ **Estadísticas de Timers**: Percentiles (p50, p95, p99), min, max, avg para operaciones
- ✅ **Endpoint de Métricas**: `/metrics` para consultar métricas en tiempo real
- ✅ **Excepciones Adicionales**: `TimeoutError`, `ConflictError`, `TooManyRequestsError`
- ✅ **Thread-Safe**: Métricas protegidas con locks para operaciones concurrentes

### v1.26.0 - Validadores y Dependencias Mejoradas

- ✅ **Validadores Adicionales**: Email, URL, budget range, positive numbers, ranges, string length
- ✅ **Dependencias de Filtrado**: `get_filter_params()` para búsqueda, rangos y estados
- ✅ **Dependencias de Fechas**: `get_date_range_params()` con validación de formato
- ✅ **Modelos de Respuesta Extendidos**: `MetadataResponse`, `ListResponse` para casos adicionales
- ✅ **Validación Mejorada**: Validadores más robustos con regex y type checking

### v1.25.0 - Decoradores y Utilidades Mejoradas

- ✅ **Decorador de Caché**: Nuevo `@cache_result(ttl)` para cachear resultados de funciones
- ✅ **Logging de Tiempo Mejorado**: Uso de `time.perf_counter()` para mayor precisión (milisegundos)
- ✅ **Validación Async/Sync**: Decorador `@validate_input` ahora soporta funciones async y sync
- ✅ **Utilidades Adicionales**: `batch_process()`, `retry_with_backoff()`, `format_bytes()`
- ✅ **Mejor Precisión**: Tiempos de ejecución reportados en milisegundos con mayor precisión

### v1.24.0 - Optimizaciones de Rendimiento y Thread-Safety

- ✅ **ServiceFactory Thread-Safe**: Implementación thread-safe con double-check locking pattern para instancias singleton
- ✅ **Rate Limiting Optimizado**: Uso de `deque` en lugar de listas para mejor rendimiento O(1) en operaciones
- ✅ **CacheMixin Mejorado**: TTL por clave individual, método `clear_expired()` para limpieza automática
- ✅ **Async Lock en Rate Limiting**: Protección thread-safe con `asyncio.Lock` para operaciones concurrentes
- ✅ **Optimización de Memoria**: Deque con maxlen para limitar memoria en rate limiting

### v1.23.0 - Arquitectura Modular Mejorada

- ✅ **Clases Base**: BaseService, StorageMixin, CacheMixin para servicios comunes
- ✅ **Interfaces**: IStorageService, IChatService, IDesignService para contratos claros
- ✅ **Service Factory**: Factory pattern para gestión centralizada de instancias de servicios
- ✅ **Validators Centralizados**: Validator class con validaciones reutilizables
- ✅ **Config Factory**: Factory para configuraciones de aplicación
- ✅ **Organización Mejorada**: Servicios organizados por categorías en __init__.py

### v1.22.3 - Decoradores y Utilidades Adicionales

- ✅ **Decoradores Útiles**: log_execution_time, retry_on_failure, validate_input
- ✅ **Utilidades Adicionales**: generate_id, truncate_text, safe_divide, parse_bool, clamp
- ✅ **Dependencies Mejoradas**: get_sort_params para ordenamiento, validación con Query
- ✅ **Type Hints Mejorados**: Uso de ParamSpec y TypeVar para mejor tipado

### v1.22.2 - Mejoras en Servicios Principales

- ✅ **Servicios Mejorados**: StorageService y ChatService ahora usan logging estructurado y excepciones personalizadas
- ✅ **Manejo de Errores en Storage**: Mejor manejo de errores de permisos y JSON corrupto
- ✅ **Logging Contextual**: Logging con información contextual en todos los servicios principales
- ✅ **Excepciones Consistentes**: Uso de NotFoundError en lugar de ValueError en ChatService

### v1.22.1 - Mejoras Adicionales de Código y Estructura

- ✅ **Modelos de Respuesta Estandarizados**: Modelos Pydantic para respuestas consistentes (SuccessResponse, ErrorResponse, PaginatedResponse)
- ✅ **Dependencies Reutilizables**: Dependencies de FastAPI para API keys y paginación
- ✅ **Rutas Mejoradas**: Uso de excepciones personalizadas en todas las rutas principales
- ✅ **Logging Mejorado**: Logging estructurado con contexto en todas las rutas
- ✅ **Manejo de Errores Robusto**: Manejo de errores parciales sin fallar completamente

### v1.22.0 - Mejoras de Infraestructura y Seguridad

- ✅ **Configuración Mejorada**: Sistema de configuración robusto con validaciones, tipos y variables de entorno
- ✅ **Manejo de Errores Centralizado**: Excepciones personalizadas con códigos de error y detalles estructurados
- ✅ **Logging Estructurado**: Sistema de logging con formato JSON/texto, niveles configurables y contexto enriquecido
- ✅ **Middleware de Seguridad**: Headers de seguridad, rate limiting, logging de requests y manejo centralizado de errores
- ✅ **Validaciones Mejoradas**: Validaciones robustas en modelos Pydantic con mensajes de error claros
- ✅ **Documentación Actualizada**: README actualizado con todas las opciones de configuración

### v1.22.0 - Utilidades y Herramientas Adicionales

- ✅ **Visualización**: Arquitectura, curvas de entrenamiento
- ✅ **Comparación**: Comparación de múltiples modelos
- ✅ **Batch Processing**: Procesamiento optimizado por lotes
- ✅ **Memory Optimization**: Optimización de memoria
- ✅ **Model Conversion**: Conversión entre formatos
- ✅ **Advanced Metrics**: Tracking y cálculo de métricas avanzadas

### v1.21.0 - Herramientas Avanzadas de Entrenamiento

- ✅ **Validación Avanzada**: Testing exhaustivo, múltiples métricas
- ✅ **Data Augmentation**: Pipelines personalizados, múltiples técnicas
- ✅ **Loss Functions**: Funciones de pérdida personalizadas
- ✅ **Optimizers**: Optimizers avanzados configurables
- ✅ **LR Finder**: Búsqueda automática de learning rate óptimo
- ✅ **Model Debugging**: Debugging, verificación de gradientes, detección de NaN

### v1.20.0 - Funcionalidades Expertas de ML

- ✅ **Advanced Transformers**: GPT, BERT, T5 y más arquitecturas
- ✅ **Advanced Diffusion**: ControlNet, LoRA para diffusion
- ✅ **Prompt Engineering**: Templates, optimización, RAG
- ✅ **Model Optimization**: ONNX, TensorRT, graph optimizations
- ✅ **Distributed Training**: Horovod, DeepSpeed, FSDP
- ✅ **Advanced Quantization**: QAT, dynamic, static quantization

### v1.19.0 - Técnicas Avanzadas de Machine Learning

- ✅ **Transfer Learning**: Modelos pre-entrenados, adaptación a nuevas tareas
- ✅ **Multi-task Learning**: Modelos con múltiples tareas, capas compartidas
- ✅ **Continual Learning**: Aprendizaje secuencial, prevención de forgetting
- ✅ **Neural Architecture Search**: Búsqueda automática de arquitecturas
- ✅ **AutoML**: Automatización completa del pipeline ML
- ✅ **Model Ensembling**: Ensembles, voting, stacking, blending

### v1.18.0 - ML Ops y Gestión Avanzada de Modelos

- ✅ **Evaluación de Modelos**: Métricas avanzadas, cross-validation, comparación
- ✅ **Hyperparameter Tuning**: Optuna, optimización bayesiana, visualización
- ✅ **Compresión de Modelos**: Pruning, cuantización, distillation, low-rank
- ✅ **Interpretabilidad**: Explicaciones, atención, importancia, reportes
- ✅ **Model Registry**: Versionado, etapas, búsqueda, comparación
- ✅ **Monitoreo en Producción**: Salud, drift, alertas, dashboard

### v1.17.0 - Optimización de Rendimiento y Técnicas Avanzadas

- ✅ **Optimización de Rendimiento**: Multi-GPU, distributed training, mixed precision, profiling
- ✅ **Pipelines de Datos**: DataLoaders eficientes, augmentation, splits automáticos
- ✅ **Gestión de Configuración**: YAML, configs predefinidas, fusión de configs
- ✅ **Logging de Experimentos**: Tensorboard, WandB, métricas, imágenes, histogramas
- ✅ **Model Serving**: Exportación (TorchScript/ONNX/TensorRT), cuantización, deployments
- ✅ **Técnicas Avanzadas**: Schedulers, early stopping, gradient clipping, checkpointing

### v1.16.0 - Deep Learning y Modelos Avanzados

- ✅ **Deep Learning Integrado**: Modelos personalizados, entrenamiento con PyTorch, checkpoints
- ✅ **Modelos de Difusión**: Generación de imágenes con Stable Diffusion, variaciones de diseños
- ✅ **Fine-tuning de LLMs**: Preparación de datasets, fine-tuning con LoRA/full/p-tuning
- ✅ **Embeddings y Búsqueda Semántica**: Generación de embeddings, búsqueda semántica, diseños similares
- ✅ **Experiment Tracking**: Tracking de experimentos, runs, métricas, comparación de resultados
- ✅ **Integración con Gradio**: Demos interactivos, aplicaciones para diseñador y modelos ML

### v1.15.0 - Tecnologías Finales Avanzadas

- ✅ **IA Multimodal**: Generación multimodal, análisis de múltiples formatos
- ✅ **Comportamiento Predictivo**: Predicción de comportamiento, tráfico futuro
- ✅ **Gestión de Residuos Inteligente**: Contenedores, sensores, analytics, reciclaje
- ✅ **Análisis de Tráfico y Flujo**: Heatmaps, bottlenecks, optimización
- ✅ **Energía Renovable**: Sistemas renovables, generación, créditos, ahorros
- ✅ **Recomendaciones Híbridas**: Combinación colaborativa + ML, explicaciones

### v1.14.0 - Tecnologías Avanzadas Adicionales

- ✅ **Biometría y Reconocimiento**: Enroll, verificación, control de acceso
- ✅ **Realidad Extendida (XR)**: Experiencias XR, showrooms, sesiones
- ✅ **Big Data**: Datasets, queries masivos, análisis de grandes volúmenes
- ✅ **Automatización Robótica**: Registro de robots, asignación de tareas, tracking
- ✅ **Análisis de Video**: Análisis de video, detección de objetos, reconocimiento
- ✅ **Cadena de Suministro**: Proveedores, órdenes, tracking, forecasting

### v1.13.0 - Tecnologías de Próxima Generación

- ✅ **Quantum Computing**: Circuitos cuánticos, ejecución, optimización
- ✅ **Edge Computing**: Dispositivos edge, despliegue, sincronización
- ✅ **Federated Learning**: Modelos federados, rondas, participantes
- ✅ **Análisis de Grafos**: Creación de grafos, análisis, métricas
- ✅ **Simulación Avanzada**: Monte Carlo, eventos discretos, agentes
- ✅ **Logística y Transporte**: Envíos, tracking, optimización de rutas

### v1.12.0 - Tecnologías de Vanguardia

- ✅ **ML Avanzado**: Entrenamiento de modelos personalizados, predicciones, insights
- ✅ **Asistentes de Voz**: Skills para Alexa, Google Assistant, Siri, procesamiento de comandos
- ✅ **Realidad Mixta**: Experiencias MR, showrooms virtuales, sesiones interactivas
- ✅ **Análisis de Mercado en Tiempo Real**: Datos de mercado, tendencias, inteligencia
- ✅ **Recomendaciones Colaborativas**: Filtrado colaborativo, usuarios similares, recomendaciones
- ✅ **Gestión de Energía Inteligente**: Dispositivos, consumo, optimización, ahorros

### v1.11.0 - Tecnologías Futuras y Avanzadas

- ✅ **Blockchain y NFTs**: Contratos inteligentes, NFTs de diseños, verificación de propiedad
- ✅ **Sostenibilidad**: Huella de carbono, evaluación de sostenibilidad, certificaciones
- ✅ **Análisis de Comportamiento**: Perfiles de clientes, heatmaps, journey analysis
- ✅ **Sistemas de Seguridad**: Registro de sistemas, eventos, alertas, estado
- ✅ **Mantenimiento Predictivo**: Predicción de fallas, calendarios, recomendaciones
- ✅ **Sentimiento en Tiempo Real**: Streams, agregados, alertas, tendencias

### v1.10.0 - IoT, ERP y Tecnologías Avanzadas

- ✅ **Realidad Aumentada/Virtual**: Experiencias AR, tours VR, previews
- ✅ **Integración IoT**: Dispositivos, sensores, lecturas, analytics
- ✅ **Inventario Inteligente**: Predicción de demanda, alertas, rotación
- ✅ **Analytics en Tiempo Real**: Métricas, dashboards, detección de anomalías
- ✅ **Integración ERP**: SAP, Oracle, Dynamics, NetSuite, QuickBooks
- ✅ **Compliance y Regulaciones**: Evaluación automática, certificados

### v1.9.0 - Funcionalidades Avanzadas y Automatización

- ✅ **IA Generativa Avanzada**: Generación de imágenes, videos, modelos 3D, copy de marketing
- ✅ **Automatización de Workflows**: Crear, ejecutar, disparar workflows automáticos
- ✅ **Sistema de Reservas**: Servicios, disponibilidad, slots, gestión de citas
- ✅ **Análisis de ROI Avanzado**: ROI, NPV, IRR, comparación de escenarios
- ✅ **Documentación Automática**: Generación automática de documentación completa
- ✅ **Integración de Pagos**: Múltiples proveedores (Stripe, PayPal, Square, MercadoPago)

### v1.8.0 - Engagement y Funcionalidades Avanzadas

- ✅ **Analytics Avanzado**: Tracking, funnels, retención, journey
- ✅ **Gamificación**: Niveles, puntos, logros, badges, leaderboard
- ✅ **Marketplace**: Vender/comprar diseños, reviews, favoritos
- ✅ **Integración CRM**: Contactos, deals, pipeline, sincronización
- ✅ **Sistema de Lealtad**: Puntos, tiers, recompensas, referidos
- ✅ **Análisis de Competencia en Tiempo Real**: Monitoreo continuo, comparación, alertas

- ✅ **Analytics Avanzado**: Tracking de eventos, funnels, retención, journey de usuario
- ✅ **Gamificación**: Niveles, puntos, logros, badges, leaderboard
- ✅ **Marketplace**: Vender/comprar diseños, reviews, favoritos
- ✅ **Integración CRM**: Contactos, deals, pipeline, sincronización
- ✅ **Sistema de Lealtad**: Puntos, tiers, recompensas, referidos

### v1.7.0 - Funcionalidades de Negocio

- ✅ **Integración con Proveedores**: Registro, cotizaciones, recomendaciones
- ✅ **Sistema de Facturación**: Suscripciones, facturas, pagos, planes
- ✅ **Análisis de Sentimiento**: Análisis de texto y feedback
- ✅ **Recomendaciones ML**: Personalizadas, basadas en éxito, perfil de usuario
- ✅ **Integración con Redes Sociales**: Contenido, calendario, anuncios
- ✅ **Sistema de A/B Testing**: Tests, variantes, conversiones, resultados

### v1.6.0 - Integraciones y Exportación Avanzada

- ✅ **Integración con APIs Externas**: Google Maps, clima, lugares cercanos
- ✅ **Sistema de Backup**: Crear, restaurar, listar, exportar backups
- ✅ **Exportación Avanzada**: CAD/DXF, 3D (OBJ/STL), SVG, PDF avanzado
- ✅ **Sistema de Webhooks**: Registro, eventos, notificaciones automáticas
- ✅ **Sistema de Caché**: Caché inteligente, decoradores, limpieza automática

### v1.5.0 - Funcionalidades Enterprise

- ✅ **Sistema de Autenticación**: Registro, login, JWT tokens, gestión de usuarios
- ✅ **Optimización Avanzada**: Optimización de presupuesto, layout, marketing
- ✅ **Análisis Predictivo**: Predicción de éxito, ingresos, tráfico de clientes
- ✅ **Monitoreo y Alertas**: Salud de diseños, alertas, métricas, tracking

### v1.4.0 - Funcionalidades Premium

- ✅ **Sistema de Reportes Avanzados**: Reportes completos, PDF, Excel
- ✅ **Colaboración y Compartir**: Compartir diseños con permisos, comentarios, respuestas
- ✅ **Dashboard Completo**: Estadísticas, tendencias, insights, actividad reciente
- ✅ **Templates Predefinidos**: 6+ templates listos para usar
- ✅ **Análisis de Tendencias**: Tendencias de tipos, estilos, presupuestos, predicciones
- ✅ **Sistema de Notificaciones**: Notificaciones por usuario, tipos, prioridades

### v1.3.0 - Funcionalidades Avanzadas

- ✅ **Comparación de Diseños**: Compara múltiples diseños por costo, rentabilidad, estilo, etc.
- ✅ **Planos Técnicos Detallados**: Planos eléctricos, plomería, HVAC, iluminación, accesibilidad, seguridad
- ✅ **Sistema de Feedback**: Agregar feedback, generar sugerencias de mejora, iterar diseños
- ✅ **Recomendaciones Inteligentes**: Acciones inmediatas, optimizaciones, alertas de riesgo, oportunidades
- ✅ **Análisis de Ubicación**: Análisis de tráfico, visibilidad, accesibilidad, demografía, competencia
- ✅ **Sistema de Versionado**: Crear versiones, comparar versiones, aprobar versiones, historial completo

### v1.2.0 - Análisis Avanzados

- ✅ **Análisis de Competencia**: Análisis completo de competidores en el área
- ✅ **Análisis Financiero**: Proyecciones financieras, punto de equilibrio, análisis de 12 meses
- ✅ **Recomendaciones de Inventario**: Guía completa de inventario por tipo de tienda
- ✅ **Sistema de KPIs**: Métricas y KPIs personalizados por tipo de negocio
- ✅ **Dashboard de Métricas**: Métricas clave para seguimiento diario, semanal y mensual
- ✅ **Análisis Integrado**: Todos los análisis incluidos automáticamente en cada diseño

### v1.1.0 - Mejoras Avanzadas

- ✅ **Integración LLM Robusta**: Chat inteligente con OpenAI GPT-4
- ✅ **Extracción Inteligente de Información**: Uso de LLM para extraer información del chat
- ✅ **Sistema de Persistencia**: Almacenamiento de diseños en archivos JSON
- ✅ **Exportación de Diseños**: Exportar a JSON, Markdown y HTML
- ✅ **Mejores Validaciones**: Validación robusta de datos y manejo de errores
- ✅ **Respuestas Contextuales Mejoradas**: Chat más inteligente y conversacional
- ✅ **Templates Mejorados**: Planes de marketing y decoración más detallados

### Características Adicionales

- **Persistencia**: Los diseños se guardan automáticamente en `storage/designs/`
- **Exportación**: Exporta tus diseños en múltiples formatos
- **Chat Inteligente**: El chat ahora usa LLM para respuestas más naturales
- **Extracción Automática**: La IA extrae automáticamente información de la conversación

## 🚧 Próximas Mejoras

- [ ] Integración con modelos 3D
- [ ] Generación de planos técnicos
- [ ] Estimación de costos más detallada
- [ ] Integración con proveedores de muebles
- [ ] Exportación a formatos CAD
- [ ] Vista previa interactiva 3D
- [ ] Base de datos para producción
- [ ] Sistema de usuarios y autenticación

## 📄 Licencia

Propietaria - Blatam Academy

