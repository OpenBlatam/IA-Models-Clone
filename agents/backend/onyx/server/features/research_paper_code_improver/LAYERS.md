# Organización por Capas - Research Paper Code Improver

## Capa 1: Presentation Layer

### API Routes (`api/`)
- `routes.py` - Endpoints REST principales
- `dashboard_routes.py` - Dashboard web HTML
- `schemas.py` - Esquemas Pydantic para validación
- `decorators.py` - Decoradores para manejo de errores
- `helpers.py` - Factory functions y utilidades

**Responsabilidades:**
- Manejo de requests HTTP
- Validación de entrada
- Transformación de respuestas
- Manejo de errores HTTP

## Capa 2: Application Layer

### Core Application (`core/` - Módulos principales)

**Paper Processing:**
- `paper_extractor.py` - Extracción de información de papers
- `paper_storage.py` - Almacenamiento persistente de papers

**Model Training:**
- `model_trainer.py` - Entrenamiento básico de modelos
- `advanced_model_trainer.py` - Entrenamiento avanzado
- `transformer_finetuner.py` - Fine-tuning de transformers
- `distributed_training.py` - Entrenamiento distribuido

**Code Improvement:**
- `code_improver.py` - Mejora de código principal
- `code_analyzer.py` - Análisis estático de código
- `batch_processor.py` - Procesamiento en lote

**RAG & Search:**
- `rag_engine.py` - Motor RAG
- `vector_store.py` - Almacenamiento vectorial
- `smart_search.py` - Búsqueda inteligente
- `advanced_search.py` - Búsqueda avanzada

**Utilities:**
- `common_utils.py` - Utilidades compartidas (device, tensors)
- `core_utils.py` - Utilidades core (logging, paths, storage)
- `constants.py` - Constantes del sistema
- `base_classes.py` - Clases base (BaseManager, BaseTrainer, etc.)

## Capa 3: Infrastructure Layer

### Storage & Persistence
- `paper_storage.py` - Almacenamiento de papers
- `file_storage.py` - Almacenamiento de archivos
- `model_registry.py` - Registro de modelos
- `model_versioning.py` - Versionado de modelos
- `backup_manager.py` - Gestión de backups

### Caching
- `cache_manager.py` - Gestión de cache
- `model_caching.py` - Cache de modelos
- `distributed_cache.py` - Cache distribuido

### External Integrations
- `github_integration.py` (utils/) - GitHub API
- `git_integration.py` - Git operations
- `integration_manager.py` - Gestión de integraciones

### Processing Utilities
- `pdf_processor.py` (utils/) - Procesamiento PDF
- `link_downloader.py` (utils/) - Descarga de links
- `exporters.py` (utils/) - Exportación de resultados

### Monitoring & Metrics
- `metrics_collector.py` - Recolección de métricas
- `health_monitor.py` - Monitoreo de salud
- `production_monitor.py` - Monitoreo de producción
- `model_monitoring.py` - Monitoreo de modelos
- `analytics_engine.py` - Motor de analytics

## Capa 4: Deep Learning Layer

### Training
- `advanced_model_trainer.py` - Entrenador avanzado
- `distributed_training.py` - Entrenamiento distribuido
- `early_stopping.py` - Early stopping
- `model_checkpointer.py` - Checkpointing
- `training_callbacks.py` - Callbacks de entrenamiento
- `training_config.py` - Configuración de entrenamiento

### Optimization
- `model_optimization_pipeline.py` - Pipeline de optimización
- `hyperparameter_optimizer.py` - Optimización de hiperparámetros
- `hyperparameter_autotuning.py` - Auto-tuning
- `learning_rate_finder.py` - Búsqueda de learning rate
- `optimizer_scheduler.py` - Optimizadores y schedulers
- `gradient_clipping.py` - Gradient clipping
- `mixed_precision.py` - Mixed precision training
- `weight_initialization.py` - Inicialización de pesos

### Model Compression
- `model_compression.py` - Compresión básica
- `advanced_compression.py` - Compresión avanzada
- `advanced_pruning.py` - Pruning avanzado
- `advanced_quantization.py` - Cuantización avanzada

### Model Management
- `model_registry.py` - Registro de modelos
- `model_versioning.py` - Versionado
- `model_lifecycle.py` - Ciclo de vida
- `model_recommendation.py` - Recomendación
- `model_export.py` - Exportación
- `model_serving.py` - Serving
- `model_serving_optimizer.py` - Optimización de serving

### Evaluation
- `model_evaluator.py` - Evaluación básica
- `model_benchmarking.py` - Benchmarking
- `model_profiler.py` - Profiling
- `model_validation.py` - Validación
- `model_robustness.py` - Robustez
- `model_calibration.py` - Calibración
- `model_comparison.py` - Comparación
- `cross_validation.py` - Cross validation
- `model_quality_assurance.py` - QA de modelos

### Advanced Learning
- `automl_system.py` - Sistema AutoML
- `neural_architecture_search.py` - Neural Architecture Search
- `transfer_learning.py` - Transfer learning
- `knowledge_distillation.py` - Knowledge distillation
- `continual_learning.py` - Continual learning
- `federated_learning.py` - Federated learning
- `active_learning.py` - Active learning
- `meta_learning.py` - Meta learning

### Ensemble & Advanced
- `model_ensemble.py` - Ensemble básico
- `advanced_ensemble.py` - Ensemble avanzado
- `uncertainty_estimation.py` - Estimación de incertidumbre

### Data Processing
- `data_preprocessing.py` - Preprocesamiento
- `data_augmentation.py` - Data augmentation
- `data_pipeline.py` - Pipeline de datos
- `data_pipeline_manager.py` - Gestión de pipelines
- `dataloader_optimizer.py` - Optimización de dataloaders
- `feature_store.py` - Feature store

### Specialized Models
- `diffusion_pipeline.py` - Diffusion models
- `multimodal_pipeline.py` - Multimodal
- `custom_attention.py` - Attention personalizado

### Loss Functions
- `loss_function_manager.py` - Gestión de loss functions

### Debugging & Analysis
- `model_debugging.py` - Debugging
- `nan_inf_detector.py` - Detección de NaN/Inf
- `model_interpretability.py` - Interpretabilidad
- `model_fairness.py` - Fairness
- `model_security.py` - Seguridad de modelos
- `model_visualization.py` - Visualización
- `model_efficiency.py` - Eficiencia
- `model_documentation.py` - Documentación
- `model_reproducibility.py` - Reproducibilidad

### Performance & Cost
- `performance_predictor.py` - Predicción de performance
- `cost_estimator.py` - Estimación de costos
- `performance_optimizer.py` - Optimización de performance

### Serving & Inference
- `batch_inference.py` - Inferencia en lote
- `model_serving.py` - Serving de modelos
- `model_serving_optimizer.py` - Optimización de serving

## Capa 5: Enterprise Features Layer

### Security
- `auth_manager.py` - Gestión de autenticación
- `security_manager.py` - Gestión de seguridad
- `advanced_security.py` - Seguridad avanzada
- `secret_management.py` - Gestión de secrets

### API Management
- `api_gateway.py` - API Gateway
- `api_versioning.py` - Versionado de API
- `api_throttling.py` - Throttling
- `advanced_rate_limiting.py` - Rate limiting avanzado
- `rate_limiter.py` - Rate limiter básico
- `api_testing.py` - Testing de API
- `api_mocking.py` - Mocking de API
- `api_documentation.py` - Documentación de API

### Resilience
- `circuit_breaker.py` - Circuit breaker
- `retry_policies.py` - Políticas de retry
- `distributed_locking.py` - Distributed locking
- `chaos_engineering.py` - Chaos engineering

### Observability
- `distributed_tracing.py` - Distributed tracing
- `message_queue.py` - Message queue
- `notification_system.py` - Sistema de notificaciones
- `alert_system.py` - Sistema de alertas

### Workflow & Orchestration
- `workflow_engine.py` - Motor de workflows
- `task_queue.py` - Cola de tareas
- `scheduled_tasks.py` - Tareas programadas
- `event_sourcing.py` - Event sourcing

### Collaboration
- `collaboration_system.py` - Sistema de colaboración
- `realtime_collaboration.py` - Colaboración en tiempo real
- `websocket_manager.py` - Gestión de WebSockets
- `streaming_api.py` - API streaming

### Enterprise Management
- `multi_tenant.py` - Multi-tenant
- `billing_system.py` - Sistema de billing
- `compliance_manager.py` - Gestión de compliance
- `disaster_recovery.py` - Disaster recovery
- `deployment_manager.py` - Gestión de deployment
- `model_rollback.py` - Rollback de modelos
- `database_migrations.py` - Migraciones de BD

### Testing & Quality
- `test_generator.py` - Generación de tests
- `test_runner.py` - Ejecución de tests
- `advanced_testing.py` - Testing avanzado
- `ab_testing.py` - A/B testing
- `performance_testing.py` - Performance testing

### Integration & Transformation
- `request_transformer.py` - Transformación de requests
- `batch_api.py` - API batch
- `graphql_api.py` - GraphQL API
- `advanced_validator.py` - Validación avanzada

### Documentation & Reporting
- `doc_generator.py` - Generación de documentación
- `report_generator.py` - Generación de reportes
- `interactive_docs.py` - Documentación interactiva

### Other Enterprise
- `plugin_system.py` - Sistema de plugins
- `template_system.py` - Sistema de plantillas
- `version_manager.py` - Gestión de versiones
- `feedback_system.py` - Sistema de feedback
- `recommendation_engine.py` - Motor de recomendaciones
- `webhook_manager.py` - Gestión de webhooks
- `service_discovery.py` - Service discovery
- `load_balancer.py` - Load balancer
- `auto_scaler.py` - Auto-scaling
- `feature_flags.py` - Feature flags
- `ml_pipeline.py` - ML Pipeline
- `ml_learner.py` - ML Learner
- `gradio_integration.py` - Integración Gradio

## Resumen por Capa

### Presentation Layer (5 módulos)
- API routes, schemas, decorators, helpers, dashboard

### Application Layer (15 módulos principales)
- Paper extraction, model training, code improvement, RAG, utilities

### Infrastructure Layer (25 módulos)
- Storage, caching, integrations, monitoring, processing

### Deep Learning Layer (80+ módulos)
- Training, optimization, evaluation, advanced learning, serving

### Enterprise Features Layer (60+ módulos)
- Security, API management, resilience, observability, enterprise tools

## Dependencias entre Capas

```
Presentation → Application → Infrastructure
                ↓
         Deep Learning
                ↓
         Enterprise Features
```

**Reglas:**
- Capas superiores pueden usar capas inferiores
- Capas inferiores NO usan capas superiores
- Misma capa puede tener dependencias internas
- Deep Learning puede usar Infrastructure directamente

