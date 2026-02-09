# Arquitectura - Research Paper Code Improver

## Visión General

Sistema de IA enterprise que entrena modelos basados en papers de investigación y utiliza ese conocimiento para mejorar código de GitHub mediante técnicas de RAG (Retrieval Augmented Generation) y fine-tuning.

## Principios Arquitectónicos

- **Modularidad**: Separación clara de responsabilidades
- **Escalabilidad**: Diseñado para crecer horizontalmente
- **Mantenibilidad**: Código limpio y bien organizado
- **Extensibilidad**: Fácil agregar nuevas funcionalidades
- **Performance**: Optimizado para velocidad y eficiencia
- **Seguridad**: Best practices de seguridad implementadas

## Capas Arquitectónicas

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   REST API   │  │   Dashboard  │  │   WebSocket  │  │
│  │  (FastAPI)   │  │   (HTML/JS)  │  │  (Real-time) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Routes     │  │  Decorators  │  │   Helpers    │  │
│  │  (Endpoints) │  │  (Error      │  │  (Factories) │  │
│  │              │  │   Handling)  │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                    Business Logic Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Paper      │  │    Model     │  │     Code     │  │
│  │  Extractor   │  │   Trainer    │  │  Improver    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Vector     │  │     RAG      │  │   Code       │  │
│  │   Store      │  │   Engine     │  │  Analyzer    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Storage    │  │    Cache     │  │   Metrics    │  │
│  │  (Papers,    │  │   Manager    │  │  Collector   │  │
│  │   Models)    │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Vector DB  │  │   GitHub     │  │   PDF        │  │
│  │  (ChromaDB)  │  │ Integration  │  │  Processor   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                    Deep Learning Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Training    │  │ Optimization │  │  Evaluation  │  │
│  │  Modules     │  │   Pipeline   │  │   Modules    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  AutoML      │  │  Ensemble    │  │  Serving     │  │
│  │  System      │  │   Methods    │  │  Optimizer   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Estructura de Directorios

```
research_paper_code_improver/
├── api/                          # Presentation Layer
│   ├── routes.py                # Endpoints principales
│   ├── schemas.py               # Esquemas Pydantic
│   ├── decorators.py            # Decoradores (error handling)
│   ├── helpers.py               # Factory functions
│   └── dashboard_routes.py      # Dashboard web
│
├── core/                         # Business Logic Layer
│   ├── paper_extractor.py       # Extracción de papers
│   ├── model_trainer.py         # Entrenamiento
│   ├── code_improver.py         # Mejora de código
│   ├── vector_store.py          # Vector database
│   ├── rag_engine.py            # RAG implementation
│   ├── paper_storage.py         # Persistencia de papers
│   ├── code_analyzer.py         # Análisis de código
│   ├── cache_manager.py         # Gestión de cache
│   ├── batch_processor.py       # Procesamiento en lote
│   │
│   ├── common_utils.py          # Utilidades compartidas
│   ├── constants.py             # Constantes
│   ├── base_classes.py          # Clases base
│   ├── core_utils.py            # Utilidades core
│   │
│   └── [150+ módulos DL]        # Deep Learning modules
│
├── utils/                        # Infrastructure Layer
│   ├── pdf_processor.py          # Procesamiento PDF
│   ├── link_downloader.py       # Descarga de papers
│   ├── github_integration.py    # GitHub API
│   └── exporters.py             # Exportación de resultados
│
├── models/                       # Data Models
│   └── paper_model.py           # Modelos Pydantic
│
├── config/                       # Configuration
│   └── settings.py              # Configuración del sistema
│
├── training/                     # Training Scripts
│   └── train.py                 # Script de entrenamiento
│
├── data/                         # Data Storage
│   ├── papers/                  # PDFs descargados
│   ├── embeddings/              # Embeddings generados
│   ├── models/                  # Modelos entrenados
│   └── vector_db/               # ChromaDB data
│
└── main.py                      # Application Entry Point
```

## Flujos de Datos Principales

### 1. Flujo de Extracción de Papers

```
User Upload/URL
    │
    ├─→ [API Route] → PaperExtractor
    │                      │
    │                      ├─→ PDFProcessor (si PDF)
    │                      │      └─→ Extrae: texto, metadatos, secciones
    │                      │
    │                      └─→ LinkDownloader (si URL)
    │                             └─→ Descarga → PDFProcessor
    │
    ├─→ PaperStorage.save_paper()
    │      └─→ Guarda en data/papers/
    │
    └─→ VectorStore.add_paper()
           └─→ Genera embeddings → ChromaDB
```

### 2. Flujo de Entrenamiento

```
Papers Collection
    │
    ├─→ ModelTrainer.prepare_training_data()
    │      └─→ Crea ejemplos: instruction-input-output
    │
    ├─→ ModelTrainer.train_from_papers()
    │      ├─→ Fine-tuning del modelo base
    │      ├─→ Guarda checkpoints
    │      └─→ Guarda modelo final
    │
    └─→ ModelRegistry.register()
           └─→ Metadata del modelo
```

### 3. Flujo de Mejora de Código

```
GitHub Repo + File Path
    │
    ├─→ CodeImprover.improve_code()
    │      │
    │      ├─→ GitHubIntegration.get_file()
    │      │      └─→ Obtiene código original
    │      │
    │      ├─→ RAGEngine.retrieve_relevant_papers()
    │      │      ├─→ VectorStore.search()
    │      │      └─→ Papers relevantes al código
    │      │
    │      ├─→ CodeAnalyzer.analyze_code()
    │      │      └─→ Análisis estático
    │      │
    │      └─→ LLM Inference (con contexto RAG)
    │             └─→ Código mejorado + sugerencias
    │
    └─→ CacheManager.cache_result()
           └─→ Cache para futuras consultas similares
```

## Componentes Principales

### Presentation Layer

**API Routes** (`api/routes.py`)
- Endpoints REST para todas las operaciones
- Manejo de errores centralizado con decoradores
- Validación de entrada con Pydantic

**Decorators** (`api/decorators.py`)
- `@handle_api_errors()` - Manejo consistente de errores
- Logging automático
- Transformación de excepciones a HTTP responses

**Helpers** (`api/helpers.py`)
- Factory functions para crear instancias
- Resolución de paths
- Configuración consistente

### Business Logic Layer

**PaperExtractor**
- Extrae información de PDFs y URLs
- Normaliza datos de papers
- Integra con PaperStorage

**ModelTrainer**
- Prepara datos de entrenamiento
- Fine-tuning de modelos
- Gestión de versiones

**CodeImprover**
- Orquesta el proceso de mejora
- Integra RAG, análisis y LLM
- Genera mejoras y sugerencias

**RAGEngine**
- Búsqueda semántica de papers
- Construcción de contexto
- Integración con LLMs

**VectorStore**
- Almacenamiento de embeddings
- Búsqueda por similitud
- Gestión de colecciones

### Infrastructure Layer

**PaperStorage**
- Persistencia de papers en filesystem
- Índice JSON para búsqueda rápida
- Gestión de metadatos

**CacheManager**
- Cache de resultados de mejoras
- TTL configurable
- Evicción LRU

**MetricsCollector**
- Recolección de métricas
- Estadísticas de uso
- Performance tracking

### Deep Learning Layer

**Training Modules**
- AdvancedModelTrainer
- DistributedTrainer
- EarlyStopping
- ModelCheckpointer

**Optimization Modules**
- ModelOptimizationPipeline
- HyperparameterAutoTuner
- AutoMLSystem
- LearningRateFinder

**Evaluation Modules**
- ModelEvaluator
- ModelBenchmarkingSuite
- ModelQualityAssurance
- CrossValidation

## Patrones Arquitectónicos

### 1. Factory Pattern
- `create_code_improver()` - Crea instancias de CodeImprover
- `get_paper_storage()` - Singleton de PaperStorage

### 2. Repository Pattern
- `PaperStorage` - Abstrae acceso a datos de papers
- `VectorStore` - Abstrae acceso a vector database

### 3. Strategy Pattern
- Múltiples estrategias de optimización
- Diferentes métodos de ensemble
- Varios algoritmos de búsqueda

### 4. Decorator Pattern
- `@handle_api_errors()` - Añade manejo de errores
- `@timing_decorator()` - Añade medición de tiempo

### 5. Singleton Pattern
- `get_paper_storage()` - Una instancia de PaperStorage
- Instancias globales de servicios principales

### 6. Dependency Injection
- Instancias pasadas como parámetros
- Fácil testing y mocking

## Decisiones de Diseño

### Por qué FastAPI
- Performance: Async/await nativo
- Type hints: Validación automática
- Auto-documentación: OpenAPI/Swagger
- Moderno: Basado en estándares

### Por qué ChromaDB
- Embeddings: Optimizado para vectores
- Simplicidad: Fácil de usar
- Persistencia: Almacenamiento local
- Performance: Búsqueda rápida

### Por qué RAG
- Contexto relevante: Papers específicos al código
- Actualizable: Nuevos papers sin reentrenar
- Explicable: Fuentes claras de mejoras
- Eficiente: No requiere fine-tuning constante

### Modularidad
- Lazy loading: Solo carga lo necesario
- Imports explícitos: Dependencias claras
- Separación de concerns: Cada módulo una responsabilidad

## Escalabilidad

### Horizontal Scaling
- Stateless API: Múltiples instancias
- Vector DB distribuida: ChromaDB cluster
- Cache compartido: Redis (futuro)

### Vertical Scaling
- Async processing: No bloquea requests
- Batch processing: Procesa múltiples archivos
- Caching: Reduce carga computacional

### Performance Optimizations
- Lazy imports: Carga rápida
- Connection pooling: Reutiliza conexiones
- Batch operations: Agrupa operaciones
- Caching: Evita recálculos

## Seguridad

### Autenticación
- JWT tokens para GitHub API
- API keys para LLM services
- Environment variables para secrets

### Validación
- Pydantic schemas: Validación de entrada
- File type checking: Solo PDFs permitidos
- Size limits: Archivos limitados

### Sanitización
- Input sanitization: Previene injection
- Path validation: Previene path traversal
- Rate limiting: Previene abuse

## Deployment

### Desarrollo
```bash
uvicorn main:app --reload --port 8030
```

### Producción
```bash
uvicorn main:app --host 0.0.0.0 --port 8030 --workers 4
```

### Docker (futuro)
- Multi-stage build
- Optimizado para producción
- Health checks

### Kubernetes (futuro)
- Horizontal Pod Autoscaler
- Service mesh
- ConfigMaps y Secrets

## Monitoreo y Observabilidad

### Métricas
- Request count y latency
- Error rates
- Model performance
- Cache hit rates

### Logging
- Structured logging
- Log levels configurables
- Request tracing

### Health Checks
- `/health` endpoint
- Component health status
- Dependency checks

## Testing Strategy

### Unit Tests
- Funciones individuales
- Mocks para dependencias externas
- Fast execution

### Integration Tests
- Flujos completos
- Real dependencies
- Database tests

### E2E Tests
- API endpoints
- Complete workflows
- Real scenarios

## Roadmap Arquitectónico

### Corto Plazo
- [ ] Subpaquetes en core/ (training/, evaluation/, etc.)
- [ ] Dependency injection container
- [ ] Redis para cache distribuido
- [ ] Database para metadata

### Mediano Plazo
- [ ] Message queue (RabbitMQ/Kafka)
- [ ] Event sourcing
- [ ] CQRS pattern
- [ ] Microservices split

### Largo Plazo
- [ ] Kubernetes deployment
- [ ] Service mesh
- [ ] Multi-region
- [ ] Edge computing

## Referencias

- [FastAPI Best Practices](https://fastapi.tiangolo.com/)
- [RAG Architecture](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Vector Databases](https://www.pinecone.io/learn/vector-database/)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

