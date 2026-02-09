# Arquitectura Modular - Gamma App

## Visión General

Esta arquitectura modular está diseñada para un sistema de IA moderno con microservicios, siguiendo principios de separación de responsabilidades, escalabilidad y mantenibilidad.

## Módulos y Propósitos

### 1. **access** - Control de Acceso y Permisos
**Propósito**: Gestiona el control de acceso basado en roles (RBAC), permisos granulares y políticas de autorización.

**Responsabilidades**:
- Validación de permisos de usuario
- Gestión de roles y políticas
- Control de acceso a recursos
- Auditoría de accesos

**Dependencias**: `auth`, `db`, `utils`

---

### 2. **agents** - Sistema de Agentes de IA
**Propósito**: Framework para agentes autónomos de IA que pueden ejecutar tareas complejas de forma independiente.

**Responsabilidades**:
- Definición de agentes y sus capacidades
- Orquestación de agentes
- Comunicación entre agentes
- Gestión del ciclo de vida de agentes

**Dependencias**: `llm`, `tools`, `tracing`, `prompts`, `context/search`

---

### 3. **auth** - Autenticación y Autorización
**Propósito**: Maneja la autenticación de usuarios, tokens JWT, OAuth, y sesiones.

**Responsabilidades**:
- Autenticación de usuarios
- Generación y validación de tokens
- Gestión de sesiones
- Integración con proveedores OAuth

**Dependencias**: `db`, `redis`, `utils`

---

### 4. **background** - Tareas en Segundo Plano
**Propósito**: Sistema de procesamiento asíncrono para tareas de larga duración.

**Responsabilidades**:
- Cola de tareas asíncronas
- Workers y procesamiento distribuido
- Retry y manejo de errores
- Monitoreo de tareas

**Dependencias**: `redis`, `db`, `tracing`, `utils`

---

### 5. **chat** - Sistema de Chat y Conversaciones
**Propósito**: Motor de chat conversacional con IA, gestión de conversaciones y contexto.

**Responsabilidades**:
- Gestión de conversaciones
- Mantenimiento de contexto
- Integración con LLMs
- Historial y persistencia

**Dependencias**: `llm`, `db`, `context/search`, `prompts`, `tracing`

---

### 6. **configs** - Configuración del Sistema
**Propósito**: Gestión centralizada de configuración, variables de entorno y settings.

**Responsabilidades**:
- Carga de configuración
- Validación de settings
- Gestión de entornos (dev, staging, prod)
- Hot-reload de configuración

**Dependencias**: `utils`

---

### 7. **connectors** - Conectores Externos
**Propósito**: Integraciones con servicios externos (APIs, bases de datos, servicios cloud).

**Responsabilidades**:
- Clientes para APIs externas
- Adaptadores para diferentes servicios
- Manejo de autenticación externa
- Retry y circuit breakers

**Dependencias**: `httpx`, `utils`, `tracing`

---

### 8. **context/search** - Búsqueda y Contexto
**Propósito**: Sistema de búsqueda semántica, recuperación de contexto y RAG (Retrieval Augmented Generation).

**Responsabilidades**:
- Búsqueda vectorial
- Embeddings y similitud
- Gestión de contexto
- RAG pipeline

**Dependencias**: `document_index`, `kg`, `llm`, `db`

---

### 9. **db** - Base de Datos
**Propósito**: Abstracción de acceso a datos, ORM, migraciones y gestión de conexiones.

**Responsabilidades**:
- Conexiones a base de datos
- Modelos y esquemas
- Migraciones
- Transacciones y queries

**Dependencias**: `configs`, `utils`

---

### 10. **document_index** - Índice de Documentos
**Propósito**: Indexación y búsqueda de documentos, gestión de embeddings y metadatos.

**Responsabilidades**:
- Indexación de documentos
- Almacenamiento de embeddings
- Búsqueda por contenido
- Gestión de versiones

**Dependencias**: `db`, `file_store`, `indexing`, `llm`

---

### 11. **evals** - Evaluación de Modelos
**Propósito**: Framework para evaluar modelos de IA, métricas y benchmarks.

**Responsabilidades**:
- Ejecución de evaluaciones
- Métricas y scoring
- Comparación de modelos
- Reportes de evaluación

**Dependencias**: `llm`, `db`, `tracing`, `utils`

---

### 12. **feature_flags** - Feature Flags
**Propósito**: Sistema de feature flags para habilitar/deshabilitar funcionalidades dinámicamente.

**Responsabilidades**:
- Gestión de flags
- Evaluación de condiciones
- Rollout gradual
- A/B testing

**Dependencias**: `db`, `redis`, `configs`

---

### 13. **federated_connectors** - Conectores Federados
**Propósito**: Conectores para sistemas federados, APIs distribuidas y servicios descentralizados.

**Responsabilidades**:
- Conexión a sistemas federados
- Sincronización de datos
- Resolución de conflictos
- Protocolos federados

**Dependencias**: `connectors`, `httpx`, `db`, `tracing`

---

### 14. **file_processing** - Procesamiento de Archivos
**Propósito**: Procesamiento de archivos (PDF, DOCX, imágenes, etc.), extracción de contenido y conversión.

**Responsabilidades**:
- Extracción de texto
- Conversión de formatos
- Procesamiento de imágenes
- OCR y parsing

**Dependencias**: `file_store`, `utils`, `tracing`

---

### 15. **file_store** - Almacenamiento de Archivos
**Propósito**: Sistema de almacenamiento de archivos (local, S3, cloud storage).

**Responsabilidades**:
- Upload/download de archivos
- Gestión de almacenamiento
- CDN y caching
- Compresión y optimización

**Dependencias**: `configs`, `utils`

---

### 16. **httpx** - Cliente HTTP
**Propósito**: Cliente HTTP asíncrono con retry, timeouts y manejo de errores.

**Responsabilidades**:
- Requests HTTP asíncronos
- Retry automático
- Circuit breakers
- Logging de requests

**Dependencias**: `configs`, `tracing`, `utils`

---

### 17. **indexing** - Sistema de Indexación
**Propósito**: Motor de indexación para búsqueda rápida, full-text search y índices vectoriales.

**Responsabilidades**:
- Creación de índices
- Actualización incremental
- Búsqueda optimizada
- Gestión de índices

**Dependencias**: `db`, `document_index`, `utils`

---

### 18. **key_value_store** - Almacenamiento Key-Value
**Propósito**: Abstracción para almacenamiento key-value (Redis, Memcached, etc.).

**Responsabilidades**:
- Operaciones CRUD
- TTL y expiración
- Transacciones
- Clustering

**Dependencias**: `redis`, `configs`, `utils`

---

### 19. **kg** - Knowledge Graph
**Propósito**: Gestión de grafos de conocimiento, relaciones entre entidades y navegación semántica.

**Responsabilidades**:
- Construcción de grafos
- Consultas de grafos
- Inferencia de relaciones
- Visualización

**Dependencias**: `db`, `llm`, `natural_language_processing`, `indexing`

---

### 20. **llm** - Large Language Models
**Propósito**: Abstracción para interactuar con diferentes modelos de lenguaje (OpenAI, Anthropic, local, etc.).

**Responsabilidades**:
- Invocación de LLMs
- Gestión de prompts
- Streaming de respuestas
- Caching de respuestas

**Dependencias**: `httpx`, `prompts`, `redis`, `tracing`, `configs`

---

### 21. **natural_language_processing** - Procesamiento de Lenguaje Natural
**Propósito**: Funciones de NLP (tokenización, NER, análisis de sentimiento, etc.).

**Responsabilidades**:
- Tokenización
- Análisis de sentimiento
- Extracción de entidades
- Análisis de texto

**Dependencias**: `llm`, `utils`

---

### 22. **onyxbot/slack** - Integración Slack
**Propósito**: Bot de Slack y integración con el workspace de Slack.

**Responsabilidades**:
- Comandos de Slack
- Eventos y webhooks
- Mensajes y notificaciones
- Integración con agentes

**Dependencias**: `agents`, `chat`, `auth`, `httpx`, `tracing`

---

### 23. **prompts** - Gestión de Prompts
**Propósito**: Sistema de gestión de prompts, templates y versionado.

**Responsabilidades**:
- Templates de prompts
- Versionado de prompts
- Optimización de prompts
- A/B testing de prompts

**Dependencias**: `db`, `llm`, `utils`

---

### 24. **redis** - Cliente Redis
**Propósito**: Cliente Redis para caching, pub/sub y colas.

**Responsabilidades**:
- Conexiones Redis
- Operaciones de cache
- Pub/Sub
- Gestión de conexiones

**Dependencias**: `configs`, `utils`

---

### 25. **secondary_llm_flows** - Flujos Secundarios de LLM
**Propósito**: Flujos alternativos de LLM para tareas especializadas (validación, refinamiento, etc.).

**Responsabilidades**:
- Flujos de validación
- Refinamiento de respuestas
- Post-procesamiento
- Flujos de fallback

**Dependencias**: `llm`, `prompts`, `tracing`, `utils`

---

### 26. **seeding** - Datos Iniciales
**Propósito**: Scripts y utilidades para poblar la base de datos con datos iniciales.

**Responsabilidades**:
- Seeders de datos
- Fixtures
- Datos de prueba
- Migración de datos

**Dependencias**: `db`, `configs`, `utils`

---

### 27. **server** - Servidor HTTP
**Propósito**: Servidor FastAPI/HTTP, rutas, middleware y endpoints.

**Responsabilidades**:
- Aplicación FastAPI
- Rutas y endpoints
- Middleware
- WebSockets

**Dependencias**: `auth`, `access`, `chat`, `agents`, `tracing`, `configs`

---

### 28. **tools** - Herramientas y Utilidades
**Propósito**: Herramientas que los agentes pueden usar (búsqueda web, calculadora, etc.).

**Responsabilidades**:
- Definición de herramientas
- Ejecución de herramientas
- Registro de herramientas
- Validación de herramientas

**Dependencias**: `httpx`, `db`, `utils`, `tracing`

---

### 29. **tracing** - Trazabilidad y Observabilidad
**Propósito**: Sistema de tracing distribuido, logging estructurado y métricas.

**Responsabilidades**:
- Distributed tracing
- Logging estructurado
- Métricas y monitoring
- Correlación de eventos

**Dependencias**: `configs`, `utils`

---

### 30. **utils** - Utilidades Generales
**Propósito**: Utilidades compartidas, helpers y funciones comunes.

**Responsabilidades**:
- Helpers generales
- Validación
- Formateo
- Constantes

**Dependencias**: Ninguna (módulo base)

---

## Mapa de Dependencias

```
utils (base)
  ├── configs
  │   ├── db
  │   ├── redis
  │   ├── httpx
  │   ├── file_store
  │   └── tracing
  │
  ├── auth
  │   ├── access
  │   └── server
  │
  ├── db
  │   ├── document_index
  │   ├── indexing
  │   ├── kg
  │   ├── prompts
  │   ├── evals
  │   ├── feature_flags
  │   └── seeding
  │
  ├── redis
  │   ├── auth
  │   ├── background
  │   ├── key_value_store
  │   └── llm
  │
  ├── httpx
  │   ├── connectors
  │   ├── federated_connectors
  │   ├── llm
  │   └── tools
  │
  ├── llm
  │   ├── agents
  │   ├── chat
  │   ├── context/search
  │   ├── document_index
  │   ├── evals
  │   ├── kg
  │   ├── natural_language_processing
  │   ├── secondary_llm_flows
  │   └── prompts
  │
  ├── prompts
  │   ├── agents
  │   ├── chat
  │   ├── llm
  │   └── secondary_llm_flows
  │
  ├── context/search
  │   ├── agents
  │   ├── chat
  │   └── document_index
  │
  ├── document_index
  │   ├── context/search
  │   └── indexing
  │
  ├── file_store
  │   ├── file_processing
  │   └── document_index
  │
  ├── agents
  │   ├── onyxbot/slack
  │   └── server
  │
  ├── chat
  │   ├── server
  │   └── onyxbot/slack
  │
  └── server
      └── (entry point)
```

## Principios de Diseño

1. **Separación de Responsabilidades**: Cada módulo tiene una responsabilidad única y bien definida.
2. **Dependencias Unidireccionales**: Las dependencias fluyen en una dirección clara, evitando ciclos.
3. **Interfaces Claras**: Cada módulo expone interfaces bien definidas a través de `base.py` y `service.py`.
4. **Testabilidad**: Cada módulo puede ser testeado de forma independiente.
5. **Escalabilidad**: Los módulos pueden escalarse independientemente.
6. **Observabilidad**: Todos los módulos integran tracing y logging.

## Convenciones de Archivos

Cada módulo debe contener:
- `__init__.py`: Exportaciones públicas del módulo
- `base.py`: Clases base, interfaces y tipos fundamentales
- `service.py`: Implementación principal del servicio del módulo

Archivos adicionales según necesidad:
- `models.py`: Modelos de datos
- `exceptions.py`: Excepciones específicas
- `config.py`: Configuración del módulo
- `tests/`: Tests del módulo

