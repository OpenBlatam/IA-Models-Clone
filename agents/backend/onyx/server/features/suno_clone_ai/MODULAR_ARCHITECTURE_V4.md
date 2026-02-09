# Arquitectura Modular V4 - Estructura Interna Completa

## 🎯 Overview

Este documento describe la arquitectura modular interna del repositorio siguiendo un diseño moderno de microservicios/IA modular. Cada módulo tiene responsabilidades claras y bien definidas.

## 📁 Estructura Completa de Módulos

```
suno_clone_ai/
├── access/                    # Control de acceso y permisos
├── agents/                    # Agentes de IA y orquestación
├── auth/                      # Autenticación y autorización
├── background/                # Tareas en segundo plano
├── chat/                      # Sistema de chat y conversación
├── configs/                   # Configuraciones del sistema
├── connectors/                # Conectores externos
├── context/                   # Gestión de contexto
│   └── search/                # Búsqueda contextual
├── db/                        # Base de datos y ORM
├── document_index/            # Indexación de documentos
├── evals/                     # Evaluación y métricas
├── feature_flags/             # Feature flags y toggles
├── federated_connectors/      # Conectores federados
├── file_processing/           # Procesamiento de archivos
├── file_store/                # Almacenamiento de archivos
├── httpx/                     # Cliente HTTP asíncrono
├── indexing/                  # Sistema de indexación
├── key_value_store/           # Almacenamiento clave-valor
├── kg/                        # Knowledge Graph
├── llm/                       # Modelos de lenguaje
├── natural_language_processing/  # Procesamiento de lenguaje natural
├── onyxbot/                   # Bot de Onyx
│   └── slack/                 # Integración con Slack
├── prompts/                   # Gestión de prompts
├── redis/                     # Cliente Redis
├── secondary_llm_flows/       # Flujos secundarios de LLM
├── seeding/                   # Datos iniciales y seeding
├── server/                    # Servidor y endpoints
├── tools/                     # Herramientas y utilidades
├── tracing/                   # Trazabilidad y observabilidad
└── utils/                     # Utilidades generales
```

## 📋 Descripción de Módulos

### 1. `access/` - Control de Acceso y Permisos

**Propósito**: Gestiona el control de acceso basado en roles (RBAC), permisos y políticas de seguridad.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para controladores de acceso
- `service.py` - Servicio de control de acceso
- `models.py` - Modelos de permisos y roles
- `policies.py` - Políticas de acceso

**Dependencias**: `auth/`, `db/`

---

### 2. `agents/` - Agentes de IA y Orquestación

**Propósito**: Define y gestiona agentes de IA, orquestación de tareas complejas y workflows inteligentes.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para agentes
- `service.py` - Servicio de orquestación de agentes
- `orchestrator.py` - Orquestador de agentes
- `registry.py` - Registro de agentes disponibles

**Dependencias**: `llm/`, `tools/`, `prompts/`, `tracing/`

---

### 3. `auth/` - Autenticación y Autorización

**Propósito**: Maneja autenticación de usuarios, tokens JWT, OAuth, y sesiones.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para autenticadores
- `service.py` - Servicio de autenticación
- `jwt_handler.py` - Manejo de tokens JWT
- `oauth.py` - Integración OAuth

**Dependencias**: `db/`, `redis/`, `configs/`

---

### 4. `background/` - Tareas en Segundo Plano

**Propósito**: Gestiona tareas asíncronas, jobs en background, y procesamiento en cola.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para tareas
- `service.py` - Servicio de tareas en background
- `queue.py` - Sistema de colas
- `worker.py` - Workers para procesar tareas

**Dependencias**: `redis/`, `db/`, `tracing/`

---

### 5. `chat/` - Sistema de Chat y Conversación

**Propósito**: Gestiona conversaciones, historial de chat, y procesamiento de mensajes.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para chat
- `service.py` - Servicio de chat
- `conversation.py` - Gestión de conversaciones
- `message_handler.py` - Procesamiento de mensajes

**Dependencias**: `llm/`, `natural_language_processing/`, `db/`, `context/`

---

### 6. `configs/` - Configuraciones del Sistema

**Propósito**: Centraliza todas las configuraciones, variables de entorno, y settings.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para configuraciones
- `settings.py` - Configuraciones principales
- `loader.py` - Cargador de configuraciones

**Dependencias**: Ninguna (módulo base)

---

### 7. `connectors/` - Conectores Externos

**Propósito**: Integraciones con servicios externos, APIs de terceros, y adaptadores.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para conectores
- `service.py` - Servicio de conectores
- `registry.py` - Registro de conectores

**Dependencias**: `httpx/`, `configs/`, `tracing/`

---

### 8. `context/` - Gestión de Contexto

**Propósito**: Gestiona contexto de conversaciones, sesiones, y estado de la aplicación.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para contexto
- `service.py` - Servicio de contexto
- `manager.py` - Gestor de contexto

**Submódulo**: `context/search/` - Búsqueda contextual

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para búsqueda
- `service.py` - Servicio de búsqueda contextual

**Dependencias**: `db/`, `indexing/`, `document_index/`

---

### 9. `db/` - Base de Datos y ORM

**Propósito**: Abstracción de base de datos, modelos ORM, migraciones, y queries.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para modelos
- `service.py` - Servicio de base de datos
- `models.py` - Modelos de datos
- `session.py` - Gestión de sesiones de DB

**Dependencias**: `configs/`

---

### 10. `document_index/` - Indexación de Documentos

**Propósito**: Indexación y búsqueda de documentos, embeddings, y RAG (Retrieval Augmented Generation).

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para indexación
- `service.py` - Servicio de indexación
- `embedder.py` - Generación de embeddings
- `retriever.py` - Recuperación de documentos

**Dependencias**: `llm/`, `indexing/`, `db/`

---

### 11. `evals/` - Evaluación y Métricas

**Propósito**: Evaluación de modelos, métricas de calidad, y benchmarking.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para evaluadores
- `service.py` - Servicio de evaluación
- `metrics.py` - Métricas de evaluación
- `benchmark.py` - Benchmarking

**Dependencias**: `llm/`, `db/`, `tracing/`

---

### 12. `feature_flags/` - Feature Flags y Toggles

**Propósito**: Gestión de feature flags, A/B testing, y toggles de funcionalidades.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para feature flags
- `service.py` - Servicio de feature flags
- `evaluator.py` - Evaluador de flags

**Dependencias**: `redis/`, `db/`, `configs/`

---

### 13. `federated_connectors/` - Conectores Federados

**Propósito**: Conectores para sistemas federados, APIs distribuidas, y servicios descentralizados.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para conectores federados
- `service.py` - Servicio de conectores federados
- `registry.py` - Registro de conectores federados

**Dependencias**: `connectors/`, `httpx/`, `tracing/`

---

### 14. `file_processing/` - Procesamiento de Archivos

**Propósito**: Procesamiento de archivos (audio, texto, imágenes), transformaciones, y validación.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para procesadores
- `service.py` - Servicio de procesamiento
- `audio_processor.py` - Procesamiento de audio
- `text_processor.py` - Procesamiento de texto

**Dependencias**: `file_store/`, `utils/`

---

### 15. `file_store/` - Almacenamiento de Archivos

**Propósito**: Almacenamiento de archivos, gestión de storage (local, S3, etc.), y acceso a archivos.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para storage
- `service.py` - Servicio de almacenamiento
- `local_store.py` - Almacenamiento local
- `s3_store.py` - Almacenamiento S3

**Dependencias**: `configs/`, `utils/`

---

### 16. `httpx/` - Cliente HTTP Asíncrono

**Propósito**: Cliente HTTP asíncrono, manejo de requests/responses, y retry logic.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para cliente HTTP
- `service.py` - Servicio HTTP
- `client.py` - Cliente HTTP principal
- `retry.py` - Lógica de reintentos

**Dependencias**: `configs/`, `tracing/`

---

### 17. `indexing/` - Sistema de Indexación

**Propósito**: Sistema de indexación general, índices invertidos, y búsqueda rápida.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para indexadores
- `service.py` - Servicio de indexación
- `inverted_index.py` - Índice invertido
- `search_engine.py` - Motor de búsqueda

**Dependencias**: `db/`, `key_value_store/`

---

### 18. `key_value_store/` - Almacenamiento Clave-Valor

**Propósito**: Almacenamiento clave-valor, caché, y datos temporales.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para KV store
- `service.py` - Servicio de KV store
- `cache.py` - Sistema de caché

**Dependencias**: `redis/`, `configs/`

---

### 19. `kg/` - Knowledge Graph

**Propósito**: Gestión de knowledge graphs, relaciones entre entidades, y grafos de conocimiento.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para KG
- `service.py` - Servicio de knowledge graph
- `graph.py` - Estructura de grafo
- `query.py` - Consultas al grafo

**Dependencias**: `db/`, `indexing/`, `llm/`

---

### 20. `llm/` - Modelos de Lenguaje

**Propósito**: Integración con LLMs, generación de texto, y manejo de modelos de IA.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para LLMs
- `service.py` - Servicio de LLM
- `provider.py` - Proveedores de LLM (OpenAI, Anthropic, etc.)
- `generator.py` - Generador de texto

**Dependencias**: `configs/`, `prompts/`, `tracing/`

---

### 21. `natural_language_processing/` - Procesamiento de Lenguaje Natural

**Propósito**: NLP, análisis de texto, extracción de entidades, y procesamiento de lenguaje.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para NLP
- `service.py` - Servicio de NLP
- `analyzer.py` - Analizador de texto
- `extractor.py` - Extractor de entidades

**Dependencias**: `llm/`, `utils/`

---

### 22. `onyxbot/` - Bot de Onyx

**Propósito**: Bot principal de Onyx y funcionalidades de bot.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para bot
- `service.py` - Servicio de bot

**Submódulo**: `onyxbot/slack/` - Integración con Slack

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para Slack
- `service.py` - Servicio de Slack
- `handler.py` - Manejador de eventos de Slack

**Dependencias**: `chat/`, `agents/`, `connectors/`

---

### 23. `prompts/` - Gestión de Prompts

**Propósito**: Gestión de prompts, templates, y construcción de prompts dinámicos.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para prompts
- `service.py` - Servicio de prompts
- `template.py` - Templates de prompts
- `builder.py` - Constructor de prompts

**Dependencias**: `configs/`, `utils/`

---

### 24. `redis/` - Cliente Redis

**Propósito**: Cliente Redis, conexiones, y operaciones Redis.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para Redis
- `service.py` - Servicio de Redis
- `client.py` - Cliente Redis principal
- `pool.py` - Pool de conexiones

**Dependencias**: `configs/`

---

### 25. `secondary_llm_flows/` - Flujos Secundarios de LLM

**Propósito**: Flujos secundarios de LLM, pipelines alternativos, y procesamiento paralelo.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para flujos
- `service.py` - Servicio de flujos secundarios
- `pipeline.py` - Pipeline de procesamiento

**Dependencias**: `llm/`, `prompts/`, `tracing/`

---

### 26. `seeding/` - Datos Iniciales y Seeding

**Propósito**: Datos iniciales, seeding de base de datos, y datos de prueba.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para seeders
- `service.py` - Servicio de seeding
- `seeder.py` - Seeder principal

**Dependencias**: `db/`, `configs/`

---

### 27. `server/` - Servidor y Endpoints

**Propósito**: Servidor FastAPI, endpoints, rutas, y API REST.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para servidor
- `service.py` - Servicio del servidor
- `app.py` - Aplicación FastAPI
- `routes.py` - Definición de rutas

**Dependencias**: `auth/`, `access/`, `chat/`, `tracing/`

---

### 28. `tools/` - Herramientas y Utilidades

**Propósito**: Herramientas reutilizables, funciones auxiliares, y utilidades específicas.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para herramientas
- `service.py` - Servicio de herramientas
- `registry.py` - Registro de herramientas

**Dependencias**: `utils/`, `llm/`

---

### 29. `tracing/` - Trazabilidad y Observabilidad

**Propósito**: Trazabilidad, logging estructurado, métricas, y observabilidad.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para tracing
- `service.py` - Servicio de tracing
- `logger.py` - Logger estructurado
- `metrics.py` - Métricas y telemetría

**Dependencias**: `configs/`

---

### 30. `utils/` - Utilidades Generales

**Propósito**: Utilidades generales, helpers, y funciones comunes.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para utilidades
- `helpers.py` - Funciones helper
- `validators.py` - Validadores

**Dependencias**: Ninguna (módulo base)

---

## 🔗 Mapa de Dependencias

```
configs/ (sin dependencias)
    ↓
utils/ (sin dependencias)
    ↓
db/ → configs/
redis/ → configs/
tracing/ → configs/
httpx/ → configs/, tracing/
file_store/ → configs/, utils/
key_value_store/ → redis/, configs/
    ↓
auth/ → db/, redis/, configs/
access/ → auth/, db/
prompts/ → configs/, utils/
llm/ → configs/, prompts/, tracing/
natural_language_processing/ → llm/, utils/
    ↓
file_processing/ → file_store/, utils/
indexing/ → db/, key_value_store/
document_index/ → llm/, indexing/, db/
context/ → db/, indexing/, document_index/
kg/ → db/, indexing/, llm/
    ↓
chat/ → llm/, natural_language_processing/, db/, context/
agents/ → llm/, tools/, prompts/, tracing/
secondary_llm_flows/ → llm/, prompts/, tracing/
tools/ → utils/, llm/
    ↓
connectors/ → httpx/, configs/, tracing/
federated_connectors/ → connectors/, httpx/, tracing/
onyxbot/ → chat/, agents/, connectors/
    ↓
background/ → redis/, db/, tracing/
evals/ → llm/, db/, tracing/
feature_flags/ → redis/, db/, configs/
seeding/ → db/, configs/
    ↓
server/ → auth/, access/, chat/, tracing/
```

## 📦 Estructura de Archivos Base

Cada módulo seguirá esta estructura mínima:

```
module_name/
├── __init__.py          # Exports principales
├── base.py              # Clase base abstracta
├── service.py           # Servicio principal
└── [archivos específicos según necesidad]
```

## 🎯 Principios de Diseño

1. **Separación de Responsabilidades**: Cada módulo tiene una responsabilidad única
2. **Dependencias Mínimas**: Módulos base no dependen de otros módulos
3. **Interfaces Claras**: Clases base definen contratos claros
4. **Inyección de Dependencias**: Servicios reciben dependencias por inyección
5. **Testabilidad**: Cada módulo es testeable de forma independiente
6. **Escalabilidad**: Arquitectura preparada para microservicios

## 🚀 Próximos Pasos

1. Generar estructura de carpetas
2. Crear archivos base para cada módulo
3. Implementar servicios principales
4. Configurar dependencias
5. Documentar APIs de cada módulo

