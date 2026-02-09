# Arquitectura Interna Modular - Robot Movement AI

## 🎯 Visión General

Arquitectura modular moderna tipo microservicios para el sistema Robot Movement AI, siguiendo las mejores prácticas de diseño de sistemas distribuidos y IA.

## 📋 Módulos del Sistema

### 1. **access** - Control de Acceso y Permisos
**Propósito**: Gestión de control de acceso basado en roles (RBAC), permisos y políticas de seguridad.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para controladores de acceso
- `service.py` - Servicio principal de gestión de acceso
- `rbac.py` - Implementación RBAC
- `policies.py` - Definición de políticas

**Dependencias**:
- `auth` - Autenticación de usuarios
- `db` - Almacenamiento de permisos
- `redis` - Cache de permisos

---

### 2. **agents** - Agentes Autónomos e IA
**Propósito**: Sistema de agentes autónomos para control robótico, planificación y toma de decisiones.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para agentes
- `service.py` - Servicio de orquestación de agentes
- `robot_agent.py` - Agente especializado para robots
- `planner_agent.py` - Agente de planificación

**Dependencias**:
- `llm` - Modelos de lenguaje para agentes
- `tools` - Herramientas disponibles para agentes
- `tracing` - Trazabilidad de decisiones
- `kg` - Knowledge graph para contexto

---

### 3. **auth** - Autenticación y Autorización
**Propósito**: Sistema completo de autenticación (JWT, OAuth2), sesiones y gestión de usuarios.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para autenticadores
- `service.py` - Servicio de autenticación
- `jwt_handler.py` - Manejo de tokens JWT
- `oauth2.py` - Integración OAuth2
- `session_manager.py` - Gestión de sesiones

**Dependencias**:
- `db` - Almacenamiento de usuarios y sesiones
- `redis` - Cache de sesiones activas
- `tracing` - Auditoría de autenticación

---

### 4. **background** - Tareas en Segundo Plano
**Propósito**: Sistema de procesamiento asíncrono, colas de tareas y workers.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para tareas
- `service.py` - Servicio de gestión de tareas
- `queue_manager.py` - Gestión de colas
- `worker.py` - Workers para procesamiento
- `scheduler.py` - Programador de tareas

**Dependencias**:
- `redis` - Colas de mensajes
- `db` - Persistencia de tareas
- `tracing` - Monitoreo de tareas

---

### 5. **chat** - Sistema de Chat y Conversación
**Propósito**: Interfaz conversacional para control robótico, procesamiento de mensajes y gestión de conversaciones.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para controladores de chat
- `service.py` - Servicio principal de chat
- `message_processor.py` - Procesamiento de mensajes
- `conversation_manager.py` - Gestión de conversaciones
- `websocket_handler.py` - Manejo de WebSockets

**Dependencias**:
- `llm` - Generación de respuestas
- `natural_language_processing` - Procesamiento de lenguaje natural
- `agents` - Agentes conversacionales
- `prompts` - Plantillas de prompts
- `tracing` - Logging de conversaciones

---

### 6. **configs** - Configuración del Sistema
**Propósito**: Gestión centralizada de configuración, variables de entorno y settings dinámicos.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para configuradores
- `service.py` - Servicio de configuración
- `env_loader.py` - Carga de variables de entorno
- `yaml_loader.py` - Carga de archivos YAML
- `hot_reload.py` - Recarga dinámica de configuración

**Dependencias**:
- `db` - Almacenamiento de configuraciones
- `redis` - Cache de configuración
- `feature_flags` - Feature flags

---

### 7. **connectors** - Conectores Externos
**Propósito**: Integración con sistemas externos, APIs de terceros y protocolos de comunicación.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para conectores
- `service.py` - Servicio de gestión de conectores
- `http_connector.py` - Conectores HTTP/REST
- `websocket_connector.py` - Conectores WebSocket
- `ros_connector.py` - Conector ROS/ROS2

**Dependencias**:
- `httpx` - Cliente HTTP asíncrono
- `tracing` - Monitoreo de conexiones
- `auth` - Autenticación para APIs externas

---

### 8. **context/search** - Búsqueda Contextual
**Propósito**: Sistema de búsqueda semántica, recuperación de información y gestión de contexto.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para buscadores
- `service.py` - Servicio de búsqueda
- `semantic_search.py` - Búsqueda semántica
- `vector_search.py` - Búsqueda vectorial
- `context_manager.py` - Gestión de contexto

**Dependencias**:
- `document_index` - Índice de documentos
- `indexing` - Sistema de indexación
- `llm` - Embeddings para búsqueda
- `kg` - Knowledge graph

---

### 9. **db** - Base de Datos
**Propósito**: Abstracción de base de datos, ORM, migraciones y gestión de conexiones.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para modelos
- `service.py` - Servicio de base de datos
- `connection_pool.py` - Pool de conexiones
- `migrations.py` - Sistema de migraciones
- `models.py` - Modelos de datos base

**Dependencias**:
- `tracing` - Monitoreo de queries
- `configs` - Configuración de BD

---

### 10. **document_index** - Índice de Documentos
**Propósito**: Indexación y búsqueda de documentos, gestión de embeddings y metadatos.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para indexadores
- `service.py` - Servicio de indexación
- `embedding_generator.py` - Generación de embeddings
- `vector_store.py` - Almacenamiento vectorial
- `metadata_manager.py` - Gestión de metadatos

**Dependencias**:
- `indexing` - Sistema de indexación
- `llm` - Modelos para embeddings
- `file_store` - Almacenamiento de archivos
- `db` - Metadatos de documentos

---

### 11. **evals** - Evaluación y Testing
**Propósito**: Sistema de evaluación de modelos, métricas, benchmarks y testing automatizado.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para evaluadores
- `service.py` - Servicio de evaluación
- `metrics.py` - Definición de métricas
- `benchmark.py` - Benchmarks
- `test_runner.py` - Ejecutor de tests

**Dependencias**:
- `llm` - Modelos a evaluar
- `db` - Almacenamiento de resultados
- `tracing` - Logging de evaluaciones

---

### 12. **feature_flags** - Feature Flags
**Propósito**: Sistema de feature flags, A/B testing y control de lanzamiento de características.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para feature flags
- `service.py` - Servicio de feature flags
- `flag_manager.py` - Gestión de flags
- `ab_testing.py` - A/B testing

**Dependencias**:
- `db` - Almacenamiento de flags
- `redis` - Cache de flags
- `tracing` - Analytics de flags

---

### 13. **federated_connectors** - Conectores Federados
**Propósito**: Conectores para sistemas federados, sincronización distribuida y federated learning.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para conectores federados
- `service.py` - Servicio de federación
- `federated_learning.py` - Federated learning
- `sync_manager.py` - Gestión de sincronización
- `node_manager.py` - Gestión de nodos

**Dependencias**:
- `connectors` - Conectores base
- `db` - Estado federado
- `tracing` - Monitoreo distribuido

---

### 14. **file_processing** - Procesamiento de Archivos
**Propósito**: Procesamiento de archivos, conversión de formatos y extracción de contenido.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para procesadores
- `service.py` - Servicio de procesamiento
- `extractors.py` - Extractores de contenido
- `converters.py` - Convertidores de formato
- `validators.py` - Validadores de archivos

**Dependencias**:
- `file_store` - Almacenamiento de archivos
- `document_index` - Indexación de contenido
- `indexing` - Sistema de indexación

---

### 15. **file_store** - Almacenamiento de Archivos
**Propósito**: Sistema de almacenamiento de archivos, gestión de objetos y storage distribuido.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para storage
- `service.py` - Servicio de almacenamiento
- `local_storage.py` - Almacenamiento local
- `s3_storage.py` - Almacenamiento S3
- `storage_manager.py` - Gestor de storage

**Dependencias**:
- `configs` - Configuración de storage
- `tracing` - Monitoreo de operaciones

---

### 16. **httpx** - Cliente HTTP Asíncrono
**Propósito**: Cliente HTTP asíncrono, gestión de requests/responses y manejo de errores HTTP.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para clientes HTTP
- `service.py` - Servicio HTTP
- `client.py` - Cliente HTTP principal
- `middleware.py` - Middleware HTTP
- `retry_handler.py` - Manejo de reintentos

**Dependencias**:
- `tracing` - Monitoreo de requests
- `configs` - Configuración HTTP

---

### 17. **indexing** - Sistema de Indexación
**Propósito**: Sistema de indexación de contenido, gestión de índices y optimización de búsquedas.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para indexadores
- `service.py` - Servicio de indexación
- `index_manager.py` - Gestor de índices
- `index_builder.py` - Constructor de índices
- `index_optimizer.py` - Optimizador de índices

**Dependencias**:
- `document_index` - Índice de documentos
- `db` - Almacenamiento de índices
- `redis` - Cache de índices

---

### 18. **key_value_store** - Almacenamiento Clave-Valor
**Propósito**: Sistema de almacenamiento clave-valor, cache distribuido y almacenamiento temporal.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para KV stores
- `service.py` - Servicio de KV store
- `cache_manager.py` - Gestor de cache
- `distributed_store.py` - Store distribuido

**Dependencias**:
- `redis` - Backend de Redis
- `db` - Persistencia opcional
- `configs` - Configuración

---

### 19. **kg** - Knowledge Graph
**Propósito**: Sistema de knowledge graph, relaciones semánticas y consultas de grafos.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para knowledge graphs
- `service.py` - Servicio de knowledge graph
- `graph_builder.py` - Constructor de grafos
- `query_engine.py` - Motor de consultas
- `relationship_manager.py` - Gestión de relaciones

**Dependencias**:
- `db` - Almacenamiento de grafos
- `llm` - Extracción de relaciones
- `indexing` - Indexación de grafos
- `context/search` - Búsqueda contextual

---

### 20. **llm** - Modelos de Lenguaje
**Propósito**: Integración con LLMs, gestión de modelos, inferencia y fine-tuning.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para LLMs
- `service.py` - Servicio de LLM
- `model_manager.py` - Gestor de modelos
- `inference_engine.py` - Motor de inferencia
- `prompt_engine.py` - Motor de prompts

**Dependencias**:
- `prompts` - Plantillas de prompts
- `tracing` - Monitoreo de inferencia
- `configs` - Configuración de modelos
- `tools` - Herramientas para LLMs

---

### 21. **natural_language_processing** - Procesamiento de Lenguaje Natural
**Propósito**: Procesamiento de texto, análisis sintáctico, extracción de entidades y NLP avanzado.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para procesadores NLP
- `service.py` - Servicio de NLP
- `tokenizer.py` - Tokenización
- `ner.py` - Named Entity Recognition
- `sentiment.py` - Análisis de sentimiento
- `parser.py` - Análisis sintáctico

**Dependencias**:
- `llm` - Modelos de lenguaje
- `context/search` - Búsqueda contextual
- `kg` - Knowledge graph

---

### 22. **onyxbot/slack** - Integración Slack
**Propósito**: Integración con Slack, bots, comandos y notificaciones.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para bots
- `service.py` - Servicio de Slack
- `bot.py` - Bot principal
- `commands.py` - Manejo de comandos
- `notifications.py` - Sistema de notificaciones

**Dependencias**:
- `chat` - Sistema de chat
- `agents` - Agentes para Slack
- `auth` - Autenticación Slack

---

### 23. **prompts** - Gestión de Prompts
**Propósito**: Sistema de gestión de prompts, plantillas, versionado y optimización.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para prompts
- `service.py` - Servicio de prompts
- `template_manager.py` - Gestor de plantillas
- `prompt_optimizer.py` - Optimizador de prompts
- `version_manager.py` - Versionado de prompts

**Dependencias**:
- `llm` - Uso de prompts
- `db` - Almacenamiento de prompts
- `evals` - Evaluación de prompts

---

### 24. **redis** - Integración Redis
**Propósito**: Cliente Redis, gestión de cache, pub/sub y operaciones Redis avanzadas.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para clientes Redis
- `service.py` - Servicio de Redis
- `client.py` - Cliente Redis principal
- `cache.py` - Sistema de cache
- `pubsub.py` - Pub/Sub

**Dependencias**:
- `configs` - Configuración de Redis
- `tracing` - Monitoreo de operaciones

---

### 25. **secondary_llm_flows** - Flujos Secundarios de LLM
**Propósito**: Flujos alternativos de LLM, fallbacks, routing de modelos y estrategias de inferencia.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para flujos
- `service.py` - Servicio de flujos
- `flow_manager.py` - Gestor de flujos
- `fallback_handler.py` - Manejo de fallbacks
- `routing_strategy.py` - Estrategias de routing

**Dependencias**:
- `llm` - Modelos principales
- `configs` - Configuración de flujos
- `tracing` - Monitoreo de flujos

---

### 26. **seeding** - Datos Iniciales
**Propósito**: Sistema de seeding de datos, inicialización de base de datos y datos de prueba.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para seeders
- `service.py` - Servicio de seeding
- `data_seeder.py` - Seeder principal
- `fixtures.py` - Datos de prueba

**Dependencias**:
- `db` - Base de datos
- `configs` - Configuración

---

### 27. **server** - Servidor Principal
**Propósito**: Servidor HTTP, API REST, GraphQL, WebSockets y endpoints principales.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para servidores
- `service.py` - Servicio de servidor
- `api_server.py` - Servidor API REST
- `graphql_server.py` - Servidor GraphQL
- `websocket_server.py` - Servidor WebSocket
- `middleware.py` - Middleware del servidor

**Dependencias**:
- `auth` - Autenticación
- `access` - Control de acceso
- `tracing` - Observabilidad
- `configs` - Configuración
- Todos los módulos de negocio

---

### 28. **tools** - Herramientas y Utilidades
**Propósito**: Herramientas para agentes, funciones ejecutables y utilidades del sistema.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para herramientas
- `service.py` - Servicio de herramientas
- `tool_registry.py` - Registro de herramientas
- `robot_tools.py` - Herramientas robóticas
- `system_tools.py` - Herramientas del sistema

**Dependencias**:
- `agents` - Uso por agentes
- `llm` - Herramientas para LLMs
- `tracing` - Logging de herramientas

---

### 29. **tracing** - Trazabilidad y Observabilidad
**Propósito**: Sistema de tracing distribuido, logging estructurado, métricas y observabilidad.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para tracers
- `service.py` - Servicio de tracing
- `tracer.py` - Tracer principal
- `logger.py` - Logger estructurado
- `metrics.py` - Sistema de métricas
- `span_manager.py` - Gestión de spans

**Dependencias**:
- `configs` - Configuración de tracing
- `db` - Almacenamiento de traces (opcional)

---

### 30. **utils** - Utilidades Generales
**Propósito**: Utilidades generales, helpers, validadores y funciones comunes.

**Archivos mínimos**:
- `__init__.py`
- `base.py` - Clase base para utilidades
- `service.py` - Servicio de utilidades
- `validators.py` - Validadores
- `formatters.py` - Formateadores
- `helpers.py` - Funciones helper

**Dependencias**:
- Ninguna (módulo base)

---

## 🔗 Mapa de Dependencias

```
┌─────────────┐
│   configs   │ (Base - sin dependencias)
└──────┬──────┘
       │
       ├──> db ──> tracing
       ├──> redis ──> tracing
       ├──> httpx ──> tracing
       │
       ├──> auth ──> db, redis, tracing
       │     │
       │     └──> access ──> db, redis
       │
       ├──> llm ──> prompts, tracing, tools
       │     │
       │     ├──> agents ──> tools, kg, tracing
       │     ├──> chat ──> nlp, prompts, agents
       │     ├──> natural_language_processing
       │     └──> secondary_llm_flows
       │
       ├──> file_store ──> tracing
       │     │
       │     └──> file_processing ──> document_index, indexing
       │
       ├──> document_index ──> indexing, llm, file_store, db
       │     │
       │     └──> context/search ──> indexing, kg
       │
       ├──> indexing ──> document_index, db, redis
       │
       ├──> kg ──> db, llm, indexing, context/search
       │
       ├──> key_value_store ──> redis, db
       │
       ├──> background ──> redis, db, tracing
       │
       ├──> connectors ──> httpx, tracing, auth
       │     │
       │     └──> federated_connectors ──> db, tracing
       │
       ├──> server ──> auth, access, tracing, configs, [todos los módulos]
       │
       ├──> onyxbot/slack ──> chat, agents, auth
       │
       ├──> tools ──> agents, llm, tracing
       │
       ├──> prompts ──> llm, db, evals
       │
       ├──> evals ──> llm, db, tracing
       │
       └──> seeding ──> db, configs
```

## 📁 Estructura de Directorios Final

```
robot_movement_ai/
├── __init__.py
├── access/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── rbac.py
│   └── policies.py
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── robot_agent.py
│   └── planner_agent.py
├── auth/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── jwt_handler.py
│   ├── oauth2.py
│   └── session_manager.py
├── background/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── queue_manager.py
│   ├── worker.py
│   └── scheduler.py
├── chat/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── message_processor.py
│   ├── conversation_manager.py
│   └── websocket_handler.py
├── configs/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── env_loader.py
│   ├── yaml_loader.py
│   └── hot_reload.py
├── connectors/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── http_connector.py
│   ├── websocket_connector.py
│   └── ros_connector.py
├── context/
│   └── search/
│       ├── __init__.py
│       ├── base.py
│       ├── service.py
│       ├── semantic_search.py
│       ├── vector_search.py
│       └── context_manager.py
├── db/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── connection_pool.py
│   ├── migrations.py
│   └── models.py
├── document_index/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── embedding_generator.py
│   ├── vector_store.py
│   └── metadata_manager.py
├── evals/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── metrics.py
│   ├── benchmark.py
│   └── test_runner.py
├── feature_flags/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── flag_manager.py
│   └── ab_testing.py
├── federated_connectors/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── federated_learning.py
│   ├── sync_manager.py
│   └── node_manager.py
├── file_processing/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── extractors.py
│   ├── converters.py
│   └── validators.py
├── file_store/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── local_storage.py
│   ├── s3_storage.py
│   └── storage_manager.py
├── httpx/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── client.py
│   ├── middleware.py
│   └── retry_handler.py
├── indexing/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── index_manager.py
│   ├── index_builder.py
│   └── index_optimizer.py
├── key_value_store/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── cache_manager.py
│   └── distributed_store.py
├── kg/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── graph_builder.py
│   ├── query_engine.py
│   └── relationship_manager.py
├── llm/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── model_manager.py
│   ├── inference_engine.py
│   └── prompt_engine.py
├── natural_language_processing/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── tokenizer.py
│   ├── ner.py
│   ├── sentiment.py
│   └── parser.py
├── onyxbot/
│   └── slack/
│       ├── __init__.py
│       ├── base.py
│       ├── service.py
│       ├── bot.py
│       ├── commands.py
│       └── notifications.py
├── prompts/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── template_manager.py
│   ├── prompt_optimizer.py
│   └── version_manager.py
├── redis/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── client.py
│   ├── cache.py
│   └── pubsub.py
├── secondary_llm_flows/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── flow_manager.py
│   ├── fallback_handler.py
│   └── routing_strategy.py
├── seeding/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── data_seeder.py
│   └── fixtures.py
├── server/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── api_server.py
│   ├── graphql_server.py
│   ├── websocket_server.py
│   └── middleware.py
├── tools/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── tool_registry.py
│   ├── robot_tools.py
│   └── system_tools.py
├── tracing/
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   ├── tracer.py
│   ├── logger.py
│   ├── metrics.py
│   └── span_manager.py
└── utils/
    ├── __init__.py
    ├── base.py
    ├── service.py
    ├── validators.py
    ├── formatters.py
    └── helpers.py
```

## 🚀 Principios de Diseño

1. **Separación de Responsabilidades**: Cada módulo tiene una responsabilidad única y bien definida
2. **Dependencias Mínimas**: Módulos dependen solo de lo estrictamente necesario
3. **Interfaces Claras**: Clases base (`base.py`) definen contratos claros
4. **Servicios Centralizados**: Cada módulo expone un servicio principal (`service.py`)
5. **Observabilidad**: Todos los módulos integran tracing para monitoreo
6. **Configuración Centralizada**: `configs` es el módulo base para configuración
7. **Extensibilidad**: Fácil agregar nuevos módulos siguiendo el patrón establecido

## 📝 Notas de Implementación

- Todos los módulos deben seguir el patrón: `base.py` (interfaces) + `service.py` (implementación)
- Las dependencias deben ser inyectadas, no importadas directamente
- Usar dependency injection para facilitar testing
- Implementar logging estructurado usando el módulo `tracing`
- Todas las operaciones asíncronas deben usar `async/await`
- Implementar circuit breakers para dependencias externas
- Cachear resultados cuando sea apropiado usando `redis` o `key_value_store`

