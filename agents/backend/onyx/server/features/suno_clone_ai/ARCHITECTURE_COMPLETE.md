# Arquitectura Completa - Ecosistema IA Modular

## 📋 Reglas de Importación y Prevención de Ciclos

### Principios Fundamentales

1. **Módulos Base (Sin Dependencias)**
   - `configs/` - No depende de nada
   - `utils/` - No depende de nada

2. **Módulos de Infraestructura (Dependen solo de Base)**
   - `db/` → `configs/`
   - `redis/` → `configs/`
   - `tracing/` → `configs/`
   - `httpx/` → `configs/`, `tracing/`

3. **Módulos de Servicios (Dependen de Infraestructura)**
   - `auth/` → `db/`, `redis/`, `configs/`
   - `llm/` → `configs/`, `prompts/`, `tracing/`
   - `prompts/` → `configs/`, `utils/`

4. **Módulos de Negocio (Dependen de Servicios)**
   - `chat/` → `llm/`, `db/`, `context/`
   - `agents/` → `llm/`, `tools/`, `prompts/`
   - `server/` → `auth/`, `access/`, `chat/`

### Reglas de Importación

```python
# ✅ CORRECTO - Import absoluto desde raíz
from configs import Settings
from db.service import DatabaseService
from llm.service import LLMService

# ✅ CORRECTO - Import relativo dentro del módulo
from .base import BaseService
from .models import User

# ❌ INCORRECTO - Import circular
# En llm/service.py NO hacer:
# from chat.service import ChatService  # ❌

# ✅ CORRECTO - Usar inyección de dependencias
class LLMService:
    def __init__(self, chat_service=None):  # Inyectado desde fuera
        self.chat_service = chat_service
```

### Prevención de Ciclos

1. **Usar TYPE_CHECKING para imports de tipo**
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from chat.service import ChatService
```

2. **Inyección de Dependencias**
   - Los servicios reciben dependencias por constructor
   - No importar servicios que dependen de ti

3. **Interfaces y Abstracciones**
   - Usar clases base (`base.py`) para definir contratos
   - Implementaciones concretas en `service.py`

## 🎯 Rol de Cada Módulo en el Ecosistema IA

### 1. `configs/` - Configuración Central
**Rol**: Base del sistema, sin dependencias. Centraliza todas las configuraciones.
**Uso en IA**: Configuración de modelos, APIs, hiperparámetros.

### 2. `utils/` - Utilidades Generales
**Rol**: Funciones helper reutilizables, sin dependencias.
**Uso en IA**: Procesamiento de datos, validaciones, transformaciones.

### 3. `db/` - Persistencia de Datos
**Rol**: Abstracción de base de datos, ORM, modelos.
**Uso en IA**: Almacenar conversaciones, historial, embeddings, metadata.

### 4. `redis/` - Caché y Estado
**Rol**: Almacenamiento rápido, sesiones, caché.
**Uso en IA**: Caché de respuestas LLM, estado de conversaciones, rate limiting.

### 5. `tracing/` - Observabilidad
**Rol**: Logging, métricas, trazabilidad.
**Uso en IA**: Tracking de llamadas LLM, latencia, tokens usados, errores.

### 6. `httpx/` - Comunicación HTTP
**Rol**: Cliente HTTP asíncrono para APIs externas.
**Uso en IA**: Llamadas a APIs de LLM (OpenAI, Anthropic, etc.).

### 7. `auth/` - Autenticación
**Rol**: Autenticación de usuarios, tokens JWT.
**Uso en IA**: Control de acceso a modelos, rate limiting por usuario.

### 8. `access/` - Control de Acceso
**Rol**: RBAC, permisos, políticas.
**Uso en IA**: Controlar qué usuarios pueden usar qué modelos/features.

### 9. `llm/` - Modelos de Lenguaje
**Rol**: Integración con LLMs, generación de texto.
**Uso en IA**: Core del sistema, generación de respuestas, embeddings.

### 10. `prompts/` - Gestión de Prompts
**Rol**: Templates, construcción de prompts dinámicos.
**Uso en IA**: Optimización de prompts, versionado, A/B testing.

### 11. `chat/` - Sistema de Chat
**Rol**: Conversaciones, historial, procesamiento de mensajes.
**Uso en IA**: Interfaz principal con usuarios, contexto de conversación.

### 12. `agents/` - Agentes de IA
**Rol**: Orquestación de tareas complejas, workflows.
**Uso en IA**: Agentes especializados, multi-step reasoning, tool use.

### 13. `tools/` - Herramientas
**Rol**: Funciones que los agentes pueden usar.
**Uso en IA**: Búsqueda web, cálculos, llamadas a APIs, RAG.

### 14. `context/` - Gestión de Contexto
**Rol**: Contexto de conversaciones, sesiones, memoria.
**Uso en IA**: Window de contexto, memoria a largo plazo, RAG.

### 15. `document_index/` - Indexación de Documentos
**Rol**: RAG, embeddings, búsqueda semántica.
**Uso en IA**: Retrieval Augmented Generation, knowledge base.

### 16. `indexing/` - Sistema de Indexación
**Rol**: Índices invertidos, búsqueda rápida.
**Uso en IA**: Búsqueda de documentos, metadata, tags.

### 17. `kg/` - Knowledge Graph
**Rol**: Grafos de conocimiento, relaciones entre entidades.
**Uso en IA**: Representación de conocimiento, reasoning sobre relaciones.

### 18. `natural_language_processing/` - NLP
**Rol**: Análisis de texto, extracción de entidades, NER.
**Uso en IA**: Preprocesamiento, análisis de sentimiento, clasificación.

### 19. `file_store/` - Almacenamiento
**Rol**: Almacenamiento de archivos (local, S3).
**Uso en IA**: Almacenar audio generado, documentos, modelos.

### 20. `file_processing/` - Procesamiento de Archivos
**Rol**: Procesamiento de audio, texto, imágenes.
**Uso en IA**: Preprocesamiento de audio, transcripción, análisis.

### 21. `key_value_store/` - KV Store
**Rol**: Almacenamiento clave-valor, caché.
**Uso en IA**: Caché de embeddings, resultados de procesamiento.

### 22. `connectors/` - Conectores Externos
**Rol**: Integraciones con servicios externos.
**Uso en IA**: APIs de terceros, servicios de IA externos.

### 23. `federated_connectors/` - Conectores Federados
**Rol**: Sistemas distribuidos, APIs federadas.
**Uso en IA**: LLMs distribuidos, inference distribuido.

### 24. `background/` - Tareas en Background
**Rol**: Jobs asíncronos, procesamiento en cola.
**Uso en IA**: Generación asíncrona, batch processing, training.

### 25. `evals/` - Evaluación
**Rol**: Métricas, benchmarking, evaluación de modelos.
**Uso en IA**: Evaluación de respuestas LLM, A/B testing, métricas.

### 26. `feature_flags/` - Feature Flags
**Rol**: Toggles, A/B testing, gradual rollout.
**Uso en IA**: Activar/desactivar modelos, features experimentales.

### 27. `secondary_llm_flows/` - Flujos Secundarios
**Rol**: Pipelines alternativos, procesamiento paralelo.
**Uso en IA**: Validación con segundo LLM, consensus, verification.

### 28. `seeding/` - Datos Iniciales
**Rol**: Seed data, inicialización de BD.
**Uso en IA**: Datos de ejemplo, prompts iniciales, configuraciones.

### 29. `server/` - Servidor
**Rol**: API REST, endpoints, FastAPI.
**Uso en IA**: Exponer funcionalidades de IA como API.

### 30. `onyxbot/` - Bot
**Rol**: Bot principal, integraciones.
**Uso en IA**: Interfaz conversacional, integración con Slack.

### 31. `tracing/` - Trazabilidad
**Rol**: Observabilidad, logging estructurado.
**Uso en IA**: Tracking completo del pipeline de IA.

## 📁 Estructura de Archivos por Módulo

Cada módulo debe tener:

```
module_name/
├── __init__.py          # Exports públicos, imports seguros
├── main.py              # Funciones base y entry points
├── base.py              # Clases base abstractas (si aplica)
├── service.py           # Servicio principal (si aplica)
├── repository.py        # Repositorio de datos (si aplica)
├── models.py            # Modelos de datos (si aplica)
└── [archivos específicos]
```

## 🔄 Flujo de Dependencias

```
configs/ (nivel 0 - base)
    ↓
utils/ (nivel 0 - base)
    ↓
db/, redis/, tracing/, httpx/ (nivel 1 - infraestructura)
    ↓
auth/, prompts/, llm/ (nivel 2 - servicios base)
    ↓
access/, tools/, nlp/ (nivel 3 - servicios especializados)
    ↓
chat/, agents/, context/ (nivel 4 - lógica de negocio)
    ↓
server/, onyxbot/ (nivel 5 - interfaz)
```

