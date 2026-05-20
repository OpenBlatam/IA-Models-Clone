# Mapa de Dependencias entre Módulos

Este documento describe las dependencias entre los módulos de la arquitectura modular.

## Visualización de Dependencias

```
utils (módulo base, sin dependencias)
  │
  ├── configs
  │   ├── db
  │   ├── redis
  │   ├── httpx
  │   ├── file_store
  │   └── tracing
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
  └── server (punto de entrada)
      └── (depende de múltiples módulos)
```

## Dependencias Detalladas

### Módulos Base (Sin Dependencias)
- **utils**: Utilidades generales, sin dependencias

### Módulos de Infraestructura
- **configs** → `utils`
- **db** → `configs`, `utils`
- **redis** → `configs`, `utils`
- **httpx** → `configs`, `tracing`, `utils`
- **tracing** → `configs`, `utils`
- **file_store** → `configs`, `utils`

### Módulos de Servicios Core
- **auth** → `db`, `redis`, `utils`
- **access** → `auth`, `db`, `utils`
- **llm** → `httpx`, `prompts`, `redis`, `tracing`, `configs`
- **prompts** → `db`, `llm`, `utils`
- **chat** → `llm`, `db`, `context/search`, `prompts`, `tracing`
- **agents** → `llm`, `tools`, `tracing`, `prompts`, `context/search`

### Módulos de Procesamiento
- **file_processing** → `file_store`, `utils`, `tracing`
- **document_index** → `db`, `file_store`, `indexing`, `llm`
- **indexing** → `db`, `document_index`, `utils`
- **context/search** → `document_index`, `kg`, `llm`, `db`
- **natural_language_processing** → `llm`, `utils`
- **kg** → `db`, `llm`, `natural_language_processing`, `indexing`

### Módulos de Integración
- **connectors** → `httpx`, `utils`, `tracing`
- **federated_connectors** → `connectors`, `httpx`, `db`, `tracing`
- **tools** → `httpx`, `db`, `utils`, `tracing`
- **onyxbot/slack** → `agents`, `chat`, `auth`, `httpx`, `tracing`

### Módulos de Flujos Especializados
- **secondary_llm_flows** → `llm`, `prompts`, `tracing`, `utils`
- **background** → `redis`, `db`, `tracing`, `utils`
- **evals** → `llm`, `db`, `tracing`, `utils`

### Módulos de Configuración y Datos
- **feature_flags** → `db`, `redis`, `configs`
- **key_value_store** → `redis`, `configs`, `utils`
- **seeding** → `db`, `configs`, `utils`

### Módulo de Entrada
- **server** → `auth`, `access`, `chat`, `agents`, `tracing`, `configs`

## Reglas de Dependencias

1. **No hay dependencias circulares**: Las dependencias fluyen en una dirección clara.
2. **Módulos base no dependen de otros módulos**: `utils` es completamente independiente.
3. **Módulos de infraestructura dependen solo de `utils` y `configs`**.
4. **Módulos de servicios dependen de infraestructura y otros servicios**.
5. **El módulo `server` es el punto de entrada y depende de múltiples módulos**.

## Orden de Inicialización

Para inicializar correctamente el sistema, los módulos deben inicializarse en este orden:

1. `utils`
2. `configs`
3. `db`, `redis`, `httpx`, `tracing`, `file_store`
4. `auth`, `llm`, `prompts`
5. `access`, `chat`, `agents`, `tools`
6. `file_processing`, `document_index`, `indexing`
7. `context/search`, `natural_language_processing`, `kg`
8. `connectors`, `federated_connectors`
9. `secondary_llm_flows`, `background`, `evals`
10. `feature_flags`, `key_value_store`, `seeding`
11. `onyxbot/slack`
12. `server`

