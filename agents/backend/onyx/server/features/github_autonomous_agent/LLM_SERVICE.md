# 🤖 Servicio LLM con OpenRouter

Este documento describe cómo usar el servicio LLM integrado que permite acceder a múltiples modelos de IA a través de OpenRouter y ejecutarlos en paralelo.

## 📋 Características

- ✅ **Soporte para OpenRouter**: Acceso a múltiples modelos de IA de diferentes proveedores
- ✅ **Ejecución Paralela**: Ejecuta múltiples modelos simultáneamente para comparar respuestas
- ✅ **Configuración Flexible**: Configura qué modelos usar y sus parámetros
- ✅ **Integración Automática**: Se integra automáticamente con el TaskProcessor para instrucciones que requieren IA

## ⚙️ Configuración

### Variables de Entorno

Agrega las siguientes variables a tu archivo `.env`:

```bash
# OpenRouter API Key (obtén una en https://openrouter.ai)
OPENROUTER_API_KEY=sk-or-v1-...

# Opcional: HTTP Referer para OpenRouter
OPENROUTER_HTTP_REFERER=https://tu-dominio.com

# Opcional: Título de la aplicación
OPENROUTER_X_TITLE=GitHub Autonomous Agent

# Modelos por defecto (separados por coma si usas formato de lista)
# O define como lista en Python: ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]
LLM_DEFAULT_MODELS=openai/gpt-4o-mini,anthropic/claude-3.5-sonnet,google/gemini-pro-1.5

# Timeout para requests LLM (segundos)
LLM_TIMEOUT=60

# Máximo de requests paralelos
LLM_MAX_PARALLEL_REQUESTS=10

# Habilitar/deshabilitar servicio LLM
LLM_ENABLED=true
```

### Modelos Disponibles

OpenRouter soporta cientos de modelos. Algunos populares:

- `openai/gpt-4o-mini` - GPT-4o Mini (rápido y económico)
- `openai/gpt-4o` - GPT-4o (más potente)
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet
- `anthropic/claude-3-opus` - Claude 3 Opus (más potente)
- `google/gemini-pro-1.5` - Google Gemini Pro 1.5
- `meta-llama/llama-3.1-70b-instruct` - Llama 3.1 70B
- `mistralai/mistral-large` - Mistral Large

Ver todos los modelos disponibles: https://openrouter.ai/models

## 🚀 Uso

### Uso Automático en TaskProcessor

El `TaskProcessor` detecta automáticamente cuando una instrucción requiere procesamiento con IA y usa múltiples modelos en paralelo:

```python
# Ejemplo de instrucción que activará LLM
instruction = "Analizar el código del repositorio y sugerir mejoras de rendimiento"
```

Palabras clave que activan LLM:
- analizar, analyze
- generar, generate
- sugerir, suggest
- revisar, review
- mejorar, improve
- optimizar, optimize
- explicar, explain
- documentar, document
- refactorizar, refactor
- crear código, create code
- escribir código, write code

### Uso Directo vía API

#### 1. Generar respuesta de un modelo

```bash
curl -X POST http://localhost:8030/api/v1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explica qué es Python en una frase",
    "model": "openai/gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

#### 2. Generar respuestas de múltiples modelos en paralelo

```bash
curl -X POST http://localhost:8030/api/v1/llm/generate-parallel \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Escribe una función Python para calcular el factorial",
    "models": [
      "openai/gpt-4o-mini",
      "anthropic/claude-3.5-sonnet",
      "google/gemini-pro-1.5"
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

#### 3. Obtener modelos disponibles

```bash
curl http://localhost:8030/api/v1/llm/models
```

#### 4. Verificar estado del servicio

```bash
curl http://localhost:8030/api/v1/llm/status
```

### Uso en Código Python

```python
from config.di_setup import get_service
from core.services.llm_service import LLMService

# Obtener servicio LLM
llm_service = get_service("llm_service")

# Generar respuesta de un modelo
response = await llm_service.generate(
    prompt="Explica qué es FastAPI",
    model="openai/gpt-4o-mini",
    temperature=0.7
)

print(response.content)

# Generar respuestas de múltiples modelos en paralelo
responses = await llm_service.generate_parallel(
    prompt="Escribe una función para validar email",
    models=["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"],
    temperature=0.7
)

for model, response in responses.items():
    print(f"\n{model}:")
    print(response.content)
```

## 📊 Respuestas Paralelas

Cuando usas `generate_parallel`, obtienes respuestas de múltiples modelos simultáneamente. Esto es útil para:

- **Comparar respuestas**: Ver cómo diferentes modelos abordan el mismo problema
- **Validación cruzada**: Verificar consistencia entre modelos
- **Redundancia**: Si un modelo falla, tienes respuestas de otros
- **Análisis de calidad**: Comparar calidad y completitud de respuestas

### Ejemplo de Respuesta Paralela

```json
{
  "success": true,
  "responses": {
    "openai/gpt-4o-mini": {
      "model": "openai/gpt-4o-mini",
      "content": "Python es un lenguaje de programación...",
      "latency_ms": 1234.5,
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 50,
        "total_tokens": 60
      }
    },
    "anthropic/claude-3.5-sonnet": {
      "model": "anthropic/claude-3.5-sonnet",
      "content": "Python es un lenguaje de alto nivel...",
      "latency_ms": 1567.8,
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 55,
        "total_tokens": 65
      }
    }
  },
  "total_models": 2,
  "successful_models": 2
}
```

## 🔧 Configuración Avanzada

### Personalizar Modelos por Tarea

Puedes especificar modelos diferentes para diferentes tipos de tareas:

```python
# Para análisis de código, usa modelos más potentes
analysis_models = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet"
]

# Para generación rápida, usa modelos más rápidos
quick_models = [
    "openai/gpt-4o-mini",
    "google/gemini-pro-1.5"
]

responses = await llm_service.generate_parallel(
    prompt=instruction,
    models=analysis_models
)
```

### Ajustar Parámetros

```python
# Respuestas más creativas
response = await llm_service.generate(
    prompt="Escribe un poema sobre Python",
    temperature=1.2,  # Más creativo
    max_tokens=500
)

# Respuestas más determinísticas
response = await llm_service.generate(
    prompt="Explica el algoritmo de ordenamiento",
    temperature=0.3,  # Más determinístico
    max_tokens=1000
)
```

## 🐛 Troubleshooting

### Error: "LLM service no disponible"

**Causa**: OpenRouter API key no configurada o servicio deshabilitado.

**Solución**:
1. Verifica que `OPENROUTER_API_KEY` esté en tu `.env`
2. Verifica que `LLM_ENABLED=true`
3. Reinicia el servidor

### Error: "Error HTTP 401"

**Causa**: API key inválida o expirada.

**Solución**:
1. Verifica tu API key en https://openrouter.ai/keys
2. Genera una nueva si es necesario

### Error: "Error HTTP 429"

**Causa**: Rate limit excedido.

**Solución**:
1. Reduce `LLM_MAX_PARALLEL_REQUESTS`
2. Espera unos minutos antes de reintentar
3. Considera actualizar tu plan de OpenRouter

### Modelo no disponible

**Causa**: El modelo especificado no existe o no está disponible.

**Solución**:
1. Verifica modelos disponibles: `GET /api/v1/llm/models`
2. Usa un modelo alternativo de la lista

## 📚 Recursos

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Models](https://openrouter.ai/models)
- [OpenRouter Pricing](https://openrouter.ai/docs/pricing)

## 💡 Mejores Prácticas

1. **Usa modelos apropiados**: Para tareas simples, usa modelos más rápidos y económicos
2. **Paralelismo moderado**: No excedas 5-10 modelos en paralelo a menos que sea necesario
3. **Manejo de errores**: Siempre verifica `response.error` antes de usar el contenido
4. **Caché cuando sea posible**: Para prompts repetitivos, considera usar caché
5. **Monitorea uso**: Revisa `usage` para controlar costos

## 🎯 Ejemplos de Uso

### Análisis de Código

```python
instruction = """
Analiza el siguiente código Python y sugiere mejoras:

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
"""

responses = await llm_service.generate_parallel(
    prompt=instruction,
    models=["openai/gpt-4o", "anthropic/claude-3.5-sonnet"]
)
```

### Generación de Documentación

```python
instruction = """
Genera documentación para esta función:

def validate_email(email: str) -> bool:
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
"""

response = await llm_service.generate(
    prompt=instruction,
    model="anthropic/claude-3.5-sonnet",
    temperature=0.5
)
```

### Refactorización de Código

```python
instruction = """
Refactoriza este código para hacerlo más eficiente y legible:

[tu código aquí]
"""

responses = await llm_service.generate_parallel(
    prompt=instruction,
    models=["openai/gpt-4o", "google/gemini-pro-1.5"]
)
```

---

**¿Necesitas ayuda?** Consulta la documentación completa o abre un issue.



