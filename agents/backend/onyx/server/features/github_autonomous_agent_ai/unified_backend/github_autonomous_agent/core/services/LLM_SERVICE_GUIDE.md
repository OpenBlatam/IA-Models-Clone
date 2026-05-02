# LLM Service - Guía de Uso

## 📋 Resumen

El `LLMService` proporciona una interfaz unificada para interactuar con modelos de lenguaje grandes (LLMs) a través de OpenRouter, soportando múltiples proveedores, caché, métricas, retry automático y más.

## 🚀 Características

- ✅ **Múltiples Modelos**: Soporte para modelos de OpenAI, Anthropic, Google, y más
- ✅ **Ejecución Paralela**: Generar respuestas de múltiples modelos simultáneamente
- ✅ **Caché Inteligente**: Cachear respuestas para reducir costos y latencia
- ✅ **Métricas Integradas**: Tracking automático de uso, latencia y errores
- ✅ **Retry Automático**: Reintentos con exponential backoff
- ✅ **Circuit Breaker**: Protección contra fallos en cascada
- ✅ **Streaming**: Soporte para respuestas en streaming
- ✅ **Análisis de Código**: Funciones especializadas para análisis de código
- ✅ **Generación de Instrucciones**: Convertir lenguaje natural en instrucciones estructuradas

## 📦 Instalación

El servicio usa `httpx` que ya está en `requirements.txt`. Solo necesitas configurar tu API key de OpenRouter:

```bash
export OPENROUTER_API_KEY="tu-api-key-aqui"
```

O en tu archivo `.env`:
```
OPENROUTER_API_KEY=tu-api-key-aqui
```

## 🔧 Uso Básico

### Inicialización

```python
from core.services import LLMService, CacheService, MetricsService

# Opcional: Inicializar servicios de soporte
cache = CacheService(max_size=1000, default_ttl=3600)
metrics = MetricsService(use_prometheus=True)

# Inicializar LLM Service
llm = LLMService(
    api_key="tu-api-key",  # Opcional si está en settings
    default_models=["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"],
    cache_service=cache,  # Opcional
    metrics_service=metrics,  # Opcional
    enable_caching=True,
    cache_ttl=3600  # 1 hora
)
```

### Generar Respuesta Simple

```python
response = await llm.generate(
    prompt="Explica qué es Python en una oración",
    model="openai/gpt-4o-mini"
)

print(response.content)
print(f"Latencia: {response.latency_ms}ms")
print(f"Tokens usados: {response.usage}")
```

### Generar con System Prompt

```python
response = await llm.generate(
    prompt="¿Cuál es la mejor práctica para manejar errores en Python?",
    system_prompt="Eres un experto en Python con 20 años de experiencia.",
    model="anthropic/claude-3.5-sonnet",
    temperature=0.7
)
```

### Generar en Paralelo

```python
responses = await llm.generate_parallel(
    prompt="Explica qué es async/await en Python",
    models=["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5"]
)

for model, response in responses.items():
    print(f"\n{model}:")
    print(response.content)
```

### Streaming

```python
async for chunk in llm.generate_stream(
    prompt="Escribe un poema sobre Python",
    model="openai/gpt-4o-mini"
):
    print(chunk, end="", flush=True)
```

## 🔍 Análisis de Código

### Análisis General

```python
code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""

response = await llm.analyze_code(
    code=code,
    language="python",
    analysis_type="general"
)

print(response.content)
```

### Tipos de Análisis

- `"general"`: Análisis general de calidad y estructura
- `"bugs"`: Detección de bugs y errores
- `"performance"`: Optimizaciones de rendimiento
- `"security"`: Vulnerabilidades de seguridad
- `"style"`: Estilo y legibilidad

```python
# Análisis de seguridad
security_analysis = await llm.analyze_code(
    code=code,
    language="python",
    analysis_type="security"
)

# Análisis de performance
perf_analysis = await llm.analyze_code(
    code=code,
    language="python",
    analysis_type="performance"
)
```

## 📝 Generación de Instrucciones

Convertir descripciones en lenguaje natural a instrucciones estructuradas:

```python
response = await llm.generate_instruction(
    description="Crear un archivo README.md con información del proyecto",
    context="Proyecto: GitHub Autonomous Agent\nLenguaje: Python"
)

print(response.content)
# Output: "create file: README.md"
```

## ⚙️ Configuración Avanzada

### Caché Personalizado

```python
# Deshabilitar caché para un request específico
response = await llm.generate(
    prompt="...",
    cache_enabled=False
)

# Usar TTL personalizado
response = await llm.generate(
    prompt="...",
    cache_ttl=7200  # 2 horas
)
```

### Parámetros del Modelo

```python
response = await llm.generate(
    prompt="...",
    temperature=0.9,  # Más creativo
    max_tokens=500,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.3
)
```

## 📊 Métricas y Estadísticas

### Obtener Estadísticas

```python
stats = llm.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Average latency: {stats['average_latency_ms']}ms")
print(f"Total tokens: {stats['total_tokens']}")
```

### Resetear Estadísticas

```python
llm.reset_stats()
```

## 🔄 Integración con Otros Servicios

### Con CacheService

```python
from core.services import CacheService, LLMService

cache = CacheService(max_size=1000, default_ttl=3600)
llm = LLMService(cache_service=cache, enable_caching=True)

# Las respuestas se cachean automáticamente
response1 = await llm.generate(prompt="...")  # Request a API
response2 = await llm.generate(prompt="...")  # Desde caché
```

### Con MetricsService

```python
from core.services import MetricsService, LLMService

metrics = MetricsService(use_prometheus=True)
llm = LLMService(metrics_service=metrics)

# Las métricas se registran automáticamente
response = await llm.generate(prompt="...")

# Ver métricas en Prometheus
# curl http://localhost:9090/metrics
```

## 🛡️ Manejo de Errores

El servicio maneja errores automáticamente:

```python
response = await llm.generate(prompt="...")

if response.error:
    print(f"Error: {response.error}")
else:
    print(response.content)
```

### Tipos de Errores

- **Rate Limit**: Demasiados requests
- **Circuit Breaker**: Servicio temporalmente no disponible
- **Validation Error**: Request inválido
- **Network Error**: Error de conexión
- **API Error**: Error de la API de OpenRouter

## 🎯 Casos de Uso

### 1. Análisis Automático de Pull Requests

```python
async def analyze_pr(code_diff: str):
    response = await llm.analyze_code(
        code=code_diff,
        analysis_type="general"
    )
    return response.content
```

### 2. Generación de Instrucciones desde Descripciones

```python
async def process_user_request(description: str, repo_context: str):
    response = await llm.generate_instruction(
        description=description,
        context=repo_context
    )
    return response.content
```

### 3. Comparación de Modelos

```python
async def compare_models(prompt: str):
    responses = await llm.generate_parallel(
        prompt=prompt,
        models=["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]
    )
    
    for model, response in responses.items():
        print(f"{model}: {len(response.content)} caracteres")
```

## 📚 Modelos Disponibles

Obtener lista de modelos disponibles:

```python
models = await llm.get_available_models()
for model in models[:10]:  # Primeros 10
    print(f"{model['id']}: {model.get('name', 'N/A')}")
```

## 🔐 Seguridad

- ✅ API keys se manejan de forma segura
- ✅ No se almacenan prompts sensibles en logs
- ✅ Caché puede ser limpiado para datos sensibles

```python
# Limpiar caché de un prompt específico
cache_key = llm._generate_cache_key(...)
llm.cache_service.delete(cache_key)
```

## 🚀 Mejores Prácticas

1. **Usa Caché**: Para prompts repetitivos, habilita caché
2. **Métricas**: Integra MetricsService para monitoreo
3. **Error Handling**: Siempre verifica `response.error`
4. **Rate Limiting**: Respeta los límites de la API
5. **Modelos Apropiados**: Usa modelos más pequeños para tareas simples
6. **Temperature**: Usa temperatura baja (0.3) para análisis, alta (0.9) para creatividad

## 📖 Referencias

- [OpenRouter API Docs](https://openrouter.ai/docs)
- [OpenRouter Models](https://openrouter.ai/models)
- [Service Layer Pattern](./ARCHITECTURE_IMPROVEMENTS_V9.md)

---

**Versión**: 1.0  
**Última actualización**: Enero 2025



