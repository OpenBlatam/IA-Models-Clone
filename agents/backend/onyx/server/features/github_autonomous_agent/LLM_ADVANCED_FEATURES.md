# 🚀 LLM Service - Advanced Features

## Resumen de Funcionalidades Avanzadas

Este documento describe las funcionalidades avanzadas agregadas al servicio LLM, incluyendo A/B testing, webhooks, versionado de prompts, testing framework, semantic caching y rate limiting avanzado.

---

## 📊 1. A/B Testing Framework

### Descripción
Framework completo para realizar A/B testing de modelos LLM, comparando diferentes variantes (modelos, prompts, configuraciones) para determinar cuál funciona mejor.

### Características
- ✅ Comparación de múltiples variantes simultáneamente
- ✅ Métricas estadísticas automáticas
- ✅ Determinación automática de ganador
- ✅ Significancia estadística
- ✅ Persistencia de resultados

### Uso

```python
from core.services.llm import get_ab_testing_framework, Variant, VariantType

framework = get_ab_testing_framework()

# Crear variantes
variants = [
    Variant(
        name="gpt-4",
        variant_type=VariantType.MODEL,
        config={"model": "openai/gpt-4"}
    ),
    Variant(
        name="claude-3",
        variant_type=VariantType.MODEL,
        config={"model": "anthropic/claude-3-opus"}
    )
]

# Crear test
test_id = framework.create_test(
    name="Model Comparison",
    description="Comparar GPT-4 vs Claude-3",
    variants=variants,
    prompt="Analiza este código...",
    evaluation_criteria=["quality", "speed", "cost"]
)

# Registrar resultados
framework.record_result(
    test_id=test_id,
    variant_name="gpt-4",
    response="...",
    latency_ms=1200,
    tokens_used=500,
    cost=0.03,
    quality_score=0.95
)

# Obtener resumen
summary = framework.get_summary(test_id)
print(f"Ganador: {summary.winner}")
```

### API Endpoints

- `POST /api/v1/llm/ab-test/create` - Crear nuevo A/B test
- `GET /api/v1/llm/ab-test/{test_id}` - Obtener test
- `GET /api/v1/llm/ab-test` - Listar tests

---

## 🔔 2. Webhooks y Notificaciones

### Descripción
Sistema completo de webhooks para recibir notificaciones sobre eventos del servicio LLM (completado, errores, experimentos, etc.).

### Características
- ✅ Múltiples eventos configurables
- ✅ Firma de payloads con secret
- ✅ Retry automático con exponential backoff
- ✅ Timeout configurable
- ✅ Headers personalizables

### Eventos Disponibles

- `generation.completed` - Generación completada
- `generation.failed` - Generación fallida
- `experiment.completed` - Experimento completado
- `ab_test.completed` - A/B test completado
- `rate_limit.exceeded` - Rate limit excedido
- `cost.threshold_reached` - Umbral de costo alcanzado
- `cache.hit` - Cache hit
- `cache.miss` - Cache miss

### Uso

```python
from core.services.llm import get_webhook_service, WebhookEvent

service = get_webhook_service()

# Registrar webhook
webhook_id = service.register_webhook(
    url="https://example.com/webhook",
    events=[
        WebhookEvent.GENERATION_COMPLETED,
        WebhookEvent.GENERATION_FAILED
    ],
    secret="my-secret-key",
    timeout=5,
    retry_count=3
)

# Disparar webhook manualmente
result = await service.trigger_webhook(
    event=WebhookEvent.GENERATION_COMPLETED,
    data={"response": "...", "model": "gpt-4"}
)
```

### API Endpoints

- `POST /api/v1/llm/webhooks/register` - Registrar webhook
- `GET /api/v1/llm/webhooks` - Listar webhooks
- `POST /api/v1/llm/webhooks/{webhook_id}/test` - Probar webhook

---

## 📝 3. Prompt Versioning System

### Descripción
Sistema de versionado de prompts con soporte para comparación, rollback y gestión de versiones.

### Características
- ✅ Versionado semántico (major.minor.patch)
- ✅ Comparación de versiones
- ✅ Rollback a versiones anteriores
- ✅ Tags y metadata
- ✅ Búsqueda y filtrado
- ✅ Estados (draft, active, archived, deprecated)

### Uso

```python
from core.services.llm import get_prompt_versioning, PromptStatus

versioning = get_prompt_versioning()

# Crear prompt
prompt_id = versioning.create_prompt(
    name="Code Analysis Prompt",
    prompt="Analiza el siguiente código...",
    version="1.0.0",
    tags=["code", "analysis"]
)

# Agregar nueva versión
new_version = versioning.add_version(
    prompt_id=prompt_id,
    prompt="Analiza el siguiente código mejorado...",
    bump_type="minor"  # 1.1.0
)

# Activar versión
versioning.set_active_version(prompt_id, new_version)

# Comparar versiones
comparison = versioning.compare_versions(
    prompt_id=prompt_id,
    version1="1.0.0",
    version2="1.1.0"
)

# Rollback
versioning.rollback_version(prompt_id, "1.0.0")
```

### API Endpoints

- `POST /api/v1/llm/prompts/create` - Crear prompt
- `GET /api/v1/llm/prompts/{prompt_id}` - Obtener prompt
- `GET /api/v1/llm/prompts` - Listar prompts

---

## 🧪 4. LLM Testing Framework

### Descripción
Framework completo para crear y ejecutar tests automatizados para evaluar la calidad y consistencia de respuestas de modelos LLM.

### Características
- ✅ Múltiples tipos de tests (functional, quality, consistency, performance, safety)
- ✅ Aserciones personalizables
- ✅ Ejecución en paralelo
- ✅ Reportes detallados
- ✅ Integración con CI/CD

### Tipos de Aserciones

- `contains` - Verificar que contiene texto
- `not_contains` - Verificar que no contiene texto
- `equals` - Verificar igualdad exacta
- `regex` - Verificar patrón regex
- `length` - Verificar longitud
- `custom` - Función personalizada

### Uso

```python
from core.services.llm import (
    get_llm_testing_framework,
    TestCase,
    TestAssertion,
    AssertionType,
    TestType
)

framework = get_llm_testing_framework()

# Crear test cases
test_cases = [
    TestCase(
        case_id="test-1",
        name="Code Analysis Test",
        prompt="Analiza este código: def hello(): print('world')",
        assertions=[
            TestAssertion(
                name="Contains function",
                assertion_type=AssertionType.CONTAINS,
                expected="def hello"
            ),
            TestAssertion(
                name="Length check",
                assertion_type=AssertionType.LENGTH,
                expected={"min": 50, "max": 500}
            )
        ]
    )
]

# Crear suite
suite_id = framework.create_test_suite(
    name="Code Analysis Suite",
    description="Tests para análisis de código",
    test_type=TestType.FUNCTIONAL,
    test_cases=test_cases,
    model="gpt-4"
)

# Ejecutar suite
result = await framework.run_test_suite(suite_id, llm_service)

print(f"Tests pasados: {result.passed_tests}/{result.total_tests}")
```

### API Endpoints

- `POST /api/v1/llm/tests/create-suite` - Crear suite de tests
- `POST /api/v1/llm/tests/{suite_id}/run` - Ejecutar suite
- `GET /api/v1/llm/tests/{suite_id}/results` - Obtener resultados

---

## 🧠 5. Semantic Caching

### Descripción
Cache inteligente basado en embeddings para encontrar respuestas similares incluso cuando el prompt no es idéntico.

### Características
- ✅ Búsqueda por similitud semántica
- ✅ TTL configurable
- ✅ LRU eviction
- ✅ Metadata personalizable
- ✅ Persistencia en disco

### Uso

```python
from core.services.llm import get_semantic_cache

cache = get_semantic_cache(
    similarity_threshold=0.85,
    max_size=1000,
    ttl_seconds=3600
)

# Guardar respuesta
key = cache.set(
    prompt="¿Qué es Python?",
    response="Python es un lenguaje de programación...",
    metadata={"model": "gpt-4", "language": "es"}
)

# Buscar respuesta similar
result = cache.get("¿Qué es el lenguaje Python?")
if result:
    response, similarity = result
    print(f"Cache hit! Similitud: {similarity:.2f}")

# Estadísticas
stats = cache.get_stats()
print(f"Items en cache: {stats['total_items']}")
```

---

## 🚦 6. Advanced Rate Limiting

### Descripción
Rate limiting sofisticado con múltiples estrategias y soporte para diferentes tipos de límites.

### Estrategias Disponibles

1. **Fixed Window** - Ventana fija de tiempo
2. **Sliding Window** - Ventana deslizante precisa
3. **Token Bucket** - Bucket de tokens con refill
4. **Leaky Bucket** - Bucket con fuga constante

### Características
- ✅ Múltiples estrategias
- ✅ Rate limiting por clave (usuario, modelo, IP, etc.)
- ✅ Burst handling
- ✅ Estadísticas detalladas

### Uso

```python
from core.services.llm import (
    get_advanced_rate_limiter,
    RateLimitStrategy
)

limiter = get_advanced_rate_limiter()

# Configurar rate limit
limiter.configure(
    key="user-123",
    limit=100,
    window_seconds=60,
    strategy=RateLimitStrategy.SLIDING_WINDOW
)

# Verificar si está permitido
info = limiter.is_allowed("user-123", tokens=1)

if info.allowed:
    print(f"Permitido. Quedan {info.remaining} requests")
else:
    print(f"Bloqueado. Retry después de {info.retry_after:.1f}s")

# Estadísticas
stats = limiter.get_stats("user-123")
```

---

## 📊 7. Dashboard & Analytics

### Endpoints de Dashboard

- `GET /api/v1/llm/dashboard/stats` - Estadísticas completas
- `GET /api/v1/llm/dashboard/analytics` - Analytics detallados

### Respuesta de Dashboard Stats

```json
{
  "llm_stats": {
    "total_requests": 1000,
    "successful_requests": 950,
    "cache_hit_rate": 0.25,
    "average_latency_ms": 1200
  },
  "cost_stats": {
    "total_cost": 50.25,
    "cost_by_model": {...}
  },
  "performance": {
    "avg_latency": 1200,
    "p95_latency": 2500
  },
  "cache_stats": {
    "total_items": 500,
    "usage_percent": 50
  },
  "rate_limit_stats": {
    "user-123": {
      "total_requests": 100,
      "blocked_requests": 5
    }
  }
}
```

---

## 🔧 Integración con LLMService

Todos estos componentes pueden integrarse con el `LLMService` principal:

```python
from core.services.llm_service import LLMService
from core.services.llm import (
    get_ab_testing_framework,
    get_webhook_service,
    get_prompt_versioning
)

llm_service = LLMService(...)

# Usar prompt versioning
versioning = get_prompt_versioning()
prompt = versioning.get_prompt("my-prompt-id")
response = await llm_service.generate(
    prompt=prompt.prompt,
    system_prompt=prompt.system_prompt
)

# Disparar webhook
webhook_service = get_webhook_service()
await webhook_service.trigger_webhook(
    WebhookEvent.GENERATION_COMPLETED,
    data={"response": response.content}
)
```

---

## 📦 Dependencias Adicionales

Para usar todas las funcionalidades, instala:

```bash
pip install sentence-transformers numpy
```

- `sentence-transformers` - Para semantic caching (embeddings)
- `numpy` - Para operaciones vectoriales

---

## 🎯 Casos de Uso

### 1. Optimización de Prompts
- Usar **Prompt Versioning** para probar diferentes versiones
- Usar **A/B Testing** para comparar efectividad
- Usar **Testing Framework** para validar calidad

### 2. Monitoreo y Alertas
- Configurar **Webhooks** para notificaciones
- Usar **Dashboard** para monitoreo en tiempo real
- Configurar **Rate Limiting** para control de uso

### 3. Optimización de Costos
- Usar **Semantic Caching** para reducir llamadas
- Usar **Cost Optimizer** (componente existente)
- Monitorear con **Dashboard Analytics**

### 4. Testing y QA
- Crear **Test Suites** para validación continua
- Integrar con CI/CD
- Usar **A/B Testing** para comparar modelos

---

## 📚 Documentación Adicional

- [LLM Service Documentation](LLM_SERVICE.md)
- [LLM Architecture](LLM_ARCHITECTURE.md)
- [LLM Components Summary](LLM_COMPONENTS_SUMMARY.md)

---

## 🚀 Próximos Pasos

1. Configurar webhooks para notificaciones
2. Crear test suites para validación
3. Configurar rate limiting por usuario/modelo
4. Habilitar semantic caching para reducir costos
5. Configurar A/B tests para optimización continua



