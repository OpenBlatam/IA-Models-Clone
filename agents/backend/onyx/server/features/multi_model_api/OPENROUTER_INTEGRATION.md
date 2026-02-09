# Integración OpenRouter

## ¿Qué incluye OpenRouter?

OpenRouter es una plataforma unificada que proporciona acceso a **más de 1,000 modelos de IA** de múltiples proveedores a través de una única API compatible con OpenAI.

### Proveedores y Modelos Incluidos

#### 🤖 OpenAI
- GPT-4, GPT-4 Turbo
- GPT-3.5 Turbo
- GPT-4o, GPT-4o-mini
- Y más variantes

#### 🧠 Anthropic
- Claude 3 Opus
- Claude 3 Sonnet
- Claude 3 Haiku
- Claude 2.1, Claude 2.0
- Claude Instant

#### 🔍 Google
- Gemini Pro
- Gemini Pro Vision
- PaLM 2
- Y más modelos de Google

#### 🦙 Meta (Llama)
- Llama 3 70B
- Llama 3 8B
- Llama 2 70B
- Llama 2 13B
- Y más variantes

#### 🌊 Mistral AI
- Mistral Large
- Mixtral 8x7B
- Mistral 7B
- Y más modelos

#### 💬 Cohere
- Command R+
- Command R
- Command
- Y más variantes

#### 🔮 Perplexity
- Llama 3 Sonar Large 32K Online
- Llama 3 Sonar Small 32K Online
- Y más modelos

#### 🌏 Otros Proveedores
- **Qwen** (Alibaba): Qwen 2.5 72B, Qwen 2.5 32B, etc.
- **01.AI**: Yi 34B Chat, Yi 6B Chat, etc.
- **DeepSeek**: DeepSeek Coder, DeepSeek Chat
- **Nous Research**: Hermes, Capybara
- **Stability AI**: Stable Beluga, etc.
- **Together AI**: Varios modelos open source
- **Anyscale**: Modelos optimizados
- Y **muchos más**...

### Características Principales

#### 1. API Unificada
- Compatible con formato OpenAI
- Migración fácil desde OpenAI
- Mismo formato de requests/responses

#### 2. Enrutamiento Inteligente
- Selección automática del mejor modelo
- Fallback automático si un modelo falla
- Optimización de costos

#### 3. Análisis y Logging
- Logs detallados de todas las llamadas
- Tracking de tokens y costos
- Métricas de rendimiento
- Dashboard centralizado

#### 4. Optimización de Costos
- Comparación de precios entre modelos
- Enrutamiento a modelos más económicos
- Análisis de costos por modelo

#### 5. Streaming
- Soporte para respuestas en streaming (implementado)
- Ideal para chatbots en tiempo real
- Mejor experiencia de usuario
- Streaming SSE en tiempo real

#### 6. Mejoras Adicionales
- Cache de lista de modelos (1 hora)
- Manejo robusto de errores con tipos específicos
- Timeout handling mejorado
- Metadata adicional en respuestas
- Soporte para modelos dinámicos

### Modelos Populares Disponibles

```python
# Modelos OpenAI
"openai/gpt-4"
"openai/gpt-4-turbo"
"openai/gpt-3.5-turbo"

# Modelos Anthropic
"anthropic/claude-3-opus"
"anthropic/claude-3-sonnet"
"anthropic/claude-3-haiku"

# Modelos Google
"google/gemini-pro"
"google/gemini-pro-vision"

# Modelos Meta
"meta-llama/llama-3-70b-instruct"
"meta-llama/llama-3-8b-instruct"

# Modelos Mistral
"mistralai/mistral-large"
"mistralai/mixtral-8x7b-instruct"

# Modelos Cohere
"cohere/command-r-plus"
"cohere/command-r"

# Y muchos más...
```

### Uso en Multi-Model API

La integración de OpenRouter permite:

1. **Acceso a 1000+ modelos** sin necesidad de múltiples APIs
2. **Comparación fácil** entre modelos de diferentes proveedores
3. **Consenso multi-modelo** usando modelos de diferentes proveedores
4. **Fallback automático** si un modelo falla
5. **Optimización de costos** eligiendo modelos más económicos

### Configuración

```bash
# Variable de entorno requerida
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Ejemplo de Uso

#### Uso Básico

```python
from multi_model_api.integrations.openrouter import openrouter_handler

# Usar GPT-4 de OpenAI
result = await openrouter_handler(
    prompt="Explain quantum computing",
    model="openai/gpt-4",
    temperature=0.7
)

# Usar Claude 3 Opus de Anthropic
result = await openrouter_handler(
    prompt="Write a poem",
    model="anthropic/claude-3-opus",
    temperature=0.9
)

# Usar Llama 3 de Meta
result = await openrouter_handler(
    prompt="Code review",
    model="meta-llama/llama-3-70b-instruct",
    temperature=0.3
)
```

#### Streaming

```python
from multi_model_api.integrations.openrouter import get_openrouter_client

client = get_openrouter_client()

async for chunk in client.chat_completion_stream(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Explain AI"}],
    temperature=0.7
):
    if chunk.get("content"):
        print(chunk["content"], end="", flush=True)
```

#### Modelos Dinámicos

```json
{
  "prompt": "Your prompt",
  "models": [
    {
      "model_type": "openrouter/gpt-4",
      "is_enabled": true,
      "openrouter_model": "qwen/qwen-2.5-72b-instruct",
      "temperature": 0.7
    }
  ]
}
```

#### Listar Modelos Disponibles

```bash
# Listar todos los modelos
GET /multi-model/openrouter/models

# Filtrar por proveedor
GET /multi-model/openrouter/models?provider=openai

# Buscar modelos
GET /multi-model/openrouter/models?search=gpt-4
```

### Beneficios

✅ **Un solo API key** para acceder a 1000+ modelos
✅ **Sin necesidad de múltiples cuentas** de diferentes proveedores
✅ **Comparación fácil** de modelos
✅ **Optimización automática** de costos y rendimiento
✅ **Logging centralizado** de todas las llamadas
✅ **Fallback automático** para alta disponibilidad

### Recursos

- [Documentación OpenRouter](https://openrouter.ai/docs)
- [Lista completa de modelos](https://openrouter.ai/models)
- [Dashboard de uso](https://openrouter.ai/activity)

