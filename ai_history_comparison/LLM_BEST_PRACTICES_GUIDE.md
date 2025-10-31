# 🤖 Mejores Prácticas para LLMs - Guía Completa

## 📋 **Índice**
1. [Configuración y Gestión de Modelos](#configuración-y-gestión-de-modelos)
2. [Prompt Engineering](#prompt-engineering)
3. [Caching y Optimización](#caching-y-optimización)
4. [Manejo de Errores y Retry](#manejo-de-errores-y-retry)
5. [Procesamiento Asíncrono](#procesamiento-asíncrono)
6. [Monitoreo y Métricas](#monitoreo-y-métricas)
7. [Seguridad y Privacidad](#seguridad-y-privacidad)
8. [Costos y Eficiencia](#costos-y-eficiencia)

---

## 🏗️ **Configuración y Gestión de Modelos**

### **1. Configuración Centralizada**
```python
@dataclass
class LLMConfig:
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    top_p: float = 1.0
    timeout: int = 30
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 3600
```

### **2. Gestión de Múltiples Proveedores**
```python
class LLMManager:
    def __init__(self):
        self.configs: Dict[str, LLMConfig] = {}
        self.clients: Dict[str, Any] = {}
    
    def register_model(self, name: str, config: LLMConfig):
        """Registrar un nuevo modelo"""
        self.configs[name] = config
        self._initialize_client(name, config)
```

### **3. Configuración por Entorno**
```python
# Desarrollo
DEV_CONFIG = LLMConfig(
    provider=LLMProvider.OPENAI,
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=2000
)

# Producción
PROD_CONFIG = LLMConfig(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4",
    temperature=0.3,  # Más determinístico
    max_tokens=4000,
    retry_attempts=5
)
```

---

## ✍️ **Prompt Engineering**

### **1. Templates Reutilizables**
```python
class PromptTemplate:
    @staticmethod
    def content_analysis_prompt(content: str, analysis_type: str = "comprehensive") -> str:
        return f"""
You are an expert content analyst. Analyze the following content and provide a comprehensive assessment.

Content to analyze:
"{content}"

Please provide analysis in the following JSON format:
{{
    "readability_score": 0.0-1.0,
    "sentiment_score": -1.0 to 1.0,
    "complexity_score": 0.0-1.0,
    "key_themes": ["theme1", "theme2"],
    "overall_quality": 0.0-1.0,
    "confidence": 0.0-1.0
}}

Analysis type: {analysis_type}
"""
```

### **2. Few-Shot Learning**
```python
def few_shot_prompt(examples: List[Dict], new_content: str) -> str:
    """Crear prompt con ejemplos"""
    examples_text = "\n".join([
        f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
        for i, ex in enumerate(examples)
    ])
    
    return f"""
You are an expert content analyst. Here are some examples:

{examples_text}

Now analyze this new content:
"{new_content}"

Provide your analysis in the same format as the examples.
"""
```

### **3. Chain of Thought**
```python
def chain_of_thought_prompt(content: str) -> str:
    """Prompt que guía el razonamiento paso a paso"""
    return f"""
Analyze the following content step by step:

Content: "{content}"

Step 1: Identify the main topics and themes
Step 2: Assess the readability and complexity
Step 3: Evaluate the sentiment and tone
Step 4: Check for consistency and coherence
Step 5: Provide overall quality assessment

For each step, explain your reasoning before giving the final score.
"""
```

### **4. Prompt Validation**
```python
def validate_prompt(prompt: str) -> Dict[str, Any]:
    """Validar prompt antes de enviarlo"""
    validation = {
        "length": len(prompt),
        "has_instructions": "analyze" in prompt.lower(),
        "has_format": "json" in prompt.lower(),
        "has_examples": "example" in prompt.lower(),
        "is_valid": True
    }
    
    if validation["length"] > 8000:  # Límite de tokens
        validation["is_valid"] = False
        validation["error"] = "Prompt too long"
    
    return validation
```

---

## 🚀 **Caching y Optimización**

### **1. Caché Inteligente**
```python
class LLMCache:
    def _generate_cache_key(self, prompt: str, model_name: str, config: Dict) -> str:
        """Generar clave de caché única"""
        content = f"{prompt}:{model_name}:{json.dumps(config, sort_keys=True)}"
        return f"llm_cache:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get(self, prompt: str, model_name: str, config: Dict) -> Optional[Dict]:
        """Obtener respuesta del caché"""
        cache_key = self._generate_cache_key(prompt, model_name, config)
        cached_data = self.redis.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        return None
```

### **2. Caché por Similitud**
```python
def semantic_cache_key(prompt: str, threshold: float = 0.9) -> str:
    """Generar clave de caché basada en similitud semántica"""
    # Usar embeddings para encontrar prompts similares
    embedding = get_embedding(prompt)
    similar_prompts = find_similar_embeddings(embedding, threshold)
    
    if similar_prompts:
        return f"semantic_cache:{similar_prompts[0]['key']}"
    
    return f"semantic_cache:{hashlib.md5(prompt.encode()).hexdigest()}"
```

### **3. Optimización de Tokens**
```python
def optimize_prompt(prompt: str, max_tokens: int = 4000) -> str:
    """Optimizar prompt para reducir tokens"""
    # Remover espacios innecesarios
    prompt = re.sub(r'\s+', ' ', prompt.strip())
    
    # Acortar instrucciones repetitivas
    prompt = re.sub(r'Please provide.*?format:', 'Format:', prompt)
    
    # Verificar longitud
    estimated_tokens = len(prompt.split()) * 1.3  # Aproximación
    
    if estimated_tokens > max_tokens:
        # Truncar manteniendo la estructura
        prompt = truncate_prompt_intelligently(prompt, max_tokens)
    
    return prompt
```

---

## 🔄 **Manejo de Errores y Retry**

### **1. Retry con Backoff Exponencial**
```python
class RetryManager:
    async def execute_with_retry(self, func, *args, **kwargs):
        """Ejecutar función con retry inteligente"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                # Backoff exponencial
                delay = self.base_delay * (self.backoff_factor ** attempt)
                await asyncio.sleep(delay)
        
        raise LLMError(f"Max retries exceeded: {str(last_exception)}")
```

### **2. Manejo de Errores Específicos**
```python
async def handle_llm_error(error: Exception) -> Dict[str, Any]:
    """Manejar errores específicos de LLM"""
    if isinstance(error, openai.RateLimitError):
        return {
            "error_type": "rate_limit",
            "retry_after": error.retry_after,
            "message": "Rate limit exceeded, retrying later"
        }
    elif isinstance(error, openai.APIError):
        return {
            "error_type": "api_error",
            "status_code": error.status_code,
            "message": f"API error: {error.message}"
        }
    elif isinstance(error, openai.Timeout):
        return {
            "error_type": "timeout",
            "message": "Request timed out"
        }
    else:
        return {
            "error_type": "unknown",
            "message": str(error)
        }
```

### **3. Circuit Breaker**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Ejecutar función con circuit breaker"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise LLMError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
```

---

## ⚡ **Procesamiento Asíncrono**

### **1. Procesamiento Concurrente**
```python
async def process_batch(prompts: List[str], model_name: str, 
                      max_concurrent: int = 5) -> List[Dict]:
    """Procesar múltiples prompts concurrentemente"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(prompt: str):
        async with semaphore:
            return await process_single(prompt, model_name)
    
    tasks = [process_with_semaphore(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results
```

### **2. Streaming de Respuestas**
```python
async def stream_response(prompt: str, model_name: str) -> AsyncGenerator[str, None]:
    """Streaming de respuesta en tiempo real"""
    client = get_client(model_name)
    
    async for chunk in client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### **3. Rate Limiting**
```python
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = []
    
    async def acquire(self):
        """Adquirir permiso para hacer request"""
        now = time.time()
        
        # Limpiar requests antiguos
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < 60]
        
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            await asyncio.sleep(sleep_time)
        
        self.requests.append(now)
```

---

## 📊 **Monitoreo y Métricas**

### **1. Métricas de Uso**
```python
class LLMMonitor:
    async def log_usage(self, model_name: str, tokens_used: int, 
                       processing_time: float, success: bool):
        """Registrar métricas de uso"""
        timestamp = datetime.now(timezone.utc)
        date_key = timestamp.strftime("%Y-%m-%d")
        
        usage_key = f"llm_usage:{date_key}:{model_name}"
        self.redis.hincrby(usage_key, "total_calls", 1)
        self.redis.hincrby(usage_key, "total_tokens", tokens_used)
        self.redis.hincrbyfloat(usage_key, "total_time", processing_time)
        
        if success:
            self.redis.hincrby(usage_key, "successful_calls", 1)
        else:
            self.redis.hincrby(usage_key, "failed_calls", 1)
```

### **2. Alertas y Notificaciones**
```python
class AlertManager:
    def __init__(self):
        self.thresholds = {
            "error_rate": 0.1,  # 10%
            "response_time": 5.0,  # 5 segundos
            "cost_per_hour": 100.0  # $100/hora
        }
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Verificar alertas basadas en métricas"""
        if metrics["error_rate"] > self.thresholds["error_rate"]:
            await self.send_alert("High error rate detected", metrics)
        
        if metrics["avg_response_time"] > self.thresholds["response_time"]:
            await self.send_alert("Slow response times", metrics)
        
        if metrics["hourly_cost"] > self.thresholds["cost_per_hour"]:
            await self.send_alert("High cost detected", metrics)
```

### **3. Dashboard de Métricas**
```python
@app.get("/metrics/llm")
async def get_llm_metrics():
    """Endpoint para métricas de LLM"""
    return {
        "total_requests": await get_total_requests(),
        "success_rate": await get_success_rate(),
        "avg_response_time": await get_avg_response_time(),
        "total_tokens": await get_total_tokens(),
        "cost_breakdown": await get_cost_breakdown(),
        "model_usage": await get_model_usage_stats()
    }
```

---

## 🔒 **Seguridad y Privacidad**

### **1. Sanitización de Datos**
```python
def sanitize_input(text: str) -> str:
    """Sanitizar entrada del usuario"""
    # Remover información sensible
    text = re.sub(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b', '[CARD_NUMBER]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Limitar longitud
    if len(text) > 10000:
        text = text[:10000] + "..."
    
    return text
```

### **2. Validación de Respuestas**
```python
def validate_llm_response(response: str) -> Dict[str, Any]:
    """Validar respuesta del LLM"""
    validation = {
        "is_valid": True,
        "contains_sensitive_data": False,
        "is_appropriate": True,
        "errors": []
    }
    
    # Verificar contenido sensible
    sensitive_patterns = [
        r'password\s*[:=]\s*\w+',
        r'api[_-]?key\s*[:=]\s*\w+',
        r'token\s*[:=]\s*\w+'
    ]
    
    for pattern in sensitive_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            validation["contains_sensitive_data"] = True
            validation["is_valid"] = False
            validation["errors"].append("Contains sensitive data")
    
    return validation
```

### **3. Logging Seguro**
```python
def safe_log_request(prompt: str, response: str) -> Dict[str, str]:
    """Log seguro sin exponer datos sensibles"""
    return {
        "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
        "response_hash": hashlib.md5(response.encode()).hexdigest(),
        "prompt_length": len(prompt),
        "response_length": len(response),
        "timestamp": datetime.now().isoformat()
    }
```

---

## 💰 **Costos y Eficiencia**

### **1. Optimización de Costos**
```python
class CostOptimizer:
    def __init__(self):
        self.model_costs = {
            "gpt-4": 0.03,  # $0.03 per 1K tokens
            "gpt-3.5-turbo": 0.002,
            "claude-3-sonnet": 0.015,
            "claude-3-haiku": 0.00025
        }
    
    def select_optimal_model(self, task_complexity: str, budget: float) -> str:
        """Seleccionar modelo óptimo basado en complejidad y presupuesto"""
        if task_complexity == "simple" and budget < 0.01:
            return "gpt-3.5-turbo"
        elif task_complexity == "complex" and budget > 0.02:
            return "gpt-4"
        else:
            return "claude-3-sonnet"
    
    def estimate_cost(self, prompt: str, model: str) -> float:
        """Estimar costo de la operación"""
        tokens = len(prompt.split()) * 1.3  # Aproximación
        cost_per_1k = self.model_costs.get(model, 0.01)
        return (tokens / 1000) * cost_per_1k
```

### **2. Caché de Costos**
```python
def cache_expensive_operations(prompt: str, model: str) -> bool:
    """Determinar si cachear operación costosa"""
    cost = estimate_cost(prompt, model)
    
    # Cachear si cuesta más de $0.01
    return cost > 0.01
```

### **3. Budget Management**
```python
class BudgetManager:
    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.daily_spent = 0.0
        self.redis = redis.Redis()
    
    async def check_budget(self, estimated_cost: float) -> bool:
        """Verificar si hay presupuesto disponible"""
        today = datetime.now().strftime("%Y-%m-%d")
        spent_key = f"budget_spent:{today}"
        
        spent = float(self.redis.get(spent_key) or 0)
        
        if spent + estimated_cost > self.daily_budget:
            return False
        
        return True
    
    async def record_cost(self, cost: float):
        """Registrar costo gastado"""
        today = datetime.now().strftime("%Y-%m-%d")
        spent_key = f"budget_spent:{today}"
        
        self.redis.incrbyfloat(spent_key, cost)
        self.redis.expire(spent_key, 86400)  # 24 horas
```

---

## 🎯 **Checklist de Mejores Prácticas**

### **✅ Configuración**
- [ ] Configuración centralizada de modelos
- [ ] Gestión de múltiples proveedores
- [ ] Configuración por entorno (dev/prod)
- [ ] Rotación de API keys

### **✅ Prompt Engineering**
- [ ] Templates reutilizables
- [ ] Few-shot learning implementado
- [ ] Chain of thought para tareas complejas
- [ ] Validación de prompts

### **✅ Performance**
- [ ] Caché inteligente implementado
- [ ] Procesamiento asíncrono
- [ ] Rate limiting configurado
- [ ] Optimización de tokens

### **✅ Confiabilidad**
- [ ] Retry con backoff exponencial
- [ ] Circuit breaker implementado
- [ ] Manejo de errores específicos
- [ ] Timeout configurado

### **✅ Monitoreo**
- [ ] Métricas de uso registradas
- [ ] Alertas configuradas
- [ ] Dashboard de métricas
- [ ] Logging estructurado

### **✅ Seguridad**
- [ ] Sanitización de entrada
- [ ] Validación de respuestas
- [ ] Logging seguro
- [ ] Protección de datos sensibles

### **✅ Costos**
- [ ] Optimización de costos
- [ ] Budget management
- [ ] Selección de modelo óptimo
- [ ] Estimación de costos

---

## 🚀 **Implementación Rápida**

Para implementar estas mejores prácticas:

1. **Instalar dependencias**:
```bash
pip install openai anthropic google-generativeai transformers redis
```

2. **Usar el archivo de mejores prácticas**:
```python
from llm_best_practices import LLMFactory, PromptTemplate

# Crear procesador
processor = LLMFactory.create_processor()

# Usar templates
prompt = PromptTemplate.content_analysis_prompt("Your content here")
result = await processor.process_single(prompt, "model_0", config)
```

3. **Configurar variables de entorno**:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

4. **Ejecutar con monitoreo**:
```python
# Con métricas y alertas
processor = LLMFactory.create_processor()
await processor.process_with_monitoring(prompt, model_name, config)
```

---

**¡Con estas mejores prácticas tendrás un sistema de LLM robusto, eficiente y seguro!** 🎉







