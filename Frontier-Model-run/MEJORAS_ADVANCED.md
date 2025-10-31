---

## 🔐 Seguridad Avanzada: Zero Trust Architecture

### Implementación de Zero Trust

```python
# security/zero_trust.py
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import jwt
import hashlib
import secrets

@dataclass
class SecurityContext:
    user_id: str
    device_id: str
    ip_address: str
    user_agent: str
    session_id: str
    permissions: List[str]
    last_verification: datetime

class ZeroTrustEngine:
    def __init__(self):
        self.verification_interval = timedelta(minutes=5)
        self.blocked_ips: Dict[str, datetime] = {}
        self.suspicious_activities: List[Dict] = []
    
    def verify_request(
        self,
        context: SecurityContext,
        resource: str,
        action: str
    ) -> tuple[bool, Optional[str]]:
        """Verificar cada request con principios Zero Trust"""
        
        # 1. Verificar identidad
        if not self._verify_identity(context):
            return False, "Identity verification failed"
        
        # 2. Verificar dispositivo
        if not self._verify_device(context):
            return False, "Device verification failed"
        
        # 3. Verificar ubicación/red
        if not self._verify_network(context):
            return False, "Network verification failed"
        
        # 4. Verificar permisos
        if not self._check_permissions(context, resource, action):
            return False, "Insufficient permissions"
        
        # 5. Verificar comportamiento anómalo
        if self._detect_anomaly(context):
            return False, "Anomalous activity detected"
        
        # 6. Verificar que no esté bloqueado
        if self._is_blocked(context.ip_address):
            return False, "IP address blocked"
        
        # Actualizar último acceso
        context.last_verification = datetime.utcnow()
        
        return True, None
    
    def _verify_identity(self, context: SecurityContext) -> bool:
        """Verificar identidad del usuario"""
        # En producción: verificar JWT, OAuth2 token, etc.
        # Verificar que el token no haya expirado
        # Verificar firma del token
        return True  # Placeholder
    
    def _verify_device(self, context: SecurityContext) -> bool:
        """Verificar que el dispositivo sea conocido y seguro"""
        # Verificar device fingerprint
        # Verificar que el dispositivo tenga políticas de seguridad
        return True  # Placeholder
    
    def _verify_network(self, context: SecurityContext) -> bool:
        """Verificar red desde donde viene la request"""
        # Verificar geolocalización
        # Verificar si está en lista de IPs permitidas
        # Verificar VPN/Proxy
        return True  # Placeholder
    
    def _check_permissions(
        self,
        context: SecurityContext,
        resource: str,
        action: str
    ) -> bool:
        """Verificar permisos con RBAC/ABAC"""
        required_permission = f"{resource}:{action}"
        return required_permission in context.permissions
    
    def _detect_anomaly(self, context: SecurityContext) -> bool:
        """Detección de comportamiento anómalo"""
        # Analizar patrones de acceso
        # Detectar velocidad anormal de requests
        # Detectar acceso desde múltiples ubicaciones simultáneas
        return False  # Placeholder
    
    def _is_blocked(self, ip_address: str) -> bool:
        """Verificar si IP está bloqueada"""
        if ip_address in self.blocked_ips:
            block_until = self.blocked_ips[ip_address]
            if datetime.utcnow() < block_until:
                return True
            else:
                # Remover de bloqueados si expiró
                del self.blocked_ips[ip_address]
        return False
    
    def block_ip(self, ip_address: str, duration: timedelta):
        """Bloquear IP temporalmente"""
        self.blocked_ips[ip_address] = datetime.utcnow() + duration
    
    def log_suspicious_activity(self, context: SecurityContext, reason: str):
        """Registrar actividad sospechosa"""
        self.suspicious_activities.append({
            "timestamp": datetime.utcnow(),
            "user_id": context.user_id,
            "ip_address": context.ip_address,
            "reason": reason
        })

# Middleware FastAPI para Zero Trust
from fastapi import Request, HTTPException, status

async def zero_trust_middleware(request: Request, call_next):
    """Middleware que aplica Zero Trust a cada request"""
    engine = ZeroTrustEngine()
    
    # Obtener contexto de seguridad
    context = SecurityContext(
        user_id=request.headers.get("X-User-ID", "anonymous"),
        device_id=request.headers.get("X-Device-ID", ""),
        ip_address=request.client.host,
        user_agent=request.headers.get("User-Agent", ""),
        session_id=request.headers.get("X-Session-ID", ""),
        permissions=extract_permissions(request),  # Implementar
        last_verification=datetime.utcnow()
    )
    
    # Verificar request
    resource = request.url.path
    action = request.method.lower()
    
    allowed, reason = engine.verify_request(context, resource, action)
    
    if not allowed:
        # Registrar actividad sospechosa
        engine.log_suspicious_activity(context, reason or "Request denied")
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=reason or "Access denied"
        )
    
    response = await call_next(request)
    
    # Agregar headers de seguridad
    response.headers["X-Zero-Trust-Verified"] = "true"
    response.headers["X-Verification-Time"] = str(datetime.utcnow())
    
    return response
```

---

## 🧪 Testing Avanzado: Property-Based Testing

### Property-Based Testing con Hypothesis

```python
# tests/property_based_testing.py
from hypothesis import given, strategies as st
import pytest
from typing import List

class ModelInferenceTester:
    @given(
        prompt=st.text(min_size=1, max_size=1000),
        max_tokens=st.integers(min_value=1, max_value=512),
        temperature=st.floats(min_value=0.0, max_value=2.0)
    )
    def test_inference_consistency(self, prompt: str, max_tokens: int, temperature: float):
        """Property: La inferencia debe ser determinística con temperature=0"""
        model = load_model()
        
        if temperature == 0.0:
            # Con temperature=0, debe ser determinístico
            result1 = model.generate(prompt, max_tokens=max_tokens, temperature=0.0)
            result2 = model.generate(prompt, max_tokens=max_tokens, temperature=0.0)
            
            assert result1 == result2, "Inference should be deterministic with temperature=0"
        
        # Property: La respuesta no debe ser más larga que max_tokens
        result = model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        assert len(result.split()) <= max_tokens, "Response should respect max_tokens"
    
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        prompts=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=32)
    )
    def test_batch_inference(self, batch_size: int, prompts: List[str]):
        """Property: Batch inference debe ser equivalente a inferencia secuencial"""
        model = load_model()
        
        # Inferencia secuencial
        sequential_results = [
            model.generate(prompt) for prompt in prompts[:batch_size]
        ]
        
        # Inferencia en batch
        batch_results = model.generate_batch(prompts[:batch_size])
        
        # Deben ser equivalentes (puede haber pequeñas diferencias por numerics)
        assert len(sequential_results) == len(batch_results)
        
        for seq, batch in zip(sequential_results, batch_results):
            # Verificar que sean similares (usar distancia de edición o embedding similarity)
            similarity = self._calculate_similarity(seq, batch)
            assert similarity > 0.95, f"Sequential and batch results should be similar"
    
    @given(
        requests=st.lists(
            st.fixed_dictionaries({
                "prompt": st.text(min_size=1, max_size=500),
                "priority": st.integers(min_value=1, max_value=10)
            }),
            min_size=1,
            max_size=100
        )
    )
    def test_priority_queue(self, requests: List[dict]):
        """Property: Requests con mayor priority deben procesarse primero"""
        queue = PriorityQueue()
        
        # Agregar requests
        for req in requests:
            queue.add(req["prompt"], priority=req["priority"])
        
        # Procesar y verificar orden
        processed = []
        while not queue.is_empty():
            processed.append(queue.pop())
        
        # Verificar que estén ordenados por priority (mayor primero)
        priorities = [r["priority"] for r in requests]
        processed_priorities = [r["priority"] for r in processed]
        
        assert processed_priorities == sorted(priorities, reverse=True), \
            "Requests should be processed in priority order"
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud entre dos textos"""
        # Usar Jaccard similarity o embedding similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
```

---

## 📈 Observabilidad: Distributed Tracing Avanzado

### Implementación Completa con OpenTelemetry

```python
# observability/distributed_tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.trace import Status, StatusCode
from contextlib import contextmanager
from typing import Optional, Dict, Any
import time

# Configurar tracer
resource = Resource.create({
    "service.name": "inference-api",
    "service.version": "2.0.0",
    "deployment.environment": "production"
})

trace.set_tracer_provider(TracerProvider(resource=resource))

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

class TracingMiddleware:
    """Middleware para tracing automático"""
    
    @staticmethod
    async def process_request(request, call_next):
        """Procesar request con tracing"""
        with tracer.start_as_current_span("request") as span:
            # Agregar atributos del span
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.route", request.url.path)
            span.set_attribute("user.id", request.headers.get("X-User-ID", "unknown"))
            
            start_time = time.time()
            
            try:
                response = await call_next(request)
                
                # Agregar atributos de respuesta
                span.set_attribute("http.status_code", response.status_code)
                span.set_status(Status(StatusCode.OK))
                
                duration = time.time() - start_time
                span.set_attribute("duration", duration)
                
                return response
                
            except Exception as e:
                # Registrar error en span
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

@contextmanager
def traced_operation(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager para operaciones con tracing"""
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

# Ejemplo de uso en endpoints
@app.post("/v1/infer")
async def infer(request: InferRequest):
    with traced_operation("inference.generate", {
        "model.name": "gpt2",
        "prompt.length": len(request.prompt),
        "max_tokens": request.max_tokens
    }) as span:
        # Preprocesamiento
        with traced_operation("inference.preprocess") as preprocess_span:
            tokenized = tokenize(request.prompt)
            preprocess_span.set_attribute("tokens.count", len(tokenized))
        
        # Inferencia
        with traced_operation("inference.model_forward") as forward_span:
            output = model.generate(tokenized)
            forward_span.set_attribute("output.length", len(output))
        
        # Postprocesamiento
        with traced_operation("inference.postprocess") as postprocess_span:
            result = detokenize(output)
            postprocess_span.set_attribute("result.length", len(result))
        
        span.set_attribute("result.length", len(result))
        
        return {"result": result}
```

---

## 🚀 Auto-Scaling Inteligente con ML

### Predictor de Carga con Machine Learning

```python
# scaling/ml_autoscaler.py
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime, timedelta

@dataclass
class LoadMetrics:
    timestamp: datetime
    requests_per_second: float
    avg_latency_ms: float
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: float
    queue_depth: int
    error_rate: float

class MLLoadPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.lookback_window = 60  # minutos
        self.prediction_horizon = 15  # minutos
    
    def train(self, historical_data: List[LoadMetrics]):
        """Entrenar modelo con datos históricos"""
        # Preparar features (últimos N minutos de métricas)
        X, y = self._prepare_training_data(historical_data)
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar modelo (predecir requests_per_second en horizonte)
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
    
    def predict_load(self, current_metrics: List[LoadMetrics]) -> float:
        """Predecir carga futura"""
        if not self.is_trained:
            # Fallback a predicción simple
            return self._simple_prediction(current_metrics)
        
        # Preparar features actuales
        X = self._prepare_features(current_metrics)
        X_scaled = self.scaler.transform([X])
        
        # Predecir
        prediction = self.model.predict(X_scaled)[0]
        
        return max(0, prediction)  # No puede ser negativo
    
    def _prepare_training_data(
        self,
        historical_data: List[LoadMetrics]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preparar datos para entrenamiento"""
        X = []
        y = []
        
        for i in range(self.lookback_window, len(historical_data) - self.prediction_horizon):
            # Features: últimas N métricas
            window = historical_data[i - self.lookback_window:i]
            features = self._prepare_features(window)
            X.append(features)
            
            # Target: requests_per_second en el horizonte
            future_metric = historical_data[i + self.prediction_horizon]
            y.append(future_metric.requests_per_second)
        
        return np.array(X), np.array(y)
    
    def _prepare_features(self, metrics: List[LoadMetrics]) -> List[float]:
        """Extraer features de una ventana de métricas"""
        if not metrics:
            return [0.0] * (self.lookback_window * 6)  # 6 features por minuto
        
        # Features: estadísticas agregadas de cada métrica
        features = []
        
        # Promedio, max, min, std de cada métrica
        metric_names = [
            'requests_per_second', 'avg_latency_ms',
            'cpu_usage_percent', 'memory_usage_percent',
            'gpu_usage_percent', 'queue_depth'
        ]
        
        for metric_name in metric_names:
            values = [getattr(m, metric_name) for m in metrics]
            features.extend([
                np.mean(values),
                np.max(values),
                np.min(values),
                np.std(values) if len(values) > 1 else 0.0
            ])
        
        # Agregar tendencia (derivada)
        if len(metrics) > 1:
            for metric_name in metric_names:
                values = [getattr(m, metric_name) for m in metrics]
                trend = (values[-1] - values[0]) / len(values)
                features.append(trend)
        
        return features
    
    def _simple_prediction(self, current_metrics: List[LoadMetrics]) -> float:
        """Predicción simple sin ML (fallback)"""
        if not current_metrics:
            return 0.0
        
        # Usar promedio móvil con tendencia
        recent_rps = [m.requests_per_second for m in current_metrics[-10:]]
        avg_rps = np.mean(recent_rps)
        
        # Calcular tendencia
        if len(recent_rps) > 1:
            trend = (recent_rps[-1] - recent_rps[0]) / len(recent_rps)
            prediction = avg_rps + trend * self.prediction_horizon
        else:
            prediction = avg_rps
        
        return max(0, prediction)

class IntelligentAutoScaler:
    def __init__(self):
        self.predictor = MLLoadPredictor()
        self.current_replicas = 1
        self.min_replicas = 1
        self.max_replicas = 10
        self.target_rps_per_replica = 50  # requests por segundo por réplica
        self.scale_up_threshold = 0.8  # Escalar si uso > 80%
        self.scale_down_threshold = 0.3  # Reducir si uso < 30%
    
    def should_scale(self, current_metrics: List[LoadMetrics]) -> tuple[bool, int]:
        """Decidir si escalar y cuántas réplicas"""
        if len(current_metrics) < self.predictor.lookback_window:
            # No hay suficientes datos, mantener actual
            return False, self.current_replicas
        
        # Predecir carga futura
        predicted_rps = self.predictor.predict_load(current_metrics)
        
        # Calcular réplicas necesarias
        required_replicas = int(np.ceil(predicted_rps / self.target_rps_per_replica))
        required_replicas = max(self.min_replicas, min(self.max_replicas, required_replicas))
        
        # Decidir si escalar
        current_usage = predicted_rps / (self.current_replicas * self.target_rps_per_replica)
        
        if current_usage > self.scale_up_threshold:
            # Necesita más réplicas
            return True, required_replicas
        elif current_usage < self.scale_down_threshold and required_replicas < self.current_replicas:
            # Puede reducir réplicas
            return True, required_replicas
        else:
            # Mantener actual
            return False, self.current_replicas
    
    def train_predictor(self, historical_data: List[LoadMetrics]):
        """Entrenar predictor con datos históricos"""
        self.predictor.train(historical_data)

# Uso
autoscaler = IntelligentAutoScaler()

# Entrenar con datos históricos
historical_metrics = load_historical_metrics()  # Implementar
autoscaler.train_predictor(historical_metrics)

# En loop de scaling
current_metrics = get_current_metrics()  # Implementar
should_scale, target_replicas = autoscaler.should_scale(current_metrics)

if should_scale:
    scale_to_replicas(target_replicas)
```

---

## 🔄 Data Pipeline: ETL para Training Data

### Pipeline Completo de Procesamiento

```python
# pipelines/etl_pipeline.py
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
import json

@dataclass
class TrainingExample:
    prompt: str
    completion: str
    metadata: Dict
    quality_score: float
    created_at: datetime

class ETLPipeline:
    def __init__(self, db_uri: str):
        self.engine = create_engine(db_uri)
        self.processed_count = 0
        self.filtered_count = 0
    
    def extract(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Extraer datos de fuente"""
        query = """
        SELECT 
            prompt,
            completion,
            metadata,
            quality_score,
            created_at
        FROM inference_logs
        WHERE created_at BETWEEN :start_date AND :end_date
        """
        
        df = pd.read_sql(query, self.engine, params={
            "start_date": start_date,
            "end_date": end_date
        })
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transformar y limpiar datos"""
        # 1. Filtrar por calidad
        df = df[df['quality_score'] >= 0.7]
        self.filtered_count += len(df[df['quality_score'] < 0.7])
        
        # 2. Limpiar texto
        df['prompt'] = df['prompt'].apply(self._clean_text)
        df['completion'] = df['completion'].apply(self._clean_text)
        
        # 3. Filtrar ejemplos muy cortos o muy largos
        df = df[
            (df['prompt'].str.len() >= 10) &
            (df['prompt'].str.len() <= 2000) &
            (df['completion'].str.len() >= 5) &
            (df['completion'].str.len() <= 2000)
        ]
        
        # 4. Parsear metadata JSON
        df['metadata'] = df['metadata'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        # 5. Agregar features derivadas
        df['prompt_length'] = df['prompt'].str.len()
        df['completion_length'] = df['completion'].str.len()
        df['total_length'] = df['prompt_length'] + df['completion_length']
        
        # 6. Remover duplicados
        df = df.drop_duplicates(subset=['prompt', 'completion'])
        
        return df
    
    def load(self, df: pd.DataFrame, output_path: str):
        """Cargar datos procesados"""
        # Guardar en formato JSONL para training
        with open(output_path, 'w') as f:
            for _, row in df.iterrows():
                example = {
                    "prompt": row['prompt'],
                    "completion": row['completion'],
                    "metadata": row['metadata']
                }
                f.write(json.dumps(example) + '\n')
        
        self.processed_count += len(df)
        
        # También guardar estadísticas
        stats = {
            "total_examples": len(df),
            "avg_prompt_length": df['prompt_length'].mean(),
            "avg_completion_length": df['completion_length'].mean(),
            "processed_at": datetime.utcnow().isoformat()
        }
        
        with open(output_path.replace('.jsonl', '_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def run(self, start_date: datetime, end_date: datetime, output_path: str):
        """Ejecutar pipeline completo"""
        print(f"Extracting data from {start_date} to {end_date}")
        df = self.extract(start_date, end_date)
        print(f"Extracted {len(df)} examples")
        
        print("Transforming data...")
        df = self.transform(df)
        print(f"After transformation: {len(df)} examples")
        print(f"Filtered out: {self.filtered_count} examples")
        
        print(f"Loading to {output_path}")
        self.load(df, output_path)
        print(f"Processed {self.processed_count} examples total")
    
    def _clean_text(self, text: str) -> str:
        """Limpiar texto"""
        # Remover caracteres especiales
        # Normalizar espacios
        # etc.
        return text.strip()
```

---

*Mejoras avanzadas agregadas - Versión 2.5*
*Nuevas secciones: Zero Trust Security, Property-Based Testing, Distributed Tracing, ML Auto-Scaling, ETL Pipeline*

