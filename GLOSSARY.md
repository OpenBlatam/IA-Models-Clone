# 📚 Glosario - Blatam Academy Features

## A

### Adaptive Cache Strategy
Estrategia de cache que ajusta automáticamente su comportamiento basado en patrones de uso observados. Mejora el hit rate comparado con estrategias estáticas.

### API Gateway
Punto de entrada único para todas las APIs del sistema. Proporciona routing, autenticación, rate limiting y monitoreo.

### Auto-Scaling
Capacidad del sistema para escalar automáticamente recursos (como cache size o workers) basado en métricas de carga.

## B

### Batch Processing
Procesamiento de múltiples requests juntos para mejorar eficiencia y throughput.

### BUL (Business Unlimited)
Sistema principal de generación de documentos empresariales con soporte para TruthGPT y KV Cache avanzado.

## C

### Cache Hit Rate
Porcentaje de requests que encuentran su resultado en el cache. Un hit rate alto indica buen uso del cache.

### Cache Miss
Cuando un request no encuentra su resultado en el cache y debe procesarse desde cero.

### Cache Warming
Proceso de precargar el cache con datos comunes antes de que el sistema entre en producción para evitar cold starts.

### Compression Ratio
Ratio de compresión aplicado al cache. 0.3 significa que el cache usa 30% del tamaño original.

### CUDA
Compute Unified Device Architecture. Plataforma de computación paralela de NVIDIA para GPUs.

## D

### Distributed Cache
Cache distribuido entre múltiples nodos o GPUs para mejor escalabilidad y rendimiento.

## E

### Eviction
Proceso de remover entradas del cache cuando está lleno, siguiendo la estrategia configurada (LRU, LFU, etc.).

## F

### Float16 (FP16)
Formato de punto flotante de 16 bits. Usado para reducir uso de memoria y aumentar velocidad en GPUs.

## G

### GPU (Graphics Processing Unit)
Unidad de procesamiento gráfico. Usado para acelerar cálculos de IA y procesamiento paralelo.

## H

### Hit Rate
Ver "Cache Hit Rate"

## I

### Inference Mode
Modo de operación del cache optimizado para inferencia (predicción) en lugar de entrenamiento.

## K

### KV Cache (Key-Value Cache)
Cache que almacena pares clave-valor, específicamente optimizado para modelos transformer que usan atención (keys y values).

## L

### Latency
Tiempo que toma procesar un request desde inicio hasta fin. Medido en milisegundos (ms).

### LFU (Least Frequently Used)
Estrategia de evicción que remueve las entradas menos frecuentemente accedidas.

### LRU (Least Recently Used)
Estrategia de evicción que remueve las entradas menos recientemente accedidas.

## M

### Memory Pool
Pool de memoria pre-allocada para reutilización, reduciendo overhead de allocaciones frecuentes.

### Mixed Precision
Técnica que usa diferentes precisiones (float32 y float16) en diferentes partes del cálculo para optimizar rendimiento.

### Multi-GPU
Sistema que usa múltiples GPUs simultáneamente para procesamiento paralelo.

### Multi-Tenant
Arquitectura donde un sistema sirve a múltiples clientes (tenants) con aislamiento de datos y recursos.

## O

### OOM (Out of Memory)
Error que ocurre cuando no hay suficiente memoria disponible (RAM o GPU) para una operación.

## P

### P50, P95, P99
Percentiles de latencia:
- **P50**: 50% de requests son más rápidos
- **P95**: 95% de requests son más rápidos
- **P99**: 99% de requests son más rápidos

### Persistence
Capacidad del cache de guardar su estado en disco para sobrevivir reinicios.

### Prefetching
Técnica de cargar datos en cache antes de que sean solicitados, basado en predicciones.

## Q

### Quantization
Técnica de reducir precisión numérica (ej: float32 → int8) para ahorrar memoria y aumentar velocidad.

## R

### Rate Limiting
Mecanismo que limita el número de requests que un cliente puede hacer en un período de tiempo.

### Redis
Sistema de cache en memoria de alto rendimiento usado como cache distribuido.

## S

### SVD (Singular Value Decomposition)
Técnica de compresión matemática usada para reducir tamaño de tensores manteniendo información importante.

## T

### Throughput
Número de requests que el sistema puede procesar por segundo (req/s).

### TruthGPT
Modelo de lenguaje usado en el sistema BUL para generación de documentos empresariales.

## U

### Ultra Adaptive KV Cache Engine
Motor de cache de nivel empresarial con características avanzadas como estrategias adaptativas, compresión, cuantización, y soporte multi-GPU.

## V

### Validation
Proceso de verificar que la configuración del sistema es válida y consistente antes de usarla.

---

**Nota**: Este glosario cubre los términos más importantes. Para más detalles, consulta las guías específicas en la documentación.

