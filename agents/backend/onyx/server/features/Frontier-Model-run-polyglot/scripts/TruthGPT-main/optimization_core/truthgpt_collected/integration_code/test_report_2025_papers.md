# 📊 Reporte de Tests - Papers 2025

**Fecha**: 2025-11-23 16:46:40

**Configuración**: batch_size=4, seq_len=32, num_tests=10


---

## 📈 Baseline (Sin Mejoras)

- **Tiempo Forward**: 0.0168s
- **Throughput**: 238.39 samples/s
- **Memoria**: 0.25 MB

## 🔬 Mejoras por Paper Individual

### AdaptiveGoT

❌ **Error en tests**


### SOLAR

**Tasa de Éxito**: 100.0%

- **Tiempo Forward**: 0.0393s (-134.5% vs baseline)
- **Throughput**: 216.71 samples/s (-9.1% vs baseline)
- **Mejora de Output**: +1.44%
- **Memoria**: 0.25 MB

**Métricas Específicas**:

- curriculum_difficulty: 0.0000
- reasoning_improvement: 0.0000
- selected_structures: ['chain', 'tree', 'graph']


**Mejoras Clave**:

- Optimización dinámica de arquitectura de razonamiento

- Aprendizaje curricular para progresión gradual

- Selección adaptativa de estructuras



### RLOfThoughts

**Tasa de Éxito**: 100.0%

- **Tiempo Forward**: 0.0035s (+79.4% vs baseline)
- **Throughput**: 1384.71 samples/s (+480.8% vs baseline)
- **Mejora de Output**: +0.51%
- **Memoria**: 0.25 MB

**Métricas Específicas**:

- block_usage: [0.30000001192092896, 0.05000000074505806, 0.05000000074505806, 0.6000000238418579]
- avg_value: 0.0785
- navigation_confidence: 0.3424
- num_blocks: 4.0000


**Mejoras Clave**:

- Mejoras generales de razonamiento



### RDoLT

**Tasa de Éxito**: 100.0%

- **Tiempo Forward**: 0.0690s (-311.3% vs baseline)
- **Throughput**: 105.26 samples/s (-55.8% vs baseline)
- **Mejora de Output**: +0.00%
- **Memoria**: 0.25 MB

**Métricas Específicas**:

- avg_decomposition_depth: 15.8533
- avg_num_subproblems: 15.8533
- knowledge_quality: 0.5000
- max_depth: 5.0000


**Mejoras Clave**:

- Descomposición recursiva de pensamientos lógicos

- Propagación de conocimiento entre subproblemas

- Scoring de calidad de pensamientos



### AMThinking

**Tasa de Éxito**: 100.0%

- **Tiempo Forward**: 0.3061s (-1724.3% vs baseline)
- **Throughput**: 26.96 samples/s (-88.7% vs baseline)
- **Mejora de Output**: +4.29%
- **Memoria**: 0.25 MB

**Métricas Específicas**:

- reasoning_quality: 0.4993
- sft_loss: 0.0000
- rl_reward: 0.0000
- num_layers: 24.0000


**Mejoras Clave**:

- Modelo denso 32B optimizado para razonamiento

- Pipeline SFT + RL para entrenamiento robusto

- Heads de razonamiento especializadas



### LADDER

**Tasa de Éxito**: 100.0%

- **Tiempo Forward**: 0.0171s (-1.8% vs baseline)
- **Throughput**: 245.00 samples/s (+2.8% vs baseline)
- **Mejora de Output**: +0.00%
- **Memoria**: 0.25 MB

**Métricas Específicas**:

- avg_decomposition_steps: 3.2566
- avg_verification_score: 0.5151
- learning_progress: 0.3628
- variant_generation_count: 50.0000
- max_steps: 5.0000
- num_variants_per_problem: 500.0000


**Mejoras Clave**:

- Auto-mejora mediante descomposición recursiva

- Generación de variantes progresivamente más simples

- TTRL para mejora en tiempo de test



### Enigmata

**Tasa de Éxito**: 100.0%

- **Tiempo Forward**: 0.0031s (+81.8% vs baseline)
- **Throughput**: 1486.00 samples/s (+523.3% vs baseline)
- **Mejora de Output**: +0.10%
- **Memoria**: 0.25 MB

**Métricas Específicas**:

- avg_puzzle_complexity: 0.5000
- avg_verification_score: 0.4922
- puzzle_solving_rate: 0.1743


**Mejoras Clave**:

- Puzzles sintéticos verificables para entrenamiento

- Generador + Verificador para RL

- Mejora de razonamiento lógico



### SPOC

**Tasa de Éxito**: 100.0%

- **Tiempo Forward**: 0.0088s (+47.6% vs baseline)
- **Throughput**: 492.81 samples/s (+106.7% vs baseline)
- **Mejora de Output**: +2.62%
- **Memoria**: 0.25 MB

**Métricas Específicas**:

- avg_verification_score: 0.5235
- correction_iterations: 1.9540
- self_correction_rate: 0.6513
- max_iterations: 3.0000


**Mejoras Clave**:

- Auto-corrección espontánea durante inferencia

- Verificación on-the-fly de soluciones

- Refinamiento iterativo



### K2Think

❌ **Error en tests**


### AdvancedMathBenchmark

❌ **Error en tests**



---

## 📋 Resumen Ejecutivo


### Top 3 Papers por Mejora:

1. **AMThinking**: +4.29% mejora

2. **SPOC**: +2.62% mejora

3. **SOLAR**: +1.44% mejora


### Recomendaciones:

- Usar combinación de papers para máximo rendimiento

- Para eficiencia: priorizar Enigmata, RLOfThoughts, SPOC
