# Refactoring Phase 12 - universal_model_benchmark_ai

## Overview
Duodécima fase de refactorización enfocada en centralizar constantes y añadir presets de configuración.

## Cambios Realizados

### 1. Nuevo Módulo de Constantes
- **Archivo**: `constants.rs` (nuevo)
- **Propósito**: Centralizar todas las constantes utilizadas en el código
- **Módulos de constantes**:
  - `percentiles` - Percentiles comunes (P50, P90, P95, P99, P99.9)
  - `time_ms` - Constantes de tiempo en milisegundos
  - `size_bytes` - Constantes de tamaño en bytes
  - `thresholds` - Umbrales de rendimiento
  - `batch_sizes` - Tamaños de batch comunes
  - `token_limits` - Límites de tokens comunes
  - `temperatures` - Valores de temperatura comunes
  - `top_p_values` - Valores de top-p comunes
  - `top_k_values` - Valores de top-k comunes
  - `retry` - Constantes de reintentos
  - `cache` - Constantes de caché
  - `benchmark` - Constantes de benchmark

### 2. Constantes de Percentiles
- `P50`, `P90`, `P95`, `P99`, `P99_9` - Percentiles estándar
- `STANDARD` - Array de percentiles estándar

### 3. Constantes de Tiempo
- `SECOND`, `MINUTE`, `HOUR` - Conversiones de tiempo
- `ACCEPTABLE_LATENCY` - Latencia aceptable para tiempo real
- `BATCH_LATENCY` - Latencia aceptable para procesamiento por lotes

### 4. Constantes de Tamaño
- `KB`, `MB`, `GB`, `TB` - Unidades de tamaño
- `SMALL_MODEL`, `MEDIUM_MODEL`, `LARGE_MODEL` - Umbrales de tamaño de modelo

### 5. Umbrales de Rendimiento
- `MIN_ACCURACY`, `GOOD_ACCURACY`, `EXCELLENT_ACCURACY` - Umbrales de precisión
- `MAX_LATENCY_P50`, `MAX_LATENCY_P95` - Umbrales de latencia
- `MIN_THROUGHPUT`, `GOOD_THROUGHPUT` - Umbrales de throughput

### 6. Presets de Configuración
- `batch_sizes` - SMALL, MEDIUM, LARGE, VERY_LARGE
- `token_limits` - SHORT, MEDIUM, LONG, VERY_LONG, MAXIMUM
- `temperatures` - VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
- `top_p_values` - VERY_FOCUSED, FOCUSED, BALANCED, DIVERSE, VERY_DIVERSE
- `top_k_values` - VERY_FOCUSED, FOCUSED, BALANCED, DIVERSE, VERY_DIVERSE

### 7. Constantes de Sistema
- `retry` - MAX_RETRIES, RETRY_DELAY_MS, BACKOFF_BASE
- `cache` - Tamaños de caché y TTL
- `benchmark` - Iteraciones, warmup, timeout

### 8. Exportaciones Actualizadas
- **Archivo**: `lib.rs`
- **Cambio**: Añadido módulo `constants` y exportaciones
- **Prelude**: Añadidas constantes al prelude
- **Documentación**: Añadido `constants` a la lista de módulos

## Uso de Constantes

### Percentiles
```rust
use benchmark_core::prelude::*;

let p95 = percentiles::P95;
let standard = percentiles::STANDARD;
```

### Tiempo
```rust
let one_second = time_ms::SECOND;
let acceptable = time_ms::ACCEPTABLE_LATENCY;
```

### Tamaño
```rust
let one_gb = size_bytes::GB;
let is_large = model_size > size_bytes::LARGE_MODEL;
```

### Umbrales
```rust
if accuracy >= thresholds::GOOD_ACCURACY {
    println!("Good accuracy!");
}

if latency <= thresholds::MAX_LATENCY_P50 {
    println!("Acceptable latency!");
}
```

### Presets
```rust
let config = BenchmarkConfig::builder()
    .model_path("model".to_string())
    .batch_size(batch_sizes::MEDIUM)
    .max_tokens(token_limits::LONG)
    .temperature(temperatures::MEDIUM)
    .top_p(top_p_values::BALANCED)
    .top_k(top_k_values::BALANCED)
    .build()?;
```

### Retry y Cache
```rust
let max_retries = retry::MAX_RETRIES;
let cache_size = cache::DEFAULT_TOKENIZATION_CACHE_SIZE;
let timeout = benchmark_constants::DEFAULT_TIMEOUT_SEC;
```

## Beneficios

1. **Centralización**: Todas las constantes en un solo lugar
2. **Consistencia**: Valores consistentes en todo el código
3. **Mantenibilidad**: Fácil actualizar valores
4. **Legibilidad**: Nombres descriptivos mejoran comprensión
5. **Presets**: Valores predefinidos para casos comunes
6. **Type Safety**: Constantes tipadas previenen errores

## Ejemplo de Uso Completo

```rust
use benchmark_core::prelude::*;

// Usar constantes para configuración
let config = BenchmarkConfig::builder()
    .model_path("model".to_string())
    .batch_size(batch_sizes::LARGE)
    .max_tokens(token_limits::LONG)
    .temperature(temperatures::MEDIUM)
    .top_p(top_p_values::BALANCED)
    .top_k(top_k_values::BALANCED)
    .build()?;

// Usar umbrales para validación
if metrics.accuracy >= thresholds::GOOD_ACCURACY &&
   metrics.latency_p50 <= thresholds::MAX_LATENCY_P50 {
    println!("Performance is good!");
}

// Usar constantes de tiempo
if latency_ms < time_ms::ACCEPTABLE_LATENCY {
    println!("Latency is acceptable");
}

// Usar constantes de tamaño
if model_size > size_bytes::LARGE_MODEL {
    println!("Large model detected");
}
```

## Próximos Pasos Sugeridos

1. Añadir más presets de configuración
2. Crear funciones helper que usen constantes
3. Añadir validación basada en constantes
4. Crear macros para configuraciones comunes
5. Documentar mejor los valores recomendados

## Notas

- Todas las constantes están organizadas por categoría
- Los nombres son descriptivos y claros
- Las constantes están tipadas correctamente
- Los presets facilitan configuración común
- Las constantes se exportan en el prelude para fácil acceso












