# Contributing to Universal Model Benchmark AI

¡Gracias por tu interés en contribuir! Este proyecto es polyglot y acepta contribuciones en múltiples lenguajes.

## Cómo Contribuir

### Agregar un Nuevo Benchmark

1. Crea un nuevo archivo en `python/benchmarks/` siguiendo el patrón de `base_benchmark.py`
2. Implementa los métodos `format_prompt` y `evaluate_answer`
3. Registra el benchmark en `python/orchestrator/main.py`

Ejemplo:

```python
from .base_benchmark import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    def __init__(self, shots: int = 0, max_samples: int = None):
        super().__init__(
            name="my_benchmark",
            dataset_name="my_dataset",
            shots=shots,
            max_samples=max_samples
        )
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        # Implementar formato de prompt
        pass
    
    def evaluate_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        # Implementar evaluación
        pass
```

### Agregar Soporte para un Nuevo Modelo

1. Asegúrate de que el modelo esté disponible en HuggingFace o localmente
2. Agrega la configuración en `config/example.yaml`
3. El `ModelLoader` debería soportarlo automáticamente

### Mejoras de Rendimiento

- **Rust**: Optimizaciones en `rust/src/inference.rs` y `rust/src/metrics.rs`
- **C++**: Kernels CUDA en `cpp/include/inference_kernel.h`
- **Go**: Mejoras de concurrencia en `go/workers/benchmark_worker.go`

## Estándares de Código

- **Python**: PEP 8, type hints, docstrings
- **Rust**: `rustfmt`, `clippy`
- **Go**: `gofmt`, `golint`
- **TypeScript**: ESLint, Prettier

## Testing

Antes de hacer commit, ejecuta:

```bash
make test
```

## Pull Requests

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/amazing-feature`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## Preguntas

Si tienes preguntas, abre un issue en GitHub.












