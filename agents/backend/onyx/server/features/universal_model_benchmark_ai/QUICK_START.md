# Quick Start Guide

## Instalación Rápida

### 1. Prerrequisitos

```bash
# Verificar Python 3.10+
python --version

# Verificar CUDA (opcional pero recomendado)
nvcc --version

# Verificar Rust
rustc --version

# Verificar Go
go version
```

### 2. Instalar Dependencias

```bash
# Instalar todo
make install

# O manualmente:
cd python && pip install -r requirements.txt
cd ../rust && cargo build --release
cd ../go && go mod download
cd ../typescript && npm install
```

### 3. Configurar

Copia el archivo de configuración de ejemplo:

```bash
cp config/example.yaml config/my_config.yaml
```

Edita `config/my_config.yaml` para seleccionar modelos y benchmarks.

### 4. Ejecutar

```bash
# Ejecutar todos los benchmarks en todos los modelos
make run

# O con configuración personalizada
python python/orchestrator/main.py --config config/my_config.yaml

# Ejecutar benchmark específico
python python/orchestrator/main.py --benchmark mmlu --model llama2-7b
```

## Ejemplo Básico

```python
from core.model_loader import ModelLoader, ModelType, QuantizationType
from benchmarks.mmlu_benchmark import MMLUBenchmark

# Cargar modelo
model_loader = ModelLoader(
    model_name="llama2-7b",
    model_path="meta-llama/Llama-2-7b-hf",
    model_type=ModelType.CAUSAL_LM,
    quantization=QuantizationType.FP16,
    device="cuda"
)

# Crear benchmark
benchmark = MMLUBenchmark(shots=5)

# Ejecutar
result = benchmark.run(model_loader)

print(f"Accuracy: {result.accuracy:.2%}")
print(f"Latency P50: {result.latency_p50:.3f}s")
```

## Resultados

Los resultados se guardan en `results/` en formato JSON. Puedes visualizarlos con:

```bash
# Ver resultados más recientes
ls -lt results/ | head -1
cat results/results_*.json | jq
```

## Troubleshooting

### Error: CUDA out of memory
- Reduce `batch_size` en la configuración
- Usa cuantización INT8 o INT4
- Reduce `max_samples`

### Error: Model not found
- Verifica que el modelo esté disponible en HuggingFace
- O proporciona un path local válido

### Error: Dataset not found
- Algunos datasets requieren aceptar términos de uso
- Ejecuta: `python -c "from datasets import load_dataset; load_dataset('dataset_name')"`

## Siguiente Paso

Lee el [README.md](README.md) completo para más detalles.












