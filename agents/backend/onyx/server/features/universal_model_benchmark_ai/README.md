# Universal Model Benchmark AI - Polyglot System

Sistema polyglot completo para probar todos los modelos de IA con todos los benchmarks disponibles, utilizando las mejores librerías open source para cada componente.

## 🎯 Características

- **Polyglot**: Python, Rust, Go, C++, TypeScript/JavaScript
- **Multi-Modelo**: Soporte para todos los modelos open source principales
- **Multi-Benchmark**: Suite completa de benchmarks estándar de la industria
- **Alto Rendimiento**: Optimizado con las mejores librerías open source
- **Escalable**: Arquitectura distribuida para ejecución concurrente
- **Extensible**: Fácil agregar nuevos modelos y benchmarks

## 📁 Estructura del Proyecto

```
universal_model_benchmark_ai/
├── python/              # Orquestación y ML frameworks
│   ├── core/           # Núcleo del sistema
│   ├── models/         # Loaders de modelos
│   ├── benchmarks/     # Implementaciones de benchmarks
│   ├── orchestrator/   # Orquestador principal
│   └── utils/          # Utilidades
├── rust/               # Operaciones de alto rendimiento
│   ├── inference/      # Motor de inferencia optimizado
│   ├── metrics/        # Cálculo de métricas
│   └── data/           # Procesamiento de datos
├── go/                 # Servicios concurrentes
│   ├── workers/        # Workers de benchmark
│   ├── scheduler/      # Planificador de tareas
│   └── api/            # API REST
├── cpp/                # Optimizaciones de bajo nivel
│   ├── kernels/        # Kernels CUDA/OpenCL
│   └── optimizations/  # Optimizaciones específicas
├── typescript/         # Frontend y API
│   ├── web/            # Interfaz web
│   ├── api/            # API TypeScript
│   └── dashboard/      # Dashboard de resultados
├── config/             # Configuraciones
├── data/               # Datos de benchmarks
└── results/            # Resultados de ejecuciones
```

## 🚀 Quick Start

### Prerrequisitos

```bash
# Python 3.10+
python --version

# Rust 1.70+
rustc --version

# Go 1.21+
go version

# Node.js 18+
node --version

# CUDA 11.8+ (opcional, para GPU)
nvcc --version
```

### Instalación

```bash
# Clonar y entrar al directorio
cd universal_model_benchmark_ai

# Instalar dependencias Python
cd python
pip install -r requirements.txt

# Compilar Rust
cd ../rust
cargo build --release

# Compilar Go
cd ../go
go build ./...

# Instalar TypeScript
cd ../typescript
npm install
```

### Uso Básico

```bash
# Ejecutar todos los benchmarks en todos los modelos
python python/orchestrator/main.py --all

# Ejecutar benchmark específico
python python/orchestrator/main.py --benchmark mmlu --model llama2

# Ejecutar con configuración personalizada
python python/orchestrator/main.py --config config/custom.yaml
```

## 📊 Benchmarks Soportados

### NLP Benchmarks
- **MMLU** (Massive Multitask Language Understanding)
- **HellaSwag** (Commonsense reasoning)
- **TruthfulQA** (Truthfulness)
- **GSM8K** (Mathematical reasoning)
- **HumanEval** (Code generation)
- **ARC** (AI2 Reasoning Challenge)
- **WinoGrande** (Commonsense reasoning)
- **PIQA** (Physical reasoning)
- **LAMBADA** (Long-range dependencies)
- **SQuAD** (Question answering)

### Vision Benchmarks
- **ImageNet** (Image classification)
- **COCO** (Object detection)
- **ADE20K** (Semantic segmentation)

### Multimodal Benchmarks
- **VQAv2** (Visual question answering)
- **TextVQA** (Text in images)

## 🤖 Modelos Soportados

### Language Models
- LLaMA 2/3 (Meta)
- Mistral (Mistral AI)
- Mixtral (Mistral AI)
- Qwen (Alibaba)
- Gemma (Google)
- Phi (Microsoft)
- Falcon (Technology Innovation Institute)
- MPT (MosaicML)
- Yi (01.AI)
- DeepSeek (DeepSeek AI)
- OpenChat (OpenChat)
- Zephyr (HuggingFace)

### Vision Models
- CLIP (OpenAI)
- BLIP (Salesforce)
- LLaVA (Microsoft)

### Multimodal Models
- LLaVA (Microsoft)
- InstructBLIP (Salesforce)

## 🏗️ Arquitectura

### Python Layer (Orquestación)
- **Librerías**: PyTorch, Transformers, vLLM, TensorRT-LLM, HuggingFace
- **Responsabilidad**: Carga de modelos, orquestación de benchmarks, análisis de resultados

### Rust Layer (Alto Rendimiento)
- **Librerías**: Candle, Candle-nn, Tokenizers
- **Responsabilidad**: Inferencia optimizada, cálculo de métricas, procesamiento de datos

### Go Layer (Concurrencia)
- **Librerías**: Goroutines, Channels, gRPC
- **Responsabilidad**: Workers concurrentes, scheduling, API REST

### C++ Layer (Optimizaciones)
- **Librerías**: CUDA, cuBLAS, cuDNN, OpenCL
- **Responsabilidad**: Kernels optimizados, operaciones de bajo nivel

### TypeScript Layer (Interfaz)
- **Librerías**: Next.js, React, TypeScript, TailwindCSS
- **Responsabilidad**: Dashboard web, visualización de resultados, API

## 📈 Métricas Reportadas

- **Latency**: Tiempo de inferencia (p50, p95, p99)
- **Throughput**: Tokens por segundo
- **Memory**: Uso de memoria (GPU/CPU)
- **Accuracy**: Precisión en benchmarks
- **Cost**: Costo estimado por inferencia
- **Energy**: Consumo energético

## 🔧 Configuración

Ver `config/example.yaml` para configuración completa.

```yaml
models:
  - name: llama2-7b
    path: meta-llama/Llama-2-7b-hf
    quantization: fp16
    
benchmarks:
  - name: mmlu
    dataset: mmlu
    shots: 5
    
execution:
  workers: 4
  batch_size: 32
  device: cuda
```

## 📝 Ejemplos

Ver `examples/` para ejemplos completos de uso.

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor lee `CONTRIBUTING.md` para más detalles.

## 📄 Licencia

MIT License

## 🙏 Agradecimientos

Este proyecto utiliza las mejores librerías open source de la comunidad.












