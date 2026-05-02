# Universal Model Benchmark AI

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![Version](https://img.shields.io/badge/version-2.4-blue.svg)
![Polyglot](https://img.shields.io/badge/Polyglot-Python%20%7C%20Rust%20%7C%20Go%20%7C%20C%2B%2B-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Benchmarks](https://img.shields.io/badge/Benchmarks-MMLU%20%7C%20HumanEval%20%7C%20GSM8K-purple.svg)

**A high-performance, polyglot benchmarking suite for comprehensive evaluation of LLMs, Vision Models, and Multimodal Systems.**

[Overview](#-overview) •
[Features](#-key-features) •
[Architecture](#-architecture) •
[Benchmarks](#-supported-benchmarks) •
[Installation](#-installation) •
[Usage](#-usage) •
[Contributing](#-contributing)

</div>

---

## 📋 Overview

**Universal Model Benchmark AI** is the definitive toolkit for evaluating Artificial Intelligence models. Designed for researchers and ML engineers, it provides a unified interface to test models against industry-standard benchmarks like MMLU, HumanEval, and GSM8K.

What sets this system apart is its **polyglot architecture**:
- **Python** for orchestration and ML framework integration.
- **Rust** for high-performance tokenization and metric calculation.
- **Go** for concurrent scheduling and distributed worker management.
- **C++** for low-level CUDA kernel optimizations.

### Why Universal Benchmark?

- **Holistic Evaluation**: Test accuracy, latency, throughput, energy consumption, and VRAM usage simultaneously.
- **Fair Comparison**: Standardized prompts and evaluation protocols ensure apples-to-apples comparisons.
- **Extensible**: Plugin system allows adding new models and datasets in minutes.

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Framework** | Support for PyTorch, TensorFlow, JAX, ONNX Runtime, and TensorRT-LLM. |
| **Comprehensive Metrics** | precise measurement of time-to-first-token (TTFT), inter-token latency, and total throughput. |
| **Distributed Testing** | Run benchmarks across massive GPU clusters using Ray or Slurm. |
| **Live Dashboard** | Real-time visualization of benchmark progress and results via a React/Next.js dashboard. |
| **Auto-Quantization** | Test performance impacts of different quantization levels (FP16, INT8, FP4) on the fly. |

## 🏗 Architecture

The system leverages the best tool for each layer of the stack.

```mermaid
graph TD
    A[User Config] --> B(Python Orchestrator)
    
    subgraph "Scheduling Layer (Go)"
    B --> C{Task Scheduler}
    C --> D[Worker Pool]
    end
    
    subgraph "Execution Layer (Python/C++)"
    D --> E[Model Loader]
    E --> F[Inference Engine]
    F --> G[CUDA Kernels]
    end
    
    subgraph "Analysis Layer (Rust)"
    F --> H[Output Stream]
    H --> I[Metric Calculator]
    I --> J[Result Aggregator]
    end
    
    J --> K[Dashboard (TypeScript)]
```

## 📊 Supported Benchmarks

### Language (LLM)
- **MMLU**: Massive Multitask Language Understanding (General Knowledge)
- **HumanEval**: Python Coding Capabilities
- **GSM8K**: Grade School Math & Logic
- **TruthfulQA**: Model safety and hallucination usage
- **HellaSwag**: Commonsense reasoning

### Vision & Multimodal
- **ImageNet**: Classification accuracy
- **COCO**: Object detection and segmentation
- **VQAv2**: Visual Question Answering
- **TextVQA**: text reading capabilities in images

## 💻 Installation

### Prerequisites

- Python 3.10+
- Rust 1.70+ (Cargo)
- Go 1.21+
- Node.js 18+
- CUDA 11.8+

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/blatam-academy/universal_model_benchmark_ai.git
   cd universal_model_benchmark_ai
   ```

2. **Install Polyglot Dependencies**
   ```bash
   # Python
   pip install -r python/requirements.txt
   
   # Rust
   cd rust && cargo build --release && cd ..
   
   # Go
   cd go && go build ./... && cd ..
   
   # UI
   cd typescript && npm install && cd ..
   ```

## ⚡ Usage

### Run a Full Benchmark Suite

```bash
python python/orchestrator/main.py \
    --model meta-llama/Llama-3-70b-hf \
    --benchmarks mmlu,humaneval,gsm8k \
    --shots 5 \
    --device cuda:0
```

### Compare Quantization Levels

```bash
python python/orchestrator/compare.py \
    --model mistralai/Mistral-7B-v0.1 \
    --levels fp16,int8,nf4 \
    --benchmark mmlu
```

### Start the Dashboard

```bash
# Terminal 1: API Server
./go/api_server

# Terminal 2: Frontend
cd typescript && npm start
```

Access the dashboard at `http://localhost:3000`.

## 📈 Example Results

| Model | MMLU (5-shot) | HumanEval | GSM8K | Tokens/s |
|-------|---------------|-----------|-------|----------|
| **Llama-3 70B** | 82.0% | 81.7% | 93.0% | 45 |
| **GPT-4** | 86.4% | 67.0% | 92.0% | -- |
| **Claude 3 Opus** | 86.8% | 84.9% | 95.0% | -- |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>Built with ❤️ by Blatam Academy</b><br>
  Part of the Onyx Server Architecture<br>
  <a href="../README.md">← Back to Main README</a>
</div>
