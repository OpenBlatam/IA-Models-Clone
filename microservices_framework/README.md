# Advanced FastAPI Microservices Framework with Deep Learning & LLM Support

A production-ready microservices framework with integrated deep learning capabilities, supporting transformers, diffusion models, and LLM development using PyTorch, Transformers, Diffusers, and Gradio.

> Part of the [Blatam Academy Integrated Platform](../README.md)


## 🏗️ Architecture Overview

### Core Principles
- **Stateless Services**: All services are designed to be stateless with external storage for persistence
- **API Gateway Integration**: Centralized routing, rate limiting, and security
- **Circuit Breakers**: Resilient service communication with automatic failure handling
- **Deep Learning Integration**: Native support for PyTorch, Transformers, and Diffusion models
- **GPU Optimization**: Automatic GPU detection and mixed precision training
- **Event-Driven Architecture**: Inter-service communication using message brokers

### Key Components

1. **LLM Service** (`services/llm_service/`)
   - Text generation with transformer models (GPT, T5, BART, etc.)
   - Text embeddings generation
   - Model caching and memory management
   - Support for causal and seq2seq models

2. **Diffusion Service** (`services/diffusion_service/`)
   - Text-to-image generation with Stable Diffusion
   - Image-to-image transformation
   - Inpainting capabilities
   - Support for SDXL and custom models

3. **Training Service** (`services/training_service/`)
   - Model fine-tuning and training
   - LoRA (Low-Rank Adaptation) support
   - Background job management
   - Training progress tracking

4. **Gradio Service** (`services/gradio_service/`)
   - Interactive web interfaces for model inference
   - Text generation demos
   - Image generation demos
   - Embeddings visualization

5. **Shared ML Utilities** (`shared/ml/`)
   - Model utilities (loading, saving, optimization)
   - Data utilities (datasets, loaders, preprocessing)
   - Common ML operations

6. **API Gateway** (`gateway/`)
   - Request routing and load balancing
   - Rate limiting and throttling
   - Authentication and authorization

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Starting Services

```bash
# Start LLM Service (port 8001)
python services/llm_service/main.py

# Start Diffusion Service (port 8002)
python services/diffusion_service/main.py

# Start Training Service (port 8003)
python services/training_service/main.py

# Start Gradio Service (port 8004)
python services/gradio_service/main.py

# Start API Gateway (port 8000)
python gateway/api_gateway.py
```

### Docker Compose (Recommended)

```bash
docker-compose up -d
```

## 📁 Project Structure

```
microservices_framework/
├── gateway/                    # API Gateway implementation
│   └── api_gateway.py
├── services/                   # Individual microservices
│   ├── llm_service/            # Transformer & LLM inference
│   │   └── main.py
│   ├── diffusion_service/      # Stable Diffusion & image generation
│   │   └── main.py
│   ├── training_service/       # Model training & fine-tuning
│   │   └── main.py
│   ├── gradio_service/         # Interactive Gradio interfaces
│   │   └── main.py
│   └── user_service/           # User management (example)
│       └── main.py
├── shared/                     # Shared libraries and utilities
│   └── ml/                     # ML utilities
│       ├── model_utils.py      # Model loading, saving, optimization
│       ├── data_utils.py       # Data loading and preprocessing
│       └── __init__.py
├── tests/                      # Test suites
├── deployment/                 # Deployment configurations
└── requirements.txt           # Python dependencies
```

## 🔧 Usage Examples

### Text Generation with LLM Service

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8001/generate",
        json={
            "prompt": "The future of artificial intelligence",
            "model_name": "gpt2",
            "max_length": 100,
            "temperature": 0.8,
            "top_p": 0.9,
        }
    )
    result = response.json()
    print(result["generated_text"])
```

### Image Generation with Diffusion Service

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8002/text-to-image",
        json={
            "prompt": "A beautiful landscape with mountains and lakes",
            "model_name": "runwayml/stable-diffusion-v1-5",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
        }
    )
    # Save image
    with open("generated_image.png", "wb") as f:
        f.write(response.content)
```

### Model Training

```python
import httpx

async with httpx.AsyncClient() as client:
    # Start training job
    response = await client.post(
        "http://localhost:8003/train",
        json={
            "model_name": "gpt2",
            "task_type": "causal_lm",
            "dataset_path": "wikitext",
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
        }
    )
    job = response.json()
    job_id = job["job_id"]
    
    # Check training status
    status_response = await client.get(f"http://localhost:8003/jobs/{job_id}/status")
    status = status_response.json()
    print(f"Progress: {status['progress']*100:.1f}%")
```

### Interactive Gradio Interface

```python
import httpx

# Launch text generation interface
response = await httpx.post(
    "http://localhost:8004/interfaces/text-generation",
    json={
        "model_name": "gpt2",
        "port": 7860,
    }
)
print(f"Interface available at: {response.json()['url']}")
```

## 🧠 Deep Learning Features

### Supported Model Types

1. **Causal Language Models (GPT-style)**
   - GPT-2, GPT-Neo, GPT-J
   - LLaMA, Mistral, Phi
   - Custom fine-tuned models

2. **Seq2Seq Models**
   - T5, BART, mT5
   - Translation and summarization models

3. **Diffusion Models**
   - Stable Diffusion v1.5, v2.x
   - Stable Diffusion XL
   - Custom diffusion models

4. **Encoder Models (Embeddings)**
   - BERT, RoBERTa
   - Sentence Transformers
   - Custom encoder models

### Training Capabilities

- **Full Fine-tuning**: Complete model training
- **LoRA**: Parameter-efficient fine-tuning
- **Mixed Precision**: FP16 training for faster training
- **Gradient Accumulation**: Support for large effective batch sizes
- **Progress Tracking**: Real-time training metrics

### Optimization Features

- **Model Caching**: Automatic model caching to reduce load times
- **GPU Memory Management**: Efficient GPU memory usage
- **Batch Processing**: Optimized batch inference
- **Quantization**: Support for model quantization (via bitsandbytes)

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Service URLs
LLM_SERVICE_URL=http://localhost:8001
DIFFUSION_SERVICE_URL=http://localhost:8002
TRAINING_SERVICE_URL=http://localhost:8003
GRADIO_SERVICE_URL=http://localhost:8004

# GPU Settings
CUDA_VISIBLE_DEVICES=0
USE_FP16=true

# Model Cache
MODEL_CACHE_DIR=./models
MAX_CACHED_MODELS=5

# Training
OUTPUT_DIR=./trained_models
WANDB_API_KEY=your_wandb_key  # Optional
```

### Service Configuration

```python
from pydantic_settings import BaseSettings

class LLMServiceSettings(BaseSettings):
    service_name: str = "llm_service"
    port: int = 8001
    device: str = "cuda"  # or "cpu", "mps"
    use_fp16: bool = True
    max_cached_models: int = 5
    
    class Config:
        env_file = ".env"
```

## 📊 Monitoring & Observability

- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **OpenTelemetry**: Distributed tracing
- **Structured Logging**: JSON logging with structlog
- **Training Metrics**: Real-time training progress tracking

## 🔒 Security

- OAuth2 with JWT tokens
- Rate limiting and DDoS protection
- Security headers (CORS, CSP, HSTS)
- Input validation and sanitization
- Model access control

## 🚀 Performance Optimization

### GPU Utilization

- Automatic GPU detection
- Mixed precision inference (FP16)
- Memory-efficient attention (xformers)
- Model quantization support

### Caching Strategy

- Model caching in memory
- Configurable cache size
- Automatic cache eviction
- Redis support for distributed caching

### Batch Processing

- Efficient batch inference
- Dynamic batching
- Async request handling

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=services --cov=shared

# Run specific service tests
pytest tests/test_llm_service.py
```

## 📚 Best Practices

### Model Development

1. **Start with Pre-trained Models**: Use HuggingFace models as base
2. **Use LoRA for Fine-tuning**: More efficient than full fine-tuning
3. **Monitor Training**: Use TensorBoard or Weights & Biases
4. **Validate Inputs**: Always validate and sanitize inputs
5. **Handle Errors Gracefully**: Implement proper error handling

### Service Development

1. **Use Async/Await**: For I/O-bound operations
2. **Implement Caching**: Cache models and frequent requests
3. **Monitor Resources**: Track GPU/CPU memory usage
4. **Log Everything**: Use structured logging
5. **Test Thoroughly**: Write unit and integration tests

### Deployment

1. **Use Docker**: Containerize services for easy deployment
2. **GPU Support**: Use nvidia-docker for GPU services
3. **Scaling**: Scale services based on load
4. **Health Checks**: Implement proper health check endpoints
5. **Monitoring**: Set up comprehensive monitoring

## 🌐 Serverless Deployment

Optimized for:
- **AWS Lambda**: With Mangum adapter
- **Azure Functions**: Python runtime
- **Google Cloud Functions**: Python 3.11+
- **Kubernetes**: With GPU node support

## 📖 API Documentation

Once services are running, access interactive API documentation:

- LLM Service: http://localhost:8001/docs
- Diffusion Service: http://localhost:8002/docs
- Training Service: http://localhost:8003/docs
- Gradio Service: http://localhost:8004/docs

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Write comprehensive tests
3. Update documentation
4. Use type hints
5. Write clear commit messages

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace transformers library
- **Diffusers**: HuggingFace diffusers library
- **Gradio**: Interactive ML interfaces
- **FastAPI**: Modern web framework

---

**Happy coding with Deep Learning! 🚀🧠**
