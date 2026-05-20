# Quick Start Guide - Deep Learning Microservices Framework

This guide will help you get started with the framework in minutes.

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended for best performance)
- 16GB+ RAM (32GB+ recommended for large models)
- 50GB+ free disk space (for model downloads)

## Step 1: Installation

```bash
# Clone or navigate to the framework directory
cd microservices_framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check Diffusers
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
```

## Step 3: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional)
# nano .env
```

## Step 4: Start Services

### Option A: Start Services Individually

```bash
# Terminal 1: LLM Service
python services/llm_service/main.py

# Terminal 2: Diffusion Service (requires GPU for best performance)
python services/diffusion_service/main.py

# Terminal 3: Training Service
python services/training_service/main.py

# Terminal 4: Gradio Service
python services/gradio_service/main.py
```

### Option B: Use Docker Compose (Recommended)

```bash
docker-compose up -d
```

## Step 5: Test the Services

### Test LLM Service

```bash
# Using curl
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "model_name": "gpt2",
    "max_length": 50,
    "temperature": 0.8
  }'

# Or using Python
python -c "
import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'http://localhost:8001/generate',
            json={
                'prompt': 'The future of AI is',
                'model_name': 'gpt2',
                'max_length': 50
            }
        )
        print(response.json())

asyncio.run(test())
"
```

### Test Diffusion Service

```bash
# Using curl (saves image to file)
curl -X POST "http://localhost:8002/text-to-image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "num_inference_steps": 20,
    "width": 512,
    "height": 512
  }' \
  --output generated_image.png
```

### Launch Gradio Interface

```bash
# Using curl
curl -X POST "http://localhost:8004/interfaces/text-generation" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt2",
    "port": 7860
  }'

# Then open http://localhost:7860 in your browser
```

## Step 6: Explore API Documentation

Open in your browser:
- LLM Service: http://localhost:8001/docs
- Diffusion Service: http://localhost:8002/docs
- Training Service: http://localhost:8003/docs
- Gradio Service: http://localhost:8004/docs

## Common Use Cases

### 1. Text Generation

```python
import httpx
import asyncio

async def generate_text():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/generate",
            json={
                "prompt": "Once upon a time",
                "model_name": "gpt2",
                "max_length": 100,
                "temperature": 0.8,
            }
        )
        result = response.json()
        print(result["generated_text"])

asyncio.run(generate_text())
```

### 2. Image Generation

```python
import httpx
import asyncio

async def generate_image():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8002/text-to-image",
            json={
                "prompt": "A futuristic city at sunset",
                "model_name": "runwayml/stable-diffusion-v1-5",
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
            }
        )
        with open("generated.png", "wb") as f:
            f.write(response.content)
        print("Image saved to generated.png")

asyncio.run(generate_image())
```

### 3. Model Training

```python
import httpx
import asyncio
import time

async def train_model():
    async with httpx.AsyncClient() as client:
        # Start training
        response = await client.post(
            "http://localhost:8003/train",
            json={
                "model_name": "gpt2",
                "task_type": "causal_lm",
                "dataset_path": "wikitext",
                "num_epochs": 1,
                "batch_size": 4,
                "use_lora": True,
            }
        )
        job = response.json()
        job_id = job["job_id"]
        print(f"Training started: {job_id}")
        
        # Monitor progress
        while True:
            status_response = await client.get(
                f"http://localhost:8003/jobs/{job_id}/status"
            )
            status = status_response.json()
            print(f"Progress: {status['progress']*100:.1f}%")
            
            if status["status"] in ["completed", "failed"]:
                break
            time.sleep(5)

asyncio.run(train_model())
```

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Errors

- Reduce batch size
- Use smaller models
- Enable gradient checkpointing
- Use LoRA instead of full fine-tuning

### Model Download Issues

```bash
# Set HuggingFace cache directory
export HF_HOME=./models
export TRANSFORMERS_CACHE=./models
```

### Port Already in Use

```bash
# Find process using port
lsof -i :8001  # On Linux/Mac
netstat -ano | findstr :8001  # On Windows

# Kill process or change port in service configuration
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for integration patterns
3. Explore the example services in `services/` directory
4. Customize services for your use case

## Getting Help

- Check service logs for errors
- Review API documentation at `/docs` endpoints
- Check GitHub issues (if applicable)
- Review service health endpoints: `/health`

---

**Happy coding! 🚀**



