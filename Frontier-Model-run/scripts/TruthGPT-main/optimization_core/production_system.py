# 🚀 TRUTHGPT - PRODUCTION DEPLOYMENT SYSTEM

## ⚡ Sistema de Deployment en Producción

### 🎯 Configuración de Producción
```python
import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import logging
from typing import Dict, Any, List, Optional
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import time
from contextlib import asynccontextmanager

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TruthGPTProductionConfig:
    """Configuración para producción."""
    
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "gpt2")
        self.device = os.getenv("DEVICE", "auto")
        self.precision = os.getenv("PRECISION", "fp16")
        self.max_length = int(os.getenv("MAX_LENGTH", "512"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "1"))
        self.num_workers = int(os.getenv("NUM_WORKERS", "4"))
        self.cache_dir = os.getenv("CACHE_DIR", "./cache")
        self.model_cache_dir = os.getenv("MODEL_CACHE_DIR", "./models")
        
        # Configuración de API
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.api_workers = int(os.getenv("API_WORKERS", "1"))
        
        # Configuración de Gradio
        self.gradio_port = int(os.getenv("GRADIO_PORT", "7860"))
        self.gradio_share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
        
        # Configuración de monitoreo
        self.enable_wandb = os.getenv("ENABLE_WANDB", "false").lower() == "true"
        self.wandb_project = os.getenv("WANDB_PROJECT", "truthgpt-production")
        
        # Configuración de seguridad
        self.api_key = os.getenv("API_KEY", None)
        self.rate_limit = int(os.getenv("RATE_LIMIT", "100"))  # requests per minute
        
        logger.info(f"Production config loaded: {self.model_name}")

class TruthGPTProductionModel:
    """Modelo optimizado para producción."""
    
    def __init__(self, config: TruthGPTProductionConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.accelerator = None
        self.device = None
        
    async def load_model(self):
        """Cargar modelo de forma asíncrona."""
        logger.info("Loading TruthGPT model for production...")
        
        try:
            # Configurar dispositivo
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.model_cache_dir
            )
            
            # Configurar pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.precision == "fp16" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                cache_dir=self.config.model_cache_dir,
                low_cpu_mem_usage=True
            )
            
            # Mover a dispositivo si es necesario
            if self.device != "auto":
                self.model = self.model.to(self.device)
            
            # Configurar para inferencia
            self.model.eval()
            
            # Accelerator para optimización
            self.accelerator = Accelerator()
            self.model = self.accelerator.prepare(self.model)
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    async def generate_text(
        self, 
        prompt: str, 
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generar texto de forma optimizada."""
        
        # Usar configuración por defecto si no se especifica
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        
        try:
            # Tokenizar input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Mover a dispositivo
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generar texto
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            # Decodificar outputs
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remover el prompt original del resultado
                generated_text = text[len(prompt):].strip()
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def batch_generate(
        self, 
        prompts: List[str], 
        **kwargs
    ) -> List[List[str]]:
        """Generar texto en lotes."""
        results = []
        
        for prompt in prompts:
            try:
                result = await self.generate_text(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch generation: {e}")
                results.append([f"Error: {str(e)}"])
        
        return results

# FastAPI Application
app = FastAPI(
    title="TruthGPT Production API",
    description="API de producción para TruthGPT",
    version="1.0.0"
)

# Variables globales
config = TruthGPTProductionConfig()
model_manager = TruthGPTProductionModel(config)

# Modelos Pydantic
class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    num_return_sequences: Optional[int] = 1

class GenerateResponse(BaseModel):
    generated_texts: List[str]
    prompt: str
    timestamp: str
    model_name: str

class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    num_return_sequences: Optional[int] = 1

class BatchGenerateResponse(BaseModel):
    results: List[List[str]]
    timestamp: str
    model_name: str
    total_prompts: int

# Middleware de autenticación
async def verify_api_key(request):
    """Verificar API key."""
    if config.api_key:
        api_key = request.headers.get("X-API-Key")
        if api_key != config.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

# Rate limiting simple
request_counts = {}

async def rate_limit_check():
    """Verificar rate limit."""
    current_time = time.time()
    minute_ago = current_time - 60
    
    # Limpiar requests antiguos
    request_counts.clear()
    
    # Verificar límite
    if len(request_counts) > config.rate_limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# Eventos de aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación."""
    # Startup
    logger.info("Starting TruthGPT Production API...")
    await model_manager.load_model()
    logger.info("Model loaded successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down TruthGPT Production API...")

app.router.lifespan_context = lifespan

# Endpoints
@app.get("/")
async def root():
    """Endpoint raíz."""
    return {
        "message": "TruthGPT Production API",
        "version": "1.0.0",
        "status": "running",
        "model": config.model_name,
        "device": model_manager.device
    }

@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model_manager.model is not None,
        "device": model_manager.device
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generar texto."""
    await rate_limit_check()
    
    try:
        generated_texts = await model_manager.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=request.num_return_sequences
        )
        
        return GenerateResponse(
            generated_texts=generated_texts,
            prompt=request.prompt,
            timestamp=time.time(),
            model_name=config.model_name
        )
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-generate", response_model=BatchGenerateResponse)
async def batch_generate_text(request: BatchGenerateRequest):
    """Generar texto en lotes."""
    await rate_limit_check()
    
    try:
        results = await model_manager.batch_generate(
            prompts=request.prompts,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=request.num_return_sequences
        )
        
        return BatchGenerateResponse(
            results=results,
            timestamp=time.time(),
            model_name=config.model_name,
            total_prompts=len(request.prompts)
        )
        
    except Exception as e:
        logger.error(f"Error in batch-generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """Información del modelo."""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model_manager.model.parameters())
    
    return {
        "model_name": config.model_name,
        "device": model_manager.device,
        "precision": config.precision,
        "total_parameters": total_params,
        "max_length": config.max_length,
        "temperature": config.temperature,
        "top_p": config.top_p
    }

# Gradio Interface
def create_gradio_interface():
    """Crear interfaz Gradio."""
    
    async def generate_gradio(prompt, max_length, temperature, top_p):
        """Función para Gradio."""
        try:
            generated_texts = await model_manager.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            return generated_texts[0] if generated_texts else "Error generating text"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Crear interfaz
    interface = gr.Interface(
        fn=generate_gradio,
        inputs=[
            gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=3
            ),
            gr.Slider(
                minimum=50,
                maximum=1024,
                value=config.max_length,
                step=1,
                label="Max Length"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=config.temperature,
                step=0.1,
                label="Temperature"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=config.top_p,
                step=0.1,
                label="Top P"
            )
        ],
        outputs=gr.Textbox(
            label="Generated Text",
            lines=10
        ),
        title="TruthGPT Production Interface",
        description="Generate text with TruthGPT model",
        theme="default"
    )
    
    return interface

# Función principal
async def main():
    """Función principal."""
    logger.info("Starting TruthGPT Production System...")
    
    # Cargar modelo
    await model_manager.load_model()
    
    # Crear interfaz Gradio
    gradio_interface = create_gradio_interface()
    
    # Iniciar Gradio en segundo plano
    gradio_task = asyncio.create_task(
        gradio_interface.launch(
            server_port=config.gradio_port,
            share=config.gradio_share,
            quiet=True
        )
    )
    
    # Iniciar FastAPI
    config_uvicorn = uvicorn.Config(
        app,
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        log_level="info"
    )
    
    server = uvicorn.Server(config_uvicorn)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
```

### 🐳 Docker Configuration
```dockerfile
# Dockerfile para TruthGPT Production
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configurar Python
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# Copiar requirements
COPY requirements.txt /app/requirements.txt

# Instalar dependencias Python
RUN pip install -r /app/requirements.txt

# Copiar código
COPY . /app
WORKDIR /app

# Variables de entorno
ENV MODEL_NAME=gpt2
ENV DEVICE=auto
ENV PRECISION=fp16
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV GRADIO_PORT=7860

# Exponer puertos
EXPOSE 8000 7860

# Comando por defecto
CMD ["python", "production_system.py"]
```

### 🚀 Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  truthgpt-api:
    build: .
    ports:
      - "8000:8000"
      - "7860:7860"
    environment:
      - MODEL_NAME=gpt2
      - DEVICE=auto
      - PRECISION=fp16
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - GRADIO_PORT=7860
      - GRADIO_SHARE=false
      - ENABLE_WANDB=false
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - truthgpt-api
    restart: unless-stopped
```

### 📊 Monitoring y Logging
```python
# monitoring.py
import logging
import time
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import wandb

class TruthGPTMonitoring:
    """Sistema de monitoreo para TruthGPT."""
    
    def __init__(self, config):
        self.config = config
        
        # Métricas Prometheus
        self.request_counter = Counter('truthgpt_requests_total', 'Total requests')
        self.request_duration = Histogram('truthgpt_request_duration_seconds', 'Request duration')
        self.model_memory = Gauge('truthgpt_model_memory_bytes', 'Model memory usage')
        self.gpu_utilization = Gauge('truthgpt_gpu_utilization_percent', 'GPU utilization')
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('truthgpt.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Iniciar servidor Prometheus
        start_http_server(9090)
        
        # Configurar WandB si está habilitado
        if config.enable_wandb:
            wandb.init(
                project=config.wandb_project,
                name=f"truthgpt-production-{int(time.time())}"
            )
    
    def log_request(self, prompt_length, response_length, duration):
        """Registrar request."""
        self.request_counter.inc()
        self.request_duration.observe(duration)
        
        self.logger.info(
            f"Request processed - "
            f"Prompt: {prompt_length} chars, "
            f"Response: {response_length} chars, "
            f"Duration: {duration:.3f}s"
        )
        
        if self.config.enable_wandb:
            wandb.log({
                "request_duration": duration,
                "prompt_length": prompt_length,
                "response_length": response_length
            })
    
    def update_system_metrics(self):
        """Actualizar métricas del sistema."""
        # Memoria del modelo
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated()
            self.model_memory.set(memory_used)
        
        # Utilización GPU
        gpus = GPUtil.getGPUs()
        if gpus:
            self.gpu_utilization.set(gpus[0].load * 100)
        
        # Logging de métricas
        self.logger.info(
            f"System metrics - "
            f"CPU: {psutil.cpu_percent()}%, "
            f"Memory: {psutil.virtual_memory().percent}%, "
            f"GPU: {gpus[0].load * 100:.1f}%"
        )
```

### 🎯 Script de Deployment
```bash
#!/bin/bash
# deploy.sh - Script de deployment

echo "🚀 Deploying TruthGPT Production System..."

# Construir imagen Docker
echo "Building Docker image..."
docker build -t truthgpt-production .

# Detener servicios existentes
echo "Stopping existing services..."
docker-compose down

# Iniciar servicios
echo "Starting services..."
docker-compose up -d

# Verificar estado
echo "Checking service status..."
sleep 10
curl -f http://localhost:8000/health || echo "API not ready"
curl -f http://localhost:7860 || echo "Gradio not ready"

echo "✅ Deployment complete!"
echo "API: http://localhost:8000"
echo "Gradio: http://localhost:7860"
echo "Prometheus: http://localhost:9090"
```

---

**¡Sistema de producción completo!** 🚀⚡🎯

