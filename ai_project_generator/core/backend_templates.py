"""
Backend Templates - Templates para generación de código backend
==============================================================

Centraliza todos los templates de código para la generación de backend,
reduciendo duplicación y mejorando mantenibilidad.
"""

from typing import Dict, Any


class BackendTemplates:
    """Templates para generación de código backend"""
    
    @staticmethod
    def main_py(project_info: Dict[str, Any], description: str, keywords: Dict[str, Any]) -> str:
        """Template para main.py"""
        imports = [
            "from fastapi import FastAPI",
            "from fastapi.middleware.cors import CORSMiddleware",
            "from fastapi.responses import JSONResponse",
            "import uvicorn",
            "",
            "from app.api import router as api_router",
            "from app.core.config import settings",
        ]
        
        if keywords.get("requires_websocket"):
            imports.insert(1, "from fastapi import WebSocket, WebSocketDisconnect")
            imports.append("from app.core.websocket_manager import ConnectionManager")
        
        websocket_manager = "manager = ConnectionManager()" if keywords.get("requires_websocket") else "# WebSocket no requerido"
        
        return f'''"""
{project_info['name'].replace('_', ' ').title()} - Backend API
{'=' * 60}

{description}
"""

{chr(10).join(imports)}

app = FastAPI(
    title="{project_info['name'].replace('_', ' ').title()} API",
    description="{description}",
    version="{project_info['version']}",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

{websocket_manager}

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {{
        "status": "ok",
        "message": "{project_info['name'].replace('_', ' ').title()} API",
        "version": "{project_info['version']}",
        "features": {keywords.get('features', [])},
        "ai_type": "{keywords.get('ai_type', 'general')}"
    }}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {{
        "status": "healthy",
        "service": "{project_info['name']}",
        "version": "{project_info['version']}"
    }}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
'''
    
    @staticmethod
    def config_py(keywords: Dict[str, Any]) -> str:
        """Template para app/core/config.py"""
        config_imports = ["from pydantic_settings import BaseSettings", "from typing import Optional"]
        
        if keywords.get("requires_pytorch") or keywords.get("is_deep_learning"):
            config_imports.append("import torch")
        
        config_fields = [
            "# App",
            "APP_NAME: str = \"AI Project\"",
            "APP_VERSION: str = \"1.0.0\"",
            "DEBUG: bool = True",
            "",
            "# API",
            "API_V1_PREFIX: str = \"/api/v1\"",
            "",
            "# CORS",
            "CORS_ORIGINS: list = [\"*\"]",
            "",
            "# Database",
            "DATABASE_URL: Optional[str] = None",
            "",
            "# AI/ML Settings",
            "OPENAI_API_KEY: Optional[str] = None",
            "ANTHROPIC_API_KEY: Optional[str] = None",
        ]
        
        if keywords.get("requires_pytorch") or keywords.get("is_deep_learning"):
            config_fields.extend([
                "",
                "# Deep Learning Settings",
                "DEVICE: str = \"cuda\" if torch.cuda.is_available() else \"cpu\"",
                "MODEL_PATH: Optional[str] = None",
                "BATCH_SIZE: int = 32",
                "LEARNING_RATE: float = 2e-5",
                "NUM_EPOCHS: int = 3",
                "GRADIENT_ACCUMULATION_STEPS: int = 1",
                "MAX_GRAD_NORM: float = 1.0",
                "USE_MIXED_PRECISION: bool = torch.cuda.is_available()",
                "SAVE_CHECKPOINTS: bool = True",
                "CHECKPOINT_DIR: str = \"./checkpoints\"",
            ])
        
        if keywords.get("is_transformer") or keywords.get("is_llm"):
            config_fields.extend([
                "",
                "# Transformer/LLM Settings",
                "MODEL_NAME: str = \"bert-base-uncased\"",
                "MAX_LENGTH: int = 512",
                "NUM_LABELS: Optional[int] = None",
                "TASK_TYPE: str = \"classification\"",
            ])
        
        if keywords.get("is_diffusion"):
            config_fields.extend([
                "",
                "# Diffusion Model Settings",
                "DIFFUSION_MODEL_ID: str = \"runwayml/stable-diffusion-v1-5\"",
                "USE_XL: bool = False",
                "NUM_INFERENCE_STEPS: int = 50",
                "GUIDANCE_SCALE: float = 7.5",
                "IMAGE_HEIGHT: int = 512",
                "IMAGE_WIDTH: int = 512",
            ])
        
        return f'''"""Configuration settings"""

{chr(10).join(config_imports)}


class Settings(BaseSettings):
    """Application settings"""
    
{chr(10).join("    " + field for field in config_fields)}
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
'''
    
    @staticmethod
    def api_init_py(keywords: Dict[str, Any]) -> str:
        """Template para app/api/__init__.py"""
        api_routes = ["from fastapi import APIRouter", "from .endpoints import ai"]
        
        if keywords.get("requires_websocket"):
            api_routes.append("from .endpoints import websocket")
        
        if keywords.get("requires_file_upload"):
            api_routes.append("from .endpoints import upload")
        
        api_routes.extend([
            "",
            "router = APIRouter()",
            "",
            "router.include_router(ai.router, prefix=\"/ai\", tags=[\"AI\"])",
        ])
        
        if keywords.get("requires_websocket"):
            api_routes.append('router.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])')
        
        if keywords.get("requires_file_upload"):
            api_routes.append('router.include_router(upload.router, prefix="/upload", tags=["Upload"])')
        
        return "\n".join(api_routes)
    
    @staticmethod
    def ai_endpoint_py(project_info: Dict[str, Any], description: str, keywords: Dict[str, Any]) -> str:
        """Template para app/api/endpoints/ai.py"""
        return f'''"""AI endpoints"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class AIRequest(BaseModel):
    """Request model for AI operations"""
    prompt: str
    context: Optional[dict] = None
    parameters: Optional[dict] = None


class AIResponse(BaseModel):
    """Response model for AI operations"""
    result: str
    metadata: Optional[dict] = None


@router.post("/process", response_model=AIResponse)
async def process_ai(request: AIRequest):
    """
    Procesa una solicitud de IA
    
    {description}
    """
    try:
        result = f"Procesado: {{request.prompt}}"
        
        return AIResponse(
            result=result,
            metadata={{"model": "default", "tokens": len(request.prompt)}}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status():
    """Obtiene el estado del servicio de IA"""
    return {{
        "status": "active",
        "service": "{project_info['name']}",
        "type": "{keywords.get('ai_type', 'general')}"
    }}
'''
    
    @staticmethod
    def ai_service_py(keywords: Dict[str, Any], project_info: Dict[str, Any]) -> str:
        """Template para app/services/ai_service.py"""
        if keywords.get("is_deep_learning") or keywords.get("requires_pytorch"):
            ai_service_imports = []
            model_class = "CustomModel"
            
            if keywords.get("is_diffusion"):
                ai_service_imports.append("from app.models.diffusion_model import DiffusionModel")
                model_class = "DiffusionModel"
            elif keywords.get("is_transformer") or keywords.get("is_llm"):
                ai_service_imports.append("from app.models.transformer_model import TransformerModel")
                model_class = "TransformerModel"
            elif keywords.get("is_deep_learning"):
                ai_service_imports.append("from app.models.custom_model import CustomModel")
                model_class = "CustomModel"
            
            return f'''"""AI Service - Lógica de negocio para IA con Deep Learning"""

import logging
import torch
from typing import Dict, Any, Optional
from pathlib import Path

{chr(10).join(ai_service_imports)}

logger = logging.getLogger(__name__)


class AIService:
    """Servicio principal de IA con soporte para Deep Learning"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Inicializa el servicio de IA"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Usando dispositivo: {{self.device}}")
        
        try:
            if "{model_class}" == "DiffusionModel":
                self.model = DiffusionModel(device=self.device)
            elif "{model_class}" == "TransformerModel":
                task_type = "{keywords.get('ai_type', 'classification')}"
                num_labels = {keywords.get('num_labels', 'None')}
                self.model = TransformerModel(
                    task_type=task_type,
                    num_labels=num_labels if num_labels else None,
                    device=self.device,
                )
            elif "{model_class}" == "CustomModel":
                self.model = CustomModel(
                    input_size=768,
                    num_classes=10,
                ).to(self.device)
            else:
                self.model = None
            
            if model_path and Path(model_path).exists():
                if hasattr(self.model, 'load'):
                    self.model.load(model_path)
                else:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Modelo cargado desde {{model_path}}")
            else:
                logger.info("Modelo inicializado (sin pesos pre-entrenados)")
        
        except Exception as e:
            logger.error(f"Error inicializando modelo: {{e}}")
            self.model = None
    
    async def process(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Procesa un prompt usando IA"""
        if self.model is None:
            return {{
                "error": "Modelo no inicializado",
                "output": None
            }}
        
        try:
            logger.info(f"Procesando prompt: {{prompt[:50]}}...")
            
            if hasattr(self.model, 'predict'):
                result = self.model.predict(prompt)
            elif hasattr(self.model, 'generate'):
                images_or_text = self.model.generate(prompt)
                result = {{"output": images_or_text}}
            else:
                result = {{"output": f"Procesado: {{prompt}}"}}
            
            return {{
                "output": result,
                "metadata": {{
                    "model": "{model_class}",
                    "device": self.device,
                    "model_loaded": self.model is not None
                }}
            }}
        
        except Exception as e:
            logger.error(f"Error procesando prompt: {{e}}", exc_info=True)
            return {{
                "error": str(e),
                "output": None
            }}
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica el estado del servicio"""
        return {{
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "device": self.device,
            "model_type": "{model_class}",
            "cuda_available": torch.cuda.is_available(),
        }}
'''
        else:
            return '''"""AI Service - Lógica de negocio para IA"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AIService:
    """Servicio principal de IA"""
    
    def __init__(self):
        """Inicializa el servicio de IA"""
        self.model = None
    
    async def process(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Procesa un prompt usando IA"""
        logger.info(f"Procesando prompt: {prompt[:50]}...")
        
        result = {
            "output": f"Resultado para: {prompt}",
            "confidence": 0.95,
            "metadata": {
                "model": "default",
                "tokens": len(prompt)
            }
        }
        
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica el estado del servicio"""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None
        }
'''
    
    @staticmethod
    def dockerfile() -> str:
        """Template para Dockerfile"""
        return '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    @staticmethod
    def env_example(keywords: Dict[str, Any]) -> str:
        """Template para .env.example"""
        env_lines = [
            "# App Configuration",
            "DEBUG=True",
            "APP_NAME=AI Project",
            "",
            "# API Keys",
            "OPENAI_API_KEY=your_openai_api_key_here",
            "ANTHROPIC_API_KEY=your_anthropic_api_key_here",
            "",
            "# Database (opcional)",
            "# DATABASE_URL=postgresql://user:password@localhost/dbname",
        ]
        
        if keywords.get("is_deep_learning") or keywords.get("requires_pytorch"):
            env_lines.extend([
                "",
                "# Deep Learning Configuration",
                "DEVICE=cuda",
                "MODEL_PATH=./models/checkpoint.pt",
                "BATCH_SIZE=32",
                "LEARNING_RATE=2.0e-5",
                "NUM_EPOCHS=3",
                "USE_MIXED_PRECISION=true",
                "",
                "# Experiment Tracking",
                "WANDB_API_KEY=your_wandb_api_key_here",
                "WANDB_PROJECT=your_project_name",
                "TENSORBOARD_LOG_DIR=./logs",
            ])
        
        if keywords.get("is_transformer") or keywords.get("is_llm"):
            env_lines.extend([
                "",
                "# Transformer/LLM Configuration",
                "MODEL_NAME=bert-base-uncased",
                "MAX_LENGTH=512",
                "USE_LORA=false",
                "LORA_R=8",
                "LORA_ALPHA=16",
            ])
        
        if keywords.get("is_diffusion"):
            env_lines.extend([
                "",
                "# Diffusion Model Configuration",
                "DIFFUSION_MODEL_ID=runwayml/stable-diffusion-v1-5",
                "USE_XL=false",
                "NUM_INFERENCE_STEPS=50",
                "GUIDANCE_SCALE=7.5",
            ])
        
        if keywords.get("requires_gradio"):
            env_lines.extend([
                "",
                "# Gradio Configuration",
                "GRADIO_SERVER_NAME=0.0.0.0",
                "GRADIO_SERVER_PORT=7860",
                "GRADIO_SHARE=false",
            ])
        
        return "\n".join(env_lines) + "\n"

