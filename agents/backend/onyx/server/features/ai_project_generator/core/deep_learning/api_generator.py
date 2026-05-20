"""
API Generator
=============

Generador de APIs REST y GraphQL para modelos de Deep Learning.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuración de API."""
    framework: str = "fastapi"  # 'fastapi', 'flask', 'graphql'
    model_path: str = "models/best_model.pth"
    port: int = 8000
    host: str = "0.0.0.0"
    enable_docs: bool = True
    enable_cors: bool = True
    batch_size: int = 1
    max_workers: int = 4


class APIGenerator:
    """
    Generador de APIs para modelos.
    """
    
    def __init__(self):
        """Inicializar generador."""
        pass
    
    def generate_fastapi(
        self,
        project_dir: Path,
        config: Optional[APIConfig] = None
    ) -> str:
        """
        Generar API FastAPI.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido del archivo API
        """
        if config is None:
            config = APIConfig()
        
        api_content = f""""""
FastAPI API para modelo de Deep Learning
==========================================

Generado automáticamente por DeepLearningGenerator
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
from pathlib import Path
import logging

from app.models import load_model, preprocess, postprocess

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deep Learning Model API",
    description="API para inferencia de modelo de Deep Learning",
    version="1.0.0"
)

# CORS
"""
        if config.enable_cors:
            api_content += """app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
        
        api_content += f"""
# Cargar modelo
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def load_model_on_startup():
    global model
    model_path = Path("{config.model_path}")
    if model_path.exists():
        model = load_model(model_path, device)
        logger.info(f"Modelo cargado desde {{model_path}}")
    else:
        logger.warning(f"Modelo no encontrado en {{model_path}}")

# Schemas
class PredictionRequest(BaseModel):
    data: List[float]
    batch_size: Optional[int] = {config.batch_size}

class PredictionResponse(BaseModel):
    prediction: List[float]
    confidence: Optional[List[float]] = None
    processing_time: float

# Endpoints
@app.get("/")
async def root():
    return {{"message": "Deep Learning Model API", "status": "running"}}

@app.get("/health")
async def health():
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.get("/ready")
async def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {{"status": "ready"}}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Preprocesar datos
        input_data = preprocess(request.data)
        
        # Inferencia
        with torch.no_grad():
            output = model(input_data)
        
        # Postprocesar
        prediction = postprocess(output)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=prediction.tolist(),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error en predicción: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Procesar batch
        predictions = []
        for request in requests:
            input_data = preprocess(request.data)
            with torch.no_grad():
                output = model(input_data)
            prediction = postprocess(output)
            predictions.append(prediction.tolist())
        
        processing_time = time.time() - start_time
        
        return {{
            "predictions": predictions,
            "batch_size": len(requests),
            "processing_time": processing_time
        }}
    except Exception as e:
        logger.error(f"Error en batch prediction: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="{config.host}", port={config.port})
"""
        
        return api_content
    
    def generate_flask(
        self,
        project_dir: Path,
        config: Optional[APIConfig] = None
    ) -> str:
        """
        Generar API Flask.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido del archivo API
        """
        if config is None:
            config = APIConfig(framework="flask")
        
        api_content = f""""""
Flask API para modelo de Deep Learning
========================================

Generado automáticamente por DeepLearningGenerator
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from pathlib import Path
import logging

from app.models import load_model, preprocess, postprocess

logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS
"""
        if config.enable_cors:
            api_content += """CORS(app)
"""
        
        api_content += f"""
# Cargar modelo
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_on_startup():
    global model
    model_path = Path("{config.model_path}")
    if model_path.exists():
        model = load_model(model_path, device)
        logger.info(f"Modelo cargado desde {{model_path}}")
    else:
        logger.warning(f"Modelo no encontrado en {{model_path}}")

load_model_on_startup()

# Endpoints
@app.route("/", methods=["GET"])
def root():
    return jsonify({{"message": "Deep Learning Model API", "status": "running"}})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({{"status": "healthy", "model_loaded": model is not None}})

@app.route("/ready", methods=["GET"])
def ready():
    if model is None:
        return jsonify({{"status": "not ready"}}), 503
    return jsonify({{"status": "ready"}})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({{"error": "Model not loaded"}}), 503
    
    try:
        import time
        start_time = time.time()
        
        data = request.json.get("data", [])
        batch_size = request.json.get("batch_size", {config.batch_size})
        
        # Preprocesar datos
        input_data = preprocess(data)
        
        # Inferencia
        with torch.no_grad():
            output = model(input_data)
        
        # Postprocesar
        prediction = postprocess(output)
        
        processing_time = time.time() - start_time
        
        return jsonify({{
            "prediction": prediction.tolist(),
            "processing_time": processing_time
        }})
    except Exception as e:
        logger.error(f"Error en predicción: {{e}}")
        return jsonify({{"error": str(e)}}), 500

if __name__ == "__main__":
    app.run(host="{config.host}", port={config.port}, debug=False)
"""
        
        return api_content
    
    def generate_all(
        self,
        project_dir: Path,
        config: Optional[APIConfig] = None
    ) -> Dict[str, str]:
        """
        Generar todas las APIs.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Diccionario con archivos generados
        """
        if config is None:
            config = APIConfig()
        
        files = {}
        api_dir = project_dir / "app" / "api"
        api_dir.mkdir(parents=True, exist_ok=True)
        
        if config.framework == "fastapi":
            api_content = self.generate_fastapi(project_dir, config)
            api_path = api_dir / "main.py"
            api_path.write_text(api_content, encoding='utf-8')
            files['app/api/main.py'] = api_content
        
        elif config.framework == "flask":
            api_content = self.generate_flask(project_dir, config)
            api_path = api_dir / "app.py"
            api_path.write_text(api_content, encoding='utf-8')
            files['app/api/app.py'] = api_content
        
        logger.info(f"API generada en {api_dir}")
        
        return files


# Instancia global
_global_api_generator: Optional[APIGenerator] = None


def get_api_generator() -> APIGenerator:
    """
    Obtener instancia global del generador de APIs.
    
    Returns:
        Instancia del generador
    """
    global _global_api_generator
    
    if _global_api_generator is None:
        _global_api_generator = APIGenerator()
    
    return _global_api_generator

