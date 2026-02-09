"""
Model Serving
REST API and serving utilities for model deployment
"""

import torch
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io

from ..inference import InferencePipeline
from ..utils import validate_image_tensor

logger = logging.getLogger(__name__)


class ModelServer:
    """
    REST API server for model serving
    """
    
    def __init__(
        self,
        inference_pipeline: InferencePipeline,
        app_name: str = "MobileNet Model Server",
    ):
        """
        Initialize model server
        
        Args:
            inference_pipeline: Inference pipeline instance
            app_name: Application name
        """
        self.pipeline = inference_pipeline
        self.app = FastAPI(title=app_name)
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "MobileNet Model Server",
                "status": "running",
                "endpoints": {
                    "predict": "/predict",
                    "predict_batch": "/predict/batch",
                    "health": "/health",
                }
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "model_loaded": self.pipeline.model is not None,
            }
        
        @self.app.post("/predict")
        async def predict(
            file: UploadFile = File(...),
            top_k: int = 5,
            return_probabilities: bool = True,
        ):
            """
            Predict on single image
            
            Args:
                file: Image file
                top_k: Number of top predictions
                return_probabilities: Return probabilities
                
            Returns:
                Prediction results
            """
            try:
                # Read image
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Predict
                result = self.pipeline.predict(
                    image,
                    return_probabilities=return_probabilities,
                    top_k=top_k
                )
                
                return JSONResponse(content=result)
            except Exception as e:
                logger.error(f"Error in prediction: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/batch")
        async def predict_batch(
            files: List[UploadFile] = File(...),
        ):
            """
            Predict on batch of images
            
            Args:
                files: List of image files
                
            Returns:
                List of prediction results
            """
            try:
                images = []
                for file in files:
                    image_bytes = await file.read()
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    images.append(image)
                
                results = self.pipeline.predict_batch(images)
                return JSONResponse(content=results)
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/tensor")
        async def predict_tensor(
            tensor_data: Dict[str, Any],
        ):
            """
            Predict on tensor data
            
            Args:
                tensor_data: Tensor data as dictionary
                
            Returns:
                Prediction results
            """
            try:
                # Convert dict to tensor
                array = np.array(tensor_data['data'])
                shape = tuple(tensor_data.get('shape', array.shape))
                array = array.reshape(shape)
                tensor = torch.from_numpy(array).float()
                
                # Validate
                tensor = validate_image_tensor(tensor)
                
                # Predict
                result = self.pipeline.predictor.predict(tensor)
                return JSONResponse(content=result)
            except Exception as e:
                logger.error(f"Error in tensor prediction: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
    
    def get_app(self) -> FastAPI:
        """Get FastAPI app"""
        return self.app
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        **kwargs
    ) -> None:
        """
        Run server
        
        Args:
            host: Server host
            port: Server port
            **kwargs: Additional uvicorn arguments
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, **kwargs)



