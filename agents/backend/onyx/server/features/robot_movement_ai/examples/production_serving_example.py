"""
Production Serving Example
==========================

Ejemplo de serving de modelo en producción.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.routing_models import ModelFactory, ModelConfig
from core.routing_optimization import (
    compile_model,
    ModelServer,
    ServingConfig,
    BatchInferencePipeline,
    AsyncInferenceServer,
    GPUOptimizer
)


def example_model_server():
    """Ejemplo de servidor de modelo."""
    logger.info("=== Model Server ===")
    
    # Crear y compilar modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    model = compile_model(model, method="torch_compile")
    
    # Crear servidor
    serving_config = ServingConfig(
        batch_size=32,
        use_cache=True,
        cache_size=1000
    )
    server = ModelServer(model, serving_config)
    
    # Simular requests
    for i in range(10):
        input_data = torch.randn(20)
        output = server.predict(input_data)
        logger.info(f"Request {i+1}: Output shape {output.shape}")
    
    # Estadísticas
    stats = server.get_stats()
    logger.info(f"Estadísticas: {stats}")


def example_batch_pipeline():
    """Ejemplo de pipeline con batching."""
    logger.info("=== Batch Inference Pipeline ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    model = compile_model(model)
    
    # Crear pipeline
    pipeline = BatchInferencePipeline(model, batch_size=8, max_wait_time=0.01)
    pipeline.start()
    
    # Simular requests
    async def run_requests():
        tasks = []
        for i in range(20):
            input_data = torch.randn(20)
            task = pipeline.predict_async(input_data, request_id=f"req_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        logger.info(f"Procesados {len(results)} requests")
        
        pipeline.stop()
    
    asyncio.run(run_requests())


def example_gpu_optimization():
    """Ejemplo de optimización GPU."""
    logger.info("=== GPU Optimization ===")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA no disponible")
        return
    
    # Habilitar optimizaciones
    GPUOptimizer.enable_all_optimizations()
    
    # Información de GPU
    info = GPUOptimizer.get_gpu_info()
    logger.info(f"GPU Info: {info}")
    
    # Optimizar modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    model = GPUOptimizer.optimize_model_for_gpu(model)
    
    logger.info("Modelo optimizado para GPU")


def example_fastapi_server():
    """Ejemplo de servidor FastAPI."""
    logger.info("=== FastAPI Server ===")
    
    try:
        from core.routing_optimization.model_serving import create_fastapi_server
        
        # Crear modelo
        config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
        model = ModelFactory.create_model("mlp", config)
        model = compile_model(model)
        
        # Crear servidor FastAPI
        app = create_fastapi_server(model)
        
        logger.info("Servidor FastAPI creado")
        logger.info("Ejecutar con: uvicorn examples.production_serving_example:app")
        
        return app
    except ImportError:
        logger.warning("FastAPI no disponible")


def main():
    """Ejecutar ejemplos."""
    logger.info("Ejecutando ejemplos de serving en producción...\n")
    
    example_model_server()
    print()
    
    example_batch_pipeline()
    print()
    
    example_gpu_optimization()
    print()
    
    # example_fastapi_server()  # Descomentar para crear servidor
    
    logger.info("=== Ejemplos completados ===")


if __name__ == "__main__":
    main()

