"""
Performance Optimization Example
================================

Ejemplo de optimizaciones de rendimiento.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.routing_models import ModelFactory, ModelConfig
from core.routing_optimization import (
    compile_model,
    optimize_for_inference,
    quantize_model,
    FastDataLoader,
    CachedDataset,
    benchmark_model
)
from core.routing_data import RouteDataset, RoutePreprocessor


def example_model_compilation():
    """Ejemplo de compilación de modelo."""
    logger.info("=== Compilación de Modelo ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Benchmark original
    logger.info("Benchmark modelo original...")
    original_metrics = benchmark_model(model, num_runs=100)
    logger.info(f"Tiempo promedio: {original_metrics['avg_inference_time_ms']:.2f} ms")
    
    # Compilar
    compiled_model = compile_model(model, method="torch_compile")
    
    # Benchmark compilado
    logger.info("Benchmark modelo compilado...")
    compiled_metrics = benchmark_model(compiled_model, num_runs=100)
    logger.info(f"Tiempo promedio: {compiled_metrics['avg_inference_time_ms']:.2f} ms")
    
    speedup = original_metrics['avg_inference_time_ms'] / compiled_metrics['avg_inference_time_ms']
    logger.info(f"Speedup: {speedup:.2f}x")


def example_quantization():
    """Ejemplo de cuantización."""
    logger.info("=== Cuantización ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Benchmark original
    original_metrics = benchmark_model(model, device="cpu", num_runs=100)
    logger.info(f"Modelo original (CPU): {original_metrics['avg_inference_time_ms']:.2f} ms")
    
    # Cuantizar
    quantized = quantize_model(model, method="dynamic")
    
    # Benchmark cuantizado
    quantized_metrics = benchmark_model(quantized.quantized_model, device="cpu", num_runs=100)
    logger.info(f"Modelo cuantizado (CPU): {quantized_metrics['avg_inference_time_ms']:.2f} ms")
    
    speedup = original_metrics['avg_inference_time_ms'] / quantized_metrics['avg_inference_time_ms']
    size_reduction = quantized.get_size_reduction()
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Reducción de tamaño: {size_reduction:.2%}")


def example_fast_data_loading():
    """Ejemplo de carga de datos rápida."""
    logger.info("=== Carga de Datos Rápida ===")
    
    # Generar datos
    features = [np.random.randn(20).astype(np.float32) for _ in range(1000)]
    targets = [np.random.randn(4).astype(np.float32) for _ in range(1000)]
    
    preprocessor = RoutePreprocessor()
    preprocessor.fit(features, targets)
    
    dataset = RouteDataset(features, targets, preprocessor=preprocessor)
    
    # Dataset con cache
    cached_dataset = CachedDataset(dataset, cache_size=500)
    
    # DataLoader estándar
    standard_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=0
    )
    
    # DataLoader rápido
    fast_loader = FastDataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    # Benchmark
    def benchmark_loader(loader, name):
        start = time.time()
        for batch in loader:
            pass
        elapsed = time.time() - start
        logger.info(f"{name}: {elapsed:.2f} segundos")
    
    benchmark_loader(standard_loader, "DataLoader estándar")
    benchmark_loader(fast_loader, "FastDataLoader")


def example_inference_optimization():
    """Ejemplo de optimización de inferencia."""
    logger.info("=== Optimización de Inferencia ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Optimizar
    optimized_model = optimize_for_inference(
        model,
        input_shape=(32, 20),  # Batch size 32
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Benchmark
    metrics = benchmark_model(optimized_model, input_shape=(32, 20), num_runs=100)
    logger.info(f"Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")


def main():
    """Ejecutar ejemplos."""
    logger.info("Ejecutando ejemplos de optimización de rendimiento...\n")
    
    if torch.cuda.is_available():
        example_model_compilation()
        print()
    
    example_quantization()
    print()
    
    example_fast_data_loading()
    print()
    
    if torch.cuda.is_available():
        example_inference_optimization()
        print()
    
    logger.info("=== Ejemplos completados ===")


if __name__ == "__main__":
    main()

