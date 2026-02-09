"""
Ultra-Fast Inference Example
=============================

Ejemplo de optimizaciones extremas para máxima velocidad.
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
    UltraFastInference,
    PrecompiledModel,
    StreamInference,
    AOTCompiler,
    DynamicBatching,
    optimize_kernels,
    benchmark_model
)


def example_ultra_fast_optimization():
    """Ejemplo de optimización ultra-rápida."""
    logger.info("=== Ultra-Fast Optimization ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Benchmark original
    logger.info("Benchmark modelo original...")
    original_metrics = benchmark_model(model, num_runs=100)
    logger.info(f"Tiempo promedio: {original_metrics['avg_inference_time_ms']:.2f} ms")
    
    # Optimización ultra-rápida
    optimizer = UltraFastInference(model, device="cuda" if torch.cuda.is_available() else "cpu")
    optimized_model = optimizer.apply_all_optimizations()
    
    # Benchmark optimizado
    logger.info("Benchmark modelo ultra-optimizado...")
    optimized_metrics = benchmark_model(optimized_model, num_runs=100)
    logger.info(f"Tiempo promedio: {optimized_metrics['avg_inference_time_ms']:.2f} ms")
    
    speedup = original_metrics['avg_inference_time_ms'] / optimized_metrics['avg_inference_time_ms']
    logger.info(f"Speedup total: {speedup:.2f}x")


def example_precompiled_model():
    """Ejemplo de modelo pre-compilado."""
    logger.info("=== Precompiled Model ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Pre-compilar
    precompiled = PrecompiledModel(model, compile_mode="max")
    
    # Benchmark
    metrics = benchmark_model(precompiled, num_runs=100)
    logger.info(f"Tiempo promedio (precompilado): {metrics['avg_inference_time_ms']:.2f} ms")


def example_stream_inference():
    """Ejemplo de inferencia con streams."""
    logger.info("=== Stream Inference ===")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA no disponible")
        return
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    model = model.cuda()
    
    # Crear stream inference
    stream_inference = StreamInference(model, num_streams=4)
    
    # Crear múltiples inputs
    inputs = [torch.randn(20) for _ in range(10)]
    
    # Procesar en paralelo
    start = time.time()
    outputs = stream_inference.predict_parallel(inputs)
    elapsed = time.time() - start
    
    logger.info(f"Procesados {len(inputs)} inputs en {elapsed*1000:.2f} ms")
    logger.info(f"Throughput: {len(inputs)/elapsed:.2f} samples/sec")


def example_dynamic_batching():
    """Ejemplo de batching dinámico."""
    logger.info("=== Dynamic Batching ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Crear dynamic batching
    dynamic_batch = DynamicBatching(
        model,
        min_batch_size=1,
        max_batch_size=128,
        target_latency=0.01
    )
    
    # Optimizar batch size
    optimal_size = dynamic_batch.optimize_batch_size()
    logger.info(f"Batch size óptimo: {optimal_size}")


def example_aot_compilation():
    """Ejemplo de compilación AOT."""
    logger.info("=== AOT Compilation ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Compilar AOT
    compiler = AOTCompiler(model, input_shape=(1, 20))
    compiled_path = compiler.compile("compiled_model_aot.pt")
    
    # Cargar y usar
    compiled_model = AOTCompiler.load_compiled(compiled_path)
    
    # Benchmark
    metrics = benchmark_model(compiled_model, num_runs=100)
    logger.info(f"Tiempo promedio (AOT): {metrics['avg_inference_time_ms']:.2f} ms")


def example_kernel_optimization():
    """Ejemplo de optimización de kernels."""
    logger.info("=== Kernel Optimization ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Optimizar kernels
    optimized = optimize_kernels(model)
    
    # Benchmark
    metrics = benchmark_model(optimized, num_runs=100)
    logger.info(f"Tiempo promedio (kernels optimizados): {metrics['avg_inference_time_ms']:.2f} ms")


def main():
    """Ejecutar ejemplos."""
    logger.info("Ejecutando ejemplos de inferencia ultra-rápida...\n")
    
    example_ultra_fast_optimization()
    print()
    
    example_precompiled_model()
    print()
    
    if torch.cuda.is_available():
        example_stream_inference()
        print()
    
    example_dynamic_batching()
    print()
    
    example_aot_compilation()
    print()
    
    example_kernel_optimization()
    print()
    
    logger.info("=== Todos los ejemplos completados ===")


if __name__ == "__main__":
    main()

