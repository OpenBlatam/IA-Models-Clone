"""
Extreme Speed Example
=====================

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
    ExtremeOptimizer,
    OptimizedInferenceEngine,
    HardwareOptimizer,
    MemoryPoolOptimizer,
    JITOptimizer,
    VectorizedOperations,
    benchmark_model
)


def example_extreme_optimization():
    """Ejemplo de optimización extrema."""
    logger.info("=== Extreme Optimization ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Benchmark original
    logger.info("Benchmark modelo original...")
    original_metrics = benchmark_model(model, num_runs=100)
    logger.info(f"Tiempo promedio: {original_metrics['avg_inference_time_ms']:.2f} ms")
    
    # Optimización extrema
    optimizer = ExtremeOptimizer(model, device="cuda" if torch.cuda.is_available() else "cpu")
    extreme_model = optimizer.apply_extreme_optimizations()
    
    # Benchmark extremo
    logger.info("Benchmark modelo extremadamente optimizado...")
    extreme_metrics = benchmark_model(extreme_model, num_runs=100)
    logger.info(f"Tiempo promedio: {extreme_metrics['avg_inference_time_ms']:.2f} ms")
    
    speedup = original_metrics['avg_inference_time_ms'] / extreme_metrics['avg_inference_time_ms']
    logger.info(f"Speedup extremo: {speedup:.2f}x")


def example_optimized_inference_engine():
    """Ejemplo de motor de inferencia optimizado."""
    logger.info("=== Optimized Inference Engine ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Crear motor optimizado
    engine = OptimizedInferenceEngine(
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cache=True,
        cache_size=10000
    )
    
    # Benchmark
    inputs = [torch.randn(20) for _ in range(100)]
    
    start = time.time()
    outputs = [engine.predict(inp) for inp in inputs]
    elapsed = time.time() - start
    
    logger.info(f"Procesados {len(inputs)} inputs en {elapsed*1000:.2f} ms")
    logger.info(f"Throughput: {len(inputs)/elapsed:.2f} samples/sec")
    
    # Batch ultra-rápido
    batch = torch.stack(inputs)
    start = time.time()
    batch_output = engine.predict_batch_ultra_fast(batch)
    elapsed = time.time() - start
    
    logger.info(f"Batch procesado en {elapsed*1000:.2f} ms")
    logger.info(f"Batch throughput: {len(inputs)/elapsed:.2f} samples/sec")


def example_hardware_optimization():
    """Ejemplo de optimización de hardware."""
    logger.info("=== Hardware Optimization ===")
    
    # Detectar hardware
    hardware = HardwareOptimizer.detect_hardware()
    logger.info(f"Hardware detectado: {hardware}")
    
    # Auto-optimizar
    HardwareOptimizer.auto_optimize()
    
    # Optimizar memory pool
    if torch.cuda.is_available():
        MemoryPoolOptimizer.optimize_memory_pool(device=0)


def example_jit_extreme():
    """Ejemplo de JIT extremo."""
    logger.info("=== JIT Extreme Optimization ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Compilar con JIT extremo
    example_input = torch.randn(1, 20)
    jit_model = JITOptimizer.compile_with_fusion(model, example_input)
    
    # Crear función optimizada
    optimized_func = JITOptimizer.create_optimized_inference_function(model)
    
    # Benchmark
    metrics = benchmark_model(jit_model, num_runs=100)
    logger.info(f"Tiempo promedio (JIT extremo): {metrics['avg_inference_time_ms']:.2f} ms")


def example_vectorized_operations():
    """Ejemplo de operaciones vectorizadas."""
    logger.info("=== Vectorized Operations ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Crear inputs
    inputs = torch.randn(1000, 20)
    
    # Predicción vectorizada
    start = time.time()
    outputs = VectorizedOperations.batch_predict_vectorized(model, inputs, batch_size=128)
    elapsed = time.time() - start
    
    logger.info(f"Procesados {len(inputs)} inputs en {elapsed*1000:.2f} ms")
    logger.info(f"Throughput: {len(inputs)/elapsed:.2f} samples/sec")


def main():
    """Ejecutar ejemplos."""
    logger.info("Ejecutando ejemplos de velocidad extrema...\n")
    
    example_extreme_optimization()
    print()
    
    example_optimized_inference_engine()
    print()
    
    example_hardware_optimization()
    print()
    
    example_jit_extreme()
    print()
    
    example_vectorized_operations()
    print()
    
    logger.info("=== Todos los ejemplos completados ===")


if __name__ == "__main__":
    main()

