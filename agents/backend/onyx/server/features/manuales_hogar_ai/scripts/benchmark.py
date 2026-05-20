"""
Benchmark Script
================

Script para hacer benchmarks del sistema.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.embeddings.embedding_service import EmbeddingService
from ml.models.manual_generator_model import ManualGeneratorModel
from ml.config.ml_config import get_ml_config

async def benchmark_embeddings():
    """Benchmark de embeddings."""
    print("🔍 Benchmarking Embeddings...")
    
    config = get_ml_config()
    service = EmbeddingService(
        model_name=config.embedding_model,
        device=config.device,
        use_cache=True
    )
    
    texts = [f"Texto de prueba {i}" for i in range(100)]
    
    # Sin caché
    start = time.time()
    _ = service.encode(texts, batch_size=32)
    time_no_cache = time.time() - start
    
    # Con caché
    start = time.time()
    _ = service.encode(texts, batch_size=32)
    time_with_cache = time.time() - start
    
    print(f"  Sin caché: {time_no_cache:.3f}s")
    print(f"  Con caché: {time_with_cache:.3f}s")
    print(f"  Mejora: {time_no_cache/time_with_cache:.2f}x")
    
    return {
        "no_cache": time_no_cache,
        "with_cache": time_with_cache,
        "improvement": time_no_cache/time_with_cache
    }

async def benchmark_generation():
    """Benchmark de generación."""
    print("📝 Benchmarking Generación...")
    
    config = get_ml_config()
    model = ManualGeneratorModel(
        model_name=config.generation_model,
        use_lora=config.use_lora,
        device=config.device
    )
    
    prompt = "Genera un manual para reparar una fuga de agua"
    
    times = []
    for _ in range(10):
        start = time.time()
        _ = model.generate(prompt, max_length=100)
        times.append(time.time() - start)
    
    avg_time = statistics.mean(times)
    print(f"  Tiempo promedio: {avg_time:.3f}s")
    print(f"  Min: {min(times):.3f}s")
    print(f"  Max: {max(times):.3f}s")
    
    return {
        "mean": avg_time,
        "min": min(times),
        "max": max(times)
    }

async def main():
    """Ejecutar benchmarks."""
    print("=" * 60)
    print("BENCHMARK DEL SISTEMA")
    print("=" * 60)
    
    results = {}
    
    # Benchmark embeddings
    results["embeddings"] = await benchmark_embeddings()
    
    print()
    
    # Benchmark generación
    results["generation"] = await benchmark_generation()
    
    print()
    print("=" * 60)
    print("RESULTADOS FINALES")
    print("=" * 60)
    print(f"Embeddings (mejora con caché): {results['embeddings']['improvement']:.2f}x")
    print(f"Generación (tiempo promedio): {results['generation']['mean']:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())




