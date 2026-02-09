"""
Base Benchmark - Base class for all benchmarks.

Provides:
- Dataset loading and management
- Benchmark execution framework
- Metrics calculation and reporting
- Progress tracking
- Memory monitoring
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from datasets import load_dataset

from .types import BenchmarkResult, BenchmarkConfig
from .executor import BenchmarkExecutor

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# BASE BENCHMARK
# ════════════════════════════════════════════════════════════════════════════════

class BaseBenchmark(ABC):
    """
    Clase base para todos los benchmarks.
    
    Provides a framework for:
    - Loading datasets
    - Running benchmarks
    - Calculating metrics
    - Tracking progress
    """
    
    def __init__(
        self,
        name: str,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[str] = None,
        shots: int = 0,
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Inicializa el benchmark.
        
        Args:
            name: Nombre del benchmark
            dataset_name: Nombre del dataset (HuggingFace)
            dataset_config: Configuración del dataset
            shots: Número de ejemplos few-shot
            batch_size: Tamaño del batch
            max_samples: Máximo número de muestras a evaluar
            cache_dir: Directorio para cache del dataset
        """
        self.name = name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.shots = shots
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        
        self.dataset = None
        self.results: List[Dict[str, Any]] = []
        self._memory_baseline: Optional[Dict[str, float]] = None
    
    def load_dataset(self, split: Optional[str] = None) -> None:
        """
        Carga el dataset del benchmark.
        
        Args:
            split: Dataset split to load (default: "test" or "validation" based on shots)
        """
        if not self.dataset_name:
            logger.warning(f"No dataset name provided for benchmark {self.name}")
            return
        
        if split is None:
            split = "test" if self.shots == 0 else "validation"
        
        logger.info(f"Loading dataset: {self.dataset_name} (split: {split})")
        
        try:
            self.dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=split,
                cache_dir=self.cache_dir,
            )
            
            if self.max_samples:
                max_samples = min(self.max_samples, len(self.dataset))
                self.dataset = self.dataset.select(range(max_samples))
                logger.info(f"Limited to {max_samples} samples")
            
            logger.info(f"Dataset loaded: {len(self.dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    @abstractmethod
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Formatea un ejemplo en un prompt.
        
        Args:
            example: Ejemplo del dataset
            
        Returns:
            Prompt formateado
        """
        pass
    
    @abstractmethod
    def evaluate_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        """
        Evalúa si la respuesta es correcta.
        
        Args:
            prediction: Respuesta del modelo
            example: Ejemplo original
            
        Returns:
            True si la respuesta es correcta
        """
        pass
    
    def run(
        self,
        model_loader,
        progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
        save_results: bool = True,
        results_dir: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Ejecuta el benchmark completo.
        
        Args:
            model_loader: Instancia de ModelLoader
            progress_callback: Callback para progreso (current, total, correct, processed)
            save_results: Whether to save individual results
            results_dir: Directory to save results (if save_results=True)
            
        Returns:
            BenchmarkResult con los resultados
        """
        executor = BenchmarkExecutor()
        result = executor.execute_benchmark(
            self,
            model_loader,
            progress_callback,
            save_results
        )
        
        # Save results if requested
        if save_results and results_dir:
            self._save_results(result, results_dir)
        
        logger.info(f"Benchmark completed: {self.name}")
        logger.info(f"Accuracy: {result.accuracy:.2%}")
        logger.info(f"Latency P50: {result.latency_p50:.3f}s")
        logger.info(f"Latency P95: {result.latency_p95:.3f}s")
        logger.info(f"Throughput: {result.throughput:.2f} tokens/s")
        
        return result
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """
        Obtiene el uso de memoria.
        
        Returns:
            Dictionary with memory usage in MB
        """
        import torch
        import psutil
        
        memory_usage = {}
        
        # GPU memory
        try:
            if torch.cuda.is_available():
                memory_usage["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
                memory_usage["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
                memory_usage["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        except Exception:
            pass
        
        # CPU memory
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage["cpu_memory_mb"] = memory_info.rss / (1024**2)
            memory_usage["cpu_virtual_mb"] = memory_info.vms / (1024**2)
        except Exception as e:
            logger.debug(f"Failed to get CPU memory: {e}")
        
        return memory_usage
    
    def _save_results(self, result: BenchmarkResult, results_dir: str) -> None:
        """
        Save benchmark results to disk.
        
        Args:
            result: Benchmark result
            results_dir: Directory to save results
        """
        import json
        
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_path = results_path / f"{self.name}_{result.model_name}_{result.timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save detailed results
        if self.results:
            details_path = results_path / f"{self.name}_{result.model_name}_{result.timestamp}_details.json"
            with open(details_path, "w") as f:
                json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get individual results."""
        return self.results
    
    def clear_results(self) -> None:
        """Clear stored results."""
        self.results = []
