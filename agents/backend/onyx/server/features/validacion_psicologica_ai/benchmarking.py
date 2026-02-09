"""
Model Benchmarking
==================
Performance benchmarking utilities
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import time
import structlog
from contextlib import contextmanager
import statistics

logger = structlog.get_logger()


class ModelBenchmark:
    """
    Model performance benchmarker
    """
    
    def __init__(self):
        """Initialize benchmarker"""
        logger.info("ModelBenchmark initialized")
    
    @contextmanager
    def _timer(self):
        """Context manager for timing"""
        start = time.time()
        yield
        end = time.time()
        return end - start
    
    def benchmark_inference(
        self,
        model: nn.Module,
        input_data: Dict[str, torch.Tensor],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance
        
        Args:
            model: Model to benchmark
            input_data: Input data
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        try:
            model.eval()
            device = next(model.parameters()).device
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(**input_data)
            
            # Synchronize GPU if available
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.time()
                    _ = model(**input_data)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end = time.time()
                    times.append((end - start) * 1000)  # Convert to ms
            
            # Calculate statistics
            mean_time = statistics.mean(times)
            median_time = statistics.median(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)
            
            # Calculate throughput
            throughput = 1000 / mean_time if mean_time > 0 else 0
            
            results = {
                "mean_inference_time_ms": mean_time,
                "median_inference_time_ms": median_time,
                "std_inference_time_ms": std_time,
                "min_inference_time_ms": min_time,
                "max_inference_time_ms": max_time,
                "throughput_inferences_per_second": throughput,
                "num_runs": num_runs,
                "device": str(device)
            }
            
            logger.info("Inference benchmark completed", results=results)
            return results
        except Exception as e:
            logger.error("Error in inference benchmark", error=str(e))
            raise
    
    def benchmark_training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        input_data: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        num_runs: int = 50,
        warmup_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark training step performance
        
        Args:
            model: Model
            optimizer: Optimizer
            loss_fn: Loss function
            input_data: Input data
            targets: Targets
            num_runs: Number of runs
            warmup_runs: Warmup runs
            
        Returns:
            Benchmark results
        """
        try:
            model.train()
            device = next(model.parameters()).device
            
            # Warmup
            for _ in range(warmup_runs):
                optimizer.zero_grad()
                outputs = model(**input_data)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Synchronize GPU if available
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                
                optimizer.zero_grad()
                outputs = model(**input_data)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.time()
                times.append((end - start) * 1000)  # Convert to ms
            
            # Calculate statistics
            mean_time = statistics.mean(times)
            median_time = statistics.median(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            results = {
                "mean_training_step_time_ms": mean_time,
                "median_training_step_time_ms": median_time,
                "std_training_step_time_ms": std_time,
                "throughput_steps_per_second": 1000 / mean_time if mean_time > 0 else 0,
                "num_runs": num_runs,
                "device": str(device)
            }
            
            logger.info("Training step benchmark completed", results=results)
            return results
        except Exception as e:
            logger.error("Error in training step benchmark", error=str(e))
            raise
    
    def benchmark_memory(
        self,
        model: nn.Module,
        input_data: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Benchmark memory usage
        
        Args:
            model: Model
            input_data: Input data
            
        Returns:
            Memory statistics
        """
        try:
            device = next(model.parameters()).device
            
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                
                model.eval()
                with torch.no_grad():
                    _ = model(**input_data)
                
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                
                return {
                    "peak_memory_mb": peak_memory,
                    "current_memory_mb": current_memory,
                    "device": str(device)
                }
            else:
                # CPU memory estimation
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                model_size_mb = (param_size + buffer_size) / (1024 ** 2)
                
                return {
                    "model_size_mb": model_size_mb,
                    "device": "cpu"
                }
        except Exception as e:
            logger.error("Error in memory benchmark", error=str(e))
            raise


class BenchmarkSuite:
    """
    Comprehensive benchmark suite
    """
    
    def __init__(self):
        """Initialize benchmark suite"""
        self.benchmarker = ModelBenchmark()
        logger.info("BenchmarkSuite initialized")
    
    def run_full_benchmark(
        self,
        model: nn.Module,
        input_data: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Run full benchmark suite
        
        Args:
            model: Model
            input_data: Input data
            targets: Targets (for training benchmark)
            optimizer: Optimizer (for training benchmark)
            loss_fn: Loss function (for training benchmark)
            
        Returns:
            Complete benchmark results
        """
        results = {
            "inference": self.benchmarker.benchmark_inference(model, input_data),
            "memory": self.benchmarker.benchmark_memory(model, input_data)
        }
        
        if targets is not None and optimizer is not None and loss_fn is not None:
            results["training"] = self.benchmarker.benchmark_training_step(
                model, optimizer, loss_fn, input_data, targets
            )
        
        return results


# Global instances
model_benchmark = ModelBenchmark()
benchmark_suite = BenchmarkSuite()




