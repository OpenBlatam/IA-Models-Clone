"""
Continuous Generation Engine for Bulk TruthGPT
=============================================

Advanced continuous generation system that runs indefinitely after receiving a query.
Integrates with all TruthGPT optimization variants and provides real-time monitoring.
"""

import asyncio
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from queue import Queue, Empty
import torch
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import signal
import sys
import os

# Add TruthGPT paths
TRUTHGPT_PATH = os.path.join(os.path.dirname(__file__), "..", "Frontier-Model-run", "scripts", "TruthGPT-main")
sys.path.append(TRUTHGPT_PATH)
sys.path.append(os.path.join(TRUTHGPT_PATH, "optimization_core"))
sys.path.append(os.path.join(TRUTHGPT_PATH, "variant_optimized"))

# Import TruthGPT components
try:
    from optimization_core import *
    from variant_optimized import *
    from enhanced_model_optimizer import UniversalModelOptimizer, UniversalOptimizationConfig
except ImportError as e:
    logging.warning(f"Some TruthGPT components not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContinuousGenerationConfig:
    """Configuration for continuous generation."""
    # Generation settings
    max_documents: int = 1000
    generation_interval: float = 0.1  # seconds between generations
    batch_size: int = 1
    max_concurrent_tasks: int = 5
    
    # Model settings
    enable_model_rotation: bool = True
    model_rotation_interval: int = 100  # documents per model
    enable_adaptive_scheduling: bool = True
    
    # Performance settings
    memory_threshold: float = 0.9  # 90% memory usage threshold
    cpu_threshold: float = 0.8     # 80% CPU usage threshold
    enable_auto_cleanup: bool = True
    cleanup_interval: int = 50      # cleanup every N documents
    
    # Quality settings
    enable_quality_filtering: bool = True
    min_content_length: int = 50
    max_content_length: int = 2000
    enable_content_diversity: bool = True
    
    # Monitoring settings
    enable_real_time_monitoring: bool = True
    metrics_collection_interval: float = 1.0  # seconds
    enable_performance_profiling: bool = True

@dataclass
class GenerationResult:
    """Result of a single generation."""
    document_id: str
    content: str
    model_used: str
    generation_time: float
    quality_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    generation_rate: float
    error_rate: float
    timestamp: datetime

class ContinuousGenerationEngine:
    """Advanced continuous generation engine."""
    
    def __init__(self, config: ContinuousGenerationConfig):
        self.config = config
        self.is_running = False
        self.generation_queue = Queue()
        self.results_queue = Queue()
        self.active_tasks = {}
        self.generated_documents = []
        self.performance_metrics = {
            "total_generated": 0,
            "total_errors": 0,
            "average_generation_time": 0.0,
            "generation_rate": 0.0,
            "quality_scores": [],
            "model_usage": {},
            "system_metrics": []
        }
        
        # Model management
        self.available_models = {}
        self.current_model_index = 0
        self.model_performance = {}
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_tasks)
        self.cleanup_thread = None
        self.monitoring_thread = None
        
        # Signal handling
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self):
        """Initialize the continuous generation engine."""
        logger.info("Initializing Continuous Generation Engine...")
        
        try:
            # Load available models
            await self._load_models()
            
            # Initialize optimization suite
            await self._initialize_optimization()
            
            # Start monitoring thread
            if self.config.enable_real_time_monitoring:
                self._start_monitoring()
            
            # Start cleanup thread
            if self.config.enable_auto_cleanup:
                self._start_cleanup()
            
            logger.info("Continuous Generation Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Continuous Generation Engine: {e}")
            raise
    
    async def _load_models(self):
        """Load available models."""
        logger.info("Loading available models...")
        
        # Mock model loading - in real implementation, this would load actual TruthGPT models
        self.available_models = {
            "ultra_optimized_deepseek": {
                "type": "ultra_optimized",
                "parameters": 809_631_744,
                "memory_usage": "high",
                "capabilities": ["reasoning", "code_generation", "optimization"],
                "performance_score": 0.95
            },
            "ultra_optimized_viral_clipper": {
                "type": "ultra_optimized",
                "parameters": 25_178_112,
                "memory_usage": "low",
                "capabilities": ["content_generation", "viral_content"],
                "performance_score": 0.85
            },
            "ultra_optimized_brandkit": {
                "type": "ultra_optimized",
                "parameters": 10_493_952,
                "memory_usage": "medium",
                "capabilities": ["brand_content", "marketing"],
                "performance_score": 0.80
            },
            "qwen_variant": {
                "type": "qwen",
                "parameters": 1_000_000_000,
                "memory_usage": "high",
                "capabilities": ["multilingual", "reasoning"],
                "performance_score": 0.90
            },
            "claude_3_5_sonnet": {
                "type": "claude",
                "parameters": 2_000_000_000,
                "memory_usage": "high",
                "capabilities": ["reasoning", "analysis", "writing"],
                "performance_score": 0.92
            }
        }
        
        logger.info(f"Loaded {len(self.available_models)} models")
    
    async def _initialize_optimization(self):
        """Initialize optimization suite."""
        try:
            optimization_config = UniversalOptimizationConfig(
                enable_fp16=True,
                enable_quantization=True,
                enable_kernel_fusion=True,
                use_mcts_optimization=True,
                use_olympiad_benchmarks=True
            )
            
            self.optimization_suite = UniversalModelOptimizer(optimization_config)
            logger.info("Optimization suite initialized")
            
        except Exception as e:
            logger.warning(f"Could not initialize optimization suite: {e}")
            self.optimization_suite = None
    
    def _start_monitoring(self):
        """Start real-time monitoring thread."""
        def monitor():
            while self.is_running:
                try:
                    metrics = self._collect_system_metrics()
                    self.performance_metrics["system_metrics"].append(metrics)
                    
                    # Keep only recent metrics
                    if len(self.performance_metrics["system_metrics"]) > 1000:
                        self.performance_metrics["system_metrics"] = \
                            self.performance_metrics["system_metrics"][-500:]
                    
                    time.sleep(self.config.metrics_collection_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring thread: {e}")
                    time.sleep(1)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        logger.info("Real-time monitoring started")
    
    def _start_cleanup(self):
        """Start cleanup thread."""
        def cleanup():
            while self.is_running:
                try:
                    time.sleep(self.config.cleanup_interval * self.config.generation_interval)
                    self._perform_cleanup()
                    
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
                    time.sleep(1)
        
        self.cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        self.cleanup_thread.start()
        logger.info("Auto-cleanup started")
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        return SystemMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            gpu_usage=torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 
                     if torch.cuda.is_available() else 0.0,
            generation_rate=self._calculate_generation_rate(),
            error_rate=self._calculate_error_rate(),
            timestamp=datetime.now()
        )
    
    def _calculate_generation_rate(self) -> float:
        """Calculate current generation rate (documents per second)."""
        if not self.performance_metrics["system_metrics"]:
            return 0.0
        
        recent_metrics = self.performance_metrics["system_metrics"][-10:]  # Last 10 measurements
        if len(recent_metrics) < 2:
            return 0.0
        
        time_diff = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
        if time_diff == 0:
            return 0.0
        
        return len(recent_metrics) / time_diff
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        total_attempts = self.performance_metrics["total_generated"] + self.performance_metrics["total_errors"]
        if total_attempts == 0:
            return 0.0
        
        return self.performance_metrics["total_errors"] / total_attempts
    
    def _perform_cleanup(self):
        """Perform system cleanup."""
        try:
            # Clean up old documents
            if len(self.generated_documents) > 1000:
                self.generated_documents = self.generated_documents[-500:]
            
            # Clean up old metrics
            if len(self.performance_metrics["system_metrics"]) > 1000:
                self.performance_metrics["system_metrics"] = \
                    self.performance_metrics["system_metrics"][-500:]
            
            # Force garbage collection
            gc.collect()
            
            logger.info("System cleanup performed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def start_continuous_generation(self, query: str, callback: Optional[Callable] = None) -> AsyncGenerator[GenerationResult, None]:
        """Start continuous generation and yield results."""
        logger.info(f"Starting continuous generation for query: {query[:100]}...")
        
        self.is_running = True
        document_count = 0
        
        try:
            while self.is_running and document_count < self.config.max_documents:
                # Check system resources
                if not self._check_system_resources():
                    logger.warning("System resources exceeded, pausing generation...")
                    await asyncio.sleep(5)
                    continue
                
                # Select model
                model_name = self._select_model()
                
                # Generate document
                start_time = time.time()
                result = await self._generate_document(query, model_name, document_count)
                generation_time = time.time() - start_time
                
                if result:
                    # Create generation result
                    generation_result = GenerationResult(
                        document_id=f"doc_{int(time.time())}_{document_count}",
                        content=result,
                        model_used=model_name,
                        generation_time=generation_time,
                        quality_score=self._calculate_quality_score(result),
                        timestamp=datetime.now(),
                        metadata={
                            "query": query,
                            "document_count": document_count,
                            "system_metrics": self._collect_system_metrics()
                        }
                    )
                    
                    # Store result
                    self.generated_documents.append(generation_result)
                    document_count += 1
                    
                    # Update metrics
                    self._update_metrics(generation_result)
                    
                    # Call callback if provided
                    if callback:
                        await callback(generation_result)
                    
                    # Yield result
                    yield generation_result
                    
                    logger.info(f"Generated document {document_count}/{self.config.max_documents} "
                              f"using {model_name} (quality: {generation_result.quality_score:.3f})")
                
                # Wait before next generation
                await asyncio.sleep(self.config.generation_interval)
                
        except KeyboardInterrupt:
            logger.info("Generation stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous generation: {e}")
        finally:
            self.is_running = False
            logger.info(f"Continuous generation completed. Generated {document_count} documents.")
    
    def _check_system_resources(self) -> bool:
        """Check if system resources are within acceptable limits."""
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        if cpu_usage > self.config.cpu_threshold * 100:
            logger.warning(f"CPU usage too high: {cpu_usage:.1f}%")
            return False
        
        if memory_usage > self.config.memory_threshold * 100:
            logger.warning(f"Memory usage too high: {memory_usage:.1f}%")
            return False
        
        return True
    
    def _select_model(self) -> str:
        """Select model for generation."""
        if not self.available_models:
            return "default"
        
        if self.config.enable_model_rotation:
            # Rotate through models
            model_names = list(self.available_models.keys())
            selected_model = model_names[self.current_model_index % len(model_names)]
            self.current_model_index += 1
            return selected_model
        else:
            # Select best performing model
            best_model = max(
                self.available_models.items(),
                key=lambda x: x[1].get("performance_score", 0.5)
            )[0]
            return best_model
    
    async def _generate_document(self, query: str, model_name: str, document_count: int) -> Optional[str]:
        """Generate a single document."""
        try:
            # This would integrate with actual TruthGPT models
            # For now, generate mock content based on the query and model
            
            base_content = f"Generated content for query: {query}\n\n"
            
            if "deepseek" in model_name:
                content = base_content + f"DeepSeek analysis (document {document_count}): " \
                    "This is a comprehensive analysis using advanced reasoning capabilities..."
            elif "viral_clipper" in model_name:
                content = base_content + f"Viral content (document {document_count}): " \
                    "Engaging and shareable content designed for maximum impact..."
            elif "brandkit" in model_name:
                content = base_content + f"Brand content (document {document_count}): " \
                    "Professional brand-focused content with marketing optimization..."
            elif "qwen" in model_name:
                content = base_content + f"Qwen analysis (document {document_count}): " \
                    "Multilingual reasoning and comprehensive content generation..."
            elif "claude" in model_name:
                content = base_content + f"Claude analysis (document {document_count}): " \
                    "Advanced reasoning and detailed analysis with high-quality writing..."
            else:
                content = base_content + f"Standard generation (document {document_count}): " \
                    "General-purpose content generation..."
            
            # Add some variation
            variation = f"\n\nVariation {document_count}: " + "x" * (document_count % 50 + 10)
            content += variation
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating document with {model_name}: {e}")
            return None
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate quality score for generated content."""
        if not content:
            return 0.0
        
        # Simple quality metrics
        length_score = min(len(content) / 500, 1.0)  # Prefer longer content
        diversity_score = len(set(content.split())) / len(content.split()) if content.split() else 0
        structure_score = 1.0 if len(content.split('\n')) > 2 else 0.5
        
        # Weighted average
        quality_score = (length_score * 0.4 + diversity_score * 0.3 + structure_score * 0.3)
        return min(quality_score, 1.0)
    
    def _update_metrics(self, result: GenerationResult):
        """Update performance metrics."""
        self.performance_metrics["total_generated"] += 1
        
        # Update average generation time
        current_avg = self.performance_metrics["average_generation_time"]
        total_generated = self.performance_metrics["total_generated"]
        
        self.performance_metrics["average_generation_time"] = (
            (current_avg * (total_generated - 1) + result.generation_time) / total_generated
        )
        
        # Update quality scores
        self.performance_metrics["quality_scores"].append(result.quality_score)
        if len(self.performance_metrics["quality_scores"]) > 100:
            self.performance_metrics["quality_scores"] = \
                self.performance_metrics["quality_scores"][-50:]
        
        # Update model usage
        model_name = result.model_used
        if model_name not in self.performance_metrics["model_usage"]:
            self.performance_metrics["model_usage"][model_name] = 0
        self.performance_metrics["model_usage"][model_name] += 1
    
    def stop(self):
        """Stop continuous generation."""
        self.is_running = False
        logger.info("Continuous generation stopped")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "total_generated": self.performance_metrics["total_generated"],
            "total_errors": self.performance_metrics["total_errors"],
            "average_generation_time": self.performance_metrics["average_generation_time"],
            "average_quality_score": np.mean(self.performance_metrics["quality_scores"]) 
                                   if self.performance_metrics["quality_scores"] else 0.0,
            "model_usage": self.performance_metrics["model_usage"],
            "current_system_metrics": self._collect_system_metrics() if self.performance_metrics["system_metrics"] else None,
            "is_running": self.is_running
        }

# Example usage
async def main():
    """Example usage of the continuous generation engine."""
    config = ContinuousGenerationConfig(
        max_documents=20,
        generation_interval=0.5,
        enable_real_time_monitoring=True,
        enable_auto_cleanup=True
    )
    
    engine = ContinuousGenerationEngine(config)
    
    try:
        await engine.initialize()
        
        query = "Explain advanced machine learning optimization techniques"
        
        print(f"Starting continuous generation for: {query}")
        print("=" * 60)
        
        async for result in engine.start_continuous_generation(query):
            print(f"Document {result.document_id}:")
            print(f"  Model: {result.model_used}")
            print(f"  Quality: {result.quality_score:.3f}")
            print(f"  Time: {result.generation_time:.3f}s")
            print(f"  Content: {result.content[:100]}...")
            print("-" * 40)
            
            # Show performance summary every 5 documents
            if engine.performance_metrics["total_generated"] % 5 == 0:
                summary = engine.get_performance_summary()
                print(f"Performance Summary:")
                print(f"  Total Generated: {summary['total_generated']}")
                print(f"  Average Quality: {summary['average_quality_score']:.3f}")
                print(f"  Model Usage: {summary['model_usage']}")
                print("=" * 60)
    
    except KeyboardInterrupt:
        print("\nStopping generation...")
    finally:
        engine.stop()
        
        # Final summary
        summary = engine.get_performance_summary()
        print(f"\nFinal Performance Summary:")
        print(f"Total Documents: {summary['total_generated']}")
        print(f"Average Generation Time: {summary['average_generation_time']:.3f}s")
        print(f"Average Quality Score: {summary['average_quality_score']:.3f}")
        print(f"Model Usage: {summary['model_usage']}")

if __name__ == "__main__":
    asyncio.run(main())










