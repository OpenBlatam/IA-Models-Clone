"""
Performance-Optimized NLP Benchmark
===================================

Script de benchmark para el sistema NLP optimizado para máximo rendimiento.
"""

import asyncio
import time
import statistics
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
import string
import psutil
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .performance_nlp_system import performance_nlp_system, PerformanceNLPConfig

logger = logging.getLogger(__name__)

class PerformanceNLPBenchmark:
    """Benchmark para el sistema NLP optimizado para rendimiento."""
    
    def __init__(self):
        self.results = {}
        self.test_texts = self._generate_test_texts()
        self.performance_metrics = {}
        self.throughput_metrics = {}
        self.memory_metrics = {}
        self.latency_metrics = {}
        
    def _generate_test_texts(self) -> List[str]:
        """Generar textos de prueba para el benchmark de rendimiento."""
        return [
            # Textos cortos para latencia
            "This is a great product!",
            "I love this service.",
            "This is terrible.",
            "The quality is excellent.",
            "I'm not satisfied with this.",
            
            # Textos medianos para rendimiento balanceado
            "The performance-optimized NLP system provides excellent throughput for natural language processing tasks. The system uses advanced optimization techniques including parallel processing, intelligent caching, and memory management.",
            "Este sistema NLP optimizado para rendimiento utiliza técnicas avanzadas de optimización incluyendo procesamiento paralelo, caché inteligente y gestión de memoria para máximo rendimiento.",
            "Le système NLP optimisé pour les performances utilise des techniques d'optimisation avancées incluant le traitement parallèle, la mise en cache intelligente et la gestion de la mémoire.",
            
            # Textos largos para pruebas de memoria
            "The comprehensive performance-optimized NLP system integrates multiple advanced optimization techniques including parallel processing, intelligent caching, memory management, GPU acceleration, and throughput optimization. The system provides sentiment analysis, named entity recognition, topic modeling, text classification, and regression capabilities with maximum performance. The optimization approach combines multiple techniques for improved speed, while the performance components handle complex processing efficiently. The system supports multiple languages and provides real-time analysis with intelligent caching for optimal performance.",
            "El sistema integral NLP optimizado para rendimiento integra múltiples técnicas avanzadas de optimización incluyendo procesamiento paralelo, caché inteligente, gestión de memoria, aceleración GPU y optimización de rendimiento. El sistema proporciona análisis de sentimientos, reconocimiento de entidades nombradas, modelado de temas, clasificación de texto y capacidades de regresión con máximo rendimiento. El enfoque de optimización combina múltiples técnicas para velocidad mejorada, mientras que los componentes de rendimiento manejan el procesamiento complejo eficientemente. El sistema soporta múltiples idiomas y proporciona análisis en tiempo real con caché inteligente para rendimiento óptimo.",
            
            # Textos técnicos para pruebas de CPU
            "The implementation utilizes state-of-the-art optimization techniques including vectorization, parallel processing, memory management, and GPU acceleration for various NLP tasks. The system employs advanced performance engineering techniques including batch processing, caching strategies, and resource optimization. The performance pipeline includes data preprocessing, feature optimization, model acceleration, and result caching. The optimization methods combine multiple techniques to improve overall performance. The performance components use advanced algorithms with attention mechanisms for complex text understanding.",
            "La implementación utiliza técnicas de optimización de última generación incluyendo vectorización, procesamiento paralelo, gestión de memoria y aceleración GPU para varias tareas de NLP. El sistema emplea técnicas avanzadas de ingeniería de rendimiento incluyendo procesamiento por lotes, estrategias de caché y optimización de recursos. La pipeline de rendimiento incluye preprocesamiento de datos, optimización de características, aceleración de modelos y caché de resultados. Los métodos de optimización combinan múltiples técnicas para mejorar el rendimiento general. Los componentes de rendimiento usan algoritmos avanzados con mecanismos de atención para comprensión compleja de texto.",
            
            # Textos con entidades para pruebas de NER
            "Apple Inc. announced new products at their headquarters in Cupertino, California. The CEO Tim Cook presented the iPhone 15 and MacBook Pro with advanced AI capabilities. The stock price increased by 5% to $180 per share. The event was attended by 1000 journalists from around the world.",
            "Google LLC desarrolló nuevos algoritmos de machine learning en su centro de investigación en Mountain View, California. El CEO Sundar Pichai presentó avances en procesamiento de lenguaje natural. Las acciones subieron 3% a $140 por acción. El evento fue cubierto por periodistas de todo el mundo.",
            
            # Textos con temas específicos para pruebas de topic modeling
            "The financial market analysis shows positive trends in technology stocks. The NASDAQ index increased by 2.5% while the S&P 500 gained 1.8%. Major tech companies like Microsoft, Amazon, and Tesla reported strong quarterly earnings. The Federal Reserve maintained interest rates at current levels. Investors remain optimistic about future growth prospects.",
            "El análisis del mercado financiero muestra tendencias positivas en acciones tecnológicas. El índice NASDAQ aumentó 2.5% mientras que el S&P 500 ganó 1.8%. Grandes compañías tecnológicas como Microsoft, Amazon y Tesla reportaron fuertes ganancias trimestrales. La Reserva Federal mantuvo las tasas de interés en niveles actuales. Los inversionistas permanecen optimistas sobre las perspectivas de crecimiento futuro."
        ]
    
    async def run_comprehensive_performance_benchmark(self) -> Dict[str, Any]:
        """Ejecutar benchmark completo de rendimiento del sistema NLP."""
        logger.info("Starting comprehensive performance NLP benchmark...")
        
        start_time = time.time()
        
        try:
            # Initialize system
            await performance_nlp_system.initialize()
            
            # Run different performance benchmark tests
            benchmark_results = {}
            
            # Throughput benchmark
            benchmark_results['throughput'] = await self._benchmark_throughput()
            
            # Latency benchmark
            benchmark_results['latency'] = await self._benchmark_latency()
            
            # Memory benchmark
            benchmark_results['memory'] = await self._benchmark_memory()
            
            # CPU benchmark
            benchmark_results['cpu'] = await self._benchmark_cpu()
            
            # GPU benchmark
            benchmark_results['gpu'] = await self._benchmark_gpu()
            
            # Cache performance benchmark
            benchmark_results['cache_performance'] = await self._benchmark_cache_performance()
            
            # Parallel processing benchmark
            benchmark_results['parallel_processing'] = await self._benchmark_parallel_processing()
            
            # Batch processing benchmark
            benchmark_results['batch_processing'] = await self._benchmark_batch_processing()
            
            # Quality vs performance benchmark
            benchmark_results['quality_vs_performance'] = await self._benchmark_quality_vs_performance()
            
            # Scalability benchmark
            benchmark_results['scalability'] = await self._benchmark_scalability()
            
            # Resource utilization benchmark
            benchmark_results['resource_utilization'] = await self._benchmark_resource_utilization()
            
            # Performance comparison
            benchmark_results['performance_comparison'] = await self._benchmark_performance_comparison()
            
            # Calculate overall benchmark results
            total_time = time.time() - start_time
            benchmark_results['summary'] = self._calculate_performance_benchmark_summary(benchmark_results, total_time)
            
            # Generate performance report
            self._generate_performance_report(benchmark_results)
            
            # Save results
            self._save_performance_benchmark_results(benchmark_results)
            
            logger.info(f"Comprehensive performance NLP benchmark completed in {total_time:.2f}s")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Comprehensive performance benchmark failed: {e}")
            raise
    
    async def _benchmark_throughput(self) -> Dict[str, Any]:
        """Benchmark de throughput."""
        logger.info("Running throughput benchmark...")
        
        start_time = time.time()
        results = []
        
        try:
            # Test different batch sizes for throughput
            batch_sizes = [10, 50, 100, 200, 500]
            
            for batch_size in batch_sizes:
                batch_start = time.time()
                
                # Create batch of texts
                batch_texts = self.test_texts[:batch_size]
                
                # Process batch
                batch_results = await performance_nlp_system.batch_analyze_performance_optimized(
                    texts=batch_texts,
                    language="en",
                    use_cache=True,
                    performance_mode="fast"
                )
                
                batch_time = time.time() - batch_start
                throughput = len(batch_texts) / batch_time if batch_time > 0 else 0
                
                results.append({
                    'batch_size': batch_size,
                    'processing_time': batch_time,
                    'throughput': throughput,
                    'texts_processed': len(batch_texts),
                    'average_quality': statistics.mean([r.quality_score for r in batch_results]),
                    'average_confidence': statistics.mean([r.confidence_score for r in batch_results])
                })
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'throughput',
                'batch_sizes_tested': batch_sizes,
                'results': results,
                'total_time': processing_time,
                'max_throughput': max(r['throughput'] for r in results),
                'average_throughput': statistics.mean([r['throughput'] for r in results]),
                'best_batch_size': max(results, key=lambda x: x['throughput'])['batch_size']
            }
            
        except Exception as e:
            logger.error(f"Throughput benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_latency(self) -> Dict[str, Any]:
        """Benchmark de latencia."""
        logger.info("Running latency benchmark...")
        
        start_time = time.time()
        results = []
        
        try:
            # Test individual text processing for latency
            for i, text in enumerate(self.test_texts[:50]):  # Test with first 50 texts
                text_start = time.time()
                
                result = await performance_nlp_system.analyze_performance_optimized(
                    text=text,
                    language="en",
                    use_cache=True,
                    performance_mode="fast"
                )
                
                text_time = time.time() - text_start
                
                results.append({
                    'text_index': i,
                    'text_length': len(text),
                    'processing_time': text_time,
                    'quality_score': result.quality_score,
                    'confidence_score': result.confidence_score,
                    'throughput': result.throughput,
                    'memory_usage': result.memory_usage
                })
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'latency',
                'texts_processed': len(results),
                'total_time': processing_time,
                'results': results,
                'average_latency': statistics.mean([r['processing_time'] for r in results]),
                'min_latency': min([r['processing_time'] for r in results]),
                'max_latency': max([r['processing_time'] for r in results]),
                'p95_latency': np.percentile([r['processing_time'] for r in results], 95),
                'p99_latency': np.percentile([r['processing_time'] for r in results], 99)
            }
            
        except Exception as e:
            logger.error(f"Latency benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_memory(self) -> Dict[str, Any]:
        """Benchmark de memoria."""
        logger.info("Running memory benchmark...")
        
        start_time = time.time()
        memory_results = []
        
        try:
            # Monitor memory usage during processing
            for i, text in enumerate(self.test_texts[:30]):
                # Get memory before processing
                memory_before = psutil.virtual_memory().used / (1024**3)  # GB
                
                # Process text
                result = await performance_nlp_system.analyze_performance_optimized(
                    text=text,
                    language="en",
                    use_cache=True,
                    performance_mode="balanced"
                )
                
                # Get memory after processing
                memory_after = psutil.virtual_memory().used / (1024**3)  # GB
                
                memory_results.append({
                    'text_index': i,
                    'text_length': len(text),
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'memory_delta': memory_after - memory_before,
                    'quality_score': result.quality_score,
                    'confidence_score': result.confidence_score,
                    'processing_time': result.processing_time
                })
                
                # Force garbage collection
                gc.collect()
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'memory',
                'texts_processed': len(memory_results),
                'total_time': processing_time,
                'memory_results': memory_results,
                'average_memory_delta': statistics.mean([m['memory_delta'] for m in memory_results]),
                'max_memory_delta': max([m['memory_delta'] for m in memory_results]),
                'min_memory_delta': min([m['memory_delta'] for m in memory_results]),
                'total_memory_used': sum([m['memory_delta'] for m in memory_results]),
                'memory_efficiency': self._calculate_memory_efficiency(memory_results)
            }
            
        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_cpu(self) -> Dict[str, Any]:
        """Benchmark de CPU."""
        logger.info("Running CPU benchmark...")
        
        start_time = time.time()
        cpu_results = []
        
        try:
            # Monitor CPU usage during processing
            for i, text in enumerate(self.test_texts[:30]):
                # Get CPU before processing
                cpu_before = psutil.cpu_percent()
                
                # Process text
                result = await performance_nlp_system.analyze_performance_optimized(
                    text=text,
                    language="en",
                    use_cache=True,
                    performance_mode="balanced"
                )
                
                # Get CPU after processing
                cpu_after = psutil.cpu_percent()
                
                cpu_results.append({
                    'text_index': i,
                    'text_length': len(text),
                    'cpu_before': cpu_before,
                    'cpu_after': cpu_after,
                    'cpu_delta': cpu_after - cpu_before,
                    'quality_score': result.quality_score,
                    'confidence_score': result.confidence_score,
                    'processing_time': result.processing_time
                })
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'cpu',
                'texts_processed': len(cpu_results),
                'total_time': processing_time,
                'cpu_results': cpu_results,
                'average_cpu_delta': statistics.mean([c['cpu_delta'] for c in cpu_results]),
                'max_cpu_delta': max([c['cpu_delta'] for c in cpu_results]),
                'min_cpu_delta': min([c['cpu_delta'] for c in cpu_results]),
                'cpu_efficiency': self._calculate_cpu_efficiency(cpu_results)
            }
            
        except Exception as e:
            logger.error(f"CPU benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_gpu(self) -> Dict[str, Any]:
        """Benchmark de GPU."""
        logger.info("Running GPU benchmark...")
        
        start_time = time.time()
        gpu_results = []
        
        try:
            # Monitor GPU usage during processing
            for i, text in enumerate(self.test_texts[:30]):
                # Get GPU before processing
                gpu_before = 0.0
                if torch.cuda.is_available():
                    gpu_before = torch.cuda.memory_allocated() / (1024**3)  # GB
                
                # Process text
                result = await performance_nlp_system.analyze_performance_optimized(
                    text=text,
                    language="en",
                    use_cache=True,
                    performance_mode="balanced"
                )
                
                # Get GPU after processing
                gpu_after = 0.0
                if torch.cuda.is_available():
                    gpu_after = torch.cuda.memory_allocated() / (1024**3)  # GB
                
                gpu_results.append({
                    'text_index': i,
                    'text_length': len(text),
                    'gpu_before': gpu_before,
                    'gpu_after': gpu_after,
                    'gpu_delta': gpu_after - gpu_before,
                    'quality_score': result.quality_score,
                    'confidence_score': result.confidence_score,
                    'processing_time': result.processing_time
                })
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'gpu',
                'texts_processed': len(gpu_results),
                'total_time': processing_time,
                'gpu_results': gpu_results,
                'average_gpu_delta': statistics.mean([g['gpu_delta'] for g in gpu_results]),
                'max_gpu_delta': max([g['gpu_delta'] for g in gpu_results]),
                'min_gpu_delta': min([g['gpu_delta'] for g in gpu_results]),
                'gpu_efficiency': self._calculate_gpu_efficiency(gpu_results)
            }
            
        except Exception as e:
            logger.error(f"GPU benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_cache_performance(self) -> Dict[str, Any]:
        """Benchmark de rendimiento de caché."""
        logger.info("Running cache performance benchmark...")
        
        start_time = time.time()
        
        try:
            # Test cache performance
            cache_results = []
            
            # First pass - no cache
            for text in self.test_texts[:20]:
                result = await performance_nlp_system.analyze_performance_optimized(
                    text=text,
                    language="en",
                    use_cache=False,
                    performance_mode="fast"
                )
                cache_results.append({
                    'text': text,
                    'cache_hit': result.cache_hit,
                    'processing_time': result.processing_time
                })
            
            # Second pass - with cache
            for text in self.test_texts[:20]:
                result = await performance_nlp_system.analyze_performance_optimized(
                    text=text,
                    language="en",
                    use_cache=True,
                    performance_mode="fast"
                )
                cache_results.append({
                    'text': text,
                    'cache_hit': result.cache_hit,
                    'processing_time': result.processing_time
                })
            
            processing_time = time.time() - start_time
            
            # Calculate cache statistics
            cache_hits = sum(1 for r in cache_results if r['cache_hit'])
            cache_misses = len(cache_results) - cache_hits
            cache_hit_rate = cache_hits / len(cache_results) if cache_results else 0
            
            return {
                'test_type': 'cache_performance',
                'texts_processed': len(cache_results),
                'total_time': processing_time,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'cache_hit_rate': cache_hit_rate,
                'average_processing_time': statistics.mean([r['processing_time'] for r in cache_results]),
                'cache_results': cache_results
            }
            
        except Exception as e:
            logger.error(f"Cache performance benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_parallel_processing(self) -> Dict[str, Any]:
        """Benchmark de procesamiento paralelo."""
        logger.info("Running parallel processing benchmark...")
        
        start_time = time.time()
        
        try:
            # Test different levels of parallelism
            parallelism_levels = [1, 2, 4, 8, 16]
            parallel_results = {}
            
            for level in parallelism_levels:
                level_start = time.time()
                
                # Create tasks for parallel processing
                tasks = []
                for text in self.test_texts[:20]:
                    task = performance_nlp_system.analyze_performance_optimized(
                        text=text,
                        language="en",
                        use_cache=True,
                        performance_mode="fast"
                    )
                    tasks.append(task)
                
                # Process in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                level_time = time.time() - level_start
                throughput = len(results) / level_time if level_time > 0 else 0
                
                parallel_results[level] = {
                    'parallelism_level': level,
                    'processing_time': level_time,
                    'throughput': throughput,
                    'texts_processed': len(results),
                    'success_rate': sum(1 for r in results if not isinstance(r, Exception)) / len(results)
                }
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'parallel_processing',
                'parallelism_levels': parallelism_levels,
                'results': parallel_results,
                'total_time': processing_time,
                'best_parallelism_level': max(parallel_results.keys(), key=lambda k: parallel_results[k]['throughput']),
                'max_throughput': max(parallel_results[k]['throughput'] for k in parallel_results.keys())
            }
            
        except Exception as e:
            logger.error(f"Parallel processing benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_batch_processing(self) -> Dict[str, Any]:
        """Benchmark de procesamiento por lotes."""
        logger.info("Running batch processing benchmark...")
        
        start_time = time.time()
        
        try:
            # Test different batch sizes
            batch_sizes = [5, 10, 20, 50, 100]
            batch_results = {}
            
            for batch_size in batch_sizes:
                batch_start = time.time()
                
                # Create batch of texts
                batch_texts = self.test_texts[:batch_size]
                
                # Process batch
                results = await performance_nlp_system.batch_analyze_performance_optimized(
                    texts=batch_texts,
                    language="en",
                    use_cache=True,
                    performance_mode="balanced"
                )
                
                batch_time = time.time() - batch_start
                throughput = len(batch_texts) / batch_time if batch_time > 0 else 0
                
                batch_results[batch_size] = {
                    'batch_size': batch_size,
                    'processing_time': batch_time,
                    'throughput': throughput,
                    'texts_processed': len(results),
                    'average_quality': statistics.mean([r.quality_score for r in results]),
                    'average_confidence': statistics.mean([r.confidence_score for r in results])
                }
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'batch_processing',
                'batch_sizes': batch_sizes,
                'results': batch_results,
                'total_time': processing_time,
                'best_batch_size': max(batch_results.keys(), key=lambda k: batch_results[k]['throughput']),
                'max_throughput': max(batch_results[k]['throughput'] for k in batch_results.keys())
            }
            
        except Exception as e:
            logger.error(f"Batch processing benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_quality_vs_performance(self) -> Dict[str, Any]:
        """Benchmark de calidad vs rendimiento."""
        logger.info("Running quality vs performance benchmark...")
        
        start_time = time.time()
        
        try:
            # Test different performance modes
            performance_modes = ['fast', 'balanced', 'quality']
            mode_results = {}
            
            for mode in performance_modes:
                mode_start = time.time()
                mode_results = []
                
                for text in self.test_texts[:20]:
                    result = await performance_nlp_system.analyze_performance_optimized(
                        text=text,
                        language="en",
                        use_cache=True,
                        performance_mode=mode
                    )
                    mode_results.append(result)
                
                mode_time = time.time() - mode_start
                throughput = len(mode_results) / mode_time if mode_time > 0 else 0
                
                mode_results[mode] = {
                    'performance_mode': mode,
                    'processing_time': mode_time,
                    'throughput': throughput,
                    'texts_processed': len(mode_results),
                    'average_quality': statistics.mean([r.quality_score for r in mode_results]),
                    'average_confidence': statistics.mean([r.confidence_score for r in mode_results]),
                    'average_processing_time': statistics.mean([r.processing_time for r in mode_results])
                }
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'quality_vs_performance',
                'performance_modes': performance_modes,
                'results': mode_results,
                'total_time': processing_time,
                'best_mode_for_quality': max(mode_results.keys(), key=lambda k: mode_results[k]['average_quality']),
                'best_mode_for_throughput': max(mode_results.keys(), key=lambda k: mode_results[k]['throughput'])
            }
            
        except Exception as e:
            logger.error(f"Quality vs performance benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark de escalabilidad."""
        logger.info("Running scalability benchmark...")
        
        start_time = time.time()
        
        try:
            # Test different scales
            scales = [10, 50, 100, 200, 500]
            scale_results = {}
            
            for scale in scales:
                scale_start = time.time()
                
                # Create texts for this scale
                scale_texts = self.test_texts[:scale]
                
                # Process at this scale
                results = await performance_nlp_system.batch_analyze_performance_optimized(
                    texts=scale_texts,
                    language="en",
                    use_cache=True,
                    performance_mode="balanced"
                )
                
                scale_time = time.time() - scale_start
                throughput = len(scale_texts) / scale_time if scale_time > 0 else 0
                
                scale_results[scale] = {
                    'scale': scale,
                    'processing_time': scale_time,
                    'throughput': throughput,
                    'texts_processed': len(results),
                    'average_quality': statistics.mean([r.quality_score for r in results]),
                    'average_confidence': statistics.mean([r.confidence_score for r in results])
                }
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'scalability',
                'scales': scales,
                'results': scale_results,
                'total_time': processing_time,
                'scalability_factor': self._calculate_scalability_factor(scale_results)
            }
            
        except Exception as e:
            logger.error(f"Scalability benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_resource_utilization(self) -> Dict[str, Any]:
        """Benchmark de utilización de recursos."""
        logger.info("Running resource utilization benchmark...")
        
        start_time = time.time()
        resource_results = []
        
        try:
            # Monitor resource utilization during processing
            for i, text in enumerate(self.test_texts[:30]):
                # Get resource usage before processing
                memory_before = psutil.virtual_memory().used / (1024**3)  # GB
                cpu_before = psutil.cpu_percent()
                gpu_before = 0.0
                if torch.cuda.is_available():
                    gpu_before = torch.cuda.memory_allocated() / (1024**3)  # GB
                
                # Process text
                result = await performance_nlp_system.analyze_performance_optimized(
                    text=text,
                    language="en",
                    use_cache=True,
                    performance_mode="balanced"
                )
                
                # Get resource usage after processing
                memory_after = psutil.virtual_memory().used / (1024**3)  # GB
                cpu_after = psutil.cpu_percent()
                gpu_after = 0.0
                if torch.cuda.is_available():
                    gpu_after = torch.cuda.memory_allocated() / (1024**3)  # GB
                
                resource_results.append({
                    'text_index': i,
                    'text_length': len(text),
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'memory_delta': memory_after - memory_before,
                    'cpu_before': cpu_before,
                    'cpu_after': cpu_after,
                    'cpu_delta': cpu_after - cpu_before,
                    'gpu_before': gpu_before,
                    'gpu_after': gpu_after,
                    'gpu_delta': gpu_after - gpu_before,
                    'quality_score': result.quality_score,
                    'confidence_score': result.confidence_score,
                    'processing_time': result.processing_time
                })
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'resource_utilization',
                'texts_processed': len(resource_results),
                'total_time': processing_time,
                'resource_results': resource_results,
                'average_memory_delta': statistics.mean([r['memory_delta'] for r in resource_results]),
                'average_cpu_delta': statistics.mean([r['cpu_delta'] for r in resource_results]),
                'average_gpu_delta': statistics.mean([r['gpu_delta'] for r in resource_results]),
                'resource_efficiency': self._calculate_resource_efficiency(resource_results)
            }
            
        except Exception as e:
            logger.error(f"Resource utilization benchmark failed: {e}")
            return {'error': str(e)}
    
    async def _benchmark_performance_comparison(self) -> Dict[str, Any]:
        """Benchmark de comparación de rendimiento."""
        logger.info("Running performance comparison benchmark...")
        
        start_time = time.time()
        
        try:
            # Compare different configurations
            configurations = [
                {'name': 'fast', 'performance_mode': 'fast', 'use_cache': True},
                {'name': 'balanced', 'performance_mode': 'balanced', 'use_cache': True},
                {'name': 'quality', 'performance_mode': 'quality', 'use_cache': True},
                {'name': 'no_cache', 'performance_mode': 'balanced', 'use_cache': False}
            ]
            
            comparison_results = {}
            
            for config in configurations:
                config_start = time.time()
                config_results = []
                
                for text in self.test_texts[:20]:
                    result = await performance_nlp_system.analyze_performance_optimized(
                        text=text,
                        language="en",
                        use_cache=config['use_cache'],
                        performance_mode=config['performance_mode']
                    )
                    config_results.append(result)
                
                config_time = time.time() - config_start
                
                comparison_results[config['name']] = {
                    'configuration': config,
                    'texts_processed': len(config_results),
                    'total_time': config_time,
                    'average_time': config_time / len(config_results),
                    'average_quality': statistics.mean([r.quality_score for r in config_results]),
                    'average_confidence': statistics.mean([r.confidence_score for r in config_results]),
                    'average_throughput': len(config_results) / config_time,
                    'average_memory_usage': statistics.mean([r.memory_usage for r in config_results])
                }
            
            processing_time = time.time() - start_time
            
            return {
                'test_type': 'performance_comparison',
                'configurations': comparison_results,
                'total_time': processing_time,
                'best_configuration': max(comparison_results.keys(), key=lambda k: comparison_results[k]['average_throughput']),
                'fastest_configuration': min(comparison_results.keys(), key=lambda k: comparison_results[k]['average_time']),
                'highest_quality_configuration': max(comparison_results.keys(), key=lambda k: comparison_results[k]['average_quality'])
            }
            
        except Exception as e:
            logger.error(f"Performance comparison benchmark failed: {e}")
            return {'error': str(e)}
    
    def _calculate_memory_efficiency(self, memory_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcular eficiencia de memoria."""
        try:
            if not memory_results:
                return {}
            
            memory_deltas = [r['memory_delta'] for r in memory_results]
            text_lengths = [r['text_length'] for r in memory_results]
            
            # Calculate memory efficiency (text length per MB of memory)
            efficiency_scores = []
            for i, (delta, length) in enumerate(zip(memory_deltas, text_lengths)):
                if delta > 0:
                    efficiency = length / delta
                    efficiency_scores.append(efficiency)
            
            return {
                'average_efficiency': statistics.mean(efficiency_scores) if efficiency_scores else 0,
                'max_efficiency': max(efficiency_scores) if efficiency_scores else 0,
                'min_efficiency': min(efficiency_scores) if efficiency_scores else 0,
                'efficiency_variance': statistics.variance(efficiency_scores) if len(efficiency_scores) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Memory efficiency calculation failed: {e}")
            return {}
    
    def _calculate_cpu_efficiency(self, cpu_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcular eficiencia de CPU."""
        try:
            if not cpu_results:
                return {}
            
            cpu_deltas = [r['cpu_delta'] for r in cpu_results]
            text_lengths = [r['text_length'] for r in cpu_results]
            
            # Calculate CPU efficiency (text length per CPU percentage)
            efficiency_scores = []
            for i, (delta, length) in enumerate(zip(cpu_deltas, text_lengths)):
                if delta > 0:
                    efficiency = length / delta
                    efficiency_scores.append(efficiency)
            
            return {
                'average_efficiency': statistics.mean(efficiency_scores) if efficiency_scores else 0,
                'max_efficiency': max(efficiency_scores) if efficiency_scores else 0,
                'min_efficiency': min(efficiency_scores) if efficiency_scores else 0,
                'efficiency_variance': statistics.variance(efficiency_scores) if len(efficiency_scores) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"CPU efficiency calculation failed: {e}")
            return {}
    
    def _calculate_gpu_efficiency(self, gpu_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcular eficiencia de GPU."""
        try:
            if not gpu_results:
                return {}
            
            gpu_deltas = [r['gpu_delta'] for r in gpu_results]
            text_lengths = [r['text_length'] for r in gpu_results]
            
            # Calculate GPU efficiency (text length per GB of GPU memory)
            efficiency_scores = []
            for i, (delta, length) in enumerate(zip(gpu_deltas, text_lengths)):
                if delta > 0:
                    efficiency = length / delta
                    efficiency_scores.append(efficiency)
            
            return {
                'average_efficiency': statistics.mean(efficiency_scores) if efficiency_scores else 0,
                'max_efficiency': max(efficiency_scores) if efficiency_scores else 0,
                'min_efficiency': min(efficiency_scores) if efficiency_scores else 0,
                'efficiency_variance': statistics.variance(efficiency_scores) if len(efficiency_scores) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"GPU efficiency calculation failed: {e}")
            return {}
    
    def _calculate_scalability_factor(self, scale_results: Dict[int, Dict[str, Any]]) -> float:
        """Calcular factor de escalabilidad."""
        try:
            if len(scale_results) < 2:
                return 0.0
            
            # Calculate scalability factor (how well throughput scales with input size)
            scales = sorted(scale_results.keys())
            throughputs = [scale_results[scale]['throughput'] for scale in scales]
            
            # Calculate scaling factor
            scaling_factor = 0.0
            for i in range(1, len(scales)):
                scale_ratio = scales[i] / scales[i-1]
                throughput_ratio = throughputs[i] / throughputs[i-1] if throughputs[i-1] > 0 else 0
                scaling_factor += throughput_ratio / scale_ratio
            
            return scaling_factor / (len(scales) - 1) if len(scales) > 1 else 0.0
            
        except Exception as e:
            logger.error(f"Scalability factor calculation failed: {e}")
            return 0.0
    
    def _calculate_resource_efficiency(self, resource_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcular eficiencia de recursos."""
        try:
            if not resource_results:
                return {}
            
            # Calculate resource efficiency metrics
            memory_efficiency = self._calculate_memory_efficiency(resource_results)
            cpu_efficiency = self._calculate_cpu_efficiency(resource_results)
            gpu_efficiency = self._calculate_gpu_efficiency(resource_results)
            
            return {
                'memory_efficiency': memory_efficiency,
                'cpu_efficiency': cpu_efficiency,
                'gpu_efficiency': gpu_efficiency,
                'overall_efficiency': (
                    memory_efficiency.get('average_efficiency', 0) +
                    cpu_efficiency.get('average_efficiency', 0) +
                    gpu_efficiency.get('average_efficiency', 0)
                ) / 3
            }
            
        except Exception as e:
            logger.error(f"Resource efficiency calculation failed: {e}")
            return {}
    
    def _calculate_performance_benchmark_summary(self, benchmark_results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Calcular resumen del benchmark de rendimiento."""
        try:
            summary = {
                'total_benchmark_time': total_time,
                'tests_completed': len([k for k, v in benchmark_results.items() if isinstance(v, dict) and 'error' not in v]),
                'tests_failed': len([k for k, v in benchmark_results.items() if isinstance(v, dict) and 'error' in v]),
                'overall_performance': {},
                'recommendations': []
            }
            
            # Calculate overall performance metrics
            all_throughputs = []
            all_latencies = []
            all_memory_usage = []
            all_quality_scores = []
            
            for test_name, test_results in benchmark_results.items():
                if isinstance(test_results, dict) and 'error' not in test_results:
                    if 'max_throughput' in test_results:
                        all_throughputs.append(test_results['max_throughput'])
                    if 'average_latency' in test_results:
                        all_latencies.append(test_results['average_latency'])
                    if 'average_memory_delta' in test_results:
                        all_memory_usage.append(test_results['average_memory_delta'])
                    if 'average_quality' in test_results:
                        all_quality_scores.append(test_results['average_quality'])
            
            if all_throughputs:
                summary['overall_performance']['max_throughput'] = max(all_throughputs)
                summary['overall_performance']['average_throughput'] = statistics.mean(all_throughputs)
            
            if all_latencies:
                summary['overall_performance']['average_latency'] = statistics.mean(all_latencies)
                summary['overall_performance']['min_latency'] = min(all_latencies)
                summary['overall_performance']['max_latency'] = max(all_latencies)
            
            if all_memory_usage:
                summary['overall_performance']['average_memory_usage'] = statistics.mean(all_memory_usage)
                summary['overall_performance']['max_memory_usage'] = max(all_memory_usage)
            
            if all_quality_scores:
                summary['overall_performance']['average_quality'] = statistics.mean(all_quality_scores)
                summary['overall_performance']['min_quality'] = min(all_quality_scores)
                summary['overall_performance']['max_quality'] = max(all_quality_scores)
            
            # Generate recommendations
            if summary['overall_performance'].get('average_latency', 0) > 5.0:
                summary['recommendations'].append("Consider optimizing for lower latency")
            
            if summary['overall_performance'].get('max_throughput', 0) < 10.0:
                summary['recommendations'].append("Consider optimizing for higher throughput")
            
            if summary['overall_performance'].get('average_memory_usage', 0) > 100.0:
                summary['recommendations'].append("Consider optimizing memory usage")
            
            if summary['overall_performance'].get('average_quality', 0) < 0.7:
                summary['recommendations'].append("Consider optimizing for higher quality")
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance benchmark summary calculation failed: {e}")
            return {'error': str(e)}
    
    def _generate_performance_report(self, benchmark_results: Dict[str, Any]):
        """Generar reporte de rendimiento."""
        try:
            # This would generate a comprehensive performance report
            # For now, just log the attempt
            logger.info("Generating performance report")
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
    
    def _save_performance_benchmark_results(self, benchmark_results: Dict[str, Any]):
        """Guardar resultados del benchmark de rendimiento."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_nlp_benchmark_results_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(benchmark_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Performance benchmark results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save performance benchmark results: {e}")

# Main performance benchmark execution

async def main():
    """Ejecutar benchmark principal de rendimiento."""
    try:
        benchmark = PerformanceNLPBenchmark()
        results = await benchmark.run_comprehensive_performance_benchmark()
        
        print("\n" + "="*80)
        print("PERFORMANCE-OPTIMIZED NLP BENCHMARK RESULTS")
        print("="*80)
        
        # Print summary
        if 'summary' in results:
            summary = results['summary']
            print(f"\nTotal Benchmark Time: {summary.get('total_benchmark_time', 0):.2f}s")
            print(f"Tests Completed: {summary.get('tests_completed', 0)}")
            print(f"Tests Failed: {summary.get('tests_failed', 0)}")
            
            if 'overall_performance' in summary:
                perf = summary['overall_performance']
                print(f"\nOverall Performance:")
                if 'max_throughput' in perf:
                    print(f"  Max Throughput: {perf['max_throughput']:.2f} texts/sec")
                if 'average_latency' in perf:
                    print(f"  Average Latency: {perf['average_latency']:.3f}s")
                if 'average_memory_usage' in perf:
                    print(f"  Average Memory Usage: {perf['average_memory_usage']:.2f} MB")
                if 'average_quality' in perf:
                    print(f"  Average Quality: {perf['average_quality']:.3f}")
            
            if 'recommendations' in summary and summary['recommendations']:
                print(f"\nRecommendations:")
                for rec in summary['recommendations']:
                    print(f"  - {rec}")
        
        # Print individual test results
        for test_name, test_results in results.items():
            if test_name != 'summary' and isinstance(test_results, dict) and 'error' not in test_results:
                print(f"\n{test_name.upper()}:")
                if 'max_throughput' in test_results:
                    print(f"  Max Throughput: {test_results['max_throughput']:.2f} texts/sec")
                if 'average_latency' in test_results:
                    print(f"  Average Latency: {test_results['average_latency']:.3f}s")
                if 'average_memory_delta' in test_results:
                    print(f"  Average Memory Delta: {test_results['average_memory_delta']:.2f} MB")
                if 'average_quality' in test_results:
                    print(f"  Average Quality: {test_results['average_quality']:.3f}")
        
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Performance benchmark execution failed: {e}")
        print(f"\nPerformance benchmark failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())












