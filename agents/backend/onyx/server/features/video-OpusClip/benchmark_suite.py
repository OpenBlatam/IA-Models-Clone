#!/usr/bin/env python3
"""
Benchmark Suite for Video-OpusClip

Comprehensive performance testing and benchmarking for all components.
"""

import asyncio
import time
import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_libraries import get_optimized_components, optimize_memory
from performance_monitor import get_benchmark_runner
from optimized_cache import get_cache_manager
from optimized_config import get_config

class VideoOpusClipBenchmark:
    """Comprehensive benchmark suite for Video-OpusClip."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.components = get_optimized_components(self.device)
        self.benchmark_runner = get_benchmark_runner()
        self.results = {}
        
    async def benchmark_video_encoder(self):
        """Benchmark video encoder performance."""
        print("🎬 Benchmarking video encoder...")
        
        encoder = self.components['video_encoder']
        
        # Test data
        batch_sizes = [1, 4, 8, 16]
        frame_counts = [10, 30, 60]
        
        for batch_size in batch_sizes:
            for num_frames in frame_counts:
                # Create dummy video data
                video_data = torch.randn(
                    batch_size, num_frames, 3, 224, 224,
                    device=self.device
                )
                
                if self.device == "cuda":
                    video_data = video_data.half()
                
                # Benchmark
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    for _ in range(10):  # Warmup
                        _ = encoder(video_data)
                    
                    torch.cuda.synchronize() if self.device == "cuda" else None
                    start_benchmark = time.perf_counter()
                    
                    for _ in range(50):  # Actual benchmark
                        _ = encoder(video_data)
                    
                    torch.cuda.synchronize() if self.device == "cuda" else None
                    end_time = time.perf_counter()
                
                # Calculate metrics
                total_time = end_time - start_benchmark
                avg_time = total_time / 50
                throughput = batch_size / avg_time
                
                key = f"encoder_b{batch_size}_f{num_frames}"
                self.results[key] = {
                    'avg_time': avg_time,
                    'throughput': throughput,
                    'batch_size': batch_size,
                    'num_frames': num_frames
                }
                
                print(f"  Batch {batch_size}, Frames {num_frames}: {avg_time:.4f}s, {throughput:.2f} samples/s")
    
    async def benchmark_diffusion_pipeline(self):
        """Benchmark diffusion pipeline performance."""
        print("🎨 Benchmarking diffusion pipeline...")
        
        pipeline = self.components['diffusion_pipeline']
        
        # Test configurations
        prompts = [
            "A beautiful sunset over mountains",
            "A cat playing with a ball",
            "A car driving on a highway"
        ]
        
        for i, prompt in enumerate(prompts):
            start_time = time.perf_counter()
            
            # Generate frames
            frames = pipeline.generate_video_frames(
                prompt=prompt,
                num_frames=10,
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=7.5
            )
            
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            avg_time_per_frame = total_time / len(frames)
            
            key = f"diffusion_prompt_{i}"
            self.results[key] = {
                'total_time': total_time,
                'avg_time_per_frame': avg_time_per_frame,
                'num_frames': len(frames),
                'prompt': prompt
            }
            
            print(f"  Prompt {i+1}: {total_time:.2f}s total, {avg_time_per_frame:.3f}s per frame")
    
    async def benchmark_text_processor(self):
        """Benchmark text processor performance."""
        print("📝 Benchmarking text processor...")
        
        processor = self.components['text_processor']
        
        # Test descriptions
        descriptions = [
            "A funny cat video with music",
            "A cooking tutorial with step-by-step instructions",
            "A travel vlog showing beautiful landscapes"
        ]
        
        for i, description in enumerate(descriptions):
            start_time = time.perf_counter()
            
            # Generate captions
            captions = []
            for _ in range(10):
                caption = processor.generate_caption(description)
                captions.append(caption)
            
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            avg_time_per_caption = total_time / len(captions)
            
            key = f"text_processor_desc_{i}"
            self.results[key] = {
                'total_time': total_time,
                'avg_time_per_caption': avg_time_per_caption,
                'num_captions': len(captions),
                'description': description
            }
            
            print(f"  Description {i+1}: {total_time:.2f}s total, {avg_time_per_caption:.3f}s per caption")
    
    async def benchmark_cache_performance(self):
        """Benchmark cache performance."""
        print("💾 Benchmarking cache performance...")
        
        cache_manager = get_cache_manager()
        
        # Test data
        test_data = {
            'video_analysis': {'url': 'test_url', 'analysis': 'test_analysis'},
            'viral_prediction': {'content': 'test_content', 'prediction': 0.85},
            'caption_generation': {'prompt': 'test_prompt', 'caption': 'test_caption'}
        }
        
        # Test cache operations
        for key, value in test_data.items():
            # Set operation
            start_time = time.perf_counter()
            await cache_manager.set_video_analysis(key, 'en', 'tiktok', value)
            set_time = time.perf_counter() - start_time
            
            # Get operation
            start_time = time.perf_counter()
            cached_value = await cache_manager.get_video_analysis(key, 'en', 'tiktok')
            get_time = time.perf_counter() - start_time
            
            key_name = f"cache_{key}"
            self.results[key_name] = {
                'set_time': set_time,
                'get_time': get_time,
                'hit': cached_value is not None
            }
            
            print(f"  {key}: Set {set_time:.4f}s, Get {get_time:.4f}s, Hit: {cached_value is not None}")
    
    async def benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        print("🧠 Benchmarking memory usage...")
        
        import psutil
        import gc
        
        # Baseline memory
        baseline_memory = psutil.virtual_memory().used / 1024**3  # GB
        
        # Test memory-intensive operations
        operations = [
            ('large_tensor_creation', self._create_large_tensors),
            ('model_loading', self._load_models),
            ('batch_processing', self._batch_processing)
        ]
        
        for op_name, op_func in operations:
            # Clear memory before test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            start_memory = psutil.virtual_memory().used / 1024**3
            
            # Run operation
            start_time = time.perf_counter()
            peak_memory = await op_func()
            end_time = time.perf_counter()
            
            end_memory = psutil.virtual_memory().used / 1024**3
            
            key = f"memory_{op_name}"
            self.results[key] = {
                'start_memory_gb': start_memory,
                'peak_memory_gb': peak_memory,
                'end_memory_gb': end_memory,
                'duration': end_time - start_time,
                'memory_increase_gb': end_memory - start_memory
            }
            
            print(f"  {op_name}: {start_memory:.2f}GB → {peak_memory:.2f}GB → {end_memory:.2f}GB")
    
    async def _create_large_tensors(self):
        """Create large tensors for memory testing."""
        import psutil
        
        tensors = []
        peak_memory = psutil.virtual_memory().used / 1024**3
        
        for i in range(10):
            tensor = torch.randn(1000, 1000, device=self.device)
            tensors.append(tensor)
            
            current_memory = psutil.virtual_memory().used / 1024**3
            peak_memory = max(peak_memory, current_memory)
        
        # Cleanup
        del tensors
        return peak_memory
    
    async def _load_models(self):
        """Load models for memory testing."""
        import psutil
        
        models = []
        peak_memory = psutil.virtual_memory().used / 1024**3
        
        # Load multiple models
        for i in range(3):
            model = torch.nn.Sequential(
                torch.nn.Linear(1000, 1000),
                torch.nn.ReLU(),
                torch.nn.Linear(1000, 1000)
            ).to(self.device)
            models.append(model)
            
            current_memory = psutil.virtual_memory().used / 1024**3
            peak_memory = max(peak_memory, current_memory)
        
        # Cleanup
        del models
        return peak_memory
    
    async def _batch_processing(self):
        """Batch processing for memory testing."""
        import psutil
        
        peak_memory = psutil.virtual_memory().used / 1024**3
        
        # Process large batches
        for batch_size in [32, 64, 128]:
            batch = torch.randn(batch_size, 3, 224, 224, device=self.device)
            
            # Simulate processing
            processed = torch.nn.functional.conv2d(
                batch, 
                torch.randn(64, 3, 3, 3, device=self.device)
            )
            
            current_memory = psutil.virtual_memory().used / 1024**3
            peak_memory = max(peak_memory, current_memory)
            
            del batch, processed
        
        return peak_memory
    
    async def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("🚀 Starting comprehensive benchmark suite...")
        print(f"📱 Device: {self.device}")
        print(f"🔧 Components: {list(self.components.keys())}")
        print()
        
        # Run all benchmarks
        await self.benchmark_video_encoder()
        print()
        
        await self.benchmark_diffusion_pipeline()
        print()
        
        await self.benchmark_text_processor()
        print()
        
        await self.benchmark_cache_performance()
        print()
        
        await self.benchmark_memory_usage()
        print()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive benchmark report."""
        print("📊 Generating benchmark report...")
        print("=" * 60)
        
        # Video Encoder Performance
        print("🎬 VIDEO ENCODER PERFORMANCE")
        print("-" * 30)
        encoder_results = {k: v for k, v in self.results.items() if k.startswith('encoder')}
        for key, result in encoder_results.items():
            print(f"  {key}: {result['avg_time']:.4f}s, {result['throughput']:.2f} samples/s")
        
        print()
        
        # Diffusion Pipeline Performance
        print("🎨 DIFFUSION PIPELINE PERFORMANCE")
        print("-" * 35)
        diffusion_results = {k: v for k, v in self.results.items() if k.startswith('diffusion')}
        for key, result in diffusion_results.items():
            print(f"  {key}: {result['total_time']:.2f}s total, {result['avg_time_per_frame']:.3f}s per frame")
        
        print()
        
        # Text Processor Performance
        print("📝 TEXT PROCESSOR PERFORMANCE")
        print("-" * 30)
        text_results = {k: v for k, v in self.results.items() if k.startswith('text_processor')}
        for key, result in text_results.items():
            print(f"  {key}: {result['total_time']:.2f}s total, {result['avg_time_per_caption']:.3f}s per caption")
        
        print()
        
        # Cache Performance
        print("💾 CACHE PERFORMANCE")
        print("-" * 20)
        cache_results = {k: v for k, v in self.results.items() if k.startswith('cache')}
        for key, result in cache_results.items():
            print(f"  {key}: Set {result['set_time']:.4f}s, Get {result['get_time']:.4f}s, Hit: {result['hit']}")
        
        print()
        
        # Memory Usage
        print("🧠 MEMORY USAGE")
        print("-" * 15)
        memory_results = {k: v for k, v in self.results.items() if k.startswith('memory')}
        for key, result in memory_results.items():
            print(f"  {key}: {result['memory_increase_gb']:.2f}GB increase, {result['duration']:.2f}s duration")
        
        print()
        print("=" * 60)
        print("✅ Benchmark suite completed!")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save benchmark results to file."""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        # Convert numpy types to native Python types
        serializable_results = {}
        for key, value in self.results.items():
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.integer, np.floating)):
                    serializable_results[key][k] = v.item()
                else:
                    serializable_results[key][k] = v
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")

async def main():
    """Main benchmark runner."""
    benchmark = VideoOpusClipBenchmark()
    await benchmark.run_full_benchmark()

if __name__ == "__main__":
    asyncio.run(main()) 