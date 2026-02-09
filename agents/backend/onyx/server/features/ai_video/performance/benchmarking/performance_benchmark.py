from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import time
import psutil
import GPUtil
from transformers import AutoModel, AutoTokenizer
from diffusers import TextToVideoPipeline
import numpy as np

        from transformers import BitsAndBytesConfig
from typing import Any, List, Dict, Optional
import logging
import asyncio
class PerformanceBenchmark:
    def __init__(self) -> Any:
        self.results = {}
    
    def benchmark_pytorch_features(self) -> Any:
        """Benchmark PyTorch 2.0+ features."""
        print("Benchmarking PyTorch features...")
        
        # Test torch.compile
        if hasattr(torch, 'compile'):
            model = torch.nn.Linear(1024, 512)
            x = torch.randn(100, 1024).cuda()
            
            # Standard model
            start_time = time.time()
            for _ in range(100):
                _ = model(x)
            torch.cuda.synchronize()
            standard_time = time.time() - start_time
            
            # Compiled model
            compiled_model = torch.compile(model, mode="reduce-overhead")
            start_time = time.time()
            for _ in range(100):
                _ = compiled_model(x)
            torch.cuda.synchronize()
            compiled_time = time.time() - start_time
            
            speedup = standard_time / compiled_time
            self.results['torch_compile_speedup'] = speedup
            print(f"Torch compile speedup: {speedup:.2f}x")
        
        # Test flash attention
        torch.backends.cuda.enable_flash_sdp(True)
        attention = torch.nn.MultiheadAttention(512, 8, batch_first=True).cuda()
        x = torch.randn(32, 100, 512).cuda()
        
        start_time = time.time()
        for _ in range(50):
            _ = attention(x, x, x)
        torch.cuda.synchronize()
        attention_time = time.time() - start_time
        
        self.results['attention_time'] = attention_time
        print(f"Attention time: {attention_time:.4f}s")
    
    def benchmark_transformers(self) -> Any:
        """Benchmark Transformers optimizations."""
        print("Benchmarking Transformers...")
        
        # Test quantization
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model with quantization
        model = AutoModel.from_pretrained(
            "bert-base-uncased",
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
        
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model(**inputs)
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        self.results['transformers_inference_time'] = inference_time
        print(f"Transformers inference time: {inference_time:.4f}s")
    
    def benchmark_diffusers(self) -> Any:
        """Benchmark Diffusers pipeline."""
        print("Benchmarking Diffusers...")
        
        pipeline = TextToVideoPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16
        )
        
        # Enable optimizations
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()
        pipeline = pipeline.to("cuda")
        
        start_time = time.time()
        video_frames = pipeline(
            "A cat walking",
            num_inference_steps=20,
            height=256,
            width=256,
            num_frames=8
        )
        torch.cuda.synchronize()
        video_time = time.time() - start_time
        
        self.results['video_generation_time'] = video_time
        print(f"Video generation time: {video_time:.2f}s")
    
    def get_system_stats(self) -> Optional[Dict[str, Any]]:
        """Get system performance stats."""
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / 1024**3,
            "memory_total_gb": psutil.virtual_memory().total / 1024**3
        }
        
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                stats[f"gpu_{i}_load"] = gpu.load * 100
                stats[f"gpu_{i}_memory_used"] = gpu.memoryUsed
                stats[f"gpu_{i}_memory_total"] = gpu.memoryTotal
                stats[f"gpu_{i}_temperature"] = gpu.temperature
        except:
            pass
        
        self.results['system_stats'] = stats
        return stats
    
    def run_all_benchmarks(self) -> Any:
        """Run all benchmarks."""
        print("🚀 Starting Performance Benchmarks")
        print("=" * 50)
        
        self.benchmark_pytorch_features()
        self.benchmark_transformers()
        self.benchmark_diffusers()
        self.get_system_stats()
        
        print("\n📊 Benchmark Results:")
        print("=" * 50)
        for key, value in self.results.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        return self.results

if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks() 