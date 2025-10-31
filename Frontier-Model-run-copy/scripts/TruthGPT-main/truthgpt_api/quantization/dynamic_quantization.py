"""
Dynamic Quantization for TruthGPT API
===================================

TensorFlow-like dynamic quantization implementation.
"""

import torch
import torch.quantization as quant
from typing import Any, Optional, Dict, List


class DynamicQuantization:
    """
    Dynamic quantization for models.
    
    Similar to tf.keras.quantization.dynamic_quantization, this class
    implements dynamic quantization for PyTorch models.
    """
    
    def __init__(self, 
                 dtype: torch.dtype = torch.qint8,
                 name: Optional[str] = None):
        """
        Initialize DynamicQuantization.
        
        Args:
            dtype: Quantization data type
            name: Optional name for the quantizer
        """
        self.dtype = dtype
        self.name = name or "dynamic_quantization"
        
        self.quantized_model = None
        self.original_model = None
    
    def quantize(self, model: Any, 
                 sample_input: Optional[torch.Tensor] = None) -> Any:
        """
        Quantize a model.
        
        Args:
            model: Model to quantize
            sample_input: Sample input for quantization
            
        Returns:
            Quantized model
        """
        print(f"🔧 Applying dynamic quantization...")
        
        # Store original model
        self.original_model = model
        
        # Set model to evaluation mode
        model.eval()
        
        # Apply dynamic quantization
        if self.dtype == torch.qint8:
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
        elif self.dtype == torch.float16:
            quantized_model = model.half()
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        self.quantized_model = quantized_model
        
        print(f"✅ Dynamic quantization completed!")
        print(f"   Original model size: {self._get_model_size(model):.2f} MB")
        print(f"   Quantized model size: {self._get_model_size(quantized_model):.2f} MB")
        print(f"   Compression ratio: {self._get_compression_ratio():.2f}x")
        
        return quantized_model
    
    def _get_model_size(self, model: Any) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def _get_compression_ratio(self) -> float:
        """Get compression ratio."""
        if self.original_model is None or self.quantized_model is None:
            return 1.0
        
        original_size = self._get_model_size(self.original_model)
        quantized_size = self._get_model_size(self.quantized_model)
        
        return original_size / quantized_size if quantized_size > 0 else 1.0
    
    def benchmark(self, 
                 model: Any,
                 quantized_model: Any,
                 sample_input: torch.Tensor,
                 num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark quantized model.
        
        Args:
            model: Original model
            quantized_model: Quantized model
            sample_input: Sample input
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        print(f"📊 Benchmarking quantized model...")
        
        # Benchmark original model
        original_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = model(sample_input)
                end_time.record()
                
                torch.cuda.synchronize()
                original_times.append(start_time.elapsed_time(end_time))
        
        # Benchmark quantized model
        quantized_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = quantized_model(sample_input)
                end_time.record()
                
                torch.cuda.synchronize()
                quantized_times.append(start_time.elapsed_time(end_time))
        
        # Calculate statistics
        original_avg = sum(original_times) / len(original_times)
        quantized_avg = sum(quantized_times) / len(quantized_times)
        
        speedup = original_avg / quantized_avg if quantized_avg > 0 else 1.0
        
        results = {
            'original_time': original_avg,
            'quantized_time': quantized_avg,
            'speedup': speedup,
            'compression_ratio': self._get_compression_ratio()
        }
        
        print(f"✅ Benchmark completed!")
        print(f"   Original time: {original_avg:.2f} ms")
        print(f"   Quantized time: {quantized_avg:.2f} ms")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Compression: {self._get_compression_ratio():.2f}x")
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get quantization configuration."""
        return {
            'name': self.name,
            'dtype': str(self.dtype),
            'type': 'dynamic'
        }
    
    def __repr__(self):
        return f"DynamicQuantization(dtype={self.dtype})"


