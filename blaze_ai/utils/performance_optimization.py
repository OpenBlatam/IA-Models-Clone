"""
Performance optimization utilities following official best practices.
PyTorch, Transformers, Diffusers, and Gradio optimization techniques.
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from typing import Dict, Any, Optional, List, Union, Callable
import time
import psutil
import gc
from contextlib import contextmanager
import warnings

class PerformanceOptimizer:
    """Comprehensive performance optimizer following best practices."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = config.get('mixed_precision', True)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Initialize optimizations
        self._setup_mixed_precision()
        self._setup_memory_optimization()
        self._setup_torch_optimizations()
        self._setup_advanced_optimizations()
        self._setup_library_optimizations()
    
    def _setup_mixed_precision(self):
        """Setup mixed precision following PyTorch best practices."""
        if self.mixed_precision and torch.cuda.is_available():
            self.scaler = amp.GradScaler()
            self.autocast = amp.autocast
            # Use bfloat16 for better numerical stability
            if hasattr(torch, 'bfloat16'):
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.scaler = None
            self.autocast = self._noop_context
            self.dtype = torch.float32
    
    def _setup_memory_optimization(self):
        """Setup memory optimization techniques."""
        if torch.cuda.is_available():
            # Enable memory efficient attention
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            
            # Enable memory efficient algorithms
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory fraction for better memory management
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Enable memory pool for better memory reuse
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.95)
    
    def _setup_torch_optimizations(self):
        """Setup PyTorch-specific optimizations."""
        # Enable channels last memory format for better performance
        if hasattr(torch, 'channels_last'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable advanced optimizations
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _setup_advanced_optimizations(self):
        """Setup advanced performance optimizations."""
        if torch.cuda.is_available():
            # Enable kernel fusion for better performance
            if hasattr(torch.backends.cuda, 'enable_kernel_fusion'):
                torch.backends.cuda.enable_kernel_fusion(True)
            
            # Enable advanced memory management
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.memory_stats()
    
    def _setup_library_optimizations(self):
        """Setup advanced library-specific optimizations."""
        if torch.cuda.is_available():
            # Enable xformers optimizations
            if self.config.get('enable_xformers', True):
                self._setup_xformers()
            
            # Enable flash attention optimizations
            if self.config.get('enable_flash_attn', True):
                self._setup_flash_attention()
            
            # Enable apex optimizations
            if self.config.get('enable_apex', True):
                self._setup_apex()
            
            # Enable triton optimizations
            if self.config.get('enable_triton', True):
                self._setup_triton()
            
            # Enable deepspeed optimizations
            if self.config.get('enable_deepspeed', False):
                self._setup_deepspeed()
    
    def _setup_xformers(self):
        """Setup xformers optimizations."""
        try:
            import xformers
            import xformers.ops as xops
            
            # Enable xformers memory efficient attention
            if hasattr(xops, 'memory_efficient_attention'):
                self.xformers_available = True
                self.xformers_ops = xops
            else:
                self.xformers_available = False
                
        except ImportError:
            warnings.warn("xformers not available, falling back to standard attention")
            self.xformers_available = False
    
    def _setup_flash_attention(self):
        """Setup flash attention optimizations."""
        try:
            import flash_attn
            
            # Enable flash attention
            self.flash_attn_available = True
            self.flash_attn = flash_attn
            
        except ImportError:
            warnings.warn("flash-attn not available, falling back to standard attention")
            self.flash_attn_available = False
    
    def _setup_apex(self):
        """Setup NVIDIA Apex optimizations."""
        try:
            import apex
            from apex import amp as apex_amp
            
            # Enable apex mixed precision
            self.apex_available = True
            self.apex_amp = apex_amp
            
        except ImportError:
            warnings.warn("NVIDIA Apex not available, using PyTorch AMP")
            self.apex_available = False
    
    def _setup_triton(self):
        """Setup Triton optimizations."""
        try:
            import triton
            
            # Enable triton optimizations
            self.triton_available = True
            self.triton = triton
            
        except ImportError:
            warnings.warn("Triton not available")
            self.triton_available = False
    
    def _setup_deepspeed(self):
        """Setup DeepSpeed optimizations."""
        try:
            import deepspeed
            
            # Enable deepspeed optimizations
            self.deepspeed_available = True
            self.deepspeed = deepspeed
            
        except ImportError:
            warnings.warn("DeepSpeed not available")
            self.deepspeed_available = False
    
    @contextmanager
    def _noop_context(self):
        """No-op context manager for when autocast is disabled."""
        yield
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive model optimizations."""
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.config.get('enable_compilation', True):
            try:
                # Use maximum optimization mode
                model = torch.compile(model, mode='max-autotune', fullgraph=True)
            except Exception as e:
                warnings.warn(f"Model compilation failed: {e}, falling back to default mode")
                try:
                    model = torch.compile(model, mode='default')
                except Exception:
                    pass
        
        # Move to device
        model = model.to(self.device)
        
        # Apply memory optimizations
        if torch.cuda.is_available():
            model = self._apply_memory_optimizations(model)
        
        # Convert to channels last for better performance
        if hasattr(torch, 'channels_last') and self.config.get('enable_channels_last', True):
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        
        # Apply quantization if enabled
        if self.config.get('enable_quantization', False):
            model = self._apply_quantization(model)
        
        # Apply library-specific optimizations
        model = self._apply_library_optimizations(model)
        
        return model
    
    def _apply_memory_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply memory optimization techniques."""
        # Enable gradient checkpointing
        if self.config.get('gradient_checkpointing', True):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            elif hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
        
        # Enable memory efficient attention
        if hasattr(model, 'config') and hasattr(model.config, 'attention_mode'):
            model.config.attention_mode = "flash_attention_2"
        
        # Enable memory efficient forward pass
        if hasattr(model, 'config') and hasattr(model.config, 'use_memory_efficient_attention'):
            model.config.use_memory_efficient_attention = True
        
        return model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization for memory efficiency."""
        try:
            if self.config.get('quantization_type') == 'int8':
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
            elif self.config.get('quantization_type') == 'fp16':
                model = model.half()
        except Exception as e:
            warnings.warn(f"Quantization failed: {e}")
        
        return model
    
    def _apply_library_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply library-specific optimizations."""
        # Apply xformers optimizations
        if self.xformers_available and hasattr(model, 'config'):
            model = self._apply_xformers_optimizations(model)
        
        # Apply flash attention optimizations
        if self.flash_attn_available and hasattr(model, 'config'):
            model = self._apply_flash_attention_optimizations(model)
        
        # Apply apex optimizations
        if self.apex_available:
            model = self._apply_apex_optimizations(model)
        
        return model
    
    def _apply_xformers_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply xformers-specific optimizations."""
        try:
            if hasattr(model.config, 'attention_mode'):
                model.config.attention_mode = "xformers"
            
            if hasattr(model.config, 'use_memory_efficient_attention'):
                model.config.use_memory_efficient_attention = True
            
            # Enable xformers memory efficient attention
            if hasattr(model, 'enable_xformers_memory_efficient_attention'):
                model.enable_xformers_memory_efficient_attention()
                
        except Exception as e:
            warnings.warn(f"xformers optimizations failed: {e}")
        
        return model
    
    def _apply_flash_attention_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply flash attention optimizations."""
        try:
            if hasattr(model.config, 'attention_mode'):
                model.config.attention_mode = "flash_attention_2"
            
            if hasattr(model.config, 'use_flash_attention'):
                model.config.use_flash_attention = True
                
        except Exception as e:
            warnings.warn(f"Flash attention optimizations failed: {e}")
        
        return model
    
    def _apply_apex_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply NVIDIA Apex optimizations."""
        try:
            # Enable apex mixed precision if available
            if self.config.get('enable_apex_amp', True):
                model = self.apex_amp.initialize(model, opt_level='O2')
                
        except Exception as e:
            warnings.warn(f"Apex optimizations failed: {e}")
        
        return model
    
    def setup_multi_gpu(self, model: nn.Module, gpu_ids: List[int]) -> nn.Module:
        """Setup multi-GPU training following best practices."""
        if len(gpu_ids) > 1:
            if self.config.get('distributed_training', False):
                # Distributed training
                model = DistributedDataParallel(
                    model,
                    device_ids=[self.device],
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                    static_graph=True,
                    broadcast_buffers=False
                )
            else:
                # DataParallel with optimizations
                model = DataParallel(
                    model, 
                    device_ids=gpu_ids,
                    dim=0,
                    output_device=gpu_ids[0]
                )
        
        return model
    
    def optimize_data_loading(self, dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """Optimize data loading following best practices."""
        # Set optimal number of workers
        if torch.cuda.is_available():
            optimal_workers = min(
                self.config.get('num_workers', 4),
                psutil.cpu_count(),
                torch.cuda.device_count() * 2
            )
            dataloader.num_workers = optimal_workers
        
        # Enable memory pinning
        dataloader.pin_memory = True
        
        # Set prefetch factor
        dataloader.prefetch_factor = self.config.get('prefetch_factor', 2)
        
        # Enable persistent workers
        dataloader.persistent_workers = True
        
        # Set pin memory device for better performance
        if hasattr(dataloader, 'pin_memory_device') and torch.cuda.is_available():
            dataloader.pin_memory_device = f"cuda:{gpu_ids[0] if 'gpu_ids' in locals() else 0}"
        
        # Enable auto-tuning for optimal performance
        if hasattr(dataloader, 'autotune_dataloader'):
            dataloader.autotune_dataloader()
        
        return dataloader
    
    def optimize_training_step(self, 
                             model: nn.Module,
                             optimizer: torch.optim.Optimizer,
                             loss_fn: Callable,
                             data: torch.Tensor,
                             targets: torch.Tensor) -> Dict[str, float]:
        """Optimized training step with mixed precision."""
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Forward pass with mixed precision
        with self.autocast():
            outputs = model(data)
            loss = loss_fn(outputs, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.get('gradient_clipping', True):
            if self.scaler:
                self.scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
        
        # Optimizer step
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        return {'loss': loss.item()}
    
    def optimize_inference(self, model: nn.Module, input_data: torch.Tensor) -> torch.Tensor:
        """Optimized inference following best practices."""
        model.eval()
        
        with torch.no_grad():
            with self.autocast():
                if torch.cuda.is_available():
                    input_data = input_data.to(self.device, non_blocking=True)
                
                # Use channels last for better performance
                if hasattr(torch, 'channels_last') and self.config.get('enable_channels_last', True):
                    try:
                        input_data = input_data.to(memory_format=torch.channels_last)
                    except Exception:
                        pass
                
                outputs = model(input_data)
        
        return outputs
    
    def memory_cleanup(self):
        """Clean up memory following best practices."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset peak memory stats
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        
        gc.collect()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'device': str(self.device),
            'mixed_precision': self.mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'dtype': str(self.dtype),
            'xformers_available': getattr(self, 'xformers_available', False),
            'flash_attn_available': getattr(self, 'flash_attn_available', False),
            'apex_available': getattr(self, 'apex_available', False),
            'triton_available': getattr(self, 'triton_available', False),
            'deepspeed_available': getattr(self, 'deepspeed_available', False)
        }
        
        if torch.cuda.is_available():
            stats.update({
                'cuda_version': torch.version.cuda,
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                'gpu_memory_free': torch.cuda.get_device_properties(0).total_memory / 1024**3 - torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_peak': torch.cuda.max_memory_allocated() / 1024**3 if hasattr(torch.cuda, 'max_memory_allocated') else 0,
                'gpu_utilization': self._get_gpu_utilization()
            })
        
        return stats
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            if hasattr(torch.cuda, 'utilization'):
                return torch.cuda.utilization()
            return 0.0
        except Exception:
            return 0.0

class DiffusionOptimizer:
    """Diffusion model optimizer following Diffusers best practices."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_library_optimizations()
    
    def _setup_library_optimizations(self):
        """Setup library-specific optimizations for diffusion models."""
        # Setup xformers
        if self.config.get('enable_xformers', True):
            try:
                import xformers
                self.xformers_available = True
            except ImportError:
                self.xformers_available = False
        
        # Setup flash attention
        if self.config.get('enable_flash_attn', True):
            try:
                import flash_attn
                self.flash_attn_available = True
            except ImportError:
                self.flash_attn_available = False
        
        # Setup triton
        if self.config.get('enable_triton', True):
            try:
                import triton
                self.triton_available = True
            except ImportError:
                self.triton_available = False
    
    def optimize_pipeline(self, pipeline) -> Any:
        """Apply diffusion pipeline optimizations."""
        try:
            # Enable xformers for faster attention
            if self.xformers_available and hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                pipeline.enable_xformers_memory_efficient_attention()
            
            # Enable flash attention
            if self.flash_attn_available and hasattr(pipeline, 'enable_flash_attention'):
                pipeline.enable_flash_attention()
            
            # Enable attention slicing
            if self.config.get('enable_attention_slicing', True):
                pipeline.enable_attention_slicing()
            
            # Enable VAE slicing
            if self.config.get('enable_vae_slicing', True):
                pipeline.enable_vae_slicing()
            
            # Enable CPU offload
            if self.config.get('enable_cpu_offload', False):
                pipeline.enable_sequential_cpu_offload()
            
            # Set memory efficient attention
            if self.config.get('memory_efficient_attention', True):
                try:
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    pipeline.unet.set_attn_processor(AttnProcessor2_0())
                except ImportError:
                    pass
            
            # Enable model offloading for memory efficiency
            if self.config.get('enable_model_offloading', True) and hasattr(pipeline, 'enable_model_cpu_offload'):
                pipeline.enable_model_cpu_offload()
            
            # Set optimal scheduler parameters
            if hasattr(pipeline, 'scheduler'):
                pipeline.scheduler = self.optimize_scheduler(pipeline.scheduler)
            
        except Exception as e:
            warnings.warn(f"Some optimizations failed: {e}")
        
        return pipeline
    
    def optimize_scheduler(self, scheduler, scheduler_type: str = "ddim"):
        """Optimize noise scheduler following best practices."""
        if scheduler_type == "ddim":
            scheduler.beta_start = 0.00085
            scheduler.beta_end = 0.012
            scheduler.beta_schedule = "scaled_linear"
        elif scheduler_type == "dpm-solver":
            scheduler.beta_start = 0.00085
            scheduler.beta_end = 0.012
            scheduler.beta_schedule = "scaled_linear"
        elif scheduler_type == "euler":
            scheduler.beta_start = 0.00085
            scheduler.beta_end = 0.012
            scheduler.beta_schedule = "scaled_linear"
        
        return scheduler

class GradioOptimizer:
    """Gradio interface optimizer following best practices."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def optimize_interface(self, interface: Any) -> Any:
        """Apply Gradio interface optimizations."""
        # Set optimal queue settings
        if hasattr(interface, 'queue'):
            interface.queue(
                max_size=self.config.get('max_queue_size', 20),
                concurrency_count=self.config.get('concurrency_count', 1),
                status_update_rate=10
            )
        
        # Enable caching for expensive operations
        if self.config.get('enable_caching', True):
            interface.cache_examples = True
        
        # Set optimal server settings
        if hasattr(interface, 'server_name'):
            interface.server_name = self.config.get('server_name', '0.0.0.0')
        
        if hasattr(interface, 'server_port'):
            interface.server_port = self.config.get('server_port', 7860)
        
        return interface
    
    def optimize_components(self, components: List[Any]) -> List[Any]:
        """Optimize individual Gradio components."""
        for component in components:
            if hasattr(component, 'container'):
                component.container = True
            
            if hasattr(component, 'show_label'):
                component.show_label = True
            
            # Enable autofocus for better UX
            if hasattr(component, 'autofocus'):
                component.autofocus = True
        
        return components

class MemoryManager:
    """Memory management following best practices."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def monitor_memory(self) -> Dict[str, float]:
        """Monitor memory usage."""
        stats = {
            'cpu_memory_percent': psutil.virtual_memory().percent,
            'cpu_memory_available_gb': psutil.virtual_memory().available / 1024**3,
            'cpu_memory_used_gb': psutil.virtual_memory().used / 1024**3,
            'cpu_memory_total_gb': psutil.virtual_memory().total / 1024**3
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'gpu_memory_free_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 - torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_peak_gb': torch.cuda.max_memory_allocated() / 1024**3 if hasattr(torch.cuda, 'max_memory_allocated') else 0
            })
        
        return stats
    
    def optimize_memory(self):
        """Apply memory optimization techniques."""
        # CPU memory optimization
        gc.collect()
        
        # GPU memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset peak memory stats
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            if gpu_memory > 8:  # More than 8GB
                recommendations.append("Consider reducing batch size")
                recommendations.append("Enable gradient checkpointing")
                recommendations.append("Use mixed precision training")
                recommendations.append("Enable model offloading")
            
            if gpu_memory > 12:  # More than 12GB
                recommendations.append("Consider using CPU offloading")
                recommendations.append("Reduce model size or use quantization")
        
        cpu_memory = psutil.virtual_memory().percent
        if cpu_memory > 80:
            recommendations.append("Reduce number of data loader workers")
            recommendations.append("Consider using smaller datasets")
            recommendations.append("Enable memory-efficient data loading")
        
        return recommendations

# Factory functions
def create_performance_optimizer(config: Dict[str, Any]) -> PerformanceOptimizer:
    """Create performance optimizer instance."""
    return PerformanceOptimizer(config)

def create_diffusion_optimizer(config: Dict[str, Any]) -> DiffusionOptimizer:
    """Create diffusion optimizer instance."""
    return DiffusionOptimizer(config)

def create_gradio_optimizer(config: Dict[str, Any]) -> GradioOptimizer:
    """Create Gradio optimizer instance."""
    return GradioOptimizer(config)

def create_memory_manager() -> MemoryManager:
    """Create memory manager instance."""
    return MemoryManager()

__all__ = [
    "PerformanceOptimizer",
    "DiffusionOptimizer", 
    "GradioOptimizer",
    "MemoryManager",
    "create_performance_optimizer",
    "create_diffusion_optimizer",
    "create_gradio_optimizer",
    "create_memory_manager"
]
