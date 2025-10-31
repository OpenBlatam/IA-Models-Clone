"""
GPU Optimization System - Mixed Precision Training
Advanced GPU utilization with automatic mixed precision
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Union
import logging
import time

class GPUOptimizedSystem:
    """Advanced GPU optimization with mixed precision training"""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.device = self._setup_device()
        self.scaler = GradScaler() if config.get('fp16', True) else None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device configuration"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # GPU optimization settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Memory optimization
            if self.config.get('memory_efficient', True):
                torch.cuda.empty_cache()
            
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")
        
        return device
    
    def load_model(self, model_name: str = "gpt2"):
        """Load model with GPU optimization"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimal settings
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.config.get('fp16', True) else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Apply optimizations
            self._apply_model_optimizations()
            
            # Multi-GPU setup
            if self.config.get('use_data_parallel', False) and torch.cuda.device_count() > 1:
                self.model = DataParallel(self.model)
                self.logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            
            self.model.to(self.device)
            self.logger.info(f"Model loaded on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _apply_model_optimizations(self):
        """Apply advanced model optimizations"""
        if self.config.get('compile_model', False):
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.logger.info("Model compiled with torch.compile")
        
        if self.config.get('fp16', True):
            self.model = self.model.half()
            self.logger.info("Model converted to FP16")
        
        if self.config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
    
    def setup_training(self, learning_rate: float = 2e-5):
        """Setup training with GPU optimization"""
        # Optimizer with mixed precision support
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=1e-7
        )
        
        self.logger.info("Training setup complete with GPU optimization")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with mixed precision"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Mixed precision training
        if self.config.get('fp16', True):
            with autocast():
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.get('gradient_accumulation_steps', 1)
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.get('gradient_clipping', True):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.get('gradient_accumulation_steps', 1)
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clipping', True):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text with GPU optimization"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                if self.config.get('fp16', True):
                    with autocast():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=max_length,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error in text generation: {e}")
            return prompt
    
    def benchmark_performance(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark GPU performance"""
        self.model.eval()
        
        # Test prompts
        test_prompts = ["The future of AI is", "Machine learning can", "Deep learning enables"]
        
        # Warmup
        for _ in range(10):
            _ = self.generate_text("Warmup")
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            for prompt in test_prompts:
                _ = self.generate_text(prompt, max_length=50)
        
        total_time = time.time() - start_time
        avg_time = total_time / (num_iterations * len(test_prompts))
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
        else:
            memory_allocated = memory_reserved = 0.0
        
        return {
            'avg_generation_time': avg_time,
            'total_time': total_time,
            'iterations': num_iterations,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'device': str(self.device)
        }
    
    def save_model(self, path: str):
        """Save model with GPU optimization"""
        try:
            if isinstance(self.model, (DataParallel, DistributedDataParallel)):
                torch.save(self.model.module.state_dict(), path)
            else:
                torch.save(self.model.state_dict(), path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'fp16': True,
        'compile_model': False,
        'gradient_checkpointing': False,
        'memory_efficient': True,
        'use_data_parallel': False,
        'gradient_accumulation_steps': 4,
        'gradient_clipping': True,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01
    }
    
    # Initialize GPU optimized system
    gpu_system = GPUOptimizedSystem(config)
    gpu_system.load_model("gpt2")
    gpu_system.setup_training()
    
    # Benchmark performance
    benchmark_results = gpu_system.benchmark_performance()
    print(f"Benchmark Results: {benchmark_results}")
    
    # Generate text
    generated = gpu_system.generate_text("The future of artificial intelligence is")
    print(f"Generated: {generated}")





