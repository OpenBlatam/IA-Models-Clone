from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import json
import os
from pathlib import Path
from efficient_finetuning import (
from transformer_models import TransformerConfig, SEOSpecificTransformer, TransformerManager
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Example script demonstrating efficient fine-tuning techniques
Comprehensive examples of LoRA, P-tuning, AdaLoRA, and other PEFT methods
"""


# Import our efficient fine-tuning modules
    LoRAConfig, PEFTConfig, EfficientFineTuningManager, PEFTTrainer,
    create_peft_config, apply_peft_to_model
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientFineTuningDemo:
    """Demonstration class for efficient fine-tuning techniques"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.transformer_manager = TransformerManager()
        logger.info(f"Using device: {self.device}")
    
    def demonstrate_lora(self) -> Any:
        """Demonstrate LoRA (Low-Rank Adaptation)"""
        logger.info("=== Demonstrating LoRA (Low-Rank Adaptation) ===")
        
        # Create base transformer
        config = TransformerConfig(
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            intermediate_size=1024,
            vocab_size=10000
        )
        
        model = SEOSpecificTransformer(config)
        model = model.to(self.device)
        
        # Original parameter count
        original_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Original model parameters: {original_params:,}")
        
        # Apply LoRA
        lora_config = LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["w_q", "w_k", "w_v", "w_o"]
        )
        
        peft_config = PEFTConfig(
            peft_type="LORA",
            lora_config=lora_config
        )
        
        peft_manager = EfficientFineTuningManager(model, peft_config)
        
        # Parameter statistics
        param_stats = peft_manager.get_parameter_count()
        logger.info(f"After LoRA:")
        logger.info(f"  Total parameters: {param_stats['total_parameters']:,}")
        logger.info(f"  Trainable parameters: {param_stats['trainable_parameters']:,}")
        logger.info(f"  Trainable percentage: {param_stats['trainable_percentage']:.2f}%")
        
        # Test forward pass
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
        attention_mask = torch.ones(batch_size, seq_len).to(self.device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logger.info(f"Output shape: {outputs['last_hidden_state'].shape}")
        
        return model, peft_manager
    
    def demonstrate_p_tuning(self) -> Any:
        """Demonstrate P-tuning"""
        logger.info("\n=== Demonstrating P-tuning ===")
        
        # Create base transformer
        config = TransformerConfig(
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            intermediate_size=1024,
            vocab_size=10000
        )
        
        model = SEOSpecificTransformer(config)
        model = model.to(self.device)
        
        # Apply P-tuning
        peft_config = PEFTConfig(
            peft_type="P_TUNING",
            num_virtual_tokens=20,
            encoder_hidden_size=128,
            encoder_num_layers=2,
            encoder_dropout=0.1
        )
        
        peft_manager = EfficientFineTuningManager(model, peft_config)
        
        # Parameter statistics
        param_stats = peft_manager.get_parameter_count()
        logger.info(f"After P-tuning:")
        logger.info(f"  Total parameters: {param_stats['total_parameters']:,}")
        logger.info(f"  Trainable parameters: {param_stats['trainable_parameters']:,}")
        logger.info(f"  Trainable percentage: {param_stats['trainable_percentage']:.2f}%")
        
        # Test forward pass with virtual tokens
        batch_size = 2
        seq_len = 64
        
        # Generate virtual token embeddings
        virtual_tokens = peft_manager.p_tuning_embeddings(batch_size)
        logger.info(f"Virtual tokens shape: {virtual_tokens.shape}")
        
        # Combine with regular input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
        attention_mask = torch.ones(batch_size, seq_len + peft_config.num_virtual_tokens).to(self.device)
        
        # Create combined embeddings
        regular_embeddings = model.embeddings(input_ids)
        combined_embeddings = torch.cat([virtual_tokens, regular_embeddings], dim=1)
        
        logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
        
        return model, peft_manager
    
    def demonstrate_adalora(self) -> Any:
        """Demonstrate AdaLoRA (Adaptive LoRA)"""
        logger.info("\n=== Demonstrating AdaLoRA (Adaptive LoRA) ===")
        
        # Create base transformer
        config = TransformerConfig(
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            intermediate_size=1024,
            vocab_size=10000
        )
        
        model = SEOSpecificTransformer(config)
        model = model.to(self.device)
        
        # Apply AdaLoRA
        peft_config = PEFTConfig(
            peft_type="ADALORA",
            target_modules=["w_q", "w_k", "w_v", "w_o"],
            init_r=12,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            deltaT=10,
            orth_reg_weight=0.5
        )
        
        peft_manager = EfficientFineTuningManager(model, peft_config)
        
        # Parameter statistics
        param_stats = peft_manager.get_parameter_count()
        logger.info(f"After AdaLoRA:")
        logger.info(f"  Total parameters: {param_stats['total_parameters']:,}")
        logger.info(f"  Trainable parameters: {param_stats['trainable_parameters']:,}")
        logger.info(f"  Trainable percentage: {param_stats['trainable_percentage']:.2f}%")
        
        # Test forward pass
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
        attention_mask = torch.ones(batch_size, seq_len).to(self.device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logger.info(f"Output shape: {outputs['last_hidden_state'].shape}")
        
        return model, peft_manager
    
    def demonstrate_prefix_tuning(self) -> Any:
        """Demonstrate Prefix Tuning"""
        logger.info("\n=== Demonstrating Prefix Tuning ===")
        
        # Create base transformer
        config = TransformerConfig(
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            intermediate_size=1024,
            vocab_size=10000
        )
        
        model = SEOSpecificTransformer(config)
        model = model.to(self.device)
        
        # Apply Prefix Tuning
        peft_config = PEFTConfig(
            peft_type="PREFIX_TUNING",
            num_prefix_tokens=20,
            prefix_projection=False
        )
        
        peft_manager = EfficientFineTuningManager(model, peft_config)
        
        # Parameter statistics
        param_stats = peft_manager.get_parameter_count()
        logger.info(f"After Prefix Tuning:")
        logger.info(f"  Total parameters: {param_stats['total_parameters']:,}")
        logger.info(f"  Trainable parameters: {param_stats['trainable_parameters']:,}")
        logger.info(f"  Trainable percentage: {param_stats['trainable_percentage']:.2f}%")
        
        # Test prefix embeddings
        batch_size = 2
        for layer_idx in range(config.num_layers):
            prefix_emb = peft_manager.prefix_embeddings.get_prefix_embeddings(layer_idx, batch_size)
            logger.info(f"Layer {layer_idx} prefix embeddings shape: {prefix_emb.shape}")
        
        return model, peft_manager
    
    def demonstrate_training(self) -> Any:
        """Demonstrate training with PEFT"""
        logger.info("\n=== Demonstrating PEFT Training ===")
        
        # Create model and apply LoRA
        config = TransformerConfig(
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
            vocab_size=1000
        )
        
        model = SEOSpecificTransformer(config)
        model = model.to(self.device)
        
        # Apply LoRA
        peft_config = create_peft_config(
            peft_type="LORA",
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["w_q", "w_k", "w_v", "w_o"]
        )
        
        peft_manager = EfficientFineTuningManager(model, peft_config)
        
        # Create trainer
        trainer = PEFTTrainer(model, peft_config)
        
        # Create dummy training data
        batch_size = 4
        seq_len = 32
        num_batches = 10
        
        logger.info("Starting training...")
        
        for batch_idx in range(num_batches):
            # Create dummy batch
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
            attention_mask = torch.ones(batch_size, seq_len).to(self.device)
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
            
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            # Training step
            metrics = trainer.train_step(batch)
            
            if batch_idx % 2 == 0:
                logger.info(f"Batch {batch_idx}: Loss = {metrics['loss']:.4f}, LR = {metrics['lr']:.6f}")
        
        # Parameter statistics after training
        param_stats = peft_manager.get_parameter_count()
        logger.info(f"Final parameter statistics:")
        logger.info(f"  Trainable parameters: {param_stats['trainable_parameters']:,}")
        logger.info(f"  Trainable percentage: {param_stats['trainable_percentage']:.2f}%")
        
        return model, peft_manager, trainer
    
    def demonstrate_save_load(self) -> Any:
        """Demonstrate saving and loading PEFT models"""
        logger.info("\n=== Demonstrating Save/Load Functionality ===")
        
        # Create model and apply LoRA
        config = TransformerConfig(
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
            vocab_size=1000
        )
        
        model = SEOSpecificTransformer(config)
        model = model.to(self.device)
        
        # Apply LoRA
        peft_config = create_peft_config(
            peft_type="LORA",
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["w_q", "w_k", "w_v", "w_o"]
        )
        
        peft_manager = EfficientFineTuningManager(model, peft_config)
        
        # Save PEFT model
        save_dir = "peft_model"
        peft_manager.save_pretrained(save_dir)
        logger.info(f"PEFT model saved to {save_dir}")
        
        # Create new model and load PEFT weights
        new_model = SEOSpecificTransformer(config)
        new_model = new_model.to(self.device)
        
        new_peft_manager = EfficientFineTuningManager(new_model, peft_config)
        new_peft_manager.load_pretrained(save_dir)
        logger.info("PEFT model loaded successfully")
        
        # Test that outputs are the same
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
        attention_mask = torch.ones(batch_size, seq_len).to(self.device)
        
        with torch.no_grad():
            outputs1 = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs2 = new_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check if outputs are similar (should be identical)
        diff = torch.abs(outputs1['last_hidden_state'] - outputs2['last_hidden_state']).max()
        logger.info(f"Maximum difference between saved and loaded model: {diff:.6f}")
        
        return model, new_model, peft_manager
    
    def demonstrate_performance_comparison(self) -> Any:
        """Demonstrate performance comparison between different PEFT methods"""
        logger.info("\n=== Demonstrating Performance Comparison ===")
        
        # Base configuration
        config = TransformerConfig(
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            intermediate_size=1024,
            vocab_size=10000
        )
        
        # Test different PEFT methods
        peft_methods = [
            ("LoRA", "LORA", {"r": 16, "lora_alpha": 32}),
            ("P-tuning", "P_TUNING", {"num_virtual_tokens": 20}),
            ("AdaLoRA", "ADALORA", {"init_r": 12, "target_r": 8}),
            ("Prefix Tuning", "PREFIX_TUNING", {"num_prefix_tokens": 20})
        ]
        
        results = {}
        
        for method_name, peft_type, kwargs in peft_methods:
            logger.info(f"\n--- Testing {method_name} ---")
            
            # Create model
            model = SEOSpecificTransformer(config)
            model = model.to(self.device)
            
            # Original parameter count
            original_params = sum(p.numel() for p in model.parameters())
            
            # Apply PEFT
            peft_config = create_peft_config(peft_type, **kwargs)
            peft_manager = EfficientFineTuningManager(model, peft_config)
            
            # Parameter statistics
            param_stats = peft_manager.get_parameter_count()
            
            # Performance test
            batch_size = 4
            seq_len = 64
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
            attention_mask = torch.ones(batch_size, seq_len).to(self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
                torch.cuda.empty_cache()
            else:
                memory_used = 0
            
            results[method_name] = {
                "original_params": original_params,
                "trainable_params": param_stats['trainable_parameters'],
                "trainable_percentage": param_stats['trainable_percentage'],
                "avg_inference_time": avg_time,
                "memory_used_mb": memory_used
            }
            
            logger.info(f"  Trainable parameters: {param_stats['trainable_parameters']:,}")
            logger.info(f"  Trainable percentage: {param_stats['trainable_percentage']:.2f}%")
            logger.info(f"  Average inference time: {avg_time:.4f}s")
            logger.info(f"  Memory used: {memory_used:.2f} MB")
        
        # Print comparison table
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE COMPARISON SUMMARY")
        logger.info("="*80)
        logger.info(f"{'Method':<15} {'Trainable %':<12} {'Params':<12} {'Time (ms)':<10} {'Memory (MB)':<12}")
        logger.info("-"*80)
        
        for method_name, result in results.items():
            logger.info(f"{method_name:<15} {result['trainable_percentage']:<12.2f} "
                       f"{result['trainable_params']:<12,} {result['avg_inference_time']*1000:<10.2f} "
                       f"{result['memory_used_mb']:<12.2f}")
        
        return results
    
    def demonstrate_orthogonal_regularization(self) -> Any:
        """Demonstrate orthogonal regularization in AdaLoRA"""
        logger.info("\n=== Demonstrating Orthogonal Regularization ===")
        
        # Create model with AdaLoRA
        config = TransformerConfig(
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
            vocab_size=1000
        )
        
        model = SEOSpecificTransformer(config)
        model = model.to(self.device)
        
        # Apply AdaLoRA
        peft_config = create_peft_config(
            peft_type="ADALORA",
            target_modules=["w_q", "w_k", "w_v", "w_o"],
            init_r=8,
            target_r=4,
            orth_reg_weight=0.5
        )
        
        peft_manager = EfficientFineTuningManager(model, peft_config)
        trainer = PEFTTrainer(model, peft_config)
        
        # Test orthogonal regularization loss
        orth_reg_loss = trainer.get_orthogonal_regularization_loss()
        logger.info(f"Initial orthogonal regularization loss: {orth_reg_loss:.6f}")
        
        # Simulate some training steps
        batch_size = 2
        seq_len = 16
        
        for step in range(5):
            # Create dummy batch
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
            attention_mask = torch.ones(batch_size, seq_len).to(self.device)
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
            
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            # Training step
            metrics = trainer.train_step(batch)
            
            # Get orthogonal regularization loss
            orth_reg_loss = trainer.get_orthogonal_regularization_loss()
            
            logger.info(f"Step {step}: Loss = {metrics['loss']:.4f}, Orth Reg = {orth_reg_loss:.6f}")
        
        return model, peft_manager, trainer
    
    def run_comprehensive_demo(self) -> Any:
        """Run comprehensive demonstration of all PEFT methods"""
        logger.info("Starting comprehensive efficient fine-tuning demo")
        
        # Run all demonstrations
        self.demonstrate_lora()
        self.demonstrate_p_tuning()
        self.demonstrate_adalora()
        self.demonstrate_prefix_tuning()
        self.demonstrate_training()
        self.demonstrate_save_load()
        self.demonstrate_performance_comparison()
        self.demonstrate_orthogonal_regularization()
        
        logger.info("Comprehensive PEFT demo completed!")

def main():
    """Main function to run the demonstration"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create demo instance
    demo = EfficientFineTuningDemo()
    
    # Run comprehensive demo
    demo.run_comprehensive_demo()

match __name__:
    case "__main__":
    main() 