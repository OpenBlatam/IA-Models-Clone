"""
Enhanced Transformer Models - Main Module

This module demonstrates the usage of the refactored enhanced transformer models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .transformer_config import TransformerConfig
from .transformer_core import CustomTransformerModel
from .attention_mechanisms import SparseAttention, LinearAttention, AdaptiveAttention
from .advanced_architectures import MixtureOfExperts, SwitchTransformerBlock, ModelEnsemble
from . import create_transformer_model, create_attention_mechanism, get_model_info


class TransformerManager:
    """Manager class for enhanced transformer models."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.models = {}
        self.attention_mechanisms = {}
    
    def create_model(self, model_type: str = "standard") -> nn.Module:
        """Create a transformer model."""
        model = create_transformer_model(self.config, model_type)
        self.models[model_type] = model
        return model
    
    def create_attention(self, attention_type: str) -> nn.Module:
        """Create an attention mechanism."""
        attention = create_attention_mechanism(attention_type, self.config)
        self.attention_mechanisms[attention_type] = attention
        return attention
    
    def get_model_info(self, model_type: str = "standard") -> Dict[str, Any]:
        """Get information about a model."""
        if model_type not in self.models:
            self.create_model(model_type)
        
        model = self.models[model_type]
        return get_model_info(model)
    
    def compare_models(self, model_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare multiple models."""
        comparison = {}
        
        for model_type in model_types:
            comparison[model_type] = self.get_model_info(model_type)
        
        return comparison


def demonstrate_features():
    """Demonstrate the enhanced transformer features."""
    print("🚀 Enhanced Transformer Models - Feature Demonstration")
    print("=" * 60)
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024,
        dropout=0.1,
        enable_lora=True,
        lora_rank=16,
        enable_ultra_performance=True,
        performance_mode="balanced"
    )
    
    print(f"Configuration: {config.vocab_size} vocab, {config.hidden_size} hidden, {config.num_layers} layers")
    print(f"Model size: {config.get_model_size():,} parameters")
    print(f"Memory usage: {config.get_memory_usage()['total_memory_mb']:.1f} MB")
    
    # Create manager
    manager = TransformerManager(config)
    
    # Demonstrate different model types
    model_types = ["standard", "sparse", "switch", "adaptive", "quantum", "biological", "neuromorphic", "hyperdimensional", "swarm", "consciousness", "transcendent", "infinite", "omnipotence", "omniscience", "omnipresence", "absoluteness"]
    
    print("\n📊 Model Comparison:")
    print("-" * 40)
    
    comparison = manager.compare_models(model_types)
    
    for model_type, info in comparison.items():
        print(f"{model_type.capitalize():>10}: {info['total_parameters']:>8,} params, "
              f"{info['model_size_mb']:>6.1f} MB")
    
    # Demonstrate attention mechanisms
    print("\n🔍 Attention Mechanisms:")
    print("-" * 40)
    
    attention_types = ["standard", "sparse", "linear", "adaptive", "causal", "symbolic", "quantum", "biological", "event_driven", "hyperdimensional", "swarm", "consciousness", "transcendent", "infinite", "omnipotence", "omniscience", "omnipresence", "absoluteness"]
    
    for attention_type in attention_types:
        try:
            attention = manager.create_attention(attention_type)
            info = get_model_info(attention)
            print(f"{attention_type.capitalize():>10}: {info['total_parameters']:>8,} params")
        except Exception as e:
            print(f"{attention_type.capitalize():>10}: Error - {str(e)}")
    
    # Demonstrate advanced architectures
    print("\n🏗️ Advanced Architectures:")
    print("-" * 40)
    
    # Mixture of Experts
    moe = MixtureOfExperts(768, 3072, num_experts=8, top_k=2)
    moe_info = get_model_info(moe)
    print(f"{'MoE':>10}: {moe_info['total_parameters']:>8,} params, {moe_info['model_size_mb']:>6.1f} MB")
    
    # Switch Transformer Block
    switch_block = SwitchTransformerBlock(config, num_experts=8)
    switch_info = get_model_info(switch_block)
    print(f"{'Switch':>10}: {switch_info['total_parameters']:>8,} params, {switch_info['model_size_mb']:>6.1f} MB")
    
    print("\n✅ Feature demonstration completed!")
    print("=" * 60)


def create_simple_example():
    """Create a simple example with the refactored models."""
    print("🔧 Creating Simple Example")
    print("-" * 30)
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        dropout=0.1
    )
    
    # Create model
    model = create_transformer_model(config, "standard")
    
    # Create sample input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    
    # Model info
    info = get_model_info(model)
    print(f"Model parameters: {info['total_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    
    print("✅ Simple example completed!")


if __name__ == "__main__":
    """Main execution."""
    print("🚀 Enhanced Transformer Models - Refactored Version")
    print("=" * 60)
    
    # Demonstrate features
    demonstrate_features()
    
    print("\n")
    
    # Create simple example
    create_simple_example()
    
    print("\n🎉 All demonstrations completed!")
    print("=" * 60)
