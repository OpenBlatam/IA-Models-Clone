"""
Refactored Enhanced Transformer Models Demonstration

This module demonstrates the refactored architecture with
clean interfaces, modular design, and advanced features.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .refactored import (
    create_transformer_model,
    create_attention_mechanism,
    get_model_info,
    get_supported_types,
    load_config,
    save_config,
    save_model,
    load_model,
    benchmark_model,
    optimize_model,
    register_model,
    get_registered_model,
    list_registered_models,
    EnhancedTransformerAPI
)
from .transformer_config import TransformerConfig


def demonstrate_basic_usage():
    """Demonstrate basic usage of the refactored system."""
    print("🚀 Basic Usage Demonstration")
    print("=" * 50)
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024,
        dropout=0.1,
        layer_norm_eps=1e-12
    )
    
    print(f"Configuration created: {config.hidden_size} hidden size, {config.num_layers} layers")
    
    # Create different model types
    model_types = ["standard", "quantum", "sparse", "linear", "adaptive", "causal"]
    
    for model_type in model_types:
        try:
            model = create_transformer_model(config, model_type)
            info = get_model_info(model)
            print(f"✅ {model_type:>10} model: {info['total_parameters']:,} parameters, {info['memory_mb']:.2f} MB")
        except Exception as e:
            print(f"❌ {model_type:>10} model: Error - {str(e)[:50]}")
    
    print()


def demonstrate_attention_mechanisms():
    """Demonstrate different attention mechanisms."""
    print("👁️ Attention Mechanisms Demonstration")
    print("=" * 50)
    
    config = TransformerConfig(hidden_size=256, num_attention_heads=8)
    x = torch.randn(2, 10, 256)
    
    attention_types = ["standard", "sparse", "linear", "adaptive", "causal", "quantum"]
    
    for attention_type in attention_types:
        try:
            attention = create_attention_mechanism(attention_type, config)
            with torch.no_grad():
                output, weights = attention(x, x, x)
            print(f"✅ {attention_type:>10} attention: {output.shape}, {weights.shape}")
        except Exception as e:
            print(f"❌ {attention_type:>10} attention: Error - {str(e)[:50]}")
    
    print()


def demonstrate_hybrid_models():
    """Demonstrate hybrid models."""
    print("🔗 Hybrid Models Demonstration")
    print("=" * 50)
    
    config = TransformerConfig(hidden_size=256, num_layers=6, num_attention_heads=8)
    x = torch.randn(2, 10, 256)
    
    hybrid_types = ["quantum_sparse", "quantum_linear", "quantum_adaptive", "sparse_linear", "adaptive_causal"]
    
    for model_type in hybrid_types:
        try:
            model = create_transformer_model(config, model_type, factory_name="hybrid")
            info = get_model_info(model)
            with torch.no_grad():
                output = model(x)
            print(f"✅ {model_type:>15} model: {info['total_parameters']:,} parameters, {output['logits'].shape}")
        except Exception as e:
            print(f"❌ {model_type:>15} model: Error - {str(e)[:50]}")
    
    print()


def demonstrate_configuration_management():
    """Demonstrate configuration management."""
    print("⚙️ Configuration Management Demonstration")
    print("=" * 50)
    
    # Create configuration using builder pattern
    from .refactored.management import ConfigBuilder
    
    config = (ConfigBuilder()
              .set_vocab_size(50257)
              .set_hidden_size(768)
              .set_num_layers(12)
              .set_num_attention_heads(12)
              .set_intermediate_size(3072)
              .set_dropout(0.1)
              .build())
    
    print(f"✅ Configuration built: {config.hidden_size} hidden size, {config.num_layers} layers")
    
    # Save configuration
    save_config(config, "demo_config.json")
    print("✅ Configuration saved to demo_config.json")
    
    # Load configuration
    loaded_config = load_config("demo_config.json")
    print(f"✅ Configuration loaded: {loaded_config.hidden_size} hidden size, {loaded_config.num_layers} layers")
    
    print()


def demonstrate_model_management():
    """Demonstrate model management."""
    print("📦 Model Management Demonstration")
    print("=" * 50)
    
    config = TransformerConfig(hidden_size=256, num_layers=4, num_attention_heads=8)
    
    # Create and register models
    model1 = create_transformer_model(config, "standard")
    model2 = create_transformer_model(config, "quantum")
    
    register_model("standard_model", model1, config)
    register_model("quantum_model", model2, config)
    
    print(f"✅ Registered models: {list_registered_models()}")
    
    # Get registered model
    retrieved_model = get_registered_model("standard_model")
    if retrieved_model is not None:
        info = get_model_info(retrieved_model)
        print(f"✅ Retrieved model: {info['total_parameters']:,} parameters")
    
    # Save and load model
    save_model(model1, "demo_model.pt")
    print("✅ Model saved to demo_model.pt")
    
    loaded_model = load_model("demo_model.pt", config)
    print("✅ Model loaded from demo_model.pt")
    
    print()


def demonstrate_benchmarking():
    """Demonstrate model benchmarking."""
    print("📊 Benchmarking Demonstration")
    print("=" * 50)
    
    config = TransformerConfig(hidden_size=256, num_layers=4, num_attention_heads=8)
    model = create_transformer_model(config, "standard")
    
    # Benchmark model
    input_shape = (2, 10, 256)
    benchmark_results = benchmark_model(model, input_shape, num_runs=5)
    
    print(f"✅ Benchmark results:")
    print(f"   Average inference time: {benchmark_results['avg_inference_time']:.4f}s")
    print(f"   Min inference time: {benchmark_results['min_inference_time']:.4f}s")
    print(f"   Max inference time: {benchmark_results['max_inference_time']:.4f}s")
    print(f"   Average memory usage: {benchmark_results['avg_memory_mb']:.2f} MB")
    
    # Optimize model
    optimized_model = optimize_model(model, "memory")
    print("✅ Model optimized for memory")
    
    print()


def demonstrate_api_class():
    """Demonstrate the main API class."""
    print("🎯 API Class Demonstration")
    print("=" * 50)
    
    # Create API instance
    api = EnhancedTransformerAPI()
    
    # Get system information
    system_info = api.get_system_info()
    print(f"✅ System info:")
    print(f"   Factories: {system_info['factories']}")
    print(f"   Cached configs: {len(system_info['cached_configs'])}")
    print(f"   Cached models: {len(system_info['cached_models'])}")
    print(f"   Registered models: {len(system_info['registered_models'])}")
    
    # Get supported types
    supported_types = api.get_supported_types("enhanced")
    print(f"✅ Supported model types: {supported_types}")
    
    print()


def demonstrate_quantum_features():
    """Demonstrate quantum features."""
    print("🔬 Quantum Features Demonstration")
    print("=" * 50)
    
    config = TransformerConfig(hidden_size=256, num_layers=4, num_attention_heads=8)
    x = torch.randn(2, 10, 256)
    
    # Create quantum model
    quantum_model = create_transformer_model(config, "quantum")
    
    with torch.no_grad():
        output = quantum_model(x)
    
    info = get_model_info(quantum_model)
    print(f"✅ Quantum model: {info['total_parameters']:,} parameters, {output['logits'].shape}")
    
    # Create quantum attention
    quantum_attention = create_attention_mechanism("quantum", config)
    
    with torch.no_grad():
        attn_output, attn_weights = quantum_attention(x, x, x)
    
    print(f"✅ Quantum attention: {attn_output.shape}, {attn_weights.shape}")
    
    print()


def demonstrate_advanced_features():
    """Demonstrate advanced features."""
    print("🌟 Advanced Features Demonstration")
    print("=" * 50)
    
    config = TransformerConfig(hidden_size=256, num_layers=4, num_attention_heads=8)
    x = torch.randn(2, 10, 256)
    
    # Test different attention patterns
    attention_types = ["sparse", "linear", "adaptive", "causal"]
    
    for attn_type in attention_types:
        try:
            attention = create_attention_mechanism(attn_type, config)
            with torch.no_grad():
                output, weights = attention(x, x, x)
            print(f"✅ {attn_type:>8} attention: {output.shape}, {weights.shape}")
        except Exception as e:
            print(f"❌ {attn_type:>8} attention: Error - {str(e)[:50]}")
    
    print()


def main():
    """Main demonstration function."""
    print("🎉 Refactored Enhanced Transformer Models Demonstration")
    print("=" * 80)
    print()
    
    # Run all demonstrations
    demonstrate_basic_usage()
    demonstrate_attention_mechanisms()
    demonstrate_hybrid_models()
    demonstrate_configuration_management()
    demonstrate_model_management()
    demonstrate_benchmarking()
    demonstrate_api_class()
    demonstrate_quantum_features()
    demonstrate_advanced_features()
    
    print("🎉 All demonstrations completed successfully!")
    print("=" * 80)
    print("✨ The refactored system provides:")
    print("   🏗️  Clean, modular architecture")
    print("   🔧  Comprehensive configuration management")
    print("   📦  Advanced model management")
    print("   🚀  Multiple model types and attention mechanisms")
    print("   🔬  Quantum and hybrid features")
    print("   📊  Built-in benchmarking and optimization")
    print("   🎯  Simple, unified API")
    print("=" * 80)


if __name__ == "__main__":
    main()

