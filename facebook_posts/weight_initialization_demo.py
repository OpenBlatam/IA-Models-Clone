"""
🚀 Weight Initialization System Demonstration
============================================

This script demonstrates the weight initialization system with official PyTorch,
Transformers, Diffusers, and Gradio best practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our weight initialization system
try:
    from weight_initialization_system import (
        WeightInitializer, 
        WeightInitConfig, 
        get_initialization_recommendations
    )
    WEIGHT_INIT_AVAILABLE = True
except ImportError:
    print("⚠️  Weight initialization system not available")
    WEIGHT_INIT_AVAILABLE = False

# Import experiment tracking if available
try:
    from experiment_tracking import ExperimentTracker
    EXPERIMENT_TRACKING_AVAILABLE = True
except ImportError:
    print("⚠️  Experiment tracking not available")
    EXPERIMENT_TRACKING_AVAILABLE = False


class DemoCNN(nn.Module):
    """Demo CNN model for weight initialization demonstration."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Convolutional layers with batch norm and ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Pooling and flatten
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class DemoTransformer(nn.Module):
    """Demo Transformer model for weight initialization demonstration."""
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Embedding and positional encoding
        x = self.embedding(x) + self.pos_encoding[:x.size(1)]
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Output projection
        x = self.layer_norm(x)
        x = self.fc_out(x)
        
        return x


class DemoDiffusion(nn.Module):
    """Demo Diffusion model for weight initialization demonstration."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        
        # U-Net style architecture
        self.inc = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        self.mid = nn.Conv2d(256, 256, 3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.outc = nn.Conv2d(64, out_channels, 3, padding=1)
        
        # Normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(64)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.bn1(self.inc(x)))
        x2 = self.relu(self.bn2(self.down1(x1)))
        x3 = self.relu(self.bn3(self.down2(x2)))
        
        # Middle
        x = self.relu(self.bn4(self.mid(x3)))
        
        # Decoder
        x = self.relu(self.bn5(self.up2(x)))
        x = self.relu(self.bn6(self.up1(x)))
        x = self.outc(x)
        
        return x


def demonstrate_weight_initialization():
    """Demonstrate different weight initialization strategies."""
    
    print("🔧 Weight Initialization Demonstration")
    print("=" * 50)
    
    if not WEIGHT_INIT_AVAILABLE:
        print("❌ Weight initialization system not available")
        return
    
    # Create models
    models = {
        'CNN': DemoCNN(),
        'Transformer': DemoTransformer(),
        'Diffusion': DemoDiffusion()
    }
    
    # Get recommendations for different architectures
    recommendations = {
        'cnn': get_initialization_recommendations('cnn'),
        'transformer': get_initialization_recommendations('transformer'),
        'rnn': get_initialization_recommendations('rnn')
    }
    
    print("\n📋 Initialization Recommendations:")
    for arch, rec in recommendations.items():
        print(f"\n{arch.upper()}:")
        for key, value in rec.items():
            print(f"  {key}: {value}")
    
    # Test different initialization methods
    methods = ['xavier_uniform', 'kaiming_uniform', 'orthogonal', 'sparse']
    
    for model_name, model in models.items():
        print(f"\n🎯 Testing {model_name} Model:")
        print("-" * 30)
        
        for method in methods:
            # Create configuration
            config = WeightInitConfig(
                method=method,
                conv_init=method,
                linear_init=method,
                attention_init=method
            )
            
            # Create initializer
            initializer = WeightInitializer(config)
            
            # Clone model to avoid modifying original
            test_model = type(model)()
            
            # Initialize weights
            summary = initializer.initialize_model(test_model, track_stats=True)
            
            print(f"  {method}: {summary['total_layers']} layers, {summary['total_parameters']} parameters")
            
            # Get statistics
            stats = initializer.get_initialization_summary()
            if 'weight_statistics' in stats:
                weight_stats = stats['weight_statistics']
                print(f"    Weight mean: {weight_stats['mean']:.4f}, std: {weight_stats['std']:.4f}")
            
            # Save statistics
            filename = f"{model_name.lower()}_{method}_stats.json"
            initializer.save_initialization_stats(filename)
            print(f"    Statistics saved to: {filename}")


def demonstrate_pytorch_best_practices():
    """Demonstrate PyTorch official best practices for weight initialization."""
    
    print("\n🔥 PyTorch Official Best Practices")
    print("=" * 50)
    
    # Create a model
    model = DemoCNN()
    
    print("📊 Model before initialization:")
    print_model_weight_stats(model)
    
    # Apply PyTorch best practices
    print("\n🔧 Applying PyTorch best practices...")
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Kaiming initialization for Conv layers with ReLU
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm: weights to 1, bias to 0
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            # Xavier initialization for linear layers
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            # Xavier initialization for embeddings
            nn.init.xavier_uniform_(module.weight)
    
    print("📊 Model after PyTorch best practices initialization:")
    print_model_weight_stats(model)


def demonstrate_modern_pytorch_features():
    """Demonstrate modern PyTorch 2.0+ features."""
    
    print("\n🚀 Modern PyTorch 2.0+ Features")
    print("=" * 50)
    
    # Create a simple model
    model = DemoCNN()
    
    # 1. TorchScript compilation
    try:
        print("🔧 Testing TorchScript compilation...")
        scripted_model = torch.jit.script(model)
        print("✅ TorchScript compilation successful")
    except Exception as e:
        print(f"❌ TorchScript compilation failed: {e}")
    
    # 2. torch.compile (PyTorch 2.0+)
    try:
        print("🔧 Testing torch.compile...")
        compiled_model = torch.compile(model, mode="default")
        print("✅ torch.compile successful")
    except Exception as e:
        print(f"❌ torch.compile failed: {e}")
    
    # 3. Automatic mixed precision
    try:
        print("🔧 Testing automatic mixed precision...")
        from torch.cuda.amp import autocast, GradScaler
        
        scaler = GradScaler()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 32, 32)
        
        with autocast():
            output = model(dummy_input)
        
        print("✅ Automatic mixed precision successful")
        print(f"   Output dtype: {output.dtype}")
        
    except Exception as e:
        print(f"❌ Automatic mixed precision failed: {e}")
    
    # 4. Memory-efficient attention
    try:
        print("🔧 Testing memory-efficient attention...")
        from torch.nn.functional import scaled_dot_product_attention
        
        # Create dummy attention inputs
        query = torch.randn(1, 8, 64, 64)
        key = torch.randn(1, 8, 64, 64)
        value = torch.randn(1, 8, 64, 64)
        
        attention_output = scaled_dot_product_attention(query, key, value)
        print("✅ Memory-efficient attention successful")
        print(f"   Output shape: {attention_output.shape}")
        
    except Exception as e:
        print(f"❌ Memory-efficient attention failed: {e}")


def demonstrate_transformers_integration():
    """Demonstrate integration with Transformers library."""
    
    print("\n🤗 Transformers Library Integration")
    print("=" * 50)
    
    try:
        from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
        
        print("✅ Transformers library available")
        
        # Test basic functionality
        print("🔧 Testing basic Transformers functionality...")
        
        # Load a small model for testing
        model_name = "distilbert-base-uncased"
        
        print(f"📥 Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test tokenization
        text = "Hello, this is a test of the Transformers library!"
        inputs = tokenizer(text, return_tensors="pt")
        
        print(f"✅ Tokenization successful")
        print(f"   Input shape: {inputs['input_ids'].shape}")
        print(f"   Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        
    except ImportError:
        print("❌ Transformers library not available")
    except Exception as e:
        print(f"❌ Transformers integration failed: {e}")


def demonstrate_diffusers_integration():
    """Demonstrate integration with Diffusers library."""
    
    print("\n🎨 Diffusers Library Integration")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        
        print("✅ Diffusers library available")
        
        # Test basic functionality
        print("🔧 Testing basic Diffusers functionality...")
        
        # Note: We won't load the full model for this demo to save time
        print("📝 Diffusers library features available:")
        print("   - Stable Diffusion pipelines")
        print("   - Custom UNet training")
        print("   - Memory optimization techniques")
        print("   - Modern schedulers")
        
        # Test if we can import key components
        from diffusers import UNet2DConditionModel, DPMSolverMultistepScheduler
        
        print("✅ Key Diffusers components imported successfully")
        
    except ImportError:
        print("❌ Diffusers library not available")
    except Exception as e:
        print(f"❌ Diffusers integration failed: {e}")


def demonstrate_gradio_integration():
    """Demonstrate integration with Gradio library."""
    
    print("\n🎯 Gradio Library Integration")
    print("=" * 50)
    
    try:
        import gradio as gr
        
        print("✅ Gradio library available")
        print(f"   Version: {gr.__version__}")
        
        # Test basic functionality
        print("🔧 Testing basic Gradio functionality...")
        
        # Create a simple interface
        def greet(name):
            return f"Hello {name}!"
        
        # Test interface creation (without launching)
        interface = gr.Interface(
            fn=greet,
            inputs="text",
            outputs="text",
            title="Test Interface"
        )
        
        print("✅ Gradio interface created successfully")
        print("   Interface type:", type(interface).__name__)
        
        # Test modern Gradio features
        print("🔧 Testing modern Gradio features...")
        
        # Test Blocks interface
        with gr.Blocks() as blocks:
            gr.Markdown("# Test Blocks Interface")
            input_text = gr.Textbox(label="Input")
            output_text = gr.Textbox(label="Output")
            button = gr.Button("Process")
            button.click(fn=greet, inputs=input_text, outputs=output_text)
        
        print("✅ Gradio Blocks interface created successfully")
        
        # Test themes
        theme = gr.themes.Soft()
        print("✅ Gradio themes available")
        
    except ImportError:
        print("❌ Gradio library not available")
    except Exception as e:
        print(f"❌ Gradio integration failed: {e}")


def print_model_weight_stats(model: nn.Module):
    """Print weight statistics for a model."""
    
    total_params = 0
    weight_stats = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            
            if param.dim() > 1:  # Only for weight matrices
                weight_stats.append({
                    'name': name,
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'min': param.min().item(),
                    'max': param.max().item(),
                    'norm': param.norm().item()
                })
    
    print(f"  Total parameters: {total_params:,}")
    
    if weight_stats:
        print("  Weight statistics:")
        for stat in weight_stats[:5]:  # Show first 5 layers
            print(f"    {stat['name']}: mean={stat['mean']:.4f}, std={stat['std']:.4f}")
        if len(weight_stats) > 5:
            print(f"    ... and {len(weight_stats) - 5} more layers")


def create_visualization_report():
    """Create a visualization report of weight initialization effects."""
    
    print("\n📊 Creating Visualization Report")
    print("=" * 50)
    
    if not WEIGHT_INIT_AVAILABLE:
        print("❌ Weight initialization system not available")
        return
    
    # Create a simple model
    model = DemoCNN()
    
    # Test different initialization methods
    methods = ['xavier_uniform', 'kaiming_uniform', 'orthogonal', 'sparse']
    method_stats = {}
    
    for method in methods:
        # Create configuration
        config = WeightInitConfig(method=method)
        initializer = WeightInitializer(config)
        
        # Clone model
        test_model = type(model)()
        
        # Initialize weights
        initializer.initialize_model(test_model, track_stats=True)
        
        # Get statistics
        stats = initializer.get_initialization_summary()
        method_stats[method] = stats
    
    # Create visualizations
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Weight Initialization Comparison', fontsize=16)
        
        # Plot weight means
        means = [method_stats[method]['weight_statistics']['mean'] for method in methods]
        axes[0, 0].bar(methods, means)
        axes[0, 0].set_title('Weight Means')
        axes[0, 0].set_ylabel('Mean Value')
        
        # Plot weight standard deviations
        stds = [method_stats[method]['weight_statistics']['std'] for method in methods]
        axes[0, 1].bar(methods, stds)
        axes[0, 1].set_title('Weight Standard Deviations')
        axes[0, 1].set_ylabel('Standard Deviation')
        
        # Plot weight norms
        norms = [method_stats[method]['weight_statistics']['norm'] for method in methods]
        axes[1, 0].bar(methods, norms)
        axes[1, 0].set_title('Weight Norms')
        axes[1, 0].set_ylabel('Norm Value')
        
        # Plot parameter counts
        param_counts = [method_stats[method]['total_layers'] for method in methods]
        axes[1, 1].bar(methods, param_counts)
        axes[1, 1].set_title('Total Layers')
        axes[1, 1].set_ylabel('Number of Layers')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = "weight_initialization_comparison.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {plot_filename}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")


def main():
    """Main demonstration function."""
    
    print("🚀 Weight Initialization System - Complete Demonstration")
    print("=" * 70)
    
    # Check system requirements
    print(f"🐍 Python version: {torch.version.python}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 CUDA version: {torch.version.cuda}")
        print(f"🎮 GPU device: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "=" * 70)
    
    # Run demonstrations
    try:
        # 1. Weight initialization system
        demonstrate_weight_initialization()
        
        # 2. PyTorch best practices
        demonstrate_pytorch_best_practices()
        
        # 3. Modern PyTorch features
        demonstrate_modern_pytorch_features()
        
        # 4. Transformers integration
        demonstrate_transformers_integration()
        
        # 5. Diffusers integration
        demonstrate_diffusers_integration()
        
        # 6. Gradio integration
        demonstrate_gradio_integration()
        
        # 7. Visualization report
        create_visualization_report()
        
        print("\n" + "=" * 70)
        print("✅ All demonstrations completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Review the generated statistics files")
        print("   2. Check the visualization report")
        print("   3. Integrate with your experiment tracking system")
        print("   4. Apply best practices to your models")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()






