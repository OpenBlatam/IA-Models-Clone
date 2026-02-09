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
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from attention_mechanisms import (
from transformer_models import TransformerConfig, SEOSpecificTransformer, TransformerManager
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Example script demonstrating advanced attention mechanisms and positional encodings
Comprehensive examples of different attention types and positional encoding methods
"""


# Import our attention mechanisms and transformer models
    MultiHeadAttention, LocalAttention, SparseAttention, AttentionWithRelativePositions,
    PositionalEncoding, LearnedPositionalEncoding, RelativePositionalEncoding, RotaryPositionalEncoding,
    AttentionFactory, PositionalEncodingFactory, create_attention_mask, create_padding_mask
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionMechanismsDemo:
    """Demonstration class for attention mechanisms and positional encodings"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.transformer_manager = TransformerManager()
        logger.info(f"Using device: {self.device}")
    
    def demonstrate_positional_encodings(self) -> Any:
        """Demonstrate different types of positional encodings"""
        logger.info("=== Demonstrating Positional Encodings ===")
        
        d_model = 512
        seq_len = 100
        batch_size = 2
        
        # Create sample input
        x = torch.randn(batch_size, seq_len, d_model).to(self.device)
        
        # Test different positional encoding types
        encoding_types = ["sinusoidal", "learned", "relative", "rotary"]
        
        for encoding_type in encoding_types:
            logger.info(f"\n--- Testing {encoding_type.upper()} Positional Encoding ---")
            
            try:
                # Create positional encoding
                if encoding_type == "relative":
                    pos_encoding = PositionalEncodingFactory.create_positional_encoding(
                        encoding_type, d_model, max_len=seq_len, max_relative_position=32
                    )
                else:
                    pos_encoding = PositionalEncodingFactory.create_positional_encoding(
                        encoding_type, d_model, max_len=seq_len
                    )
                
                pos_encoding = pos_encoding.to(self.device)
                
                # Apply positional encoding
                start_time = time.time()
                
                if encoding_type == "relative":
                    output = pos_encoding(x, seq_len)
                else:
                    # Transpose for other encoding types
                    x_transposed = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
                    output_transposed = pos_encoding(x_transposed)
                    output = output_transposed.transpose(0, 1)  # [batch_size, seq_len, d_model]
                
                end_time = time.time()
                
                logger.info(f"Output shape: {output.shape}")
                logger.info(f"Processing time: {end_time - start_time:.4f}s")
                logger.info(f"Output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
                
                # Visualize positional encoding patterns
                self._visualize_positional_encoding(pos_encoding, encoding_type, seq_len, d_model)
                
            except Exception as e:
                logger.error(f"Error with {encoding_type} encoding: {e}")
    
    def demonstrate_attention_mechanisms(self) -> Any:
        """Demonstrate different types of attention mechanisms"""
        logger.info("\n=== Demonstrating Attention Mechanisms ===")
        
        d_model = 512
        num_heads = 8
        seq_len = 128
        batch_size = 2
        
        # Create sample input
        x = torch.randn(batch_size, seq_len, d_model).to(self.device)
        
        # Test different attention types
        attention_configs = [
            {"type": "standard", "name": "Standard Multi-Head Attention"},
            {"type": "local", "name": "Local Attention", "window_size": 32},
            {"type": "sparse", "name": "Sparse Attention", "num_landmarks": 16},
            {"type": "relative", "name": "Relative Position Attention", "max_relative_position": 16}
        ]
        
        for config in attention_configs:
            logger.info(f"\n--- Testing {config['name']} ---")
            
            try:
                # Create attention mechanism
                attention_kwargs = {"dropout": 0.1}
                if "window_size" in config:
                    attention_kwargs["window_size"] = config["window_size"]
                if "num_landmarks" in config:
                    attention_kwargs["num_landmarks"] = config["num_landmarks"]
                if "max_relative_position" in config:
                    attention_kwargs["max_relative_position"] = config["max_relative_position"]
                
                attention = AttentionFactory.create_attention(
                    attention_type=config["type"],
                    d_model=d_model,
                    num_heads=num_heads,
                    **attention_kwargs
                )
                attention = attention.to(self.device)
                
                # Create attention mask
                mask = torch.ones(batch_size, seq_len, seq_len).to(self.device)
                
                # Test attention
                start_time = time.time()
                
                if config["type"] in ["local", "sparse", "relative"]:
                    output = attention(x, mask=mask)
                else:
                    output, attention_weights = attention(
                        query=x, key=x, value=x, 
                        mask=mask, need_weights=True
                    )
                
                end_time = time.time()
                
                logger.info(f"Output shape: {output.shape}")
                logger.info(f"Processing time: {end_time - start_time:.4f}s")
                logger.info(f"Output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
                
                # Visualize attention weights if available
                if config["type"] == "standard" and attention_weights is not None:
                    self._visualize_attention_weights(attention_weights[0], config["name"])
                
            except Exception as e:
                logger.error(f"Error with {config['name']}: {e}")
    
    def demonstrate_transformer_with_attention(self) -> Any:
        """Demonstrate transformer models with different attention mechanisms"""
        logger.info("\n=== Demonstrating Transformer Models with Different Attention Types ===")
        
        # Test different transformer configurations
        transformer_configs = [
            {
                "name": "Standard Transformer",
                "attention_type": "standard",
                "positional_encoding_type": "sinusoidal"
            },
            {
                "name": "Local Attention Transformer",
                "attention_type": "local",
                "positional_encoding_type": "learned",
                "attention_window_size": 64
            },
            {
                "name": "Sparse Attention Transformer",
                "attention_type": "sparse",
                "positional_encoding_type": "relative",
                "attention_num_landmarks": 32,
                "positional_encoding_max_relative_position": 16
            },
            {
                "name": "Relative Position Transformer",
                "attention_type": "relative",
                "positional_encoding_type": "rotary",
                "attention_max_relative_position": 16
            }
        ]
        
        for config in transformer_configs:
            logger.info(f"\n--- Testing {config['name']} ---")
            
            try:
                # Create transformer configuration
                transformer_config = TransformerConfig(
                    hidden_size=256,
                    num_layers=4,
                    num_heads=8,
                    intermediate_size=1024,
                    vocab_size=10000,
                    attention_type=config["attention_type"],
                    positional_encoding_type=config["positional_encoding_type"]
                )
                
                # Add specific parameters
                if "attention_window_size" in config:
                    transformer_config.attention_window_size = config["attention_window_size"]
                if "attention_num_landmarks" in config:
                    transformer_config.attention_num_landmarks = config["attention_num_landmarks"]
                if "attention_max_relative_position" in config:
                    transformer_config.attention_max_relative_position = config["attention_max_relative_position"]
                if "positional_encoding_max_relative_position" in config:
                    transformer_config.positional_encoding_max_relative_position = config["positional_encoding_max_relative_position"]
                
                # Create transformer
                transformer = SEOSpecificTransformer(transformer_config)
                transformer = transformer.to(self.device)
                
                # Create sample input
                batch_size = 2
                seq_len = 64
                input_ids = torch.randint(0, transformer_config.vocab_size, (batch_size, seq_len)).to(self.device)
                attention_mask = torch.ones(batch_size, seq_len).to(self.device)
                
                # Test forward pass
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = transformer(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True
                    )
                
                end_time = time.time()
                
                logger.info(f"Output keys: {list(outputs.keys())}")
                logger.info(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
                logger.info(f"Pooler output shape: {outputs['pooler_output'].shape}")
                logger.info(f"Processing time: {end_time - start_time:.4f}s")
                
                if outputs['attentions'] is not None:
                    logger.info(f"Number of attention layers: {len(outputs['attentions'])}")
                    logger.info(f"Attention weights shape: {outputs['attentions'][0].shape}")
                
                # Test memory usage
                self._test_memory_usage(transformer, input_ids, attention_mask)
                
            except Exception as e:
                logger.error(f"Error with {config['name']}: {e}")
    
    def demonstrate_attention_analysis(self) -> Any:
        """Demonstrate attention analysis and visualization"""
        logger.info("\n=== Demonstrating Attention Analysis ===")
        
        # Create a simple transformer for analysis
        config = TransformerConfig(
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            attention_type="standard",
            positional_encoding_type="sinusoidal",
            return_attention_weights=True
        )
        
        transformer = SEOSpecificTransformer(config)
        transformer = transformer.to(self.device)
        
        # Create sample text-like input
        batch_size = 1
        seq_len = 20
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
        attention_mask = torch.ones(batch_size, seq_len).to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Analyze attention patterns
        self._analyze_attention_patterns(outputs['attentions'], seq_len)
    
    def _visualize_positional_encoding(self, pos_encoding, encoding_type: str, seq_len: int, d_model: int):
        """Visualize positional encoding patterns"""
        try:
            # Create sample input for visualization
            x_viz = torch.zeros(1, seq_len, d_model).to(self.device)
            
            if encoding_type == "relative":
                pe_output = pos_encoding(x_viz, seq_len)
            else:
                x_viz_transposed = x_viz.transpose(0, 1)
                pe_output_transposed = pos_encoding(x_viz_transposed)
                pe_output = pe_output_transposed.transpose(0, 1)
            
            # Extract positional encoding (subtract input)
            if encoding_type == "relative":
                pe_values = pe_output[0].cpu().detach().numpy()
            else:
                pe_values = (pe_output - x_viz)[0].cpu().detach().numpy()
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Heatmap of positional encoding
            plt.subplot(2, 2, 1)
            sns.heatmap(pe_values.T, cmap='viridis', cbar=True)
            plt.title(f'{encoding_type.title()} Positional Encoding Heatmap')
            plt.xlabel('Position')
            plt.ylabel('Dimension')
            
            # First few dimensions
            plt.subplot(2, 2, 2)
            for i in range(min(8, d_model)):
                plt.plot(pe_values[:, i], label=f'Dim {i}')
            plt.title('First 8 Dimensions')
            plt.xlabel('Position')
            plt.ylabel('Value')
            plt.legend()
            
            # Statistical analysis
            plt.subplot(2, 2, 3)
            plt.hist(pe_values.flatten(), bins=50, alpha=0.7)
            plt.title('Value Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            
            # Correlation matrix
            plt.subplot(2, 2, 4)
            corr_matrix = np.corrcoef(pe_values.T)
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, cbar=True)
            plt.title('Dimension Correlation Matrix')
            
            plt.tight_layout()
            plt.savefig(f'positional_encoding_{encoding_type}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Positional encoding visualization saved as 'positional_encoding_{encoding_type}.png'")
            
        except Exception as e:
            logger.error(f"Error visualizing positional encoding: {e}")
    
    def _visualize_attention_weights(self, attention_weights: torch.Tensor, title: str):
        """Visualize attention weights"""
        try:
            # attention_weights shape: [num_heads, seq_len, seq_len]
            attention_np = attention_weights.cpu().detach().numpy()
            
            num_heads = attention_np.shape[0]
            seq_len = attention_np.shape[1]
            
            # Create subplot for each head
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            for i in range(min(num_heads, 8)):
                sns.heatmap(attention_np[i], ax=axes[i], cmap='viridis', cbar=True)
                axes[i].set_title(f'Head {i}')
                axes[i].set_xlabel('Key Position')
                axes[i].set_ylabel('Query Position')
            
            plt.suptitle(f'Attention Weights - {title}')
            plt.tight_layout()
            plt.savefig(f'attention_weights_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Attention weights visualization saved as 'attention_weights_{title.lower().replace(' ', '_')}.png'")
            
        except Exception as e:
            logger.error(f"Error visualizing attention weights: {e}")
    
    def _analyze_attention_patterns(self, attention_weights: Tuple[torch.Tensor], seq_len: int):
        """Analyze attention patterns across layers"""
        try:
            num_layers = len(attention_weights)
            
            # Calculate attention statistics
            layer_stats = []
            for layer_idx, layer_attention in enumerate(attention_weights):
                # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
                attention_np = layer_attention[0].cpu().detach().numpy()  # Remove batch dimension
                
                # Calculate statistics for each head
                head_stats = []
                for head_idx in range(attention_np.shape[0]):
                    head_weights = attention_np[head_idx]
                    
                    # Calculate entropy (measure of attention concentration)
                    entropy = -np.sum(head_weights * np.log(head_weights + 1e-8), axis=-1)
                    mean_entropy = np.mean(entropy)
                    
                    # Calculate attention span (how many positions get significant attention)
                    attention_span = np.sum(head_weights > 0.1, axis=-1)
                    mean_span = np.mean(attention_span)
                    
                    # Calculate diagonal attention (self-attention)
                    diagonal_attention = np.mean(np.diag(head_weights))
                    
                    head_stats.append({
                        'entropy': mean_entropy,
                        'attention_span': mean_span,
                        'diagonal_attention': diagonal_attention
                    })
                
                layer_stats.append(head_stats)
            
            # Visualize statistics
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Entropy across layers
            axes[0, 0].plot([np.mean([h['entropy'] for h in layer]) for layer in layer_stats], 'o-')
            axes[0, 0].set_title('Mean Attention Entropy')
            axes[0, 0].set_xlabel('Layer')
            axes[0, 0].set_ylabel('Entropy')
            
            # Attention span across layers
            axes[0, 1].plot([np.mean([h['attention_span'] for h in layer]) for layer in layer_stats], 'o-')
            axes[0, 1].set_title('Mean Attention Span')
            axes[0, 1].set_xlabel('Layer')
            axes[0, 1].set_ylabel('Span')
            
            # Diagonal attention across layers
            axes[1, 0].plot([np.mean([h['diagonal_attention'] for h in layer]) for layer in layer_stats], 'o-')
            axes[1, 0].set_title('Mean Diagonal Attention')
            axes[1, 0].set_xlabel('Layer')
            axes[1, 0].set_ylabel('Diagonal Attention')
            
            # Head diversity (std of entropy across heads)
            axes[1, 1].plot([np.std([h['entropy'] for h in layer]) for layer in layer_stats], 'o-')
            axes[1, 1].set_title('Head Diversity (Entropy Std)')
            axes[1, 1].set_xlabel('Layer')
            axes[1, 1].set_ylabel('Standard Deviation')
            
            plt.tight_layout()
            plt.savefig('attention_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Attention analysis visualization saved as 'attention_analysis.png'")
            
        except Exception as e:
            logger.error(f"Error analyzing attention patterns: {e}")
    
    def _test_memory_usage(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Test memory usage of the model"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get memory usage
                max_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                logger.info(f"Peak GPU memory usage: {max_memory:.2f} MB")
                
                torch.cuda.empty_cache()
            else:
                logger.info("CUDA not available, skipping memory test")
                
        except Exception as e:
            logger.error(f"Error testing memory usage: {e}")
    
    def run_comprehensive_demo(self) -> Any:
        """Run comprehensive demonstration of all features"""
        logger.info("Starting comprehensive attention mechanisms and positional encodings demo")
        
        # Run all demonstrations
        self.demonstrate_positional_encodings()
        self.demonstrate_attention_mechanisms()
        self.demonstrate_transformer_with_attention()
        self.demonstrate_attention_analysis()
        
        logger.info("Comprehensive demo completed!")

def main():
    """Main function to run the demonstration"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create demo instance
    demo = AttentionMechanismsDemo()
    
    # Run comprehensive demo
    demo.run_comprehensive_demo()

match __name__:
    case "__main__":
    main() 