"""
Complete Transformer and LLM System Example

This file demonstrates the complete Transformer and LLM system with all features:
1. Basic Transformer model creation and usage
2. Attention mechanisms and positional encoding
3. LLM text generation capabilities
4. Prompt engineering techniques
5. Model analysis and comparison
6. Performance optimization
7. Integration with other framework components
8. Real-world applications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import matplotlib.pyplot as plt

# Import transformer LLM system
try:
    from transformer_llm_system import (
        TransformerModel, TransformerEncoder, TransformerDecoder,
        MultiHeadAttention, PositionalEncoding, LLMGenerator,
        PromptEngineer, LLMAnalyzer, create_transformer_model
    )
    TRANSFORMER_LLM_AVAILABLE = True
except ImportError:
    print("Warning: transformer_llm_system not found. Some examples may not work.")
    TRANSFORMER_LLM_AVAILABLE = False

# Import other framework components
try:
    from pytorch_primary_framework_system import PyTorchPrimaryFrameworkSystem
    from custom_model_architectures import BaseModel
    from loss_optimization_system import LossFunctions, Optimizers, LearningRateSchedulers
    from weight_initialization_system import WeightInitializer
    FRAMEWORK_AVAILABLE = True
except ImportError:
    print("Warning: Framework components not found. Some integrations may not work.")
    FRAMEWORK_AVAILABLE = False


class CompleteTransformerLLMExample:
    """Complete example demonstrating all Transformer and LLM capabilities."""
    
    def __init__(self, config_path: str = "transformer_llm_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.models = {}
        self.generators = {}
        self.analyzers = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using default config.")
            return {}
    
    def setup_models(self):
        """Set up different Transformer models."""
        print("\n=== Setting Up Transformer Models ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping model setup.")
            return
        
        # Create different model configurations
        model_configs = {
            'small': {
                'model_type': 'encoder_decoder',
                'src_vocab_size': 1000,
                'tgt_vocab_size': 1000,
                'd_model': 256,
                'n_layers': 4,
                'n_heads': 8,
                'd_ff': 1024
            },
            'medium': {
                'model_type': 'encoder_decoder',
                'src_vocab_size': 2000,
                'tgt_vocab_size': 2000,
                'd_model': 512,
                'n_layers': 6,
                'n_heads': 8,
                'd_ff': 2048
            },
            'large': {
                'model_type': 'encoder_decoder',
                'src_vocab_size': 5000,
                'tgt_vocab_size': 5000,
                'd_model': 768,
                'n_layers': 12,
                'n_heads': 12,
                'd_ff': 3072
            }
        }
        
        # Create models
        for name, config in model_configs.items():
            self.models[name] = create_transformer_model(config)
            self.models[name].to(self.device)
            
            # Count parameters
            num_params = sum(p.numel() for p in self.models[name].parameters())
            print(f"Created {name} model with {num_params:,} parameters")
    
    def demonstrate_basic_transformer(self):
        """Demonstrate basic Transformer functionality."""
        print("\n=== Basic Transformer Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping basic transformer demonstration.")
            return
        
        # Use small model for demonstration
        model = self.models['small']
        model.eval()
        
        # Create sample data
        batch_size, seq_len = 2, 10
        src = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        tgt = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = model(src, tgt)
        
        print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        
        # Test with different sequence lengths
        for seq_len in [5, 10, 15]:
            src = torch.randint(0, 1000, (1, seq_len)).to(self.device)
            tgt = torch.randint(0, 1000, (1, seq_len)).to(self.device)
            
            with torch.no_grad():
                output = model(src, tgt)
            
            print(f"Sequence length {seq_len}: output shape {output.shape}")
    
    def demonstrate_attention_mechanisms(self):
        """Demonstrate attention mechanisms."""
        print("\n=== Attention Mechanisms Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping attention demonstration.")
            return
        
        # Create attention mechanism
        d_model, n_heads = 256, 8
        attention = MultiHeadAttention(d_model, n_heads)
        attention.to(self.device)
        
        # Create sample data
        batch_size, seq_len = 2, 10
        query = torch.randn(batch_size, seq_len, d_model).to(self.device)
        key = torch.randn(batch_size, seq_len, d_model).to(self.device)
        value = torch.randn(batch_size, seq_len, d_model).to(self.device)
        
        # Create causal mask for decoder
        mask = torch.ones(batch_size, seq_len, seq_len).to(self.device)
        mask = torch.triu(mask, diagonal=1)  # Upper triangular mask
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        # Forward pass with and without mask
        attention.eval()
        with torch.no_grad():
            output_no_mask, weights_no_mask = attention(query, key, value)
            output_with_mask, weights_with_mask = attention(query, key, value, mask)
        
        print(f"Attention output shapes: no_mask={output_no_mask.shape}, with_mask={output_with_mask.shape}")
        print(f"Attention weights shapes: no_mask={weights_no_mask.shape}, with_mask={weights_with_mask.shape}")
        
        # Analyze attention patterns
        avg_weights_no_mask = weights_no_mask.mean(dim=1)  # Average across heads
        avg_weights_with_mask = weights_with_mask.mean(dim=1)
        
        print(f"Average attention weights: no_mask={avg_weights_no_mask.shape}, with_mask={avg_weights_with_mask.shape}")
        
        # Check causal masking effect
        mask_effect = (avg_weights_with_mask == 0).float().mean()
        print(f"Masked attention positions: {mask_effect:.2%}")
    
    def demonstrate_positional_encoding(self):
        """Demonstrate positional encoding techniques."""
        print("\n=== Positional Encoding Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping positional encoding demonstration.")
            return
        
        # Create positional encoding
        d_model, max_len = 256, 100
        pos_encoding = PositionalEncoding(d_model, max_len)
        pos_encoding.to(self.device)
        
        # Test different sequence lengths
        for seq_len in [10, 20, 50]:
            x = torch.randn(seq_len, d_model).to(self.device)
            encoded = pos_encoding(x)
            
            # Analyze positional patterns
            pos_diff = encoded - x
            magnitude = pos_diff.abs().mean()
            variance = pos_diff.var()
            
            print(f"Sequence length {seq_len}: magnitude={magnitude:.4f}, variance={variance:.4f}")
        
        # Visualize positional encoding patterns
        seq_len = 20
        x = torch.randn(seq_len, d_model).to(self.device)
        encoded = pos_encoding(x)
        pos_diff = encoded - x
        
        # Plot first few dimensions
        plt.figure(figsize=(12, 8))
        for i in range(min(8, d_model)):
            plt.subplot(2, 4, i+1)
            plt.plot(pos_diff[:, i].cpu().numpy())
            plt.title(f'Dimension {i}')
            plt.xlabel('Position')
            plt.ylabel('Encoding Value')
        
        plt.tight_layout()
        plt.suptitle('Positional Encoding Patterns', y=1.02)
        plt.show()
    
    def demonstrate_text_generation(self):
        """Demonstrate LLM text generation capabilities."""
        print("\n=== Text Generation Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping text generation demonstration.")
            return
        
        # Create a simple mock tokenizer for demonstration
        class MockTokenizer:
            def __init__(self, vocab_size=1000):
                self.vocab_size = vocab_size
                self.eos_token_id = vocab_size - 1
                self.pad_token_id = vocab_size - 2
            
            def encode(self, text, return_tensors='pt'):
                # Mock encoding - convert text to random token IDs
                words = text.split()
                tokens = torch.randint(0, self.vocab_size-2, (1, len(words)))
                if return_tensors == 'pt':
                    return tokens
                return tokens.tolist()
            
            def decode(self, tokens, skip_special_tokens=True):
                # Mock decoding - convert token IDs back to text
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                return f"Generated text with {len(tokens)} tokens: {' '.join([f'token_{t}' for t in tokens])}"
        
        # Create tokenizer and generator
        tokenizer = MockTokenizer()
        model = self.models['small']
        
        # Create generator
        generator = LLMGenerator(model, tokenizer, str(self.device))
        
        # Test different generation parameters
        prompts = [
            "The future of artificial intelligence",
            "Once upon a time",
            "The quick brown fox"
        ]
        
        generation_params = [
            {'temperature': 0.5, 'top_k': 10, 'top_p': 0.8},
            {'temperature': 1.0, 'top_k': 50, 'top_p': 0.9},
            {'temperature': 1.5, 'top_k': 100, 'top_p': 0.95}
        ]
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            for i, params in enumerate(generation_params):
                try:
                    generated = generator.generate(
                        prompt,
                        max_length=20,
                        **params
                    )
                    print(f"  Params {i+1}: {generated}")
                except Exception as e:
                    print(f"  Params {i+1}: Generation failed - {e}")
    
    def demonstrate_prompt_engineering(self):
        """Demonstrate prompt engineering techniques."""
        print("\n=== Prompt Engineering Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping prompt engineering demonstration.")
            return
        
        # Create prompt engineer
        engineer = PromptEngineer()
        
        # Test different prompt templates
        instruction = "Explain the concept of machine learning"
        
        # Zero-shot prompt
        zero_shot = engineer.create_prompt(instruction, 'zero_shot')
        print(f"Zero-shot prompt:\n{zero_shot}\n")
        
        # Few-shot prompt
        examples = [
            ("What is AI?", "Artificial Intelligence is a field of computer science..."),
            ("What is deep learning?", "Deep learning is a subset of machine learning..."),
            ("What is neural networks?", "Neural networks are computational models...")
        ]
        few_shot = engineer.create_few_shot_prompt(examples, instruction)
        print(f"Few-shot prompt:\n{few_shot}\n")
        
        # Chain-of-thought prompt
        cot_instruction = "Solve this math problem: If a train travels 120 km in 2 hours, what is its speed?"
        cot_prompt = engineer.create_prompt(cot_instruction, 'chain_of_thought')
        print(f"Chain-of-thought prompt:\n{cot_prompt}\n")
        
        # Role-playing prompt
        role_instruction = "Explain quantum computing in simple terms"
        role_prompt = engineer.create_prompt(role_instruction, 'role_playing', role="a high school physics teacher")
        print(f"Role-playing prompt:\n{role_prompt}\n")
        
        # Formatting prompt
        format_instruction = "List the benefits of renewable energy"
        format_prompt = engineer.create_prompt(format_instruction, 'formatting', format_type="bullet points")
        print(f"Formatting prompt:\n{format_prompt}\n")
    
    def demonstrate_model_analysis(self):
        """Demonstrate model analysis capabilities."""
        print("\n=== Model Analysis Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping model analysis demonstration.")
            return
        
        # Create analyzer
        model = self.models['small']
        
        # Mock tokenizer for analyzer
        class MockTokenizer:
            def encode(self, text):
                return torch.randint(0, 1000, (1, len(text.split())))
        
        tokenizer = MockTokenizer()
        analyzer = LLMAnalyzer(model, tokenizer)
        
        # Analyze model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Analyze model layers
        layer_info = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding, MultiHeadAttention)):
                params = sum(p.numel() for p in module.parameters())
                layer_info.append((name, params))
        
        print("\nLayer-wise parameter distribution:")
        for name, params in sorted(layer_info, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {name}: {params:,} parameters")
    
    def demonstrate_performance_optimization(self):
        """Demonstrate performance optimization techniques."""
        print("\n=== Performance Optimization Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping performance optimization demonstration.")
            return
        
        model = self.models['medium']
        model.eval()
        
        # Create sample data
        batch_size, seq_len = 4, 20
        src = torch.randint(0, 2000, (batch_size, seq_len)).to(self.device)
        tgt = torch.randint(0, 2000, (batch_size, seq_len)).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(src, tgt)
        
        # Baseline timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(src, tgt)
        baseline_time = (time.time() - start_time) / 10
        
        print(f"Baseline inference time: {baseline_time*1000:.2f} ms")
        
        # Mixed precision optimization
        if self.device.type == 'cuda':
            try:
                with torch.cuda.amp.autocast():
                    start_time = time.time()
                    with torch.no_grad():
                        for _ in range(10):
                            _ = model(src, tgt)
                    amp_time = (time.time() - start_time) / 10
                
                print(f"Mixed precision time: {amp_time*1000:.2f} ms")
                print(f"Speedup: {baseline_time/amp_time:.2f}x")
            except Exception as e:
                print(f"Mixed precision failed: {e}")
        
        # Torch compile optimization (if available)
        try:
            compiled_model = torch.compile(model)
            compiled_model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = compiled_model(src, tgt)
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = compiled_model(src, tgt)
            compiled_time = (time.time() - start_time) / 10
            
            print(f"Compiled model time: {compiled_time*1000:.2f} ms")
            print(f"Speedup: {baseline_time/compiled_time:.2f}x")
        except Exception as e:
            print(f"Torch compile failed: {e}")
    
    def demonstrate_framework_integration(self):
        """Demonstrate integration with other framework components."""
        print("\n=== Framework Integration Demonstration ===")
        
        if not FRAMEWORK_AVAILABLE:
            print("Skipping framework integration demonstration.")
            return
        
        # Create a custom Transformer model that inherits from BaseModel
        class CustomTransformerModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
                self.transformer = TransformerModel(
                    src_vocab_size=config['src_vocab_size'],
                    tgt_vocab_size=config['tgt_vocab_size'],
                    d_model=config['d_model'],
                    n_layers=config['n_layers'],
                    n_heads=config['n_heads'],
                    d_ff=config['d_ff']
                )
            
            def forward(self, src, tgt):
                return self.transformer(src, tgt)
        
        # Create model and framework
        config = {
            'src_vocab_size': 1000,
            'tgt_vocab_size': 1000,
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 8,
            'd_ff': 1024
        }
        
        model = CustomTransformerModel(config)
        model.to(self.device)
        
        # Initialize framework components
        framework = PyTorchPrimaryFrameworkSystem()
        
        # Set up optimizer and loss function
        optimizer = Optimizers.create_optimizer(
            model.parameters(),
            optimizer_type='adam',
            lr=1e-4,
            weight_decay=1e-5
        )
        
        loss_fn = LossFunctions.cross_entropy_loss
        
        # Set up learning rate scheduler
        scheduler = LearningRateSchedulers.create_scheduler(
            optimizer,
            scheduler_type='cosine',
            T_max=100,
            eta_min=1e-6
        )
        
        print(f"Created custom Transformer model with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Optimizer: {type(optimizer).__name__}")
        print(f"Loss function: {loss_fn.__name__}")
        print(f"Scheduler: {type(scheduler).__name__}")
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        src = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        tgt = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        
        output = model(src, tgt)
        loss = loss_fn(output.view(-1, 1000), tgt.view(-1))
        
        print(f"Forward pass successful: output shape {output.shape}, loss {loss.item():.4f}")
    
    def demonstrate_real_world_applications(self):
        """Demonstrate real-world applications."""
        print("\n=== Real-World Applications Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping real-world applications demonstration.")
            return
        
        # Application 1: Machine Translation
        print("1. Machine Translation:")
        translation_prompt = "Translate the following English text to French: 'Hello, how are you today?'"
        print(f"   Prompt: {translation_prompt}")
        
        # Application 2: Text Summarization
        print("\n2. Text Summarization:")
        summarization_prompt = "Summarize the following text: 'Artificial intelligence is transforming various industries...'"
        print(f"   Prompt: {summarization_prompt}")
        
        # Application 3: Question Answering
        print("\n3. Question Answering:")
        qa_prompt = "Answer the following question: 'What is the capital of France?'"
        print(f"   Prompt: {qa_prompt}")
        
        # Application 4: Code Generation
        print("\n4. Code Generation:")
        code_prompt = "Write a Python function to calculate the factorial of a number"
        print(f"   Prompt: {code_prompt}")
        
        # Application 5: Creative Writing
        print("\n5. Creative Writing:")
        creative_prompt = "Write a short story about a robot learning to paint"
        print(f"   Prompt: {creative_prompt}")
    
    def run_complete_demonstration(self):
        """Run the complete demonstration."""
        print("=== Complete Transformer and LLM System Demonstration ===")
        print("This demonstration showcases all features of the Transformer and LLM system.")
        
        # Setup
        self.setup_models()
        
        # Core demonstrations
        self.demonstrate_basic_transformer()
        self.demonstrate_attention_mechanisms()
        self.demonstrate_positional_encoding()
        self.demonstrate_text_generation()
        self.demonstrate_prompt_engineering()
        self.demonstrate_model_analysis()
        self.demonstrate_performance_optimization()
        self.demonstrate_framework_integration()
        self.demonstrate_real_world_applications()
        
        print("\n=== Complete Demonstration Finished ===")
        print("The Transformer and LLM system is now ready for use!")
        print("\nKey Features Demonstrated:")
        print("- ✅ Transformer architectures (Encoder-Decoder)")
        print("- ✅ Multi-head attention mechanisms")
        print("- ✅ Positional encoding techniques")
        print("- ✅ LLM text generation capabilities")
        print("- ✅ Prompt engineering templates")
        print("- ✅ Model analysis and optimization")
        print("- ✅ Framework integration")
        print("- ✅ Real-world applications")


def main():
    """Main function to run the complete demonstration."""
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run complete demonstration
    example = CompleteTransformerLLMExample()
    example.run_complete_demonstration()


if __name__ == "__main__":
    main()


