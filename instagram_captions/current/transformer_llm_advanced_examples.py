"""
Advanced Transformer and LLM Examples

This module provides advanced examples demonstrating Transformer architectures and
LLM capabilities integrated with the framework, including Hugging Face Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

# Import transformer LLM system
try:
    from transformer_llm_system import (
        TransformerModel, TransformerEncoder, TransformerDecoder,
        MultiHeadAttention, PositionalEncoding, LLMGenerator,
        PromptEngineer, LLMAnalyzer, create_transformer_model,
        PreTrainedModelManager, TransformersPipeline, get_available_models
    )
    TRANSFORMER_LLM_AVAILABLE = True
except ImportError:
    print("Warning: transformer_llm_system not found. Some examples may not work.")
    TRANSFORMER_LLM_AVAILABLE = False


class AdvancedTransformerLLMExamples:
    """Advanced examples for Transformer and LLM system."""
    
    def __init__(self, config_path: str = "transformer_llm_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using default config.")
            return {}
    
    def setup_components(self):
        """Set up Transformer and LLM components."""
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Transformer LLM system not available. Skipping setup.")
            return
        
        # Create different model configurations
        self.models = {}
        
        # Small model
        small_config = {
            'model_type': 'encoder_decoder',
            'src_vocab_size': 1000,
            'tgt_vocab_size': 1000,
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 8,
            'd_ff': 1024
        }
        self.models['small'] = create_transformer_model(small_config)
        
        # Medium model
        medium_config = {
            'model_type': 'encoder_decoder',
            'src_vocab_size': 2000,
            'tgt_vocab_size': 2000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'd_ff': 2048
        }
        self.models['medium'] = create_transformer_model(medium_config)
        
        # Move models to device
        for name, model in self.models.items():
            model.to(self.device)
            print(f"Created {name} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    def demonstrate_attention_mechanisms(self):
        """Demonstrate different attention mechanisms."""
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
        
        # Forward pass
        output, attention_weights = attention(query, key, value)
        
        print(f"Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
    
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
        
        # Create sample sequence
        seq_len = 50
        x = torch.randn(seq_len, d_model).to(self.device)
        
        # Apply positional encoding
        encoded = pos_encoding(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Encoded shape: {encoded.shape}")
        
        # Analyze positional patterns
        pos_diff = encoded - x
        print(f"Positional encoding magnitude: {pos_diff.abs().mean():.4f}")
    
    def demonstrate_prompt_engineering(self):
        """Demonstrate prompt engineering techniques."""
        print("\n=== Prompt Engineering Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping prompt engineering demonstration.")
            return
        
        # Create prompt engineer
        prompt_engineer = PromptEngineer()
        
        # Zero-shot prompt
        instruction = "Translate the following text to French: Hello, how are you?"
        zero_shot = prompt_engineer.create_prompt(instruction, 'zero_shot')
        print(f"Zero-shot prompt:\n{zero_shot}\n")
        
        # Few-shot prompt
        examples = [
            ("Hello", "Bonjour"),
            ("Good morning", "Bonjour"),
            ("How are you?", "Comment allez-vous?")
        ]
        few_shot = prompt_engineer.create_few_shot_prompt(examples, "Thank you very much")
        print(f"Few-shot prompt:\n{few_shot}\n")
        
        # Chain-of-thought prompt
        cot_instruction = "Solve this math problem: If a train travels 120 km in 2 hours, what is its speed?"
        cot_prompt = prompt_engineer.create_prompt(cot_instruction, 'chain_of_thought')
        print(f"Chain-of-thought prompt:\n{cot_prompt}\n")
    
    def demonstrate_model_comparison(self):
        """Compare different Transformer model configurations."""
        print("\n=== Model Comparison Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping model comparison demonstration.")
            return
        
        # Create sample data
        batch_size, seq_len = 2, 10
        src = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        tgt = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        
        # Compare models
        results = {}
        for name, model in self.models.items():
            model.eval()
            
            # Measure parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                output = model(src, tgt)
            inference_time = time.time() - start_time
            
            results[name] = {
                'parameters': num_params,
                'inference_time': inference_time,
                'output_shape': output.shape
            }
        
        # Print comparison
        print("Model Comparison Results:")
        print("-" * 60)
        print(f"{'Model':<10} {'Parameters':<12} {'Time (ms)':<10} {'Output Shape'}")
        print("-" * 60)
        for name, result in results.items():
            print(f"{name:<10} {result['parameters']:<12,} {result['inference_time']*1000:<10.2f} {result['output_shape']}")
    
    def demonstrate_pre_trained_models(self):
        """Demonstrate pre-trained models from Hugging Face."""
        print("\n=== Pre-trained Models Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping pre-trained models demonstration.")
            return
        
        try:
            # Get available models
            available_models = get_available_models()
            print("Available model categories:")
            for category, models in available_models.items():
                print(f"  {category}: {len(models)} models")
            
            # Demonstrate text generation with GPT-2
            print("\n--- Text Generation with GPT-2 ---")
            gpt2_manager = PreTrainedModelManager("gpt2", "causal_lm")
            model_info = gpt2_manager.get_model_info()
            print(f"Model info: {model_info}")
            
            # Generate text
            prompts = [
                "The future of artificial intelligence",
                "Once upon a time in a galaxy",
                "The best way to learn programming is"
            ]
            
            for prompt in prompts:
                generated = gpt2_manager.generate_text(
                    prompt,
                    max_length=50,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9
                )
                print(f"Prompt: {prompt}")
                print(f"Generated: {generated[0]}\n")
            
            # Demonstrate embeddings with BERT
            print("--- Embeddings with BERT ---")
            bert_manager = PreTrainedModelManager("bert-base-uncased", "auto")
            
            texts = [
                "Hello world",
                "Machine learning is fascinating",
                "Transformers are powerful models"
            ]
            
            embeddings = bert_manager.get_embeddings(texts, pooling="mean")
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Sample embedding (first text): {embeddings[0][:10]}...")
            
            # Demonstrate tokenization
            print("\n--- Tokenization Demo ---")
            tokenized = bert_manager.tokenize_text("Hello world, how are you?")
            print(f"Tokenized output keys: {list(tokenized.keys())}")
            print(f"Input IDs shape: {tokenized['input_ids'].shape}")
            print(f"Attention mask shape: {tokenized['attention_mask'].shape}")
            
        except Exception as e:
            print(f"Error with pre-trained models: {e}")
    
    def demonstrate_transformers_pipelines(self):
        """Demonstrate high-level Transformers pipelines."""
        print("\n=== Transformers Pipelines Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping pipelines demonstration.")
            return
        
        try:
            # Text classification pipeline
            print("--- Text Classification Pipeline ---")
            classifier = TransformersPipeline("text-classification", "distilbert-base-uncased")
            pipeline_info = classifier.get_pipeline_info()
            print(f"Pipeline info: {pipeline_info}")
            
            texts = [
                "I love this movie!",
                "This is terrible.",
                "The weather is nice today."
            ]
            
            results = classifier.process(texts)
            for text, result in zip(texts, results):
                print(f"Text: {text}")
                print(f"Classification: {result}\n")
            
            # Question answering pipeline
            print("--- Question Answering Pipeline ---")
            qa_pipeline = TransformersPipeline("question-answering", "distilbert-base-uncased")
            
            context = "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was constructed from 1887 to 1889 and is named after engineer Gustave Eiffel."
            questions = [
                "What is the Eiffel Tower?",
                "When was it constructed?",
                "Who is it named after?"
            ]
            
            for question in questions:
                result = qa_pipeline.process({
                    "question": question,
                    "context": context
                })
                print(f"Question: {question}")
                print(f"Answer: {result['answer']}")
                print(f"Confidence: {result['score']:.3f}\n")
            
        except Exception as e:
            print(f"Error with pipelines: {e}")
    
    def demonstrate_fine_tuning_preparation(self):
        """Demonstrate preparation for fine-tuning pre-trained models."""
        print("\n=== Fine-tuning Preparation Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping fine-tuning demonstration.")
            return
        
        try:
            # Load a model for fine-tuning
            print("--- Loading Model for Fine-tuning ---")
            model_manager = PreTrainedModelManager("bert-base-uncased", "sequence_classification")
            
            # Get model information
            model_info = model_manager.get_model_info()
            print(f"Model ready for fine-tuning:")
            print(f"  Parameters: {model_info['num_parameters']:,}")
            print(f"  Trainable: {model_info['trainable_parameters']:,}")
            print(f"  Vocab size: {model_info['vocab_size']}")
            print(f"  Max length: {model_info['max_length']}")
            
            # Demonstrate tokenization for training
            print("\n--- Tokenization for Training ---")
            training_texts = [
                "This is a positive example",
                "This is a negative example",
                "Another positive case",
                "Another negative case"
            ]
            
            tokenized = model_manager.tokenize_text(
                training_texts,
                max_length=128,
                padding=True,
                truncation=True
            )
            
            print(f"Tokenized batch shape: {tokenized['input_ids'].shape}")
            print(f"Attention mask shape: {tokenized['attention_mask'].shape}")
            
            # Show how to prepare labels
            labels = [1, 0, 1, 0]  # Binary classification labels
            print(f"Labels: {labels}")
            
            print("\nModel is ready for fine-tuning with custom dataset!")
            
        except Exception as e:
            print(f"Error with fine-tuning preparation: {e}")
    
    def demonstrate_model_analysis(self):
        """Demonstrate model analysis capabilities."""
        print("\n=== Model Analysis Demonstration ===")
        
        if not TRANSFORMER_LLM_AVAILABLE:
            print("Skipping model analysis demonstration.")
            return
        
        try:
            # Analyze custom model
            model = self.models['small']
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Custom Model Analysis:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
            
            # Analyze pre-trained model
            print("\nPre-trained Model Analysis:")
            gpt2_manager = PreTrainedModelManager("gpt2", "causal_lm")
            model_info = gpt2_manager.get_model_info()
            
            print(f"  Model: {model_info['model_name']}")
            print(f"  Parameters: {model_info['num_parameters']:,}")
            print(f"  Trainable: {model_info['trainable_parameters']:,}")
            print(f"  Vocab size: {model_info['vocab_size']}")
            print(f"  Max length: {model_info['max_length']}")
            
            # Memory usage estimation
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"\nGPU Memory Usage:")
                print(f"  Allocated: {memory_allocated:.2f} MB")
                print(f"  Reserved: {memory_reserved:.2f} MB")
            
        except Exception as e:
            print(f"Error with model analysis: {e}")
    
    def run_all_examples(self):
        """Run all demonstration examples."""
        print("=== Advanced Transformer and LLM Examples ===")
        
        self.setup_components()
        self.demonstrate_attention_mechanisms()
        self.demonstrate_positional_encoding()
        self.demonstrate_prompt_engineering()
        self.demonstrate_model_comparison()
        self.demonstrate_pre_trained_models()
        self.demonstrate_transformers_pipelines()
        self.demonstrate_fine_tuning_preparation()
        self.demonstrate_model_analysis()
        
        print("\n=== All Examples Complete ===")
        print("\nKey Features Demonstrated:")
        print("- ✅ Custom Transformer architectures")
        print("- ✅ Attention mechanisms and positional encoding")
        print("- ✅ Prompt engineering techniques")
        print("- ✅ Model comparison and analysis")
        print("- ✅ Pre-trained models from Hugging Face")
        print("- ✅ High-level Transformers pipelines")
        print("- ✅ Fine-tuning preparation")
        print("- ✅ Model analysis and memory usage")


def main():
    """Main function to run advanced examples."""
    examples = AdvancedTransformerLLMExamples()
    examples.run_all_examples()


if __name__ == "__main__":
    main()
