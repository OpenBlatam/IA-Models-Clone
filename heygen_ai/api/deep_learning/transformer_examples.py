from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import logging
import time
        from .transformers import create_transformer_model
        from .transformers import create_transformer_model
        from .transformers import create_transformer_model
        from .transformers import MultiHeadAttention
        from .transformers import MultiHeadAttention
        from .llm_utils import create_simple_vocabulary
        from .llm_utils import Tokenizer
        from .llm_utils import create_simple_vocabulary, Tokenizer
        from .transformers import create_transformer_model
        from .llm_utils import TextGenerator, GenerationConfig
        from .llm_utils import create_simple_vocabulary, Tokenizer
        from .transformers import create_transformer_model
        from .llm_utils import LLMInference
        from .llm_utils import create_simple_vocabulary, Tokenizer
        from .transformers import create_transformer_model
        from .llm_utils import LLMTraining
        from .transformers import create_transformer_model
        from .transformers import MultiHeadAttention
            from .transformers import create_transformer_model
from typing import Any, List, Dict, Optional
import asyncio
"""
Transformer and LLM Examples for HeyGen AI.

Comprehensive examples demonstrating transformer architectures and Large Language Models
following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


class TransformerExamples:
    """Examples of transformer model usage."""

    @staticmethod
    def basic_transformer_example():
        """Basic transformer model example."""
        
        # Create vocabulary
        vocabulary_size = 1000
        embedding_dimensions = 256
        num_attention_heads = 8
        num_layers = 6
        feed_forward_dimensions = 1024
        
        # Create model
        
        model = create_transformer_model(
            model_type="transformer",
            vocabulary_size=vocabulary_size,
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            feed_forward_dimensions=feed_forward_dimensions,
            dropout_probability=0.1
        )
        
        # Create sample data
        batch_size = 4
        seq_len = 32
        
        input_ids = torch.randint(0, vocabulary_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocabulary_size, (batch_size, seq_len))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, target_ids)
        
        logger.info(f"Transformer output shape: {outputs.shape}")
        logger.info(f"Expected shape: ({batch_size}, {seq_len}, {vocabulary_size})")
        
        return model, outputs

    @staticmethod
    def gpt_model_example():
        """GPT model example."""
        
        # Create vocabulary
        vocabulary_size = 1000
        embedding_dimensions = 256
        num_attention_heads = 8
        num_layers = 6
        feed_forward_dimensions = 1024
        
        # Create GPT model
        
        model = create_transformer_model(
            model_type="gpt",
            vocabulary_size=vocabulary_size,
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            feed_forward_dimensions=feed_forward_dimensions,
            dropout_probability=0.1
        )
        
        # Create sample data
        batch_size = 4
        seq_len = 32
        
        input_ids = torch.randint(0, vocabulary_size, (batch_size, seq_len))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
        
        logger.info(f"GPT output shape: {outputs.shape}")
        logger.info(f"Expected shape: ({batch_size}, {seq_len}, {vocabulary_size})")
        
        return model, outputs

    @staticmethod
    def bert_model_example():
        """BERT model example."""
        
        # Create vocabulary
        vocabulary_size = 1000
        embedding_dimensions = 256
        num_attention_heads = 8
        num_layers = 6
        feed_forward_dimensions = 1024
        
        # Create BERT model
        
        model = create_transformer_model(
            model_type="bert",
            vocabulary_size=vocabulary_size,
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            feed_forward_dimensions=feed_forward_dimensions,
            dropout_probability=0.1
        )
        
        # Create sample data
        batch_size = 4
        seq_len = 32
        
        input_ids = torch.randint(0, vocabulary_size, (batch_size, seq_len))
        token_type_ids = torch.randint(0, 2, (batch_size, seq_len))
        masked_lm_labels = torch.randint(-1, vocabulary_size, (batch_size, seq_len))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                masked_lm_labels=masked_lm_labels
            )
        
        logger.info(f"BERT sequence output shape: {outputs['sequence_output'].shape}")
        logger.info(f"BERT MLM logits shape: {outputs['mlm_logits'].shape}")
        logger.info(f"BERT MLM loss: {outputs['mlm_loss'].item():.4f}")
        
        return model, outputs

    @staticmethod
    def attention_visualization_example():
        """Attention visualization example."""
        
        # Create multi-head attention
        
        embedding_dimensions = 256
        num_attention_heads = 8
        
        attention = MultiHeadAttention(
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            dropout_probability=0.1
        )
        
        # Create sample data
        batch_size = 2
        seq_len = 16
        
        query = torch.randn(batch_size, seq_len, embedding_dimensions)
        key = torch.randn(batch_size, seq_len, embedding_dimensions)
        value = torch.randn(batch_size, seq_len, embedding_dimensions)
        
        # Forward pass
        attention.eval()
        with torch.no_grad():
            output, attention_weights = attention(query, key, value)
        
        logger.info(f"Attention output shape: {output.shape}")
        logger.info(f"Attention weights shape: {attention_weights.shape}")
        logger.info(f"Attention weights sum per head: {attention_weights.sum(dim=-1).mean(dim=(0, 1, 2))}")
        
        return attention, attention_weights

    @staticmethod
    def relative_position_encoding_example():
        """Relative position encoding example."""
        
        # Create multi-head attention with relative position encoding
        
        embedding_dimensions = 256
        num_attention_heads = 8
        
        attention = MultiHeadAttention(
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            dropout_probability=0.1,
            use_relative_position=True,
            max_relative_position=64
        )
        
        # Create sample data
        batch_size = 2
        seq_len = 32
        
        query = torch.randn(batch_size, seq_len, embedding_dimensions)
        key = torch.randn(batch_size, seq_len, embedding_dimensions)
        value = torch.randn(batch_size, seq_len, embedding_dimensions)
        
        # Forward pass
        attention.eval()
        with torch.no_grad():
            output, attention_weights = attention(query, key, value)
        
        logger.info(f"Relative position attention output shape: {output.shape}")
        logger.info(f"Relative position attention weights shape: {attention_weights.shape}")
        
        return attention, attention_weights


class LLMExamples:
    """Examples of LLM usage."""

    @staticmethod
    def tokenizer_example():
        """Tokenizer example."""
        
        # Create simple vocabulary
        texts = [
            "Hello world this is a test",
            "Another example text for tokenization",
            "Machine learning is fascinating",
            "Transformers are powerful models"
        ]
        
        
        vocabulary = create_simple_vocabulary(texts, min_frequency=1)
        
        # Create tokenizer
        
        tokenizer = Tokenizer(vocabulary)
        
        # Test encoding and decoding
        test_text = "Hello world machine learning"
        encoded = tokenizer.encode(test_text, add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        
        logger.info(f"Original text: {test_text}")
        logger.info(f"Encoded tokens: {encoded}")
        logger.info(f"Decoded text: {decoded}")
        
        # Batch encoding
        batch_texts = ["Hello world", "Machine learning", "Transformers"]
        batch_encoded = tokenizer.batch_encode(batch_texts, max_length=10, padding=True)
        
        logger.info(f"Batch input IDs shape: {batch_encoded['input_ids'].shape}")
        logger.info(f"Batch attention mask shape: {batch_encoded['attention_mask'].shape}")
        
        return tokenizer, batch_encoded

    @staticmethod
    def text_generation_example():
        """Text generation example."""
        
        # Create vocabulary and tokenizer
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning algorithms are powerful tools",
            "Natural language processing is a fascinating field",
            "Deep learning models can understand complex patterns"
        ]
        
        
        vocabulary = create_simple_vocabulary(texts, min_frequency=1)
        tokenizer = Tokenizer(vocabulary)
        
        # Create simple model
        vocabulary_size = len(vocabulary)
        embedding_dimensions = 128
        num_attention_heads = 4
        num_layers = 2
        feed_forward_dimensions = 512
        
        
        model = create_transformer_model(
            model_type="gpt",
            vocabulary_size=vocabulary_size,
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            feed_forward_dimensions=feed_forward_dimensions
        )
        
        # Create text generator
        
        generation_config = GenerationConfig(
            max_length=20,
            temperature=1.0,
            top_k=10,
            top_p=0.9,
            do_sample=True
        )
        
        generator = TextGenerator(model, tokenizer)
        
        # Generate text
        prompt = "The quick brown"
        generated_text = generator.generate(prompt, generation_config)
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated text: {generated_text}")
        
        return generator, generated_text

    @staticmethod
    def llm_inference_example():
        """LLM inference example."""
        
        # Create vocabulary and tokenizer
        texts = [
            "Hello world this is a test",
            "Machine learning is amazing",
            "Transformers are powerful",
            "Natural language processing"
        ]
        
        
        vocabulary = create_simple_vocabulary(texts, min_frequency=1)
        tokenizer = Tokenizer(vocabulary)
        
        # Create model
        vocabulary_size = len(vocabulary)
        embedding_dimensions = 128
        num_attention_heads = 4
        num_layers = 2
        feed_forward_dimensions = 512
        
        
        model = create_transformer_model(
            model_type="gpt",
            vocabulary_size=vocabulary_size,
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            feed_forward_dimensions=feed_forward_dimensions
        )
        
        # Create LLM inference
        
        llm_inference = LLMInference(model, tokenizer)
        
        # Generate text
        prompt = "Hello world"
        generated_text = llm_inference.generate_text(
            prompt=prompt,
            max_length=15,
            temperature=1.0,
            top_k=10,
            top_p=0.9
        )
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated text: {generated_text}")
        
        # Get embeddings
        test_texts = ["Hello world", "Machine learning"]
        embeddings = llm_inference.get_embeddings(test_texts)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Embedding similarity: {F.cosine_similarity(embeddings[0:1], embeddings[1:2]).item():.4f}")
        
        return llm_inference, generated_text, embeddings

    @staticmethod
    def llm_training_example():
        """LLM training example."""
        
        # Create vocabulary and tokenizer
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning algorithms are powerful tools for data analysis",
            "Natural language processing enables computers to understand human language",
            "Deep learning models can learn complex patterns from large datasets",
            "Transformers have revolutionized the field of natural language processing"
        ]
        
        
        vocabulary = create_simple_vocabulary(texts, min_frequency=1)
        tokenizer = Tokenizer(vocabulary)
        
        # Create model
        vocabulary_size = len(vocabulary)
        embedding_dimensions = 128
        num_attention_heads = 4
        num_layers = 2
        feed_forward_dimensions = 512
        
        
        model = create_transformer_model(
            model_type="gpt",
            vocabulary_size=vocabulary_size,
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            feed_forward_dimensions=feed_forward_dimensions
        )
        
        # Create LLM training
        
        llm_training = LLMTraining(model, tokenizer)
        
        # Prepare training data
        training_data = llm_training.prepare_training_data(
            texts=texts,
            max_length=16,
            stride=8
        )
        
        logger.info(f"Number of training samples: {len(training_data)}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Training loop
        num_epochs = 5
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in training_data:
                loss = llm_training.train_step(batch, optimizer)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        
        # Evaluate model
        evaluation_metrics = llm_training.evaluate(texts[:2])
        logger.info(f"Evaluation metrics: {evaluation_metrics}")
        
        return llm_training, evaluation_metrics


class AdvancedTransformerExamples:
    """Advanced transformer examples."""

    @staticmethod
    def multi_task_transformer_example():
        """Multi-task transformer example."""
        
        # Create vocabulary
        vocabulary_size = 1000
        embedding_dimensions = 256
        num_attention_heads = 8
        num_layers = 6
        feed_forward_dimensions = 1024
        
        # Create transformer model
        
        model = create_transformer_model(
            model_type="transformer",
            vocabulary_size=vocabulary_size,
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            feed_forward_dimensions=feed_forward_dimensions
        )
        
        # Create sample data for multiple tasks
        batch_size = 4
        seq_len = 32
        
        # Translation task
        source_ids = torch.randint(0, vocabulary_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocabulary_size, (batch_size, seq_len))
        
        # Summarization task
        document_ids = torch.randint(0, vocabulary_size, (batch_size, seq_len * 2))
        summary_ids = torch.randint(0, vocabulary_size, (batch_size, seq_len // 2))
        
        # Forward pass for translation
        model.eval()
        with torch.no_grad():
            translation_outputs = model(source_ids, target_ids)
            summary_outputs = model(document_ids, summary_ids)
        
        logger.info(f"Translation output shape: {translation_outputs.shape}")
        logger.info(f"Summary output shape: {summary_outputs.shape}")
        
        return model, translation_outputs, summary_outputs

    @staticmethod
    def attention_analysis_example():
        """Attention analysis example."""
        
        # Create multi-head attention
        
        embedding_dimensions = 256
        num_attention_heads = 8
        
        attention = MultiHeadAttention(
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            dropout_probability=0.1,
            use_relative_position=True
        )
        
        # Create sample data with specific patterns
        batch_size = 1
        seq_len = 16
        
        # Create input with repeating patterns
        query = torch.randn(batch_size, seq_len, embedding_dimensions)
        key = torch.randn(batch_size, seq_len, embedding_dimensions)
        value = torch.randn(batch_size, seq_len, embedding_dimensions)
        
        # Forward pass
        attention.eval()
        with torch.no_grad():
            output, attention_weights = attention(query, key, value)
        
        # Analyze attention patterns
        attention_weights_mean = attention_weights.mean(dim=1)  # Average over heads
        attention_weights_max = attention_weights.max(dim=1)[0]  # Max over heads
        
        logger.info(f"Attention weights mean shape: {attention_weights_mean.shape}")
        logger.info(f"Attention weights max shape: {attention_weights_max.shape}")
        
        # Analyze attention distribution
        attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1)
        logger.info(f"Attention entropy shape: {attention_entropy.shape}")
        logger.info(f"Average attention entropy: {attention_entropy.mean().item():.4f}")
        
        return attention, attention_weights, attention_entropy

    @staticmethod
    def transformer_benchmark_example():
        """Transformer benchmark example."""
        
        # Create different model sizes
        model_configs = [
            {"name": "Small", "embedding_dim": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 512},
            {"name": "Medium", "embedding_dim": 256, "num_heads": 8, "num_layers": 4, "ff_dim": 1024},
            {"name": "Large", "embedding_dim": 512, "num_heads": 16, "num_layers": 8, "ff_dim": 2048}
        ]
        
        vocabulary_size = 1000
        batch_size = 4
        seq_len = 64
        
        benchmark_results = {}
        
        for config in model_configs:
            logger.info(f"Benchmarking {config['name']} model...")
            
            # Create model
            
            model = create_transformer_model(
                model_type="gpt",
                vocabulary_size=vocabulary_size,
                embedding_dimensions=config["embedding_dim"],
                num_attention_heads=config["num_heads"],
                num_layers=config["num_layers"],
                feed_forward_dimensions=config["ff_dim"]
            )
            
            # Create sample data
            input_ids = torch.randint(0, vocabulary_size, (batch_size, seq_len))
            
            # Benchmark forward pass
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_ids)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(input_ids)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100
            benchmark_results[config["name"]] = {
                "avg_time": avg_time,
                "params": sum(p.numel() for p in model.parameters()),
                "embedding_dim": config["embedding_dim"],
                "num_heads": config["num_heads"],
                "num_layers": config["num_layers"]
            }
            
            logger.info(f"{config['name']} model - Avg time: {avg_time:.4f}s, Params: {benchmark_results[config['name']]['params']:,}")
        
        return benchmark_results


def run_transformer_llm_examples():
    """Run all transformer and LLM examples."""
    
    logger.info("Running Transformer and LLM Examples")
    logger.info("=" * 60)
    
    # Basic transformer examples
    logger.info("\n1. Basic Transformer Examples:")
    transformer_model, transformer_outputs = TransformerExamples.basic_transformer_example()
    
    logger.info("\n2. GPT Model Example:")
    gpt_model, gpt_outputs = TransformerExamples.gpt_model_example()
    
    logger.info("\n3. BERT Model Example:")
    bert_model, bert_outputs = TransformerExamples.bert_model_example()
    
    logger.info("\n4. Attention Visualization Example:")
    attention_model, attention_weights = TransformerExamples.attention_visualization_example()
    
    logger.info("\n5. Relative Position Encoding Example:")
    relative_attention, relative_weights = TransformerExamples.relative_position_encoding_example()
    
    # LLM examples
    logger.info("\n6. Tokenizer Example:")
    tokenizer, batch_encoded = LLMExamples.tokenizer_example()
    
    logger.info("\n7. Text Generation Example:")
    generator, generated_text = LLMExamples.text_generation_example()
    
    logger.info("\n8. LLM Inference Example:")
    llm_inference, inference_text, embeddings = LLMExamples.llm_inference_example()
    
    logger.info("\n9. LLM Training Example:")
    llm_training, training_metrics = LLMExamples.llm_training_example()
    
    # Advanced examples
    logger.info("\n10. Multi-Task Transformer Example:")
    multi_task_model, translation_outputs, summary_outputs = AdvancedTransformerExamples.multi_task_transformer_example()
    
    logger.info("\n11. Attention Analysis Example:")
    attention_analysis, analysis_weights, attention_entropy = AdvancedTransformerExamples.attention_analysis_example()
    
    logger.info("\n12. Transformer Benchmark Example:")
    benchmark_results = AdvancedTransformerExamples.transformer_benchmark_example()
    
    logger.info("\nAll transformer and LLM examples completed successfully!")
    
    return {
        "transformer_models": {
            "basic": transformer_model,
            "gpt": gpt_model,
            "bert": bert_model,
            "attention": attention_model,
            "relative_attention": relative_attention
        },
        "llm_components": {
            "tokenizer": tokenizer,
            "generator": generator,
            "inference": llm_inference,
            "training": llm_training
        },
        "advanced": {
            "multi_task": multi_task_model,
            "attention_analysis": attention_analysis,
            "benchmark": benchmark_results
        },
        "outputs": {
            "transformer_outputs": transformer_outputs,
            "gpt_outputs": gpt_outputs,
            "bert_outputs": bert_outputs,
            "attention_weights": attention_weights,
            "relative_weights": relative_weights,
            "batch_encoded": batch_encoded,
            "generated_text": generated_text,
            "inference_text": inference_text,
            "embeddings": embeddings,
            "training_metrics": training_metrics,
            "translation_outputs": translation_outputs,
            "summary_outputs": summary_outputs,
            "analysis_weights": analysis_weights,
            "attention_entropy": attention_entropy
        }
    }


if __name__ == "__main__":
    # Run examples
    examples = run_transformer_llm_examples()
    logger.info("Transformer and LLM Examples completed!") 