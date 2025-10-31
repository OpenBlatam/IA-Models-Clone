from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import numpy as np
import logging
from typing import List, Dict, Any
from pathlib import Path
import json
from tokenization_utils import (
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import asyncio
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Example Usage of Advanced Tokenization and Sequence Handling
Demonstrates comprehensive tokenization utilities for SEO text processing
"""


    AdvancedTokenizer, SequenceHandler, TokenizedDataset, TokenizationPipeline,
    TokenizationConfig, SequenceConfig, analyze_tokenization_quality,
    optimize_tokenization_config, create_data_collator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_basic_tokenization():
    """Example of basic tokenization functionality"""
    
    logger.info("=== Basic Tokenization Example ===")
    
    # Create tokenization configuration
    config = TokenizationConfig(
        model_name="bert-base-uncased",
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True
    )
    
    # Initialize advanced tokenizer
    tokenizer = AdvancedTokenizer(config)
    
    # Sample SEO texts
    sample_texts = [
        "SEO optimization is crucial for website visibility and search engine rankings.",
        "Digital marketing strategies include content marketing, social media, and PPC advertising.",
        "Keyword research helps identify high-value search terms for content optimization.",
        "Technical SEO involves website structure, page speed, and mobile optimization.",
        "Content marketing focuses on creating valuable, relevant content for target audiences."
    ]
    
    logger.info(f"Sample texts: {len(sample_texts)}")
    
    # Tokenize individual texts
    for i, text in enumerate(sample_texts):
        logger.info(f"\nText {i+1}: {text[:50]}...")
        
        # Tokenize with caching
        result = tokenizer.tokenize_text(text, use_cache=True)
        
        logger.info(f"Input IDs shape: {result['input_ids'].shape}")
        logger.info(f"Attention mask shape: {result['attention_mask'].shape}")
        logger.info(f"Token count: {result['input_ids'].shape[-1]}")
        
        # Decode tokens back to text
        decoded = tokenizer.tokenizer.decode(result['input_ids'][0], skip_special_tokens=True)
        logger.info(f"Decoded: {decoded[:50]}...")
    
    # Get tokenization statistics
    stats = tokenizer.get_stats()
    logger.info(f"\nTokenization Statistics:")
    logger.info(f"Total tokens: {stats.total_tokens}")
    logger.info(f"Unique tokens: {stats.unique_tokens}")
    logger.info(f"Average sequence length: {stats.avg_sequence_length:.2f}")
    logger.info(f"Vocabulary size: {stats.vocabulary_size}")

def example_batch_tokenization():
    """Example of batch tokenization with optimization"""
    
    logger.info("\n=== Batch Tokenization Example ===")
    
    # Create configuration
    config = TokenizationConfig(
        model_name="bert-base-uncased",
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    tokenizer = AdvancedTokenizer(config)
    
    # Sample batch of texts
    batch_texts = [
        "SEO optimization techniques for better rankings.",
        "Content marketing strategies for business growth.",
        "Technical SEO audit checklist for websites.",
        "Local SEO optimization for small businesses.",
        "E-commerce SEO best practices for online stores."
    ]
    
    logger.info(f"Batch size: {len(batch_texts)}")
    
    # Tokenize batch
    batch_result = tokenizer.tokenize_batch(batch_texts, use_cache=True)
    
    logger.info(f"Batch input IDs shape: {batch_result['input_ids'].shape}")
    logger.info(f"Batch attention mask shape: {batch_result['attention_mask'].shape}")
    
    # Process individual items in batch
    for i in range(len(batch_texts)):
        input_ids = batch_result['input_ids'][i]
        attention_mask = batch_result['attention_mask'][i]
        
        logger.info(f"Item {i+1}: {batch_texts[i][:30]}...")
        logger.info(f"  Input IDs: {input_ids.shape}")
        logger.info(f"  Non-padding tokens: {(attention_mask == 1).sum().item()}")

def example_sequence_handling():
    """Example of advanced sequence handling for long texts"""
    
    logger.info("\n=== Sequence Handling Example ===")
    
    # Create sequence configuration
    sequence_config = SequenceConfig(
        max_sequence_length=256,
        chunk_strategy="sentence",
        overlap_strategy="sliding_window",
        overlap_size=50,
        preserve_boundaries=True
    )
    
    sequence_handler = SequenceHandler(sequence_config)
    
    # Load tokenizer for sequence handling
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Long text example
    long_text = """
    Search Engine Optimization (SEO) is a comprehensive digital marketing strategy that focuses on improving a website's visibility in search engine results pages (SERPs). 
    
    The primary goal of SEO is to increase organic traffic by optimizing various on-page and off-page factors. On-page SEO includes optimizing content, meta tags, headings, and internal linking structure. Technical SEO involves improving website speed, mobile responsiveness, and crawlability.
    
    Off-page SEO focuses on building quality backlinks from authoritative websites and improving domain authority. Content marketing plays a crucial role in SEO by creating valuable, relevant content that answers user queries and satisfies search intent.
    
    Keyword research is fundamental to SEO success, helping identify high-value search terms with reasonable competition. Local SEO is essential for businesses targeting specific geographic areas, involving optimization for local search results and Google My Business profiles.
    
    E-commerce SEO requires special attention to product pages, category pages, and user experience optimization. Regular SEO audits help identify issues and opportunities for improvement, ensuring sustained organic growth.
    """
    
    logger.info(f"Original text length: {len(long_text)} characters")
    
    # Split text into chunks
    chunks = sequence_handler.split_text_into_chunks(long_text, tokenizer)
    
    logger.info(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        logger.info(f"\nChunk {i+1}:")
        logger.info(f"Length: {len(chunk)} characters")
        logger.info(f"Content: {chunk[:100]}...")
    
    # Create sliding windows
    windows = sequence_handler.create_sliding_windows(long_text, tokenizer)
    
    logger.info(f"\nNumber of sliding windows: {len(windows)}")
    
    for i, window in enumerate(windows):
        logger.info(f"Window {i+1}: {len(window)} characters")

def example_tokenized_dataset():
    """Example of using TokenizedDataset with caching"""
    
    logger.info("\n=== Tokenized Dataset Example ===")
    
    # Create tokenizer configuration
    config = TokenizationConfig(
        model_name="bert-base-uncased",
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    tokenizer = AdvancedTokenizer(config)
    
    # Sample data
    texts = [
        "SEO optimization improves website rankings.",
        "Content marketing drives organic traffic.",
        "Technical SEO enhances user experience.",
        "Local SEO targets geographic audiences.",
        "E-commerce SEO optimizes product pages."
    ]
    
    labels = [1, 1, 1, 0, 0]  # 1 for good, 0 for needs improvement
    
    # Create dataset with caching
    cache_dir = "./tokenization_cache"
    dataset = TokenizedDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=128,
        cache_dir=cache_dir
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Access dataset items
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        logger.info(f"\nItem {i+1}:")
        logger.info(f"Input IDs shape: {item['input_ids'].shape}")
        logger.info(f"Attention mask shape: {item['attention_mask'].shape}")
        logger.info(f"Label: {item['labels'].item()}")
    
    # Save cache
    dataset.save_cache()
    
    # Create data loader
    data_collator = create_data_collator(tokenizer.tokenizer, "sequence_classification")
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=data_collator
    )
    
    logger.info(f"\nDataLoader created with batch size: 2")
    
    # Iterate through batches
    for batch_idx, batch in enumerate(dataloader):
        logger.info(f"\nBatch {batch_idx + 1}:")
        logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        
        if batch_idx >= 1:  # Show only first 2 batches
            break

def example_tokenization_pipeline():
    """Example of complete tokenization pipeline"""
    
    logger.info("\n=== Tokenization Pipeline Example ===")
    
    # Create configurations
    tokenizer_config = TokenizationConfig(
        model_name="bert-base-uncased",
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True
    )
    
    sequence_config = SequenceConfig(
        max_sequence_length=256,
        chunk_strategy="sentence",
        overlap_strategy="sliding_window",
        overlap_size=50
    )
    
    # Create pipeline
    pipeline = TokenizationPipeline(tokenizer_config, sequence_config)
    
    # Sample texts
    texts = [
        "SEO optimization is essential for digital marketing success.",
        "This is a very long text that will be processed by the tokenization pipeline. " * 20,
        "Short text for testing.",
        "Content marketing and SEO work together to improve search rankings and drive organic traffic to websites."
    ]
    
    logger.info(f"Processing {len(texts)} texts through pipeline")
    
    # Process individual texts
    for i, text in enumerate(texts):
        logger.info(f"\nText {i+1}: {text[:50]}...")
        
        result = pipeline.process_text(text)
        
        logger.info(f"Input IDs shape: {result['input_ids'].shape}")
        logger.info(f"Token count: {result['input_ids'].shape[-1]}")
    
    # Process batch
    batch_result = pipeline.process_batch(texts)
    
    logger.info(f"\nBatch processing:")
    logger.info(f"Batch input IDs shape: {batch_result['input_ids'].shape}")
    logger.info(f"Batch attention mask shape: {batch_result['attention_mask'].shape}")
    
    # Get pipeline statistics
    stats = pipeline.get_statistics()
    logger.info(f"\nPipeline Statistics:")
    logger.info(f"Total tokens processed: {stats['tokenization_stats'].total_tokens}")
    logger.info(f"Cache size: {stats['cache_info']['cache_size']}")

def example_quality_analysis():
    """Example of tokenization quality analysis"""
    
    logger.info("\n=== Tokenization Quality Analysis Example ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Sample texts for analysis
    analysis_texts = [
        "SEO optimization techniques for better search rankings.",
        "Digital marketing strategies include content marketing and social media advertising.",
        "Technical SEO involves website structure and page speed optimization.",
        "Local SEO targets specific geographic areas for business growth.",
        "E-commerce SEO requires product page optimization and user experience improvements.",
        "Content marketing focuses on creating valuable, relevant content for target audiences.",
        "Keyword research helps identify high-value search terms for content optimization.",
        "Backlink building is crucial for improving domain authority and search rankings.",
        "Mobile optimization is essential for modern SEO success.",
        "User experience optimization improves engagement and conversion rates."
    ]
    
    # Analyze tokenization quality
    analysis = analyze_tokenization_quality(tokenizer, analysis_texts)
    
    logger.info("Tokenization Quality Analysis Results:")
    logger.info(f"Total texts analyzed: {analysis['total_texts']}")
    logger.info(f"Average text length: {analysis['avg_text_length']:.2f} characters")
    logger.info(f"Average tokens per text: {analysis['avg_tokens_per_text']:.2f}")
    logger.info(f"Average input IDs per text: {analysis['avg_input_ids_per_text']:.2f}")
    logger.info(f"Vocabulary coverage ratio: {analysis['vocabulary_coverage_ratio']:.4f}")
    
    # Show sequence length distribution
    logger.info(f"\nSequence Length Distribution:")
    for length, count in sorted(analysis['sequence_length_distribution'].items()):
        logger.info(f"  Length {length}: {count} texts")
    
    # Show most common tokens
    logger.info(f"\nMost Common Tokens:")
    sorted_tokens = sorted(analysis['vocabulary_coverage'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]
    for token, count in sorted_tokens:
        logger.info(f"  '{token}': {count} occurrences")

def example_config_optimization():
    """Example of tokenization configuration optimization"""
    
    logger.info("\n=== Configuration Optimization Example ===")
    
    # Sample texts for optimization
    optimization_texts = [
        "SEO optimization is crucial for website visibility.",
        "Digital marketing strategies include content marketing, social media, and PPC advertising.",
        "Technical SEO involves website structure, page speed, and mobile optimization.",
        "Content marketing focuses on creating valuable, relevant content for target audiences.",
        "Keyword research helps identify high-value search terms for content optimization.",
        "Local SEO optimization for small businesses in specific geographic areas.",
        "E-commerce SEO best practices for online stores and product page optimization.",
        "Backlink building strategies for improving domain authority and search rankings.",
        "Mobile optimization techniques for modern SEO success and user experience.",
        "User experience optimization for improved engagement and conversion rates."
    ]
    
    # Base configuration
    base_config = TokenizationConfig(
        model_name="bert-base-uncased",
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    logger.info("Base Configuration:")
    logger.info(f"Model: {base_config.model_name}")
    logger.info(f"Max length: {base_config.max_length}")
    
    # Optimize configuration
    optimized_config = optimize_tokenization_config(optimization_texts, base_config)
    
    logger.info("\nOptimized Configuration:")
    logger.info(f"Model: {optimized_config.model_name}")
    logger.info(f"Max length: {optimized_config.max_length}")
    logger.info(f"Truncation: {optimized_config.truncation}")
    logger.info(f"Padding: {optimized_config.padding}")
    
    # Test both configurations
    base_tokenizer = AdvancedTokenizer(base_config)
    optimized_tokenizer = AdvancedTokenizer(optimized_config)
    
    # Compare tokenization results
    test_text = optimization_texts[0]
    
    base_result = base_tokenizer.tokenize_text(test_text)
    optimized_result = optimized_tokenizer.tokenize_text(test_text)
    
    logger.info(f"\nComparison for text: {test_text}")
    logger.info(f"Base config tokens: {base_result['input_ids'].shape[-1]}")
    logger.info(f"Optimized config tokens: {optimized_result['input_ids'].shape[-1]}")

def example_advanced_features():
    """Example of advanced tokenization features"""
    
    logger.info("\n=== Advanced Features Example ===")
    
    # Create advanced configuration
    config = TokenizationConfig(
        model_name="bert-base-uncased",
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
        return_length=True,
        add_special_tokens=True
    )
    
    tokenizer = AdvancedTokenizer(config)
    
    # Test text with special tokens
    test_text = "SEO optimization [MASK] for better rankings."
    
    logger.info(f"Test text: {test_text}")
    
    # Tokenize with all features
    result = tokenizer.tokenize_text(
        test_text,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
        return_length=True
    )
    
    logger.info(f"Input IDs: {result['input_ids'].shape}")
    logger.info(f"Attention mask: {result['attention_mask'].shape}")
    
    if 'special_tokens_mask' in result:
        logger.info(f"Special tokens mask: {result['special_tokens_mask'].shape}")
    
    if 'offsets_mapping' in result:
        logger.info(f"Offsets mapping: {len(result['offsets_mapping'])} mappings")
    
    if 'length' in result:
        logger.info(f"Length: {result['length']}")
    
    # Test cache functionality
    logger.info(f"\nCache size before: {len(tokenizer.cache)}")
    
    # Tokenize same text again (should use cache)
    result2 = tokenizer.tokenize_text(test_text)
    logger.info(f"Cache size after: {len(tokenizer.cache)}")
    
    # Clear cache
    tokenizer.clear_cache()
    logger.info(f"Cache size after clearing: {len(tokenizer.cache)}")
    
    # Test batch processing with different lengths
    batch_texts = [
        "Short text.",
        "Medium length text for testing tokenization.",
        "This is a longer text that will be processed by the advanced tokenizer with various features enabled."
    ]
    
    batch_result = tokenizer.tokenize_batch(batch_texts)
    
    logger.info(f"\nBatch processing results:")
    logger.info(f"Batch input IDs shape: {batch_result['input_ids'].shape}")
    
    for i, text in enumerate(batch_texts):
        input_ids = batch_result['input_ids'][i]
        attention_mask = batch_result['attention_mask'][i]
        
        # Count non-padding tokens
        non_padding = (attention_mask == 1).sum().item()
        total_tokens = attention_mask.shape[-1]
        
        logger.info(f"Text {i+1}: {non_padding}/{total_tokens} tokens (non-padding/total)")

async def main():
    """Main function to run all examples"""
    
    logger.info("Starting Advanced Tokenization and Sequence Handling Examples")
    
    try:
        # Run all examples
        example_basic_tokenization()
        example_batch_tokenization()
        example_sequence_handling()
        example_tokenized_dataset()
        example_tokenization_pipeline()
        example_quality_analysis()
        example_config_optimization()
        example_advanced_features()
        
        logger.info("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 