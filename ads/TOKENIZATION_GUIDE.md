# Advanced Tokenization and Sequence Handling Guide

This guide covers the comprehensive tokenization and sequence handling implementation for the ads generation feature, including text preprocessing, advanced tokenization strategies, and sequence optimization.

## Overview

The tokenization system provides:
- **Advanced Text Preprocessing**: Normalization, cleaning, and optimization
- **Intelligent Tokenization**: Caching, optimization, and model-specific handling
- **Sequence Management**: Training and inference sequence creation
- **Performance Optimization**: Batch processing and memory management
- **Analysis Tools**: Text complexity analysis and statistics

## Architecture

### Core Components

1. **TextPreprocessor**: Handles text cleaning and normalization
2. **AdvancedTokenizer**: Provides caching and optimization
3. **SequenceManager**: Manages training and inference sequences
4. **OptimizedAdsDataset**: Dataset with caching and optimization
5. **TokenizationService**: Main service orchestrating all components

### Key Features

- **Caching**: Intelligent caching of tokenization results
- **Special Tokens**: Domain-specific tokens for ads generation
- **Sequence Optimization**: Automatic sequence length optimization
- **Batch Processing**: Efficient batch tokenization
- **Memory Management**: Optimized memory usage for large datasets

## Text Preprocessing

### Normalization

```python
from onyx.server.features.ads.tokenization_service import TextPreprocessor

preprocessor = TextPreprocessor()

# Normalize text
normalized_text = preprocessor.normalize_text("Hello World! Visit https://example.com")
# Result: "hello world! visit [URL]"
```

### Cleaning for Ads

```python
# Clean text specifically for ads
clean_text = preprocessor.clean_ads_text(
    "Amazing product for sale! Call 555-1234 or email info@example.com",
    remove_stopwords=False
)
```

### Keyword Extraction

```python
# Extract keywords from text
keywords = preprocessor.extract_keywords(
    "Our premium product offers amazing features and benefits",
    max_keywords=5
)
# Result: ['premium', 'product', 'amazing', 'features', 'benefits']
```

### Text Segmentation

```python
# Segment long text into smaller chunks
segments = preprocessor.segment_text(
    "Long text content...",
    max_segment_length=512
)
```

## Advanced Tokenization

### Basic Tokenization

```python
from onyx.server.features.ads.tokenization_service import AdvancedTokenizer

tokenizer = AdvancedTokenizer("microsoft/DialoGPT-medium")

# Tokenize text with caching
tokens = tokenizer.tokenize_text(
    "Generate an ad for our product",
    max_length=512,
    use_cache=True
)
```

### Ads-Specific Tokenization

```python
# Tokenize ads prompt with structured format
tokens = tokenizer.tokenize_ads_prompt(
    prompt="Generate an ad for our product",
    target_audience="Young professionals",
    keywords=["premium", "quality", "affordable"],
    brand="OurBrand",
    max_length=512
)
```

### Special Tokens

The system includes domain-specific special tokens:
- `[AD_START]` / `[AD_END]`: Ad content boundaries
- `[TARGET_AUDIENCE]`: Target audience specification
- `[KEYWORDS]`: Keyword list
- `[BRAND]`: Brand name
- `[CTA]`: Call-to-action
- `[URL]`, `[EMAIL]`, `[PHONE]`: Contact information placeholders

## Sequence Management

### Training Sequences

```python
from onyx.server.features.ads.tokenization_service import SequenceManager

sequence_manager = SequenceManager(tokenizer)

# Create training sequences
sequences = sequence_manager.create_training_sequences(
    prompts=["Generate an ad for product X"],
    targets=["Amazing product X offers..."],
    max_length=512
)
```

### Inference Sequences

```python
# Create sequence for inference
sequence = sequence_manager.create_inference_sequence(
    prompt="Generate an ad for our new product",
    max_length=512
)
```

### Sequence Padding

```python
# Pad sequences to same length
padded_sequences = sequence_manager.pad_sequences(
    sequences,
    padding="max_length",
    max_length=512
)
```

## Dataset Management

### Optimized Dataset

```python
from onyx.server.features.ads.tokenization_service import OptimizedAdsDataset

# Create dataset with caching
dataset = OptimizedAdsDataset(
    data=training_data,
    tokenizer=tokenizer,
    max_length=512,
    use_cache=True
)
```

### DataLoader Creation

```python
from onyx.server.features.ads.tokenization_service import TokenizationService

tokenization_service = TokenizationService()

# Create training dataset with DataLoader
dataloader = await tokenization_service.create_training_dataset(
    ads_data=training_data,
    max_length=512,
    batch_size=8
)
```

## Text Analysis

### Complexity Analysis

```python
# Analyze text complexity
analysis = await tokenization_service.analyze_text_complexity(
    "Your text content here"
)

# Returns:
# {
#     'token_count': 15,
#     'word_count': 12,
#     'sentence_count': 2,
#     'avg_word_length': 4.5,
#     'avg_sentence_length': 6.0,
#     'unique_words': 10,
#     'vocabulary_diversity': 0.83,
#     'keywords': ['text', 'content'],
#     'complexity_score': 1.25
# }
```

### Tokenization Statistics

```python
# Get statistics for dataset
stats = await tokenization_service.get_tokenization_stats(ads_data)

# Returns:
# {
#     'total_items': 1000,
#     'total_tokens': 50000,
#     'total_words': 45000,
#     'avg_tokens_per_item': 50,
#     'avg_words_per_item': 45,
#     'avg_complexity_score': 1.11,
#     'vocabulary_size': 50257
# }
```

## API Endpoints

### Text Preprocessing

```bash
POST /tokenization/preprocess
{
    "text": "Your text to preprocess",
    "remove_stopwords": false,
    "normalize": true
}
```

### Tokenization

```bash
POST /tokenization/tokenize
{
    "text": "Text to tokenize",
    "max_length": 512,
    "model_name": "microsoft/DialoGPT-medium",
    "include_analysis": true
}
```

### Ads Prompt Tokenization

```bash
POST /tokenization/tokenize-ads-prompt
{
    "prompt": "Generate an ad for our product",
    "target_audience": "Young professionals",
    "keywords": ["premium", "quality"],
    "brand": "OurBrand",
    "max_length": 512
}
```

### Sequence Optimization

```bash
POST /tokenization/optimize-sequences
{
    "texts": ["Text 1", "Text 2", "Text 3"],
    "target_token_count": 512,
    "model_name": "microsoft/DialoGPT-medium"
}
```

### Batch Tokenization

```bash
POST /tokenization/batch-tokenize
{
    "texts": ["Text 1", "Text 2", "Text 3"],
    "max_length": 512,
    "model_name": "microsoft/DialoGPT-medium",
    "include_analysis": true
}
```

### Text Analysis

```bash
POST /tokenization/analyze-text
{
    "text": "Text to analyze",
    "model_name": "microsoft/DialoGPT-medium"
}
```

## Performance Optimization

### Caching Strategy

- **Tokenization Cache**: Caches tokenization results for repeated texts
- **Model Cache**: Caches loaded tokenizers and models
- **Sequence Cache**: Caches processed sequences in dataset
- **Redis Cache**: Distributed caching for production environments

### Memory Management

- **Lazy Loading**: Tokenizers loaded only when needed
- **Batch Processing**: Efficient batch tokenization
- **Memory Cleanup**: Automatic cleanup of unused resources
- **Garbage Collection**: Proper cleanup of temporary objects

### Optimization Techniques

1. **Sequence Length Optimization**: Automatic segmentation of long texts
2. **Batch Size Optimization**: Dynamic batch size based on memory
3. **Parallel Processing**: Concurrent tokenization for batch operations
4. **Memory Mapping**: Efficient handling of large datasets

## Integration with Fine-tuning

### Training Data Preparation

```python
# Prepare training data with advanced tokenization
training_data, total_samples = await fine_tuning_service.prepare_training_data(
    user_id=user_id,
    model_name="microsoft/DialoGPT-medium",
    max_samples=1000
)

# Create dataset with tokenization service
dataset = await tokenization_service.create_training_dataset(
    ads_data=training_data,
    max_length=512,
    batch_size=8
)
```

### Model Training

```python
# Fine-tune with optimized tokenization
metrics = await fine_tuning_service.fine_tune_lora(
    user_id=user_id,
    base_model_name="microsoft/DialoGPT-medium",
    epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_length=512
)
```

### Inference

```python
# Generate with fine-tuned model using advanced tokenization
generated_text = await fine_tuning_service.generate_with_finetuned_model(
    user_id=user_id,
    prompt="Generate an ad for our product",
    base_model_name="microsoft/DialoGPT-medium",
    max_length=200,
    temperature=0.7
)
```

## Configuration

### Environment Variables

```bash
# Tokenization settings
TOKENIZATION_CACHE_SIZE=1000
TOKENIZATION_MAX_LENGTH=512
TOKENIZATION_BATCH_SIZE=8
TOKENIZATION_USE_CACHE=true

# Model settings
DEFAULT_MODEL_NAME=microsoft/DialoGPT-medium
TOKENIZATION_DEVICE=cuda  # or cpu
```

### Settings Configuration

```python
# In optimized_config.py
class TokenizationSettings(BaseSettings):
    cache_size: int = 1000
    max_length: int = 512
    batch_size: int = 8
    use_cache: bool = True
    default_model: str = "microsoft/DialoGPT-medium"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

## Monitoring and Logging

### Performance Metrics

- Tokenization speed (tokens/second)
- Cache hit rates
- Memory usage
- Processing time per request
- Error rates

### Health Checks

```bash
GET /tokenization/health
```

### Logging

```python
# Log tokenization events
logger.info(f"Tokenized {token_count} tokens in {processing_time}ms")
logger.warning(f"Cache miss for text: {text[:50]}...")
logger.error(f"Tokenization failed: {error}")
```

## Best Practices

### Text Preprocessing

1. **Always normalize text** before tokenization
2. **Use domain-specific cleaning** for ads content
3. **Extract keywords** for better model understanding
4. **Segment long texts** to avoid truncation

### Tokenization

1. **Use caching** for repeated texts
2. **Choose appropriate max_length** based on model
3. **Include special tokens** for domain-specific features
4. **Monitor token count** to avoid memory issues

### Sequence Management

1. **Optimize sequence lengths** for batch processing
2. **Use proper padding** strategies
3. **Handle variable-length sequences** efficiently
4. **Cache processed sequences** when possible

### Performance

1. **Use batch processing** for multiple texts
2. **Monitor memory usage** and adjust batch sizes
3. **Implement proper error handling** for failed tokenizations
4. **Use async operations** for I/O-bound tasks

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or sequence length
2. **Slow Tokenization**: Enable caching and use batch processing
3. **Model Loading Errors**: Check model name and dependencies
4. **Cache Issues**: Clear cache or increase cache size

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('onyx.server.features.ads.tokenization_service').setLevel(logging.DEBUG)
```

### Performance Profiling

```python
# Profile tokenization performance
import time
start_time = time.time()
tokens = tokenizer.tokenize_text(text)
processing_time = time.time() - start_time
print(f"Tokenization took {processing_time:.3f} seconds")
```

## Future Enhancements

1. **Multi-language Support**: Support for multiple languages
2. **Custom Tokenizers**: Domain-specific tokenizer training
3. **Advanced Caching**: Distributed caching with Redis
4. **Real-time Optimization**: Dynamic sequence optimization
5. **Model-specific Optimizations**: Optimizations for different model architectures

This comprehensive tokenization system provides the foundation for efficient and accurate ads generation with proper text handling, sequence management, and performance optimization. 