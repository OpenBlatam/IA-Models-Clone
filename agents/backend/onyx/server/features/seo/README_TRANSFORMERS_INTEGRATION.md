# Enhanced Transformers Library Integration for SEO Service

This module provides comprehensive integration with the Hugging Face Transformers library, offering advanced capabilities for working with pre-trained models and tokenizers specifically designed for SEO tasks.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Advanced Features](#advanced-features)
7. [Model Management](#model-management)
8. [Tokenization](#tokenization)
9. [Pipeline Creation](#pipeline-creation)
10. [Fine-tuning](#fine-tuning)
11. [SEO-Specific Features](#seo-specific-features)
12. [Performance Optimization](#performance-optimization)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)
15. [API Reference](#api-reference)

## Overview

The enhanced Transformers library integration provides:

- **Advanced Model Loading**: Comprehensive support for all major transformer models
- **Flexible Tokenization**: Multiple tokenization strategies with custom options
- **Pipeline Creation**: Easy-to-use pipelines for various NLP tasks
- **Fine-tuning Support**: Complete fine-tuning workflow with Transformers Trainer
- **SEO-Specific Features**: Specialized functionality for SEO content analysis
- **Performance Optimization**: Memory-efficient loading and inference
- **Production Ready**: Optimized for deployment and scaling

## Features

### Core Features
- ✅ Support for all major transformer models (BERT, RoBERTa, GPT-2, T5, etc.)
- ✅ Advanced tokenization with custom options
- ✅ Multiple pooling strategies for embeddings
- ✅ Pipeline creation for various NLP tasks
- ✅ Fine-tuning with Transformers Trainer
- ✅ SEO-specific content analysis
- ✅ Batch processing capabilities
- ✅ Model information and size estimation

### Advanced Features
- ✅ Mixed precision training and inference
- ✅ Gradient checkpointing for memory efficiency
- ✅ Custom model registries
- ✅ Advanced embedding extraction
- ✅ Content similarity computation
- ✅ Model saving and loading
- ✅ Comprehensive error handling

## Architecture

### Components

```
Transformers Integration
├── TransformersModelManager
│   ├── Model Loading
│   ├── Tokenization
│   ├── Embedding Extraction
│   └── Pipeline Creation
├── SEOSpecificTransformers
│   ├── SEO Content Analysis
│   ├── Content Similarity
│   └── Batch Processing
├── TransformersUtilities
│   ├── Model Information
│   ├── Size Estimation
│   └── Configuration Optimization
└── Enhanced LLMIntegration
    ├── Advanced Text Generation
    ├── Multiple Pooling Strategies
    └── SEO Metrics Extraction
```

## Installation

### Prerequisites

```bash
# Install Transformers library
pip install transformers

# Install additional dependencies
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn
pip install tqdm
```

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd blatam-academy/agents/backend/onyx/server/features/seo

# Install requirements
pip install -r requirements.deep_learning.txt
```

## Quick Start

### Basic Model Loading

```python
from transformers_integration import TransformersModelManager, TransformersConfig

# Create configuration
config = TransformersConfig(
    model_name="bert-base-uncased",
    task_type="sequence_classification",
    max_length=512,
    use_mixed_precision=True
)

# Create manager and load model
manager = TransformersModelManager(config)
model, tokenizer = manager.load_model_and_tokenizer()

print(f"Model loaded: {type(model).__name__}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Advanced Tokenization

```python
# Basic tokenization
tokens = manager.tokenize_text("SEO optimization tips")

# Advanced tokenization with custom options
advanced_tokens = manager.tokenize_text(
    "SEO optimization tips",
    add_special_tokens=True,
    return_attention_mask=True,
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=128,
    return_special_tokens_mask=True
)

print(f"Token count: {len(advanced_tokens['input_ids'][0])}")
```

### Embedding Extraction

```python
# Extract embeddings with different pooling strategies
text = "SEO optimization for better rankings"

# Mean pooling (default)
embeddings_mean = manager.get_embeddings(text, pooling_strategy="mean")

# CLS token pooling
embeddings_cls = manager.get_embeddings(text, pooling_strategy="cls")

# Max pooling
embeddings_max = manager.get_embeddings(text, pooling_strategy="max")

# Attention-based pooling
embeddings_attention = manager.get_embeddings(text, pooling_strategy="attention")

print(f"Embedding shapes: {embeddings_mean.shape}")
```

## Advanced Features

### Model Registry

```python
# Access model registry
registry = manager.model_registry

# Available model types
print(f"Available model types: {list(registry.keys())}")

# BERT models
bert_models = registry['bert']
print(f"BERT sequence classification: {bert_models['sequence_classification']}")
```

### Pipeline Creation

```python
from transformers_integration import PipelineConfig

# Create pipeline configuration
pipeline_config = PipelineConfig(
    task="text-classification",
    model="bert-base-uncased",
    device=0 if torch.cuda.is_available() else -1,
    batch_size=4
)

# Create pipeline
pipeline_obj = manager.create_pipeline(pipeline_config)

# Use pipeline
result = pipeline_obj("This is a positive review")
print(f"Classification result: {result}")
```

### SEO-Specific Analysis

```python
from transformers_integration import SEOSpecificTransformers

# Create SEO transformers
seo_transformers = SEOSpecificTransformers(config)

# Setup for SEO analysis
seo_transformers.setup_seo_model(task="sequence_classification", num_labels=4)

# Analyze SEO content
content = """
This is a comprehensive SEO content about digital marketing strategies.
It includes relevant keywords and provides valuable information to readers.
"""

analysis = seo_transformers.analyze_seo_content(content)
print(f"SEO Analysis: {analysis['metrics']}")
```

## Model Management

### Loading Different Model Types

```python
# Load BERT for sequence classification
bert_config = TransformersConfig(
    model_name="bert-base-uncased",
    task_type="sequence_classification"
)
bert_manager = TransformersModelManager(bert_config)
bert_model, bert_tokenizer = bert_manager.load_model_and_tokenizer()

# Load GPT-2 for text generation
gpt2_config = TransformersConfig(
    model_name="gpt2-medium",
    task_type="text_generation"
)
gpt2_manager = TransformersModelManager(gpt2_config)
gpt2_model, gpt2_tokenizer = gpt2_manager.load_model_and_tokenizer()

# Load T5 for summarization
t5_config = TransformersConfig(
    model_name="t5-small",
    task_type="summarization"
)
t5_manager = TransformersModelManager(t5_config)
t5_model, t5_tokenizer = t5_manager.load_model_and_tokenizer()
```

### Model Information and Size Estimation

```python
from transformers_integration import TransformersUtilities

# Get available models
available_models = TransformersUtilities.get_available_models()
print(f"Available BERT models: {available_models['bert']}")

# Get model information
model_info = TransformersUtilities.get_model_info("bert-base-uncased")
print(f"Model info: {model_info}")

# Estimate model size
size_info = TransformersUtilities.estimate_model_size("bert-base-uncased")
print(f"Model size: {size_info['memory_fp32_gb']:.2f} GB (FP32)")
print(f"Model size: {size_info['memory_fp16_gb']:.2f} GB (FP16)")

# Create optimized configuration
optimized_config = TransformersUtilities.create_optimized_config(
    "bert-base-uncased",
    use_mixed_precision=True,
    gradient_checkpointing=True
)
```

## Tokenization

### Advanced Tokenization Options

```python
# Tokenizer configuration
tokenizer_config = TokenizerConfig(
    model_name="bert-base-uncased",
    use_fast=True,
    add_special_tokens=True,
    return_attention_mask=True,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512,
    return_overflowing_tokens=False,
    return_special_tokens_mask=False,
    return_offsets_mapping=False,
    return_length=False
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.model_name)

# Test different tokenization scenarios
texts = [
    "Short text.",
    "This is a medium length text for testing.",
    "This is a very long text that will be truncated during tokenization to demonstrate the truncation functionality of the tokenizer."
]

for text in texts:
    # Basic tokenization
    tokens = tokenizer.tokenize(text)
    print(f"Text: {text[:50]}... | Tokens: {len(tokens)}")
    
    # Encoding with options
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=tokenizer_config.max_length,
        return_tensors='pt',
        return_special_tokens_mask=True,
        return_attention_mask=True
    )
    
    print(f"Input IDs shape: {encoding['input_ids'].shape}")
```

## Pipeline Creation

### Different Pipeline Types

```python
# Text classification pipeline
classification_pipeline = pipeline(
    task="text-classification",
    model="bert-base-uncased",
    device=-1
)

# Text generation pipeline
generation_pipeline = pipeline(
    task="text-generation",
    model="gpt2",
    device=-1
)

# Summarization pipeline
summarization_pipeline = pipeline(
    task="summarization",
    model="t5-small",
    device=-1
)

# Translation pipeline
translation_pipeline = pipeline(
    task="translation_en_to_fr",
    model="t5-small",
    device=-1
)

# Fill-mask pipeline
fill_mask_pipeline = pipeline(
    task="fill-mask",
    model="bert-base-uncased",
    device=-1
)

# Test pipelines
text = "This is a positive review about the product."
classification_result = classification_pipeline(text)
print(f"Classification: {classification_result}")

generation_result = generation_pipeline("The future of AI", max_length=50)
print(f"Generation: {generation_result}")

summarization_result = summarization_pipeline("This is a long text that needs to be summarized.")
print(f"Summarization: {summarization_result}")

fill_mask_result = fill_mask_pipeline("The [MASK] is bright today.")
print(f"Fill-mask: {fill_mask_result}")
```

## Fine-tuning

### Complete Fine-tuning Workflow

```python
from transformers import TrainingArguments, Trainer
from transformers.data import DataCollatorWithPadding

# Create dataset
class SEOTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Sample data
train_texts = [
    "Excellent SEO content with great optimization.",
    "Poor quality content with no SEO value.",
    "Outstanding digital marketing strategy.",
    "Terrible website design and SEO."
]

train_labels = [1, 0, 1, 0]  # 1 for good, 0 for bad

# Create datasets
train_dataset = SEOTextDataset(train_texts, train_labels, tokenizer)
eval_dataset = SEOTextDataset(train_texts[:2], train_labels[:2], tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    eval_steps=1000,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./fine_tuned_model")
```

## SEO-Specific Features

### Content Analysis

```python
# Create SEO transformers
seo_transformers = SEOSpecificTransformers(config)

# Setup for SEO analysis
seo_transformers.setup_seo_model(task="sequence_classification", num_labels=4)

# Analyze SEO content
content = """
This is a comprehensive SEO content about digital marketing strategies.
It includes relevant keywords and provides valuable information to readers.
The content is well-structured and optimized for search engines.
"""

analysis = seo_transformers.analyze_seo_content(content)

print(f"Content length: {analysis['metrics']['content_length']}")
print(f"Word count: {analysis['metrics']['word_count']}")
print(f"Has keywords: {analysis['metrics']['has_keywords']}")
print(f"Has links: {analysis['metrics']['has_links']}")
print(f"Analysis: {analysis['analysis'][:200]}...")
```

### Content Similarity

```python
# Calculate similarity between content pieces
content1 = "SEO optimization tips for better rankings"
content2 = "Search engine optimization strategies for improved visibility"
content3 = "Cooking recipes for beginners"

similarity_12 = seo_transformers.get_content_similarity(content1, content2)
similarity_13 = seo_transformers.get_content_similarity(content1, content3)

print(f"Similarity between content1 and content2: {similarity_12:.4f}")
print(f"Similarity between content1 and content3: {similarity_13:.4f}")
```

### Batch Processing

```python
# Process multiple texts efficiently
texts = [
    "SEO optimization techniques",
    "Digital marketing strategies",
    "Content creation tips",
    "Search engine ranking factors"
]

# Batch embedding extraction
batch_embeddings = []
for text in texts:
    embeddings = manager.get_embeddings(text)
    batch_embeddings.append(embeddings)

batch_embeddings_tensor = torch.cat(batch_embeddings, dim=0)
print(f"Batch embeddings shape: {batch_embeddings_tensor.shape}")

# Batch text generation
prompts = [
    "Write SEO tips:",
    "Create content strategy:",
    "Optimize website:"
]

generated_texts = llm.batch_generate(prompts, max_length=100)
for prompt, generated in zip(prompts, generated_texts):
    print(f"{prompt} {generated[:50]}...")
```

## Performance Optimization

### Memory Optimization

```python
# Enable mixed precision
config = TransformersConfig(
    model_name="bert-base-uncased",
    use_mixed_precision=True,
    torch_dtype="float16"
)

# Enable gradient checkpointing
config = TransformersConfig(
    model_name="bert-base-uncased",
    gradient_checkpointing=True
)

# Optimize memory usage
config = TransformersConfig(
    model_name="bert-base-uncased",
    use_cache=False,  # Disable KV cache for training
    max_length=512  # Limit sequence length
)
```

### Batch Processing Optimization

```python
# Optimize batch size based on memory
config = TransformersConfig(
    model_name="bert-base-uncased",
    max_length=512
)

# Use data loading optimization
dataloader = DataLoader(
    dataset,
    batch_size=16,  # Adjust based on GPU memory
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### GPU Memory Management

```python
# Clear cache periodically
import torch
torch.cuda.empty_cache()

# Monitor memory usage
from pytorch_configuration import PyTorchConfiguration
pytorch_config = PyTorchConfiguration()
memory_info = pytorch_config.get_memory_info()
print(f"GPU memory: {memory_info}")

# Use gradient accumulation
config = TrainingArguments(
    gradient_accumulation_steps=4,
    per_device_train_batch_size=4
)
```

## Best Practices

### Model Selection

1. **Choose appropriate model size**:
   - Small models (BERT-base, DistilBERT) for quick experiments
   - Medium models (BERT-large, RoBERTa) for production
   - Large models (GPT-2 large, T5-large) for maximum performance

2. **Consider task requirements**:
   - BERT/RoBERTa for understanding tasks
   - GPT-2 for generation tasks
   - T5 for text-to-text tasks

3. **Optimize for memory**:
   - Use mixed precision training
   - Enable gradient checkpointing
   - Adjust batch size based on available memory

### Tokenization Best Practices

1. **Choose appropriate max_length**:
   - 512 for most tasks
   - 1024 for longer documents
   - 2048+ for very long content

2. **Use appropriate padding**:
   - `max_length` for fixed-size batches
   - `longest` for variable-size batches
   - `do_not_pad` for inference

3. **Handle special tokens**:
   - Always set pad_token for generation models
   - Use appropriate special tokens for your task

### Pipeline Best Practices

1. **Choose appropriate device**:
   - Use GPU for large models and batch processing
   - Use CPU for small models and single predictions

2. **Optimize batch size**:
   - Start with batch_size=1 and increase
   - Monitor memory usage
   - Balance speed vs memory

3. **Handle errors gracefully**:
   - Always wrap pipeline calls in try-except
   - Provide fallback options
   - Log errors for debugging

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   ```python
   # Reduce batch size
   config = TransformersConfig(max_length=256)
   
   # Enable mixed precision
   config = TransformersConfig(use_mixed_precision=True)
   
   # Use gradient checkpointing
   config = TransformersConfig(gradient_checkpointing=True)
   ```

2. **Slow Tokenization**:
   ```python
   # Use fast tokenizer
   tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
   
   # Batch tokenization
   tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
   ```

3. **Model Loading Errors**:
   ```python
   # Check model availability
   from transformers import AutoConfig
   config = AutoConfig.from_pretrained(model_name)
   
   # Use local files only
   config = TransformersConfig(local_files_only=True)
   ```

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model information
model_info = manager.get_model_info()
print(f"Model info: {model_info}")

# Check tokenizer information
tokenizer_info = {
    'vocab_size': tokenizer.vocab_size,
    'model_max_length': tokenizer.model_max_length,
    'pad_token': tokenizer.pad_token,
    'eos_token': tokenizer.eos_token
}
print(f"Tokenizer info: {tokenizer_info}")

# Monitor memory usage
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## API Reference

### TransformersConfig

```python
@dataclass
class TransformersConfig:
    model_name: str = "bert-base-uncased"
    model_type: str = "bert"
    task_type: str = "sequence_classification"
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    return_tensors: str = "pt"
    device: str = "auto"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    use_auth_token: Optional[str] = None
    revision: Optional[str] = None
    mirror: Optional[str] = None
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    torch_dtype: Optional[str] = None
```

### TokenizerConfig

```python
@dataclass
class TokenizerConfig:
    model_name: str = "bert-base-uncased"
    use_fast: bool = True
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    return_tensors: str = "pt"
    padding: str = "max_length"
    truncation: bool = True
    max_length: int = 512
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False
```

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    task: str = "text-classification"
    model: str = "bert-base-uncased"
    tokenizer: Optional[str] = None
    device: int = -1
    batch_size: int = 1
    top_k: int = 5
    temperature: float = 1.0
    do_sample: bool = False
    max_length: int = 50
    num_return_sequences: int = 1
    return_all_scores: bool = False
    function_to_apply: Optional[str] = None
```

### TransformersModelManager

```python
class TransformersModelManager:
    def __init__(self, config: TransformersConfig)
    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]
    def create_pipeline(self, pipeline_config: PipelineConfig) -> Pipeline
    def tokenize_text(self, text: Union[str, List[str]], **kwargs) -> BatchEncoding
    def encode_text(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]
    def predict(self, text: Union[str, List[str]], **kwargs) -> Any
    def get_embeddings(self, text: Union[str, List[str]], pooling_strategy: str = "mean") -> torch.Tensor
    def fine_tune(self, train_dataset, eval_dataset=None, training_args: TrainingArguments = None) -> Trainer
    def save_model(self, save_path: str)
    def load_saved_model(self, load_path: str)
```

### SEOSpecificTransformers

```python
class SEOSpecificTransformers:
    def __init__(self, config: TransformersConfig)
    def setup_seo_model(self, task: str = "sequence_classification", num_labels: int = 2)
    def analyze_seo_content(self, content: str) -> Dict[str, Any]
    def batch_analyze_seo_content(self, contents: List[str]) -> List[Dict[str, Any]]
    def generate_seo_content(self, prompt: str, max_length: int = 200) -> str
    def get_content_similarity(self, content1: str, content2: str) -> float
    def create_seo_pipeline(self, task: str = "text-classification") -> Pipeline
```

### TransformersUtilities

```python
class TransformersUtilities:
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]
    @staticmethod
    def estimate_model_size(model_name: str) -> Dict[str, Any]
    @staticmethod
    def create_optimized_config(model_name: str, **kwargs) -> TransformersConfig
```

## File Structure

```
transformers_integration/
├── transformers_integration.py          # Main integration module
├── example_transformers_integration.py  # Usage examples
├── README_TRANSFORMERS_INTEGRATION.md   # This documentation
└── tests/
    └── test_transformers_integration.py # Unit tests
```

## Contributing

1. Follow the existing code style and conventions
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure all tests pass before submitting

## License

This module is part of the Blatam Academy SEO Service and follows the same licensing terms.

---

For more information, see the main [README_DEEP_LEARNING.md](README_DEEP_LEARNING.md) file. 