# 🔤 Advanced Tokenization & Sequence Handling System

A comprehensive implementation of advanced tokenization and sequence handling for text data processing, with support for multiple models, intelligent sequence processing, and comprehensive text preprocessing.

## 🎯 Features

### **Advanced Tokenization**
- **Multi-Model Support**: GPT-2, BERT, T5, and custom tokenizers
- **Fast Tokenizer Integration**: Support for both fast and slow tokenizers
- **Special Token Management**: Automatic setup of BOS, EOS, SEP, CLS tokens
- **Metadata Tracking**: Comprehensive tokenization statistics and analysis
- **Batch Processing**: Efficient batch tokenization with memory optimization

### **Intelligent Sequence Processing**
- **Flexible Padding Strategies**: Longest, max_length, and dynamic padding
- **Smart Truncation**: Multiple truncation strategies for different use cases
- **Sliding Window**: Handle long sequences with overlapping windows
- **Attention Mask Generation**: Automatic attention mask creation
- **Token Type IDs**: Support for sequence pair tasks

### **Comprehensive Text Preprocessing**
- **Text Cleaning**: Remove URLs, emails, phone numbers, normalize whitespace
- **Text Normalization**: Lowercase, accent removal, Unicode normalization
- **Intelligent Chunking**: Split long texts with overlapping chunks
- **Data Augmentation**: Random masking, insertion, and token swapping
- **Multi-language Support**: Basic support for various languages

### **Advanced Data Handling**
- **Data Collators**: Factory for different task types (LM, Seq2Seq, Classification)
- **Batch Processing**: Memory-efficient batch processing
- **Dataset Preparation**: Ready-to-use batches for training
- **Performance Optimization**: Benchmarking and memory profiling

## 📁 File Structure

```
advanced_tokenization_system.py    # Main system implementation
tokenization_demo.py               # Comprehensive demonstration script
requirements_tokenization.txt      # Dependencies
ADVANCED_TOKENIZATION_README.md   # This documentation
```

## 🛠️ Installation

### Basic Installation

```bash
# Install core requirements
pip install torch transformers numpy

# Install all requirements
pip install -r requirements_tokenization.txt
```

### Advanced Installation (Optional)

```bash
# For GPU acceleration
pip install cupy-cuda11x  # Replace with your CUDA version

# For distributed processing
pip install dask ray

# For advanced text analysis
pip install spacy nltk textblob
```

## 🚀 Quick Start

### 1. Basic Tokenization

```python
from advanced_tokenization_system import TokenizationConfig, AdvancedTokenizer

# Configuration
config = TokenizationConfig(
    model_name="gpt2",
    max_length=512,
    padding="longest",
    return_attention_mask=True
)

# Create tokenizer
tokenizer = AdvancedTokenizer(config)

# Tokenize text
result = tokenizer.tokenize_with_metadata("Hello, world!")
print(f"Tokens: {len(result['input_ids'][0])}")
```

### 2. Sequence Processing

```python
from advanced_tokenization_system import SequenceConfig, SequenceProcessor

# Configuration
config = SequenceConfig(
    max_sequence_length=1024,
    target_sequence_length=512,
    padding_strategy="longest",
    truncation_strategy="longest_first"
)

# Create processor
processor = SequenceProcessor(config)

# Process sequences
sequences = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4, 5])]
processed = processor.process_sequences(sequences)
print(f"Processed shape: {processed.shape}")
```

### 3. Text Preprocessing

```python
from advanced_tokenization_system import TextPreprocessor

# Create preprocessor
preprocessor = TextPreprocessor()

# Clean and normalize text
cleaned = preprocessor.clean_text(
    "Check out https://example.com and contact info@example.com",
    remove_urls=True,
    remove_emails=True
)

normalized = preprocessor.normalize_text(cleaned, lowercase=True)
print(f"Cleaned: {cleaned}")
```

### 4. Complete Pipeline

```python
from advanced_tokenization_system import (
    TokenizationConfig, SequenceConfig, AdvancedTextProcessor
)

# Configuration
tokenization_config = TokenizationConfig(
    model_name="gpt2",
    max_length=512,
    return_attention_mask=True
)

sequence_config = SequenceConfig(
    target_sequence_length=512,
    handle_long_sequences=True
)

# Create processor
processor = AdvancedTextProcessor(tokenization_config, sequence_config)

# Process text
result = processor.process_text(
    "Your text here",
    clean_text=True,
    normalize_text=True
)

# Create dataset-ready batch
texts = ["Text 1", "Text 2", "Text 3"]
batch = processor.create_dataset_ready_batch(texts)
print(f"Batch shape: {batch['input_ids'].shape}")
```

## 📊 Configuration Options

### Tokenization Configuration

```python
@dataclass
class TokenizationConfig:
    model_name: str = "gpt2"                    # Model to use
    tokenizer_type: str = "auto"                # auto, bert, gpt2, t5
    use_fast_tokenizer: bool = True             # Use fast tokenizer
    max_length: int = 512                       # Maximum sequence length
    padding: str = "longest"                    # Padding strategy
    return_attention_mask: bool = True          # Return attention masks
    return_token_type_ids: bool = True          # Return token type IDs
    add_special_tokens: bool = True             # Add special tokens
```

### Sequence Configuration

```python
@dataclass
class SequenceConfig:
    max_sequence_length: int = 1024             # Maximum sequence length
    target_sequence_length: int = 512           # Target sequence length
    padding_strategy: str = "longest"           # Padding strategy
    truncation_strategy: str = "longest_first"  # Truncation strategy
    handle_long_sequences: bool = True          # Handle long sequences
    sliding_window: bool = False                # Use sliding window
    window_size: int = 512                      # Window size
    window_stride: int = 256                    # Window stride
```

## 🎯 Advanced Usage

### Custom Tokenizer Integration

```python
from transformers import AutoTokenizer
from advanced_tokenization_system import AdvancedTokenizer, TokenizationConfig

# Load custom tokenizer
custom_tokenizer = AutoTokenizer.from_pretrained("your-model")

# Create configuration
config = TokenizationConfig(
    model_name="custom",
    tokenizer_type="custom"
)

# Create advanced tokenizer
advanced_tokenizer = AdvancedTokenizer(config)
advanced_tokenizer.tokenizer = custom_tokenizer
```

### Sliding Window for Long Sequences

```python
from advanced_tokenization_system import SequenceProcessor, SequenceConfig

config = SequenceConfig(
    sliding_window=True,
    window_size=512,
    window_stride=256
)

processor = SequenceProcessor(config)

# Long sequence
long_sequence = torch.randn(1000, 768)
windows = processor.create_sliding_windows(long_sequence)
print(f"Created {len(windows)} windows")
```

### Text Augmentation

```python
from advanced_tokenization_system import TextPreprocessor

preprocessor = TextPreprocessor()

# Create augmentations
augmented = preprocessor.create_text_augmentations(
    "This is a sample text.",
    methods=["random_mask", "random_insert", "random_swap"]
)

for i, text in enumerate(augmented):
    print(f"Augmentation {i+1}: {text}")
```

### Data Collators for Different Tasks

```python
from advanced_tokenization_system import DataCollatorFactory

# Language modeling
lm_collator = DataCollatorFactory.create_collator(
    "language_modeling",
    tokenizer=tokenizer,
    mlm=False
)

# Sequence classification
cls_collator = DataCollatorFactory.create_collator(
    "sequence_classification",
    tokenizer=tokenizer
)

# Token classification
tok_collator = DataCollatorFactory.create_collator(
    "token_classification",
    tokenizer=tokenizer
)
```

## 🔬 Demonstration

Run the comprehensive demonstration:

```bash
python tokenization_demo.py
```

This demonstrates:
- Basic tokenization capabilities
- Advanced sequence processing
- Text preprocessing features
- Batch processing
- Data collators
- Performance benchmarking
- Statistics and analysis

## 📈 Performance Optimization

### Memory Optimization
- Use `batch_size` parameter for memory-efficient processing
- Enable `sliding_window` for long sequences
- Use appropriate `padding_strategy`
- Monitor memory usage with built-in profiling

### Speed Optimization
- Use `use_fast_tokenizer=True` when possible
- Enable `verbose=False` for production
- Use appropriate batch sizes
- Profile performance with built-in benchmarking

### Example Optimization

```python
# Optimized configuration
tokenization_config = TokenizationConfig(
    model_name="gpt2",
    max_length=512,
    verbose=False,  # Disable logging for speed
    use_fast_tokenizer=True
)

sequence_config = SequenceConfig(
    target_sequence_length=512,
    handle_long_sequences=True,
    sliding_window=True,
    window_size=512,
    window_stride=256
)

# Process in batches
processor = AdvancedTextProcessor(tokenization_config, sequence_config)
results = processor.process_batch(texts, batch_size=16)
```

## 🧪 Testing and Validation

### Unit Tests

```python
import pytest
from advanced_tokenization_system import AdvancedTokenizer, TokenizationConfig

def test_tokenizer_initialization():
    config = TokenizationConfig(model_name="gpt2")
    tokenizer = AdvancedTokenizer(config)
    assert tokenizer.tokenizer is not None

def test_text_tokenization():
    config = TokenizationConfig(model_name="gpt2")
    tokenizer = AdvancedTokenizer(config)
    result = tokenizer.tokenize_text("Hello world")
    assert 'input_ids' in result
    assert len(result['input_ids'][0]) > 0
```

### Benchmarking

```python
from advanced_tokenization_system import AdvancedTextProcessor
import time

# Performance test
processor = AdvancedTextProcessor(tokenization_config, sequence_config)
texts = ["Sample text " * 100] * 100

start_time = time.time()
results = processor.process_batch(texts, batch_size=32)
end_time = time.time()

print(f"Processed {len(texts)} texts in {end_time - start_time:.2f}s")
print(f"Speed: {len(texts) / (end_time - start_time):.1f} texts/second")
```

## 🤝 Integration with Transformers

### Hugging Face Integration

```python
from transformers import AutoTokenizer, AutoModel
from advanced_tokenization_system import AdvancedTextProcessor

# Load model and tokenizer
model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create processor
processor = AdvancedTextProcessor(tokenization_config, sequence_config)

# Process text
result = processor.process_text("Your text here")
input_ids = result['processed_input_ids']

# Use with model
with torch.no_grad():
    outputs = model(input_ids)
```

### Custom Model Integration

```python
class CustomModelWithTokenization(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AdvancedTextProcessor(
            TokenizationConfig(model_name=model_name),
            SequenceConfig()
        )
    
    def forward(self, texts: List[str]):
        # Process texts
        processed = self.processor.create_dataset_ready_batch(texts)
        
        # Forward pass
        return self.model(**processed)
```

## 📚 Language Support

### Multi-language Tokenization

```python
# Chinese
from advanced_tokenization_system import TextPreprocessor
preprocessor = TextPreprocessor()

# Basic Chinese support
chinese_text = "人工智能正在改变世界"
cleaned = preprocessor.clean_text(chinese_text)

# Japanese
japanese_text = "人工知能は世界を変えています"
cleaned = preprocessor.clean_text(japanese_text)

# Korean
korean_text = "인공지능이 세상을 바꾸고 있습니다"
cleaned = preprocessor.clean_text(korean_text)
```

### Domain-Specific Processing

```python
# Medical text
medical_config = TokenizationConfig(
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
)

# Legal text
legal_config = TokenizationConfig(
    model_name="nlpaueb/legal-bert-base-uncased"
)

# Financial text
financial_config = TokenizationConfig(
    model_name="ProsusAI/finbert"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Tokenizer Loading Errors**
   ```python
   # Solution: Check model name and internet connection
   config = TokenizationConfig(model_name="gpt2")
   tokenizer = AdvancedTokenizer(config)
   ```

2. **Memory Issues with Long Sequences**
   ```python
   # Solution: Enable sliding window
   sequence_config = SequenceConfig(
       sliding_window=True,
       window_size=512,
       window_stride=256
   )
   ```

3. **Special Token Errors**
   ```python
   # Solution: Ensure pad token is set
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token
   ```

### Debug Mode

```python
# Enable verbose logging
config = TokenizationConfig(verbose=True)

# Check tokenizer state
print(f"Vocabulary size: {tokenizer.tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.tokenizer.special_tokens_map}")

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
```

## 📄 License

This implementation is provided under the MIT License. See the main project license for details.

## 🙏 Acknowledgments

- Hugging Face team for the excellent Transformers library
- PyTorch team for the deep learning framework
- OpenAI for GPT models and tokenization
- Google Research for BERT and T5
- Microsoft Research for various language models

---

**Ready to implement advanced tokenization and sequence handling! 🔤**






