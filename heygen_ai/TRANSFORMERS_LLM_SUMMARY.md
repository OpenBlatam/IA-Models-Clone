# Transformers and LLM Implementation Summary for HeyGen AI

## Overview
Comprehensive implementation of transformer architectures and Large Language Models (LLMs) for deep learning, providing state-of-the-art transformer models, advanced attention mechanisms, and complete LLM utilities for text generation and processing.

## Core Components

### 1. **Transformer Architectures** (`transformers.py`)

#### Multi-Head Attention
- **MultiHeadAttention**: Advanced attention mechanism with relative position encoding
- **Attention Features**: Support for causal masking, padding masks, and custom attention patterns
- **Relative Position Encoding**: Optional relative position bias for better sequence modeling

#### Transformer Blocks
- **TransformerBlock**: Complete transformer block with self-attention and feed-forward network
- **Cross-Attention Support**: Optional cross-attention for encoder-decoder architectures
- **Advanced Features**: Layer normalization, dropout, and flexible activation functions

#### Positional Encoding
- **PositionalEncoding**: Sinusoidal positional encoding for sequence modeling
- **Configurable**: Support for custom sequence lengths and dropout

#### Complete Models
- **TransformerModel**: Full encoder-decoder transformer for sequence-to-sequence tasks
- **GPTModel**: Autoregressive language model for text generation
- **BERTModel**: Bidirectional encoder for masked language modeling

#### Model Features
```python
# Multi-head attention
attention = MultiHeadAttention(
    embedding_dimensions=768,
    num_attention_heads=12,
    dropout_probability=0.1,
    use_relative_position=True,
    max_relative_position=512
)

# Transformer block
transformer_block = TransformerBlock(
    embedding_dimensions=768,
    num_attention_heads=12,
    feed_forward_dimensions=3072,
    dropout_probability=0.1,
    use_cross_attention=True,
    activation_function="gelu"
)

# Complete transformer model
transformer = TransformerModel(
    vocabulary_size=50000,
    embedding_dimensions=768,
    num_attention_heads=12,
    num_encoder_layers=12,
    num_decoder_layers=12,
    feed_forward_dimensions=3072,
    use_relative_position=True
)

# GPT model
gpt = GPTModel(
    vocabulary_size=50000,
    embedding_dimensions=768,
    num_attention_heads=12,
    num_layers=12,
    feed_forward_dimensions=3072
)

# BERT model
bert = BERTModel(
    vocabulary_size=50000,
    embedding_dimensions=768,
    num_attention_heads=12,
    num_layers=12,
    feed_forward_dimensions=3072
)
```

### 2. **LLM Utilities** (`llm_utils.py`)

#### Tokenization
- **Tokenizer**: Complete tokenization pipeline with vocabulary management
- **Special Tokens**: Support for PAD, UNK, BOS, EOS, and MASK tokens
- **Batch Processing**: Efficient batch encoding and decoding

#### Text Generation
- **TextGenerator**: Advanced text generation with multiple sampling strategies
- **GenerationConfig**: Comprehensive configuration for generation parameters
- **Sampling Methods**: Greedy, beam search, top-k, top-p, and temperature sampling

#### LLM Inference
- **LLMInference**: Complete inference pipeline for LLMs
- **Text Generation**: High-level text generation interface
- **Embedding Extraction**: Get embeddings from text inputs
- **Perplexity Calculation**: Model evaluation using perplexity

#### LLM Training
- **LLMTraining**: Training utilities for LLM models
- **Data Preparation**: Automatic training data preparation with sliding windows
- **Training Loop**: Complete training pipeline with loss computation
- **Evaluation**: Model evaluation with perplexity metrics

#### Utility Features
```python
# Tokenizer
tokenizer = Tokenizer(
    vocabulary=vocabulary,
    special_tokens={"pad": "<PAD>", "unk": "<UNK>", "bos": "<BOS>", "eos": "<EOS>"}
)

# Text generation
generation_config = GenerationConfig(
    max_length=100,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    num_beams=1
)

generator = TextGenerator(model, tokenizer)
generated_text = generator.generate(prompt, generation_config)

# LLM inference
llm_inference = LLMInference(model, tokenizer)
generated_text = llm_inference.generate_text(
    prompt="Hello world",
    max_length=50,
    temperature=0.8,
    top_k=20
)

embeddings = llm_inference.get_embeddings(["Hello world", "Machine learning"])
perplexity = llm_inference.compute_perplexity(["Test text for evaluation"])

# LLM training
llm_training = LLMTraining(model, tokenizer)
training_data = llm_training.prepare_training_data(texts, max_length=512, stride=128)
loss = llm_training.train_step(batch, optimizer)
metrics = llm_training.evaluate(evaluation_texts)
```

### 3. **Examples and Demonstrations** (`transformer_examples.py`)

#### Basic Examples
- **TransformerExamples**: Basic transformer model demonstrations
- **LLMExamples**: LLM usage examples and utilities
- **AdvancedTransformerExamples**: Advanced transformer applications

#### Example Implementations
```python
# Basic transformer example
transformer_model, outputs = TransformerExamples.basic_transformer_example()

# GPT model example
gpt_model, outputs = TransformerExamples.gpt_model_example()

# BERT model example
bert_model, outputs = TransformerExamples.bert_model_example()

# Attention visualization
attention_model, weights = TransformerExamples.attention_visualization_example()

# Tokenizer example
tokenizer, batch_data = LLMExamples.tokenizer_example()

# Text generation
generator, text = LLMExamples.text_generation_example()

# LLM inference
inference, text, embeddings = LLMExamples.llm_inference_example()

# LLM training
training, metrics = LLMExamples.llm_training_example()

# Multi-task transformer
multi_task_model, trans_outputs, sum_outputs = AdvancedTransformerExamples.multi_task_transformer_example()

# Attention analysis
attention_analysis, weights, entropy = AdvancedTransformerExamples.attention_analysis_example()

# Transformer benchmark
benchmark_results = AdvancedTransformerExamples.transformer_benchmark_example()
```

## Advanced Features

### 1. **Multi-Head Attention**

#### Advanced Attention Mechanism
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimensions, num_attention_heads, dropout_probability=0.1,
                 bias=True, use_relative_position=False, max_relative_position=512):
        super().__init__()
        self.embedding_dimensions = embedding_dimensions
        self.num_attention_heads = num_attention_heads
        self.head_dimensions = embedding_dimensions // num_attention_heads
        self.scaling_factor = self.head_dimensions ** -0.5
        
        # Linear projections
        self.query_projection = nn.Linear(embedding_dimensions, embedding_dimensions, bias=bias)
        self.key_projection = nn.Linear(embedding_dimensions, embedding_dimensions, bias=bias)
        self.value_projection = nn.Linear(embedding_dimensions, embedding_dimensions, bias=bias)
        self.output_projection = nn.Linear(embedding_dimensions, embedding_dimensions, bias=bias)
        
        # Relative position encoding
        if use_relative_position:
            self.relative_position_embeddings = nn.Parameter(
                torch.randn(2 * max_relative_position + 1, self.head_dimensions)
            )
            self.relative_position_bias = nn.Parameter(
                torch.randn(2 * max_relative_position + 1, num_attention_heads)
            )

    def forward(self, query, key, value, attention_mask=None, key_padding_mask=None, causal_mask=False):
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling_factor
        
        # Add relative position bias if enabled
        if self.use_relative_position:
            relative_position_bias = self._compute_relative_position_bias(query.shape[1], key.shape[1])
            attention_scores = attention_scores + relative_position_bias
        
        # Apply attention masks
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        if causal_mask:
            causal_mask = self._create_causal_mask(query.shape[1], key.shape[1], query.device)
            attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply softmax and compute output
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
```

### 2. **Transformer Block**

#### Complete Transformer Block
```python
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimensions, num_attention_heads, feed_forward_dimensions,
                 dropout_probability=0.1, layer_norm_eps=1e-6, use_relative_position=False,
                 use_cross_attention=False, activation_function="gelu"):
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            dropout_probability=dropout_probability,
            use_relative_position=use_relative_position
        )
        
        # Cross-attention (if enabled)
        if use_cross_attention:
            self.cross_attention = MultiHeadAttention(
                embedding_dimensions=embedding_dimensions,
                num_attention_heads=num_attention_heads,
                dropout_probability=dropout_probability,
                use_relative_position=use_relative_position
            )
            self.cross_attention_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        
        # Layer normalization
        self.self_attention_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        self.feed_forward_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimensions, feed_forward_dimensions),
            self._get_activation_function(activation_function),
            nn.Dropout(dropout_probability),
            nn.Linear(feed_forward_dimensions, embedding_dimensions),
            nn.Dropout(dropout_probability)
        )

    def forward(self, input_tensor, cross_attention_input=None, attention_mask=None,
                cross_attention_mask=None, causal_mask=False):
        # Self-attention
        residual = input_tensor
        input_tensor = self.self_attention_layer_norm(input_tensor)
        attention_output, _ = self.self_attention(
            query=input_tensor, key=input_tensor, value=input_tensor,
            attention_mask=attention_mask, causal_mask=causal_mask
        )
        input_tensor = residual + attention_output
        
        # Cross-attention (if enabled)
        if self.use_cross_attention and cross_attention_input is not None:
            residual = input_tensor
            input_tensor = self.cross_attention_layer_norm(input_tensor)
            cross_attention_output, _ = self.cross_attention(
                query=input_tensor, key=cross_attention_input, value=cross_attention_input,
                attention_mask=cross_attention_mask
            )
            input_tensor = residual + cross_attention_output
        
        # Feed-forward network
        residual = input_tensor
        input_tensor = self.feed_forward_layer_norm(input_tensor)
        feed_forward_output = self.feed_forward(input_tensor)
        input_tensor = residual + feed_forward_output
        
        return input_tensor
```

### 3. **GPT Model**

#### Autoregressive Language Model
```python
class GPTModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimensions, num_attention_heads, num_layers,
                 feed_forward_dimensions, max_sequence_length=2048, dropout_probability=0.1,
                 layer_norm_eps=1e-6, use_relative_position=False, activation_function="gelu",
                 pad_token_id=0):
        super().__init__()
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocabulary_size, embedding_dimensions, padding_idx=pad_token_id)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dimensions=embedding_dimensions,
            max_sequence_length=max_sequence_length,
            dropout_probability=dropout_probability
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_dimensions=embedding_dimensions,
                num_attention_heads=num_attention_heads,
                feed_forward_dimensions=feed_forward_dimensions,
                dropout_probability=dropout_probability,
                layer_norm_eps=layer_norm_eps,
                use_relative_position=use_relative_position,
                activation_function=activation_function
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization and output projection
        self.final_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        self.output_projection = nn.Linear(embedding_dimensions, vocabulary_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        # Token embeddings
        embeddings = self.token_embeddings(input_ids)
        embeddings = self.positional_encoding(embeddings)
        
        # Apply transformer layers
        for layer in self.layers:
            embeddings = layer(
                input_tensor=embeddings,
                attention_mask=attention_mask,
                causal_mask=True
            )
        
        # Final layer normalization and output projection
        embeddings = self.final_layer_norm(embeddings)
        logits = self.output_projection(embeddings)
        
        return logits

    def generate(self, input_ids, max_length, temperature=1.0, top_k=None, top_p=1.0,
                do_sample=True, pad_token_id=None):
        # Autoregressive generation
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        output = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # Get logits for next token
                logits = self.forward(output)[:, -1, :]
                
                if do_sample:
                    # Apply temperature and sampling
                    logits = logits / temperature
                    
                    if top_k is not None:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                        logits = torch.full_like(logits, float('-inf'))
                        logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append next token
                output = torch.cat([output, next_token], dim=-1)
                
                # Check if all sequences are complete
                if pad_token_id is not None and (next_token == pad_token_id).all():
                    break
        
        return output
```

### 4. **BERT Model**

#### Bidirectional Encoder
```python
class BERTModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimensions, num_attention_heads, num_layers,
                 feed_forward_dimensions, max_sequence_length=512, dropout_probability=0.1,
                 layer_norm_eps=1e-6, use_relative_position=False, activation_function="gelu",
                 pad_token_id=0, mask_token_id=103, type_vocabulary_size=2):
        super().__init__()
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocabulary_size, embedding_dimensions, padding_idx=pad_token_id)
        
        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(type_vocabulary_size, embedding_dimensions)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dimensions=embedding_dimensions,
            max_sequence_length=max_sequence_length,
            dropout_probability=dropout_probability
        )
        
        # Layer normalization for embeddings
        self.embedding_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        self.embedding_dropout = nn.Dropout(dropout_probability)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_dimensions=embedding_dimensions,
                num_attention_heads=num_attention_heads,
                feed_forward_dimensions=feed_forward_dimensions,
                dropout_probability=dropout_probability,
                layer_norm_eps=layer_norm_eps,
                use_relative_position=use_relative_position,
                activation_function=activation_function
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization and MLM head
        self.final_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        self.mlm_head = nn.Sequential(
            nn.Linear(embedding_dimensions, embedding_dimensions),
            nn.GELU(),
            nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps),
            nn.Linear(embedding_dimensions, vocabulary_size)
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Token embeddings
        token_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + token_type_embeddings
        embeddings = self.positional_encoding(embeddings)
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Apply transformer layers
        for layer in self.layers:
            embeddings = layer(
                input_tensor=embeddings,
                attention_mask=attention_mask,
                causal_mask=False
            )
        
        # Final layer normalization
        sequence_output = self.final_layer_norm(embeddings)
        
        # MLM predictions
        mlm_logits = self.mlm_head(sequence_output)
        
        outputs = {
            "sequence_output": sequence_output,
            "mlm_logits": mlm_logits
        }
        
        # Compute MLM loss if labels are provided
        if masked_lm_labels is not None:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.vocabulary_size),
                masked_lm_labels.view(-1),
                ignore_index=-100
            )
            outputs["mlm_loss"] = mlm_loss
        
        return outputs
```

## Complete Usage Example

```python
# Create vocabulary and tokenizer
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning algorithms are powerful tools",
    "Natural language processing is fascinating"
]

from .llm_utils import create_simple_vocabulary, Tokenizer
vocabulary = create_simple_vocabulary(texts, min_frequency=1)
tokenizer = Tokenizer(vocabulary)

# Create GPT model
from .transformers import create_transformer_model
model = create_transformer_model(
    model_type="gpt",
    vocabulary_size=len(vocabulary),
    embedding_dimensions=256,
    num_attention_heads=8,
    num_layers=6,
    feed_forward_dimensions=1024
)

# Create LLM inference
from .llm_utils import LLMInference
llm_inference = LLMInference(model, tokenizer)

# Generate text
generated_text = llm_inference.generate_text(
    prompt="The quick brown",
    max_length=50,
    temperature=0.8,
    top_k=20,
    top_p=0.9
)

print(f"Generated text: {generated_text}")

# Get embeddings
embeddings = llm_inference.get_embeddings(["Hello world", "Machine learning"])
print(f"Embeddings shape: {embeddings.shape}")

# Compute perplexity
perplexity = llm_inference.compute_perplexity(["Test text for evaluation"])
print(f"Perplexity: {perplexity:.4f}")
```

## Key Benefits

### 1. **Comprehensive Transformer Models**
- **Multi-Head Attention**: Advanced attention with relative position encoding
- **Transformer Blocks**: Complete blocks with cross-attention support
- **Positional Encoding**: Sinusoidal encoding for sequence modeling
- **Multiple Architectures**: Transformer, GPT, and BERT implementations

### 2. **Advanced LLM Utilities**
- **Tokenization**: Complete tokenization pipeline with vocabulary management
- **Text Generation**: Multiple sampling strategies and generation methods
- **Inference Pipeline**: High-level inference interface with embeddings and evaluation
- **Training Utilities**: Complete training pipeline with data preparation

### 3. **Production-Ready Features**
- **Flexible Configuration**: Extensive parameter customization
- **Error Handling**: Robust error checking and validation
- **Performance Optimization**: Efficient implementations with GPU support
- **Comprehensive Examples**: Complete examples for all components

### 4. **Research and Development**
- **Attention Analysis**: Tools for analyzing attention patterns
- **Model Benchmarking**: Performance comparison utilities
- **Multi-Task Support**: Support for various NLP tasks
- **Extensible Architecture**: Easy to extend and customize

The transformers and LLM implementation provides a comprehensive framework for building and using state-of-the-art transformer models and Large Language Models, with extensive utilities for text generation, processing, and analysis. 