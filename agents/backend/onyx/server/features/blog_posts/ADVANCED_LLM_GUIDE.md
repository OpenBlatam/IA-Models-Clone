# Advanced LLM Integration Guide

## Overview

This guide covers advanced LLM integration with modern PyTorch practices, transformers, quantization, and production-ready features. It provides comprehensive implementations for training, fine-tuning, and deploying large language models.

## Key Features

### 1. Modern LLM Training
- **Latest Transformers**: Up-to-date model architectures and APIs
- **Quantization**: 4-bit and 8-bit quantization for memory efficiency
- **PEFT Support**: Parameter-efficient fine-tuning with LoRA
- **Flash Attention**: Memory-efficient attention mechanisms
- **Gradient Checkpointing**: Memory-efficient training

### 2. Multiple Model Types
- **Causal Models**: GPT-2, LLaMA, Mistral for text generation
- **Sequence Classification**: BERT for classification tasks
- **Conditional Generation**: T5 for summarization and translation
- **Custom Architectures**: Support for custom model types

### 3. Production Features
- **Model Compilation**: torch.compile for performance optimization
- **Distributed Training**: Multi-GPU training support
- **Model Serialization**: Safe model saving and loading
- **Pipeline Integration**: Production-ready inference pipelines
- **Error Handling**: Comprehensive error handling and recovery

## LLM Configuration

### Basic Configuration

```python
from advanced_llm_integration import LLMConfig, AdvancedLLMTrainer

# Basic configuration for GPT-2
config = LLMConfig(
    model_name="gpt2",
    model_type="causal",
    task="text_generation",
    max_length=512,
    batch_size=4,
    learning_rate=5e-5,
    num_epochs=3,
    use_peft=True,
    quantization="4bit"
)

trainer = AdvancedLLMTrainer(config)
```

### Advanced Configuration

```python
# Advanced configuration with all optimizations
config = LLMConfig(
    model_name="microsoft/DialoGPT-medium",
    model_type="causal",
    task="text_generation",
    max_length=1024,
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=5,
    warmup_steps=500,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=True,
    bf16=False,
    use_peft=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    quantization="4bit",
    use_flash_attention=True,
    use_gradient_checkpointing=True,
    max_grad_norm=1.0,
    save_steps=500,
    eval_steps=500,
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    evaluation_strategy="steps",
    save_strategy="steps",
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to=None
)
```

## Model Types and Architectures

### 1. Causal Language Models

```python
# GPT-2 Configuration
gpt2_config = LLMConfig(
    model_name="gpt2",
    model_type="causal",
    task="text_generation",
    max_length=512,
    batch_size=4,
    learning_rate=5e-5,
    use_peft=True,
    quantization="4bit"
)

# LLaMA Configuration
llama_config = LLMConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    model_type="causal",
    task="text_generation",
    max_length=2048,
    batch_size=2,
    learning_rate=1e-4,
    use_peft=True,
    quantization="4bit",
    use_flash_attention=True
)

# Mistral Configuration
mistral_config = LLMConfig(
    model_name="mistralai/Mistral-7B-v0.1",
    model_type="causal",
    task="text_generation",
    max_length=2048,
    batch_size=2,
    learning_rate=1e-4,
    use_peft=True,
    quantization="4bit",
    use_flash_attention=True
)
```

### 2. Sequence Classification Models

```python
# BERT Configuration
bert_config = LLMConfig(
    model_name="bert-base-uncased",
    model_type="sequence_classification",
    task="classification",
    max_length=512,
    batch_size=16,
    learning_rate=2e-5,
    use_peft=True,
    quantization="4bit"
)

# RoBERTa Configuration
roberta_config = LLMConfig(
    model_name="roberta-base",
    model_type="sequence_classification",
    task="classification",
    max_length=512,
    batch_size=16,
    learning_rate=1e-5,
    use_peft=True,
    quantization="4bit"
)
```

### 3. Conditional Generation Models

```python
# T5 Configuration
t5_config = LLMConfig(
    model_name="t5-small",
    model_type="conditional_generation",
    task="summarization",
    max_length=512,
    batch_size=8,
    learning_rate=1e-4,
    use_peft=True,
    quantization="4bit"
)

# BART Configuration
bart_config = LLMConfig(
    model_name="facebook/bart-base",
    model_type="conditional_generation",
    task="summarization",
    max_length=1024,
    batch_size=4,
    learning_rate=3e-5,
    use_peft=True,
    quantization="4bit"
)
```

## Training and Fine-tuning

### 1. Basic Training

```python
from advanced_llm_integration import AdvancedLLMTrainer

# Initialize trainer
config = LLMConfig(
    model_name="gpt2",
    model_type="causal",
    max_length=128,
    batch_size=4,
    num_epochs=3,
    use_peft=True,
    quantization="4bit"
)

trainer = AdvancedLLMTrainer(config)

# Prepare training data
train_texts = [
    "This is a positive example for training.",
    "This is a negative example for training.",
    "I love this product and would recommend it.",
    "I hate this product and would not recommend it."
]

train_labels = [1, 0, 1, 0]

# Train model
trainer_result = trainer.train(train_texts, train_labels)

# Save model
trainer.save_model("./trained_model")
```

### 2. Advanced Training with Validation

```python
# Training with validation data
train_texts = [
    "This is a positive example.",
    "This is a negative example.",
    "I love this product.",
    "I hate this product."
]

train_labels = [1, 0, 1, 0]

val_texts = [
    "This is a test positive example.",
    "This is a test negative example."
]

val_labels = [1, 0]

# Train with validation
trainer_result = trainer.train(
    train_texts, train_labels,
    val_texts, val_labels
)
```

### 3. PEFT Fine-tuning

```python
# PEFT configuration for efficient fine-tuning
config = LLMConfig(
    model_name="gpt2",
    model_type="causal",
    use_peft=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    quantization="4bit"
)

trainer = AdvancedLLMTrainer(config)

# Train with PEFT
trainer_result = trainer.train(train_texts, train_labels)

# Save PEFT model
trainer.save_model("./peft_model")
```

### 4. Quantized Training

```python
# 4-bit quantized training
config = LLMConfig(
    model_name="gpt2",
    model_type="causal",
    quantization="4bit",
    use_peft=True,
    use_gradient_checkpointing=True
)

trainer = AdvancedLLMTrainer(config)

# Train quantized model
trainer_result = trainer.train(train_texts, train_labels)
```

## Text Generation

### 1. Basic Text Generation

```python
# Generate text with trained model
generated_text = trainer.generate_text(
    prompt="Hello, how are you?",
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(f"Generated: {generated_text}")
```

### 2. Advanced Generation Parameters

```python
# Advanced generation with custom parameters
generated_text = trainer.generate_text(
    prompt="The future of AI is",
    max_length=200,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    length_penalty=1.0,
    early_stopping=True
)
```

### 3. Batch Generation

```python
# Generate text for multiple prompts
prompts = [
    "The future of AI is",
    "Machine learning can",
    "Deep learning models"
]

predictions = trainer.predict(prompts)

for prompt, prediction in zip(prompts, predictions):
    print(f"Prompt: {prompt}")
    print(f"Generated: {prediction['generated_text']}")
    print()
```

## Production Pipeline

### 1. Basic Pipeline

```python
from advanced_llm_integration import LLMPipeline

# Create production pipeline
pipeline = LLMPipeline("./trained_model", config)

# Generate text
result = pipeline.generate("Hello, how are you?", max_length=50)
print(f"Generated: {result}")
```

### 2. Batch Processing

```python
# Batch generation
prompts = [
    "Hello, how are you?",
    "What is machine learning?",
    "Explain deep learning"
]

results = pipeline.batch_generate(
    prompts,
    max_length=100,
    temperature=0.7
)

for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Result: {result}")
    print()
```

### 3. Classification Pipeline

```python
# Classification pipeline
config = LLMConfig(
    model_name="bert-base-uncased",
    model_type="sequence_classification",
    use_peft=True,
    quantization="4bit"
)

pipeline = LLMPipeline("./classification_model", config)

# Classify texts
texts = [
    "I love this product!",
    "This is terrible quality.",
    "It's okay, nothing special."
]

results = pipeline.classify(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Probabilities: {result['probabilities']}")
    print()
```

## Model Optimization

### 1. Memory Optimization

```python
# Memory-efficient configuration
config = LLMConfig(
    model_name="gpt2",
    model_type="causal",
    use_gradient_checkpointing=True,
    use_flash_attention=True,
    quantization="4bit",
    batch_size=1,
    gradient_accumulation_steps=8
)

trainer = AdvancedLLMTrainer(config)
```

### 2. Performance Optimization

```python
# Performance-optimized configuration
config = LLMConfig(
    model_name="gpt2",
    model_type="causal",
    fp16=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    use_flash_attention=True
)

trainer = AdvancedLLMTrainer(config)
```

### 3. Model Compilation

```python
# The trainer automatically applies torch.compile if available
config = LLMConfig(
    model_name="gpt2",
    model_type="causal",
    use_peft=False,  # Compilation works better without PEFT
    quantization="none"
)

trainer = AdvancedLLMTrainer(config)
# Model is automatically compiled with torch.compile
```

## Error Handling and Monitoring

### 1. Comprehensive Error Handling

```python
try:
    trainer = AdvancedLLMTrainer(config)
    trainer_result = trainer.train(train_texts, train_labels)
except Exception as e:
    logger.error(f"Training failed: {e}")
    # Handle error appropriately
```

### 2. Model Loading Error Handling

```python
try:
    pipeline = LLMPipeline("./model_path", config)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Fallback to default model or handle error
```

### 3. Generation Error Handling

```python
def safe_generate(pipeline, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return pipeline.generate(prompt)
        except Exception as e:
            logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return "Generation failed"
```

## Integration Examples

### 1. Integration with Experiment Tracking

```python
from experiment_tracking import experiment_tracking

with experiment_tracking("llm_experiment", config) as tracker:
    # Train model
    trainer = AdvancedLLMTrainer(config)
    trainer_result = trainer.train(train_texts, train_labels)
    
    # Log metrics
    tracker.log_metrics({
        "training_loss": trainer_result.state.log_history[-1]["loss"],
        "learning_rate": trainer_result.state.log_history[-1]["learning_rate"]
    })
    
    # Test generation
    generated = trainer.generate_text("Test prompt")
    tracker.log_text("generated_text", generated)
```

### 2. Integration with Version Control

```python
from version_control import version_control

with version_control("llm_project", auto_commit=True) as vc:
    # Train model
    trainer = AdvancedLLMTrainer(config)
    trainer_result = trainer.train(train_texts, train_labels)
    
    # Save model
    trainer.save_model("./models/llm_model")
    
    # Version model
    model_version = vc.version_model(
        model_name="llm_model",
        model_path="./models/llm_model",
        metadata={
            "architecture": config.model_name,
            "task": config.task,
            "training_loss": trainer_result.state.log_history[-1]["loss"]
        },
        description="Trained LLM for text generation"
    )
```

### 3. Integration with Gradio Interface

```python
import gradio as gr
from advanced_llm_integration import LLMPipeline

def create_llm_interface():
    # Load pipeline
    pipeline = LLMPipeline("./trained_model", config)
    
    def generate_text(prompt, max_length, temperature):
        return pipeline.generate(
            prompt,
            max_length=int(max_length),
            temperature=float(temperature)
        )
    
    # Create interface
    interface = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your prompt..."),
            gr.Slider(minimum=10, maximum=500, value=100, label="Max Length"),
            gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature")
        ],
        outputs=gr.Textbox(label="Generated Text"),
        title="Advanced LLM Text Generation"
    )
    
    return interface

# Launch interface
interface = create_llm_interface()
interface.launch()
```

## Best Practices

### 1. Model Selection

```python
# Choose appropriate model size
models = {
    "small": "gpt2",                    # 124M parameters
    "medium": "microsoft/DialoGPT-medium",  # 345M parameters
    "large": "gpt2-large",              # 774M parameters
    "xl": "gpt2-xl"                     # 1.5B parameters
}

# Consider task requirements
task_models = {
    "text_generation": "gpt2",
    "classification": "bert-base-uncased",
    "summarization": "t5-small",
    "translation": "t5-base"
}
```

### 2. Training Configuration

```python
# Optimal learning rates
learning_rates = {
    "gpt2": 5e-5,
    "bert": 2e-5,
    "t5": 1e-4,
    "llama": 1e-4
}

# Batch sizes for different GPU memory
batch_sizes = {
    "8GB": 1,
    "16GB": 4,
    "24GB": 8,
    "40GB": 16
}
```

### 3. PEFT Configuration

```python
# LoRA configurations
lora_configs = {
    "small": {"r": 8, "alpha": 16, "dropout": 0.1},
    "medium": {"r": 16, "alpha": 32, "dropout": 0.1},
    "large": {"r": 32, "alpha": 64, "dropout": 0.1}
}
```

## Performance Monitoring

### 1. Training Performance

```python
def monitor_training_performance(trainer):
    # Monitor GPU usage
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    # Monitor training metrics
    for log in trainer.state.log_history:
        print(f"Step {log['step']}: Loss = {log['loss']:.4f}")
```

### 2. Generation Performance

```python
def benchmark_generation(pipeline, prompts, num_runs=10):
    import time
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        results = pipeline.batch_generate(prompts)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"Average generation time: {avg_time:.2f} seconds")
    return avg_time
```

## Conclusion

This guide provides comprehensive coverage of advanced LLM integration with modern PyTorch practices. Key takeaways include:

- **Model Selection**: Choose appropriate models based on task and resources
- **PEFT Integration**: Use parameter-efficient fine-tuning for large models
- **Quantization**: Apply quantization for memory efficiency
- **Production Pipeline**: Create robust production-ready pipelines
- **Performance Optimization**: Apply modern optimizations for better performance
- **Error Handling**: Implement comprehensive error handling and monitoring
- **Integration**: Seamlessly integrate with experiment tracking and version control

These practices ensure production-ready, efficient, and maintainable LLM systems. 