# Transformers Integration Guide

## Overview

This guide covers the integration of Hugging Face Transformers with the Email Sequence AI System, providing powerful NLP capabilities for email generation, analysis, and optimization.

## 🚀 Quick Start

### Installation

Transformers is already installed in the system. Verify installation:

```bash
python -c "import transformers; print(transformers.__version__)"
```

### Basic Usage

```python
from transformers import pipeline

# Text generation for emails
generator = pipeline("text-generation", model="gpt2")
email_content = generator("Dear valued customer,", max_length=100)

# Sentiment analysis
classifier = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
sentiment = classifier("We're excited to introduce our new product!")

# Summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summary = summarizer(long_email_text)
```

## 📧 Email Sequence Generation

### Text Generation Models

| Model | Size | Use Case | Performance |
|-------|------|----------|-------------|
| `gpt2` | 548MB | General email content | Fast, good quality |
| `gpt2-medium` | 1.5GB | Higher quality emails | Slower, better quality |
| `gpt2-large` | 3.1GB | Premium content | Slowest, best quality |

### Example: Generate Email Sequence

```python
from transformers import pipeline

def generate_email_sequence(topic, audience, num_emails=5):
    generator = pipeline("text-generation", model="gpt2")
    
    sequence = []
    prompts = [
        f"Write a professional introduction email about {topic} for {audience}:",
        f"Write a follow-up email explaining the benefits of {topic} for {audience}:",
        f"Write a call-to-action email for {topic} targeting {audience}:",
        f"Write a testimonial email about {topic} for {audience}:",
        f"Write a final follow-up email for {topic} targeting {audience}:"
    ]
    
    for i, prompt in enumerate(prompts[:num_emails], 1):
        result = generator(prompt, max_length=150, do_sample=True, temperature=0.7)
        content = result[0]['generated_text'].replace(prompt, "").strip()
        
        sequence.append({
            "email_number": i,
            "subject": f"Email {i}: {topic}",
            "content": content,
            "type": ["introduction", "benefits", "cta", "testimonial", "followup"][i-1]
        })
    
    return sequence
```

## 😊 Sentiment Analysis

### Available Models

| Model | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | 91% | Fast | General sentiment |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 94% | Medium | Social media style |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 89% | Medium | Multilingual |

### Example: Analyze Email Sentiment

```python
def analyze_email_sentiment(email_content):
    classifier = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    
    result = classifier(email_content)
    return {
        "sentiment": result[0]['label'],
        "confidence": result[0]['score'],
        "content": email_content
    }

# Usage
emails = [
    "We're excited to introduce our new product!",
    "We're sorry to inform you about the delay.",
    "Thank you for your feedback."
]

for email in emails:
    analysis = analyze_email_sentiment(email)
    print(f"Email: {email}")
    print(f"Sentiment: {analysis['sentiment']} ({analysis['confidence']:.2f})")
```

## 📝 Text Summarization

### Available Models

| Model | Max Length | Quality | Speed |
|-------|------------|---------|-------|
| `sshleifer/distilbart-cnn-12-6` | 1024 | Good | Fast |
| `facebook/bart-large-cnn` | 1024 | Excellent | Medium |
| `google/pegasus-xsum` | 512 | Very Good | Fast |

### Example: Summarize Long Emails

```python
def summarize_email(email_content, max_length=100):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    result = summarizer(email_content, max_length=max_length, min_length=30)
    return result[0]['summary_text']

# Usage
long_email = """
Dear valued customer,
[Long email content here...]
Best regards,
Team
"""

summary = summarize_email(long_email)
print(f"Original: {len(long_email)} characters")
print(f"Summary: {len(summary)} characters")
print(f"Compression: {len(summary)/len(long_email)*100:.1f}%")
```

## 🏷️ Email Categorization

### Zero-Shot Classification

```python
def categorize_email(email_content, categories):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    result = classifier(email_content, candidate_labels=categories)
    return {
        "top_category": result['labels'][0],
        "confidence": result['scores'][0],
        "all_categories": dict(zip(result['labels'], result['scores']))
    }

# Usage
categories = ["sales", "marketing", "support", "newsletter", "promotional"]
email = "We're excited to announce our new product launch!"

categorization = categorize_email(email, categories)
print(f"Category: {categorization['top_category']}")
print(f"Confidence: {categorization['confidence']:.2f}")
```

## 🔄 Translation Support

### Multilingual Email Support

```python
def translate_email(email_content, target_language="Spanish"):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    
    result = translator(email_content)
    return result[0]['translation_text']

# Usage
english_email = "Dear customer, thank you for your interest in our product."
spanish_email = translate_email(english_email)
print(f"English: {english_email}")
print(f"Spanish: {spanish_email}")
```

## ⚡ Performance Optimization

### Memory Management

```python
import torch
from transformers import pipeline

# Use CPU for smaller models
generator = pipeline("text-generation", model="gpt2", device="cpu")

# Use GPU if available
if torch.cuda.is_available():
    generator = pipeline("text-generation", model="gpt2", device="cuda")

# Batch processing for multiple emails
def batch_generate_emails(prompts, batch_size=4):
    generator = pipeline("text-generation", model="gpt2")
    
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_results = generator(batch, max_length=100)
        results.extend(batch_results)
    
    return results
```

### Model Caching

```python
# Models are automatically cached in ~/.cache/huggingface/
# To clear cache:
import shutil
shutil.rmtree("~/.cache/huggingface/")

# To use a different cache directory:
import os
os.environ["TRANSFORMERS_CACHE"] = "/path/to/cache"
```

## 🎯 Advanced Features

### Custom Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_email(email_content):
    tokens = tokenizer.encode(email_content, return_tensors="pt")
    return {
        "tokens": tokens,
        "token_count": len(tokens[0]),
        "max_length": tokenizer.model_max_length
    }
```

### Model Fine-tuning Preparation

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

def prepare_for_fine_tuning(model_name="gpt2"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
```

## 🔧 Integration with Email Sequence AI System

### Using with EmailSequenceTransformers Class

```python
from examples.transformers_examples import EmailSequenceTransformers

# Initialize
email_transformer = EmailSequenceTransformers()

# Load models
email_transformer.load_text_generation_model()
email_transformer.load_classification_model()
email_transformer.load_summarization_model()

# Generate sequence
sequence = email_transformer.generate_email_sequence(
    topic="AI Email Marketing Platform",
    target_audience="small business owners"
)

# Analyze sentiment
for email in sequence:
    sentiment = email_transformer.analyze_email_sentiment(email['content'])
    print(f"Email {email['email_number']}: {sentiment['sentiment']}")
```

### Integration with Gradio Interface

```python
import gradio as gr
from transformers import pipeline

def create_transformers_interface():
    generator = pipeline("text-generation", model="gpt2")
    classifier = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    
    def generate_and_analyze(prompt):
        # Generate content
        generated = generator(prompt, max_length=100)[0]['generated_text']
        
        # Analyze sentiment
        sentiment = classifier(generated)[0]
        
        return generated, f"{sentiment['label']} ({sentiment['score']:.2f})"
    
    interface = gr.Interface(
        fn=generate_and_analyze,
        inputs=gr.Textbox(label="Email Prompt"),
        outputs=[
            gr.Textbox(label="Generated Email"),
            gr.Textbox(label="Sentiment")
        ],
        title="Email Generation with Sentiment Analysis"
    )
    
    return interface
```

## 📊 Model Comparison

### Performance Metrics

| Task | Model | Accuracy | Speed | Memory | Best For |
|------|-------|----------|-------|--------|----------|
| Text Generation | GPT-2 | 85% | Fast | 548MB | General emails |
| Text Generation | GPT-2 Medium | 90% | Medium | 1.5GB | Quality emails |
| Sentiment | DistilBERT | 91% | Fast | 268MB | Quick analysis |
| Sentiment | RoBERTa | 94% | Medium | 500MB | High accuracy |
| Summarization | DistilBART | 88% | Fast | 1.2GB | Quick summaries |
| Summarization | BART Large | 92% | Slow | 1.6GB | Quality summaries |

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```python
   # Use smaller models
   generator = pipeline("text-generation", model="gpt2", device="cpu")
   
   # Clear cache
   import torch
   torch.cuda.empty_cache()
   ```

2. **Model Download Issues**
   ```python
   # Set cache directory
   import os
   os.environ["TRANSFORMERS_CACHE"] = "/path/with/space"
   
   # Use offline mode if models are cached
   generator = pipeline("text-generation", model="gpt2", local_files_only=True)
   ```

3. **Slow Performance**
   ```python
   # Use GPU if available
   device = "cuda" if torch.cuda.is_available() else "cpu"
   generator = pipeline("text-generation", model="gpt2", device=device)
   
   # Use smaller models
   generator = pipeline("text-generation", model="distilgpt2")
   ```

### Error Handling

```python
def safe_transformers_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Transformers error: {e}")
        return None

# Usage
result = safe_transformers_call(
    pipeline("text-generation", model="gpt2"),
    "Hello world",
    max_length=50
)
```

## 📈 Best Practices

1. **Model Selection**
   - Use smaller models for development and testing
   - Use larger models for production quality
   - Consider model size vs. performance trade-offs

2. **Memory Management**
   - Clear GPU cache regularly
   - Use CPU for smaller models
   - Batch process when possible

3. **Error Handling**
   - Always wrap Transformers calls in try-except
   - Provide fallback options
   - Log errors for debugging

4. **Performance Optimization**
   - Cache models when possible
   - Use appropriate batch sizes
   - Monitor memory usage

## 🔗 Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Model Hub](https://huggingface.co/models)
- [Pipeline Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

## 🎯 Next Steps

1. **Fine-tune Models**: Train models on your specific email data
2. **Custom Pipelines**: Create specialized pipelines for email tasks
3. **Performance Tuning**: Optimize for your specific use case
4. **Integration**: Connect with the full Email Sequence AI System
5. **Monitoring**: Add performance monitoring and logging

---

*This guide covers the essential Transformers integration for the Email Sequence AI System. For advanced usage, refer to the official Transformers documentation.* 