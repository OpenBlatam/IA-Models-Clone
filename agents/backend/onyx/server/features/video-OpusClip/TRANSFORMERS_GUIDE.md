# Transformers Guide for Video-OpusClip

Complete guide to using the Hugging Face Transformers library in your Video-OpusClip system for advanced AI capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Core Components](#core-components)
4. [Model Loading & Usage](#model-loading--usage)
5. [Text Processing](#text-processing)
6. [Video Caption Generation](#video-caption-generation)
7. [Optimization Techniques](#optimization-techniques)
8. [Integration with Video-OpusClip](#integration-with-video-opusclip)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [Examples](#examples)

## Overview

The Transformers library provides state-of-the-art natural language processing models that are essential for your Video-OpusClip system. It enables:

- **Text Generation**: Creating viral captions and descriptions
- **Text Classification**: Analyzing video content and sentiment
- **Translation**: Multi-language support for global audiences
- **Summarization**: Creating concise video descriptions
- **Question Answering**: Interactive video content analysis
- **Named Entity Recognition**: Extracting key information from videos

## Installation & Setup

### Current Dependencies

Your Video-OpusClip system already includes Transformers in the requirements:

```txt
# From requirements_complete.txt
transformers>=4.30.0
tokenizers>=0.13.0
sentencepiece>=0.1.99
accelerate>=0.20.0
peft>=0.4.0
bitsandbytes>=0.41.0
```

### Installation Commands

```bash
# Install basic Transformers
pip install transformers

# Install with all optimizations
pip install transformers[torch] accelerate

# Install for production
pip install transformers[torch,accelerate,peft] bitsandbytes

# Install from your requirements
pip install -r requirements_complete.txt
```

### Verify Installation

```python
import transformers
print(f"Transformers version: {transformers.__version__}")

# Test basic functionality
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("✅ Transformers installation successful!")
```

## Core Components

### 1. Tokenizers

```python
from transformers import AutoTokenizer, GPT2Tokenizer, T5Tokenizer

# Auto tokenizer (recommended)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Specific tokenizers
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
```

### 2. Models

```python
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM, T5ForConditionalGeneration, GPT2LMHeadModel
)

# Auto models (recommended)
model = AutoModel.from_pretrained("gpt2")
causal_model = AutoModelForCausalLM.from_pretrained("gpt2")
classifier = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
```

### 3. Pipelines

```python
from transformers import pipeline

# Text generation
generator = pipeline("text-generation", model="gpt2")

# Sentiment analysis
classifier = pipeline("sentiment-analysis")

# Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
```

## Model Loading & Usage

### Basic Model Loading

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str, device: str = "auto"):
    """Load a transformer model with optimization."""
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer

# Usage
model, tokenizer = load_model("gpt2")
```

### Optimized Model Loading

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class OptimizedModelLoader:
    """Optimized model loading with caching and memory management."""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = cache_dir
        self.loaded_models = {}
    
    def load_model(
        self,
        model_name: str,
        model_type: str = "causal",
        device: str = "auto",
        use_8bit: bool = False,
        use_4bit: bool = False
    ):
        """Load model with advanced optimizations."""
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check cache
        cache_key = f"{model_name}_{model_type}_{device}_{use_8bit}_{use_4bit}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimizations
        if model_type == "causal":
            model_class = AutoModelForCausalLM
        elif model_type == "classification":
            model_class = AutoModelForSequenceClassification
        else:
            model_class = AutoModel
        
        # Load with quantization if requested
        if use_8bit and device == "cuda":
            model = model_class.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                cache_dir=self.cache_dir
            )
        elif use_4bit and device == "cuda":
            model = model_class.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto",
                cache_dir=self.cache_dir
            )
        else:
            model = model_class.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
                cache_dir=self.cache_dir
            )
        
        # Cache the result
        self.loaded_models[cache_key] = (model, tokenizer)
        
        return model, tokenizer

# Usage
loader = OptimizedModelLoader()
model, tokenizer = loader.load_model("gpt2", use_8bit=True)
```

## Text Processing

### Text Generation

```python
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    num_return_sequences: int = 1
):
    """Generate text using transformer model."""
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode outputs
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

# Usage
prompt = "Generate a viral caption for a funny cat video:"
captions = generate_text(model, tokenizer, prompt, max_length=50)
```

### Text Classification

```python
from transformers import pipeline

def analyze_sentiment(texts: List[str]):
    """Analyze sentiment of text using transformers."""
    
    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    
    results = classifier(texts)
    return results

# Usage
texts = [
    "This video is absolutely amazing!",
    "I don't like this content at all.",
    "It's okay, nothing special."
]
sentiments = analyze_sentiment(texts)
```

### Text Summarization

```python
def summarize_text(text: str, max_length: int = 150):
    """Summarize text using transformers."""
    
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    
    summary = summarizer(text, max_length=max_length, min_length=30)
    return summary[0]['summary_text']

# Usage
long_text = "Your long video description here..."
summary = summarize_text(long_text)
```

## Video Caption Generation

### Caption Generation Pipeline

```python
class VideoCaptionGenerator:
    """Generate captions for videos using transformers."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model, self.tokenizer = load_model(model_name)
        
    def generate_viral_caption(
        self,
        video_description: str,
        platform: str = "tiktok",
        style: str = "funny",
        max_length: int = 50
    ):
        """Generate viral caption for video."""
        
        # Create platform-specific prompts
        prompts = {
            "tiktok": f"Create a viral TikTok caption for: {video_description}",
            "youtube": f"Write an engaging YouTube title for: {video_description}",
            "instagram": f"Generate an Instagram caption for: {video_description}"
        }
        
        prompt = prompts.get(platform, f"Generate a caption for: {video_description}")
        
        # Add style modifier
        if style == "funny":
            prompt += " Make it funny and engaging."
        elif style == "dramatic":
            prompt += " Make it dramatic and attention-grabbing."
        elif style == "informative":
            prompt += " Make it informative and educational."
        
        # Generate caption
        captions = generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_length=max_length,
            temperature=0.8
        )
        
        return captions[0] if captions else ""
    
    def generate_multiple_captions(
        self,
        video_description: str,
        num_captions: int = 5,
        styles: List[str] = ["funny", "dramatic", "informative"]
    ):
        """Generate multiple caption variations."""
        
        captions = []
        for style in styles:
            for _ in range(num_captions // len(styles)):
                caption = self.generate_viral_caption(
                    video_description,
                    style=style
                )
                captions.append({
                    "text": caption,
                    "style": style
                })
        
        return captions

# Usage
generator = VideoCaptionGenerator()
caption = generator.generate_viral_caption(
    "A cat playing with a laser pointer",
    platform="tiktok",
    style="funny"
)
```

### Multi-Language Support

```python
def translate_caption(
    caption: str,
    target_language: str = "es",
    source_language: str = "en"
):
    """Translate caption to target language."""
    
    translator = pipeline(
        "translation",
        model=f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"
    )
    
    translation = translator(caption)
    return translation[0]['translation_text']

# Usage
english_caption = "This cat is absolutely hilarious!"
spanish_caption = translate_caption(english_caption, "es")
```

## Optimization Techniques

### Memory Optimization

```python
def optimize_model_memory(model, tokenizer):
    """Optimize model for memory efficiency."""
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Use half precision
    if torch.cuda.is_available():
        model = model.half()
    
    # Enable attention slicing
    if hasattr(model, 'enable_attention_slicing'):
        model.enable_attention_slicing()
    
    return model

# Usage
model, tokenizer = load_model("gpt2")
model = optimize_model_memory(model, tokenizer)
```

### Batch Processing

```python
def batch_generate_captions(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 4
):
    """Generate captions in batches for efficiency."""
    
    all_captions = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode batch
        batch_captions = []
        for output in outputs:
            caption = tokenizer.decode(output, skip_special_tokens=True)
            batch_captions.append(caption)
        
        all_captions.extend(batch_captions)
    
    return all_captions

# Usage
prompts = [
    "Generate caption for cat video",
    "Generate caption for dog video",
    "Generate caption for bird video",
    "Generate caption for fish video"
]
captions = batch_generate_captions(model, tokenizer, prompts)
```

### Caching

```python
from functools import lru_cache
import hashlib

class CachedCaptionGenerator:
    """Caption generator with caching for performance."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model, self.tokenizer = load_model(model_name)
        self.cache = {}
    
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for prompt and parameters."""
        key_data = f"{prompt}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def generate_cached_caption(self, prompt: str, max_length: int = 50):
        """Generate caption with caching."""
        
        cache_key = self._get_cache_key(prompt, max_length=max_length)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate new caption
        captions = generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_length=max_length
        )
        
        caption = captions[0] if captions else ""
        self.cache[cache_key] = caption
        
        return caption

# Usage
cached_generator = CachedCaptionGenerator()
caption = cached_generator.generate_cached_caption("Funny cat video")
```

## Integration with Video-OpusClip

### Integration with Existing Components

```python
from optimized_libraries import OptimizedTextProcessor
from enhanced_error_handling import safe_load_ai_model, safe_model_inference

class VideoOpusClipTransformers:
    """Transformers integration for Video-OpusClip system."""
    
    def __init__(self):
        self.text_processor = OptimizedTextProcessor()
        self.models = {}
    
    def load_model_safely(self, model_name: str):
        """Safely load transformer model with error handling."""
        
        try:
            model_data = safe_load_ai_model(model_name)
            self.models[model_name] = model_data
            return model_data
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def generate_caption_safely(
        self,
        video_description: str,
        model_name: str = "gpt2",
        max_length: int = 50
    ):
        """Safely generate caption with error handling."""
        
        if model_name not in self.models:
            self.load_model_safely(model_name)
        
        model_data = self.models[model_name]
        
        try:
            result = safe_model_inference(
                model_data,
                f"Generate viral caption: {video_description}",
                max_length=max_length
            )
            return result
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "Amazing video! 🔥"
    
    def analyze_video_content(self, video_description: str):
        """Analyze video content using transformers."""
        
        # Sentiment analysis
        sentiment = self.text_processor.generate_caption(video_description)
        
        # Content classification
        classifier = pipeline("text-classification")
        classification = classifier(video_description)
        
        return {
            "sentiment": sentiment,
            "classification": classification,
            "description": video_description
        }

# Usage
video_opus = VideoOpusClipTransformers()
caption = video_opus.generate_caption_safely("Cat playing with laser")
```

### API Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class CaptionRequest(BaseModel):
    video_description: str
    platform: str = "tiktok"
    style: str = "funny"
    max_length: int = 50

class CaptionResponse(BaseModel):
    caption: str
    style: str
    platform: str

@app.post("/generate-caption", response_model=CaptionResponse)
async def generate_caption(request: CaptionRequest):
    """Generate caption via API."""
    
    try:
        generator = VideoCaptionGenerator()
        caption = generator.generate_viral_caption(
            request.video_description,
            platform=request.platform,
            style=request.style,
            max_length=request.max_length
        )
        
        return CaptionResponse(
            caption=caption,
            style=request.style,
            platform=request.platform
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Usage
# POST /generate-caption
# {
#   "video_description": "Funny cat video",
#   "platform": "tiktok",
#   "style": "funny"
# }
```

## Advanced Features

### Fine-tuning Support

```python
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset

class CaptionDataset(Dataset):
    """Dataset for caption fine-tuning."""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

def fine_tune_model(
    model,
    tokenizer,
    training_texts: List[str],
    output_dir: str = "./fine_tuned_model"
):
    """Fine-tune model on custom data."""
    
    # Prepare dataset
    dataset = CaptionDataset(training_texts, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        warmup_steps=500,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

# Usage
training_texts = [
    "Amazing video! 🔥",
    "This is incredible! 😍",
    "Can't stop watching! 😂"
]
fine_tune_model(model, tokenizer, training_texts)
```

### Model Quantization

```python
def quantize_model(model, tokenizer, quantization_type: str = "8bit"):
    """Quantize model for memory efficiency."""
    
    if quantization_type == "8bit":
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model.name_or_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
    
    elif quantization_type == "4bit":
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model.name_or_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
    
    return model, tokenizer

# Usage
model, tokenizer = quantize_model(model, tokenizer, "8bit")
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```python
   # Solution: Use quantization
   model, tokenizer = quantize_model(model, tokenizer, "8bit")
   
   # Or use gradient checkpointing
   model.gradient_checkpointing_enable()
   ```

2. **Model Loading Errors**
   ```python
   # Solution: Clear cache and retry
   from transformers import clear_cache
   clear_cache()
   
   # Or use specific model revision
   model = AutoModel.from_pretrained("model_name", revision="main")
   ```

3. **Tokenization Errors**
   ```python
   # Solution: Add padding token
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token
   ```

4. **Generation Quality Issues**
   ```python
   # Solution: Adjust generation parameters
   outputs = model.generate(
       **inputs,
       temperature=0.7,  # Lower for more focused
       top_p=0.9,       # Nucleus sampling
       repetition_penalty=1.2,  # Avoid repetition
       do_sample=True
   )
   ```

### Performance Optimization

```python
# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model.generate(**inputs)

# Batch processing
def process_batch(texts, batch_size=4):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Process batch
        results.extend(process_single_batch(batch))
    return results
```

## Examples

### Complete Caption Generation System

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict
import logging

class CompleteCaptionSystem:
    """Complete caption generation system for Video-OpusClip."""
    
    def __init__(self):
        self.models = {}
        self.pipelines = {}
        self.setup_models()
    
    def setup_models(self):
        """Setup all required models."""
        
        # Text generation model
        self.models['gpt2'], self.tokenizer = load_model("gpt2")
        
        # Sentiment analysis
        self.pipelines['sentiment'] = pipeline("sentiment-analysis")
        
        # Translation
        self.pipelines['translation'] = pipeline("translation")
        
        # Summarization
        self.pipelines['summarization'] = pipeline("summarization")
    
    def generate_complete_caption(
        self,
        video_description: str,
        target_language: str = "en",
        platforms: List[str] = ["tiktok", "youtube", "instagram"]
    ) -> Dict[str, str]:
        """Generate captions for multiple platforms and languages."""
        
        results = {}
        
        for platform in platforms:
            # Generate base caption
            base_caption = self.generate_viral_caption(
                video_description,
                platform=platform
            )
            
            # Translate if needed
            if target_language != "en":
                translated_caption = self.translate_caption(
                    base_caption,
                    target_language
                )
                results[f"{platform}_{target_language}"] = translated_caption
            else:
                results[platform] = base_caption
        
        return results
    
    def generate_viral_caption(self, description: str, platform: str) -> str:
        """Generate platform-specific viral caption."""
        
        prompts = {
            "tiktok": f"Create a viral TikTok caption: {description}",
            "youtube": f"Write engaging YouTube title: {description}",
            "instagram": f"Generate Instagram caption: {description}"
        }
        
        prompt = prompts.get(platform, f"Generate caption: {description}")
        
        captions = generate_text(
            self.models['gpt2'],
            self.tokenizer,
            prompt,
            max_length=50,
            temperature=0.8
        )
        
        return captions[0] if captions else ""
    
    def translate_caption(self, caption: str, target_language: str) -> str:
        """Translate caption to target language."""
        
        try:
            translation = self.pipelines['translation'](
                caption,
                model=f"Helsinki-NLP/opus-mt-en-{target_language}"
            )
            return translation[0]['translation_text']
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            return caption

# Usage
system = CompleteCaptionSystem()
captions = system.generate_complete_caption(
    "Funny cat playing with laser pointer",
    target_language="es",
    platforms=["tiktok", "youtube"]
)
```

This comprehensive guide covers all aspects of using Transformers in your Video-OpusClip system. The library provides powerful capabilities for text generation, analysis, and processing that are essential for creating viral video content. 