"""
Concise Technical Examples - Deep Learning & NLP
Accurate Python implementations with best practices
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Model Loading with Mixed Precision
def load_optimized_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

# 2. Efficient Text Generation
def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. Custom Dataset with Tokenization
class OptimizedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }

# 4. Training Loop with Mixed Precision
def train_with_amp(model, dataloader, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            with autocast():
                outputs = model(**inputs)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

# 5. Sentiment Analysis Pipeline
def setup_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1
    )

# 6. Batch Processing
def batch_generate(model, tokenizer, prompts, max_length=100):
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            do_sample=True,
            temperature=0.7
        )
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# 7. Model Evaluation
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
            num_batches += 1
    
    return total_loss / num_batches

# 8. Gradient Clipping
def train_with_clipping(model, dataloader, max_grad_norm=1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

# 9. Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

# 10. Performance Profiling
def profile_model(model, dataloader, num_batches=10):
    from torch.profiler import profile, record_function, ProfilerActivity
    
    model.eval()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                inputs = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    _ = model(**inputs)
    
    prof.export_chrome_trace("model_trace.json")

# Usage Examples
if __name__ == "__main__":
    # Load model
    model, tokenizer = load_optimized_model()
    
    # Generate text
    result = generate_text(model, tokenizer, "The future of AI is")
    print(f"Generated: {result}")
    
    # Setup sentiment analysis
    sentiment_pipeline = setup_sentiment_pipeline()
    sentiment = sentiment_pipeline("I love this technology!")
    print(f"Sentiment: {sentiment}")
    
    # Batch generation
    prompts = ["AI is", "Machine learning", "Deep learning"]
    results = batch_generate(model, tokenizer, prompts)
    print(f"Batch results: {results}")





