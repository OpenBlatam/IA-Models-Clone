from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
from transformers import (
from transformers.utils import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import logging as python_logging
from pathlib import Path
import json
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Pre-trained Models and Tokenizers Implementation
Using Transformers library for efficient model loading and tokenization.
"""

    AutoTokenizer, AutoModel, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForQuestionAnswering, AutoModelForMaskedLM,
    AutoConfig, AutoFeatureExtractor, AutoProcessor,
    pipeline, PreTrainedTokenizer, PreTrainedModel,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup, set_seed
)

# Configure logging
logging.set_verbosity_info()
logger = python_logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for pre-trained models."""
    model_name: str = "bert-base-uncased"
    model_type: str = "bert"  # bert, gpt2, t5, roberta, etc.
    task_type: str = "classification"  # classification, generation, qa, token_classification
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    device: str = "auto"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    local_files_only: bool = False

class PreTrainedModelManager:
    """Manages pre-trained models and tokenizers from HuggingFace."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.config_model = None
        self._initialize_components()
    
    def _initialize_components(self) -> Any:
        """Initialize tokenizer, model, and configuration."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=self.config.use_fast_tokenizer,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token = "[PAD]"
        
        # Load model configuration
        self.config_model = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only
        )
        
        # Load model based on task type
        self._load_model_for_task()
        
        # Move model to device
        self.model.to(self.device)
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_model_for_task(self) -> Any:
        """Load appropriate model for the specified task."""
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "cache_dir": self.config.cache_dir,
            "local_files_only": self.config.local_files_only
        }
        
        if self.config.task_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                **model_kwargs
            )
        elif self.config.task_type == "generation":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                **model_kwargs
            )
        elif self.config.task_type == "qa":
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                **model_kwargs
            )
        elif self.config.task_type == "token_classification":
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                **model_kwargs
            )
        elif self.config.task_type == "masked_lm":
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                **model_kwargs
            )
        else:
            # Default to base model
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                **model_kwargs
            )
    
    def tokenize_text(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize text using the loaded tokenizer."""
        tokenize_kwargs = {
            "max_length": self.config.max_length,
            "padding": self.config.padding,
            "truncation": self.config.truncation,
            "return_tensors": self.config.return_tensors,
            **kwargs
        }
        
        return self.tokenizer(text, **tokenize_kwargs)
    
    def encode_text(self, text: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, **kwargs)
    
    def decode_tokens(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings."""
        return {
            "pad_token": self.tokenizer.pad_token_id,
            "eos_token": self.tokenizer.eos_token_id,
            "bos_token": self.tokenizer.bos_token_id,
            "unk_token": self.tokenizer.unk_token_id,
            "mask_token": self.tokenizer.mask_token_id,
            "sep_token": self.tokenizer.sep_token_id,
            "cls_token": self.tokenizer.cls_token_id
        }

class TextClassificationPipeline:
    """Pipeline for text classification using pre-trained models."""
    
    def __init__(self, model_config: ModelConfig, num_labels: int = 2):
        
    """__init__ function."""
self.model_config = model_config
        self.model_config.task_type = "classification"
        self.model_manager = PreTrainedModelManager(model_config)
        self.num_labels = num_labels
        
        # Update model for specific number of labels
        if hasattr(self.model_manager.model.config, 'num_labels'):
            self.model_manager.model.config.num_labels = num_labels
    
    def predict(self, texts: Union[str, List[str]], return_probs: bool = False) -> Union[int, List[int], np.ndarray]:
        """Predict class labels for input texts."""
        self.model_manager.model.eval()
        
        # Tokenize inputs
        inputs = self.model_manager.tokenize_text(texts)
        inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model_manager.model(**inputs)
            logits = outputs.logits
            
            if return_probs:
                probs = torch.softmax(logits, dim=-1)
                return probs.cpu().numpy()
            else:
                predictions = torch.argmax(logits, dim=-1)
                return predictions.cpu().numpy()

class TextGenerationPipeline:
    """Pipeline for text generation using pre-trained models."""
    
    def __init__(self, model_config: ModelConfig):
        
    """__init__ function."""
self.model_config = model_config
        self.model_config.task_type = "generation"
        self.model_manager = PreTrainedModelManager(model_config)
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate text from a prompt."""
        self.model_manager.model.eval()
        
        # Tokenize input
        inputs = self.model_manager.tokenize_text(prompt)
        inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.model_manager.tokenizer.pad_token_id,
            "eos_token_id": self.model_manager.tokenizer.eos_token_id,
            **kwargs
        }
        
        with torch.no_grad():
            outputs = self.model_manager.model.generate(**inputs, **gen_kwargs)
        
        # Decode generated sequences
        generated_texts = []
        for output in outputs:
            generated_text = self.model_manager.decode_tokens(
                output, 
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        return generated_texts

class QuestionAnsweringPipeline:
    """Pipeline for question answering using pre-trained models."""
    
    def __init__(self, model_config: ModelConfig):
        
    """__init__ function."""
self.model_config = model_config
        self.model_config.task_type = "qa"
        self.model_manager = PreTrainedModelManager(model_config)
    
    def answer_question(
        self, 
        question: str, 
        context: str,
        max_answer_length: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """Answer a question based on the given context."""
        self.model_manager.model.eval()
        
        # Format input for QA
        inputs = self.model_manager.tokenizer(
            question,
            context,
            max_length=self.model_config.max_length,
            padding=self.model_config.padding,
            truncation=self.model_config.truncation,
            return_tensors=self.model_config.return_tensors
        )
        inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model_manager.model(**inputs)
            
            # Get start and end logits
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # Find best answer span
            start_index = torch.argmax(start_logits)
            end_index = torch.argmax(end_logits)
            
            # Ensure end_index >= start_index
            if end_index < start_index:
                end_index = start_index + max_answer_length
            
            # Extract answer tokens
            answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
            answer = self.model_manager.decode_tokens(answer_tokens, skip_special_tokens=True)
            
            # Calculate confidence scores
            start_score = torch.softmax(start_logits, dim=-1)[0][start_index].item()
            end_score = torch.softmax(end_logits, dim=-1)[0][end_index].item()
            confidence = (start_score + end_score) / 2
        
        return {
            "answer": answer,
            "start_index": start_index.item(),
            "end_index": end_index.item(),
            "confidence": confidence,
            "start_score": start_score,
            "end_score": end_score
        }

class TokenClassificationPipeline:
    """Pipeline for token classification (NER, POS tagging, etc.)."""
    
    def __init__(self, model_config: ModelConfig, label2id: Optional[Dict[str, int]] = None):
        
    """__init__ function."""
self.model_config = model_config
        self.model_config.task_type = "token_classification"
        self.model_manager = PreTrainedModelManager(model_config)
        self.label2id = label2id or {}
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def predict_tokens(self, text: str) -> List[Dict[str, Any]]:
        """Predict labels for each token in the text."""
        self.model_manager.model.eval()
        
        # Tokenize input
        inputs = self.model_manager.tokenize_text(text)
        inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model_manager.model(**inputs)
            logits = outputs.logits
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        predictions = predictions[0].cpu().numpy()
        
        # Get tokens for alignment
        tokens = self.model_manager.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Align predictions with tokens
        results = []
        for token, pred_id in zip(tokens, predictions):
            if token in self.model_manager.tokenizer.special_tokens_map.values():
                continue
            
            label = self.id2label.get(pred_id, f"LABEL_{pred_id}")
            results.append({
                "token": token,
                "label": label,
                "label_id": pred_id
            })
        
        return results

class ModelRegistry:
    """Registry for managing multiple pre-trained models."""
    
    def __init__(self) -> Any:
        self.models = {}
        self.configs = {}
    
    def register_model(self, name: str, config: ModelConfig):
        """Register a model configuration."""
        self.configs[name] = config
        logger.info(f"Registered model: {name}")
    
    def load_model(self, name: str) -> PreTrainedModelManager:
        """Load a registered model."""
        if name not in self.configs:
            raise ValueError(f"Model '{name}' not found in registry")
        
        if name not in self.models:
            self.models[name] = PreTrainedModelManager(self.configs[name])
        
        return self.models[name]
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.configs.keys())
    
    def unload_model(self, name: str):
        """Unload a model from memory."""
        if name in self.models:
            del self.models[name]
            logger.info(f"Unloaded model: {name}")

class HuggingFacePipeline:
    """Wrapper for HuggingFace pipeline API."""
    
    def __init__(self, task: str, model_name: str, **kwargs):
        
    """__init__ function."""
self.task = task
        self.model_name = model_name
        self.pipeline = pipeline(task, model=model_name, **kwargs)
    
    def __call__(self, inputs, **kwargs) -> Any:
        """Execute the pipeline."""
        return self.pipeline(inputs, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "task": self.task,
            "model_name": self.model_name,
            "model_type": type(self.pipeline.model).__name__,
            "tokenizer_type": type(self.pipeline.tokenizer).__name__
        }

# Example usage and demonstration
def demonstrate_pretrained_models():
    """Demonstrate various pre-trained model capabilities."""
    
    # 1. Text Classification
    print("=== Text Classification ===")
    classification_config = ModelConfig(
        model_name="distilbert-base-uncased",
        task_type="classification",
        max_length=128
    )
    
    classifier = TextClassificationPipeline(classification_config, num_labels=2)
    texts = [
        "I love this movie!",
        "This is terrible.",
        "Amazing performance!"
    ]
    
    predictions = classifier.predict(texts)
    print(f"Predictions: {predictions}")
    
    # 2. Text Generation
    print("\n=== Text Generation ===")
    generation_config = ModelConfig(
        model_name="gpt2",
        task_type="generation",
        max_length=256
    )
    
    generator = TextGenerationPipeline(generation_config)
    prompt = "The future of artificial intelligence is"
    
    generated_texts = generator.generate(
        prompt,
        max_length=50,
        temperature=0.8,
        num_return_sequences=2
    )
    
    for i, text in enumerate(generated_texts):
        print(f"Generated {i+1}: {text}")
    
    # 3. Question Answering
    print("\n=== Question Answering ===")
    qa_config = ModelConfig(
        model_name="distilbert-base-cased-distilled-squad",
        task_type="qa",
        max_length=384
    )
    
    qa_pipeline = QuestionAnsweringPipeline(qa_config)
    question = "What is the capital of France?"
    context = "Paris is the capital and most populous city of France."
    
    answer = qa_pipeline.answer_question(question, context)
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['confidence']:.3f}")
    
    # 4. Model Registry
    print("\n=== Model Registry ===")
    registry = ModelRegistry()
    
    # Register different models
    registry.register_model("bert-classifier", ModelConfig(
        model_name="bert-base-uncased",
        task_type="classification"
    ))
    
    registry.register_model("gpt2-generator", ModelConfig(
        model_name="gpt2",
        task_type="generation"
    ))
    
    print(f"Available models: {registry.get_available_models()}")
    
    # Load a model from registry
    bert_model = registry.load_model("bert-classifier")
    print(f"Loaded model vocab size: {bert_model.get_vocab_size()}")

if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Demonstrate pre-trained models
    demonstrate_pretrained_models() 