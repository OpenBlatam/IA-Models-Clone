"""
Pre-trained Models and Tokenizers System
Comprehensive demonstration of using Transformers library for pre-trained models and tokenizers
"""

import torch
import torch.nn as nn
from transformers import (
    # Tokenizers
    AutoTokenizer, GPT2Tokenizer, BERTTokenizer, T5Tokenizer, RobertaTokenizer,
    PreTrainedTokenizer, PreTrainedTokenizerFast,
    
    # Pre-trained Models
    AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    AutoModelForMaskedLM, AutoModelForSeq2SeqLM,
    
    # Specific Model Classes
    GPT2LMHeadModel, GPT2ForSequenceClassification,
    BertModel, BertForSequenceClassification, BertForTokenClassification,
    T5ForConditionalGeneration, T5Tokenizer,
    RobertaModel, RobertaForSequenceClassification,
    
    # Pipeline and Utilities
    pipeline, Pipeline,
    
    # Training
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    DataCollatorWithPadding, DataCollatorForTokenClassification,
    
    # Model Configuration
    AutoConfig, PretrainedConfig,
    
    # Utilities
    PreTrainedModel, PreTrainedTokenizer
)
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import time
import numpy as np
import json
import os


@dataclass
class ModelConfig:
    """Configuration for pre-trained models and tokenizers."""
    
    # Model selection
    model_name: str = "gpt2"  # gpt2, bert-base-uncased, t5-base, roberta-base
    model_type: str = "causal_lm"  # causal_lm, sequence_classification, token_classification, qa, masked_lm, seq2seq
    
    # Tokenizer settings
    max_length: int = 512
    padding: str = "max_length"  # max_length, longest, do_not_pad
    truncation: bool = True
    return_tensors: str = "pt"  # pt, tf, np
    
    # Model settings
    torch_dtype: str = "auto"  # auto, float16, float32
    device_map: str = "auto"  # auto, cpu, cuda
    low_cpu_mem_usage: bool = True
    
    # Training settings
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 3


class PreTrainedModelManager:
    """Manager for loading and using pre-trained models and tokenizers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config_obj = None
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        logging.info(f"Pre-trained model manager initialized with {config.model_name}")
    
    def _load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer using Transformers library."""
        try:
            # Load configuration
            self.config_obj = AutoConfig.from_pretrained(self.config.model_name)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token'):
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token_id
            
            # Load model based on type
            self.model = self._load_model_by_type()
            
            # Move model to device
            self.model.to(self.device)
            
            logging.info(f"Successfully loaded {self.config.model_name}")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def _load_model_by_type(self) -> PreTrainedModel:
        """Load model based on specified type."""
        model_kwargs = {
            "torch_dtype": torch.float16 if self.config.torch_dtype == "float16" else torch.float32,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage
        }
        
        if self.config.device_map != "auto":
            model_kwargs["device_map"] = self.config.device_map
        
        if self.config.model_type == "causal_lm":
            return AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )
        elif self.config.model_type == "sequence_classification":
            return AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name, num_labels=2, **model_kwargs
            )
        elif self.config.model_type == "token_classification":
            return AutoModelForTokenClassification.from_pretrained(
                self.config.model_name, num_labels=9, **model_kwargs
            )
        elif self.config.model_type == "qa":
            return AutoModelForQuestionAnswering.from_pretrained(
                self.config.model_name, **model_kwargs
            )
        elif self.config.model_type == "masked_lm":
            return AutoModelForMaskedLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )
        elif self.config.model_type == "seq2seq":
            return AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )
        else:
            # Default to base model
            return AutoModel.from_pretrained(self.config.model_name, **model_kwargs)
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text using the pre-trained tokenizer."""
        return self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=self.config.return_tensors
        )
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=self.config.return_tensors
        )
    
    def decode_tokens(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size of the tokenizer."""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings."""
        return {
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': self.tokenizer.bos_token_id,
            'unk_token_id': self.tokenizer.unk_token_id,
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token_id': self.tokenizer.sep_token_id,
            'mask_token_id': self.tokenizer.mask_token_id
        }


class ModelTypeDemonstrator:
    """Demonstrate different types of pre-trained models."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def load_causal_language_model(self, model_name: str = "gpt2"):
        """Load a causal language model (GPT-2, GPT-Neo, etc.)."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.models["causal_lm"] = model
        self.tokenizers["causal_lm"] = tokenizer
        
        logging.info(f"Loaded causal language model: {model_name}")
        return model, tokenizer
    
    def load_sequence_classification_model(self, model_name: str = "bert-base-uncased"):
        """Load a sequence classification model (BERT, RoBERTa, etc.)."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        
        self.models["sequence_classification"] = model
        self.tokenizers["sequence_classification"] = tokenizer
        
        logging.info(f"Loaded sequence classification model: {model_name}")
        return model, tokenizer
    
    def load_token_classification_model(self, model_name: str = "bert-base-uncased"):
        """Load a token classification model (NER, POS tagging, etc.)."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=9
        )
        
        self.models["token_classification"] = model
        self.tokenizers["token_classification"] = tokenizer
        
        logging.info(f"Loaded token classification model: {model_name}")
        return model, tokenizer
    
    def load_question_answering_model(self, model_name: str = "bert-base-uncased"):
        """Load a question answering model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        self.models["question_answering"] = model
        self.tokenizers["question_answering"] = tokenizer
        
        logging.info(f"Loaded question answering model: {model_name}")
        return model, tokenizer
    
    def load_masked_language_model(self, model_name: str = "bert-base-uncased"):
        """Load a masked language model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        self.models["masked_lm"] = model
        self.tokenizers["masked_lm"] = tokenizer
        
        logging.info(f"Loaded masked language model: {model_name}")
        return model, tokenizer
    
    def load_seq2seq_model(self, model_name: str = "t5-base"):
        """Load a sequence-to-sequence model (T5, BART, etc.)."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.models["seq2seq"] = model
        self.tokenizers["seq2seq"] = tokenizer
        
        logging.info(f"Loaded sequence-to-sequence model: {model_name}")
        return model, tokenizer


class TokenizerDemonstrator:
    """Demonstrate different tokenizer capabilities."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def demonstrate_tokenization(self, text: str):
        """Demonstrate various tokenization methods."""
        print(f"\n=== Tokenization Demonstration for: '{text}' ===")
        
        # Basic tokenization
        tokens = self.tokenizer.tokenize(text)
        print(f"Tokenized: {tokens}")
        
        # Token to ID mapping
        token_ids = self.tokenizer.encode(text)
        print(f"Token IDs: {token_ids}")
        
        # Full encoding with attention mask
        encoding = self.tokenizer(
            text,
            max_length=20,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        print(f"Full encoding: {encoding}")
        
        # Decoding
        decoded = self.tokenizer.decode(token_ids)
        print(f"Decoded: {decoded}")
        
        # Special tokens
        special_tokens = self.get_special_tokens_info()
        print(f"Special tokens: {special_tokens}")
    
    def get_special_tokens_info(self) -> Dict[str, Any]:
        """Get information about special tokens."""
        return {
            'pad_token': self.tokenizer.pad_token,
            'eos_token': self.tokenizer.eos_token,
            'bos_token': self.tokenizer.bos_token,
            'unk_token': self.tokenizer.unk_token,
            'cls_token': self.tokenizer.cls_token,
            'sep_token': self.tokenizer.sep_token,
            'mask_token': self.tokenizer.mask_token,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'vocab_size': self.tokenizer.vocab_size
        }
    
    def demonstrate_batch_tokenization(self, texts: List[str]):
        """Demonstrate batch tokenization."""
        print(f"\n=== Batch Tokenization Demonstration ===")
        
        # Batch encoding
        batch_encoding = self.tokenizer(
            texts,
            max_length=20,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        print(f"Batch input_ids shape: {batch_encoding['input_ids'].shape}")
        print(f"Batch attention_mask shape: {batch_encoding['attention_mask'].shape}")
        
        # Decode batch
        for i, text in enumerate(texts):
            decoded = self.tokenizer.decode(batch_encoding['input_ids'][i])
            print(f"Text {i+1}: {decoded}")


class PipelineDemonstrator:
    """Demonstrate using Transformers pipelines."""
    
    def __init__(self):
        self.pipelines = {}
    
    def create_text_generation_pipeline(self, model_name: str = "gpt2"):
        """Create a text generation pipeline."""
        pipeline_obj = pipeline("text-generation", model=model_name)
        self.pipelines["text_generation"] = pipeline_obj
        return pipeline_obj
    
    def create_sentiment_analysis_pipeline(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Create a sentiment analysis pipeline."""
        pipeline_obj = pipeline("sentiment-analysis", model=model_name)
        self.pipelines["sentiment_analysis"] = pipeline_obj
        return pipeline_obj
    
    def create_question_answering_pipeline(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """Create a question answering pipeline."""
        pipeline_obj = pipeline("question-answering", model=model_name)
        self.pipelines["question_answering"] = pipeline_obj
        return pipeline_obj
    
    def create_named_entity_recognition_pipeline(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        """Create a named entity recognition pipeline."""
        pipeline_obj = pipeline("ner", model=model_name)
        self.pipelines["ner"] = pipeline_obj
        return pipeline_obj
    
    def create_translation_pipeline(self, model_name: str = "Helsinki-NLP/opus-mt-en-es"):
        """Create a translation pipeline."""
        pipeline_obj = pipeline("translation", model=model_name)
        self.pipelines["translation"] = pipeline_obj
        return pipeline_obj
    
    def demonstrate_pipelines(self):
        """Demonstrate all available pipelines."""
        print("\n=== Pipeline Demonstrations ===")
        
        # Text generation
        if "text_generation" in self.pipelines:
            result = self.pipelines["text_generation"]("The future of AI is", max_length=50)
            print(f"Text generation: {result[0]['generated_text']}")
        
        # Sentiment analysis
        if "sentiment_analysis" in self.pipelines:
            result = self.pipelines["sentiment_analysis"]("I love this technology!")
            print(f"Sentiment analysis: {result[0]}")
        
        # Question answering
        if "question_answering" in self.pipelines:
            result = self.pipelines["question_answering"](
                question="What is artificial intelligence?",
                context="Artificial intelligence is a branch of computer science."
            )
            print(f"Question answering: {result}")
        
        # Named entity recognition
        if "ner" in self.pipelines:
            result = self.pipelines["ner"]("Apple Inc. is headquartered in Cupertino, California.")
            print(f"NER: {result}")
        
        # Translation
        if "translation" in self.pipelines:
            result = self.pipelines["translation"]("Hello, how are you?")
            print(f"Translation: {result[0]['translation_text']}")


class ModelComparison:
    """Compare different pre-trained models."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def load_models_for_comparison(self):
        """Load different models for comparison."""
        models_to_load = [
            ("gpt2", "causal_lm"),
            ("bert-base-uncased", "sequence_classification"),
            ("roberta-base", "sequence_classification"),
            ("t5-base", "seq2seq")
        ]
        
        for model_name, model_type in models_to_load:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                if model_type == "causal_lm":
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                elif model_type == "sequence_classification":
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, num_labels=2
                    )
                elif model_type == "seq2seq":
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                
                logging.info(f"Loaded {model_name} for comparison")
                
            except Exception as e:
                logging.warning(f"Failed to load {model_name}: {e}")
    
    def compare_model_sizes(self) -> Dict[str, Dict[str, Any]]:
        """Compare model sizes and parameters."""
        comparison = {}
        
        for model_name, model in self.models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            comparison[model_name] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
                'vocab_size': self.tokenizers[model_name].vocab_size
            }
        
        return comparison
    
    def print_comparison(self):
        """Print model comparison."""
        comparison = self.compare_model_sizes()
        
        print("\n=== Model Comparison ===")
        print(f"{'Model':<20} {'Parameters':<15} {'Size (MB)':<12} {'Vocab Size':<12}")
        print("-" * 60)
        
        for model_name, stats in comparison.items():
            print(f"{model_name:<20} {stats['total_parameters']:<15,} "
                  f"{stats['model_size_mb']:<12.1f} {stats['vocab_size']:<12,}")


class TrainingDemonstrator:
    """Demonstrate training with pre-trained models."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def setup_training(self, config: ModelConfig):
        """Setup training configuration."""
        self.training_args = TrainingArguments(
            output_dir="./training_output",
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            save_strategy="epoch",
            logging_steps=10,
            remove_unused_columns=False
        )
    
    def create_dataset(self, texts: List[str]) -> Dataset:
        """Create a dataset for training."""
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "labels": encoding["input_ids"].flatten()
                }
        
        return TextDataset(texts, self.tokenizer, 128)
    
    def train_model(self, train_texts: List[str]):
        """Train the model using the Trainer class."""
        # Create dataset
        train_dataset = self.create_dataset(train_texts)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        logging.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model("./trained_model")
        self.tokenizer.save_pretrained("./trained_model")
        
        logging.info("Training completed!")
        return trainer


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Pre-trained Models and Tokenizers Demonstration ===\n")
    
    # 1. Basic model and tokenizer loading
    config = ModelConfig(model_name="gpt2", model_type="causal_lm")
    model_manager = PreTrainedModelManager(config)
    
    # Demonstrate tokenization
    text = "The Transformers library provides easy access to pre-trained models."
    tokenized = model_manager.tokenize_text(text)
    print(f"Tokenized text: {tokenized}")
    
    # 2. Different model types
    demonstrator = ModelTypeDemonstrator()
    demonstrator.load_causal_language_model("gpt2")
    demonstrator.load_sequence_classification_model("bert-base-uncased")
    demonstrator.load_token_classification_model("bert-base-uncased")
    demonstrator.load_question_answering_model("bert-base-uncased")
    demonstrator.load_masked_language_model("bert-base-uncased")
    demonstrator.load_seq2seq_model("t5-base")
    
    # 3. Tokenizer demonstrations
    tokenizer_demo = TokenizerDemonstrator(demonstrator.tokenizers["causal_lm"])
    tokenizer_demo.demonstrate_tokenization("Hello, world!")
    tokenizer_demo.demonstrate_batch_tokenization([
        "First sentence.",
        "Second sentence with more words.",
        "Third sentence."
    ])
    
    # 4. Pipeline demonstrations
    pipeline_demo = PipelineDemonstrator()
    pipeline_demo.create_text_generation_pipeline()
    pipeline_demo.create_sentiment_analysis_pipeline()
    pipeline_demo.create_question_answering_pipeline()
    pipeline_demo.create_named_entity_recognition_pipeline()
    pipeline_demo.demonstrate_pipelines()
    
    # 5. Model comparison
    comparison = ModelComparison()
    comparison.load_models_for_comparison()
    comparison.print_comparison()
    
    # 6. Training demonstration
    sample_texts = [
        "The Transformers library is amazing for NLP tasks.",
        "Pre-trained models save time and improve performance.",
        "Tokenizers handle text preprocessing efficiently.",
        "Fine-tuning adapts models to specific domains."
    ]
    
    training_demo = TrainingDemonstrator(
        demonstrator.models["causal_lm"],
        demonstrator.tokenizers["causal_lm"]
    )
    training_demo.setup_training(config)
    # training_demo.train_model(sample_texts)  # Uncomment to run training
    
    print("\n=== Demonstration Completed Successfully! ===")





