#!/usr/bin/env python3
"""
Transformers Library Integration for Blaze AI
Demonstrates working with pre-trained models and tokenizers from Hugging Face
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering, AutoModelForMaskedLM,
    AutoConfig, AutoModelForSeq2SeqLM, pipeline, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, DataCollatorForTokenClassification,
    DataCollatorForSeq2SeqLM, EarlyStoppingCallback, IntervalStrategy,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup, PreTrainedTokenizer, PreTrainedModel
)
from transformers.utils import logging as transformers_logging
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
import json
import os
from datasets import Dataset, load_dataset
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for pre-trained models"""
    model_name: str = "gpt2"  # gpt2, bert-base-uncased, t5-base, etc.
    model_type: str = "causal_lm"  # causal_lm, sequence_classification, token_classification, etc.
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    device: str = "auto"  # auto, cpu, cuda, mps


@dataclass
class TokenizerConfig:
    """Configuration for tokenizers"""
    tokenizer_name: str = "gpt2"
    add_special_tokens: bool = True
    padding_side: str = "right"
    model_max_length: int = 512
    use_fast: bool = True


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 10
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


class TransformersModelManager:
    """Manager for working with pre-trained models and tokenizers"""
    
    def __init__(self, model_config: ModelConfig, tokenizer_config: TokenizerConfig):
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config
        
        # Auto-detect device
        if model_config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                     "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(model_config.device)
        
        self.tokenizer = None
        self.model = None
        self.model_type = None
    
    def load_tokenizer(self, tokenizer_name: str = None) -> PreTrainedTokenizer:
        """Load pre-trained tokenizer"""
        
        tokenizer_name = tokenizer_name or self.tokenizer_config.tokenizer_name
        
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                add_special_tokens=self.tokenizer_config.add_special_tokens,
                padding_side=self.tokenizer_config.padding_side,
                model_max_length=self.tokenizer_config.model_max_length,
                use_fast=self.tokenizer_config.use_fast
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Tokenizer loaded successfully: {self.tokenizer.__class__.__name__}")
            logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")
            logger.info(f"Model max length: {self.tokenizer.model_max_length}")
            
            return self.tokenizer
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def load_model(self, model_name: str = None, model_type: str = None) -> PreTrainedModel:
        """Load pre-trained model based on type"""
        
        model_name = model_name or self.model_config.model_name
        model_type = model_type or self.model_config.model_type
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        
        try:
            if model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model_type = "causal_lm"
            elif model_type == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model_type = "sequence_classification"
            elif model_type == "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(model_name)
                self.model_type = "token_classification"
            elif model_type == "question_answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
                self.model_type = "question_answering"
            elif model_type == "masked_lm":
                self.model = AutoModelForMaskedLM.from_pretrained(model_name)
                self.model_type = "masked_lm"
            elif model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.model_type = "seq2seq"
            else:
                # Auto-detect model type
                self.model = AutoModel.from_pretrained(model_name)
                self.model_type = "auto"
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully: {self.model.__class__.__name__}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            logger.info(f"Device: {self.device}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            "model_type": self.model_type,
            "model_class": self.model.__class__.__name__,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "config": self.model.config.to_dict() if hasattr(self.model, 'config') else {}
        }
        
        return info


class TextGenerationManager:
    """Manager for text generation tasks"""
    
    def __init__(self, model_manager: TransformersModelManager):
        self.model_manager = model_manager
        self.tokenizer = model_manager.tokenizer
        self.model = model_manager.model
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 1.0,
                     top_k: int = 50, top_p: float = 0.9, do_sample: bool = True,
                     num_return_sequences: int = 1) -> List[str]:
        """Generate text using the loaded model"""
        
        if self.model_type != "causal_lm":
            raise ValueError("Model must be a causal language model for text generation")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_manager.model_config.max_length
        ).to(self.model_manager.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs['attention_mask']
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Remove input tokens from output
            generated_tokens = output[inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def complete_sentence(self, sentence: str, max_new_tokens: int = 20) -> str:
        """Complete a given sentence"""
        
        return self.generate_text(sentence, max_length=len(sentence.split()) + max_new_tokens)[0]


class SequenceClassificationManager:
    """Manager for sequence classification tasks"""
    
    def __init__(self, model_manager: TransformersModelManager):
        self.model_manager = model_manager
        self.tokenizer = model_manager.tokenizer
        self.model = model_manager.model
    
    def classify_text(self, texts: Union[str, List[str]], return_probs: bool = False) -> Union[Dict, List[Dict]]:
        """Classify text(s) using the loaded model"""
        
        if self.model_type != "sequence_classification":
            raise ValueError("Model must be a sequence classification model")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.model_manager.model_config.max_length,
            return_tensors="pt"
        ).to(self.model_manager.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        # Process results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            result = {
                "text": texts[i],
                "prediction": pred.item(),
                "confidence": prob[pred].item()
            }
            
            if return_probs:
                result["probabilities"] = prob.tolist()
            
            results.append(result)
        
        return results[0] if len(texts) == 1 else results


class TokenClassificationManager:
    """Manager for token classification tasks (NER, POS tagging, etc.)"""
    
    def __init__(self, model_manager: TransformersModelManager):
        self.model_manager = model_manager
        self.tokenizer = model_manager.tokenizer
        self.model = model_manager.model
    
    def classify_tokens(self, text: str, return_offsets: bool = True) -> List[Dict]:
        """Classify tokens in text (e.g., Named Entity Recognition)"""
        
        if self.model_type != "token_classification":
            raise ValueError("Model must be a token classification model")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_manager.model_config.max_length,
            return_offsets_mapping=return_offsets
        )
        
        # Move to device
        inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        # Process results
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        predictions = predictions[0].cpu().tolist()
        
        results = []
        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            if token in [self.tokenizer.pad_token, self.tokenizer.sep_token]:
                continue
            
            result = {
                "token": token,
                "prediction": pred,
                "label": self.model.config.id2label[pred] if hasattr(self.model.config, 'id2label') else str(pred)
            }
            
            if return_offsets and 'offset_mapping' in inputs:
                start, end = inputs['offset_mapping'][0][i]
                result["start"] = start.item()
                result["end"] = end.item()
            
            results.append(result)
        
        return results


class QuestionAnsweringManager:
    """Manager for question answering tasks"""
    
    def __init__(self, model_manager: TransformersModelManager):
        self.model_manager = model_manager
        self.tokenizer = model_manager.tokenizer
        self.model = model_manager.model
    
    def answer_question(self, question: str, context: str, max_answer_length: int = 50) -> Dict:
        """Answer a question based on context"""
        
        if self.model_type != "question_answering":
            raise ValueError("Model must be a question answering model")
        
        # Tokenize inputs
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_manager.model_config.max_length,
            return_overflowing_tokens=True,
            stride=128
        ).to(self.model_manager.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
        
        # Find best answer span
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        
        # Ensure answer_end >= answer_start
        if answer_end <= answer_start:
            answer_end = answer_start + 1
        
        # Decode answer
        answer_tokens = inputs['input_ids'][0][answer_start:answer_end]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate confidence scores
        start_score = answer_start_scores[0][answer_start].item()
        end_score = answer_end_scores[0][answer_end - 1].item()
        confidence = (start_score + end_score) / 2
        
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "start": answer_start.item(),
            "end": answer_end.item(),
            "confidence": confidence
        }


class FineTuningManager:
    """Manager for fine-tuning pre-trained models"""
    
    def __init__(self, model_manager: TransformersModelManager, training_config: TrainingConfig):
        self.model_manager = model_manager
        self.training_config = training_config
        self.tokenizer = model_manager.tokenizer
        self.model = model_manager.model
    
    def prepare_dataset(self, texts: List[str], labels: Optional[List[int]] = None,
                       task_type: str = "classification") -> Dataset:
        """Prepare dataset for fine-tuning"""
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.model_manager.model_config.max_length,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset_dict = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }
        
        if labels is not None:
            dataset_dict["labels"] = labels
        
        return Dataset.from_dict(dataset_dict)
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Dataset = None,
                       compute_metrics: Optional[callable] = None) -> Trainer:
        """Create trainer for fine-tuning"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=self.training_config.learning_rate,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            warmup_steps=self.training_config.warmup_steps,
            weight_decay=self.training_config.weight_decay,
            logging_steps=self.training_config.logging_steps,
            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            report_to=None,  # Disable wandb/tensorboard
            push_to_hub=False
        )
        
        # Data collator
        if self.model_manager.model_type == "causal_lm":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        elif self.model_manager.model_type == "token_classification":
            data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        elif self.model_manager.model_type == "seq2seq":
            data_collator = DataCollatorForSeq2SeqLM(tokenizer=self.tokenizer)
        else:
            data_collator = None
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        return trainer
    
    def fine_tune(self, train_dataset: Dataset, eval_dataset: Dataset = None,
                  compute_metrics: Optional[callable] = None) -> Dict[str, Any]:
        """Fine-tune the model"""
        
        logger.info("Starting fine-tuning...")
        
        # Create trainer
        trainer = self.create_trainer(train_dataset, eval_dataset, compute_metrics)
        
        # Train model
        train_result = trainer.train()
        
        # Evaluate model
        eval_result = None
        if eval_dataset is not None:
            eval_result = trainer.evaluate()
        
        # Save model
        trainer.save_model("./fine_tuned_model")
        self.tokenizer.save_pretrained("./fine_tuned_model")
        
        logger.info("Fine-tuning completed successfully!")
        
        return {
            "train_result": train_result,
            "eval_result": eval_result,
            "model_path": "./fine_tuned_model"
        }


class PipelineManager:
    """Manager for using Hugging Face pipelines"""
    
    def __init__(self):
        self.pipelines = {}
    
    def create_pipeline(self, task: str, model: str, **kwargs) -> Any:
        """Create a pipeline for a specific task"""
        
        logger.info(f"Creating pipeline for task: {task} with model: {model}")
        
        try:
            pipeline_obj = pipeline(task, model=model, **kwargs)
            self.pipelines[task] = pipeline_obj
            
            logger.info(f"Pipeline created successfully: {task}")
            return pipeline_obj
            
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            raise
    
    def text_generation_pipeline(self, model: str = "gpt2") -> Any:
        """Create text generation pipeline"""
        return self.create_pipeline("text-generation", model)
    
    def sentiment_analysis_pipeline(self, model: str = "distilbert-base-uncased-finetuned-sst-2-english") -> Any:
        """Create sentiment analysis pipeline"""
        return self.create_pipeline("sentiment-analysis", model)
    
    def named_entity_recognition_pipeline(self, model: str = "dbmdz/bert-large-cased-finetuned-conll03-english") -> Any:
        """Create NER pipeline"""
        return self.create_pipeline("ner", model)
    
    def question_answering_pipeline(self, model: str = "distilbert-base-cased-distilled-squad") -> Any:
        """Create question answering pipeline"""
        return self.create_pipeline("question-answering", model)
    
    def summarization_pipeline(self, model: str = "facebook/bart-large-cnn") -> Any:
        """Create summarization pipeline"""
        return self.create_pipeline("summarization", model)
    
    def translation_pipeline(self, model: str = "Helsinki-NLP/opus-mt-en-es") -> Any:
        """Create translation pipeline"""
        return self.create_pipeline("translation", model)


class TransformersExperiments:
    """Collection of experiments using the Transformers library"""
    
    @staticmethod
    def demonstrate_text_generation():
        """Demonstrate text generation with pre-trained models"""
        
        logger.info("Demonstrating text generation...")
        
        # Create model manager
        model_config = ModelConfig(
            model_name="gpt2",
            model_type="causal_lm",
            max_length=128
        )
        
        tokenizer_config = TokenizerConfig(tokenizer_name="gpt2")
        
        model_manager = TransformersModelManager(model_config, tokenizer_config)
        
        # Load tokenizer and model
        tokenizer = model_manager.load_tokenizer()
        model = model_manager.load_model()
        
        # Create text generation manager
        gen_manager = TextGenerationManager(model_manager)
        
        # Generate text
        prompts = [
            "The future of artificial intelligence",
            "Once upon a time in Silicon Valley",
            "The best way to learn machine learning is"
        ]
        
        results = []
        for prompt in prompts:
            generated = gen_manager.generate_text(
                prompt,
                max_length=50,
                temperature=0.8,
                top_k=50,
                do_sample=True
            )
            results.append({"prompt": prompt, "generated": generated[0]})
        
        logger.info("Text generation demonstration completed")
        return model_manager, gen_manager, results
    
    @staticmethod
    def demonstrate_classification():
        """Demonstrate sequence classification"""
        
        logger.info("Demonstrating sequence classification...")
        
        # Create model manager
        model_config = ModelConfig(
            model_name="distilbert-base-uncased-finetuned-sst-2-english",
            model_type="sequence_classification",
            max_length=128
        )
        
        tokenizer_config = TokenizerConfig(tokenizer_name="distilbert-base-uncased")
        
        model_manager = TransformersModelManager(model_config, tokenizer_config)
        
        # Load tokenizer and model
        tokenizer = model_manager.load_tokenizer()
        model = model_manager.load_model()
        
        # Create classification manager
        class_manager = SequenceClassificationManager(model_manager)
        
        # Classify texts
        texts = [
            "I love this movie, it's fantastic!",
            "This is the worst experience ever.",
            "The weather is okay today.",
            "Amazing performance by the team!"
        ]
        
        results = class_manager.classify_text(texts, return_probs=True)
        
        logger.info("Classification demonstration completed")
        return model_manager, class_manager, results
    
    @staticmethod
    def demonstrate_pipelines():
        """Demonstrate Hugging Face pipelines"""
        
        logger.info("Demonstrating Hugging Face pipelines...")
        
        # Create pipeline manager
        pipeline_manager = PipelineManager()
        
        # Create various pipelines
        pipelines = {}
        
        # Text generation
        pipelines["text_generation"] = pipeline_manager.text_generation_pipeline()
        
        # Sentiment analysis
        pipelines["sentiment"] = pipeline_manager.sentiment_analysis_pipeline()
        
        # NER
        pipelines["ner"] = pipeline_manager.named_entity_recognition_pipeline()
        
        # Test pipelines
        test_text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."
        
        results = {}
        
        # Sentiment analysis
        sentiment_result = pipelines["sentiment"](test_text)
        results["sentiment"] = sentiment_result
        
        # NER
        ner_result = pipelines["ner"](test_text)
        results["ner"] = ner_result
        
        # Text generation
        gen_result = pipelines["text_generation"]("The future of technology", max_length=30)
        results["generation"] = gen_result
        
        logger.info("Pipeline demonstration completed")
        return pipeline_manager, pipelines, results


def main():
    """Main execution function"""
    logger.info("Starting Transformers Library Demonstrations...")
    
    # Demonstrate text generation
    logger.info("Testing text generation with pre-trained models...")
    gen_model_manager, gen_manager, gen_results = TransformersExperiments.demonstrate_text_generation()
    
    # Demonstrate classification
    logger.info("Testing sequence classification...")
    class_model_manager, class_manager, class_results = TransformersExperiments.demonstrate_classification()
    
    # Demonstrate pipelines
    logger.info("Testing Hugging Face pipelines...")
    pipeline_manager, pipelines, pipeline_results = TransformersExperiments.demonstrate_pipelines()
    
    # Create comprehensive model manager
    logger.info("Creating comprehensive model manager...")
    
    comprehensive_config = ModelConfig(
        model_name="bert-base-uncased",
        model_type="sequence_classification",
        max_length=512
    )
    
    comprehensive_tokenizer_config = TokenizerConfig(
        tokenizer_name="bert-base-uncased",
        model_max_length=512
    )
    
    comprehensive_manager = TransformersModelManager(comprehensive_config, comprehensive_tokenizer_config)
    
    # Load comprehensive model
    comprehensive_tokenizer = comprehensive_manager.load_tokenizer()
    comprehensive_model = comprehensive_manager.load_model()
    
    # Get model information
    model_info = comprehensive_manager.get_model_info()
    
    # Summary
    logger.info("Transformers Library Summary:")
    logger.info(f"Text generation models tested: ✓")
    logger.info(f"Classification models tested: ✓")
    logger.info(f"Pipelines tested: ✓")
    logger.info(f"Comprehensive model loaded: ✓")
    logger.info(f"Model parameters: {model_info.get('parameters', 0):,}")
    logger.info(f"Model type: {model_info.get('model_type', 'Unknown')}")
    
    logger.info("Transformers Library demonstrations completed successfully!")


if __name__ == "__main__":
    main()
