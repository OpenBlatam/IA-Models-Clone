from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import time
import json
import os
    from transformers import (
    from datasets import Dataset, load_dataset
from typing import Any, List, Dict, Optional
import asyncio
"""
Pre-trained Models and Tokenizers for HeyGen AI.

Integration with Hugging Face Transformers library for working with pre-trained models
and tokenizers following PEP 8 style guidelines.
"""


try:
        AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        AutoModelForMaskedLM, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoModelForQuestionAnswering,
        pipeline, Pipeline, PreTrainedTokenizer, PreTrainedModel,
        TrainingArguments, Trainer, DataCollatorWithPadding,
        TextGenerationPipeline, TextClassificationPipeline,
        QuestionAnsweringPipeline, TokenClassificationPipeline,
        FillMaskPipeline, TranslationPipeline, SummarizationPipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Install with: pip install transformers datasets")

logger = logging.getLogger(__name__)


class PreTrainedModelManager:
    """Manager for pre-trained models and tokenizers."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize pre-trained model manager.

        Args:
            cache_dir: Directory to cache models and tokenizers.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required. Install with: pip install transformers datasets")
        
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.loaded_models = {}
        self.loaded_tokenizers = {}

    def load_tokenizer(
        self,
        model_name: str,
        use_fast: bool = True,
        **kwargs
    ) -> PreTrainedTokenizer:
        """Load pre-trained tokenizer.

        Args:
            model_name: Name or path of the model.
            use_fast: Whether to use fast tokenizer.
            **kwargs: Additional arguments for tokenizer.

        Returns:
            PreTrainedTokenizer: Loaded tokenizer.
        """
        if model_name in self.loaded_tokenizers:
            return self.loaded_tokenizers[model_name]
        
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            cache_dir=self.cache_dir,
            **kwargs
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.loaded_tokenizers[model_name] = tokenizer
        return tokenizer

    def load_model(
        self,
        model_name: str,
        model_type: str = "auto",
        device: Optional[torch.device] = None,
        **kwargs
    ) -> PreTrainedModel:
        """Load pre-trained model.

        Args:
            model_name: Name or path of the model.
            model_type: Type of model to load.
            device: Device to load model on.
            **kwargs: Additional arguments for model.

        Returns:
            PreTrainedModel: Loaded model.
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model: {model_name} on {device}")
        
        # Load model based on type
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                **kwargs
            )
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                **kwargs
            )
        elif model_type == "masked_lm":
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                **kwargs
            )
        elif model_type == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                **kwargs
            )
        elif model_type == "token_classification":
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                **kwargs
            )
        elif model_type == "question_answering":
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                **kwargs
            )
        else:
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                **kwargs
            )
        
        model.to(device)
        model.eval()
        
        self.loaded_models[model_name] = model
        return model

    def create_pipeline(
        self,
        task: str,
        model_name: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Pipeline:
        """Create Hugging Face pipeline.

        Args:
            task: Task type (text-generation, text-classification, etc.).
            model_name: Name or path of the model.
            device: Device to run pipeline on.
            **kwargs: Additional arguments for pipeline.

        Returns:
            Pipeline: Created pipeline.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Creating pipeline for task: {task} with model: {model_name}")
        
        return pipeline(
            task=task,
            model=model_name,
            device=device,
            cache_dir=self.cache_dir,
            **kwargs
        )


class TextGenerationManager:
    """Manager for text generation with pre-trained models."""

    def __init__(self, model_manager: PreTrainedModelManager):
        """Initialize text generation manager.

        Args:
            model_manager: Pre-trained model manager.
        """
        self.model_manager = model_manager
        self.generation_pipelines = {}

    def create_generation_pipeline(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> TextGenerationPipeline:
        """Create text generation pipeline.

        Args:
            model_name: Name or path of the model.
            device: Device to run pipeline on.
            **kwargs: Additional arguments for pipeline.

        Returns:
            TextGenerationPipeline: Created pipeline.
        """
        if model_name in self.generation_pipelines:
            return self.generation_pipelines[model_name]
        
        pipeline = self.model_manager.create_pipeline(
            task="text-generation",
            model_name=model_name,
            device=device,
            **kwargs
        )
        
        self.generation_pipelines[model_name] = pipeline
        return pipeline

    def generate_text(
        self,
        model_name: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate text using pre-trained model.

        Args:
            model_name: Name or path of the model.
            prompt: Input prompt.
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            do_sample: Whether to use sampling.
            num_return_sequences: Number of sequences to return.
            **kwargs: Additional generation parameters.

        Returns:
            List[str]: Generated texts.
        """
        pipeline = self.create_generation_pipeline(model_name)
        
        generation_kwargs = {
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            **kwargs
        }
        
        results = pipeline(prompt, **generation_kwargs)
        
        if isinstance(results, list):
            return [result["generated_text"] for result in results]
        else:
            return [results["generated_text"]]

    def batch_generate(
        self,
        model_name: str,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> List[List[str]]:
        """Generate text for multiple prompts.

        Args:
            model_name: Name or path of the model.
            prompts: List of input prompts.
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            do_sample: Whether to use sampling.
            **kwargs: Additional generation parameters.

        Returns:
            List[List[str]]: Generated texts for each prompt.
        """
        pipeline = self.create_generation_pipeline(model_name)
        
        generation_kwargs = {
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            **kwargs
        }
        
        all_results = []
        for prompt in prompts:
            results = pipeline(prompt, **generation_kwargs)
            if isinstance(results, list):
                generated_texts = [result["generated_text"] for result in results]
            else:
                generated_texts = [results["generated_text"]]
            all_results.append(generated_texts)
        
        return all_results


class TextClassificationManager:
    """Manager for text classification with pre-trained models."""

    def __init__(self, model_manager: PreTrainedModelManager):
        """Initialize text classification manager.

        Args:
            model_manager: Pre-trained model manager.
        """
        self.model_manager = model_manager
        self.classification_pipelines = {}

    def create_classification_pipeline(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> TextClassificationPipeline:
        """Create text classification pipeline.

        Args:
            model_name: Name or path of the model.
            device: Device to run pipeline on.
            **kwargs: Additional arguments for pipeline.

        Returns:
            TextClassificationPipeline: Created pipeline.
        """
        if model_name in self.classification_pipelines:
            return self.classification_pipelines[model_name]
        
        pipeline = self.model_manager.create_pipeline(
            task="text-classification",
            model_name=model_name,
            device=device,
            **kwargs
        )
        
        self.classification_pipelines[model_name] = pipeline
        return pipeline

    def classify_text(
        self,
        model_name: str,
        text: str,
        return_all_scores: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Classify text using pre-trained model.

        Args:
            model_name: Name or path of the model.
            text: Input text to classify.
            return_all_scores: Whether to return all class scores.
            **kwargs: Additional classification parameters.

        Returns:
            Dict[str, Any]: Classification results.
        """
        pipeline = self.create_classification_pipeline(model_name)
        
        classification_kwargs = {
            "return_all_scores": return_all_scores,
            **kwargs
        }
        
        result = pipeline(text, **classification_kwargs)
        return result

    def batch_classify(
        self,
        model_name: str,
        texts: List[str],
        return_all_scores: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Classify multiple texts.

        Args:
            model_name: Name or path of the model.
            texts: List of texts to classify.
            return_all_scores: Whether to return all class scores.
            **kwargs: Additional classification parameters.

        Returns:
            List[Dict[str, Any]]: Classification results for each text.
        """
        pipeline = self.create_classification_pipeline(model_name)
        
        classification_kwargs = {
            "return_all_scores": return_all_scores,
            **kwargs
        }
        
        results = []
        for text in texts:
            result = pipeline(text, **classification_kwargs)
            results.append(result)
        
        return results


class QuestionAnsweringManager:
    """Manager for question answering with pre-trained models."""

    def __init__(self, model_manager: PreTrainedModelManager):
        """Initialize question answering manager.

        Args:
            model_manager: Pre-trained model manager.
        """
        self.model_manager = model_manager
        self.qa_pipelines = {}

    def create_qa_pipeline(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> QuestionAnsweringPipeline:
        """Create question answering pipeline.

        Args:
            model_name: Name or path of the model.
            device: Device to run pipeline on.
            **kwargs: Additional arguments for pipeline.

        Returns:
            QuestionAnsweringPipeline: Created pipeline.
        """
        if model_name in self.qa_pipelines:
            return self.qa_pipelines[model_name]
        
        pipeline = self.model_manager.create_pipeline(
            task="question-answering",
            model_name=model_name,
            device=device,
            **kwargs
        )
        
        self.qa_pipelines[model_name] = pipeline
        return pipeline

    def answer_question(
        self,
        model_name: str,
        question: str,
        context: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Answer question using pre-trained model.

        Args:
            model_name: Name or path of the model.
            question: Question to answer.
            context: Context to search for answer.
            **kwargs: Additional QA parameters.

        Returns:
            Dict[str, Any]: Answer and confidence scores.
        """
        pipeline = self.create_qa_pipeline(model_name)
        
        result = pipeline(question=question, context=context, **kwargs)
        return result

    def batch_answer_questions(
        self,
        model_name: str,
        questions: List[str],
        contexts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Answer multiple questions.

        Args:
            model_name: Name or path of the model.
            questions: List of questions.
            contexts: List of contexts.
            **kwargs: Additional QA parameters.

        Returns:
            List[Dict[str, Any]]: Answers for each question.
        """
        pipeline = self.create_qa_pipeline(model_name)
        
        results = []
        for question, context in zip(questions, contexts):
            result = pipeline(question=question, context=context, **kwargs)
            results.append(result)
        
        return results


class TranslationManager:
    """Manager for translation with pre-trained models."""

    def __init__(self, model_manager: PreTrainedModelManager):
        """Initialize translation manager.

        Args:
            model_manager: Pre-trained model manager.
        """
        self.model_manager = model_manager
        self.translation_pipelines = {}

    def create_translation_pipeline(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> TranslationPipeline:
        """Create translation pipeline.

        Args:
            model_name: Name or path of the model.
            device: Device to run pipeline on.
            **kwargs: Additional arguments for pipeline.

        Returns:
            TranslationPipeline: Created pipeline.
        """
        if model_name in self.translation_pipelines:
            return self.translation_pipelines[model_name]
        
        pipeline = self.model_manager.create_pipeline(
            task="translation",
            model_name=model_name,
            device=device,
            **kwargs
        )
        
        self.translation_pipelines[model_name] = pipeline
        return pipeline

    def translate_text(
        self,
        model_name: str,
        text: str,
        **kwargs
    ) -> str:
        """Translate text using pre-trained model.

        Args:
            model_name: Name or path of the model.
            text: Text to translate.
            **kwargs: Additional translation parameters.

        Returns:
            str: Translated text.
        """
        pipeline = self.create_translation_pipeline(model_name)
        
        result = pipeline(text, **kwargs)
        return result[0]["translation_text"]

    def batch_translate(
        self,
        model_name: str,
        texts: List[str],
        **kwargs
    ) -> List[str]:
        """Translate multiple texts.

        Args:
            model_name: Name or path of the model.
            texts: List of texts to translate.
            **kwargs: Additional translation parameters.

        Returns:
            List[str]: Translated texts.
        """
        pipeline = self.create_translation_pipeline(model_name)
        
        results = []
        for text in texts:
            result = pipeline(text, **kwargs)
            results.append(result[0]["translation_text"])
        
        return results


class SummarizationManager:
    """Manager for text summarization with pre-trained models."""

    def __init__(self, model_manager: PreTrainedModelManager):
        """Initialize summarization manager.

        Args:
            model_manager: Pre-trained model manager.
        """
        self.model_manager = model_manager
        self.summarization_pipelines = {}

    def create_summarization_pipeline(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> SummarizationPipeline:
        """Create summarization pipeline.

        Args:
            model_name: Name or path of the model.
            device: Device to run pipeline on.
            **kwargs: Additional arguments for pipeline.

        Returns:
            SummarizationPipeline: Created pipeline.
        """
        if model_name in self.summarization_pipelines:
            return self.summarization_pipelines[model_name]
        
        pipeline = self.model_manager.create_pipeline(
            task="summarization",
            model_name=model_name,
            device=device,
            **kwargs
        )
        
        self.summarization_pipelines[model_name] = pipeline
        return pipeline

    def summarize_text(
        self,
        model_name: str,
        text: str,
        max_length: int = 150,
        min_length: int = 50,
        **kwargs
    ) -> str:
        """Summarize text using pre-trained model.

        Args:
            model_name: Name or path of the model.
            text: Text to summarize.
            max_length: Maximum summary length.
            min_length: Minimum summary length.
            **kwargs: Additional summarization parameters.

        Returns:
            str: Summarized text.
        """
        pipeline = self.create_summarization_pipeline(model_name)
        
        summarization_kwargs = {
            "max_length": max_length,
            "min_length": min_length,
            **kwargs
        }
        
        result = pipeline(text, **summarization_kwargs)
        return result[0]["summary_text"]

    def batch_summarize(
        self,
        model_name: str,
        texts: List[str],
        max_length: int = 150,
        min_length: int = 50,
        **kwargs
    ) -> List[str]:
        """Summarize multiple texts.

        Args:
            model_name: Name or path of the model.
            texts: List of texts to summarize.
            max_length: Maximum summary length.
            min_length: Minimum summary length.
            **kwargs: Additional summarization parameters.

        Returns:
            List[str]: Summarized texts.
        """
        pipeline = self.create_summarization_pipeline(model_name)
        
        summarization_kwargs = {
            "max_length": max_length,
            "min_length": min_length,
            **kwargs
        }
        
        results = []
        for text in texts:
            result = pipeline(text, **summarization_kwargs)
            results.append(result[0]["summary_text"])
        
        return results


class PreTrainedModelTrainer:
    """Trainer for fine-tuning pre-trained models."""

    def __init__(self, model_manager: PreTrainedModelManager):
        """Initialize pre-trained model trainer.

        Args:
            model_manager: Pre-trained model manager.
        """
        self.model_manager = model_manager

    def prepare_dataset(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        **kwargs
    ) -> Dataset:
        """Prepare dataset for training.

        Args:
            texts: List of input texts.
            labels: List of labels (for classification tasks).
            tokenizer_name: Name of tokenizer to use.
            max_length: Maximum sequence length.
            **kwargs: Additional dataset parameters.

        Returns:
            Dataset: Prepared dataset.
        """
        tokenizer = self.model_manager.load_tokenizer(tokenizer_name)
        
        # Tokenize texts
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"]
        }
        
        if labels is not None:
            dataset_dict["labels"] = labels
        
        return Dataset.from_dict(dataset_dict)

    def fine_tune_model(
        self,
        model_name: str,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./fine_tuned_model",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        logging_dir: str = "./logs",
        logging_steps: int = 100,
        save_steps: int = 1000,
        eval_steps: int = 1000,
        **kwargs
    ) -> Trainer:
        """Fine-tune pre-trained model.

        Args:
            model_name: Name or path of the model.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            output_dir: Output directory for model.
            num_train_epochs: Number of training epochs.
            per_device_train_batch_size: Training batch size per device.
            per_device_eval_batch_size: Evaluation batch size per device.
            learning_rate: Learning rate.
            warmup_steps: Number of warmup steps.
            weight_decay: Weight decay.
            logging_dir: Logging directory.
            logging_steps: Logging frequency.
            save_steps: Model saving frequency.
            eval_steps: Evaluation frequency.
            **kwargs: Additional training parameters.

        Returns:
            Trainer: Trained model trainer.
        """
        # Load model and tokenizer
        model = self.model_manager.load_model(model_name)
        tokenizer = self.model_manager.load_tokenizer(model_name)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=logging_dir,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset is not None else False,
            **kwargs
        )
        
        # Create data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        return trainer


def create_pretrained_model_manager(cache_dir: Optional[str] = None) -> PreTrainedModelManager:
    """Create pre-trained model manager.

    Args:
        cache_dir: Directory to cache models and tokenizers.

    Returns:
        PreTrainedModelManager: Created manager.
    """
    return PreTrainedModelManager(cache_dir=cache_dir)


def get_available_models() -> Dict[str, List[str]]:
    """Get list of available pre-trained models.

    Returns:
        Dict[str, List[str]]: Available models by task.
    """
    return {
        "text-generation": [
            "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
            "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B",
            "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"
        ],
        "text-classification": [
            "bert-base-uncased", "bert-large-uncased",
            "distilbert-base-uncased", "roberta-base", "roberta-large"
        ],
        "question-answering": [
            "bert-base-uncased", "bert-large-uncased",
            "distilbert-base-uncased", "deepset/roberta-base-squad2"
        ],
        "translation": [
            "Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-en-de",
            "Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-en-it"
        ],
        "summarization": [
            "facebook/bart-base", "facebook/bart-large",
            "t5-base", "t5-large", "google/pegasus-large"
        ],
        "token-classification": [
            "bert-base-uncased", "bert-large-uncased",
            "distilbert-base-uncased", "dbmdz/bert-large-cased-finetuned-conll03-english"
        ]
    } 