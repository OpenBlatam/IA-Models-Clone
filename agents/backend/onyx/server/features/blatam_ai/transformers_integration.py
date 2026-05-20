"""
Blatam AI - Transformers Integration Engine v6.0.0
Ultra-optimized integration with Hugging Face Transformers library
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForQuestionAnswering, AutoModelForMaskedLM,
    GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer,
    BertModel, BertTokenizer, RobertaModel, RobertaTokenizer,
    LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM, MistralTokenizer,
    GenerationConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq, PreTrainedTokenizer, PreTrainedModel,
    pipeline, TextGenerationPipeline, TranslationPipeline, SummarizationPipeline,
    QuestionAnsweringPipeline, TextClassificationPipeline, TokenClassificationPipeline,
    FillMaskPipeline, FeatureExtractionPipeline
)
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformersManager:
    """Advanced manager for Hugging Face Transformers integration."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
            
    def load_model_and_tokenizer(self, model_name: str, task: str = "auto") -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load pre-trained model and tokenizer for specific task."""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Load model based on task
            if task == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(model_name)
            elif task == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            elif task == "sequence_classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            elif task == "token_classification":
                model = AutoModelForTokenClassification.from_pretrained(model_name)
            elif task == "question_answering":
                model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            elif task == "masked_lm":
                model = AutoModelForMaskedLM.from_pretrained(model_name)
            else:
                # Auto-detect task
                model = AutoModel.from_pretrained(model_name)
                
            # Move to device
            model = model.to(self.device)
            
            # Store references
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            logger.info(f"Successfully loaded {model_name} for {task} on {self.device}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
            
    def create_pipeline(self, model_name: str, task: str, **kwargs) -> Any:
        """Create Hugging Face pipeline for specific task."""
        try:
            pipeline_obj = pipeline(
                task=task,
                model=model_name,
                device=self.device,
                **kwargs
            )
            
            self.pipelines[f"{model_name}_{task}"] = pipeline_obj
            logger.info(f"Created {task} pipeline for {model_name}")
            return pipeline_obj
            
        except Exception as e:
            logger.error(f"Error creating pipeline for {model_name}: {e}")
            raise
            
    def text_generation(self, model_name: str, prompt: str, max_length: int = 100,
                       temperature: float = 0.7, top_p: float = 0.9,
                       do_sample: bool = True, num_return_sequences: int = 1) -> List[str]:
        """Generate text using pre-trained model."""
        if model_name not in self.models:
            self.load_model_and_tokenizer(model_name, "causal_lm")
            
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts
        
    def sequence_classification(self, model_name: str, texts: List[str]) -> List[Dict[str, Any]]:
        """Perform sequence classification."""
        if model_name not in self.models:
            self.load_model_and_tokenizer(model_name, "sequence_classification")
            
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize inputs
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'text': texts[i],
                'prediction': pred.item(),
                'confidence': prob[pred].item(),
                'probabilities': prob.tolist()
            })
            
        return results
        
    def question_answering(self, model_name: str, question: str, context: str) -> Dict[str, Any]:
        """Perform question answering."""
        if model_name not in self.models:
            self.load_model_and_tokenizer(model_name, "question_answering")
            
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize inputs
        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get answer
        with torch.no_grad():
            outputs = model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            
        # Decode answer
        answer_tokens = inputs['input_ids'][0][answer_start:answer_end]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return {
            'question': question,
            'context': context,
            'answer': answer,
            'confidence': (outputs.start_logits[0][answer_start] + outputs.end_logits[0][answer_end-1]).item()
        }
        
    def text_summarization(self, model_name: str, text: str, max_length: int = 150,
                          min_length: int = 50) -> str:
        """Perform text summarization."""
        if model_name not in self.models:
            self.load_model_and_tokenizer(model_name, "seq2seq")
            
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Decode summary
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    def token_classification(self, model_name: str, text: str) -> List[Dict[str, Any]]:
        """Perform token classification (NER, POS tagging, etc.)."""
        if model_name not in self.models:
            self.load_model_and_tokenizer(model_name, "token_classification")
            
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
        # Format results
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        results = []
        
        for i, (token, pred) in enumerate(zip(tokens, predictions[0])):
            if token not in [tokenizer.pad_token, tokenizer.sep_token, tokenizer.cls_token]:
                results.append({
                    'token': token,
                    'prediction': pred.item(),
                    'position': i
                })
                
        return results
        
    def feature_extraction(self, model_name: str, texts: List[str]) -> torch.Tensor:
        """Extract features/embeddings from text."""
        if model_name not in self.models:
            self.load_model_and_tokenizer(model_name, "feature_extraction")
            
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize inputs
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use last hidden state as features
            features = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            
        return features
        
    def fine_tune_model(self, model_name: str, training_data: List[Dict[str, str]],
                       task: str, output_dir: str = "./fine_tuned_model",
                       num_epochs: int = 3, batch_size: int = 4,
                       learning_rate: float = 5e-5) -> str:
        """Fine-tune pre-trained model on custom data."""
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer(model_name, task)
        
        # Prepare dataset
        train_dataset = self._prepare_dataset(training_data, tokenizer, task)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            remove_unused_columns=False,
            push_to_hub=False
        )
        
        # Data collator
        if task == "causal_lm":
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        else:
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
            
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model fine-tuned and saved to {output_dir}")
        return output_dir
        
    def _prepare_dataset(self, training_data: List[Dict[str, str]], 
                        tokenizer: PreTrainedTokenizer, task: str) -> Any:
        """Prepare dataset for fine-tuning."""
        from torch.utils.data import Dataset
        
        class TextDataset(Dataset):
            def __init__(self, data, tokenizer, task, max_length=512):
                self.data = data
                self.tokenizer = tokenizer
                self.task = task
                self.max_length = max_length
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                item = self.data[idx]
                
                if self.task == "causal_lm":
                    text = item.get('text', '')
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    return {
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze()
                    }
                elif self.task == "seq2seq":
                    input_text = item.get('input', '')
                    target_text = item.get('target', '')
                    
                    inputs = self.tokenizer(
                        input_text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    targets = self.tokenizer(
                        target_text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': inputs['input_ids'].squeeze(),
                        'attention_mask': inputs['attention_mask'].squeeze(),
                        'labels': targets['input_ids'].squeeze()
                    }
                else:
                    # Default to causal LM
                    text = item.get('text', '')
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    return {
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze()
                    }
                    
        return TextDataset(training_data, tokenizer, task)

def main():
    """Main examples for transformers integration."""
    # Initialize manager
    manager = TransformersManager()
    
    # Example: Text generation with GPT-2
    try:
        generated_texts = manager.text_generation(
            "gpt2",
            "The future of artificial intelligence is",
            max_length=50,
            temperature=0.8
        )
        logger.info(f"Generated text: {generated_texts[0]}")
    except Exception as e:
        logger.info(f"GPT-2 example skipped: {e}")
        
    # Example: Sequence classification with BERT
    try:
        texts = ["I love this movie!", "This is terrible.", "It's okay, I guess."]
        classifications = manager.sequence_classification("bert-base-uncased", texts)
        for result in classifications:
            logger.info(f"Text: {result['text']}, Prediction: {result['prediction']}")
    except Exception as e:
        logger.info(f"BERT example skipped: {e}")
        
    print("Transformers integration engine ready!")

if __name__ == "__main__":
    main()

