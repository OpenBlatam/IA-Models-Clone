"""
Transformers Models - HuggingFace Integration
============================================

Integration with HuggingFace Transformers library for pre-trained models.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
import logging

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        AutoModelForSequenceClassification, AutoModelForSeq2SeqLM,
        CLIPTextModel, CLIPTokenizer, T5Tokenizer, T5EncoderModel,
        TrainingArguments, Trainer, DataCollatorWithPadding
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Install with: pip install transformers")

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from .base_model import BaseModel


class HuggingFaceModel(BaseModel):
    """
    Wrapper for HuggingFace pre-trained models.
    
    Supports:
    - BERT, GPT, T5, and other transformer models
    - Text classification
    - Text generation
    - Sequence-to-sequence tasks
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "classification",  # classification, generation, seq2seq
        num_labels: Optional[int] = None,
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        Initialize HuggingFace model.
        
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased')
            task_type: Type of task (classification, generation, seq2seq)
            num_labels: Number of labels for classification
            device: Target device
            torch_dtype: Model dtype (float16, bfloat16, float32)
            **kwargs: Additional model arguments
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required")
        
        super().__init__(device)
        
        self.model_name = model_name
        self.task_type = task_type
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Load model based on task type
        try:
            if task_type == "classification" and num_labels:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            elif task_type == "generation":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            elif task_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        self._initialized = True
        
        logger.info(f"✅ HuggingFace model loaded: {model_name} on {self.device}")
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: Text or list of texts
            max_length: Maximum sequence length
            padding: Whether to pad
            truncation: Whether to truncate
            return_tensors: Return format (pt, np, etc.)
        
        Returns:
            Tokenized inputs
        """
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Text or list of texts
            max_length: Maximum sequence length
        
        Returns:
            Embeddings tensor
        """
        inputs = self.tokenize(texts, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            if hasattr(outputs, 'last_hidden_state'):
                # Use [CLS] token or mean pooling
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs[0][:, 0, :]
        
        return embeddings
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def predict(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        return_probs: bool = False
    ) -> Dict[str, Any]:
        """
        Make predictions.
        
        Args:
            texts: Text or list of texts
            max_length: Maximum sequence length
            return_probs: Whether to return probabilities
        
        Returns:
            Predictions dictionary
        """
        inputs = self.tokenize(texts, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            if self.task_type == "classification":
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                result = {
                    "predictions": preds.cpu().numpy().tolist(),
                }
                
                if return_probs:
                    result["probabilities"] = probs.cpu().numpy().tolist()
                
                return result
            
            elif self.task_type == "generation":
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                generated_texts = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                return {"generated_texts": generated_texts}
            
            else:
                return {"outputs": outputs}


class CLIPTextEncoder(BaseModel):
    """
    CLIP text encoder for multi-modal tasks.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None
    ):
        """
        Initialize CLIP text encoder.
        
        Args:
            model_name: CLIP model name
            device: Target device
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required")
        
        super().__init__(device)
        
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.model = CLIPTextModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            logger.info(f"✅ CLIP model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode texts using CLIP.
        
        Args:
            texts: Text or list of texts
        
        Returns:
            Text embeddings
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.pooler_output or outputs.last_hidden_state[:, 0, :]



