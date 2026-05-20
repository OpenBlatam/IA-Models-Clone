"""
Advanced LLM System - Sistema avanzado de LLM con Transformers
================================================================
Integración completa con Transformers, fine-tuning con LoRA, y optimizaciones
"""

import logging
import torch
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        BitsAndBytesConfig, get_linear_schedule_with_warmup
    )
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available")

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Tipos de modelos soportados"""
    CAUSAL_LM = "causal_lm"
    SEQ2SEQ = "seq2seq"
    ENCODER_DECODER = "encoder_decoder"


@dataclass
class LLMConfig:
    """Configuración de modelo LLM"""
    model_name: str
    model_type: ModelType
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    use_gpu: bool = True
    use_quantization: bool = False
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


class AdvancedLLMSystem:
    """Sistema avanzado de LLM con Transformers"""
    
    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required")
        
        self.models: Dict[str, Dict[str, Any]] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_id: str, config: LLMConfig) -> Dict[str, Any]:
        """Carga un modelo LLM"""
        try:
            # Configurar cuantización si está habilitada
            quantization_config = None
            if config.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Cargar tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Cargar modelo
            if config.model_type == ModelType.CAUSAL_LM:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if config.use_gpu else None,
                    torch_dtype=torch.float16 if config.use_gpu else torch.float32
                )
            elif config.model_type == ModelType.SEQ2SEQ:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if config.use_gpu else None,
                    torch_dtype=torch.float16 if config.use_gpu else torch.float32
                )
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            # Aplicar LoRA si está habilitado
            if config.use_lora:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM if config.model_type == ModelType.CAUSAL_LM else TaskType.SEQ2SEQ,
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
                )
                model = get_peft_model(model, peft_config)
            
            if not config.use_gpu:
                model = model.to(self.device)
            
            self.models[model_id] = {
                "model": model,
                "config": config,
                "loaded_at": datetime.now().isoformat()
            }
            self.tokenizers[model_id] = tokenizer
            
            logger.info(f"Model {model_id} loaded successfully")
            
            return {
                "model_id": model_id,
                "status": "loaded",
                "device": str(self.device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def generate_text(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Genera texto con el modelo"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")
        
        model_data = self.models[model_id]
        model = model_data["model"]
        config = model_data["config"]
        tokenizer = self.tokenizers[model_id]
        
        # Tokenizar input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Parámetros de generación
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or config.max_length,
            "temperature": kwargs.get("temperature", config.temperature),
            "top_p": kwargs.get("top_p", config.top_p),
            "top_k": kwargs.get("top_k", config.top_k),
            "do_sample": kwargs.get("do_sample", config.do_sample),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        
        # Generar
        model.eval()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        
        # Decodificar
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def prepare_training_data(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 512
    ) -> torch.utils.data.Dataset:
        """Prepara datos para entrenamiento"""
        def tokenize_function(examples):
            return tokenizer(
                examples,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
        
        tokenized = tokenize_function(texts)
        
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}
            
            def __len__(self):
                return len(self.encodings["input_ids"])
        
        return TextDataset(tokenized)
    
    def fine_tune(
        self,
        model_id: str,
        training_texts: List[str],
        validation_texts: Optional[List[str]] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        output_dir: str = "./models/finetuned"
    ) -> Dict[str, Any]:
        """Fine-tune del modelo con LoRA"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")
        
        model_data = self.models[model_id]
        model = model_data["model"]
        config = model_data["config"]
        tokenizer = self.tokenizers[model_id]
        
        # Preparar datos
        train_dataset = self.prepare_training_data(training_texts, tokenizer, config.max_length)
        val_dataset = None
        if validation_texts:
            val_dataset = self.prepare_training_data(validation_texts, tokenizer, config.max_length)
        
        # Configurar entrenamiento
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=500 if val_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            fp16=config.use_gpu and torch.cuda.is_available(),
            gradient_accumulation_steps=4,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # Entrenar
        logger.info(f"Starting fine-tuning for model {model_id}")
        train_result = trainer.train()
        
        # Guardar modelo
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        return {
            "model_id": model_id,
            "status": "fine_tuned",
            "training_loss": train_result.training_loss,
            "output_dir": output_dir,
            "epochs": num_epochs,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información del modelo"""
        if model_id not in self.models:
            return None
        
        model_data = self.models[model_id]
        model = model_data["model"]
        
        return {
            "model_id": model_id,
            "config": model_data["config"].__dict__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": str(self.device),
            "loaded_at": model_data["loaded_at"]
        }




