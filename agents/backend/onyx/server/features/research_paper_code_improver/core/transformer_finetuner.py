"""
Transformer Fine-Tuner - Fine-tuning eficiente de transformers con LoRA/P-tuning
==================================================================================
"""

import logging
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuración de LoRA"""
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class FineTuningConfig:
    """Configuración de fine-tuning"""
    model_name: str
    use_lora: bool = True
    use_4bit: bool = False
    use_8bit: bool = False
    lora_config: Optional[LoRAConfig] = None
    max_length: int = 512
    output_dir: str = "./fine_tuned_models"


class TransformerFineTuner:
    """Fine-tuner de transformers con LoRA/P-tuning"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo con configuración de cuantización opcional"""
        model_kwargs = {}
        
        # Configuración de bits and bytes
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs["quantization_config"] = bnb_config
        elif self.config.use_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = bnb_config
        
        # Cargar modelo base
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Aplicar LoRA si está habilitado
        if self.config.use_lora:
            self._apply_lora()
        
        logger.info(f"Modelo {self.config.model_name} cargado")
    
    def _apply_lora(self):
        """Aplica LoRA al modelo"""
        if self.config.lora_config is None:
            lora_config = LoRAConfig()
        else:
            lora_config = self.config.lora_config
        
        # Determinar target modules si no se especifican
        if lora_config.target_modules is None:
            # Detectar módulos automáticamente
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            if hasattr(model_config, "hidden_size"):
                # Para modelos GPT-like
                lora_config.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            else:
                # Fallback genérico
                lora_config.target_modules = ["query", "value", "key"]
        
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=TaskType.CAUSAL_LM if lora_config.task_type == "CAUSAL_LM" else TaskType.SEQ_2_SEQ_LM
        )
        
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()
        logger.info("LoRA aplicado al modelo")
    
    def prepare_dataset(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """Prepara dataset para fine-tuning"""
        dataset = []
        
        for i, text in enumerate(texts):
            encoded = self.tokenizer(
                text,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            sample = {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze()
            }
            
            if labels is not None:
                sample["labels"] = labels[i]
            
            dataset.append(sample)
        
        return dataset
    
    def train(
        self,
        train_dataset: List[Dict[str, Any]],
        eval_dataset: Optional[List[Dict[str, Any]]] = None,
        training_args: Optional[Dict[str, Any]] = None
    ):
        """Entrena el modelo"""
        from transformers import DataCollatorForLanguageModeling
        
        # Argumentos de entrenamiento por defecto
        default_args = {
            "output_dir": self.config.output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "evaluation_strategy": "steps" if eval_dataset else "no",
            "eval_steps": 500 if eval_dataset else None,
            "save_total_limit": 3,
            "load_best_model_at_end": True if eval_dataset else False,
            "fp16": torch.cuda.is_available(),
        }
        
        if training_args:
            default_args.update(training_args)
        
        training_arguments = TrainingArguments(**default_args)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Para causal LM
        )
        
        # Trainer
        model_to_train = self.peft_model if self.peft_model else self.model
        
        trainer = Trainer(
            model=model_to_train,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Entrenar
        trainer.train()
        
        # Guardar modelo
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"Modelo fine-tuneado guardado en {self.config.output_dir}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Genera texto con el modelo fine-tuneado"""
        model_to_use = self.peft_model if self.peft_model else self.model
        model_to_use.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model_to_use.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text




