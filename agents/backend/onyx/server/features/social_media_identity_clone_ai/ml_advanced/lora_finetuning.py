"""
Fine-tuning con LoRA para personalización de modelos
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import json

from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuración de LoRA"""
    r: int = 8  # Rank
    lora_alpha: int = 16
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class LoRAFineTuner:
    """Fine-tuner usando LoRA para personalización"""
    
    def __init__(self):
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
    
    def prepare_model_for_lora(
        self,
        base_model_name: str = "gpt2",
        lora_config: Optional[LoRAConfig] = None
    ) -> tuple:
        """
        Prepara modelo para fine-tuning con LoRA
        
        Args:
            base_model_name: Nombre del modelo base
            lora_config: Configuración de LoRA
            
        Returns:
            Tupla (model, tokenizer)
        """
        try:
            logger.info(f"Cargando modelo base: {base_model_name}")
            
            # Cargar tokenizer y modelo
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Configurar LoRA
            if lora_config is None:
                lora_config = LoRAConfig()
            
            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules or ["c_attn", "c_proj"],
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                task_type=TaskType.CAUSAL_LM
            )
            
            # Aplicar LoRA
            model = get_peft_model(model, peft_config)
            model = model.to(self.device)
            
            self.model = model
            
            logger.info("Modelo preparado para LoRA fine-tuning")
            logger.info(f"Parámetros entrenables: {model.num_parameters()}")
            
            return model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error preparando modelo para LoRA: {e}", exc_info=True)
            raise
    
    def prepare_dataset(
        self,
        texts: List[str],
        max_length: int = 512
    ) -> torch.utils.data.Dataset:
        """
        Prepara dataset para entrenamiento
        
        Args:
            texts: Lista de textos
            max_length: Longitud máxima
            
        Returns:
            Dataset preparado
        """
        class TextDataset(torch.utils.data.Dataset):
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
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten()
                }
        
        return TextDataset(texts, self.tokenizer, max_length)
    
    def fine_tune(
        self,
        train_texts: List[str],
        validation_texts: Optional[List[str]] = None,
        output_dir: str = "./models/finetuned",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 4
    ) -> Dict[str, Any]:
        """
        Fine-tune modelo con LoRA
        
        Args:
            train_texts: Textos de entrenamiento
            validation_texts: Textos de validación
            output_dir: Directorio de salida
            num_epochs: Número de épocas
            batch_size: Tamaño de batch
            learning_rate: Learning rate
            gradient_accumulation_steps: Pasos de acumulación de gradiente
            
        Returns:
            Resultados del entrenamiento
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Modelo no preparado. Llama prepare_model_for_lora primero")
            
            # Preparar datasets
            train_dataset = self.prepare_dataset(train_texts)
            val_dataset = self.prepare_dataset(validation_texts) if validation_texts else None
            
            # Configurar entrenamiento
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                fp16=self.device == "cuda",
                logging_steps=10,
                save_steps=100,
                evaluation_strategy="epoch" if val_dataset else "no",
                save_total_limit=3,
                load_best_model_at_end=True if val_dataset else False,
                report_to="none"  # Cambiar a "wandb" o "tensorboard" si se desea
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator
            )
            
            # Entrenar
            logger.info("Iniciando fine-tuning...")
            train_result = trainer.train()
            
            # Guardar modelo
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info("Fine-tuning completado")
            
            return {
                "success": True,
                "train_loss": train_result.training_loss,
                "output_dir": output_dir,
                "epochs": num_epochs
            }
            
        except Exception as e:
            logger.error(f"Error en fine-tuning: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Genera texto usando modelo fine-tuned
        
        Args:
            prompt: Prompt inicial
            max_length: Longitud máxima
            temperature: Temperature para sampling
            top_p: Top-p sampling
            
        Returns:
            Texto generado
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Modelo no cargado")
        
        try:
            self.model.eval()
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generando texto: {e}", exc_info=True)
            raise


# Singleton global
_lora_finetuner: Optional[LoRAFineTuner] = None


def get_lora_finetuner() -> LoRAFineTuner:
    """Obtiene instancia singleton del fine-tuner"""
    global _lora_finetuner
    if _lora_finetuner is None:
        _lora_finetuner = LoRAFineTuner()
    return _lora_finetuner




