"""
Trainer para Fine-tuning de Modelos
====================================

Sistema de entrenamiento para fine-tuning de modelos de generación.
"""

import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import wandb
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ManualDataset(Dataset):
    """Dataset para entrenamiento de manuales."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512
    ):
        """
        Inicializar dataset.
        
        Args:
            texts: Lista de textos (problema + manual)
            tokenizer: Tokenizer
            max_length: Longitud máxima
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenizar
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


class ManualTrainer:
    """Trainer para fine-tuning de modelos."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        use_lora: bool = True,
        device: Optional[str] = None,
        use_wandb: bool = False,
        project_name: str = "manuales-hogar-ai"
    ):
        """
        Inicializar trainer.
        
        Args:
            model_name: Nombre del modelo base
            use_lora: Usar LoRA para fine-tuning
            device: Dispositivo
            use_wandb: Usar Weights & Biases
            project_name: Nombre del proyecto
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lora = use_lora
        self.use_wandb = use_wandb
        
        logger.info(f"Inicializando trainer: {model_name} en {self.device}")
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Aplicar LoRA
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA configurado")
        
        self.model.to(self.device)
        
        # Inicializar wandb
        if use_wandb:
            wandb.init(project=project_name)
    
    def prepare_dataset(
        self,
        problems: List[str],
        manuals: List[str],
        max_length: int = 512
    ) -> ManualDataset:
        """
        Preparar dataset.
        
        Args:
            problems: Lista de problemas
            manuals: Lista de manuales
            max_length: Longitud máxima
        
        Returns:
            Dataset preparado
        """
        # Combinar problema + manual
        texts = [
            f"PROBLEMA: {prob}\n\nMANUAL:\n{manual}"
            for prob, manual in zip(problems, manuals)
        ]
        
        return ManualDataset(texts, self.tokenizer, max_length)
    
    def train(
        self,
        train_dataset: ManualDataset,
        val_dataset: Optional[ManualDataset] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        output_dir: str = "./models/finetuned",
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100
    ):
        """
        Entrenar modelo.
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación (opcional)
            num_epochs: Número de épocas
            batch_size: Tamaño de batch
            learning_rate: Learning rate
            gradient_accumulation_steps: Pasos de acumulación
            output_dir: Directorio de salida
            save_steps: Pasos para guardar
            eval_steps: Pasos para evaluar
            logging_steps: Pasos para logging
        """
        try:
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                fp16=self.device == "cuda",
                logging_steps=logging_steps,
                save_steps=save_steps,
                eval_steps=eval_steps if val_dataset else None,
                evaluation_strategy="steps" if val_dataset else "no",
                save_total_limit=3,
                load_best_model_at_end=True if val_dataset else False,
                report_to="wandb" if self.use_wandb else None,
                remove_unused_columns=False
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            logger.info("Iniciando entrenamiento...")
            trainer.train()
            
            # Guardar modelo final
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Entrenamiento completado. Modelo guardado en {output_dir}")
        
        except Exception as e:
            logger.error(f"Error en entrenamiento: {str(e)}")
            raise
    
    def evaluate(
        self,
        test_dataset: ManualDataset,
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Evaluar modelo.
        
        Args:
            test_dataset: Dataset de prueba
            batch_size: Tamaño de batch
        
        Returns:
            Métricas de evaluación
        """
        try:
            self.model.eval()
            
            dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            total_loss = 0.0
            num_batches = 0
            
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Evaluando"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            metrics = {
                "loss": avg_loss,
                "perplexity": perplexity
            }
            
            logger.info(f"Métricas de evaluación: {metrics}")
            return metrics
        
        except Exception as e:
            logger.error(f"Error en evaluación: {str(e)}")
            return {}




