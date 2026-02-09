"""
Sistema de Fine-Tuning con LoRA
=================================
Fine-tuning eficiente de modelos transformers
"""

from typing import Dict, Any, List, Optional, Tuple
import structlog
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class PsychologicalDataset(Dataset):
    """Dataset para entrenamiento psicológico"""
    
    texts: List[str]
    labels: Dict[str, List[float]]  # Rasgos de personalidad, sentimientos, etc.
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "text": self.texts[idx],
            "labels": {k: v[idx] for k, v in self.labels.items()}
        }


class LoRATrainer:
    """
    Trainer para fine-tuning con LoRA (Low-Rank Adaptation)
    """
    
    def __init__(
        self,
        base_model_name: str = "distilbert-base-uncased",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1
    ):
        """
        Inicializar trainer
        
        Args:
            base_model_name: Nombre del modelo base
            lora_r: Rango de LoRA
            lora_alpha: Alpha de LoRA
            lora_dropout: Dropout de LoRA
        """
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModel.from_pretrained(base_model_name)
            
            # Configurar LoRA
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_lin", "v_lin"]  # Módulos objetivo
            )
            
            self.model = get_peft_model(base_model, lora_config)
            
            logger.info("LoRA model initialized", model=base_model_name)
        except ImportError:
            logger.warning("peft library not available, LoRA disabled")
        except Exception as e:
            logger.error("Error initializing LoRA model", error=str(e))
    
    def prepare_dataset(
        self,
        texts: List[str],
        labels: Dict[str, List[float]]
    ) -> Dataset:
        """
        Preparar dataset
        
        Args:
            texts: Lista de textos
            labels: Etiquetas
            
        Returns:
            Dataset preparado
        """
        return PsychologicalDataset(texts, labels)
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        output_dir: str = "./models/finetuned"
    ) -> Dict[str, Any]:
        """
        Entrenar modelo
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación (opcional)
            num_epochs: Número de épocas
            batch_size: Tamaño de batch
            learning_rate: Tasa de aprendizaje
            output_dir: Directorio de salida
            
        Returns:
            Métricas de entrenamiento
        """
        if self.model is None:
            return {"error": "Model not initialized"}
        
        try:
            # Argumentos de entrenamiento
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="epoch" if val_dataset else "no",
                load_best_model_at_end=True if val_dataset else False,
                fp16=torch.cuda.is_available(),  # Mixed precision
                dataloader_num_workers=2
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True
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
            
            # Entrenar
            train_result = trainer.train()
            
            # Guardar modelo
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info("Model fine-tuned successfully", output_dir=output_dir)
            
            return {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics.get("train_runtime", 0),
                "output_dir": output_dir
            }
            
        except Exception as e:
            logger.error("Error during fine-tuning", error=str(e))
            return {"error": str(e)}


class ModelEvaluator:
    """Evaluador de modelos"""
    
    def __init__(self, model, tokenizer):
        """
        Inicializar evaluador
        
        Args:
            model: Modelo a evaluar
            tokenizer: Tokenizador
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        texts: List[str],
        true_labels: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Evaluar modelo
        
        Args:
            texts: Textos de prueba
            true_labels: Etiquetas verdaderas
            
        Returns:
            Métricas de evaluación
        """
        predictions = {}
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # Procesar outputs según el modelo
                # (simplificado)
        
        # Calcular métricas
        metrics = {}
        for label_name, true_values in true_labels.items():
            if label_name in predictions:
                pred_values = predictions[label_name]
                # MSE, MAE, etc.
                mse = np.mean((np.array(true_values) - np.array(pred_values))**2)
                mae = np.mean(np.abs(np.array(true_values) - np.array(pred_values)))
                metrics[f"{label_name}_mse"] = float(mse)
                metrics[f"{label_name}_mae"] = float(mae)
        
        return metrics


# Instancia global del trainer
lora_trainer = LoRATrainer()




