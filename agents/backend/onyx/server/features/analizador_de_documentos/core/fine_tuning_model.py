"""
Modelo de Fine-Tuning para Analizador de Documentos
====================================================

Sistema completo para fine-tuning de modelos de lenguaje
específicamente para análisis de documentos.
"""

import os
import logging
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuración para fine-tuning"""
    model_name: str = "bert-base-multilingual-cased"
    num_labels: int = 2
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "./fine_tuned_models"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    seed: int = 42


class DocumentDataset(Dataset):
    """Dataset para fine-tuning"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[Union[int, List[int]]],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }
        
        if isinstance(label, list):
            item["labels"] = torch.tensor(label, dtype=torch.float)
        else:
            item["labels"] = torch.tensor(label, dtype=torch.long)
        
        return item


class FineTuningModel:
    """
    Modelo de Fine-Tuning para análisis de documentos
    
    Permite entrenar modelos personalizados para tareas específicas:
    - Clasificación de documentos
    - Detección de sentimiento
    - Categorización
    - Etiquetado multi-label
    """
    
    def __init__(
        self,
        config: Optional[FineTuningConfig] = None,
        model_path: Optional[str] = None
    ):
        """
        Inicializar modelo de fine-tuning
        
        Args:
            config: Configuración de fine-tuning
            model_path: Ruta a modelo pre-entrenado (opcional)
        """
        self.config = config or FineTuningConfig()
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._initialize_model()
        
        logger.info(f"FineTuningModel inicializado en {self.device}")
    
    def _initialize_model(self):
        """Inicializar modelo y tokenizer"""
        logger.info(f"Inicializando modelo: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=os.path.join(
                Path(__file__).parent.parent.parent,
                "models",
                "cache"
            )
        )
        
        # Determinar si es clasificación o regresión
        if self.config.num_labels == 1:
            # Regresión
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=1,
                cache_dir=os.path.join(
                    Path(__file__).parent.parent.parent,
                    "models",
                    "cache"
                )
            )
        else:
            # Clasificación
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels,
                cache_dir=os.path.join(
                    Path(__file__).parent.parent.parent,
                    "models",
                    "cache"
                )
            )
        
        self.model.to(self.device)
        logger.info(f"Modelo inicializado con {self.config.num_labels} labels")
    
    def prepare_dataset(
        self,
        texts: List[str],
        labels: List[Union[int, List[int]]],
        train_split: float = 0.8
    ) -> Tuple[DocumentDataset, DocumentDataset]:
        """
        Preparar dataset para entrenamiento
        
        Args:
            texts: Lista de textos
            labels: Lista de etiquetas
            train_split: Proporción para entrenamiento
        
        Returns:
            Tupla con (train_dataset, eval_dataset)
        """
        # Dividir datos
        split_idx = int(len(texts) * train_split)
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx]
        eval_texts = texts[split_idx:]
        eval_labels = labels[split_idx:]
        
        train_dataset = DocumentDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.config.max_length
        )
        
        eval_dataset = DocumentDataset(
            eval_texts,
            eval_labels,
            self.tokenizer,
            self.config.max_length
        )
        
        logger.info(
            f"Dataset preparado: {len(train_dataset)} train, "
            f"{len(eval_dataset)} eval"
        )
        
        return train_dataset, eval_dataset
    
    def compute_metrics(self, eval_pred):
        """Calcular métricas de evaluación"""
        predictions, labels = eval_pred
        
        if self.config.num_labels == 1:
            # Regresión
            predictions = predictions.flatten()
            labels = labels.flatten()
            mse = np.mean((predictions - labels) ** 2)
            mae = np.mean(np.abs(predictions - labels))
            return {
                "mse": mse,
                "mae": mae,
                "rmse": np.sqrt(mse)
            }
        else:
            # Clasificación
            if len(labels.shape) > 1:
                # Multi-label
                predictions = (torch.sigmoid(torch.tensor(predictions)) > 0.5).numpy()
                predictions = predictions.astype(int)
            else:
                # Single-label
                predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels,
                predictions,
                average="weighted",
                zero_division=0
            )
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
    
    def train(
        self,
        train_dataset: DocumentDataset,
        eval_dataset: Optional[DocumentDataset] = None,
        custom_callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Entrenar el modelo
        
        Args:
            train_dataset: Dataset de entrenamiento
            eval_dataset: Dataset de evaluación (opcional)
            custom_callbacks: Callbacks personalizados
        
        Returns:
            Diccionario con métricas de entrenamiento
        """
        # Crear directorio de salida
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            fp16=self.config.fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            seed=self.config.seed,
            report_to="none"  # No usar wandb/tensorboard por defecto
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Callbacks
        callbacks = []
        if eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # Entrenar
        logger.info("Iniciando entrenamiento...")
        train_result = trainer.train()
        
        # Evaluar
        if eval_dataset is not None:
            logger.info("Evaluando modelo...")
            eval_result = trainer.evaluate()
            logger.info(f"Métricas de evaluación: {eval_result}")
        else:
            eval_result = {}
        
        # Guardar modelo final
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Guardar configuración
        config_path = os.path.join(self.config.output_dir, "fine_tuning_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Modelo entrenado y guardado en {self.config.output_dir}")
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            **eval_result
        }
    
    async def classify(self, text: str) -> Dict[str, float]:
        """
        Clasificar un texto usando el modelo fine-tuned
        
        Args:
            text: Texto a clasificar
        
        Returns:
            Diccionario con probabilidades por clase
        """
        if self.model is None:
            raise ValueError("Modelo no inicializado. Entrena o carga un modelo primero.")
        
        self.model.eval()
        
        # Tokenizar
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Predecir
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            
            if self.config.num_labels == 1:
                # Regresión
                prediction = logits.item()
                return {"prediction": float(prediction)}
            else:
                # Clasificación
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                probabilities = probabilities.cpu().numpy()[0]
                
                # Crear diccionario con labels
                results = {}
                for i, prob in enumerate(probabilities):
                    label = f"class_{i}"
                    results[label] = float(prob)
                
                return results
    
    def save(self, path: str):
        """
        Guardar modelo completo
        
        Args:
            path: Ruta donde guardar
        """
        os.makedirs(path, exist_ok=True)
        
        # Guardar modelo y tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Guardar configuración
        config_path = os.path.join(path, "fine_tuning_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Modelo guardado en {path}")
    
    @classmethod
    def load(cls, path: str) -> "FineTuningModel":
        """
        Cargar modelo desde disco
        
        Args:
            path: Ruta al modelo guardado
        
        Returns:
            Instancia de FineTuningModel
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado en {path}")
        
        # Cargar configuración
        config_path = os.path.join(path, "fine_tuning_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config = FineTuningConfig(**config_dict)
        else:
            config = FineTuningConfig()
        
        # Crear instancia
        instance = cls(config=config)
        
        # Cargar modelo y tokenizer
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model.to(instance.device)
        
        logger.info(f"Modelo cargado desde {path}")
        
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        return {
            "model_name": self.config.model_name,
            "num_labels": self.config.num_labels,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "max_length": self.config.max_length
        }


if __name__ == "__main__":
    # Ejemplo de uso
    config = FineTuningConfig(
        num_labels=3,
        output_dir="./test_fine_tuned_model"
    )
    
    model = FineTuningModel(config=config)
    print("Modelo de Fine-Tuning inicializado correctamente")
    print(f"Información: {model.get_model_info()}")
















