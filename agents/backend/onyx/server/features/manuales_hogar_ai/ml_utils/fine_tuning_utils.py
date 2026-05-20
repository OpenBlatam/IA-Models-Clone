"""
Fine-Tuning Utils - Utilidades de Fine-Tuning
=============================================

Utilidades para fine-tuning de modelos transformers con LoRA y otras técnicas.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Intentar importar transformers
try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    _has_transformers = True
except ImportError:
    _has_transformers = False
    logger.warning("transformers library not available")


@dataclass
class FineTuningConfig:
    """Configuración de fine-tuning"""
    model_name: str = "bert-base-uncased"
    output_dir: str = "./fine_tuned_model"
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_length: int = 512
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1


class LoRALayer(nn.Module):
    """
    Capa LoRA (Low-Rank Adaptation) para fine-tuning eficiente.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1
    ):
        """
        Inicializar capa LoRA.
        
        Args:
            in_features: Features de entrada
            out_features: Features de salida
            rank: Rango de la descomposición
            alpha: Factor de escalado
            dropout: Dropout rate
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.dropout(x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora_to_linear(
    linear_layer: nn.Linear,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1
) -> nn.Module:
    """
    Aplicar LoRA a capa linear.
    
    Args:
        linear_layer: Capa linear
        rank: Rango de LoRA
        alpha: Factor de escalado
        dropout: Dropout rate
        
    Returns:
        Módulo con LoRA aplicado
    """
    lora = LoRALayer(
        linear_layer.in_features,
        linear_layer.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )
    
    # Crear wrapper que combina original + LoRA
    class LoRALinear(nn.Module):
        def __init__(self, original, lora):
            super().__init__()
            self.original = original
            self.lora = lora
        
        def forward(self, x):
            return self.original(x) + self.lora(x)
    
    return LoRALinear(linear_layer, lora)


class LoRATrainer:
    """
    Trainer para fine-tuning con LoRA.
    """
    
    def __init__(
        self,
        model_name: str,
        config: FineTuningConfig
    ):
        """
        Inicializar LoRA trainer.
        
        Args:
            model_name: Nombre del modelo
            config: Configuración de fine-tuning
        """
        if not _has_transformers:
            raise ImportError("transformers library is required for LoRA training")
        
        self.config = config
        self.model_name = model_name
        
        # Cargar modelo y tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determinar tipo de modelo
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=config.num_labels if hasattr(config, 'num_labels') else 2
            )
        except Exception:
            self.model = AutoModel.from_pretrained(model_name)
        
        # Aplicar LoRA si está habilitado
        if config.use_lora:
            self._apply_lora()
    
    def _apply_lora(self) -> None:
        """Aplicar LoRA a capas de atención del modelo"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and ("attention" in name.lower() or "query" in name.lower() or "value" in name.lower()):
                lora_module = apply_lora_to_linear(
                    module,
                    rank=self.config.lora_r,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout
                )
                # Reemplazar módulo
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = self.model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                setattr(parent, child_name, lora_module)
        
        logger.info("LoRA applied to model")
    
    def prepare_training_args(self) -> TrainingArguments:
        """
        Preparar argumentos de entrenamiento.
        
        Returns:
            TrainingArguments
        """
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=100,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False
        )
    
    def tokenize_data(self, texts: List[str], labels: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenizar datos.
        
        Args:
            texts: Lista de textos
            labels: Labels opcionales
            
        Returns:
            Diccionario tokenizado
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        if labels is not None:
            encodings["labels"] = torch.tensor(labels)
        
        return encodings




