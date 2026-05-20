"""
Transfer Learning Service - Transfer Learning
==============================================

Sistema para aplicar transfer learning con modelos pre-entrenados.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoConfig,
    )
    import torch
    import torch.nn as nn
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")


class TransferStrategy(str, Enum):
    """Estrategias de transfer learning"""
    FEATURE_EXTRACTION = "feature_extraction"  # Freeze all, train only head
    FINE_TUNING = "fine_tuning"  # Train all layers
    LAYER_WISE = "layer_wise"  # Unfreeze layers progressively


@dataclass
class TransferLearningConfig:
    """Configuración de transfer learning"""
    base_model: str
    num_labels: int
    strategy: TransferStrategy = TransferStrategy.FINE_TUNING
    freeze_layers: Optional[List[int]] = None
    learning_rate: float = 2e-5
    use_classifier: bool = True
    dropout: float = 0.1


class TransferLearningService:
    """Servicio de transfer learning"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        logger.info("TransferLearningService initialized")
    
    def create_transfer_model(
        self,
        model_id: str,
        config: TransferLearningConfig
    ) -> Any:
        """Crear modelo con transfer learning"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available")
        
        try:
            if config.use_classifier:
                model = AutoModelForSequenceClassification.from_pretrained(
                    config.base_model,
                    num_labels=config.num_labels,
                )
            else:
                base_model = AutoModel.from_pretrained(config.base_model)
                
                # Add custom classifier
                class CustomModel(nn.Module):
                    def __init__(self, base_model, num_labels, dropout):
                        super().__init__()
                        self.base_model = base_model
                        self.dropout = nn.Dropout(dropout)
                        self.classifier = nn.Linear(
                            base_model.config.hidden_size,
                            num_labels
                        )
                    
                    def forward(self, **inputs):
                        outputs = self.base_model(**inputs)
                        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
                        pooled = self.dropout(pooled)
                        return self.classifier(pooled)
                
                model = CustomModel(base_model, config.num_labels, config.dropout)
            
            # Apply transfer strategy
            if config.strategy == TransferStrategy.FEATURE_EXTRACTION:
                # Freeze all parameters
                for param in model.parameters():
                    param.requires_grad = False
                
                # Unfreeze classifier
                if hasattr(model, 'classifier'):
                    for param in model.classifier.parameters():
                        param.requires_grad = True
                elif hasattr(model, 'score'):
                    for param in model.score.parameters():
                        param.requires_grad = True
            
            elif config.strategy == TransferStrategy.LAYER_WISE and config.freeze_layers:
                # Freeze specific layers
                for layer_idx in config.freeze_layers:
                    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                        if layer_idx < len(model.encoder.layer):
                            for param in model.encoder.layer[layer_idx].parameters():
                                param.requires_grad = False
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.base_model)
            
            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer
            
            logger.info(f"Transfer learning model {model_id} created")
            return model
            
        except Exception as e:
            logger.error(f"Error creating transfer model: {e}")
            raise
    
    def get_trainable_parameters(self, model_id: str) -> Dict[str, Any]:
        """Obtener parámetros entrenables"""
        model = self.models.get(model_id)
        if not model:
            return {"error": "Model not found"}
        
        if not TRANSFORMERS_AVAILABLE:
            return {"error": "Transformers not available"}
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
        }
    
    def unfreeze_layers(
        self,
        model_id: str,
        layer_indices: List[int]
    ) -> bool:
        """Descongelar capas específicas"""
        model = self.models.get(model_id)
        if not model:
            return False
        
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                for layer_idx in layer_indices:
                    if layer_idx < len(model.encoder.layer):
                        for param in model.encoder.layer[layer_idx].parameters():
                            param.requires_grad = True
            
            logger.info(f"Unfroze layers {layer_indices} in model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unfreezing layers: {e}")
            return False




