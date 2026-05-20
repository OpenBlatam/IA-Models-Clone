"""
Modular Integration with HuggingFace Transformers
Provides seamless integration with pre-trained models
"""

from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoFeatureExtractor,
        AutoConfig,
        Wav2Vec2Model,
        Wav2Vec2Processor,
        PreTrainedModel,
        PreTrainedTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class HuggingFaceModelWrapper(nn.Module):
    """
    Wrapper for HuggingFace models with modular integration
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "feature_extraction",
        freeze_base: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required")
        
        self.model_name = model_name
        self.task_type = task_type
        self.device = device
        
        # Load model and tokenizer/processor
        try:
            self.base_model = AutoModel.from_pretrained(model_name)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except:
                self.tokenizer = None
            
            try:
                self.processor = AutoFeatureExtractor.from_pretrained(model_name)
            except:
                self.processor = None
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
        
        # Freeze base model if requested
        if freeze_base:
            self._freeze_base_model()
        
        # Move to device
        self.base_model = self.base_model.to(device)
    
    def _freeze_base_model(self):
        """Freeze all parameters in base model"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Base model frozen")
    
    def unfreeze_layers(self, num_layers: int = 1):
        """Unfreeze last N layers"""
        if hasattr(self.base_model, 'encoder'):
            layers = self.base_model.encoder.layer
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(f"Unfroze last {num_layers} layers")
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through base model"""
        outputs = self.base_model(**inputs)
        
        # Extract features (pooler_output or last_hidden_state)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # Mean pooling
            return outputs.last_hidden_state.mean(dim=1)
        else:
            return outputs[0]


class TransformerMusicEncoder(nn.Module):
    """
    Music encoder using pre-trained transformer with task-specific head
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        task_type: str = "classification",
        freeze_base: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        # Wrap HuggingFace model
        self.base_model = HuggingFaceModelWrapper(
            model_name=model_name,
            task_type=task_type,
            freeze_base=freeze_base,
            device=device
        )
        
        # Get hidden size from config
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # Task-specific head
        if task_type == "classification":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_classes)
            )
        elif task_type == "regression":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)
            )
        else:
            self.head = nn.Identity()
        
        self._init_head()
    
    def _init_head(self):
        """Initialize task head"""
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass"""
        # Get base model features
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs
        }
        features = self.base_model(inputs)
        
        # Apply task head
        output = self.head(features)
        return output


class LoRATransformerWrapper(nn.Module):
    """
    Wrapper for LoRA fine-tuning of transformers
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "classification",
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required")
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            PEFT_AVAILABLE = True
        except ImportError:
            PEFT_AVAILABLE = False
            raise ImportError("PEFT library required for LoRA")
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Default target modules
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
        
        # LoRA configuration
        peft_task_type = (
            TaskType.FEATURE_EXTRACTION if task_type == "feature_extraction"
            else TaskType.CLASSIFICATION
        )
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=peft_task_type
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        logger.info(f"Applied LoRA with r={r}, alpha={lora_alpha}")
    
    def forward(self, **inputs) -> torch.Tensor:
        """Forward pass"""
        outputs = self.model(**inputs)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state.mean(dim=1)
        else:
            return outputs[0]



