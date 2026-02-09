"""
Custom Model Architectures

Implements custom nn.Module classes following best practices:
- Proper weight initialization
- Normalization layers
- Attention mechanisms
- Custom heads for different tasks
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """
    Custom classification head with dropout and normalization
    
    Can be attached to any transformer model for classification tasks.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize classification head
        
        Args:
            hidden_size: Size of hidden layer
            num_labels: Number of classification labels
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = dropout
        
        self.dropout_layer = nn.Dropout(dropout)
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None
        
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            hidden_states: Hidden states from transformer (batch_size, seq_len, hidden_size)
            
        Returns:
            Logits (batch_size, num_labels)
        """
        # Use [CLS] token or mean pooling
        if len(hidden_states.shape) == 3:
            # Use first token ([CLS])
            pooled = hidden_states[:, 0, :]
        else:
            pooled = hidden_states
        
        # Apply layer norm if enabled
        if self.layer_norm:
            pooled = self.layer_norm(pooled)
        
        # Dropout
        pooled = self.dropout_layer(pooled)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class RegressionHead(nn.Module):
    """
    Custom regression head
    
    For regression tasks (e.g., sentiment score prediction).
    """
    
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize regression head
        
        Args:
            hidden_size: Size of hidden layer
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.dropout_layer = nn.Dropout(dropout)
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None
        
        self.regressor = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights"""
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            hidden_states: Hidden states from transformer
            
        Returns:
            Regression output (batch_size, 1)
        """
        # Pooling
        if len(hidden_states.shape) == 3:
            pooled = hidden_states[:, 0, :]  # [CLS] token
        else:
            pooled = hidden_states
        
        # Layer norm
        if self.layer_norm:
            pooled = self.layer_norm(pooled)
        
        # Dropout
        pooled = self.dropout_layer(pooled)
        
        # Regress
        output = self.regressor(pooled)
        
        return output


class TransformerClassifier(nn.Module):
    """
    Complete transformer-based classifier
    
    Combines a transformer backbone with a custom classification head.
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_custom_head: bool = True
    ):
        """
        Initialize transformer classifier
        
        Args:
            model_name: Name of transformer model
            num_labels: Number of classification labels
            dropout: Dropout rate
            freeze_backbone: Whether to freeze transformer weights
            use_custom_head: Whether to use custom classification head
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load transformer
        self.backbone = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        if use_custom_head:
            self.classifier = ClassificationHead(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout=dropout
            )
        else:
            # Use default head from model
            self.classifier = nn.Linear(hidden_size, num_labels)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            token_type_ids: Token type ids
            
        Returns:
            Logits (batch_size, num_labels)
        """
        # Get hidden states from backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Classify
        logits = self.classifier(hidden_states)
        
        return logits


class MultiTaskModel(nn.Module):
    """
    Multi-task learning model
    
    Shares a transformer backbone with multiple task-specific heads.
    """
    
    def __init__(
        self,
        model_name: str,
        tasks: Dict[str, int],
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Initialize multi-task model
        
        Args:
            model_name: Name of transformer model
            tasks: Dictionary mapping task names to number of labels
            dropout: Dropout rate
            freeze_backbone: Whether to freeze transformer weights
        """
        super().__init__()
        self.model_name = model_name
        self.tasks = tasks
        
        # Load transformer backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create task-specific heads
        self.heads = nn.ModuleDict()
        for task_name, num_labels in tasks.items():
            self.heads[task_name] = ClassificationHead(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout=dropout
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            task: Specific task to run (None for all tasks)
            
        Returns:
            Dictionary with task outputs
        """
        # Get hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Run task-specific heads
        results = {}
        
        if task:
            # Single task
            if task in self.heads:
                results[task] = self.heads[task](hidden_states)
        else:
            # All tasks
            for task_name, head in self.heads.items():
                results[task_name] = head(hidden_states)
        
        return results


class WeightInitializer:
    """
    Utility class for proper weight initialization
    """
    
    @staticmethod
    def init_weights_xavier(module: nn.Module) -> None:
        """Initialize weights using Xavier uniform"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @staticmethod
    def init_weights_kaiming(module: nn.Module) -> None:
        """Initialize weights using Kaiming normal"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def init_weights_bert(module: nn.Module) -> None:
        """Initialize weights following BERT initialization"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    @staticmethod
    def apply_initialization(
        model: nn.Module,
        method: str = "xavier"
    ) -> None:
        """
        Apply initialization to model
        
        Args:
            model: Model to initialize
            method: Initialization method (xavier, kaiming, bert)
        """
        if method == "xavier":
            model.apply(WeightInitializer.init_weights_xavier)
        elif method == "kaiming":
            model.apply(WeightInitializer.init_weights_kaiming)
        elif method == "bert":
            model.apply(WeightInitializer.init_weights_bert)
        else:
            raise ValueError(f"Unknown initialization method: {method}")


class AttentionVisualizer(nn.Module):
    """
    Wrapper to extract and visualize attention weights
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention visualizer
        
        Args:
            model: Model with attention layers
        """
        super().__init__()
        self.model = model
        self.attention_weights = []
        self.hooks = []
    
    def register_hooks(self) -> None:
        """Register hooks to capture attention weights"""
        def hook_fn(module, input, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                self.attention_weights.append(output.attentions.detach().cpu())
        
        # Register hooks on attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
    
    def remove_hooks(self) -> None:
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def forward(self, *args, **kwargs):
        """Forward pass with attention capture"""
        self.attention_weights = []
        output = self.model(*args, **kwargs)
        return output
    
    def get_attention_weights(self) -> list:
        """Get captured attention weights"""
        return self.attention_weights















