"""
Efficient Fine-tuning System with LoRA, P-tuning, and Advanced Parameter-Efficient Methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import math
import warnings


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"  # none, all, lora_only
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class PtuningConfig:
    """Configuration for P-tuning v2."""
    num_virtual_tokens: int = 20
    token_dim: int = 768
    num_transformer_submodules: int = 2
    num_attention_heads: int = 12
    num_layers: int = 12
    encoder_hidden_size: int = 768
    prefix_projection: bool = False
    
    
@dataclass
class AdapterConfig:
    """Configuration for Adapter layers."""
    reduction_factor: int = 16
    non_linearity: str = "gelu"
    adapter_dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["attention", "feed_forward"]


class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) implementation for Linear layers."""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA parameters
        self.lora_A = Parameter(torch.randn(rank, original_layer.in_features))
        self.lora_B = Parameter(torch.zeros(original_layer.out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        self._initialize_lora_weights()
    
    def _initialize_lora_weights(self):
        """Initialize LoRA weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        # Original layer output
        original_output = self.original_layer(x)
        
        # LoRA adaptation
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into original layer (for inference)."""
        if self.rank > 0:
            # Compute LoRA weight update
            lora_weight = self.lora_B @ self.lora_A * self.scaling
            
            # Add to original weights
            self.original_layer.weight.data += lora_weight
            
            # Reset LoRA parameters
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()
    
    def get_lora_parameters(self):
        """Get LoRA parameters for optimization."""
        return [self.lora_A, self.lora_B]


class LoRAAttention(nn.Module):
    """LoRA adaptation for attention layers."""
    
    def __init__(
        self,
        original_attention: nn.Module,
        config: LoRAConfig
    ):
        super().__init__()
        self.original_attention = original_attention
        self.config = config
        
        # Apply LoRA to target modules
        self.lora_layers = nn.ModuleDict()
        
        for name, module in original_attention.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in config.target_modules):
                lora_layer = LoRALinear(
                    module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout
                )
                self.lora_layers[name] = lora_layer
                
                # Replace original module
                self._replace_module(original_attention, name, lora_layer)
    
    def _replace_module(self, parent_module, module_name, new_module):
        """Replace a module in the parent module."""
        names = module_name.split('.')
        for name in names[:-1]:
            parent_module = getattr(parent_module, name)
        setattr(parent_module, names[-1], new_module)
    
    def forward(self, *args, **kwargs):
        """Forward pass through LoRA attention."""
        return self.original_attention(*args, **kwargs)
    
    def get_lora_parameters(self):
        """Get all LoRA parameters."""
        params = []
        for lora_layer in self.lora_layers.values():
            params.extend(lora_layer.get_lora_parameters())
        return params


class PrefixEncoder(nn.Module):
    """Prefix encoder for P-tuning v2."""
    
    def __init__(self, config: PtuningConfig):
        super().__init__()
        self.config = config
        
        # Prefix tokens
        self.prefix_tokens = Parameter(torch.randn(config.num_virtual_tokens, config.token_dim))
        
        # Optional prefix projection
        if config.prefix_projection:
            self.prefix_projection = nn.Sequential(
                nn.Linear(config.token_dim, config.encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(config.encoder_hidden_size, config.num_layers * 2 * config.token_dim)
            )
        else:
            self.prefix_projection = nn.Linear(
                config.token_dim, 
                config.num_layers * 2 * config.token_dim
            )
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize prefix encoder weights."""
        nn.init.normal_(self.prefix_tokens, mean=0.0, std=0.02)
        
        for module in self.prefix_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """Generate prefix representations."""
        # Expand prefix tokens for batch
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply projection
        prefix_states = self.prefix_projection(prefix_tokens)
        prefix_states = self.dropout(prefix_states)
        
        # Reshape for transformer layers
        # [batch_size, num_virtual_tokens, num_layers * 2 * token_dim]
        # -> [batch_size, num_layers, 2, num_virtual_tokens, token_dim]
        prefix_states = prefix_states.view(
            batch_size,
            self.config.num_virtual_tokens,
            self.config.num_layers,
            2,
            self.config.token_dim
        )
        prefix_states = prefix_states.permute(0, 2, 3, 1, 4)
        
        return prefix_states
    
    def get_virtual_tokens(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get virtual tokens for key and value."""
        prefix_states = self.forward(batch_size)
        
        # Split into key and value
        past_key_values = []
        for layer_idx in range(self.config.num_layers):
            key = prefix_states[:, layer_idx, 0]  # [batch_size, num_virtual_tokens, token_dim]
            value = prefix_states[:, layer_idx, 1]  # [batch_size, num_virtual_tokens, token_dim]
            past_key_values.append((key, value))
        
        return past_key_values


class AdapterLayer(nn.Module):
    """Adapter layer for efficient fine-tuning."""
    
    def __init__(
        self,
        hidden_size: int,
        reduction_factor: int = 16,
        non_linearity: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_size = hidden_size // reduction_factor
        
        # Down projection
        self.down_proj = nn.Linear(hidden_size, self.adapter_size)
        
        # Non-linearity
        if non_linearity == "gelu":
            self.activation = nn.GELU()
        elif non_linearity == "relu":
            self.activation = nn.ReLU()
        elif non_linearity == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
        
        # Up projection
        self.up_proj = nn.Linear(self.adapter_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize adapter weights."""
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapter."""
        # Down projection
        adapter_input = self.down_proj(x)
        adapter_input = self.activation(adapter_input)
        adapter_input = self.dropout(adapter_input)
        
        # Up projection
        adapter_output = self.up_proj(adapter_input)
        
        # Residual connection
        return x + adapter_output


class EfficientFineTuner:
    """Main class for efficient fine-tuning techniques."""
    
    def __init__(
        self,
        model: nn.Module,
        method: str = "lora",  # lora, ptuning, adapter, bitfit
        config: Optional[Union[LoRAConfig, PtuningConfig, AdapterConfig]] = None
    ):
        self.model = model
        self.method = method
        self.config = config or self._get_default_config(method)
        
        # Apply efficient fine-tuning
        self._apply_efficient_finetuning()
        
        # Track trainable parameters
        self.trainable_params = []
        self._collect_trainable_parameters()
    
    def _get_default_config(self, method: str):
        """Get default configuration for method."""
        if method == "lora":
            return LoRAConfig()
        elif method == "ptuning":
            return PtuningConfig()
        elif method == "adapter":
            return AdapterConfig()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _apply_efficient_finetuning(self):
        """Apply efficient fine-tuning technique."""
        if self.method == "lora":
            self._apply_lora()
        elif self.method == "ptuning":
            self._apply_ptuning()
        elif self.method == "adapter":
            self._apply_adapter()
        elif self.method == "bitfit":
            self._apply_bitfit()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _apply_lora(self):
        """Apply LoRA to the model."""
        print(f"🔧 Applying LoRA with rank {self.config.rank}")
        
        # Find and replace linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in self.config.target_modules):
                # Create LoRA layer
                lora_layer = LoRALinear(
                    module,
                    rank=self.config.rank,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout
                )
                
                # Replace module
                self._replace_module(name, lora_layer)
        
        # Freeze non-LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
    
    def _apply_ptuning(self):
        """Apply P-tuning v2 to the model."""
        print(f"🔧 Applying P-tuning v2 with {self.config.num_virtual_tokens} virtual tokens")
        
        # Add prefix encoder
        self.prefix_encoder = PrefixEncoder(self.config)
        
        # Freeze original model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _apply_adapter(self):
        """Apply Adapter layers to the model."""
        print(f"🔧 Applying Adapter layers with reduction factor {self.config.reduction_factor}")
        
        # Add adapter layers after attention and feed-forward
        for name, module in self.model.named_modules():
            if any(target in name for target in self.config.target_modules):
                # Add adapter layer
                if hasattr(module, 'hidden_size'):
                    hidden_size = module.hidden_size
                elif hasattr(module, 'config') and hasattr(module.config, 'hidden_size'):
                    hidden_size = module.config.hidden_size
                else:
                    hidden_size = 768  # Default
                
                adapter = AdapterLayer(
                    hidden_size,
                    reduction_factor=self.config.reduction_factor,
                    non_linearity=self.config.non_linearity,
                    dropout=self.config.adapter_dropout
                )
                
                # Insert adapter after module
                self._insert_adapter(name, adapter)
        
        # Freeze non-adapter parameters
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False
    
    def _apply_bitfit(self):
        """Apply BitFit (bias-only fine-tuning)."""
        print("🔧 Applying BitFit (bias-only fine-tuning)")
        
        # Freeze all parameters except biases
        for name, param in self.model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False
    
    def _replace_module(self, module_name: str, new_module: nn.Module):
        """Replace a module in the model."""
        names = module_name.split('.')
        parent = self.model
        
        for name in names[:-1]:
            parent = getattr(parent, name)
        
        setattr(parent, names[-1], new_module)
    
    def _insert_adapter(self, module_name: str, adapter: AdapterLayer):
        """Insert adapter after a module."""
        # This is a simplified implementation
        # In practice, you'd need to modify the forward pass
        names = module_name.split('.')
        parent = self.model
        
        for name in names[:-1]:
            parent = getattr(parent, name)
        
        # Add adapter as an attribute
        setattr(parent, f"{names[-1]}_adapter", adapter)
    
    def _collect_trainable_parameters(self):
        """Collect trainable parameters."""
        self.trainable_params = []
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                self.trainable_params.append(param)
                trainable_params += param.numel()
        
        # Add prefix encoder parameters if using P-tuning
        if hasattr(self, 'prefix_encoder'):
            for param in self.prefix_encoder.parameters():
                self.trainable_params.append(param)
                trainable_params += param.numel()
        
        print(f"📊 Parameter Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer."""
        return self.trainable_params
    
    def save_efficient_weights(self, filepath: str):
        """Save only the efficient fine-tuning weights."""
        if self.method == "lora":
            # Save LoRA weights
            lora_weights = {}
            for name, module in self.model.named_modules():
                if isinstance(module, LoRALinear):
                    lora_weights[name] = {
                        'lora_A': module.lora_A.data,
                        'lora_B': module.lora_B.data
                    }
            torch.save(lora_weights, filepath)
        
        elif self.method == "ptuning":
            # Save prefix encoder weights
            torch.save(self.prefix_encoder.state_dict(), filepath)
        
        elif self.method == "adapter":
            # Save adapter weights
            adapter_weights = {}
            for name, module in self.model.named_modules():
                if isinstance(module, AdapterLayer):
                    adapter_weights[name] = module.state_dict()
            torch.save(adapter_weights, filepath)
        
        elif self.method == "bitfit":
            # Save bias weights
            bias_weights = {}
            for name, param in self.model.named_parameters():
                if "bias" in name and param.requires_grad:
                    bias_weights[name] = param.data
            torch.save(bias_weights, filepath)
        
        print(f"✅ Efficient fine-tuning weights saved to: {filepath}")
    
    def load_efficient_weights(self, filepath: str):
        """Load efficient fine-tuning weights."""
        weights = torch.load(filepath, map_location='cpu')
        
        if self.method == "lora":
            # Load LoRA weights
            for name, module in self.model.named_modules():
                if isinstance(module, LoRALinear) and name in weights:
                    module.lora_A.data = weights[name]['lora_A']
                    module.lora_B.data = weights[name]['lora_B']
        
        elif self.method == "ptuning":
            # Load prefix encoder weights
            if hasattr(self, 'prefix_encoder'):
                self.prefix_encoder.load_state_dict(weights)
        
        elif self.method == "adapter":
            # Load adapter weights
            for name, module in self.model.named_modules():
                if isinstance(module, AdapterLayer) and name in weights:
                    module.load_state_dict(weights[name])
        
        elif self.method == "bitfit":
            # Load bias weights
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.data = weights[name]
        
        print(f"✅ Efficient fine-tuning weights loaded from: {filepath}")


class EfficientTrainer:
    """Trainer for efficient fine-tuning methods."""
    
    def __init__(
        self,
        model: nn.Module,
        fine_tuner: EfficientFineTuner,
        training_args: TrainingArguments
    ):
        self.model = model
        self.fine_tuner = fine_tuner
        self.training_args = training_args
        
        # Setup optimizer for trainable parameters only
        self.optimizer = torch.optim.AdamW(
            fine_tuner.get_trainable_parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay
        )
    
    def train(self, train_dataset, eval_dataset=None):
        """Train the model with efficient fine-tuning."""
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=None,  # Will be set externally
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            optimizers=(self.optimizer, None)  # optimizer, scheduler
        )
        
        # Train
        print(f"🚀 Starting efficient fine-tuning with {self.fine_tuner.method}")
        trainer.train()
        
        # Save efficient weights
        self.fine_tuner.save_efficient_weights("efficient_weights.pt")
        
        return trainer


def create_efficient_finetuner(
    model_name: str,
    method: str = "lora",
    **method_kwargs
) -> Tuple[nn.Module, EfficientFineTuner]:
    """Create model and efficient fine-tuner."""
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create configuration
    if method == "lora":
        config = LoRAConfig(**method_kwargs)
    elif method == "ptuning":
        config = PtuningConfig(**method_kwargs)
    elif method == "adapter":
        config = AdapterConfig(**method_kwargs)
    else:
        config = None
    
    # Create fine-tuner
    fine_tuner = EfficientFineTuner(model, method, config)
    
    return model, fine_tuner


# Usage example
def main():
    """Main function demonstrating efficient fine-tuning."""
    
    # Create efficient fine-tuner with LoRA
    model, fine_tuner = create_efficient_finetuner(
        model_name="gpt2",
        method="lora",
        rank=16,
        alpha=32.0,
        target_modules=["c_attn", "c_proj"]
    )
    
    print(f"✅ Efficient Fine-tuning System Ready!")
    print(f"   Method: {fine_tuner.method}")
    print(f"   Trainable parameters: {len(fine_tuner.get_trainable_parameters())}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./efficient_finetuning_output",
        learning_rate=2e-4,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        logging_steps=100,
        fp16=True
    )
    
    # Create trainer
    trainer = EfficientTrainer(model, fine_tuner, training_args)
    
    print("🎯 Ready for efficient fine-tuning!")


if __name__ == "__main__":
    main()






