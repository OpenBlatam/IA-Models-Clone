"""
Multi-Modal Fusion Engine for Export IA
Advanced multi-modal learning with attention mechanisms and cross-modal alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import json
import random
from pathlib import Path
from collections import defaultdict, deque
import copy
import cv2
from PIL import Image
import librosa
import transformers
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

@dataclass
class MultiModalConfig:
    """Configuration for multi-modal fusion"""
    # Modality types
    modalities: List[str] = None  # text, image, audio, video, tabular
    
    # Fusion strategies
    fusion_strategy: str = "attention"  # early, late, attention, cross_attention, transformer
    
    # Attention mechanisms
    attention_type: str = "multi_head"  # multi_head, self_attention, cross_attention, hierarchical
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Cross-modal alignment
    enable_cross_modal_alignment: bool = True
    alignment_loss_weight: float = 1.0
    contrastive_temperature: float = 0.07
    
    # Modality-specific encoders
    text_encoder: str = "bert"  # bert, roberta, distilbert, custom
    image_encoder: str = "resnet"  # resnet, vit, efficientnet, custom
    audio_encoder: str = "wav2vec2"  # wav2vec2, mel_spectrogram, custom
    video_encoder: str = "3d_cnn"  # 3d_cnn, transformer, custom
    
    # Feature dimensions
    text_dim: int = 768
    image_dim: int = 2048
    audio_dim: int = 1024
    video_dim: int = 2048
    tabular_dim: int = 512
    
    # Fusion dimensions
    fusion_dim: int = 1024
    hidden_dim: int = 512
    output_dim: int = 256
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    
    # Data augmentation
    enable_augmentation: bool = True
    text_augmentation: bool = True
    image_augmentation: bool = True
    audio_augmentation: bool = True
    
    # Evaluation metrics
    evaluation_metrics: List[str] = None  # accuracy, f1, mse, mae, cosine_similarity

class TextEncoder(nn.Module):
    """Text encoder for multi-modal fusion"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        if config.text_encoder == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        elif config.text_encoder == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.encoder = AutoModel.from_pretrained("roberta-base")
        else:
            # Custom text encoder
            self.encoder = nn.Sequential(
                nn.Embedding(10000, 256),
                nn.LSTM(256, 512, batch_first=True),
                nn.Linear(512, config.text_dim)
            )
            
        self.projection = nn.Linear(config.text_dim, config.fusion_dim)
        
    def forward(self, text_inputs):
        """Forward pass for text encoding"""
        
        if hasattr(self, 'tokenizer'):
            # Use pre-trained tokenizer
            if isinstance(text_inputs, str):
                text_inputs = [text_inputs]
            encoded = self.tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True)
            outputs = self.encoder(**encoded)
            text_features = outputs.last_hidden_state.mean(dim=1)
        else:
            # Use custom encoder
            text_features = self.encoder(text_inputs)
            
        # Project to fusion dimension
        text_features = self.projection(text_features)
        
        return text_features

class ImageEncoder(nn.Module):
    """Image encoder for multi-modal fusion"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        if config.image_encoder == "resnet":
            import torchvision.models as models
            self.encoder = models.resnet50(pretrained=True)
            self.encoder.fc = nn.Identity()  # Remove final classification layer
        elif config.image_encoder == "vit":
            from transformers import ViTModel
            self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        else:
            # Custom CNN encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, config.image_dim)
            )
            
        self.projection = nn.Linear(config.image_dim, config.fusion_dim)
        
    def forward(self, image_inputs):
        """Forward pass for image encoding"""
        
        if hasattr(self.encoder, 'forward'):
            if self.config.image_encoder == "vit":
                outputs = self.encoder(image_inputs)
                image_features = outputs.last_hidden_state.mean(dim=1)
            else:
                image_features = self.encoder(image_inputs)
        else:
            image_features = self.encoder(image_inputs)
            
        # Project to fusion dimension
        image_features = self.projection(image_features)
        
        return image_features

class AudioEncoder(nn.Module):
    """Audio encoder for multi-modal fusion"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        if config.audio_encoder == "wav2vec2":
            from transformers import Wav2Vec2Model
            self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        else:
            # Custom audio encoder
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(256, config.audio_dim)
            )
            
        self.projection = nn.Linear(config.audio_dim, config.fusion_dim)
        
    def forward(self, audio_inputs):
        """Forward pass for audio encoding"""
        
        if hasattr(self.encoder, 'forward'):
            if self.config.audio_encoder == "wav2vec2":
                outputs = self.encoder(audio_inputs)
                audio_features = outputs.last_hidden_state.mean(dim=1)
            else:
                audio_features = self.encoder(audio_inputs)
        else:
            audio_features = self.encoder(audio_inputs)
            
        # Project to fusion dimension
        audio_features = self.projection(audio_features)
        
        return audio_features

class VideoEncoder(nn.Module):
    """Video encoder for multi-modal fusion"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        if config.video_encoder == "3d_cnn":
            # 3D CNN for video
            self.encoder = nn.Sequential(
                nn.Conv3d(3, 64, (3, 3, 3), padding=1),
                nn.ReLU(),
                nn.MaxPool3d((1, 2, 2)),
                nn.Conv3d(64, 128, (3, 3, 3), padding=1),
                nn.ReLU(),
                nn.MaxPool3d((2, 2, 2)),
                nn.Conv3d(128, 256, (3, 3, 3), padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(256, config.video_dim)
            )
        else:
            # Custom video encoder
            self.encoder = nn.Sequential(
                nn.Conv3d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(256, config.video_dim)
            )
            
        self.projection = nn.Linear(config.video_dim, config.fusion_dim)
        
    def forward(self, video_inputs):
        """Forward pass for video encoding"""
        
        video_features = self.encoder(video_inputs)
        video_features = self.projection(video_features)
        
        return video_features

class TabularEncoder(nn.Module):
    """Tabular data encoder for multi-modal fusion"""
    
    def __init__(self, config: MultiModalConfig, num_features: int):
        super().__init__()
        self.config = config
        
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.tabular_dim)
        )
        
        self.projection = nn.Linear(config.tabular_dim, config.fusion_dim)
        
    def forward(self, tabular_inputs):
        """Forward pass for tabular encoding"""
        
        tabular_features = self.encoder(tabular_inputs)
        tabular_features = self.projection(tabular_features)
        
        return tabular_features

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for multi-modal fusion"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.d_model = config.fusion_dim
        self.d_k = self.d_model // self.num_heads
        
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(self, query, key, value, mask=None):
        """Forward pass for multi-head attention"""
        
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.W_o(context)
        
        return output, attention_weights

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        
    def forward(self, modality1_features, modality2_features):
        """Forward pass for cross-modal attention"""
        
        # Cross-attention: modality1 attends to modality2
        attended_features, attention_weights = self.attention(
            modality1_features, modality2_features, modality2_features
        )
        
        return attended_features, attention_weights

class MultiModalFusion(nn.Module):
    """Multi-modal fusion module"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        self.modalities = config.modalities or ['text', 'image']
        
        # Initialize encoders
        self.encoders = nn.ModuleDict()
        
        if 'text' in self.modalities:
            self.encoders['text'] = TextEncoder(config)
        if 'image' in self.modalities:
            self.encoders['image'] = ImageEncoder(config)
        if 'audio' in self.modalities:
            self.encoders['audio'] = AudioEncoder(config)
        if 'video' in self.modalities:
            self.encoders['video'] = VideoEncoder(config)
        if 'tabular' in self.modalities:
            self.encoders['tabular'] = TabularEncoder(config, num_features=10)  # Default
            
        # Fusion mechanisms
        if config.fusion_strategy == "attention":
            self.fusion_attention = MultiHeadAttention(config)
        elif config.fusion_strategy == "cross_attention":
            self.cross_attention = CrossModalAttention(config)
        elif config.fusion_strategy == "transformer":
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.fusion_dim,
                    nhead=config.num_attention_heads,
                    dropout=config.attention_dropout
                ),
                num_layers=3
            )
            
        # Output layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.fusion_dim * len(self.modalities), config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Cross-modal alignment
        if config.enable_cross_modal_alignment:
            self.alignment_projections = nn.ModuleDict({
                modality: nn.Linear(config.fusion_dim, config.fusion_dim)
                for modality in self.modalities
            })
            
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-modal fusion"""
        
        # Encode each modality
        modality_features = {}
        for modality, encoder in self.encoders.items():
            if modality in inputs:
                modality_features[modality] = encoder(inputs[modality])
                
        # Apply cross-modal alignment
        if self.config.enable_cross_modal_alignment:
            modality_features = self._apply_cross_modal_alignment(modality_features)
            
        # Fusion strategy
        if self.config.fusion_strategy == "early":
            # Concatenate features
            fused_features = torch.cat(list(modality_features.values()), dim=-1)
            
        elif self.config.fusion_strategy == "late":
            # Average features
            fused_features = torch.stack(list(modality_features.values()), dim=0).mean(dim=0)
            
        elif self.config.fusion_strategy == "attention":
            # Attention-based fusion
            modality_tokens = torch.stack(list(modality_features.values()), dim=1)
            attended_features, attention_weights = self.fusion_attention(
                modality_tokens, modality_tokens, modality_tokens
            )
            fused_features = attended_features.mean(dim=1)
            
        elif self.config.fusion_strategy == "cross_attention":
            # Cross-modal attention
            modality_list = list(modality_features.values())
            if len(modality_list) >= 2:
                attended_features, attention_weights = self.cross_attention(
                    modality_list[0], modality_list[1]
                )
                fused_features = attended_features.mean(dim=1)
            else:
                fused_features = modality_list[0]
                
        elif self.config.fusion_strategy == "transformer":
            # Transformer-based fusion
            modality_tokens = torch.stack(list(modality_features.values()), dim=1)
            transformer_output = self.transformer(modality_tokens)
            fused_features = transformer_output.mean(dim=1)
            
        else:
            # Default: concatenation
            fused_features = torch.cat(list(modality_features.values()), dim=-1)
            
        # Final output
        output = self.fusion_layer(fused_features)
        
        return {
            'fused_features': output,
            'modality_features': modality_features,
            'attention_weights': attention_weights if 'attention_weights' in locals() else None
        }
        
    def _apply_cross_modal_alignment(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply cross-modal alignment"""
        
        aligned_features = {}
        
        for modality, features in modality_features.items():
            if modality in self.alignment_projections:
                aligned_features[modality] = self.alignment_projections[modality](features)
            else:
                aligned_features[modality] = features
                
        return aligned_features

class ContrastiveLoss(nn.Module):
    """Contrastive loss for cross-modal alignment"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        self.temperature = config.contrastive_temperature
        
    def forward(self, modality1_features, modality2_features):
        """Compute contrastive loss"""
        
        # Normalize features
        modality1_features = F.normalize(modality1_features, dim=-1)
        modality2_features = F.normalize(modality2_features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(modality1_features, modality2_features.T) / self.temperature
        
        # Positive pairs (diagonal)
        batch_size = modality1_features.size(0)
        labels = torch.arange(batch_size).to(modality1_features.device)
        
        # Contrastive loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class MultiModalFusionEngine:
    """Main Multi-Modal Fusion Engine"""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.model = MultiModalFusion(config)
        self.contrastive_loss = ContrastiveLoss(config) if config.enable_cross_modal_alignment else None
        
        # Training metrics
        self.training_metrics = defaultdict(list)
        self.evaluation_metrics = defaultdict(list)
        
    def train(self, train_dataloader, val_dataloader=None, num_epochs=None):
        """Train the multi-modal fusion model"""
        
        if num_epochs is None:
            num_epochs = self.config.max_epochs
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch['inputs'])
                
                # Compute loss
                loss = criterion(outputs['fused_features'], batch['targets'])
                
                # Add contrastive loss if enabled
                if self.contrastive_loss and len(batch['inputs']) >= 2:
                    modality_names = list(batch['inputs'].keys())
                    contrastive_loss = self.contrastive_loss(
                        outputs['modality_features'][modality_names[0]],
                        outputs['modality_features'][modality_names[1]]
                    )
                    loss += self.config.alignment_loss_weight * contrastive_loss
                    
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Compute accuracy (simplified)
                with torch.no_grad():
                    predictions = outputs['fused_features']
                    targets = batch['targets']
                    accuracy = torch.mean((torch.abs(predictions - targets) < 0.1).float())
                    train_accuracy += accuracy.item()
                    
            # Validation
            if val_dataloader is not None:
                val_loss, val_accuracy = self._evaluate(val_dataloader, criterion)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                           f"Train Acc = {train_accuracy:.4f}, "
                           f"Val Loss = {val_loss:.4f}, "
                           f"Val Acc = {val_accuracy:.4f}")
                           
                # Store metrics
                self.training_metrics['epoch'].append(epoch)
                self.training_metrics['train_loss'].append(train_loss)
                self.training_metrics['train_accuracy'].append(train_accuracy)
                self.training_metrics['val_loss'].append(val_loss)
                self.training_metrics['val_accuracy'].append(val_accuracy)
                
            else:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                           f"Train Acc = {train_accuracy:.4f}")
                           
    def _evaluate(self, dataloader, criterion):
        """Evaluate the model"""
        
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(batch['inputs'])
                loss = criterion(outputs['fused_features'], batch['targets'])
                
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = outputs['fused_features']
                targets = batch['targets']
                accuracy = torch.mean((torch.abs(predictions - targets) < 0.1).float())
                total_accuracy += accuracy.item()
                
        return total_loss / len(dataloader), total_accuracy / len(dataloader)
        
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions"""
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            return outputs['fused_features']
            
    def get_modality_importance(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get importance of each modality"""
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            
            if outputs['attention_weights'] is not None:
                # Extract attention weights
                attention_weights = outputs['attention_weights']
                modality_importance = {}
                
                for i, modality in enumerate(self.config.modalities):
                    modality_importance[modality] = attention_weights[0, i].mean().item()
                    
                return modality_importance
            else:
                # Fallback: use feature magnitudes
                modality_importance = {}
                for modality, features in outputs['modality_features'].items():
                    modality_importance[modality] = torch.norm(features).item()
                    
                return modality_importance
                
    def save_model(self, filepath: str):
        """Save the model"""
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_metrics': dict(self.training_metrics)
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load the model"""
        
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_metrics = defaultdict(list, checkpoint.get('training_metrics', {}))

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test multi-modal fusion
    print("Testing Multi-Modal Fusion Engine...")
    
    # Create config
    config = MultiModalConfig(
        modalities=['text', 'image'],
        fusion_strategy='attention',
        attention_type='multi_head',
        num_attention_heads=8,
        enable_cross_modal_alignment=True,
        text_encoder='custom',
        image_encoder='custom'
    )
    
    # Create engine
    fusion_engine = MultiModalFusionEngine(config)
    
    # Create dummy data
    batch_size = 4
    dummy_inputs = {
        'text': torch.randint(0, 1000, (batch_size, 10)),
        'image': torch.randn(batch_size, 3, 224, 224)
    }
    dummy_targets = torch.randn(batch_size, config.output_dim)
    
    # Test forward pass
    outputs = fusion_engine.model(dummy_inputs)
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    print(f"Modality features: {list(outputs['modality_features'].keys())}")
    
    # Test modality importance
    importance = fusion_engine.get_modality_importance(dummy_inputs)
    print(f"Modality importance: {importance}")
    
    # Test prediction
    predictions = fusion_engine.predict(dummy_inputs)
    print(f"Predictions shape: {predictions.shape}")
    
    print("\nMulti-modal fusion engine initialized successfully!")
























