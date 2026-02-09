#!/usr/bin/env python3
"""
Advanced Multi-Modal Learning Support System for Frontier Model Training
Provides comprehensive multi-modal data processing, fusion, and learning capabilities.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
from PIL import Image
import librosa
import soundfile as sf
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import transformers
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import clip
import sentence_transformers
from sentence_transformers import SentenceTransformer
import openai
import whisper
import torchaudio
import torchaudio.transforms as T
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec, Doc2Vec
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class ModalityType(Enum):
    """Modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"
    GRAPH = "graph"
    TIME_SERIES = "time_series"
    POINT_CLOUD = "point_cloud"
    MULTIMODAL = "multimodal"

class FusionStrategy(Enum):
    """Fusion strategies."""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    INTERMEDIATE_FUSION = "intermediate_fusion"
    ATTENTION_FUSION = "attention_fusion"
    CROSS_MODAL_ATTENTION = "cross_modal_attention"
    HIERARCHICAL_FUSION = "hierarchical_fusion"
    ADAPTIVE_FUSION = "adaptive_fusion"

class AlignmentMethod(Enum):
    """Alignment methods."""
    CONTRASTIVE_LEARNING = "contrastive_learning"
    TRIPLET_LOSS = "triplet_loss"
    COSINE_SIMILARITY = "cosine_similarity"
    CROSS_MODAL_TRANSFORMER = "cross_modal_transformer"
    VISION_LANGUAGE_PRETRAINING = "vision_language_pretraining"

@dataclass
class MultiModalConfig:
    """Multi-modal configuration."""
    modalities: List[ModalityType] = None
    fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION
    alignment_method: AlignmentMethod = AlignmentMethod.CONTRASTIVE_LEARNING
    embedding_dim: int = 512
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_sequence_length: int = 512
    image_size: int = 224
    audio_sample_rate: int = 16000
    device: str = "auto"
    enable_pretrained_models: bool = True
    enable_fine_tuning: bool = True
    enable_data_augmentation: bool = True
    enable_contrastive_learning: bool = True

@dataclass
class MultiModalData:
    """Multi-modal data sample."""
    sample_id: str
    modalities: Dict[ModalityType, Any]
    labels: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class MultiModalModel:
    """Multi-modal model."""
    model_id: str
    name: str
    modalities: List[ModalityType]
    fusion_strategy: FusionStrategy
    architecture: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: datetime

class TextProcessor:
    """Text processing module."""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer and model
        if config.enable_pretrained_models:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.model = AutoModel.from_pretrained("bert-base-uncased")
            except Exception as e:
                self.logger.warning(f"Failed to load BERT: {e}")
                self.tokenizer = None
                self.model = None
        else:
            self.tokenizer = None
            self.model = None
        
        # Initialize sentence transformer
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text data."""
        processed = {
            'raw_text': text,
            'tokens': [],
            'embeddings': None,
            'features': {}
        }
        
        # Tokenization
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(text)
            processed['tokens'] = tokens
            
            # Get embeddings
            inputs = self.tokenizer(text, return_tensors='pt', 
                                  max_length=self.config.max_sequence_length,
                                  padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Pooled embeddings
                processed['embeddings'] = embeddings.numpy()
        
        # Sentence transformer embeddings
        if self.sentence_transformer:
            sentence_embeddings = self.sentence_transformer.encode(text)
            processed['sentence_embeddings'] = sentence_embeddings
        
        # Extract features
        processed['features'] = self._extract_text_features(text)
        
        return processed
    
    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract text features."""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(text.split('.')),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'punctuation_count': sum(1 for c in text if c in '.,!?;:'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        }
        
        return features

class ImageProcessor:
    """Image processing module."""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize pretrained models
        if config.enable_pretrained_models:
            try:
                self.resnet = torchvision.models.resnet50(pretrained=True)
                self.resnet.eval()
            except Exception as e:
                self.logger.warning(f"Failed to load ResNet: {e}")
                self.resnet = None
            
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
            except Exception as e:
                self.logger.warning(f"Failed to load CLIP: {e}")
                self.clip_model = None
                self.clip_preprocess = None
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image data."""
        processed = {
            'image_path': image_path,
            'tensor': None,
            'embeddings': {},
            'features': {}
        }
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Convert to tensor
            tensor = self.transform(image)
            processed['tensor'] = tensor
            
            # Extract embeddings
            if self.resnet:
                with torch.no_grad():
                    resnet_features = self.resnet(tensor.unsqueeze(0))
                    processed['embeddings']['resnet'] = resnet_features.numpy()
            
            if self.clip_model and self.clip_preprocess:
                with torch.no_grad():
                    clip_input = self.clip_preprocess(image).unsqueeze(0)
                    clip_features = self.clip_model.encode_image(clip_input)
                    processed['embeddings']['clip'] = clip_features.numpy()
            
            # Extract features
            processed['features'] = self._extract_image_features(image)
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            processed['error'] = str(e)
        
        return processed
    
    def _extract_image_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract image features."""
        features = {
            'width': image.width,
            'height': image.height,
            'aspect_ratio': image.width / image.height,
            'total_pixels': image.width * image.height,
            'brightness': np.mean(np.array(image)),
            'contrast': np.std(np.array(image))
        }
        
        # Convert to numpy for additional features
        img_array = np.array(image)
        
        # Color features
        if len(img_array.shape) == 3:
            features['red_mean'] = np.mean(img_array[:, :, 0])
            features['green_mean'] = np.mean(img_array[:, :, 1])
            features['blue_mean'] = np.mean(img_array[:, :, 2])
        
        return features

class AudioProcessor:
    """Audio processing module."""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize audio transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.audio_sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        
        # Initialize pretrained models
        if config.enable_pretrained_models:
            try:
                self.whisper_model = whisper.load_model("base")
            except Exception as e:
                self.logger.warning(f"Failed to load Whisper: {e}")
                self.whisper_model = None
    
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process audio data."""
        processed = {
            'audio_path': audio_path,
            'waveform': None,
            'spectrogram': None,
            'embeddings': {},
            'features': {}
        }
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != self.config.audio_sample_rate:
                resampler = T.Resample(sample_rate, self.config.audio_sample_rate)
                waveform = resampler(waveform)
            
            processed['waveform'] = waveform
            processed['sample_rate'] = self.config.audio_sample_rate
            
            # Generate spectrogram
            spectrogram = self.mel_transform(waveform)
            processed['spectrogram'] = spectrogram
            
            # Extract embeddings
            if self.whisper_model:
                with torch.no_grad():
                    whisper_features = self.whisper_model.encode_audio(waveform)
                    processed['embeddings']['whisper'] = whisper_features.numpy()
            
            # Extract features
            processed['features'] = self._extract_audio_features(waveform, spectrogram)
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            processed['error'] = str(e)
        
        return processed
    
    def _extract_audio_features(self, waveform: torch.Tensor, spectrogram: torch.Tensor) -> Dict[str, float]:
        """Extract audio features."""
        features = {
            'duration': waveform.shape[1] / self.config.audio_sample_rate,
            'sample_rate': self.config.audio_sample_rate,
            'amplitude_mean': torch.mean(torch.abs(waveform)).item(),
            'amplitude_std': torch.std(waveform).item(),
            'zero_crossing_rate': self._calculate_zcr(waveform),
            'spectral_centroid': self._calculate_spectral_centroid(spectrogram),
            'spectral_rolloff': self._calculate_spectral_rolloff(spectrogram),
            'mfcc_mean': torch.mean(spectrogram).item(),
            'mfcc_std': torch.std(spectrogram).item()
        }
        
        return features
    
    def _calculate_zcr(self, waveform: torch.Tensor) -> float:
        """Calculate zero crossing rate."""
        diff = torch.diff(torch.sign(waveform))
        zcr = torch.sum(torch.abs(diff)) / (2 * waveform.shape[1])
        return zcr.item()
    
    def _calculate_spectral_centroid(self, spectrogram: torch.Tensor) -> float:
        """Calculate spectral centroid."""
        freqs = torch.linspace(0, self.config.audio_sample_rate // 2, spectrogram.shape[1])
        centroid = torch.sum(freqs * spectrogram.mean(dim=0)) / torch.sum(spectrogram.mean(dim=0))
        return centroid.item()
    
    def _calculate_spectral_rolloff(self, spectrogram: torch.Tensor) -> float:
        """Calculate spectral rolloff."""
        cumsum = torch.cumsum(spectrogram.mean(dim=0), dim=0)
        threshold = 0.85 * cumsum[-1]
        rolloff_idx = torch.where(cumsum >= threshold)[0]
        if len(rolloff_idx) > 0:
            freqs = torch.linspace(0, self.config.audio_sample_rate // 2, spectrogram.shape[1])
            return freqs[rolloff_idx[0]].item()
        return 0.0

class VideoProcessor:
    """Video processing module."""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize image processor for frames
        self.image_processor = ImageProcessor(config)
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video data."""
        processed = {
            'video_path': video_path,
            'frames': [],
            'embeddings': {},
            'features': {}
        }
        
        try:
            # Extract frames
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            processed['frames'] = frames
            
            # Process frames
            frame_features = []
            for i, frame in enumerate(frames[:10]):  # Limit to first 10 frames
                # Convert to PIL Image
                frame_pil = Image.fromarray(frame)
                
                # Process frame
                frame_processed = self.image_processor.process_image(frame_pil)
                frame_features.append(frame_processed['features'])
            
            # Aggregate frame features
            processed['features'] = self._aggregate_frame_features(frame_features)
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            processed['error'] = str(e)
        
        return processed
    
    def _aggregate_frame_features(self, frame_features: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate features across frames."""
        if not frame_features:
            return {}
        
        aggregated = {}
        
        # Get all feature keys
        all_keys = set()
        for features in frame_features:
            all_keys.update(features.keys())
        
        # Aggregate each feature
        for key in all_keys:
            values = [f.get(key, 0) for f in frame_features]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated

class TabularProcessor:
    """Tabular data processing module."""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_tabular(self, data: np.ndarray, column_names: List[str] = None) -> Dict[str, Any]:
        """Process tabular data."""
        processed = {
            'data': data,
            'column_names': column_names or [f'feature_{i}' for i in range(data.shape[1])],
            'embeddings': None,
            'features': {}
        }
        
        # Create embeddings using simple MLP
        embeddings = self._create_tabular_embeddings(data)
        processed['embeddings'] = embeddings
        
        # Extract features
        processed['features'] = self._extract_tabular_features(data)
        
        return processed
    
    def _create_tabular_embeddings(self, data: np.ndarray) -> np.ndarray:
        """Create embeddings for tabular data."""
        # Simple linear projection
        embedding_dim = self.config.embedding_dim
        input_dim = data.shape[1]
        
        # Create random projection matrix
        projection_matrix = np.random.randn(input_dim, embedding_dim)
        embeddings = np.dot(data, projection_matrix)
        
        return embeddings
    
    def _extract_tabular_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract tabular features."""
        features = {
            'num_rows': data.shape[0],
            'num_columns': data.shape[1],
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'missing_values': np.isnan(data).sum(),
            'correlation_mean': np.mean(np.corrcoef(data.T))
        }
        
        return features

class GraphProcessor:
    """Graph processing module."""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_graph(self, graph: nx.Graph) -> Dict[str, Any]:
        """Process graph data."""
        processed = {
            'graph': graph,
            'embeddings': {},
            'features': {}
        }
        
        # Extract graph embeddings
        processed['embeddings'] = self._extract_graph_embeddings(graph)
        
        # Extract graph features
        processed['features'] = self._extract_graph_features(graph)
        
        return processed
    
    def _extract_graph_embeddings(self, graph: nx.Graph) -> Dict[str, np.ndarray]:
        """Extract graph embeddings."""
        embeddings = {}
        
        # Node embeddings using random walk
        if len(graph.nodes()) > 0:
            node_embeddings = np.random.randn(len(graph.nodes()), self.config.embedding_dim)
            embeddings['nodes'] = node_embeddings
        
        # Graph-level embeddings
        graph_embeddings = np.random.randn(self.config.embedding_dim)
        embeddings['graph'] = graph_embeddings
        
        return embeddings
    
    def _extract_graph_features(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract graph features."""
        features = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'average_clustering': nx.average_clustering(graph),
            'transitivity': nx.transitivity(graph)
        }
        
        # Additional features if graph is connected
        if nx.is_connected(graph):
            features['diameter'] = nx.diameter(graph)
            features['radius'] = nx.radius(graph)
            features['average_shortest_path_length'] = nx.average_shortest_path_length(graph)
        
        return features

class MultiModalFusion:
    """Multi-modal fusion module."""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize fusion layers
        self._init_fusion_layers()
    
    def _init_fusion_layers(self):
        """Initialize fusion layers based on strategy."""
        if self.config.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout
            )
        elif self.config.fusion_strategy == FusionStrategy.CROSS_MODAL_ATTENTION:
            self.cross_modal_attention = nn.ModuleDict({
                'text_to_image': nn.MultiheadAttention(
                    embed_dim=self.config.embedding_dim,
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout
                ),
                'image_to_text': nn.MultiheadAttention(
                    embed_dim=self.config.embedding_dim,
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout
                )
            })
        
        # Projection layers
        self.projection_layers = nn.ModuleDict({
            modality.value: nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
            for modality in self.config.modalities
        })
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.config.embedding_dim * len(self.config.modalities), self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.embedding_dim)
        )
    
    def fuse_modalities(self, modality_embeddings: Dict[ModalityType, torch.Tensor]) -> torch.Tensor:
        """Fuse embeddings from different modalities."""
        # Project embeddings to same dimension
        projected_embeddings = {}
        for modality, embedding in modality_embeddings.items():
            if modality.value in self.projection_layers:
                projected = self.projection_layers[modality.value](embedding)
                projected_embeddings[modality] = projected
        
        if self.config.fusion_strategy == FusionStrategy.EARLY_FUSION:
            # Concatenate embeddings
            fused = torch.cat(list(projected_embeddings.values()), dim=-1)
            fused = self.fusion_layer(fused)
            
        elif self.config.fusion_strategy == FusionStrategy.LATE_FUSION:
            # Average embeddings
            fused = torch.mean(torch.stack(list(projected_embeddings.values())), dim=0)
            
        elif self.config.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            # Use attention mechanism
            embeddings_list = list(projected_embeddings.values())
            if len(embeddings_list) > 1:
                # Self-attention across modalities
                fused, _ = self.attention_fusion(
                    embeddings_list[0].unsqueeze(0),
                    embeddings_list[1].unsqueeze(0),
                    embeddings_list[1].unsqueeze(0)
                )
                fused = fused.squeeze(0)
            else:
                fused = embeddings_list[0]
            
        elif self.config.fusion_strategy == FusionStrategy.CROSS_MODAL_ATTENTION:
            # Cross-modal attention
            if ModalityType.TEXT in projected_embeddings and ModalityType.IMAGE in projected_embeddings:
                text_emb = projected_embeddings[ModalityType.TEXT].unsqueeze(0)
                image_emb = projected_embeddings[ModalityType.IMAGE].unsqueeze(0)
                
                # Text attends to image
                text_attended, _ = self.cross_modal_attention['text_to_image'](
                    text_emb, image_emb, image_emb
                )
                
                # Image attends to text
                image_attended, _ = self.cross_modal_attention['image_to_text'](
                    image_emb, text_emb, text_emb
                )
                
                # Combine attended representations
                fused = torch.cat([text_attended.squeeze(0), image_attended.squeeze(0)], dim=-1)
                fused = self.fusion_layer(fused)
            else:
                fused = torch.mean(torch.stack(list(projected_embeddings.values())), dim=0)
        
        else:
            # Default: average fusion
            fused = torch.mean(torch.stack(list(projected_embeddings.values())), dim=0)
        
        return fused

class MultiModalModel(nn.Module):
    """Multi-modal neural network model."""
    
    def __init__(self, config: MultiModalConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Initialize processors
        self.text_processor = TextProcessor(config)
        self.image_processor = ImageProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.video_processor = VideoProcessor(config)
        self.tabular_processor = TabularProcessor(config)
        self.graph_processor = GraphProcessor(config)
        
        # Initialize fusion
        self.fusion = MultiModalFusion(config)
        
        # Initialize classifiers
        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_classes)
        )
        
        # Contrastive learning components
        if config.enable_contrastive_learning:
            self.contrastive_head = nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.embedding_dim)
            )
    
    def forward(self, modality_data: Dict[ModalityType, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        modality_embeddings = {}
        
        # Process each modality
        for modality, data in modality_data.items():
            if modality == ModalityType.TEXT:
                processed = self.text_processor.process_text(data)
                if processed['embeddings'] is not None:
                    modality_embeddings[modality] = torch.FloatTensor(processed['embeddings'])
            
            elif modality == ModalityType.IMAGE:
                processed = self.image_processor.process_image(data)
                if 'resnet' in processed['embeddings']:
                    modality_embeddings[modality] = torch.FloatTensor(processed['embeddings']['resnet'])
            
            elif modality == ModalityType.AUDIO:
                processed = self.audio_processor.process_audio(data)
                if 'whisper' in processed['embeddings']:
                    modality_embeddings[modality] = torch.FloatTensor(processed['embeddings']['whisper'])
            
            elif modality == ModalityType.TABULAR:
                processed = self.tabular_processor.process_tabular(data)
                if processed['embeddings'] is not None:
                    modality_embeddings[modality] = torch.FloatTensor(processed['embeddings'])
        
        # Fuse modalities
        if modality_embeddings:
            fused_embedding = self.fusion.fuse_modalities(modality_embeddings)
            
            # Classification
            logits = self.classifier(fused_embedding)
            
            # Contrastive learning
            contrastive_output = None
            if self.config.enable_contrastive_learning:
                contrastive_output = self.contrastive_head(fused_embedding)
            
            return {
                'logits': logits,
                'fused_embedding': fused_embedding,
                'contrastive_output': contrastive_output,
                'modality_embeddings': modality_embeddings
            }
        else:
            # Return zero outputs if no embeddings
            batch_size = 1
            return {
                'logits': torch.zeros(batch_size, self.num_classes),
                'fused_embedding': torch.zeros(batch_size, self.config.embedding_dim),
                'contrastive_output': torch.zeros(batch_size, self.config.embedding_dim),
                'modality_embeddings': {}
            }

class MultiModalDataset(Dataset):
    """Multi-modal dataset."""
    
    def __init__(self, data_samples: List[MultiModalData], config: MultiModalConfig):
        self.data_samples = data_samples
        self.config = config
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # Process modalities
        processed_modalities = {}
        
        for modality, data in sample.modalities.items():
            if modality == ModalityType.TEXT:
                processed_modalities[modality] = data
            elif modality == ModalityType.IMAGE:
                processed_modalities[modality] = data
            elif modality == ModalityType.AUDIO:
                processed_modalities[modality] = data
            elif modality == ModalityType.TABULAR:
                processed_modalities[modality] = data
        
        return {
            'modalities': processed_modalities,
            'labels': sample.labels,
            'sample_id': sample.sample_id
        }

class MultiModalLearningSystem:
    """Main multi-modal learning system."""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Models and data
        self.models: Dict[str, MultiModalModel] = {}
        self.datasets: Dict[str, MultiModalDataset] = {}
        self.training_history: Dict[str, List[Dict[str, float]]] = {}
    
    def _init_database(self) -> str:
        """Initialize multi-modal database."""
        db_path = Path("./multimodal.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multimodal_models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    modalities TEXT NOT NULL,
                    fusion_strategy TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    performance_metrics TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multimodal_data (
                    sample_id TEXT PRIMARY KEY,
                    modalities TEXT NOT NULL,
                    labels TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    session_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    config TEXT NOT NULL,
                    performance_metrics TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES multimodal_models (model_id)
                )
            """)
        
        return str(db_path)
    
    def create_model(self, model_id: str, name: str, num_classes: int) -> MultiModalModel:
        """Create multi-modal model."""
        model = MultiModalModel(self.config, num_classes)
        model = model.to(self.device)
        
        self.models[model_id] = model
        
        # Save model info
        self._save_model_info(model_id, name)
        
        console.print(f"[green]Created multi-modal model: {model_id}[/green]")
        return model
    
    def create_dataset(self, dataset_id: str, data_samples: List[MultiModalData]) -> MultiModalDataset:
        """Create multi-modal dataset."""
        dataset = MultiModalDataset(data_samples, self.config)
        self.datasets[dataset_id] = dataset
        
        # Save dataset info
        self._save_dataset_info(dataset_id, data_samples)
        
        console.print(f"[green]Created multi-modal dataset: {dataset_id}[/green]")
        return dataset
    
    def train_model(self, model_id: str, dataset_id: str, 
                   num_epochs: int = 100) -> Dict[str, Any]:
        """Train multi-modal model."""
        if model_id not in self.models or dataset_id not in self.datasets:
            console.print("[red]Model or dataset not found[/red]")
            return {}
        
        model = self.models[model_id]
        dataset = self.datasets[dataset_id]
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        training_history = []
        
        console.print(f"[blue]Training model {model_id} on dataset {dataset_id}[/blue]")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            model.train()
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch['modalities'])
                logits = outputs['logits']
                
                # Calculate loss
                labels = batch['labels']
                if isinstance(labels, dict):
                    # Use first available label
                    label_key = list(labels.keys())[0]
                    label_values = labels[label_key]
                    if isinstance(label_values, list):
                        label_tensor = torch.LongTensor(label_values)
                    else:
                        label_tensor = torch.LongTensor([label_values])
                else:
                    label_tensor = torch.LongTensor([labels])
                
                loss = criterion(logits, label_tensor.to(self.device))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == label_tensor.to(self.device)).float().mean()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                num_batches += 1
            
            # Record metrics
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': avg_accuracy
            })
            
            # Log progress
            if epoch % 10 == 0:
                console.print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")
        
        # Save training history
        self.training_history[model_id] = training_history
        
        console.print(f"[green]Training completed for model {model_id}[/green]")
        
        return {
            'training_history': training_history,
            'final_loss': training_history[-1]['loss'],
            'final_accuracy': training_history[-1]['accuracy']
        }
    
    def evaluate_model(self, model_id: str, dataset_id: str) -> Dict[str, float]:
        """Evaluate multi-modal model."""
        if model_id not in self.models or dataset_id not in self.datasets:
            return {}
        
        model = self.models[model_id]
        dataset = self.datasets[dataset_id]
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(batch['modalities'])
                logits = outputs['logits']
                
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                
                labels = batch['labels']
                if isinstance(labels, dict):
                    label_key = list(labels.keys())[0]
                    label_values = labels[label_key]
                    if isinstance(label_values, list):
                        all_labels.extend(label_values)
                    else:
                        all_labels.append(label_values)
                else:
                    all_labels.append(labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def visualize_training_progress(self, model_id: str, output_path: str = None) -> str:
        """Visualize training progress."""
        if model_id not in self.training_history:
            console.print("[red]No training history found[/red]")
            return ""
        
        if output_path is None:
            output_path = f"multimodal_training_{model_id}.png"
        
        history = self.training_history[model_id]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot loss
        epochs = [h['epoch'] for h in history]
        losses = [h['loss'] for h in history]
        accuracies = [h['accuracy'] for h in history]
        
        ax1.plot(epochs, losses, 'b-', label='Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Multi-Modal Training Progress')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, accuracies, 'r-', label='Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Multi-Modal Training Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Training progress visualization saved: {output_path}[/green]")
        return output_path
    
    def _save_model_info(self, model_id: str, name: str):
        """Save model information to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO multimodal_models 
                (model_id, name, modalities, fusion_strategy, architecture, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                model_id, name,
                json.dumps([m.value for m in self.config.modalities]),
                self.config.fusion_strategy.value,
                json.dumps({'embedding_dim': self.config.embedding_dim, 'hidden_dim': self.config.hidden_dim}),
                datetime.now().isoformat()
            ))
    
    def _save_dataset_info(self, dataset_id: str, data_samples: List[MultiModalData]):
        """Save dataset information to database."""
        with sqlite3.connect(self.db_path) as conn:
            for sample in data_samples:
                conn.execute("""
                    INSERT OR REPLACE INTO multimodal_data 
                    (sample_id, modalities, labels, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    sample.sample_id,
                    json.dumps([m.value for m in sample.modalities.keys()]),
                    json.dumps(sample.labels),
                    json.dumps(sample.metadata),
                    sample.created_at.isoformat()
                ))

def main():
    """Main function for multi-modal learning CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Modal Learning System")
    parser.add_argument("--modalities", nargs="+",
                       choices=["text", "image", "audio", "video", "tabular"],
                       default=["text", "image"], help="Modalities to use")
    parser.add_argument("--fusion-strategy", type=str,
                       choices=["early_fusion", "late_fusion", "attention_fusion", "cross_modal_attention"],
                       default="attention_fusion", help="Fusion strategy")
    parser.add_argument("--embedding-dim", type=int, default=512,
                       help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden dimension")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create multi-modal configuration
    modalities = [ModalityType(modality) for modality in args.modalities]
    config = MultiModalConfig(
        modalities=modalities,
        fusion_strategy=FusionStrategy(args.fusion_strategy),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Create multi-modal learning system
    multimodal_system = MultiModalLearningSystem(config)
    
    # Create sample data
    sample_data = []
    for i in range(100):
        modalities_data = {}
        
        if ModalityType.TEXT in modalities:
            modalities_data[ModalityType.TEXT] = f"Sample text {i}"
        
        if ModalityType.IMAGE in modalities:
            modalities_data[ModalityType.IMAGE] = f"sample_image_{i}.jpg"
        
        if ModalityType.AUDIO in modalities:
            modalities_data[ModalityType.AUDIO] = f"sample_audio_{i}.wav"
        
        if ModalityType.TABULAR in modalities:
            modalities_data[ModalityType.TABULAR] = np.random.randn(10)
        
        sample = MultiModalData(
            sample_id=f"sample_{i}",
            modalities=modalities_data,
            labels={'class': i % 5},  # 5 classes
            metadata={'source': 'synthetic'},
            created_at=datetime.now()
        )
        sample_data.append(sample)
    
    # Create dataset
    dataset = multimodal_system.create_dataset("sample_dataset", sample_data)
    
    # Create model
    model = multimodal_system.create_model("sample_model", "Multi-Modal Sample Model", num_classes=5)
    
    # Train model
    training_results = multimodal_system.train_model("sample_model", "sample_dataset", args.num_epochs)
    
    # Evaluate model
    evaluation_results = multimodal_system.evaluate_model("sample_model", "sample_dataset")
    
    # Show results
    console.print(f"[green]Multi-modal learning completed[/green]")
    console.print(f"[blue]Final accuracy: {evaluation_results.get('accuracy', 0):.4f}[/blue]")
    console.print(f"[blue]Final loss: {training_results.get('final_loss', 0):.4f}[/blue]")
    
    # Create visualization
    multimodal_system.visualize_training_progress("sample_model")

if __name__ == "__main__":
    main()
