"""
Advanced Data Pipeline for Export IA
Refactored data processing with PyTorch best practices and optimizations
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchaudio.transforms as audio_transforms

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import json
import pickle
from pathlib import Path
import cv2
from PIL import Image
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data pipeline"""
    # Data paths
    data_dir: str
    train_dir: str = None
    val_dir: str = None
    test_dir: str = None
    
    # Data parameters
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Image parameters
    image_size: Tuple[int, int] = (224, 224)
    image_channels: int = 3
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # Text parameters
    max_seq_length: int = 512
    vocab_size: int = 50000
    tokenizer_name: str = "bert-base-uncased"
    
    # Audio parameters
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets"""
    
    def __init__(self, config: DataConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.data = []
        self.labels = []
        self._load_data()
        
    @abstractmethod
    def _load_data(self):
        """Load and preprocess data"""
        pass
        
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index"""
        pass
        
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.data)
        
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        if not self.labels:
            return torch.ones(1)
            
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        num_classes = len(unique_labels)
        
        weights = torch.zeros(num_classes)
        for i, count in enumerate(counts):
            weights[i] = total_samples / (num_classes * count)
            
        return weights

class ImageDataset(BaseDataset):
    """Dataset for image data with advanced augmentation"""
    
    def __init__(self, config: DataConfig, split: str = "train"):
        super().__init__(config, split)
        self._setup_transforms()
        
    def _load_data(self):
        """Load image data"""
        if self.split == "train" and self.config.train_dir:
            data_dir = Path(self.config.train_dir)
        elif self.split == "val" and self.config.val_dir:
            data_dir = Path(self.config.val_dir)
        elif self.split == "test" and self.config.test_dir:
            data_dir = Path(self.config.test_dir)
        else:
            data_dir = Path(self.config.data_dir)
            
        if not data_dir.exists():
            logger.warning(f"Data directory {data_dir} does not exist")
            return
            
        # Load images from directory structure
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_path in class_dir.glob("*.jpg"):
                    self.data.append(str(img_path))
                    self.labels.append(class_name)
                for img_path in class_dir.glob("*.png"):
                    self.data.append(str(img_path))
                    self.labels.append(class_name)
                    
        # Encode labels
        self.label_encoder = LabelEncoder()
        if self.labels:
            self.labels = self.label_encoder.fit_transform(self.labels)
            
    def _setup_transforms(self):
        """Setup image transforms"""
        if self.split == "train" and self.config.use_augmentation:
            # Training transforms with augmentation
            self.transform = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.Normalize(
                    mean=self.config.normalize_mean,
                    std=self.config.normalize_std
                ),
                ToTensorV2()
            ])
        else:
            # Validation/test transforms
            self.transform = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.Normalize(
                    mean=self.config.normalize_mean,
                    std=self.config.normalize_std
                ),
                ToTensorV2()
            ])
            
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get image item"""
        img_path = self.data[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': img_path
        }

class TextDataset(BaseDataset):
    """Dataset for text data with tokenization"""
    
    def __init__(self, config: DataConfig, split: str = "train"):
        super().__init__(config, split)
        self._setup_tokenizer()
        
    def _load_data(self):
        """Load text data"""
        data_file = Path(self.config.data_dir) / f"{self.split}.json"
        
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                self.data.append(item['text'])
                self.labels.append(item['label'])
        else:
            # Create dummy data for testing
            logger.warning(f"Data file {data_file} not found, creating dummy data")
            self.data = [f"Sample text {i}" for i in range(100)]
            self.labels = [i % 5 for i in range(100)]
            
        # Encode labels
        self.label_encoder = LabelEncoder()
        if self.labels:
            self.labels = self.label_encoder.fit_transform(self.labels)
            
    def _setup_tokenizer(self):
        """Setup tokenizer"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        except ImportError:
            logger.warning("Transformers not available, using simple tokenizer")
            self.tokenizer = None
            
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get text item"""
        text = self.data[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            # Use transformer tokenizer
            encoding = self.tokenizer(
                text,
                max_length=self.config.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Simple tokenization
            tokens = text.split()[:self.config.max_seq_length]
            input_ids = [hash(token) % self.config.vocab_size for token in tokens]
            
            # Pad to max length
            if len(input_ids) < self.config.max_seq_length:
                input_ids.extend([0] * (self.config.max_seq_length - len(input_ids)))
            else:
                input_ids = input_ids[:self.config.max_seq_length]
                
            attention_mask = [1 if token != 0 else 0 for token in input_ids]
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }

class AudioDataset(BaseDataset):
    """Dataset for audio data with spectrogram conversion"""
    
    def __init__(self, config: DataConfig, split: str = "train"):
        super().__init__(config, split)
        self._setup_audio_transforms()
        
    def _load_data(self):
        """Load audio data"""
        data_file = Path(self.config.data_dir) / f"{self.split}.json"
        
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                self.data.append(item['audio_path'])
                self.labels.append(item['label'])
        else:
            # Create dummy data for testing
            logger.warning(f"Data file {data_file} not found, creating dummy data")
            self.data = [f"sample_audio_{i}.wav" for i in range(100)]
            self.labels = [i % 5 for i in range(100)]
            
        # Encode labels
        self.label_encoder = LabelEncoder()
        if self.labels:
            self.labels = self.label_encoder.fit_transform(self.labels)
            
    def _setup_audio_transforms(self):
        """Setup audio transforms"""
        self.mel_transform = audio_transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )
        
        self.amplitude_to_db = audio_transforms.AmplitudeToDB()
        
        if self.split == "train" and self.config.use_augmentation:
            self.augmentation = audio_transforms.Compose([
                audio_transforms.TimeMasking(time_mask_param=20),
                audio_transforms.FrequencyMasking(freq_mask_param=20)
            ])
        else:
            self.augmentation = None
            
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get audio item"""
        audio_path = self.data[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            # Convert to mel spectrogram
            mel_spec = self.mel_transform(torch.tensor(audio))
            mel_spec = self.amplitude_to_db(mel_spec)
            
            # Apply augmentation
            if self.augmentation:
                mel_spec = self.augmentation(mel_spec)
                
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {e}")
            # Create dummy spectrogram
            mel_spec = torch.randn(self.config.n_mels, 100)
            
        return {
            'spectrogram': mel_spec,
            'label': torch.tensor(label, dtype=torch.long),
            'audio_path': audio_path
        }

class MultiModalDataset(BaseDataset):
    """Dataset for multi-modal data (text + image + audio)"""
    
    def __init__(self, config: DataConfig, split: str = "train"):
        super().__init__(config, split)
        self._setup_modality_datasets()
        
    def _load_data(self):
        """Load multi-modal data"""
        data_file = Path(self.config.data_dir) / f"{self.split}_multimodal.json"
        
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                self.data.append({
                    'text': item.get('text', ''),
                    'image_path': item.get('image_path', ''),
                    'audio_path': item.get('audio_path', ''),
                    'metadata': item.get('metadata', {})
                })
                self.labels.append(item['label'])
        else:
            # Create dummy data for testing
            logger.warning(f"Data file {data_file} not found, creating dummy data")
            for i in range(100):
                self.data.append({
                    'text': f"Sample text {i}",
                    'image_path': f"sample_image_{i}.jpg",
                    'audio_path': f"sample_audio_{i}.wav",
                    'metadata': {'id': i}
                })
                self.labels.append(i % 5)
                
        # Encode labels
        self.label_encoder = LabelEncoder()
        if self.labels:
            self.labels = self.label_encoder.fit_transform(self.labels)
            
    def _setup_modality_datasets(self):
        """Setup individual modality datasets"""
        # Create text dataset
        text_config = DataConfig(
            data_dir=self.config.data_dir,
            max_seq_length=self.config.max_seq_length,
            tokenizer_name=self.config.tokenizer_name
        )
        self.text_dataset = TextDataset(text_config, self.split)
        
        # Create image dataset
        image_config = DataConfig(
            data_dir=self.config.data_dir,
            image_size=self.config.image_size,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std
        )
        self.image_dataset = ImageDataset(image_config, self.split)
        
        # Create audio dataset
        audio_config = DataConfig(
            data_dir=self.config.data_dir,
            sample_rate=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        self.audio_dataset = AudioDataset(audio_config, self.split)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get multi-modal item"""
        item_data = self.data[idx]
        label = self.labels[idx]
        
        # Get text features
        text_item = self.text_dataset.__getitem__(idx)
        
        # Get image features
        image_item = self.image_dataset.__getitem__(idx)
        
        # Get audio features
        audio_item = self.audio_dataset.__getitem__(idx)
        
        return {
            'text_input_ids': text_item['input_ids'],
            'text_attention_mask': text_item['attention_mask'],
            'image': image_item['image'],
            'spectrogram': audio_item['spectrogram'],
            'label': torch.tensor(label, dtype=torch.long),
            'metadata': item_data['metadata']
        }

class BalancedSampler(Sampler):
    """Balanced sampler for imbalanced datasets"""
    
    def __init__(self, dataset: BaseDataset, replacement: bool = True):
        self.dataset = dataset
        self.replacement = replacement
        
        # Calculate class frequencies
        self.class_counts = np.bincount(dataset.labels)
        self.num_samples = len(dataset)
        
        # Calculate weights for each sample
        self.weights = torch.zeros(self.num_samples)
        for idx, label in enumerate(dataset.labels):
            self.weights[idx] = 1.0 / self.class_counts[label]
            
    def __iter__(self):
        if self.replacement:
            return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())
        else:
            # Without replacement, we need to be more careful
            indices = []
            for class_idx in range(len(self.class_counts)):
                class_indices = [i for i, label in enumerate(self.dataset.labels) if label == class_idx]
                # Sample equal number from each class
                max_samples_per_class = self.num_samples // len(self.class_counts)
                sampled_indices = np.random.choice(class_indices, 
                                                 min(len(class_indices), max_samples_per_class), 
                                                 replace=False)
                indices.extend(sampled_indices)
            return iter(indices)
            
    def __len__(self):
        return self.num_samples

class DataPipeline:
    """Advanced data pipeline with preprocessing and augmentation"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.datasets = {}
        self.data_loaders = {}
        self.label_encoders = {}
        
    def create_dataset(self, dataset_type: str, split: str = "train") -> BaseDataset:
        """Create dataset of specified type"""
        if dataset_type == "image":
            dataset = ImageDataset(self.config, split)
        elif dataset_type == "text":
            dataset = TextDataset(self.config, split)
        elif dataset_type == "audio":
            dataset = AudioDataset(self.config, split)
        elif dataset_type == "multimodal":
            dataset = MultiModalDataset(self.config, split)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        self.datasets[f"{dataset_type}_{split}"] = dataset
        return dataset
        
    def create_data_loader(self, dataset: BaseDataset, use_balanced_sampling: bool = False) -> DataLoader:
        """Create data loader with optional balanced sampling"""
        sampler = None
        if use_balanced_sampling:
            sampler = BalancedSampler(dataset)
            
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
    def create_train_val_test_loaders(self, dataset_type: str) -> Dict[str, DataLoader]:
        """Create train, validation, and test data loaders"""
        loaders = {}
        
        for split in ["train", "val", "test"]:
            dataset = self.create_dataset(dataset_type, split)
            use_balanced = (split == "train")  # Use balanced sampling only for training
            loaders[split] = self.create_data_loader(dataset, use_balanced)
            
        return loaders
        
    def get_dataset_statistics(self, dataset: BaseDataset) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        if not dataset.labels:
            return {}
            
        stats = {
            'total_samples': len(dataset),
            'num_classes': len(set(dataset.labels)),
            'class_distribution': {},
            'class_weights': dataset.get_class_weights().tolist()
        }
        
        # Calculate class distribution
        unique_labels, counts = np.unique(dataset.labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            stats['class_distribution'][str(label)] = {
                'count': int(count),
                'percentage': float(count / len(dataset) * 100)
            }
            
        return stats
        
    def save_dataset_info(self, dataset: BaseDataset, save_path: str):
        """Save dataset information to file"""
        info = {
            'config': self.config.__dict__,
            'statistics': self.get_dataset_statistics(dataset),
            'label_encoder': dataset.label_encoder.classes_.tolist() if hasattr(dataset, 'label_encoder') else None
        }
        
        with open(save_path, 'w') as f:
            json.dump(info, f, indent=2)
            
    def load_dataset_info(self, load_path: str) -> Dict[str, Any]:
        """Load dataset information from file"""
        with open(load_path, 'r') as f:
            return json.load(f)

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data pipeline
    print("Testing Data Pipeline...")
    
    # Create config
    config = DataConfig(
        data_dir="./test_data",
        batch_size=16,
        num_workers=2,
        image_size=(224, 224),
        max_seq_length=128,
        sample_rate=16000
    )
    
    # Create data pipeline
    pipeline = DataPipeline(config)
    
    # Test text dataset
    print("\nTesting Text Dataset...")
    text_dataset = pipeline.create_dataset("text", "train")
    print(f"Text dataset size: {len(text_dataset)}")
    
    if len(text_dataset) > 0:
        sample = text_dataset[0]
        print(f"Sample text item keys: {sample.keys()}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        
    # Test image dataset
    print("\nTesting Image Dataset...")
    image_dataset = pipeline.create_dataset("image", "train")
    print(f"Image dataset size: {len(image_dataset)}")
    
    if len(image_dataset) > 0:
        sample = image_dataset[0]
        print(f"Sample image item keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        
    # Test audio dataset
    print("\nTesting Audio Dataset...")
    audio_dataset = pipeline.create_dataset("audio", "train")
    print(f"Audio dataset size: {len(audio_dataset)}")
    
    if len(audio_dataset) > 0:
        sample = audio_dataset[0]
        print(f"Sample audio item keys: {sample.keys()}")
        print(f"Spectrogram shape: {sample['spectrogram'].shape}")
        
    # Test multi-modal dataset
    print("\nTesting Multi-Modal Dataset...")
    multimodal_dataset = pipeline.create_dataset("multimodal", "train")
    print(f"Multi-modal dataset size: {len(multimodal_dataset)}")
    
    if len(multimodal_dataset) > 0:
        sample = multimodal_dataset[0]
        print(f"Sample multi-modal item keys: {sample.keys()}")
        
    # Test data loaders
    print("\nTesting Data Loaders...")
    if len(text_dataset) > 0:
        text_loader = pipeline.create_data_loader(text_dataset)
        print(f"Text data loader created with {len(text_loader)} batches")
        
        # Test one batch
        for batch in text_loader:
            print(f"Batch keys: {batch.keys()}")
            print(f"Batch input_ids shape: {batch['input_ids'].shape}")
            break
            
    # Test dataset statistics
    print("\nTesting Dataset Statistics...")
    if len(text_dataset) > 0:
        stats = pipeline.get_dataset_statistics(text_dataset)
        print(f"Dataset statistics: {stats}")
        
    print("\nData pipeline refactored successfully!")
























