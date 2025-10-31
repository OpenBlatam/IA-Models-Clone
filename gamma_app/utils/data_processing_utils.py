"""
Gamma App - Data Processing Utilities
Advanced data processing and transformation utilities for ML/AI workflows
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import cv2
from PIL import Image
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

class DataType(Enum):
    """Data types"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    GRAPH = "graph"

class ProcessingMode(Enum):
    """Processing modes"""
    TRAINING = "training"
    INFERENCE = "inference"
    VALIDATION = "validation"
    TESTING = "testing"

@dataclass
class DataConfig:
    """Data processing configuration"""
    data_type: DataType
    processing_mode: ProcessingMode
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    max_length: int = 512
    image_size: Tuple[int, int] = (224, 224)
    audio_sample_rate: int = 22050
    normalize: bool = True
    augment: bool = False
    cache: bool = True

@dataclass
class ProcessingResult:
    """Data processing result"""
    success: bool
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0

class TextProcessor:
    """Advanced text processing utilities"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            logger.info(f"Initialized text processor with {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing text processor: {e}")
    
    def tokenize_text(self, text: str, max_length: int = 512, padding: bool = True, truncation: bool = True) -> Dict[str, torch.Tensor]:
        """Tokenize text using transformer tokenizer"""
        try:
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            
            tokens = self.tokenizer(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors="pt"
            )
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise
    
    def encode_text(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """Encode texts to embeddings"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            
            # Tokenize texts
            tokens = self.tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract text features"""
        try:
            features = {
                'length': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(text.split('.')),
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
                'unique_words': len(set(text.lower().split())),
                'punctuation_count': sum(1 for c in text if c in '.,!?;:'),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        try:
            import re
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters (keep alphanumeric and basic punctuation)
            text = re.sub(r'[^\w\s.,!?;:]', '', text)
            
            # Normalize case
            text = text.strip().lower()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

class ImageProcessor:
    """Advanced image processing utilities"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from file"""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int] = None) -> Image.Image:
        """Resize image"""
        try:
            if size is None:
                size = self.target_size
            
            image = image.resize(size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            raise
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for model input"""
        try:
            # Convert to float and normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            image = (image - np.array(self.normalize_mean)) / np.array(self.normalize_std)
            
            return image
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            raise
    
    def augment_image(self, image: Image.Image, augment_config: Dict[str, Any] = None) -> Image.Image:
        """Apply data augmentation to image"""
        try:
            if augment_config is None:
                augment_config = {
                    'rotation': 10,
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.2
                }
            
            # Random rotation
            if 'rotation' in augment_config:
                import random
                angle = random.uniform(-augment_config['rotation'], augment_config['rotation'])
                image = image.rotate(angle)
            
            # Random brightness, contrast, saturation
            from PIL import ImageEnhance
            
            if 'brightness' in augment_config:
                enhancer = ImageEnhance.Brightness(image)
                factor = 1 + random.uniform(-augment_config['brightness'], augment_config['brightness'])
                image = enhancer.enhance(factor)
            
            if 'contrast' in augment_config:
                enhancer = ImageEnhance.Contrast(image)
                factor = 1 + random.uniform(-augment_config['contrast'], augment_config['contrast'])
                image = enhancer.enhance(factor)
            
            if 'saturation' in augment_config:
                enhancer = ImageEnhance.Color(image)
                factor = 1 + random.uniform(-augment_config['saturation'], augment_config['saturation'])
                image = enhancer.enhance(factor)
            
            return image
            
        except Exception as e:
            logger.error(f"Error augmenting image: {e}")
            return image
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract image features"""
        try:
            features = {
                'shape': image.shape,
                'mean': np.mean(image),
                'std': np.std(image),
                'min': np.min(image),
                'max': np.max(image),
                'histogram': np.histogram(image, bins=256)[0].tolist()
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return {}
    
    def process_image(self, image_path: str, config: DataConfig) -> torch.Tensor:
        """Process image for model input"""
        try:
            # Load image
            image = self.load_image(image_path)
            
            # Resize
            image = self.resize_image(image, config.image_size)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Normalize
            if config.normalize:
                image_array = self.normalize_image(image_array)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC to CHW
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

class AudioProcessor:
    """Advanced audio processing utilities"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract audio features"""
        try:
            features = {
                'mfcc': librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13).mean(axis=1).tolist(),
                'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate).mean(),
                'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate).mean(),
                'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio).mean(),
                'chroma': librosa.feature.chroma_stft(y=audio, sr=self.sample_rate).mean(axis=1).tolist(),
                'tonnetz': librosa.feature.tonnetz(y=audio, sr=self.sample_rate).mean(axis=1).tolist(),
                'duration': len(audio) / self.sample_rate,
                'rms': librosa.feature.rms(y=audio).mean()
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio"""
        try:
            # Normalize to [-1, 1]
            audio = audio / np.max(np.abs(audio))
            return audio
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            raise
    
    def augment_audio(self, audio: np.ndarray, augment_config: Dict[str, Any] = None) -> np.ndarray:
        """Apply data augmentation to audio"""
        try:
            if augment_config is None:
                augment_config = {
                    'noise_factor': 0.005,
                    'time_shift': 0.2,
                    'pitch_shift': 2
                }
            
            import random
            
            # Add noise
            if 'noise_factor' in augment_config:
                noise = np.random.normal(0, augment_config['noise_factor'], audio.shape)
                audio = audio + noise
            
            # Time shift
            if 'time_shift' in augment_config:
                shift = int(random.uniform(-augment_config['time_shift'], augment_config['time_shift']) * len(audio))
                audio = np.roll(audio, shift)
            
            # Pitch shift
            if 'pitch_shift' in augment_config:
                n_steps = random.uniform(-augment_config['pitch_shift'], augment_config['pitch_shift'])
                audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error augmenting audio: {e}")
            return audio
    
    def process_audio(self, audio_path: str, config: DataConfig) -> torch.Tensor:
        """Process audio for model input"""
        try:
            # Load audio
            audio, sr = self.load_audio(audio_path)
            
            # Normalize
            if config.normalize:
                audio = self.normalize_audio(audio)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            return audio_tensor
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise

class TabularProcessor:
    """Advanced tabular data processing utilities"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def load_data(self, file_path: str, file_type: str = "csv") -> pd.DataFrame:
        """Load tabular data from file"""
        try:
            if file_type == "csv":
                data = pd.read_csv(file_path)
            elif file_type == "excel":
                data = pd.read_excel(file_path)
            elif file_type == "json":
                data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame, target_column: str = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess tabular data"""
        try:
            # Handle missing values
            data = data.fillna(data.mean(numeric_only=True))
            data = data.fillna(data.mode().iloc[0])
            
            # Separate features and target
            if target_column and target_column in data.columns:
                X = data.drop(columns=[target_column])
                y = data[target_column]
            else:
                X = data
                y = None
            
            # Encode categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    X[col] = self.encoders[col].fit_transform(X[col])
                else:
                    X[col] = self.encoders[col].transform(X[col])
            
            # Scale numerical features
            numerical_columns = X.select_dtypes(include=[np.number]).columns
            for col in numerical_columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    X[col] = self.scalers[col].fit_transform(X[[col]]).flatten()
                else:
                    X[col] = self.scalers[col].transform(X[[col]]).flatten()
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Convert to numpy arrays
            X_array = X.values.astype(np.float32)
            y_array = y.values if y is not None else None
            
            return X_array, y_array
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets"""
        try:
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise

class DataProcessor:
    """Main data processing class"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.tabular_processor = TabularProcessor()
        self.cache = {}
    
    async def process_data(
        self,
        data: Any,
        config: DataConfig,
        cache_key: str = None
    ) -> ProcessingResult:
        """Process data based on type and configuration"""
        try:
            start_time = time.time()
            
            # Check cache
            if config.cache and cache_key and cache_key in self.cache:
                logger.info(f"Using cached data for {cache_key}")
                return self.cache[cache_key]
            
            result = ProcessingResult(success=False, data=None)
            
            if config.data_type == DataType.TEXT:
                result = await self._process_text_data(data, config)
            elif config.data_type == DataType.IMAGE:
                result = await self._process_image_data(data, config)
            elif config.data_type == DataType.AUDIO:
                result = await self._process_audio_data(data, config)
            elif config.data_type == DataType.TABULAR:
                result = await self._process_tabular_data(data, config)
            else:
                result.errors.append(f"Unsupported data type: {config.data_type}")
            
            result.processing_time = time.time() - start_time
            
            # Cache result
            if config.cache and cache_key:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return ProcessingResult(
                success=False,
                data=None,
                errors=[str(e)],
                processing_time=time.time() - start_time
            )
    
    async def _process_text_data(self, data: Union[str, List[str]], config: DataConfig) -> ProcessingResult:
        """Process text data"""
        try:
            if isinstance(data, str):
                data = [data]
            
            # Tokenize texts
            tokens = []
            for text in data:
                tokenized = self.text_processor.tokenize_text(text, config.max_length)
                tokens.append(tokenized)
            
            # Stack tensors
            input_ids = torch.stack([t['input_ids'].squeeze() for t in tokens])
            attention_mask = torch.stack([t['attention_mask'].squeeze() for t in tokens])
            
            processed_data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            return ProcessingResult(
                success=True,
                data=processed_data,
                metadata={'num_texts': len(data), 'max_length': config.max_length}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                errors=[f"Text processing error: {str(e)}"]
            )
    
    async def _process_image_data(self, data: Union[str, List[str]], config: DataConfig) -> ProcessingResult:
        """Process image data"""
        try:
            if isinstance(data, str):
                data = [data]
            
            # Process images
            images = []
            for image_path in data:
                image_tensor = self.image_processor.process_image(image_path, config)
                images.append(image_tensor)
            
            # Stack tensors
            processed_data = torch.stack(images)
            
            return ProcessingResult(
                success=True,
                data=processed_data,
                metadata={'num_images': len(data), 'image_size': config.image_size}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                errors=[f"Image processing error: {str(e)}"]
            )
    
    async def _process_audio_data(self, data: Union[str, List[str]], config: DataConfig) -> ProcessingResult:
        """Process audio data"""
        try:
            if isinstance(data, str):
                data = [data]
            
            # Process audio files
            audio_tensors = []
            for audio_path in data:
                audio_tensor = self.audio_processor.process_audio(audio_path, config)
                audio_tensors.append(audio_tensor)
            
            # Pad sequences to same length
            max_length = max(len(tensor) for tensor in audio_tensors)
            padded_tensors = []
            for tensor in audio_tensors:
                if len(tensor) < max_length:
                    padding = torch.zeros(max_length - len(tensor))
                    tensor = torch.cat([tensor, padding])
                padded_tensors.append(tensor)
            
            processed_data = torch.stack(padded_tensors)
            
            return ProcessingResult(
                success=True,
                data=processed_data,
                metadata={'num_audio_files': len(data), 'sample_rate': config.audio_sample_rate}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                errors=[f"Audio processing error: {str(e)}"]
            )
    
    async def _process_tabular_data(self, data: Union[str, pd.DataFrame], config: DataConfig) -> ProcessingResult:
        """Process tabular data"""
        try:
            if isinstance(data, str):
                df = self.tabular_processor.load_data(data)
            else:
                df = data
            
            # Preprocess data
            X, y = self.tabular_processor.preprocess_data(df)
            
            processed_data = {
                'features': torch.from_numpy(X),
                'targets': torch.from_numpy(y) if y is not None else None
            }
            
            return ProcessingResult(
                success=True,
                data=processed_data,
                metadata={'num_samples': len(X), 'num_features': X.shape[1]}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                errors=[f"Tabular processing error: {str(e)}"]
            )
    
    def create_dataloader(
        self,
        data: Any,
        config: DataConfig,
        shuffle: bool = None
    ) -> DataLoader:
        """Create PyTorch DataLoader"""
        try:
            if shuffle is None:
                shuffle = config.shuffle
            
            # Create dataset
            if isinstance(data, torch.Tensor):
                dataset = TensorDataset(data)
            elif isinstance(data, dict):
                if 'input_ids' in data and 'attention_mask' in data:
                    dataset = TensorDataset(data['input_ids'], data['attention_mask'])
                elif 'features' in data and 'targets' in data:
                    if data['targets'] is not None:
                        dataset = TensorDataset(data['features'], data['targets'])
                    else:
                        dataset = TensorDataset(data['features'])
                else:
                    raise ValueError("Unsupported data format for DataLoader")
            else:
                raise ValueError("Unsupported data type for DataLoader")
            
            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )
            
            return dataloader
            
        except Exception as e:
            logger.error(f"Error creating DataLoader: {e}")
            raise
    
    def clear_cache(self):
        """Clear processing cache"""
        self.cache.clear()
        logger.info("Data processing cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'cache_keys': list(self.cache.keys())
        }

# Global data processor instance
data_processor = DataProcessor()

async def process_data(data: Any, config: DataConfig, cache_key: str = None) -> ProcessingResult:
    """Process data using global processor"""
    return await data_processor.process_data(data, config, cache_key)

def create_dataloader(data: Any, config: DataConfig, shuffle: bool = None) -> DataLoader:
    """Create DataLoader using global processor"""
    return data_processor.create_dataloader(data, config, shuffle)

def clear_data_cache():
    """Clear data processing cache"""
    data_processor.clear_cache()

def get_data_cache_stats() -> Dict[str, Any]:
    """Get data processing cache statistics"""
    return data_processor.get_cache_stats()
























