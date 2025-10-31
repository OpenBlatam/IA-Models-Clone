#!/usr/bin/env python3
"""
Advanced Data Preprocessing and Augmentation Tools for Frontier Model Training
Provides comprehensive data processing, augmentation, and quality assurance.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
from PIL import Image
import librosa
import soundfile as sf
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import datasets
from datasets import Dataset as HFDataset, DatasetDict

console = Console()

class DataType(Enum):
    """Types of data."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TABULAR = "tabular"
    MULTIMODAL = "multimodal"
    TIME_SERIES = "time_series"

class AugmentationType(Enum):
    """Types of data augmentation."""
    ROTATION = "rotation"
    FLIP = "flip"
    NOISE = "noise"
    TRANSLATION = "translation"
    SCALING = "scaling"
    CROPPING = "cropping"
    COLOR_JITTER = "color_jitter"
    BLUR = "blur"
    ELASTIC_DEFORMATION = "elastic_deformation"
    MIXUP = "mixup"
    CUTMIX = "cutmix"

@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_type: DataType
    input_path: str
    output_path: str
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    validation_split: float = 0.2
    test_split: float = 0.1
    augmentation_enabled: bool = True
    augmentation_factor: float = 2.0
    quality_check: bool = True
    normalization: bool = True
    encoding: str = "utf-8"

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    augmentation_type: AugmentationType
    probability: float = 0.5
    intensity: float = 0.5
    parameters: Dict[str, Any] = None

@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    total_samples: int
    valid_samples: int
    invalid_samples: int
    quality_score: float
    missing_data_percentage: float
    duplicate_percentage: float
    outliers_percentage: float
    issues: List[str] = None
    recommendations: List[str] = None

class TextProcessor:
    """Advanced text data processing."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_text_data(self, texts: List[str]) -> Dict[str, Any]:
        """Process text data with various techniques."""
        console.print("[blue]Processing text data...[/blue]")
        
        processed_data = {
            "original_texts": texts,
            "processed_texts": [],
            "statistics": {},
            "quality_report": None
        }
        
        # Clean and preprocess texts
        cleaned_texts = self._clean_texts(texts)
        processed_data["processed_texts"] = cleaned_texts
        
        # Generate statistics
        stats = self._generate_text_statistics(cleaned_texts)
        processed_data["statistics"] = stats
        
        # Quality assessment
        quality_report = self._assess_text_quality(cleaned_texts)
        processed_data["quality_report"] = quality_report
        
        return processed_data
    
    def _clean_texts(self, texts: List[str]) -> List[str]:
        """Clean and preprocess text data."""
        cleaned_texts = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            # Basic cleaning
            text = text.strip()
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = ' '.join(text.split())  # Remove extra whitespace
            
            # Remove special characters (optional)
            # text = re.sub(r'[^\w\s]', '', text)
            
            cleaned_texts.append(text)
        
        return cleaned_texts
    
    def _generate_text_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Generate text statistics."""
        if not texts:
            return {}
        
        lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]
        
        stats = {
            "total_texts": len(texts),
            "avg_word_count": np.mean(lengths),
            "max_word_count": np.max(lengths),
            "min_word_count": np.min(lengths),
            "avg_char_count": np.mean(char_lengths),
            "max_char_count": np.max(char_lengths),
            "min_char_count": np.min(char_lengths),
            "vocabulary_size": len(set(' '.join(texts).split())),
            "unique_texts": len(set(texts))
        }
        
        return stats
    
    def _assess_text_quality(self, texts: List[str]) -> DataQualityReport:
        """Assess text data quality."""
        total_samples = len(texts)
        valid_samples = sum(1 for text in texts if text.strip())
        invalid_samples = total_samples - valid_samples
        
        # Check for duplicates
        unique_texts = set(texts)
        duplicate_percentage = (total_samples - len(unique_texts)) / total_samples * 100
        
        # Check for empty or very short texts
        short_texts = sum(1 for text in texts if len(text.split()) < 3)
        short_percentage = short_texts / total_samples * 100
        
        # Calculate quality score
        quality_score = (valid_samples / total_samples) * (1 - duplicate_percentage / 100) * (1 - short_percentage / 100)
        
        issues = []
        recommendations = []
        
        if invalid_samples > 0:
            issues.append(f"{invalid_samples} invalid texts found")
            recommendations.append("Remove or fix invalid texts")
        
        if duplicate_percentage > 10:
            issues.append(f"High duplicate rate: {duplicate_percentage:.1f}%")
            recommendations.append("Consider removing duplicates")
        
        if short_percentage > 20:
            issues.append(f"Many short texts: {short_percentage:.1f}%")
            recommendations.append("Filter out very short texts")
        
        return DataQualityReport(
            total_samples=total_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            quality_score=quality_score,
            missing_data_percentage=invalid_samples / total_samples * 100,
            duplicate_percentage=duplicate_percentage,
            outliers_percentage=short_percentage,
            issues=issues,
            recommendations=recommendations
        )

class ImageProcessor:
    """Advanced image data processing."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_image_data(self, image_paths: List[str]) -> Dict[str, Any]:
        """Process image data with various techniques."""
        console.print("[blue]Processing image data...[/blue]")
        
        processed_data = {
            "image_paths": image_paths,
            "processed_images": [],
            "statistics": {},
            "quality_report": None
        }
        
        # Load and process images
        images = self._load_images(image_paths)
        processed_data["processed_images"] = images
        
        # Generate statistics
        stats = self._generate_image_statistics(images)
        processed_data["statistics"] = stats
        
        # Quality assessment
        quality_report = self._assess_image_quality(images)
        processed_data["quality_report"] = quality_report
        
        return processed_data
    
    def _load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """Load images from paths."""
        images = []
        
        for path in image_paths:
            try:
                image = cv2.imread(path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                else:
                    self.logger.warning(f"Could not load image: {path}")
            except Exception as e:
                self.logger.error(f"Error loading image {path}: {e}")
        
        return images
    
    def _generate_image_statistics(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """Generate image statistics."""
        if not images:
            return {}
        
        heights = [img.shape[0] for img in images]
        widths = [img.shape[1] for img in images]
        channels = [img.shape[2] if len(img.shape) > 2 else 1 for img in images]
        
        stats = {
            "total_images": len(images),
            "avg_height": np.mean(heights),
            "avg_width": np.mean(widths),
            "min_height": np.min(heights),
            "max_height": np.max(heights),
            "min_width": np.min(widths),
            "max_width": np.max(widths),
            "avg_channels": np.mean(channels),
            "unique_sizes": len(set((h, w) for h, w in zip(heights, widths)))
        }
        
        return stats
    
    def _assess_image_quality(self, images: List[np.ndarray]) -> DataQualityReport:
        """Assess image data quality."""
        total_samples = len(images)
        valid_samples = sum(1 for img in images if img is not None and img.size > 0)
        invalid_samples = total_samples - valid_samples
        
        # Check for corrupted images
        corrupted_images = 0
        for img in images:
            if img is None or img.size == 0:
                corrupted_images += 1
        
        # Check for very small images
        small_images = sum(1 for img in images if img is not None and img.shape[0] * img.shape[1] < 1000)
        small_percentage = small_images / total_samples * 100
        
        # Calculate quality score
        quality_score = valid_samples / total_samples * (1 - small_percentage / 100)
        
        issues = []
        recommendations = []
        
        if invalid_samples > 0:
            issues.append(f"{invalid_samples} invalid images found")
            recommendations.append("Remove or fix invalid images")
        
        if small_percentage > 20:
            issues.append(f"Many small images: {small_percentage:.1f}%")
            recommendations.append("Filter out very small images")
        
        return DataQualityReport(
            total_samples=total_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            quality_score=quality_score,
            missing_data_percentage=invalid_samples / total_samples * 100,
            duplicate_percentage=0.0,  # Would need hash comparison
            outliers_percentage=small_percentage,
            issues=issues,
            recommendations=recommendations
        )

class AudioProcessor:
    """Advanced audio data processing."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_audio_data(self, audio_paths: List[str]) -> Dict[str, Any]:
        """Process audio data with various techniques."""
        console.print("[blue]Processing audio data...[/blue]")
        
        processed_data = {
            "audio_paths": audio_paths,
            "processed_audio": [],
            "statistics": {},
            "quality_report": None
        }
        
        # Load and process audio files
        audio_data = self._load_audio_files(audio_paths)
        processed_data["processed_audio"] = audio_data
        
        # Generate statistics
        stats = self._generate_audio_statistics(audio_data)
        processed_data["statistics"] = stats
        
        # Quality assessment
        quality_report = self._assess_audio_quality(audio_data)
        processed_data["quality_report"] = quality_report
        
        return processed_data
    
    def _load_audio_files(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """Load audio files from paths."""
        audio_data = []
        
        for path in audio_paths:
            try:
                # Load audio file
                audio, sr = librosa.load(path, sr=None)
                
                audio_info = {
                    "path": path,
                    "audio": audio,
                    "sample_rate": sr,
                    "duration": len(audio) / sr,
                    "channels": 1 if len(audio.shape) == 1 else audio.shape[0]
                }
                
                audio_data.append(audio_info)
                
            except Exception as e:
                self.logger.error(f"Error loading audio {path}: {e}")
        
        return audio_data
    
    def _generate_audio_statistics(self, audio_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate audio statistics."""
        if not audio_data:
            return {}
        
        durations = [info["duration"] for info in audio_data]
        sample_rates = [info["sample_rate"] for info in audio_data]
        
        stats = {
            "total_files": len(audio_data),
            "avg_duration": np.mean(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "avg_sample_rate": np.mean(sample_rates),
            "unique_sample_rates": len(set(sample_rates)),
            "total_duration": np.sum(durations)
        }
        
        return stats
    
    def _assess_audio_quality(self, audio_data: List[Dict[str, Any]]) -> DataQualityReport:
        """Assess audio data quality."""
        total_samples = len(audio_data)
        valid_samples = sum(1 for info in audio_data if info["audio"] is not None)
        invalid_samples = total_samples - valid_samples
        
        # Check for very short audio files
        short_audio = sum(1 for info in audio_data if info["duration"] < 1.0)
        short_percentage = short_audio / total_samples * 100
        
        # Check for very long audio files
        long_audio = sum(1 for info in audio_data if info["duration"] > 300.0)
        long_percentage = long_audio / total_samples * 100
        
        # Calculate quality score
        quality_score = valid_samples / total_samples * (1 - short_percentage / 100) * (1 - long_percentage / 100)
        
        issues = []
        recommendations = []
        
        if invalid_samples > 0:
            issues.append(f"{invalid_samples} invalid audio files found")
            recommendations.append("Remove or fix invalid audio files")
        
        if short_percentage > 20:
            issues.append(f"Many short audio files: {short_percentage:.1f}%")
            recommendations.append("Filter out very short audio files")
        
        if long_percentage > 10:
            issues.append(f"Many long audio files: {long_percentage:.1f}%")
            recommendations.append("Consider splitting long audio files")
        
        return DataQualityReport(
            total_samples=total_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            quality_score=quality_score,
            missing_data_percentage=invalid_samples / total_samples * 100,
            duplicate_percentage=0.0,
            outliers_percentage=short_percentage + long_percentage,
            issues=issues,
            recommendations=recommendations
        )

class DataAugmenter:
    """Advanced data augmentation for various data types."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def augment_text_data(self, texts: List[str]) -> List[str]:
        """Augment text data."""
        console.print(f"[blue]Augmenting text data with {self.config.augmentation_type.value}...[/blue]")
        
        augmented_texts = []
        
        for text in texts:
            # Original text
            augmented_texts.append(text)
            
            # Apply augmentation
            if np.random.random() < self.config.probability:
                if self.config.augmentation_type == AugmentationType.NOISE:
                    augmented_text = self._add_text_noise(text)
                elif self.config.augmentation_type == AugmentationType.TRANSLATION:
                    augmented_text = self._translate_text(text)
                else:
                    augmented_text = text
                
                augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def augment_image_data(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Augment image data."""
        console.print(f"[blue]Augmenting image data with {self.config.augmentation_type.value}...[/blue]")
        
        augmented_images = []
        
        for image in images:
            # Original image
            augmented_images.append(image)
            
            # Apply augmentation
            if np.random.random() < self.config.probability:
                if self.config.augmentation_type == AugmentationType.ROTATION:
                    augmented_image = self._rotate_image(image)
                elif self.config.augmentation_type == AugmentationType.FLIP:
                    augmented_image = self._flip_image(image)
                elif self.config.augmentation_type == AugmentationType.NOISE:
                    augmented_image = self._add_image_noise(image)
                elif self.config.augmentation_type == AugmentationType.COLOR_JITTER:
                    augmented_image = self._color_jitter_image(image)
                else:
                    augmented_image = image
                
                augmented_images.append(augmented_image)
        
        return augmented_images
    
    def augment_audio_data(self, audio_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Augment audio data."""
        console.print(f"[blue]Augmenting audio data with {self.config.augmentation_type.value}...[/blue]")
        
        augmented_audio = []
        
        for audio_info in audio_data:
            # Original audio
            augmented_audio.append(audio_info)
            
            # Apply augmentation
            if np.random.random() < self.config.probability:
                if self.config.augmentation_type == AugmentationType.NOISE:
                    augmented_info = self._add_audio_noise(audio_info)
                elif self.config.augmentation_type == AugmentationType.SCALING:
                    augmented_info = self._scale_audio(audio_info)
                else:
                    augmented_info = audio_info
                
                augmented_audio.append(augmented_info)
        
        return augmented_audio
    
    def _add_text_noise(self, text: str) -> str:
        """Add noise to text."""
        words = text.split()
        if len(words) < 2:
            return text
        
        # Randomly swap words
        if np.random.random() < 0.3:
            i, j = np.random.choice(len(words), 2, replace=False)
            words[i], words[j] = words[j], words[i]
        
        # Randomly delete words
        if np.random.random() < 0.2:
            if len(words) > 1:
                words.pop(np.random.randint(len(words)))
        
        return ' '.join(words)
    
    def _translate_text(self, text: str) -> str:
        """Translate text (simplified - would use actual translation service)."""
        # This is a placeholder - in practice, you'd use a translation service
        return text
    
    def _rotate_image(self, image: np.ndarray) -> np.ndarray:
        """Rotate image."""
        angle = np.random.uniform(-30, 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated
    
    def _flip_image(self, image: np.ndarray) -> np.ndarray:
        """Flip image."""
        if np.random.random() < 0.5:
            return cv2.flip(image, 1)  # Horizontal flip
        else:
            return cv2.flip(image, 0)  # Vertical flip
    
    def _add_image_noise(self, image: np.ndarray) -> np.ndarray:
        """Add noise to image."""
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def _color_jitter_image(self, image: np.ndarray) -> np.ndarray:
        """Apply color jitter to image."""
        # Brightness
        brightness = np.random.uniform(0.8, 1.2)
        image = image * brightness
        
        # Contrast
        contrast = np.random.uniform(0.8, 1.2)
        image = image * contrast
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _add_audio_noise(self, audio_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to audio."""
        audio = audio_info["audio"]
        noise = np.random.normal(0, 0.01, audio.shape)
        noisy_audio = audio + noise
        
        augmented_info = audio_info.copy()
        augmented_info["audio"] = noisy_audio
        return augmented_info
    
    def _scale_audio(self, audio_info: Dict[str, Any]) -> Dict[str, Any]:
        """Scale audio amplitude."""
        audio = audio_info["audio"]
        scale_factor = np.random.uniform(0.8, 1.2)
        scaled_audio = audio * scale_factor
        
        augmented_info = audio_info.copy()
        augmented_info["audio"] = scaled_audio
        return augmented_info

class DataManager:
    """Main data management class."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors based on data type
        if config.data_type == DataType.TEXT:
            self.processor = TextProcessor(config)
        elif config.data_type == DataType.IMAGE:
            self.processor = ImageProcessor(config)
        elif config.data_type == DataType.AUDIO:
            self.processor = AudioProcessor(config)
        else:
            raise ValueError(f"Unsupported data type: {config.data_type}")
    
    def process_data(self, data: Union[List[str], List[np.ndarray], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Process data using appropriate processor."""
        return self.processor.process_data(data)
    
    def augment_data(self, data: Union[List[str], List[np.ndarray], List[Dict[str, Any]]], 
                    augmentation_config: AugmentationConfig) -> Union[List[str], List[np.ndarray], List[Dict[str, Any]]]:
        """Augment data using specified augmentation."""
        augmenter = DataAugmenter(augmentation_config)
        
        if self.config.data_type == DataType.TEXT:
            return augmenter.augment_text_data(data)
        elif self.config.data_type == DataType.IMAGE:
            return augmenter.augment_image_data(data)
        elif self.config.data_type == DataType.AUDIO:
            return augmenter.augment_audio_data(data)
        else:
            return data
    
    def create_dataset(self, data: Union[List[str], List[np.ndarray], List[Dict[str, Any]]],
                      labels: Optional[List[Any]] = None) -> Dataset:
        """Create PyTorch dataset from processed data."""
        
        class CustomDataset(Dataset):
            def __init__(self, data, labels=None):
                self.data = data
                self.labels = labels
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                if self.labels is not None:
                    return self.data[idx], self.labels[idx]
                return self.data[idx]
        
        return CustomDataset(data, labels)
    
    def create_dataloader(self, dataset: Dataset, 
                         batch_size: Optional[int] = None,
                         shuffle: Optional[bool] = None,
                         num_workers: Optional[int] = None) -> DataLoader:
        """Create PyTorch dataloader from dataset."""
        
        batch_size = batch_size or self.config.batch_size
        shuffle = shuffle if shuffle is not None else self.config.shuffle
        num_workers = num_workers or self.config.num_workers
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def split_data(self, data: List[Any], labels: Optional[List[Any]] = None) -> Tuple[List[Any], List[Any], List[Any]]:
        """Split data into train, validation, and test sets."""
        
        if labels is not None:
            # Stratified split
            train_data, temp_data, train_labels, temp_labels = train_test_split(
                data, labels, 
                test_size=self.config.validation_split + self.config.test_split,
                random_state=42,
                stratify=labels
            )
            
            val_data, test_data, val_labels, test_labels = train_test_split(
                temp_data, temp_labels,
                test_size=self.config.test_split / (self.config.validation_split + self.config.test_split),
                random_state=42,
                stratify=temp_labels
            )
            
            return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
        else:
            # Random split
            train_data, temp_data = train_test_split(
                data, 
                test_size=self.config.validation_split + self.config.test_split,
                random_state=42
            )
            
            val_data, test_data = train_test_split(
                temp_data,
                test_size=self.config.test_split / (self.config.validation_split + self.config.test_split),
                random_state=42
            )
            
            return train_data, val_data, test_data

def main():
    """Main function for data processing CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Processing and Augmentation Tools")
    parser.add_argument("--data-type", type=str, 
                       choices=["text", "image", "audio", "tabular"],
                       required=True, help="Type of data to process")
    parser.add_argument("--input-path", type=str, required=True, help="Input data path")
    parser.add_argument("--output-path", type=str, required=True, help="Output data path")
    parser.add_argument("--augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--augmentation-type", type=str,
                       choices=["rotation", "flip", "noise", "translation", "scaling"],
                       default="noise", help="Type of augmentation")
    parser.add_argument("--quality-check", action="store_true", help="Enable quality checking")
    
    args = parser.parse_args()
    
    # Create data configuration
    config = DataConfig(
        data_type=DataType(args.data_type),
        input_path=args.input_path,
        output_path=args.output_path,
        augmentation_enabled=args.augmentation,
        quality_check=args.quality_check
    )
    
    # Create data manager
    manager = DataManager(config)
    
    # Load data (simplified - in practice, you'd load from actual files)
    if args.data_type == "text":
        data = ["Sample text 1", "Sample text 2", "Sample text 3"]
    elif args.data_type == "image":
        data = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
    elif args.data_type == "audio":
        data = [{"audio": np.random.randn(1000), "sample_rate": 22050, "duration": 1.0} for _ in range(3)]
    else:
        data = []
    
    # Process data
    processed_data = manager.process_data(data)
    
    # Apply augmentation if enabled
    if args.augmentation:
        augmentation_config = AugmentationConfig(
            augmentation_type=AugmentationType(args.augmentation_type),
            probability=0.5
        )
        augmented_data = manager.augment_data(data, augmentation_config)
        processed_data["augmented_data"] = augmented_data
    
    # Save processed data
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    console.print(f"[green]Processed data saved to: {args.output_path}[/green]")
    
    # Display quality report if available
    if processed_data.get("quality_report"):
        report = processed_data["quality_report"]
        console.print(f"[blue]Quality Score: {report.quality_score:.2f}[/blue]")
        if report.issues:
            console.print(f"[yellow]Issues: {', '.join(report.issues)}[/yellow]")
        if report.recommendations:
            console.print(f"[green]Recommendations: {', '.join(report.recommendations)}[/green]")

if __name__ == "__main__":
    main()
