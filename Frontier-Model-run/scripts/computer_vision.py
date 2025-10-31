#!/usr/bin/env python3
"""
Advanced Computer Vision Pipeline for Frontier Model Training
Provides comprehensive CV algorithms, image processing, and deep learning capabilities.
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
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import PIL
from PIL import Image, ImageEnhance, ImageFilter
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet, VGG, DenseNet, EfficientNet, VisionTransformer
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class CVTask(Enum):
    """Computer vision tasks."""
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    FACE_RECOGNITION = "face_recognition"
    OPTICAL_CHARACTER_RECOGNITION = "ocr"
    IMAGE_GENERATION = "image_generation"
    IMAGE_RESTORATION = "image_restoration"
    STYLE_TRANSFER = "style_transfer"
    DEPTH_ESTIMATION = "depth_estimation"
    SUPER_RESOLUTION = "super_resolution"

class CVModel(Enum):
    """Computer vision models."""
    # Classification models
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    VGG16 = "vgg16"
    VGG19 = "vgg19"
    DENSENET121 = "densenet121"
    EFFICIENTNET_B0 = "efficientnet_b0"
    VIT_BASE = "vit_base_patch16_224"
    MOBILENET_V2 = "mobilenet_v2"
    INCEPTION_V3 = "inception_v3"
    
    # Detection models
    YOLO_V5 = "yolo_v5"
    YOLO_V8 = "yolo_v8"
    RCNN = "rcnn"
    FAST_RCNN = "fast_rcnn"
    FASTER_RCNN = "faster_rcnn"
    RETINANET = "retinanet"
    SSD = "ssd"
    
    # Segmentation models
    U_NET = "u_net"
    DEEPLAB_V3 = "deeplab_v3"
    PSPNET = "pspnet"
    FPN = "fpn"
    LINKNET = "linknet"
    MANET = "manet"

class ImageProcessing(Enum):
    """Image processing operations."""
    RESIZE = "resize"
    CROP = "crop"
    ROTATE = "rotate"
    FLIP = "flip"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE = "hue"
    BLUR = "blur"
    SHARPEN = "sharpen"
    NOISE = "noise"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"
    EDGE_DETECTION = "edge_detection"
    MORPHOLOGICAL = "morphological"

class AugmentationStrategy(Enum):
    """Data augmentation strategies."""
    BASIC = "basic"
    ADVANCED = "advanced"
    MIXUP = "mixup"
    CUTMIX = "cutmix"
    CUTOUT = "cutout"
    RANDOM_ERASING = "random_erasing"
    AUTO_AUGMENT = "auto_augment"
    RANDAUGMENT = "randaugment"
    TRIVIAL_AUGMENT = "trivial_augment"

@dataclass
class CVConfig:
    """Computer vision configuration."""
    task: CVTask = CVTask.CLASSIFICATION
    model: CVModel = CVModel.RESNET50
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    num_classes: int = 1000
    pretrained: bool = True
    augmentation_strategy: AugmentationStrategy = AugmentationStrategy.ADVANCED
    enable_mixed_precision: bool = True
    enable_gradient_clipping: bool = True
    enable_learning_rate_scheduling: bool = True
    enable_early_stopping: bool = True
    enable_model_checkpointing: bool = True
    enable_tensorboard_logging: bool = True
    enable_visualization: bool = True
    enable_data_analysis: bool = True
    device: str = "auto"

@dataclass
class ImageData:
    """Image data container."""
    image_id: str
    image_path: str
    image_array: np.ndarray
    label: Optional[Any] = None
    metadata: Dict[str, Any] = None
    processed_image: Optional[np.ndarray] = None

@dataclass
class CVModelResult:
    """CV model result."""
    result_id: str
    task: CVTask
    model: CVModel
    performance_metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    model_state: Dict[str, Any]
    created_at: datetime

class ImageProcessor:
    """Image processing engine."""
    
    def __init__(self, config: CVConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_image(self, image_path: str) -> ImageData:
        """Load image from path."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Create image data
            image_data = ImageData(
                image_id=f"img_{int(time.time())}",
                image_path=image_path,
                image_array=image_array,
                metadata={
                    'size': image.size,
                    'mode': image.mode,
                    'format': image.format
                }
            )
            
            return image_data
            
        except Exception as e:
            self.logger.error(f"Image loading failed: {e}")
            return self._create_fallback_image(image_path)
    
    def process_image(self, image_data: ImageData, operations: List[ImageProcessing]) -> ImageData:
        """Process image with specified operations."""
        processed_image = image_data.image_array.copy()
        
        for operation in operations:
            if operation == ImageProcessing.RESIZE:
                processed_image = self._resize_image(processed_image)
            elif operation == ImageProcessing.CROP:
                processed_image = self._crop_image(processed_image)
            elif operation == ImageProcessing.ROTATE:
                processed_image = self._rotate_image(processed_image)
            elif operation == ImageProcessing.FLIP:
                processed_image = self._flip_image(processed_image)
            elif operation == ImageProcessing.BRIGHTNESS:
                processed_image = self._adjust_brightness(processed_image)
            elif operation == ImageProcessing.CONTRAST:
                processed_image = self._adjust_contrast(processed_image)
            elif operation == ImageProcessing.BLUR:
                processed_image = self._blur_image(processed_image)
            elif operation == ImageProcessing.SHARPEN:
                processed_image = self._sharpen_image(processed_image)
            elif operation == ImageProcessing.NOISE:
                processed_image = self._add_noise(processed_image)
            elif operation == ImageProcessing.HISTOGRAM_EQUALIZATION:
                processed_image = self._histogram_equalization(processed_image)
            elif operation == ImageProcessing.EDGE_DETECTION:
                processed_image = self._edge_detection(processed_image)
        
        image_data.processed_image = processed_image
        return image_data
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image."""
        return cv2.resize(image, self.config.image_size)
    
    def _crop_image(self, image: np.ndarray) -> np.ndarray:
        """Crop image."""
        h, w = image.shape[:2]
        crop_size = min(h, w) // 2
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        return image[start_h:start_h+crop_size, start_w:start_w+crop_size]
    
    def _rotate_image(self, image: np.ndarray) -> np.ndarray:
        """Rotate image."""
        angle = np.random.uniform(-30, 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    def _flip_image(self, image: np.ndarray) -> np.ndarray:
        """Flip image."""
        if np.random.random() > 0.5:
            return cv2.flip(image, 1)  # Horizontal flip
        return image
    
    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Adjust brightness."""
        factor = np.random.uniform(0.8, 1.2)
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adjust contrast."""
        factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def _blur_image(self, image: np.ndarray) -> np.ndarray:
        """Blur image."""
        kernel_size = np.random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image."""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add noise to image."""
        noise = np.random.normal(0, 25, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization."""
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            return cv2.equalizeHist(image)
    
    def _edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Apply edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    def _create_fallback_image(self, image_path: str) -> ImageData:
        """Create fallback image."""
        fallback_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return ImageData(
            image_id=f"fallback_{int(time.time())}",
            image_path=image_path,
            image_array=fallback_image,
            metadata={'fallback': True}
        )

class DataAugmentation:
    """Data augmentation engine."""
    
    def __init__(self, config: CVConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create augmentation pipeline."""
        if self.config.augmentation_strategy == AugmentationStrategy.BASIC:
            return A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        elif self.config.augmentation_strategy == AugmentationStrategy.ADVANCED:
            return A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RGBShift(p=0.3),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.GridDistortion(p=0.3),
                A.ElasticTransform(p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def augment_image(self, image: np.ndarray) -> torch.Tensor:
        """Apply augmentation to image."""
        try:
            augmented = self.augmentation_pipeline(image=image)
            return augmented['image']
        except Exception as e:
            self.logger.error(f"Augmentation failed: {e}")
            # Fallback to basic processing
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)

class CVModelFactory:
    """CV model factory."""
    
    def __init__(self, config: CVConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def create_model(self) -> nn.Module:
        """Create CV model."""
        console.print(f"[blue]Creating {self.config.model.value} model for {self.config.task.value}...[/blue]")
        
        try:
            if self.config.task == CVTask.CLASSIFICATION:
                return self._create_classification_model()
            elif self.config.task == CVTask.OBJECT_DETECTION:
                return self._create_detection_model()
            elif self.config.task == CVTask.SEMANTIC_SEGMENTATION:
                return self._create_segmentation_model()
            else:
                return self._create_classification_model()
                
        except Exception as e:
            self.logger.error(f"Model creation failed: {e}")
            return self._create_fallback_model()
    
    def _create_classification_model(self) -> nn.Module:
        """Create classification model."""
        if self.config.model == CVModel.RESNET50:
            model = models.resnet50(pretrained=self.config.pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        elif self.config.model == CVModel.RESNET101:
            model = models.resnet101(pretrained=self.config.pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        elif self.config.model == CVModel.VGG16:
            model = models.vgg16(pretrained=self.config.pretrained)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.config.num_classes)
        elif self.config.model == CVModel.DENSENET121:
            model = models.densenet121(pretrained=self.config.pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, self.config.num_classes)
        elif self.config.model == CVModel.EFFICIENTNET_B0:
            try:
                import timm
                model = timm.create_model('efficientnet_b0', pretrained=self.config.pretrained, num_classes=self.config.num_classes)
            except ImportError:
                model = models.resnet50(pretrained=self.config.pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        elif self.config.model == CVModel.VIT_BASE:
            try:
                import timm
                model = timm.create_model('vit_base_patch16_224', pretrained=self.config.pretrained, num_classes=self.config.num_classes)
            except ImportError:
                model = models.resnet50(pretrained=self.config.pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        else:
            model = models.resnet50(pretrained=self.config.pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        
        return model.to(self.device)
    
    def _create_detection_model(self) -> nn.Module:
        """Create object detection model."""
        # Simplified detection model
        class DetectionModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.backbone = models.resnet50(pretrained=True)
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
                self.classifier = nn.Conv2d(2048, num_classes, 1)
                self.bbox_regressor = nn.Conv2d(2048, 4, 1)
            
            def forward(self, x):
                features = self.backbone(x)
                class_logits = self.classifier(features)
                bbox_preds = self.bbox_regressor(features)
                return class_logits, bbox_preds
        
        return DetectionModel(self.config.num_classes).to(self.device)
    
    def _create_segmentation_model(self) -> nn.Module:
        """Create segmentation model."""
        try:
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights="imagenet" if self.config.pretrained else None,
                in_channels=3,
                classes=self.config.num_classes
            )
            return model.to(self.device)
        except ImportError:
            # Fallback to simple segmentation model
            class SegmentationModel(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.backbone = models.resnet50(pretrained=True)
                    self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
                    self.decoder = nn.ConvTranspose2d(2048, num_classes, 32, 32)
                
                def forward(self, x):
                    features = self.backbone(x)
                    output = self.decoder(features)
                    return F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            return SegmentationModel(self.config.num_classes).to(self.device)
    
    def _create_fallback_model(self) -> nn.Module:
        """Create fallback model."""
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        return model.to(self.device)

class CVTrainer:
    """CV training engine."""
    
    def __init__(self, config: CVConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader) -> Dict[str, Any]:
        """Train CV model."""
        console.print(f"[blue]Training {self.config.model.value} model...[/blue]")
        
        model = model.to(self.device)
        
        # Initialize optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # Training metrics
        training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                if self.config.enable_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            # Update history
            training_history['train_loss'].append(train_loss / len(train_loader))
            training_history['train_accuracy'].append(train_accuracy)
            training_history['val_loss'].append(val_loss / len(val_loader))
            training_history['val_accuracy'].append(val_accuracy)
            
            # Learning rate scheduling
            if self.config.enable_learning_rate_scheduling:
                scheduler.step()
            
            # Early stopping
            if self.config.enable_early_stopping:
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:  # Patience of 10 epochs
                    console.print(f"[blue]Early stopping at epoch {epoch}[/blue]")
                    break
            
            # Log progress
            if epoch % 10 == 0:
                console.print(f"[blue]Epoch {epoch}: Train Acc = {train_accuracy:.2f}%, Val Acc = {val_accuracy:.2f}%[/blue]")
        
        return {
            'training_history': training_history,
            'best_val_accuracy': best_val_accuracy,
            'final_train_accuracy': training_history['train_accuracy'][-1],
            'final_val_accuracy': training_history['val_accuracy'][-1]
        }

class CVSystem:
    """Main computer vision system."""
    
    def __init__(self, config: CVConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.image_processor = ImageProcessor(config)
        self.data_augmentation = DataAugmentation(config)
        self.model_factory = CVModelFactory(config)
        self.trainer = CVTrainer(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.cv_results: Dict[str, CVModelResult] = {}
    
    def _init_database(self) -> str:
        """Initialize CV database."""
        db_path = Path("./computer_vision.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cv_models (
                    model_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    training_history TEXT NOT NULL,
                    model_state TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def run_cv_experiment(self, train_data: List[str], val_data: List[str], 
                         labels: List[int] = None) -> CVModelResult:
        """Run complete CV experiment."""
        console.print(f"[blue]Starting CV experiment with {self.config.task.value}...[/blue]")
        
        start_time = time.time()
        result_id = f"cv_exp_{int(time.time())}"
        
        # Create model
        model = self.model_factory.create_model()
        
        # Create data loaders
        train_loader = self._create_data_loader(train_data, labels, is_training=True)
        val_loader = self._create_data_loader(val_data, labels, is_training=False)
        
        # Train model
        training_result = self.trainer.train_model(model, train_loader, val_loader)
        
        # Evaluate model
        performance_metrics = self._evaluate_model(model, val_loader)
        
        # Create CV result
        cv_result = CVModelResult(
            result_id=result_id,
            task=self.config.task,
            model=self.config.model,
            performance_metrics=performance_metrics,
            training_history=training_result['training_history'],
            model_state={
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            },
            created_at=datetime.now()
        )
        
        # Store result
        self.cv_results[result_id] = cv_result
        
        # Save to database
        self._save_cv_result(cv_result)
        
        experiment_time = time.time() - start_time
        console.print(f"[green]CV experiment completed in {experiment_time:.2f} seconds[/green]")
        console.print(f"[blue]Final accuracy: {performance_metrics.get('accuracy', 0):.4f}[/blue]")
        
        return cv_result
    
    def _create_data_loader(self, image_paths: List[str], labels: List[int], 
                          is_training: bool = True) -> DataLoader:
        """Create data loader."""
        class ImageDataset(Dataset):
            def __init__(self, image_paths, labels, processor, augmenter, is_training):
                self.image_paths = image_paths
                self.labels = labels or [0] * len(image_paths)
                self.processor = processor
                self.augmenter = augmenter
                self.is_training = is_training
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                # Load image
                image_data = self.processor.load_image(self.image_paths[idx])
                
                # Apply augmentation if training
                if self.is_training:
                    image_tensor = self.augmenter.augment_image(image_data.image_array)
                else:
                    # Basic preprocessing for validation
                    image_tensor = torch.from_numpy(image_data.image_array).permute(2, 0, 1).float() / 255.0
                    image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
                
                return image_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
        
        dataset = ImageDataset(image_paths, labels, self.image_processor, self.data_augmentation, is_training)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=is_training)
    
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance."""
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _save_cv_result(self, result: CVModelResult):
        """Save CV result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cv_models 
                (model_id, task, model_name, performance_metrics,
                 training_history, model_state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.task.value,
                result.model.value,
                json.dumps(result.performance_metrics),
                json.dumps(result.training_history),
                json.dumps(result.model_state),
                result.created_at.isoformat()
            ))
    
    def visualize_cv_results(self, result: CVModelResult, 
                           output_path: str = None) -> str:
        """Visualize CV training results."""
        if output_path is None:
            output_path = f"cv_training_{result.result_id}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        if 'train_loss' in result.training_history:
            axes[0, 0].plot(result.training_history['train_loss'], label='Train Loss')
            axes[0, 0].plot(result.training_history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Training accuracy
        if 'train_accuracy' in result.training_history:
            axes[0, 1].plot(result.training_history['train_accuracy'], label='Train Acc')
            axes[0, 1].plot(result.training_history['val_accuracy'], label='Val Acc')
            axes[0, 1].set_title('Training Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Performance metrics
        performance_metrics = result.performance_metrics
        metric_names = list(performance_metrics.keys())
        metric_values = list(performance_metrics.values())
        
        axes[1, 0].bar(metric_names, metric_values)
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Model info
        model_state = result.model_state
        info_names = list(model_state.keys())
        info_values = list(model_state.values())
        
        axes[1, 1].bar(info_names, info_values)
        axes[1, 1].set_title('Model Information')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]CV visualization saved: {output_path}[/green]")
        return output_path
    
    def get_cv_summary(self) -> Dict[str, Any]:
        """Get CV system summary."""
        if not self.cv_results:
            return {'total_experiments': 0}
        
        total_experiments = len(self.cv_results)
        
        # Calculate average metrics
        accuracies = [result.performance_metrics.get('accuracy', 0) for result in self.cv_results.values()]
        f1_scores = [result.performance_metrics.get('f1_score', 0) for result in self.cv_results.values()]
        
        avg_accuracy = np.mean(accuracies)
        avg_f1 = np.mean(f1_scores)
        
        # Best performing experiment
        best_result = max(self.cv_results.values(), 
                         key=lambda x: x.performance_metrics.get('accuracy', 0))
        
        return {
            'total_experiments': total_experiments,
            'average_accuracy': avg_accuracy,
            'average_f1_score': avg_f1,
            'best_accuracy': best_result.performance_metrics.get('accuracy', 0),
            'best_experiment_id': best_result.result_id,
            'tasks_used': list(set(result.task.value for result in self.cv_results.values())),
            'models_used': list(set(result.model.value for result in self.cv_results.values()))
        }

def main():
    """Main function for CV CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Computer Vision System")
    parser.add_argument("--task", type=str,
                       choices=["classification", "object_detection", "semantic_segmentation"],
                       default="classification", help="CV task")
    parser.add_argument("--model", type=str,
                       choices=["resnet50", "vgg16", "densenet121", "efficientnet_b0"],
                       default="resnet50", help="CV model")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224],
                       help="Image size (height width)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--num-classes", type=int, default=1000,
                       help="Number of classes")
    parser.add_argument("--augmentation", type=str,
                       choices=["basic", "advanced", "mixup"],
                       default="advanced", help="Augmentation strategy")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create CV configuration
    config = CVConfig(
        task=CVTask(args.task),
        model=CVModel(args.model),
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_classes=args.num_classes,
        augmentation_strategy=AugmentationStrategy(args.augmentation),
        device=args.device
    )
    
    # Create CV system
    cv_system = CVSystem(config)
    
    # Create sample data (in practice, you'd load real image paths)
    train_data = [f"sample_image_{i}.jpg" for i in range(100)]
    val_data = [f"sample_image_{i}.jpg" for i in range(100, 120)]
    labels = list(range(10)) * 10  # 10 classes
    
    # Run CV experiment
    result = cv_system.run_cv_experiment(train_data, val_data, labels)
    
    # Show results
    console.print(f"[green]CV experiment completed[/green]")
    console.print(f"[blue]Task: {result.task.value}[/blue]")
    console.print(f"[blue]Model: {result.model.value}[/blue]")
    console.print(f"[blue]Final accuracy: {result.performance_metrics.get('accuracy', 0):.4f}[/blue]")
    
    # Create visualization
    cv_system.visualize_cv_results(result)
    
    # Show summary
    summary = cv_system.get_cv_summary()
    console.print(f"[blue]Summary: {summary}[/blue]")

if __name__ == "__main__":
    main()
