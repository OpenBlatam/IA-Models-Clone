"""
Computer Vision Engine for Export IA
Advanced computer vision with object detection, segmentation, and image processing
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
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import torchvision.transforms as transforms
from torchvision.models import detection, segmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ultralytics
from ultralytics import YOLO
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

logger = logging.getLogger(__name__)

@dataclass
class ComputerVisionConfig:
    """Configuration for computer vision"""
    # Task types
    task_type: str = "detection"  # detection, segmentation, classification, keypoints, tracking
    
    # Model types
    model_type: str = "yolo"  # yolo, rcnn, faster_rcnn, mask_rcnn, retinanet, detr
    
    # YOLO parameters
    yolo_version: str = "yolov8"  # yolov5, yolov8, yolov9
    yolo_model_size: str = "n"  # n, s, m, l, x
    
    # R-CNN parameters
    rcnn_backbone: str = "resnet50"  # resnet50, resnet101, resnext50, efficientnet
    rcnn_num_classes: int = 80
    rcnn_confidence_threshold: float = 0.5
    rcnn_nms_threshold: float = 0.4
    
    # Detection parameters
    detection_confidence_threshold: float = 0.5
    detection_nms_threshold: float = 0.4
    detection_max_detections: int = 100
    
    # Segmentation parameters
    segmentation_threshold: float = 0.5
    segmentation_mask_threshold: float = 0.5
    
    # Classification parameters
    classification_top_k: int = 5
    classification_confidence_threshold: float = 0.1
    
    # Image preprocessing
    image_size: Tuple[int, int] = (640, 640)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Data augmentation
    enable_augmentation: bool = True
    augmentation_probability: float = 0.5
    augmentation_types: List[str] = None  # horizontal_flip, vertical_flip, rotation, brightness, contrast
    
    # Post-processing
    enable_nms: bool = True
    enable_soft_nms: bool = False
    enable_multi_scale: bool = False
    
    # Visualization
    enable_visualization: bool = True
    visualization_confidence_threshold: float = 0.3
    visualization_colors: List[Tuple[int, int, int]] = None
    
    # Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    num_workers: int = 4

class ObjectDetector:
    """Object detection using various models"""
    
    def __init__(self, config: ComputerVisionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.transform = None
        
        # Initialize model
        self._initialize_model()
        
        # Initialize transforms
        self._initialize_transforms()
        
    def _initialize_model(self):
        """Initialize detection model"""
        
        if self.config.model_type == "yolo":
            self._initialize_yolo()
        elif self.config.model_type == "rcnn":
            self._initialize_rcnn()
        elif self.config.model_type == "faster_rcnn":
            self._initialize_faster_rcnn()
        elif self.config.model_type == "mask_rcnn":
            self._initialize_mask_rcnn()
        elif self.config.model_type == "retinanet":
            self._initialize_retinanet()
        elif self.config.model_type == "detr":
            self._initialize_detr()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
    def _initialize_yolo(self):
        """Initialize YOLO model"""
        
        if self.config.yolo_version == "yolov8":
            model_name = f"yolov8{self.config.yolo_model_size}.pt"
            self.model = YOLO(model_name)
        elif self.config.yolo_version == "yolov5":
            # YOLOv5 implementation
            self.model = self._build_yolov5()
        else:
            raise ValueError(f"Unsupported YOLO version: {self.config.yolo_version}")
            
    def _build_yolov5(self):
        """Build YOLOv5 model"""
        
        # Simplified YOLOv5 implementation
        class YOLOv5(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 6, 2, 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, 2, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256, 512, 3, 2, 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(512, 1024, 3, 2, 1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU()
                )
                
                self.head = nn.Sequential(
                    nn.Conv2d(1024, 512, 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(512, 256, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256, 85, 1)  # 80 classes + 5 (x, y, w, h, conf)
                )
                
            def forward(self, x):
                x = self.backbone(x)
                x = self.head(x)
                return x
                
        return YOLOv5()
        
    def _initialize_rcnn(self):
        """Initialize R-CNN model"""
        
        self.model = detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            num_classes=self.config.rcnn_num_classes
        )
        
    def _initialize_faster_rcnn(self):
        """Initialize Faster R-CNN model"""
        
        self.model = detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            num_classes=self.config.rcnn_num_classes
        )
        
    def _initialize_mask_rcnn(self):
        """Initialize Mask R-CNN model"""
        
        self.model = detection.maskrcnn_resnet50_fpn(
            pretrained=True,
            num_classes=self.config.rcnn_num_classes
        )
        
    def _initialize_retinanet(self):
        """Initialize RetinaNet model"""
        
        self.model = detection.retinanet_resnet50_fpn(
            pretrained=True,
            num_classes=self.config.rcnn_num_classes
        )
        
    def _initialize_detr(self):
        """Initialize DETR model"""
        
        # Simplified DETR implementation
        class DETR(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torchvision.models.resnet50(pretrained=True)
                self.backbone.fc = nn.Identity()
                
                self.transformer = nn.Transformer(
                    d_model=512,
                    nhead=8,
                    num_encoder_layers=6,
                    num_decoder_layers=6
                )
                
                self.classifier = nn.Linear(512, self.config.rcnn_num_classes + 1)
                self.bbox_regressor = nn.Linear(512, 4)
                
            def forward(self, x):
                features = self.backbone(x)
                features = features.unsqueeze(1)
                
                transformer_output = self.transformer(features, features)
                
                class_logits = self.classifier(transformer_output)
                bbox_coords = self.bbox_regressor(transformer_output)
                
                return class_logits, bbox_coords
                
        self.model = DETR()
        
    def _initialize_transforms(self):
        """Initialize image transforms"""
        
        self.transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
        
    def detect(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """Detect objects in image"""
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
            
        # Apply transforms
        if self.config.model_type == "yolo":
            # YOLO handles its own preprocessing
            results = self.model(image)
            
            # Parse YOLO results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = {
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'confidence': box.conf[0].cpu().numpy(),
                            'class_id': int(box.cls[0].cpu().numpy()),
                            'class_name': self.model.names[int(box.cls[0].cpu().numpy())]
                        }
                        detections.append(detection)
                        
            return {
                'detections': detections,
                'num_detections': len(detections)
            }
            
        else:
            # Other models
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.config.model_type == "detr":
                    class_logits, bbox_coords = self.model(input_tensor)
                    # Parse DETR results
                    detections = self._parse_detr_results(class_logits, bbox_coords)
                else:
                    outputs = self.model(input_tensor)
                    detections = self._parse_rcnn_results(outputs)
                    
            return {
                'detections': detections,
                'num_detections': len(detections)
            }
            
    def _parse_rcnn_results(self, outputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
        """Parse R-CNN model results"""
        
        detections = []
        
        for output in outputs:
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            for i in range(len(boxes)):
                if scores[i] > self.config.detection_confidence_threshold:
                    detection = {
                        'bbox': boxes[i],
                        'confidence': scores[i],
                        'class_id': labels[i],
                        'class_name': f'class_{labels[i]}'
                    }
                    detections.append(detection)
                    
        return detections
        
    def _parse_detr_results(self, class_logits: torch.Tensor, bbox_coords: torch.Tensor) -> List[Dict[str, Any]]:
        """Parse DETR model results"""
        
        detections = []
        
        # Apply softmax to class logits
        class_probs = F.softmax(class_logits, dim=-1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(class_probs, k=10, dim=-1)
        
        for i in range(top_probs.size(1)):
            if top_probs[0, i] > self.config.detection_confidence_threshold:
                class_id = top_indices[0, i].item()
                if class_id < self.config.rcnn_num_classes:  # Skip background class
                    detection = {
                        'bbox': bbox_coords[0, i].cpu().numpy(),
                        'confidence': top_probs[0, i].item(),
                        'class_id': class_id,
                        'class_name': f'class_{class_id}'
                    }
                    detections.append(detection)
                    
        return detections
        
    def visualize_detections(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                           save_path: str = None) -> np.ndarray:
        """Visualize object detections"""
        
        if not self.config.enable_visualization:
            return image
            
        # Create visualization
        vis_image = image.copy()
        
        # Define colors
        colors = self.config.visualization_colors or [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]
        
        for detection in detections:
            if detection['confidence'] > self.config.visualization_confidence_threshold:
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                color = colors[detection['class_id'] % len(colors)]
                
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                          
        if save_path:
            cv2.imwrite(save_path, vis_image)
            
        return vis_image

class ImageSegmenter:
    """Image segmentation using various models"""
    
    def __init__(self, config: ComputerVisionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize segmentation model"""
        
        if self.config.model_type == "mask_rcnn":
            self.model = segmentation.maskrcnn_resnet50_fpn(pretrained=True)
        elif self.config.model_type == "deeplabv3":
            self.model = segmentation.deeplabv3_resnet50(pretrained=True)
        elif self.config.model_type == "fcn":
            self.model = segmentation.fcn_resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported segmentation model: {self.config.model_type}")
            
    def segment(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """Segment image"""
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
            
        # Transform image
        transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
        # Parse results
        if self.config.model_type == "mask_rcnn":
            return self._parse_mask_rcnn_results(outputs)
        else:
            return self._parse_segmentation_results(outputs)
            
    def _parse_mask_rcnn_results(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Parse Mask R-CNN results"""
        
        segments = []
        
        for output in outputs:
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            masks = output['masks'].cpu().numpy()
            
            for i in range(len(boxes)):
                if scores[i] > self.config.segmentation_threshold:
                    segment = {
                        'bbox': boxes[i],
                        'confidence': scores[i],
                        'class_id': labels[i],
                        'mask': masks[i][0],
                        'class_name': f'class_{labels[i]}'
                    }
                    segments.append(segment)
                    
        return {
            'segments': segments,
            'num_segments': len(segments)
        }
        
    def _parse_segmentation_results(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Parse segmentation results"""
        
        # Get predicted segmentation
        predicted = torch.argmax(outputs['out'], dim=1)
        predicted = predicted.squeeze(0).cpu().numpy()
        
        return {
            'segmentation_map': predicted,
            'num_classes': len(np.unique(predicted))
        }
        
    def visualize_segmentation(self, image: np.ndarray, segmentation_result: Dict[str, Any], 
                             save_path: str = None) -> np.ndarray:
        """Visualize segmentation results"""
        
        if not self.config.enable_visualization:
            return image
            
        vis_image = image.copy()
        
        if 'segments' in segmentation_result:
            # Instance segmentation
            for segment in segmentation_result['segments']:
                if segment['confidence'] > self.config.segmentation_threshold:
                    mask = segment['mask']
                    mask = (mask > self.config.segmentation_mask_threshold).astype(np.uint8)
                    
                    # Create colored mask
                    color = np.random.randint(0, 255, 3)
                    colored_mask = np.zeros_like(vis_image)
                    colored_mask[mask == 1] = color
                    
                    # Blend with original image
                    vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
                    
        elif 'segmentation_map' in segmentation_result:
            # Semantic segmentation
            seg_map = segmentation_result['segmentation_map']
            
            # Create colored segmentation map
            colored_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
            
            for class_id in np.unique(seg_map):
                if class_id > 0:  # Skip background
                    mask = seg_map == class_id
                    color = np.random.randint(0, 255, 3)
                    colored_seg[mask] = color
                    
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 0.7, colored_seg, 0.3, 0)
            
        if save_path:
            cv2.imwrite(save_path, vis_image)
            
        return vis_image

class ImageClassifier:
    """Image classification using various models"""
    
    def __init__(self, config: ComputerVisionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize classification model"""
        
        if self.config.rcnn_backbone == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
        elif self.config.rcnn_backbone == "resnet101":
            self.model = torchvision.models.resnet101(pretrained=True)
        elif self.config.rcnn_backbone == "efficientnet":
            self.model = torchvision.models.efficientnet_b0(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {self.config.rcnn_backbone}")
            
        # Modify final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.config.rcnn_num_classes)
        
    def classify(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """Classify image"""
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
            
        # Transform image
        transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=self.config.classification_top_k, dim=1)
        
        predictions = []
        for i in range(self.config.classification_top_k):
            if top_probs[0, i] > self.config.classification_confidence_threshold:
                prediction = {
                    'class_id': top_indices[0, i].item(),
                    'confidence': top_probs[0, i].item(),
                    'class_name': f'class_{top_indices[0, i].item()}'
                }
                predictions.append(prediction)
                
        return {
            'predictions': predictions,
            'num_predictions': len(predictions)
        }

class ComputerVisionEngine:
    """Main Computer Vision Engine"""
    
    def __init__(self, config: ComputerVisionConfig):
        self.config = config
        
        # Initialize components
        if config.task_type == "detection":
            self.detector = ObjectDetector(config)
        elif config.task_type == "segmentation":
            self.segmenter = ImageSegmenter(config)
        elif config.task_type == "classification":
            self.classifier = ImageClassifier(config)
        else:
            raise ValueError(f"Unsupported task type: {config.task_type}")
            
        # Results storage
        self.results = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        
    def process_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor], 
                     save_path: str = None) -> Dict[str, Any]:
        """Process image based on task type"""
        
        start_time = time.time()
        
        if self.config.task_type == "detection":
            results = self.detector.detect(image)
            
            # Visualize if enabled
            if self.config.enable_visualization and isinstance(image, np.ndarray):
                vis_image = self.detector.visualize_detections(image, results['detections'], save_path)
                results['visualization'] = vis_image
                
        elif self.config.task_type == "segmentation":
            results = self.segmenter.segment(image)
            
            # Visualize if enabled
            if self.config.enable_visualization and isinstance(image, np.ndarray):
                vis_image = self.segmenter.visualize_segmentation(image, results, save_path)
                results['visualization'] = vis_image
                
        elif self.config.task_type == "classification":
            results = self.classifier.classify(image)
            
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
            
        # Add processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        # Store results
        self.results[self.config.task_type].append(results)
        
        return results
        
    def process_batch(self, images: List[Union[np.ndarray, Image.Image, torch.Tensor]], 
                     save_paths: List[str] = None) -> List[Dict[str, Any]]:
        """Process batch of images"""
        
        results = []
        
        for i, image in enumerate(images):
            save_path = save_paths[i] if save_paths else None
            result = self.process_image(image, save_path)
            results.append(result)
            
        return results
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        metrics = {
            'total_images_processed': sum(len(results) for results in self.results.values()),
            'average_processing_time': 0.0,
            'task_type': self.config.task_type,
            'model_type': self.config.model_type
        }
        
        # Calculate average processing time
        all_times = []
        for results in self.results.values():
            for result in results:
                if 'processing_time' in result:
                    all_times.append(result['processing_time'])
                    
        if all_times:
            metrics['average_processing_time'] = np.mean(all_times)
            
        return metrics
        
    def save_results(self, filepath: str):
        """Save results to file"""
        
        results_data = {
            'results': dict(self.results),
            'performance_metrics': self.get_performance_metrics(),
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, default=str)
            
    def load_results(self, filepath: str):
        """Load results from file"""
        
        with open(filepath, 'r') as f:
            results_data = json.load(f)
            
        self.results = defaultdict(list, results_data['results'])

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test computer vision
    print("Testing Computer Vision Engine...")
    
    # Create config
    config = ComputerVisionConfig(
        task_type="detection",
        model_type="yolo",
        yolo_version="yolov8",
        yolo_model_size="n",
        enable_visualization=True
    )
    
    # Create engine
    cv_engine = ComputerVisionEngine(config)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test detection
    print("Testing object detection...")
    results = cv_engine.process_image(dummy_image)
    print(f"Detection results: {results}")
    
    # Test batch processing
    print("Testing batch processing...")
    batch_images = [dummy_image, dummy_image, dummy_image]
    batch_results = cv_engine.process_batch(batch_images)
    print(f"Batch results: {len(batch_results)} images processed")
    
    # Get performance metrics
    metrics = cv_engine.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    print("\nComputer vision engine initialized successfully!")
























