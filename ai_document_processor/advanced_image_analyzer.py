"""
Advanced Image Analysis and Computer Vision Module
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from transformers import pipeline, CLIPProcessor, CLIPModel
from ultralytics import YOLO
import easyocr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from config import settings
from models import ProcessingStatus

logger = logging.getLogger(__name__)


class AdvancedImageAnalyzer:
    """Advanced Image Analysis and Computer Vision Engine"""
    
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.yolo_model = None
        self.ocr_reader = None
        self.image_classifier = None
        self.face_detector = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the advanced image analyzer"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing Advanced Image Analyzer...")
            
            # Initialize CLIP model for image-text understanding
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Initialize YOLO for object detection
            self.yolo_model = YOLO('yolov8n.pt')
            
            # Initialize OCR reader
            self.ocr_reader = easyocr.Reader(['en', 'es', 'fr', 'de', 'it', 'pt'])
            
            # Initialize image classification pipeline
            self.image_classifier = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                return_all_scores=True
            )
            
            # Initialize face detection
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            self.initialized = True
            logger.info("Advanced Image Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing advanced image analyzer: {e}")
            raise
    
    async def analyze_image(self, image_path: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """Comprehensive image analysis"""
        if not self.initialized:
            await self.initialize()
        
        if analysis_types is None:
            analysis_types = ["objects", "text", "faces", "classification", "features"]
        
        start_time = time.time()
        results = {
            "image_path": image_path,
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Perform requested analyses
            if "objects" in analysis_types:
                results["object_detection"] = await self._detect_objects(image)
            
            if "text" in analysis_types:
                results["text_extraction"] = await self._extract_text_from_image(image_path)
            
            if "faces" in analysis_types:
                results["face_analysis"] = await self._analyze_faces(image)
            
            if "classification" in analysis_types:
                results["image_classification"] = await self._classify_image(image_path)
            
            if "features" in analysis_types:
                results["visual_features"] = await self._extract_visual_features(image)
            
            if "similarity" in analysis_types:
                results["similarity_analysis"] = await self._analyze_image_similarity(image_path)
            
            if "quality" in analysis_types:
                results["quality_analysis"] = await self._analyze_image_quality(image)
            
            if "aesthetics" in analysis_types:
                results["aesthetic_analysis"] = await self._analyze_aesthetics(image)
            
            if "color_analysis" in analysis_types:
                results["color_analysis"] = await self._analyze_colors(image)
            
            if "composition" in analysis_types:
                results["composition_analysis"] = await self._analyze_composition(image)
            
            results["processing_time"] = time.time() - start_time
            results["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["processing_time"] = time.time() - start_time
        
        return results
    
    async def _detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects in image using YOLO"""
        try:
            # Run YOLO detection
            results = self.yolo_model(image)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        detections.append({
                            "class": class_name,
                            "confidence": float(confidence),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "area": float((x2 - x1) * (y2 - y1))
                        })
            
            return {
                "total_objects": len(detections),
                "detections": detections,
                "object_classes": list(set([d["class"] for d in detections]))
            }
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return {"total_objects": 0, "detections": [], "object_classes": []}
    
    async def _extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image using OCR"""
        try:
            # Use EasyOCR for text extraction
            results = self.ocr_reader.readtext(image_path)
            
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Confidence threshold
                    extracted_text.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": bbox
                    })
            
            # Combine all text
            full_text = " ".join([item["text"] for item in extracted_text])
            
            return {
                "full_text": full_text,
                "text_blocks": extracted_text,
                "total_text_blocks": len(extracted_text),
                "average_confidence": np.mean([item["confidence"] for item in extracted_text]) if extracted_text else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return {"full_text": "", "text_blocks": [], "total_text_blocks": 0, "average_confidence": 0.0}
    
    async def _analyze_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze faces in image"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            
            face_analysis = []
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = image[y:y+h, x:x+w]
                
                # Analyze face properties
                face_properties = {
                    "bbox": [int(x), int(y), int(x+w), int(y+h)],
                    "size": int(w * h),
                    "aspect_ratio": float(w / h),
                    "center": [int(x + w/2), int(y + h/2)]
                }
                
                # Additional face analysis could be added here
                # (emotion detection, age estimation, etc.)
                
                face_analysis.append(face_properties)
            
            return {
                "total_faces": len(faces),
                "faces": face_analysis,
                "face_density": len(faces) / (image.shape[0] * image.shape[1]) * 1000000  # faces per megapixel
            }
            
        except Exception as e:
            logger.error(f"Error analyzing faces: {e}")
            return {"total_faces": 0, "faces": [], "face_density": 0.0}
    
    async def _classify_image(self, image_path: str) -> Dict[str, Any]:
        """Classify image content"""
        try:
            # Use image classification pipeline
            results = self.image_classifier(image_path)
            
            return {
                "predictions": results,
                "top_prediction": max(results, key=lambda x: x["score"]) if results else None,
                "confidence": max(results, key=lambda x: x["score"])["score"] if results else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return {"predictions": [], "top_prediction": None, "confidence": 0.0}
    
    async def _extract_visual_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract visual features from image"""
        try:
            features = {}
            
            # Color features
            features["dominant_colors"] = await self._get_dominant_colors(image)
            features["color_histogram"] = await self._get_color_histogram(image)
            
            # Texture features
            features["texture_features"] = await self._get_texture_features(image)
            
            # Shape features
            features["shape_features"] = await self._get_shape_features(image)
            
            # Edge features
            features["edge_features"] = await self._get_edge_features(image)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {}
    
    async def _get_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Get dominant colors using K-means clustering"""
        try:
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Calculate color percentages
            color_counts = np.bincount(labels)
            color_percentages = color_counts / len(labels) * 100
            
            dominant_colors = []
            for i, color in enumerate(colors):
                dominant_colors.append({
                    "rgb": color.tolist(),
                    "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    "percentage": float(color_percentages[i])
                })
            
            return dominant_colors
            
        except Exception as e:
            logger.error(f"Error getting dominant colors: {e}")
            return []
    
    async def _get_color_histogram(self, image: np.ndarray) -> Dict[str, Any]:
        """Get color histogram features"""
        try:
            # Calculate histograms for each color channel
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            # Normalize histograms
            hist_b = hist_b.flatten() / hist_b.sum()
            hist_g = hist_g.flatten() / hist_g.sum()
            hist_r = hist_r.flatten() / hist_r.sum()
            
            return {
                "blue_histogram": hist_b.tolist(),
                "green_histogram": hist_g.tolist(),
                "red_histogram": hist_r.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error getting color histogram: {e}")
            return {}
    
    async def _get_texture_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract texture features using Local Binary Patterns"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Local Binary Pattern
            from skimage.feature import local_binary_pattern
            
            # Parameters for LBP
            radius = 1
            n_points = 8 * radius
            
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate LBP histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            # Calculate texture statistics
            texture_features = {
                "lbp_histogram": hist.tolist(),
                "texture_energy": float(np.sum(hist**2)),
                "texture_entropy": float(-np.sum(hist * np.log2(hist + 1e-7))),
                "texture_contrast": float(np.sum(hist * np.arange(len(hist))**2))
            }
            
            return texture_features
            
        except Exception as e:
            logger.error(f"Error getting texture features: {e}")
            return {}
    
    async def _get_shape_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract shape features from image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shape_features = {
                "total_contours": len(contours),
                "largest_contour_area": 0.0,
                "contour_areas": [],
                "contour_perimeters": []
            }
            
            if contours:
                # Calculate contour properties
                areas = [cv2.contourArea(contour) for contour in contours]
                perimeters = [cv2.arcLength(contour, True) for contour in contours]
                
                shape_features.update({
                    "largest_contour_area": float(max(areas)),
                    "contour_areas": [float(area) for area in areas],
                    "contour_perimeters": [float(perimeter) for perimeter in perimeters],
                    "average_contour_area": float(np.mean(areas)),
                    "average_contour_perimeter": float(np.mean(perimeters))
                })
            
            return shape_features
            
        except Exception as e:
            logger.error(f"Error getting shape features: {e}")
            return {}
    
    async def _get_edge_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract edge features from image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge statistics
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_density = edge_pixels / total_pixels
            
            # Calculate edge orientation histogram
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            orientation = np.arctan2(sobel_y, sobel_x)
            
            # Create orientation histogram
            hist, _ = np.histogram(orientation, bins=36, range=(-np.pi, np.pi))
            hist = hist.astype(float) / hist.sum()
            
            edge_features = {
                "edge_density": float(edge_density),
                "total_edge_pixels": int(edge_pixels),
                "orientation_histogram": hist.tolist(),
                "dominant_orientation": float(np.argmax(hist) * 10 - 180)  # Convert to degrees
            }
            
            return edge_features
            
        except Exception as e:
            logger.error(f"Error getting edge features: {e}")
            return {}
    
    async def _analyze_image_similarity(self, image_path: str) -> Dict[str, Any]:
        """Analyze image similarity using CLIP embeddings"""
        try:
            # Load and process image
            image = Image.open(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Get image embeddings
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return {
                "image_embedding": image_features.tolist()[0],
                "embedding_dimension": len(image_features[0])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image similarity: {e}")
            return {"image_embedding": [], "embedding_dimension": 0}
    
    async def _analyze_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image quality metrics"""
        try:
            # Convert to grayscale for some calculations
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = gray.std()
            
            # Calculate noise level (simplified)
            noise_level = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
            
            # Calculate dynamic range
            dynamic_range = np.max(gray) - np.min(gray)
            
            quality_metrics = {
                "sharpness": float(laplacian_var),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "noise_level": float(noise_level),
                "dynamic_range": float(dynamic_range),
                "resolution": f"{image.shape[1]}x{image.shape[0]}",
                "aspect_ratio": float(image.shape[1] / image.shape[0])
            }
            
            # Quality score (0-100)
            quality_score = min(100, max(0, 
                (laplacian_var / 1000) * 30 +  # Sharpness contribution
                (contrast / 100) * 25 +        # Contrast contribution
                (1 - noise_level / 50) * 25 +  # Noise contribution
                (dynamic_range / 255) * 20     # Dynamic range contribution
            ))
            
            quality_metrics["quality_score"] = float(quality_score)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing image quality: {e}")
            return {}
    
    async def _analyze_aesthetics(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze aesthetic properties of image"""
        try:
            # Convert to PIL Image for aesthetic analysis
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Rule of thirds analysis
            width, height = pil_image.size
            rule_of_thirds_score = self._calculate_rule_of_thirds_score(image)
            
            # Color harmony analysis
            color_harmony_score = self._calculate_color_harmony_score(image)
            
            # Symmetry analysis
            symmetry_score = self._calculate_symmetry_score(image)
            
            # Balance analysis
            balance_score = self._calculate_balance_score(image)
            
            aesthetic_metrics = {
                "rule_of_thirds_score": rule_of_thirds_score,
                "color_harmony_score": color_harmony_score,
                "symmetry_score": symmetry_score,
                "balance_score": balance_score,
                "overall_aesthetic_score": (rule_of_thirds_score + color_harmony_score + symmetry_score + balance_score) / 4
            }
            
            return aesthetic_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing aesthetics: {e}")
            return {}
    
    def _calculate_rule_of_thirds_score(self, image: np.ndarray) -> float:
        """Calculate rule of thirds score"""
        try:
            height, width = image.shape[:2]
            
            # Define rule of thirds lines
            vertical_lines = [width // 3, 2 * width // 3]
            horizontal_lines = [height // 3, 2 * height // 3]
            
            # This is a simplified implementation
            # In practice, you'd analyze the distribution of important elements
            # along these lines
            
            return 0.7  # Placeholder score
            
        except Exception as e:
            logger.error(f"Error calculating rule of thirds score: {e}")
            return 0.0
    
    def _calculate_color_harmony_score(self, image: np.ndarray) -> float:
        """Calculate color harmony score"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Analyze hue distribution
            hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hue_hist = hue_hist.flatten() / hue_hist.sum()
            
            # Calculate color harmony based on hue distribution
            # This is a simplified implementation
            harmony_score = 1.0 - np.std(hue_hist)
            
            return float(max(0, min(1, harmony_score)))
            
        except Exception as e:
            logger.error(f"Error calculating color harmony score: {e}")
            return 0.0
    
    def _calculate_symmetry_score(self, image: np.ndarray) -> float:
        """Calculate symmetry score"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Check horizontal symmetry
            height, width = gray.shape
            top_half = gray[:height//2, :]
            bottom_half = cv2.flip(gray[height//2:, :], 0)
            
            # Resize if necessary
            if top_half.shape != bottom_half.shape:
                bottom_half = cv2.resize(bottom_half, (top_half.shape[1], top_half.shape[0]))
            
            # Calculate similarity
            horizontal_symmetry = 1.0 - np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float))) / 255.0
            
            return float(max(0, min(1, horizontal_symmetry)))
            
        except Exception as e:
            logger.error(f"Error calculating symmetry score: {e}")
            return 0.0
    
    def _calculate_balance_score(self, image: np.ndarray) -> float:
        """Calculate visual balance score"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate center of mass
            moments = cv2.moments(gray)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                
                # Calculate distance from image center
                image_center_x = gray.shape[1] / 2
                image_center_y = gray.shape[0] / 2
                
                distance = np.sqrt((cx - image_center_x)**2 + (cy - image_center_y)**2)
                max_distance = np.sqrt((image_center_x)**2 + (image_center_y)**2)
                
                balance_score = 1.0 - (distance / max_distance)
                return float(max(0, min(1, balance_score)))
            
            return 0.5  # Default score
            
        except Exception as e:
            logger.error(f"Error calculating balance score: {e}")
            return 0.0
    
    async def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive color analysis"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Calculate color statistics
            color_stats = {
                "rgb_mean": np.mean(image, axis=(0, 1)).tolist(),
                "rgb_std": np.std(image, axis=(0, 1)).tolist(),
                "hsv_mean": np.mean(hsv, axis=(0, 1)).tolist(),
                "lab_mean": np.mean(lab, axis=(0, 1)).tolist()
            }
            
            # Calculate color diversity
            pixels = image.reshape(-1, 3)
            unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))))
            color_diversity = unique_colors / (image.shape[0] * image.shape[1])
            
            color_stats["color_diversity"] = float(color_diversity)
            color_stats["unique_colors"] = int(unique_colors)
            
            return color_stats
            
        except Exception as e:
            logger.error(f"Error analyzing colors: {e}")
            return {}
    
    async def _analyze_composition(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image composition"""
        try:
            height, width = image.shape[:2]
            
            # Calculate aspect ratio
            aspect_ratio = width / height
            
            # Analyze image orientation
            orientation = "landscape" if aspect_ratio > 1.2 else "portrait" if aspect_ratio < 0.8 else "square"
            
            # Calculate center of mass
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            moments = cv2.moments(gray)
            
            composition_analysis = {
                "aspect_ratio": float(aspect_ratio),
                "orientation": orientation,
                "resolution": f"{width}x{height}",
                "total_pixels": int(width * height)
            }
            
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                
                composition_analysis.update({
                    "center_of_mass": [float(cx), float(cy)],
                    "center_of_mass_normalized": [float(cx / width), float(cy / height)]
                })
            
            return composition_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing composition: {e}")
            return {}


# Global advanced image analyzer instance
advanced_image_analyzer = AdvancedImageAnalyzer()


async def initialize_advanced_image_analyzer():
    """Initialize the advanced image analyzer"""
    await advanced_image_analyzer.initialize()














