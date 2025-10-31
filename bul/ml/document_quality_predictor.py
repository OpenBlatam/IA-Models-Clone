"""
BUL Document Quality Predictor
==============================

Advanced ML models for document quality prediction and optimization.
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import uuid
import os
from pathlib import Path

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

class QualityMetric(str, Enum):
    """Document quality metrics"""
    READABILITY = "readability"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    PROFESSIONALISM = "professionalism"
    STRUCTURE = "structure"
    LANGUAGE_QUALITY = "language_quality"

class ModelType(str, Enum):
    """ML model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    ENSEMBLE = "ensemble"

@dataclass
class DocumentFeatures:
    """Document features for ML prediction"""
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    avg_word_length: float
    readability_score: float
    business_area_score: float
    document_type_score: float
    language_complexity: float
    structure_score: float
    keyword_density: float
    technical_terms_count: int
    passive_voice_ratio: float
    active_voice_ratio: float
    punctuation_density: float
    capitalization_ratio: float
    numbers_count: int
    bullet_points_count: int
    headings_count: int
    metadata_score: float

@dataclass
class QualityPrediction:
    """Quality prediction result"""
    document_id: str
    predicted_quality: float
    confidence_score: float
    quality_breakdown: Dict[str, float]
    recommendations: List[str]
    model_used: str
    prediction_timestamp: datetime
    feature_importance: Dict[str, float]

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    model_type: ModelType
    mse: float
    rmse: float
    mae: float
    r2_score: float
    cross_val_score: float
    training_time: float
    prediction_time: float
    last_trained: datetime
    is_active: bool

class DocumentQualityPredictor:
    """Document Quality Predictor using ML models"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        
        # ML Models
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        
        # Training data
        self.training_data: List[Dict[str, Any]] = []
        self.feature_columns: List[str] = []
        
        # Model paths
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize different model types
            self.models = {
                "random_forest": RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                "linear_regression": LinearRegression(),
                "ridge_regression": Ridge(alpha=1.0),
                "ensemble": None  # Will be created dynamically
            }
            
            # Initialize scalers
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
            
            # Initialize label encoders
            self.label_encoders["business_area"] = LabelEncoder()
            self.label_encoders["document_type"] = LabelEncoder()
            
            # Load existing models if available
            self._load_models()
            
            # Initialize feature columns
            self.feature_columns = [
                'word_count', 'sentence_count', 'paragraph_count',
                'avg_sentence_length', 'avg_word_length', 'readability_score',
                'business_area_score', 'document_type_score', 'language_complexity',
                'structure_score', 'keyword_density', 'technical_terms_count',
                'passive_voice_ratio', 'active_voice_ratio', 'punctuation_density',
                'capitalization_ratio', 'numbers_count', 'bullet_points_count',
                'headings_count', 'metadata_score'
            ]
            
            self.logger.info("Document Quality Predictor initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize Document Quality Predictor: {e}")
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            for model_name in self.models.keys():
                model_path = self.models_dir / f"{model_name}_model.joblib"
                scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
                
                if model_path.exists() and scaler_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                    self.logger.info(f"Loaded {model_name} model from disk")
        
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                if model is not None:
                    model_path = self.models_dir / f"{model_name}_model.joblib"
                    scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
                    
                    joblib.dump(model, model_path)
                    joblib.dump(self.scalers[model_name], scaler_path)
                    self.logger.info(f"Saved {model_name} model to disk")
        
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def extract_document_features(self, content: str, business_area: BusinessArea, document_type: DocumentType) -> DocumentFeatures:
        """Extract features from document content"""
        try:
            # Basic text statistics
            words = content.split()
            sentences = content.split('.')
            paragraphs = content.split('\n\n')
            
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            paragraph_count = len([p for p in paragraphs if p.strip()])
            
            # Average lengths
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
            
            # Readability score (simplified Flesch Reading Ease)
            readability_score = self._calculate_readability_score(content)
            
            # Business area and document type scores
            business_area_score = self._get_business_area_score(business_area)
            document_type_score = self._get_document_type_score(document_type)
            
            # Language complexity
            language_complexity = self._calculate_language_complexity(content)
            
            # Structure score
            structure_score = self._calculate_structure_score(content)
            
            # Keyword density
            keyword_density = self._calculate_keyword_density(content)
            
            # Technical terms count
            technical_terms_count = self._count_technical_terms(content)
            
            # Voice ratios
            passive_voice_ratio = self._calculate_passive_voice_ratio(content)
            active_voice_ratio = 1.0 - passive_voice_ratio
            
            # Punctuation and capitalization
            punctuation_density = content.count('.') + content.count(',') + content.count(';') + content.count(':')
            punctuation_density = punctuation_density / word_count if word_count > 0 else 0
            
            capitalization_ratio = sum(1 for c in content if c.isupper()) / len(content) if content else 0
            
            # Numbers and formatting
            numbers_count = sum(1 for word in words if word.isdigit())
            bullet_points_count = content.count('•') + content.count('-') + content.count('*')
            headings_count = content.count('#') + content.count('**')
            
            # Metadata score
            metadata_score = self._calculate_metadata_score(business_area, document_type)
            
            return DocumentFeatures(
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                avg_sentence_length=avg_sentence_length,
                avg_word_length=avg_word_length,
                readability_score=readability_score,
                business_area_score=business_area_score,
                document_type_score=document_type_score,
                language_complexity=language_complexity,
                structure_score=structure_score,
                keyword_density=keyword_density,
                technical_terms_count=technical_terms_count,
                passive_voice_ratio=passive_voice_ratio,
                active_voice_ratio=active_voice_ratio,
                punctuation_density=punctuation_density,
                capitalization_ratio=capitalization_ratio,
                numbers_count=numbers_count,
                bullet_points_count=bullet_points_count,
                headings_count=headings_count,
                metadata_score=metadata_score
            )
        
        except Exception as e:
            self.logger.error(f"Error extracting document features: {e}")
            return DocumentFeatures(
                word_count=0, sentence_count=0, paragraph_count=0,
                avg_sentence_length=0, avg_word_length=0, readability_score=0,
                business_area_score=0, document_type_score=0, language_complexity=0,
                structure_score=0, keyword_density=0, technical_terms_count=0,
                passive_voice_ratio=0, active_voice_ratio=0, punctuation_density=0,
                capitalization_ratio=0, numbers_count=0, bullet_points_count=0,
                headings_count=0, metadata_score=0
            )
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate simplified readability score"""
        try:
            words = content.split()
            sentences = [s for s in content.split('.') if s.strip()]
            
            if not words or not sentences:
                return 0.0
            
            avg_words_per_sentence = len(words) / len(sentences)
            avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
            
            # Simplified Flesch Reading Ease
            score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            return max(0, min(100, score)) / 100  # Normalize to 0-1
        
        except Exception as e:
            self.logger.error(f"Error calculating readability score: {e}")
            return 0.5
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _get_business_area_score(self, business_area: BusinessArea) -> float:
        """Get business area score"""
        business_area_scores = {
            BusinessArea.FINANCE: 0.9,
            BusinessArea.MARKETING: 0.8,
            BusinessArea.HUMAN_RESOURCES: 0.85,
            BusinessArea.OPERATIONS: 0.8,
            BusinessArea.SALES: 0.75,
            BusinessArea.TECHNOLOGY: 0.9,
            BusinessArea.LEGAL: 0.95,
            BusinessArea.CONSULTING: 0.85
        }
        return business_area_scores.get(business_area, 0.7)
    
    def _get_document_type_score(self, document_type: DocumentType) -> float:
        """Get document type score"""
        document_type_scores = {
            DocumentType.PROPOSAL: 0.9,
            DocumentType.REPORT: 0.85,
            DocumentType.PRESENTATION: 0.8,
            DocumentType.CONTRACT: 0.95,
            DocumentType.MANUAL: 0.8,
            DocumentType.EMAIL: 0.7,
            DocumentType.MEMO: 0.75,
            DocumentType.PLAN: 0.85
        }
        return document_type_scores.get(document_type, 0.7)
    
    def _calculate_language_complexity(self, content: str) -> float:
        """Calculate language complexity score"""
        try:
            words = content.split()
            if not words:
                return 0.0
            
            # Count complex words (more than 3 syllables)
            complex_words = sum(1 for word in words if self._count_syllables(word) > 3)
            complexity_ratio = complex_words / len(words)
            
            # Count technical terms
            technical_terms = sum(1 for word in words if len(word) > 8)
            technical_ratio = technical_terms / len(words)
            
            return (complexity_ratio + technical_ratio) / 2
        
        except Exception as e:
            self.logger.error(f"Error calculating language complexity: {e}")
            return 0.5
    
    def _calculate_structure_score(self, content: str) -> float:
        """Calculate document structure score"""
        try:
            score = 0.0
            
            # Check for headings
            if '#' in content or '**' in content:
                score += 0.3
            
            # Check for bullet points
            if '•' in content or '-' in content or '*' in content:
                score += 0.2
            
            # Check for paragraphs
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                score += 0.2
            
            # Check for numbers/lists
            if any(char.isdigit() for char in content):
                score += 0.1
            
            # Check for proper capitalization
            sentences = [s for s in content.split('.') if s.strip()]
            proper_caps = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
            if sentences:
                score += 0.2 * (proper_caps / len(sentences))
            
            return min(1.0, score)
        
        except Exception as e:
            self.logger.error(f"Error calculating structure score: {e}")
            return 0.5
    
    def _calculate_keyword_density(self, content: str) -> float:
        """Calculate keyword density"""
        try:
            words = content.lower().split()
            if not words:
                return 0.0
            
            # Count business-related keywords
            business_keywords = [
                'strategy', 'analysis', 'implementation', 'optimization',
                'efficiency', 'performance', 'growth', 'revenue', 'profit',
                'market', 'customer', 'product', 'service', 'quality'
            ]
            
            keyword_count = sum(1 for word in words if word in business_keywords)
            return keyword_count / len(words)
        
        except Exception as e:
            self.logger.error(f"Error calculating keyword density: {e}")
            return 0.0
    
    def _count_technical_terms(self, content: str) -> int:
        """Count technical terms in content"""
        try:
            words = content.split()
            technical_terms = [
                'algorithm', 'database', 'software', 'hardware', 'network',
                'security', 'protocol', 'framework', 'architecture', 'system',
                'integration', 'deployment', 'configuration', 'optimization'
            ]
            
            return sum(1 for word in words if word.lower() in technical_terms)
        
        except Exception as e:
            self.logger.error(f"Error counting technical terms: {e}")
            return 0
    
    def _calculate_passive_voice_ratio(self, content: str) -> float:
        """Calculate passive voice ratio (simplified)"""
        try:
            words = content.lower().split()
            if not words:
                return 0.0
            
            # Simple passive voice indicators
            passive_indicators = ['was', 'were', 'been', 'being', 'is', 'are', 'am']
            passive_count = sum(1 for word in words if word in passive_indicators)
            
            return passive_count / len(words)
        
        except Exception as e:
            self.logger.error(f"Error calculating passive voice ratio: {e}")
            return 0.0
    
    def _calculate_metadata_score(self, business_area: BusinessArea, document_type: DocumentType) -> float:
        """Calculate metadata quality score"""
        try:
            score = 0.0
            
            # Business area relevance
            if business_area != BusinessArea.OTHER:
                score += 0.5
            
            # Document type specificity
            if document_type != DocumentType.OTHER:
                score += 0.5
            
            return score
        
        except Exception as e:
            self.logger.error(f"Error calculating metadata score: {e}")
            return 0.0
    
    async def predict_document_quality(
        self,
        content: str,
        business_area: BusinessArea,
        document_type: DocumentType,
        model_name: str = "ensemble"
    ) -> QualityPrediction:
        """Predict document quality using ML models"""
        try:
            # Extract features
            features = self.extract_document_features(content, business_area, document_type)
            
            # Convert to array
            feature_array = np.array([
                features.word_count, features.sentence_count, features.paragraph_count,
                features.avg_sentence_length, features.avg_word_length, features.readability_score,
                features.business_area_score, features.document_type_score, features.language_complexity,
                features.structure_score, features.keyword_density, features.technical_terms_count,
                features.passive_voice_ratio, features.active_voice_ratio, features.punctuation_density,
                features.capitalization_ratio, features.numbers_count, features.bullet_points_count,
                features.headings_count, features.metadata_score
            ]).reshape(1, -1)
            
            # Scale features
            if model_name in self.scalers:
                feature_array = self.scalers[model_name].transform(feature_array)
            
            # Make prediction
            if model_name == "ensemble":
                prediction = await self._ensemble_predict(feature_array)
                confidence = 0.9  # Ensemble typically has higher confidence
            else:
                if model_name in self.models and self.models[model_name] is not None:
                    prediction = self.models[model_name].predict(feature_array)[0]
                    confidence = 0.8
                else:
                    # Fallback to simple heuristic
                    prediction = self._heuristic_quality_score(features)
                    confidence = 0.6
            
            # Generate quality breakdown
            quality_breakdown = self._generate_quality_breakdown(features)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(features, prediction)
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model_name)
            
            document_id = str(uuid.uuid4())
            
            return QualityPrediction(
                document_id=document_id,
                predicted_quality=float(prediction),
                confidence_score=confidence,
                quality_breakdown=quality_breakdown,
                recommendations=recommendations,
                model_used=model_name,
                prediction_timestamp=datetime.now(),
                feature_importance=feature_importance
            )
        
        except Exception as e:
            self.logger.error(f"Error predicting document quality: {e}")
            # Return fallback prediction
            return QualityPrediction(
                document_id=str(uuid.uuid4()),
                predicted_quality=0.5,
                confidence_score=0.3,
                quality_breakdown={},
                recommendations=["Unable to analyze document quality"],
                model_used="fallback",
                prediction_timestamp=datetime.now(),
                feature_importance={}
            )
    
    async def _ensemble_predict(self, features: np.ndarray) -> float:
        """Make ensemble prediction using multiple models"""
        try:
            predictions = []
            weights = []
            
            for model_name, model in self.models.items():
                if model is not None and model_name != "ensemble":
                    try:
                        pred = model.predict(features)[0]
                        predictions.append(pred)
                        
                        # Weight based on model performance
                        if model_name in self.model_performance:
                            weight = self.model_performance[model_name].r2_score
                        else:
                            weight = 0.5
                        weights.append(weight)
                    except Exception as e:
                        self.logger.warning(f"Error with {model_name} model: {e}")
                        continue
            
            if predictions:
                # Weighted average
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize weights
                ensemble_prediction = np.average(predictions, weights=weights)
                return float(ensemble_prediction)
            else:
                return 0.5  # Fallback
        
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return 0.5
    
    def _heuristic_quality_score(self, features: DocumentFeatures) -> float:
        """Calculate heuristic quality score as fallback"""
        try:
            score = 0.0
            
            # Readability (30%)
            score += features.readability_score * 0.3
            
            # Structure (25%)
            score += features.structure_score * 0.25
            
            # Business relevance (20%)
            score += (features.business_area_score + features.document_type_score) / 2 * 0.2
            
            # Language quality (15%)
            language_quality = 1.0 - features.language_complexity
            score += language_quality * 0.15
            
            # Content richness (10%)
            content_richness = min(1.0, features.keyword_density * 10)
            score += content_richness * 0.1
            
            return min(1.0, max(0.0, score))
        
        except Exception as e:
            self.logger.error(f"Error calculating heuristic quality score: {e}")
            return 0.5
    
    def _generate_quality_breakdown(self, features: DocumentFeatures) -> Dict[str, float]:
        """Generate quality breakdown by category"""
        try:
            return {
                "readability": features.readability_score,
                "structure": features.structure_score,
                "business_relevance": (features.business_area_score + features.document_type_score) / 2,
                "language_quality": 1.0 - features.language_complexity,
                "content_richness": min(1.0, features.keyword_density * 10),
                "technical_accuracy": min(1.0, features.technical_terms_count / 10),
                "formatting": (features.bullet_points_count + features.headings_count) / 20,
                "metadata_quality": features.metadata_score
            }
        
        except Exception as e:
            self.logger.error(f"Error generating quality breakdown: {e}")
            return {}
    
    def _generate_recommendations(self, features: DocumentFeatures, predicted_quality: float) -> List[str]:
        """Generate improvement recommendations"""
        try:
            recommendations = []
            
            if features.readability_score < 0.6:
                recommendations.append("Improve readability by using shorter sentences and simpler words")
            
            if features.structure_score < 0.5:
                recommendations.append("Add headings, bullet points, and better paragraph structure")
            
            if features.language_complexity > 0.7:
                recommendations.append("Reduce language complexity by using simpler terms")
            
            if features.keyword_density < 0.02:
                recommendations.append("Include more relevant business keywords")
            
            if features.passive_voice_ratio > 0.3:
                recommendations.append("Use more active voice instead of passive voice")
            
            if features.technical_terms_count < 3:
                recommendations.append("Add more technical terms relevant to your business area")
            
            if features.bullet_points_count == 0 and features.headings_count == 0:
                recommendations.append("Add visual structure with headings and bullet points")
            
            if predicted_quality < 0.6:
                recommendations.append("Overall document quality is below average - consider comprehensive revision")
            
            if not recommendations:
                recommendations.append("Document quality is good - minor improvements could enhance it further")
            
            return recommendations
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations"]
    
    def _get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if model_name in self.models and self.models[model_name] is not None:
                model = self.models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                else:
                    return {}
                
                # Create feature importance dictionary
                feature_importance = {}
                for i, feature in enumerate(self.feature_columns):
                    if i < len(importances):
                        feature_importance[feature] = float(importances[i])
                
                return feature_importance
            else:
                return {}
        
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
    
    async def train_models(self, training_data: List[Dict[str, Any]]) -> Dict[str, ModelPerformance]:
        """Train ML models with provided data"""
        try:
            if not training_data:
                self.logger.warning("No training data provided")
                return {}
            
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if X.empty or len(y) == 0:
                self.logger.warning("Invalid training data")
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model_performances = {}
            
            # Train each model
            for model_name, model in self.models.items():
                if model_name == "ensemble":
                    continue
                
                try:
                    start_time = time.time()
                    
                    # Scale features
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_test_scaled = self.scalers[model_name].transform(X_test)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    cv_score = cv_scores.mean()
                    
                    training_time = time.time() - start_time
                    
                    # Store performance
                    performance = ModelPerformance(
                        model_name=model_name,
                        model_type=ModelType(model_name),
                        mse=mse,
                        rmse=rmse,
                        mae=mae,
                        r2_score=r2,
                        cross_val_score=cv_score,
                        training_time=training_time,
                        prediction_time=0.001,  # Estimated
                        last_trained=datetime.now(),
                        is_active=True
                    )
                    
                    model_performances[model_name] = performance
                    self.model_performance[model_name] = performance
                    
                    self.logger.info(f"Trained {model_name} model - R²: {r2:.3f}, RMSE: {rmse:.3f}")
                
                except Exception as e:
                    self.logger.error(f"Error training {model_name} model: {e}")
                    continue
            
            # Save models
            self._save_models()
            
            return model_performances
        
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return {}
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data for ML models"""
        try:
            X_data = []
            y_data = []
            
            for data_point in training_data:
                # Extract features
                features = data_point.get('features', {})
                quality_score = data_point.get('quality_score', 0.5)
                
                # Convert features to array
                feature_array = []
                for feature_name in self.feature_columns:
                    feature_array.append(features.get(feature_name, 0.0))
                
                X_data.append(feature_array)
                y_data.append(quality_score)
            
            X = pd.DataFrame(X_data, columns=self.feature_columns)
            y = np.array(y_data)
            
            return X, y
        
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), np.array([])
    
    async def get_model_performance(self) -> Dict[str, ModelPerformance]:
        """Get performance metrics for all models"""
        return self.model_performance
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get document quality predictor system status"""
        try:
            total_models = len(self.models)
            trained_models = len([m for m in self.models.values() if m is not None])
            active_models = len([p for p in self.model_performance.values() if p.is_active])
            
            # Calculate average performance
            if self.model_performance:
                avg_r2 = np.mean([p.r2_score for p in self.model_performance.values()])
                avg_rmse = np.mean([p.rmse for p in self.model_performance.values()])
                avg_mae = np.mean([p.mae for p in self.model_performance.values()])
            else:
                avg_r2 = avg_rmse = avg_mae = 0.0
            
            return {
                'total_models': total_models,
                'trained_models': trained_models,
                'active_models': active_models,
                'average_r2_score': round(avg_r2, 3),
                'average_rmse': round(avg_rmse, 3),
                'average_mae': round(avg_mae, 3),
                'feature_count': len(self.feature_columns),
                'training_data_size': len(self.training_data),
                'system_health': 'active' if trained_models > 0 else 'inactive'
            }
        
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {}

# Global document quality predictor
_document_quality_predictor: Optional[DocumentQualityPredictor] = None

def get_document_quality_predictor() -> DocumentQualityPredictor:
    """Get the global document quality predictor"""
    global _document_quality_predictor
    if _document_quality_predictor is None:
        _document_quality_predictor = DocumentQualityPredictor()
    return _document_quality_predictor

# Document quality predictor router
document_quality_router = APIRouter(prefix="/document-quality", tags=["Document Quality"])

@document_quality_router.post("/predict")
async def predict_document_quality_endpoint(
    content: str = Field(..., description="Document content"),
    business_area: BusinessArea = Field(..., description="Business area"),
    document_type: DocumentType = Field(..., description="Document type"),
    model_name: str = Field("ensemble", description="ML model to use")
):
    """Predict document quality using ML models"""
    try:
        predictor = get_document_quality_predictor()
        prediction = await predictor.predict_document_quality(
            content, business_area, document_type, model_name
        )
        return {"prediction": asdict(prediction), "success": True}
    
    except Exception as e:
        logger.error(f"Error predicting document quality: {e}")
        raise HTTPException(status_code=500, detail="Failed to predict document quality")

@document_quality_router.post("/train")
async def train_models_endpoint(training_data: List[Dict[str, Any]]):
    """Train ML models with provided data"""
    try:
        predictor = get_document_quality_predictor()
        performances = await predictor.train_models(training_data)
        return {"performances": {k: asdict(v) for k, v in performances.items()}, "success": True}
    
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail="Failed to train models")

@document_quality_router.get("/performance")
async def get_model_performance_endpoint():
    """Get model performance metrics"""
    try:
        predictor = get_document_quality_predictor()
        performances = await predictor.get_model_performance()
        return {"performances": {k: asdict(v) for k, v in performances.items()}}
    
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model performance")

@document_quality_router.get("/status")
async def get_system_status_endpoint():
    """Get document quality predictor system status"""
    try:
        predictor = get_document_quality_predictor()
        status = await predictor.get_system_status()
        return {"status": status}
    
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

