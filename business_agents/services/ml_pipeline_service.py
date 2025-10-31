"""
ML Pipeline Service
===================

Advanced machine learning pipeline service for predictive analytics,
automated insights, and intelligent business recommendations.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    RECOMMENDATION = "recommendation"

class DataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"

@dataclass
class DataSchema:
    column_name: str
    data_type: DataType
    is_target: bool = False
    is_feature: bool = True
    missing_percentage: float = 0.0
    unique_values: int = 0
    description: str = ""

@dataclass
class ModelMetrics:
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    silhouette_score: Optional[float] = None
    cross_val_score: Optional[float] = None

@dataclass
class PredictionResult:
    predictions: List[Any]
    probabilities: Optional[List[float]] = None
    confidence_scores: Optional[List[float]] = None
    model_metrics: Optional[ModelMetrics] = None
    feature_importance: Optional[Dict[str, float]] = None

@dataclass
class MLPipeline:
    pipeline_id: str
    name: str
    model_type: ModelType
    model: Any
    scaler: Optional[Any] = None
    encoder: Optional[Any] = None
    vectorizer: Optional[Any] = None
    feature_columns: List[str] = None
    target_column: str = ""
    data_schema: List[DataSchema] = None
    training_metrics: Optional[ModelMetrics] = None
    created_at: datetime = None
    last_trained: datetime = None
    is_trained: bool = False

class MLPipelineService:
    """
    Advanced machine learning pipeline service for business intelligence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipelines: Dict[str, MLPipeline] = {}
        self.models_dir = Path(config.get("models_dir", "./models"))
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize default models
        self._initialize_default_models()
        
    def _initialize_default_models(self):
        """Initialize default ML models for common business use cases."""
        
        # Customer churn prediction
        churn_pipeline = MLPipeline(
            pipeline_id="customer_churn",
            name="Customer Churn Prediction",
            model_type=ModelType.CLASSIFICATION,
            model=RandomForestClassifier(n_estimators=100, random_state=42),
            feature_columns=[],
            target_column="churn",
            data_schema=[],
            created_at=datetime.now()
        )
        self.pipelines[churn_pipeline.pipeline_id] = churn_pipeline
        
        # Sales forecasting
        sales_pipeline = MLPipeline(
            pipeline_id="sales_forecast",
            name="Sales Forecasting",
            model_type=ModelType.REGRESSION,
            model=RandomForestRegressor(n_estimators=100, random_state=42),
            feature_columns=[],
            target_column="sales",
            data_schema=[],
            created_at=datetime.now()
        )
        self.pipelines[sales_pipeline.pipeline_id] = sales_pipeline
        
        # Customer segmentation
        segmentation_pipeline = MLPipeline(
            pipeline_id="customer_segmentation",
            name="Customer Segmentation",
            model_type=ModelType.CLUSTERING,
            model=KMeans(n_clusters=4, random_state=42),
            feature_columns=[],
            target_column="",
            data_schema=[],
            created_at=datetime.now()
        )
        self.pipelines[segmentation_pipeline.pipeline_id] = segmentation_pipeline
        
    async def create_pipeline(
        self,
        pipeline_id: str,
        name: str,
        model_type: ModelType,
        feature_columns: List[str],
        target_column: str = "",
        model_params: Dict[str, Any] = None
    ) -> MLPipeline:
        """Create a new ML pipeline."""
        
        try:
            # Create model based on type
            model = self._create_model(model_type, model_params or {})
            
            # Create pipeline
            pipeline = MLPipeline(
                pipeline_id=pipeline_id,
                name=name,
                model_type=model_type,
                model=model,
                feature_columns=feature_columns,
                target_column=target_column,
                data_schema=[],
                created_at=datetime.now()
            )
            
            self.pipelines[pipeline_id] = pipeline
            
            logger.info(f"Created ML pipeline: {pipeline_id}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create pipeline {pipeline_id}: {str(e)}")
            raise
            
    def _create_model(self, model_type: ModelType, params: Dict[str, Any]):
        """Create ML model based on type."""
        
        if model_type == ModelType.CLASSIFICATION:
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                random_state=params.get('random_state', 42),
                max_depth=params.get('max_depth', None)
            )
        elif model_type == ModelType.REGRESSION:
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                random_state=params.get('random_state', 42),
                max_depth=params.get('max_depth', None)
            )
        elif model_type == ModelType.CLUSTERING:
            return KMeans(
                n_clusters=params.get('n_clusters', 4),
                random_state=params.get('random_state', 42)
            )
        elif model_type == ModelType.ANOMALY_DETECTION:
            return DBSCAN(
                eps=params.get('eps', 0.5),
                min_samples=params.get('min_samples', 5)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    async def train_pipeline(
        self,
        pipeline_id: str,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> ModelMetrics:
        """Train ML pipeline with provided data."""
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        pipeline = self.pipelines[pipeline_id]
        
        try:
            # Prepare data
            X, y = self._prepare_training_data(pipeline, training_data)
            
            # Handle text data if needed
            if pipeline.model_type == ModelType.NLP:
                X = self._process_text_data(pipeline, X)
            
            # Scale numerical features
            if pipeline.model_type in [ModelType.CLASSIFICATION, ModelType.REGRESSION]:
                pipeline.scaler = StandardScaler()
                X = pipeline.scaler.fit_transform(X)
            
            # Train model
            if pipeline.model_type == ModelType.CLUSTERING:
                pipeline.model.fit(X)
                predictions = pipeline.model.predict(X)
                metrics = self._calculate_clustering_metrics(X, predictions)
            else:
                pipeline.model.fit(X, y)
                
                # Calculate metrics
                if validation_data is not None:
                    X_val, y_val = self._prepare_training_data(pipeline, validation_data)
                    if pipeline.scaler:
                        X_val = pipeline.scaler.transform(X_val)
                    metrics = self._calculate_metrics(pipeline, X_val, y_val)
                else:
                    metrics = self._calculate_metrics(pipeline, X, y)
            
            # Update pipeline
            pipeline.training_metrics = metrics
            pipeline.is_trained = True
            pipeline.last_trained = datetime.now()
            
            # Save pipeline
            await self._save_pipeline(pipeline)
            
            logger.info(f"Trained pipeline {pipeline_id} successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to train pipeline {pipeline_id}: {str(e)}")
            raise
            
    def _prepare_training_data(self, pipeline: MLPipeline, data: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare training data for model."""
        
        # Select feature columns
        feature_data = data[pipeline.feature_columns].copy()
        
        # Handle missing values
        feature_data = self._handle_missing_values(feature_data)
        
        # Encode categorical variables
        feature_data = self._encode_categorical_variables(pipeline, feature_data)
        
        # Convert to numpy array
        X = feature_data.values
        
        # Prepare target variable if needed
        y = None
        if pipeline.target_column and pipeline.target_column in data.columns:
            y = data[pipeline.target_column].values
            
            # Encode target if categorical
            if pipeline.model_type == ModelType.CLASSIFICATION:
                if not pipeline.encoder:
                    pipeline.encoder = LabelEncoder()
                    y = pipeline.encoder.fit_transform(y)
                else:
                    y = pipeline.encoder.transform(y)
        
        return X, y
        
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data."""
        
        # For numerical columns, fill with median
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())
        
        # For categorical columns, fill with mode
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
        
        return data
        
    def _encode_categorical_variables(self, pipeline: MLPipeline, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in pipeline.feature_columns:
                continue
                
            # Use label encoding for now (could be improved with one-hot encoding)
            if col not in getattr(pipeline, 'label_encoders', {}):
                if not hasattr(pipeline, 'label_encoders'):
                    pipeline.label_encoders = {}
                pipeline.label_encoders[col] = LabelEncoder()
                data[col] = pipeline.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                data[col] = pipeline.label_encoders[col].transform(data[col].astype(str))
        
        return data
        
    def _process_text_data(self, pipeline: MLPipeline, X: np.ndarray) -> np.ndarray:
        """Process text data for NLP models."""
        
        if not pipeline.vectorizer:
            pipeline.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X_processed = pipeline.vectorizer.fit_transform(X.flatten()).toarray()
        else:
            X_processed = pipeline.vectorizer.transform(X.flatten()).toarray()
        
        return X_processed
        
    def _calculate_metrics(self, pipeline: MLPipeline, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Calculate model metrics."""
        
        predictions = pipeline.model.predict(X)
        
        if pipeline.model_type == ModelType.CLASSIFICATION:
            return ModelMetrics(
                accuracy=accuracy_score(y, predictions),
                precision=precision_score(y, predictions, average='weighted'),
                recall=recall_score(y, predictions, average='weighted'),
                f1_score=f1_score(y, predictions, average='weighted'),
                cross_val_score=np.mean(cross_val_score(pipeline.model, X, y, cv=5))
            )
        elif pipeline.model_type == ModelType.REGRESSION:
            mse = mean_squared_error(y, predictions)
            return ModelMetrics(
                mse=mse,
                rmse=np.sqrt(mse),
                r2_score=r2_score(y, predictions),
                cross_val_score=np.mean(cross_val_score(pipeline.model, X, y, cv=5))
            )
        else:
            return ModelMetrics()
            
    def _calculate_clustering_metrics(self, X: np.ndarray, predictions: np.ndarray) -> ModelMetrics:
        """Calculate clustering metrics."""
        
        try:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(X, predictions)
            return ModelMetrics(silhouette_score=silhouette)
        except:
            return ModelMetrics()
            
    async def predict(
        self,
        pipeline_id: str,
        data: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> PredictionResult:
        """Make predictions using trained pipeline."""
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        pipeline = self.pipelines[pipeline_id]
        
        if not pipeline.is_trained:
            raise ValueError(f"Pipeline {pipeline_id} is not trained")
        
        try:
            # Convert data to DataFrame if needed
            if isinstance(data, list):
                data = pd.DataFrame(data)
            
            # Prepare data
            X = self._prepare_prediction_data(pipeline, data)
            
            # Make predictions
            predictions = pipeline.model.predict(X)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(pipeline.model, 'predict_proba'):
                probabilities = pipeline.model.predict_proba(X).tolist()
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(pipeline, X, predictions)
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(pipeline.model, 'feature_importances_'):
                feature_importance = dict(zip(pipeline.feature_columns, pipeline.model.feature_importances_))
            
            return PredictionResult(
                predictions=predictions.tolist(),
                probabilities=probabilities,
                confidence_scores=confidence_scores,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for pipeline {pipeline_id}: {str(e)}")
            raise
            
    def _prepare_prediction_data(self, pipeline: MLPipeline, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for prediction."""
        
        # Select feature columns
        feature_data = data[pipeline.feature_columns].copy()
        
        # Handle missing values
        feature_data = self._handle_missing_values(feature_data)
        
        # Encode categorical variables
        feature_data = self._encode_categorical_variables(pipeline, feature_data)
        
        # Scale if scaler exists
        if pipeline.scaler:
            feature_data = pipeline.scaler.transform(feature_data)
        
        return feature_data
        
    def _calculate_confidence_scores(self, pipeline: MLPipeline, X: np.ndarray, predictions: np.ndarray) -> List[float]:
        """Calculate confidence scores for predictions."""
        
        if hasattr(pipeline.model, 'predict_proba'):
            probabilities = pipeline.model.predict_proba(X)
            # Use max probability as confidence score
            return np.max(probabilities, axis=1).tolist()
        else:
            # For regression models, use distance from mean as confidence
            if hasattr(pipeline.model, 'oob_score_'):
                return [pipeline.model.oob_score_] * len(predictions)
            else:
                return [0.8] * len(predictions)  # Default confidence
                
    async def analyze_data(self, data: pd.DataFrame) -> List[DataSchema]:
        """Analyze data and create schema."""
        
        schema = []
        
        for column in data.columns:
            data_type = self._infer_data_type(data[column])
            missing_percentage = (data[column].isnull().sum() / len(data)) * 100
            unique_values = data[column].nunique()
            
            schema.append(DataSchema(
                column_name=column,
                data_type=data_type,
                missing_percentage=missing_percentage,
                unique_values=unique_values,
                description=f"{data_type.value} column with {missing_percentage:.1f}% missing values"
            ))
        
        return schema
        
    def _infer_data_type(self, series: pd.Series) -> DataType:
        """Infer data type from pandas series."""
        
        if series.dtype == 'bool':
            return DataType.BOOLEAN
        elif pd.api.types.is_numeric_dtype(series):
            return DataType.NUMERICAL
        elif pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME
        elif series.dtype == 'object':
            # Check if it's text data
            if series.str.len().mean() > 20:
                return DataType.TEXT
            else:
                return DataType.CATEGORICAL
        else:
            return DataType.CATEGORICAL
            
    async def get_feature_importance(self, pipeline_id: str) -> Dict[str, float]:
        """Get feature importance for trained model."""
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        pipeline = self.pipelines[pipeline_id]
        
        if not pipeline.is_trained or not hasattr(pipeline.model, 'feature_importances_'):
            return {}
        
        return dict(zip(pipeline.feature_columns, pipeline.model.feature_importances_))
        
    async def get_model_insights(self, pipeline_id: str) -> Dict[str, Any]:
        """Get insights about the trained model."""
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        pipeline = self.pipelines[pipeline_id]
        
        insights = {
            "pipeline_id": pipeline_id,
            "name": pipeline.name,
            "model_type": pipeline.model_type.value,
            "is_trained": pipeline.is_trained,
            "created_at": pipeline.created_at.isoformat() if pipeline.created_at else None,
            "last_trained": pipeline.last_trained.isoformat() if pipeline.last_trained else None,
            "feature_count": len(pipeline.feature_columns),
            "metrics": asdict(pipeline.training_metrics) if pipeline.training_metrics else None
        }
        
        if pipeline.is_trained:
            insights["feature_importance"] = await self.get_feature_importance(pipeline_id)
            
            # Add model-specific insights
            if hasattr(pipeline.model, 'n_estimators'):
                insights["n_estimators"] = pipeline.model.n_estimators
            if hasattr(pipeline.model, 'n_clusters'):
                insights["n_clusters"] = pipeline.model.n_clusters
                
        return insights
        
    async def _save_pipeline(self, pipeline: MLPipeline):
        """Save pipeline to disk."""
        
        try:
            pipeline_path = self.models_dir / f"{pipeline.pipeline_id}.pkl"
            
            # Create a serializable version of the pipeline
            pipeline_data = {
                "pipeline_id": pipeline.pipeline_id,
                "name": pipeline.name,
                "model_type": pipeline.model_type.value,
                "feature_columns": pipeline.feature_columns,
                "target_column": pipeline.target_column,
                "data_schema": [asdict(schema) for schema in pipeline.data_schema] if pipeline.data_schema else [],
                "training_metrics": asdict(pipeline.training_metrics) if pipeline.training_metrics else None,
                "created_at": pipeline.created_at.isoformat() if pipeline.created_at else None,
                "last_trained": pipeline.last_trained.isoformat() if pipeline.last_trained else None,
                "is_trained": pipeline.is_trained
            }
            
            # Save pipeline metadata
            with open(pipeline_path, 'wb') as f:
                pickle.dump(pipeline_data, f)
            
            # Save model separately
            model_path = self.models_dir / f"{pipeline.pipeline_id}_model.pkl"
            joblib.dump(pipeline.model, model_path)
            
            # Save scaler if exists
            if pipeline.scaler:
                scaler_path = self.models_dir / f"{pipeline.pipeline_id}_scaler.pkl"
                joblib.dump(pipeline.scaler, scaler_path)
            
            # Save encoder if exists
            if pipeline.encoder:
                encoder_path = self.models_dir / f"{pipeline.pipeline_id}_encoder.pkl"
                joblib.dump(pipeline.encoder, encoder_path)
            
            logger.info(f"Saved pipeline {pipeline.pipeline_id} to disk")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline {pipeline.pipeline_id}: {str(e)}")
            
    async def load_pipeline(self, pipeline_id: str) -> MLPipeline:
        """Load pipeline from disk."""
        
        try:
            pipeline_path = self.models_dir / f"{pipeline_id}.pkl"
            
            if not pipeline_path.exists():
                raise FileNotFoundError(f"Pipeline {pipeline_id} not found on disk")
            
            # Load pipeline metadata
            with open(pipeline_path, 'rb') as f:
                pipeline_data = pickle.load(f)
            
            # Recreate pipeline
            pipeline = MLPipeline(
                pipeline_id=pipeline_data["pipeline_id"],
                name=pipeline_data["name"],
                model_type=ModelType(pipeline_data["model_type"]),
                model=None,  # Will be loaded separately
                feature_columns=pipeline_data["feature_columns"],
                target_column=pipeline_data["target_column"],
                data_schema=[DataSchema(**schema) for schema in pipeline_data["data_schema"]] if pipeline_data["data_schema"] else [],
                training_metrics=ModelMetrics(**pipeline_data["training_metrics"]) if pipeline_data["training_metrics"] else None,
                created_at=datetime.fromisoformat(pipeline_data["created_at"]) if pipeline_data["created_at"] else None,
                last_trained=datetime.fromisoformat(pipeline_data["last_trained"]) if pipeline_data["last_trained"] else None,
                is_trained=pipeline_data["is_trained"]
            )
            
            # Load model
            model_path = self.models_dir / f"{pipeline_id}_model.pkl"
            if model_path.exists():
                pipeline.model = joblib.load(model_path)
            
            # Load scaler if exists
            scaler_path = self.models_dir / f"{pipeline_id}_scaler.pkl"
            if scaler_path.exists():
                pipeline.scaler = joblib.load(scaler_path)
            
            # Load encoder if exists
            encoder_path = self.models_dir / f"{pipeline_id}_encoder.pkl"
            if encoder_path.exists():
                pipeline.encoder = joblib.load(encoder_path)
            
            self.pipelines[pipeline_id] = pipeline
            
            logger.info(f"Loaded pipeline {pipeline_id} from disk")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load pipeline {pipeline_id}: {str(e)}")
            raise
            
    async def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all available pipelines."""
        
        pipelines = []
        for pipeline in self.pipelines.values():
            pipelines.append({
                "pipeline_id": pipeline.pipeline_id,
                "name": pipeline.name,
                "model_type": pipeline.model_type.value,
                "is_trained": pipeline.is_trained,
                "created_at": pipeline.created_at.isoformat() if pipeline.created_at else None,
                "last_trained": pipeline.last_trained.isoformat() if pipeline.last_trained else None,
                "feature_count": len(pipeline.feature_columns)
            })
        
        return pipelines
        
    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete pipeline and its files."""
        
        try:
            # Remove from memory
            if pipeline_id in self.pipelines:
                del self.pipelines[pipeline_id]
            
            # Remove files
            files_to_remove = [
                f"{pipeline_id}.pkl",
                f"{pipeline_id}_model.pkl",
                f"{pipeline_id}_scaler.pkl",
                f"{pipeline_id}_encoder.pkl"
            ]
            
            for filename in files_to_remove:
                file_path = self.models_dir / filename
                if file_path.exists():
                    file_path.unlink()
            
            logger.info(f"Deleted pipeline {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete pipeline {pipeline_id}: {str(e)}")
            return False
            
    async def retrain_pipeline(
        self,
        pipeline_id: str,
        new_data: pd.DataFrame,
        incremental: bool = True
    ) -> ModelMetrics:
        """Retrain pipeline with new data."""
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        pipeline = self.pipelines[pipeline_id]
        
        if incremental and pipeline.is_trained:
            # For incremental learning, we would need to implement online learning
            # For now, we'll do full retraining
            pass
        
        # Retrain with new data
        return await self.train_pipeline(pipeline_id, new_data)
        
    async def get_prediction_explanation(
        self,
        pipeline_id: str,
        prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get explanation for a specific prediction."""
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        pipeline = self.pipelines[pipeline_id]
        
        if not pipeline.is_trained:
            raise ValueError(f"Pipeline {pipeline_id} is not trained")
        
        try:
            # Make prediction
            result = await self.predict(pipeline_id, [prediction_data])
            
            explanation = {
                "prediction": result.predictions[0],
                "confidence": result.confidence_scores[0] if result.confidence_scores else None,
                "feature_contributions": {}
            }
            
            # Calculate feature contributions
            if result.feature_importance:
                for feature, importance in result.feature_importance.items():
                    feature_value = prediction_data.get(feature, 0)
                    explanation["feature_contributions"][feature] = {
                        "value": feature_value,
                        "importance": importance,
                        "contribution": feature_value * importance
                    }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to get prediction explanation: {str(e)}")
            raise





























