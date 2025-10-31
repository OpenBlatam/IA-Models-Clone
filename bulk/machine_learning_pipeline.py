"""
BUL Machine Learning Pipeline
============================

Advanced machine learning pipeline for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pickle
from dataclasses import dataclass
from enum import Enum
import yaml

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTaskType(Enum):
    """Machine learning task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TEXT_ANALYSIS = "text_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    DOCUMENT_CLASSIFICATION = "document_classification"

class MLModelType(Enum):
    """Machine learning model types."""
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NAIVE_BAYES = "naive_bayes"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"

@dataclass
class MLModel:
    """Machine learning model definition."""
    id: str
    name: str
    task_type: MLTaskType
    model_type: MLModelType
    model_path: str
    accuracy: float
    created_at: datetime
    last_trained: datetime
    parameters: Dict[str, Any] = None
    feature_importance: List[float] = None

@dataclass
class MLDataset:
    """Machine learning dataset definition."""
    id: str
    name: str
    task_type: MLTaskType
    data_path: str
    features: List[str]
    target: str
    size: int
    created_at: datetime
    description: str = ""

class MachineLearningPipeline:
    """Advanced machine learning pipeline for BUL system."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.models = {}
        self.datasets = {}
        self.training_history = []
        self.prediction_history = []
        self.init_ml_environment()
        self.load_models()
        self.load_datasets()
    
    def init_ml_environment(self):
        """Initialize machine learning environment."""
        print("🤖 Initializing machine learning environment...")
        
        # Create ML directories
        self.ml_dir = Path("ml_models")
        self.ml_dir.mkdir(exist_ok=True)
        
        self.data_dir = Path("ml_data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.results_dir = Path("ml_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print("✅ Machine learning environment initialized")
    
    def load_models(self):
        """Load existing machine learning models."""
        models_file = self.ml_dir / "models.json"
        if models_file.exists():
            with open(models_file, 'r') as f:
                models_data = json.load(f)
            
            for model_id, model_data in models_data.items():
                model = MLModel(
                    id=model_id,
                    name=model_data['name'],
                    task_type=MLTaskType(model_data['task_type']),
                    model_type=MLModelType(model_data['model_type']),
                    model_path=model_data['model_path'],
                    accuracy=model_data['accuracy'],
                    created_at=datetime.fromisoformat(model_data['created_at']),
                    last_trained=datetime.fromisoformat(model_data['last_trained']),
                    parameters=model_data.get('parameters', {}),
                    feature_importance=model_data.get('feature_importance', [])
                )
                self.models[model_id] = model
        
        print(f"✅ Loaded {len(self.models)} machine learning models")
    
    def load_datasets(self):
        """Load existing datasets."""
        datasets_file = self.ml_dir / "datasets.json"
        if datasets_file.exists():
            with open(datasets_file, 'r') as f:
                datasets_data = json.load(f)
            
            for dataset_id, dataset_data in datasets_data.items():
                dataset = MLDataset(
                    id=dataset_id,
                    name=dataset_data['name'],
                    task_type=MLTaskType(dataset_data['task_type']),
                    data_path=dataset_data['data_path'],
                    features=dataset_data['features'],
                    target=dataset_data['target'],
                    size=dataset_data['size'],
                    created_at=datetime.fromisoformat(dataset_data['created_at']),
                    description=dataset_data.get('description', '')
                )
                self.datasets[dataset_id] = dataset
        
        print(f"✅ Loaded {len(self.datasets)} datasets")
    
    def create_dataset(self, dataset_id: str, name: str, task_type: MLTaskType,
                      data_path: str, features: List[str], target: str,
                      description: str = "") -> MLDataset:
        """Create a new dataset."""
        # Load data to get size
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            size = len(df)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            size = 0
        
        dataset = MLDataset(
            id=dataset_id,
            name=name,
            task_type=task_type,
            data_path=data_path,
            features=features,
            target=target,
            size=size,
            created_at=datetime.now(),
            description=description
        )
        
        self.datasets[dataset_id] = dataset
        self._save_datasets()
        
        print(f"✅ Created dataset: {name} ({size} samples)")
        return dataset
    
    def train_model(self, model_id: str, name: str, task_type: MLTaskType,
                   model_type: MLModelType, dataset_id: str,
                   parameters: Dict[str, Any] = None) -> MLModel:
        """Train a machine learning model."""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset = self.datasets[dataset_id]
        
        print(f"🤖 Training model: {name}")
        print(f"   Dataset: {dataset.name}")
        print(f"   Task: {task_type.value}")
        print(f"   Model: {model_type.value}")
        
        try:
            # Load dataset
            if dataset.data_path.endswith('.csv'):
                df = pd.read_csv(dataset.data_path)
            elif dataset.data_path.endswith('.json'):
                df = pd.read_json(dataset.data_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset.data_path}")
            
            # Prepare features and target
            X = df[dataset.features]
            y = df[dataset.target]
            
            # Handle text features
            if task_type == MLTaskType.TEXT_ANALYSIS or task_type == MLTaskType.SENTIMENT_ANALYSIS:
                # Combine text features
                text_features = []
                for feature in dataset.features:
                    if df[feature].dtype == 'object':
                        text_features.append(feature)
                
                if text_features:
                    # Combine all text features
                    X_text = df[text_features].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                    
                    # Vectorize text
                    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                    X_vectorized = vectorizer.fit_transform(X_text)
                    
                    # Convert to DataFrame
                    X = pd.DataFrame(X_vectorized.toarray())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            if model_type == MLModelType.RANDOM_FOREST:
                model = RandomForestClassifier(
                    n_estimators=parameters.get('n_estimators', 100),
                    random_state=42
                )
            elif model_type == MLModelType.LOGISTIC_REGRESSION:
                model = LogisticRegression(
                    random_state=42,
                    max_iter=parameters.get('max_iter', 1000)
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_.tolist()
            
            # Save model
            model_path = self.ml_dir / f"{model_id}.joblib"
            joblib.dump(model, model_path)
            
            # Create model object
            ml_model = MLModel(
                id=model_id,
                name=name,
                task_type=task_type,
                model_type=model_type,
                model_path=str(model_path),
                accuracy=accuracy,
                created_at=datetime.now(),
                last_trained=datetime.now(),
                parameters=parameters or {},
                feature_importance=feature_importance
            )
            
            self.models[model_id] = ml_model
            self._save_models()
            
            # Log training
            self._log_training(model_id, dataset_id, accuracy, parameters)
            
            print(f"✅ Model trained successfully")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Model saved: {model_path}")
            
            return ml_model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, model_id: str, data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """Make predictions using a trained model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        
        print(f"🔮 Making predictions with model: {model_info.name}")
        
        try:
            # Load model
            model = joblib.load(model_info.model_path)
            
            # Prepare data
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()
            
            # Handle text features
            if model_info.task_type == MLTaskType.TEXT_ANALYSIS or model_info.task_type == MLTaskType.SENTIMENT_ANALYSIS:
                # This would need the same vectorizer used during training
                # For now, we'll assume the data is already prepared
                pass
            
            # Make predictions
            predictions = model.predict(df)
            probabilities = None
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(df)
            
            # Log prediction
            self._log_prediction(model_id, len(df), predictions.tolist())
            
            result = {
                'model_id': model_id,
                'model_name': model_info.name,
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'predicted_at': datetime.now().isoformat(),
                'input_size': len(df)
            }
            
            print(f"✅ Predictions completed")
            print(f"   Predictions: {len(predictions)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def evaluate_model(self, model_id: str, test_data_path: str) -> Dict[str, Any]:
        """Evaluate a model on test data."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        
        print(f"📊 Evaluating model: {model_info.name}")
        
        try:
            # Load model
            model = joblib.load(model_info.model_path)
            
            # Load test data
            if test_data_path.endswith('.csv'):
                df = pd.read_csv(test_data_path)
            elif test_data_path.endswith('.json'):
                df = pd.read_json(test_data_path)
            else:
                raise ValueError(f"Unsupported file format: {test_data_path}")
            
            # Prepare features and target
            X = df[model_info.parameters.get('features', [])]
            y = df[model_info.parameters.get('target', '')]
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            
            # Generate classification report
            report = classification_report(y, y_pred, output_dict=True)
            
            evaluation_result = {
                'model_id': model_id,
                'model_name': model_info.name,
                'accuracy': accuracy,
                'classification_report': report,
                'evaluated_at': datetime.now().isoformat(),
                'test_samples': len(df)
            }
            
            print(f"✅ Model evaluation completed")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Test samples: {len(df)}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def create_sample_dataset(self, dataset_id: str, task_type: MLTaskType) -> MLDataset:
        """Create a sample dataset for demonstration."""
        print(f"📊 Creating sample dataset: {dataset_id}")
        
        if task_type == MLTaskType.DOCUMENT_CLASSIFICATION:
            # Create sample document classification dataset
            data = {
                'text': [
                    'This is a marketing strategy document',
                    'Financial report for Q1 2024',
                    'HR policy update for remote work',
                    'Sales presentation for new product',
                    'Technical documentation for API',
                    'Legal contract template',
                    'Marketing campaign proposal',
                    'Budget planning document',
                    'Employee handbook update',
                    'Product specification sheet'
                ],
                'category': [
                    'marketing', 'finance', 'hr', 'sales', 'technical',
                    'legal', 'marketing', 'finance', 'hr', 'technical'
                ],
                'priority': [1, 2, 1, 2, 1, 3, 2, 2, 1, 1]
            }
            
            df = pd.DataFrame(data)
            data_path = self.data_dir / f"{dataset_id}.csv"
            df.to_csv(data_path, index=False)
            
            dataset = self.create_dataset(
                dataset_id=dataset_id,
                name=f"Sample {task_type.value.title()} Dataset",
                task_type=task_type,
                data_path=str(data_path),
                features=['text'],
                target='category',
                description=f"Sample dataset for {task_type.value} demonstration"
            )
            
        elif task_type == MLTaskType.SENTIMENT_ANALYSIS:
            # Create sample sentiment analysis dataset
            data = {
                'text': [
                    'This product is amazing and works perfectly',
                    'Terrible quality, would not recommend',
                    'Good value for money, satisfied with purchase',
                    'Poor customer service, very disappointed',
                    'Excellent features and easy to use',
                    'Waste of money, product broke immediately',
                    'Great quality and fast delivery',
                    'Not worth the price, very basic',
                    'Love this product, highly recommend',
                    'Average product, nothing special'
                ],
                'sentiment': [
                    'positive', 'negative', 'positive', 'negative', 'positive',
                    'negative', 'positive', 'negative', 'positive', 'neutral'
                ]
            }
            
            df = pd.DataFrame(data)
            data_path = self.data_dir / f"{dataset_id}.csv"
            df.to_csv(data_path, index=False)
            
            dataset = self.create_dataset(
                dataset_id=dataset_id,
                name=f"Sample {task_type.value.title()} Dataset",
                task_type=task_type,
                data_path=str(data_path),
                features=['text'],
                target='sentiment',
                description=f"Sample dataset for {task_type.value} demonstration"
            )
        
        else:
            raise ValueError(f"Unsupported task type for sample dataset: {task_type}")
        
        return dataset
    
    def _save_models(self):
        """Save models to file."""
        models_data = {}
        for model_id, model in self.models.items():
            models_data[model_id] = {
                'id': model.id,
                'name': model.name,
                'task_type': model.task_type.value,
                'model_type': model.model_type.value,
                'model_path': model.model_path,
                'accuracy': model.accuracy,
                'created_at': model.created_at.isoformat(),
                'last_trained': model.last_trained.isoformat(),
                'parameters': model.parameters or {},
                'feature_importance': model.feature_importance or []
            }
        
        with open(self.ml_dir / "models.json", 'w') as f:
            json.dump(models_data, f, indent=2)
    
    def _save_datasets(self):
        """Save datasets to file."""
        datasets_data = {}
        for dataset_id, dataset in self.datasets.items():
            datasets_data[dataset_id] = {
                'id': dataset.id,
                'name': dataset.name,
                'task_type': dataset.task_type.value,
                'data_path': dataset.data_path,
                'features': dataset.features,
                'target': dataset.target,
                'size': dataset.size,
                'created_at': dataset.created_at.isoformat(),
                'description': dataset.description
            }
        
        with open(self.ml_dir / "datasets.json", 'w') as f:
            json.dump(datasets_data, f, indent=2)
    
    def _log_training(self, model_id: str, dataset_id: str, accuracy: float, parameters: Dict[str, Any]):
        """Log training event."""
        training_event = {
            'model_id': model_id,
            'dataset_id': dataset_id,
            'accuracy': accuracy,
            'parameters': parameters,
            'trained_at': datetime.now().isoformat()
        }
        
        self.training_history.append(training_event)
        
        # Save training history
        with open(self.ml_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _log_prediction(self, model_id: str, input_size: int, predictions: List[Any]):
        """Log prediction event."""
        prediction_event = {
            'model_id': model_id,
            'input_size': input_size,
            'predictions': predictions,
            'predicted_at': datetime.now().isoformat()
        }
        
        self.prediction_history.append(prediction_event)
        
        # Save prediction history
        with open(self.ml_dir / "prediction_history.json", 'w') as f:
            json.dump(self.prediction_history, f, indent=2)
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Get training history for this model
        model_training = [t for t in self.training_history if t['model_id'] == model_id]
        
        # Get prediction history for this model
        model_predictions = [p for p in self.prediction_history if p['model_id'] == model_id]
        
        return {
            'model_id': model_id,
            'model_name': model.name,
            'accuracy': model.accuracy,
            'task_type': model.task_type.value,
            'model_type': model.model_type.value,
            'created_at': model.created_at.isoformat(),
            'last_trained': model.last_trained.isoformat(),
            'training_runs': len(model_training),
            'total_predictions': sum(p['input_size'] for p in model_predictions),
            'feature_importance': model.feature_importance
        }
    
    def generate_ml_report(self) -> str:
        """Generate machine learning pipeline report."""
        report = f"""
BUL Machine Learning Pipeline Report
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODELS
------
Total Models: {len(self.models)}
"""
        
        for model_id, model in self.models.items():
            report += f"""
{model.name} ({model_id}):
  Task Type: {model.task_type.value}
  Model Type: {model.model_type.value}
  Accuracy: {model.accuracy:.4f}
  Created: {model.created_at.strftime('%Y-%m-%d %H:%M:%S')}
  Last Trained: {model.last_trained.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report += f"""
DATASETS
--------
Total Datasets: {len(self.datasets)}
"""
        
        for dataset_id, dataset in self.datasets.items():
            report += f"""
{dataset.name} ({dataset_id}):
  Task Type: {dataset.task_type.value}
  Size: {dataset.size} samples
  Features: {', '.join(dataset.features)}
  Target: {dataset.target}
  Created: {dataset.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report += f"""
TRAINING HISTORY
----------------
Total Training Runs: {len(self.training_history)}
"""
        
        for training in self.training_history[-5:]:  # Show last 5 training runs
            report += f"""
{training['trained_at']}: {training['model_id']}
  Dataset: {training['dataset_id']}
  Accuracy: {training['accuracy']:.4f}
"""
        
        report += f"""
PREDICTION HISTORY
------------------
Total Predictions: {sum(p['input_size'] for p in self.prediction_history)}
"""
        
        for prediction in self.prediction_history[-5:]:  # Show last 5 predictions
            report += f"""
{prediction['predicted_at']}: {prediction['model_id']}
  Input Size: {prediction['input_size']}
  Predictions: {len(prediction['predictions'])}
"""
        
        return report

def main():
    """Main machine learning pipeline function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Machine Learning Pipeline")
    parser.add_argument("--create-dataset", help="Create a new dataset")
    parser.add_argument("--create-sample", help="Create a sample dataset")
    parser.add_argument("--train-model", help="Train a new model")
    parser.add_argument("--predict", help="Make predictions with a model")
    parser.add_argument("--evaluate", help="Evaluate a model")
    parser.add_argument("--list-models", action="store_true", help="List all models")
    parser.add_argument("--list-datasets", action="store_true", help="List all datasets")
    parser.add_argument("--model-performance", help="Get model performance metrics")
    parser.add_argument("--report", action="store_true", help="Generate ML pipeline report")
    parser.add_argument("--task-type", choices=['classification', 'regression', 'clustering', 'text_analysis', 'sentiment_analysis', 'document_classification'],
                       help="Task type for dataset/model")
    parser.add_argument("--model-type", choices=['random_forest', 'logistic_regression', 'svm', 'naive_bayes', 'neural_network', 'transformer'],
                       help="Model type for training")
    parser.add_argument("--data-path", help="Path to dataset file")
    parser.add_argument("--features", help="Comma-separated list of features")
    parser.add_argument("--target", help="Target variable name")
    parser.add_argument("--name", help="Name for dataset/model")
    
    args = parser.parse_args()
    
    pipeline = MachineLearningPipeline()
    
    print("🤖 BUL Machine Learning Pipeline")
    print("=" * 40)
    
    if args.create_dataset:
        if not all([args.data_path, args.features, args.target, args.task_type]):
            print("❌ Error: --data-path, --features, --target, and --task-type are required")
            return 1
        
        features = args.features.split(',')
        dataset = pipeline.create_dataset(
            dataset_id=args.create_dataset,
            name=args.name or f"Dataset {args.create_dataset}",
            task_type=MLTaskType(args.task_type),
            data_path=args.data_path,
            features=features,
            target=args.target
        )
        print(f"✅ Created dataset: {dataset.name}")
    
    elif args.create_sample:
        if not args.task_type:
            print("❌ Error: --task-type is required for sample dataset")
            return 1
        
        dataset = pipeline.create_sample_dataset(
            dataset_id=args.create_sample,
            task_type=MLTaskType(args.task_type)
        )
        print(f"✅ Created sample dataset: {dataset.name}")
    
    elif args.train_model:
        if not all([args.dataset_id, args.model_type, args.task_type]):
            print("❌ Error: --dataset-id, --model-type, and --task-type are required")
            return 1
        
        model = pipeline.train_model(
            model_id=args.train_model,
            name=args.name or f"Model {args.train_model}",
            task_type=MLTaskType(args.task_type),
            model_type=MLModelType(args.model_type),
            dataset_id=args.dataset_id
        )
        print(f"✅ Trained model: {model.name}")
    
    elif args.predict:
        # This would need input data - for demo purposes, we'll create sample data
        sample_data = {
            'text': 'This is a sample document for prediction'
        }
        
        result = pipeline.predict(args.predict, sample_data)
        print(f"✅ Predictions completed")
        print(f"   Predictions: {result['predictions']}")
    
    elif args.evaluate:
        if not args.test_data_path:
            print("❌ Error: --test-data-path is required for evaluation")
            return 1
        
        result = pipeline.evaluate_model(args.evaluate, args.test_data_path)
        print(f"✅ Model evaluation completed")
        print(f"   Accuracy: {result['accuracy']:.4f}")
    
    elif args.list_models:
        models = pipeline.models
        if models:
            print(f"\n🤖 Machine Learning Models ({len(models)}):")
            print("-" * 50)
            for model_id, model in models.items():
                print(f"{model.name} ({model_id}):")
                print(f"  Task: {model.task_type.value}")
                print(f"  Type: {model.model_type.value}")
                print(f"  Accuracy: {model.accuracy:.4f}")
                print()
        else:
            print("No models found.")
    
    elif args.list_datasets:
        datasets = pipeline.datasets
        if datasets:
            print(f"\n📊 Datasets ({len(datasets)}):")
            print("-" * 50)
            for dataset_id, dataset in datasets.items():
                print(f"{dataset.name} ({dataset_id}):")
                print(f"  Task: {dataset.task_type.value}")
                print(f"  Size: {dataset.size} samples")
                print(f"  Features: {', '.join(dataset.features)}")
                print()
        else:
            print("No datasets found.")
    
    elif args.model_performance:
        performance = pipeline.get_model_performance(args.model_performance)
        print(f"\n📊 Model Performance: {performance['model_name']}")
        print(f"   Accuracy: {performance['accuracy']:.4f}")
        print(f"   Training Runs: {performance['training_runs']}")
        print(f"   Total Predictions: {performance['total_predictions']}")
    
    elif args.report:
        report = pipeline.generate_ml_report()
        print(report)
        
        # Save report
        report_file = f"ml_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        print(f"🤖 Models: {len(pipeline.models)}")
        print(f"📊 Datasets: {len(pipeline.datasets)}")
        print(f"📈 Training Runs: {len(pipeline.training_history)}")
        print(f"🔮 Predictions: {sum(p['input_size'] for p in pipeline.prediction_history)}")
        print(f"\n💡 Use --create-sample document_classification to create a sample dataset")
        print(f"💡 Use --train-model to train a new model")
        print(f"💡 Use --list-models to see all models")
        print(f"💡 Use --report to generate ML pipeline report")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
