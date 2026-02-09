from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from model_training_evaluation import (
        from model_training_evaluation import ModelMetadata, TrainingConfig, EvaluationMetrics, ModelType
from typing import Any, List, Dict, Optional
import logging
"""
Model Training and Evaluation Demo

This demo showcases the comprehensive model training and evaluation system
for cybersecurity applications, including:
- Training different types of models
- Hyperparameter optimization
- Model evaluation and comparison
- A/B testing
- Production deployment
- Performance monitoring
"""



    ModelTrainer, ModelEvaluator, HyperparameterOptimizer,
    ModelVersionManager, ModelDeploymentManager, ModelType,
    TrainingConfig, EvaluationMetrics, create_model_trainer,
    calculate_model_complexity, validate_model_performance,
    run_ab_test
)


class ModelTrainingEvaluationDemo:
    """Comprehensive demo for model training and evaluation."""
    
    def __init__(self) -> Any:
        self.demo_dir = Path("./demo_output")
        self.demo_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.demo_dir / "data").mkdir(exist_ok=True)
        (self.demo_dir / "models").mkdir(exist_ok=True)
        (self.demo_dir / "logs").mkdir(exist_ok=True)
        (self.demo_dir / "artifacts").mkdir(exist_ok=True)
        
        self.results = {}
        
    async def run_comprehensive_demo(self) -> Any:
        """Run the complete demo showcasing all features."""
        print("🚀 Starting Comprehensive Model Training and Evaluation Demo")
        print("=" * 80)
        
        # Generate synthetic datasets
        await self._generate_synthetic_datasets()
        
        # Demo 1: Threat Detection Model Training
        await self._demo_threat_detection_training()
        
        # Demo 2: Anomaly Detection Model Training
        await self._demo_anomaly_detection_training()
        
        # Demo 3: Hyperparameter Optimization
        await self._demo_hyperparameter_optimization()
        
        # Demo 4: Model Evaluation and Comparison
        await self._demo_model_evaluation()
        
        # Demo 5: Model Versioning and Management
        await self._demo_model_versioning()
        
        # Demo 6: Production Deployment
        await self._demo_production_deployment()
        
        # Demo 7: A/B Testing
        await self._demo_ab_testing()
        
        # Demo 8: Performance Monitoring
        await self._demo_performance_monitoring()
        
        # Save results
        self._save_demo_results()
        
        print("\n✅ Demo completed successfully!")
        print(f"Results saved to: {self.demo_dir / 'demo_results.json'}")
    
    async def _generate_synthetic_datasets(self) -> Any:
        """Generate synthetic datasets for demonstration."""
        print("\n📊 Generating Synthetic Datasets...")
        
        # Threat Detection Dataset
        X_threat, y_threat = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # Convert to text-like features for threat detection
        threat_texts = []
        for features in X_threat:
            # Create synthetic text based on features
            text = f"Network packet with features: {', '.join([f'f{i}={v:.3f}' for i, v in enumerate(features[:10])])}"
            threat_texts.append(text)
        
        threat_df = pd.DataFrame({
            'text': threat_texts,
            'label': y_threat
        })
        
        threat_df.to_csv(self.demo_dir / "data" / "threat_detection.csv", index=False)
        
        # Anomaly Detection Dataset
        X_anomaly, y_anomaly = make_regression(
            n_samples=8000,
            n_features=50,
            n_informative=30,
            noise=0.1,
            random_state=42
        )
        
        # Add some anomalies
        anomaly_indices = np.random.choice(len(X_anomaly), size=800, replace=False)
        for idx in anomaly_indices:
            X_anomaly[idx] += np.random.normal(0, 2, X_anomaly.shape[1])
            y_anomaly[idx] = 1  # Mark as anomaly
        y_anomaly[~np.isin(np.arange(len(y_anomaly)), anomaly_indices)] = 0
        
        anomaly_df = pd.DataFrame({
            'features': [json.dumps(features.tolist()) for features in X_anomaly],
            'label': y_anomaly
        })
        
        anomaly_df.to_csv(self.demo_dir / "data" / "anomaly_detection.csv", index=False)
        
        # Test datasets
        X_test_threat, y_test_threat = make_classification(
            n_samples=2000, n_features=20, n_classes=2, random_state=123
        )
        test_threat_texts = [
            f"Test network packet: {', '.join([f'f{i}={v:.3f}' for i, v in enumerate(features[:10])])}"
            for features in X_test_threat
        ]
        test_threat_df = pd.DataFrame({
            'text': test_threat_texts,
            'label': y_test_threat
        })
        test_threat_df.to_csv(self.demo_dir / "data" / "test_threat_detection.csv", index=False)
        
        X_test_anomaly, y_test_anomaly = make_regression(
            n_samples=1500, n_features=50, random_state=123
        )
        test_anomaly_df = pd.DataFrame({
            'features': [json.dumps(features.tolist()) for features in X_test_anomaly],
            'label': y_test_anomaly
        })
        test_anomaly_df.to_csv(self.demo_dir / "data" / "test_anomaly_detection.csv", index=False)
        
        print(f"✅ Generated datasets:")
        print(f"   - Threat detection: {len(threat_df)} samples")
        print(f"   - Anomaly detection: {len(anomaly_df)} samples")
        print(f"   - Test datasets: {len(test_threat_df)} + {len(test_anomaly_df)} samples")
    
    async def _demo_threat_detection_training(self) -> Any:
        """Demo threat detection model training."""
        print("\n🛡️ Demo 1: Threat Detection Model Training")
        print("-" * 50)
        
        # Create training config
        config = TrainingConfig(
            model_type=ModelType.THREAT_DETECTION,
            model_name="distilbert-base-uncased",
            dataset_path=str(self.demo_dir / "data" / "threat_detection.csv"),
            output_dir=str(self.demo_dir / "models" / "threat_detection"),
            logging_dir=str(self.demo_dir / "logs" / "threat_detection"),
            num_epochs=2,  # Reduced for demo
            batch_size=16,
            learning_rate=2e-5,
            validation_split=0.2,
            test_split=0.1
        )
        
        print(f"Training configuration:")
        print(f"   - Model: {config.model_name}")
        print(f"   - Epochs: {config.num_epochs}")
        print(f"   - Batch size: {config.batch_size}")
        print(f"   - Learning rate: {config.learning_rate}")
        
        # Train model
        start_time = time.time()
        trainer = ModelTrainer(config)
        metadata = await trainer.train()
        training_time = time.time() - start_time
        
        # Store results
        self.results["threat_detection_training"] = {
            "metadata": metadata.__dict__,
            "training_time": training_time,
            "model_complexity": calculate_model_complexity(trainer.model)
        }
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"   - Model ID: {metadata.model_id}")
        print(f"   - Model size: {metadata.evaluation_metrics.model_size_mb:.2f} MB")
        print(f"   - Training time: {metadata.evaluation_metrics.training_time:.2f} seconds")
    
    async def _demo_anomaly_detection_training(self) -> Any:
        """Demo anomaly detection model training."""
        print("\n🔍 Demo 2: Anomaly Detection Model Training")
        print("-" * 50)
        
        # Create training config
        config = TrainingConfig(
            model_type=ModelType.ANOMALY_DETECTION,
            model_name="autoencoder",
            dataset_path=str(self.demo_dir / "data" / "anomaly_detection.csv"),
            output_dir=str(self.demo_dir / "models" / "anomaly_detection"),
            logging_dir=str(self.demo_dir / "logs" / "anomaly_detection"),
            num_epochs=3,  # Reduced for demo
            batch_size=32,
            learning_rate=1e-3,
            validation_split=0.2,
            test_split=0.1
        )
        
        print(f"Training configuration:")
        print(f"   - Model: {config.model_name}")
        print(f"   - Epochs: {config.num_epochs}")
        print(f"   - Batch size: {config.batch_size}")
        print(f"   - Learning rate: {config.learning_rate}")
        
        # Train model
        start_time = time.time()
        trainer = ModelTrainer(config)
        metadata = await trainer.train()
        training_time = time.time() - start_time
        
        # Store results
        self.results["anomaly_detection_training"] = {
            "metadata": metadata.__dict__,
            "training_time": training_time,
            "model_complexity": calculate_model_complexity(trainer.model)
        }
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"   - Model ID: {metadata.model_id}")
        print(f"   - Model size: {metadata.evaluation_metrics.model_size_mb:.2f} MB")
        print(f"   - Training time: {metadata.evaluation_metrics.training_time:.2f} seconds")
    
    async def _demo_hyperparameter_optimization(self) -> Any:
        """Demo hyperparameter optimization."""
        print("\n⚙️ Demo 3: Hyperparameter Optimization")
        print("-" * 50)
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(
            model_type=ModelType.THREAT_DETECTION,
            dataset_path=str(self.demo_dir / "data" / "threat_detection.csv")
        )
        
        print("Running hyperparameter optimization...")
        print("   - Trials: 5 (reduced for demo)")
        print("   - Timeout: 10 minutes")
        
        # Run optimization
        start_time = time.time()
        best_params = optimizer.optimize(n_trials=5)  # Reduced for demo
        optimization_time = time.time() - start_time
        
        # Store results
        self.results["hyperparameter_optimization"] = {
            "best_params": best_params,
            "optimization_time": optimization_time,
            "study_info": {
                "n_trials": len(optimizer.study.trials),
                "best_value": optimizer.study.best_value,
                "best_trial": optimizer.study.best_trial.number
            }
        }
        
        print(f"✅ Optimization completed in {optimization_time:.2f} seconds")
        print(f"   - Best parameters: {best_params}")
        print(f"   - Best F1 score: {optimizer.study.best_value:.4f}")
        print(f"   - Trials completed: {len(optimizer.study.trials)}")
    
    async def _demo_model_evaluation(self) -> Any:
        """Demo model evaluation and comparison."""
        print("\n📈 Demo 4: Model Evaluation and Comparison")
        print("-" * 50)
        
        # Get trained models
        threat_model_path = self.demo_dir / "models" / "threat_detection" / "final_model"
        anomaly_model_path = self.demo_dir / "models" / "anomaly_detection" / "final_model"
        
        # Evaluate threat detection model
        print("Evaluating threat detection model...")
        threat_evaluator = ModelEvaluator(
            str(threat_model_path),
            ModelType.THREAT_DETECTION
        )
        threat_metrics = await threat_evaluator.evaluate(
            str(self.demo_dir / "data" / "test_threat_detection.csv")
        )
        
        # Evaluate anomaly detection model
        print("Evaluating anomaly detection model...")
        anomaly_evaluator = ModelEvaluator(
            str(anomaly_model_path),
            ModelType.ANOMALY_DETECTION
        )
        anomaly_metrics = await anomaly_evaluator.evaluate(
            str(self.demo_dir / "data" / "test_anomaly_detection.csv")
        )
        
        # Performance validation
        thresholds = {
            "min_accuracy": 0.7,
            "min_f1": 0.6,
            "max_fpr": 0.2,
            "max_inference_time": 1.0
        }
        
        threat_valid = validate_model_performance(threat_metrics, thresholds)
        anomaly_valid = validate_model_performance(anomaly_metrics, thresholds)
        
        # Store results
        self.results["model_evaluation"] = {
            "threat_detection": {
                "metrics": threat_metrics.__dict__,
                "meets_thresholds": threat_valid
            },
            "anomaly_detection": {
                "metrics": anomaly_metrics.__dict__,
                "meets_thresholds": anomaly_valid
            },
            "thresholds": thresholds
        }
        
        print("✅ Evaluation completed:")
        print(f"   Threat Detection:")
        print(f"     - Accuracy: {threat_metrics.accuracy:.4f}")
        print(f"     - F1 Score: {threat_metrics.f1_score:.4f}")
        print(f"     - Inference time: {threat_metrics.inference_time:.4f}s")
        print(f"     - Meets thresholds: {threat_valid}")
        
        print(f"   Anomaly Detection:")
        print(f"     - Accuracy: {anomaly_metrics.accuracy:.4f}")
        print(f"     - F1 Score: {anomaly_metrics.f1_score:.4f}")
        print(f"     - Inference time: {anomaly_metrics.inference_time:.4f}s")
        print(f"     - Meets thresholds: {anomaly_valid}")
    
    async def _demo_model_versioning(self) -> Any:
        """Demo model versioning and management."""
        print("\n📦 Demo 5: Model Versioning and Management")
        print("-" * 50)
        
        # Create version manager
        version_manager = ModelVersionManager(str(self.demo_dir / "models"))
        
        # Register models from training
        threat_metadata = self.results["threat_detection_training"]["metadata"]
        anomaly_metadata = self.results["anomaly_detection_training"]["metadata"]
        
        # Convert back to ModelMetadata objects
        
        threat_config = TrainingConfig(**threat_metadata["training_config"])
        threat_metrics = EvaluationMetrics(**threat_metadata["evaluation_metrics"])
        
        threat_model_metadata = ModelMetadata(
            model_id=threat_metadata["model_id"],
            model_type=ModelType.THREAT_DETECTION,
            model_name=threat_metadata["model_name"],
            version=threat_metadata["version"],
            created_at=datetime.fromisoformat(threat_metadata["created_at"]),
            training_config=threat_config,
            evaluation_metrics=threat_metrics,
            dataset_info=threat_metadata["dataset_info"],
            hyperparameters=threat_metadata["hyperparameters"],
            dependencies=threat_metadata["dependencies"],
            model_path=threat_metadata["model_path"],
            artifacts_path=threat_metadata["artifacts_path"],
            is_production=False,
            tags=["demo", "threat_detection"]
        )
        
        version_manager.register_model(threat_model_metadata)
        
        # List models
        models = version_manager.list_models()
        print(f"✅ Registered {len(models)} models:")
        for model in models:
            print(f"   - {model['model_name']} v{model['version']} ({model['model_type']})")
        
        # Set production model
        version_manager.set_production_model(threat_model_metadata.model_id, ModelType.THREAT_DETECTION)
        production_model = version_manager.get_production_model(ModelType.THREAT_DETECTION)
        
        print(f"✅ Production model set: {production_model['model_name']} v{production_model['version']}")
        
        # Store results
        self.results["model_versioning"] = {
            "registered_models": models,
            "production_model": production_model
        }
    
    async def _demo_production_deployment(self) -> Any:
        """Demo production deployment."""
        print("\n🚀 Demo 6: Production Deployment")
        print("-" * 50)
        
        # Create deployment manager
        deployment_manager = ModelDeploymentManager(str(self.demo_dir / "models"))
        
        # Deploy threat detection model
        print("Deploying threat detection model...")
        deployment_id = await deployment_manager.deploy_model(ModelType.THREAT_DETECTION)
        
        # Test predictions
        test_inputs = [
            "Normal network traffic with standard HTTP requests",
            "Suspicious packet with unusual port scanning activity",
            "Malicious email containing phishing links",
            "Legitimate user login attempt"
        ]
        
        print("Making test predictions...")
        predictions = []
        for input_text in test_inputs:
            prediction = await deployment_manager.predict(deployment_id, input_text)
            predictions.append({
                "input": input_text,
                "prediction": prediction
            })
            print(f"   '{input_text[:50]}...' -> {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")
        
        # List deployed models
        deployed_models = deployment_manager.list_deployed_models()
        print(f"✅ Deployed models: {deployed_models}")
        
        # Store results
        self.results["production_deployment"] = {
            "deployment_id": deployment_id,
            "deployed_models": deployed_models,
            "test_predictions": predictions
        }
        
        # Cleanup
        deployment_manager.undeploy_model(deployment_id)
    
    async def _demo_ab_testing(self) -> Any:
        """Demo A/B testing between models."""
        print("\n🔄 Demo 7: A/B Testing")
        print("-" * 50)
        
        # Create two different models for A/B testing
        print("Creating two models for A/B testing...")
        
        # Model A: Standard configuration
        config_a = TrainingConfig(
            model_type=ModelType.THREAT_DETECTION,
            model_name="distilbert-base-uncased",
            dataset_path=str(self.demo_dir / "data" / "threat_detection.csv"),
            output_dir=str(self.demo_dir / "models" / "ab_test" / "model_a"),
            num_epochs=1,
            batch_size=16,
            learning_rate=2e-5
        )
        
        trainer_a = ModelTrainer(config_a)
        metadata_a = await trainer_a.train()
        
        # Model B: Different configuration
        config_b = TrainingConfig(
            model_type=ModelType.THREAT_DETECTION,
            model_name="distilbert-base-uncased",
            dataset_path=str(self.demo_dir / "data" / "threat_detection.csv"),
            output_dir=str(self.demo_dir / "models" / "ab_test" / "model_b"),
            num_epochs=1,
            batch_size=32,
            learning_rate=1e-5
        )
        
        trainer_b = ModelTrainer(config_b)
        metadata_b = await trainer_b.train()
        
        # Register models
        version_manager = ModelVersionManager(str(self.demo_dir / "models"))
        version_manager.register_model(metadata_a)
        version_manager.register_model(metadata_b)
        
        # Run A/B test
        print("Running A/B test...")
        ab_results = await run_ab_test(
            metadata_a.model_id,
            metadata_b.model_id,
            str(self.demo_dir / "data" / "test_threat_detection.csv"),
            traffic_split=0.5
        )
        
        print(f"✅ A/B test completed:")
        print(f"   Model A F1: {ab_results['model_a']['metrics']['f1_score']:.4f}")
        print(f"   Model B F1: {ab_results['model_b']['metrics']['f1_score']:.4f}")
        print(f"   Winner: {ab_results['winner']}")
        
        # Store results
        self.results["ab_testing"] = ab_results
    
    async def _demo_performance_monitoring(self) -> Any:
        """Demo performance monitoring."""
        print("\n📊 Demo 8: Performance Monitoring")
        print("-" * 50)
        
        # Create deployment manager
        deployment_manager = ModelDeploymentManager(str(self.demo_dir / "models"))
        
        # Get production model
        version_manager = ModelVersionManager(str(self.demo_dir / "models"))
        production_model = version_manager.get_production_model(ModelType.THREAT_DETECTION)
        
        if production_model:
            # Deploy model
            deployment_id = await deployment_manager.deploy_model(ModelType.THREAT_DETECTION)
            
            # Simulate production load
            print("Simulating production load...")
            test_inputs = [
                "Normal network traffic",
                "Suspicious activity detected",
                "Malicious payload identified",
                "Legitimate user request",
                "Potential security threat"
            ] * 20  # 100 total requests
            
            performance_metrics = {
                "total_requests": len(test_inputs),
                "predictions": [],
                "response_times": [],
                "errors": 0
            }
            
            start_time = time.time()
            
            for i, input_text in enumerate(test_inputs):
                try:
                    request_start = time.time()
                    prediction = await deployment_manager.predict(deployment_id, input_text)
                    request_time = time.time() - request_start
                    
                    performance_metrics["predictions"].append(prediction)
                    performance_metrics["response_times"].append(request_time)
                    
                    if i % 20 == 0:
                        print(f"   Processed {i+1}/{len(test_inputs)} requests...")
                        
                except Exception as e:
                    performance_metrics["errors"] += 1
                    print(f"   Error on request {i+1}: {e}")
            
            total_time = time.time() - start_time
            
            # Calculate performance statistics
            response_times = performance_metrics["response_times"]
            performance_stats = {
                "total_time": total_time,
                "requests_per_second": len(test_inputs) / total_time,
                "avg_response_time": np.mean(response_times),
                "median_response_time": np.median(response_times),
                "p95_response_time": np.percentile(response_times, 95),
                "p99_response_time": np.percentile(response_times, 99),
                "min_response_time": np.min(response_times),
                "max_response_time": np.max(response_times),
                "error_rate": performance_metrics["errors"] / len(test_inputs),
                "success_rate": (len(test_inputs) - performance_metrics["errors"]) / len(test_inputs)
            }
            
            print(f"✅ Performance monitoring completed:")
            print(f"   - Total requests: {performance_metrics['total_requests']}")
            print(f"   - Requests/second: {performance_stats['requests_per_second']:.2f}")
            print(f"   - Avg response time: {performance_stats['avg_response_time']:.4f}s")
            print(f"   - P95 response time: {performance_stats['p95_response_time']:.4f}s")
            print(f"   - Error rate: {performance_stats['error_rate']:.2%}")
            print(f"   - Success rate: {performance_stats['success_rate']:.2%}")
            
            # Store results
            self.results["performance_monitoring"] = {
                "performance_metrics": performance_metrics,
                "performance_stats": performance_stats
            }
            
            # Cleanup
            deployment_manager.undeploy_model(deployment_id)
        else:
            print("⚠️ No production model available for performance monitoring")
    
    def _save_demo_results(self) -> Any:
        """Save demo results to file."""
        results_file = self.demo_dir / "demo_results.json"
        
        # Convert datetime objects to strings
        def convert_datetime(obj) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj
        
        # Convert results to JSON-serializable format
        serializable_results = json.loads(
            json.dumps(self.results, default=convert_datetime, indent=2)
        )
        
        with open(results_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📁 Demo results saved to: {results_file}")
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self) -> Any:
        """Generate a summary report of the demo."""
        report_file = self.demo_dir / "demo_summary.md"
        
        with open(report_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("# Model Training and Evaluation Demo Summary\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Demo Overview\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("This demo showcases a comprehensive model training and evaluation system for cybersecurity applications.\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Key Features Demonstrated\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("1. **Threat Detection Model Training** - Training transformer-based models for threat detection\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("2. **Anomaly Detection Model Training** - Training autoencoder models for anomaly detection\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("3. **Hyperparameter Optimization** - Automated hyperparameter tuning using Optuna\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("4. **Model Evaluation** - Comprehensive evaluation metrics and performance validation\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("5. **Model Versioning** - Model registration, versioning, and production management\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("6. **Production Deployment** - Model deployment and real-time inference\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("7. **A/B Testing** - Model comparison and selection\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("8. **Performance Monitoring** - Production performance metrics and monitoring\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Results Summary\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "threat_detection_training" in self.results:
                threat_metrics = self.results["threat_detection_training"]["metadata"]["evaluation_metrics"]
                f.write(f"### Threat Detection Model\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Model ID: {self.results['threat_detection_training']['metadata']['model_id']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Training Time: {threat_metrics['training_time']:.2f} seconds\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Model Size: {threat_metrics['model_size_mb']:.2f} MB\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "anomaly_detection_training" in self.results:
                anomaly_metrics = self.results["anomaly_detection_training"]["metadata"]["evaluation_metrics"]
                f.write(f"### Anomaly Detection Model\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Model ID: {self.results['anomaly_detection_training']['metadata']['model_id']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Training Time: {anomaly_metrics['training_time']:.2f} seconds\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Model Size: {anomaly_metrics['model_size_mb']:.2f} MB\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "model_evaluation" in self.results:
                f.write("### Model Evaluation Results\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                threat_eval = self.results["model_evaluation"]["threat_detection"]["metrics"]
                anomaly_eval = self.results["model_evaluation"]["anomaly_detection"]["metrics"]
                
                f.write(f"**Threat Detection:**\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Accuracy: {threat_eval['accuracy']:.4f}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- F1 Score: {threat_eval['f1_score']:.4f}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Inference Time: {threat_eval['inference_time']:.4f}s\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                
                f.write(f"**Anomaly Detection:**\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Accuracy: {anomaly_eval['accuracy']:.4f}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- F1 Score: {anomaly_eval['f1_score']:.4f}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Inference Time: {anomaly_eval['inference_time']:.4f}s\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "performance_monitoring" in self.results:
                perf_stats = self.results["performance_monitoring"]["performance_stats"]
                f.write("### Performance Monitoring Results\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Requests/Second: {perf_stats['requests_per_second']:.2f}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Average Response Time: {perf_stats['avg_response_time']:.4f}s\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- P95 Response Time: {perf_stats['p95_response_time']:.4f}s\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"- Success Rate: {perf_stats['success_rate']:.2%}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Files Generated\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `demo_results.json` - Complete demo results\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `data/` - Synthetic datasets\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `models/` - Trained models\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `logs/` - Training logs\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `artifacts/` - Model artifacts\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Next Steps\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("1. Review the generated models and their performance\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("2. Deploy models to production environment\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("3. Set up continuous monitoring and retraining pipelines\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("4. Implement automated model updates and rollbacks\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("5. Add more sophisticated evaluation metrics\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        print(f"📋 Summary report generated: {report_file}")


async def main():
    """Main demo function."""
    demo = ModelTrainingEvaluationDemo()
    await demo.run_comprehensive_demo()


match __name__:
    case "__main__":
    asyncio.run(main()) 