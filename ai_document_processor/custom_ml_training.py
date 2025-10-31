"""
Custom Machine Learning Model Training and Deployment Module
"""

import asyncio
import logging
import os
import time
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import uuid
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
import joblib
import wandb
from datasets import Dataset as HFDataset
import optuna

from config import settings
from models import ProcessingStatus

logger = logging.getLogger(__name__)


class CustomMLTraining:
    """Custom Machine Learning Model Training and Deployment Engine"""
    
    def __init__(self):
        self.training_jobs = {}
        self.deployed_models = {}
        self.model_registry = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize the custom ML training system"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing Custom ML Training System...")
            
            # Initialize Weights & Biases if configured
            if hasattr(settings, 'wandb_project') and settings.wandb_project:
                wandb.init(project=settings.wandb_project, mode="disabled")
            
            # Create model directories
            self.models_dir = Path("./models")
            self.models_dir.mkdir(exist_ok=True)
            
            self.training_dir = Path("./training")
            self.training_dir.mkdir(exist_ok=True)
            
            self.deployment_dir = Path("./deployment")
            self.deployment_dir.mkdir(exist_ok=True)
            
            self.initialized = True
            logger.info("Custom ML Training System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing custom ML training system: {e}")
            raise
    
    async def create_training_job(self, job_config: Dict[str, Any]) -> str:
        """Create a new training job"""
        if not self.initialized:
            await self.initialize()
        
        job_id = str(uuid.uuid4())
        
        try:
            # Validate job configuration
            validated_config = self._validate_job_config(job_config)
            
            # Create training job
            training_job = {
                "job_id": job_id,
                "config": validated_config,
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "progress": 0.0,
                "metrics": {},
                "model_path": None,
                "error_message": None
            }
            
            self.training_jobs[job_id] = training_job
            
            # Save job configuration
            job_file = self.training_dir / f"{job_id}_config.json"
            with open(job_file, 'w') as f:
                json.dump(training_job, f, indent=2)
            
            logger.info(f"Created training job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error creating training job: {e}")
            raise
    
    async def start_training(self, job_id: str) -> Dict[str, Any]:
        """Start training a model"""
        try:
            if job_id not in self.training_jobs:
                raise ValueError(f"Training job {job_id} not found")
            
            job = self.training_jobs[job_id]
            
            # Update job status
            job["status"] = "training"
            job["updated_at"] = datetime.now().isoformat()
            
            # Start training asynchronously
            asyncio.create_task(self._train_model_async(job_id))
            
            return {
                "job_id": job_id,
                "status": "training_started",
                "message": "Training has been started"
            }
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            raise
    
    async def _train_model_async(self, job_id: str):
        """Train model asynchronously"""
        try:
            job = self.training_jobs[job_id]
            config = job["config"]
            
            # Update progress
            job["progress"] = 10.0
            job["updated_at"] = datetime.now().isoformat()
            
            # Load and prepare data
            dataset = await self._load_dataset(config["dataset_path"], config["task_type"])
            train_loader, val_loader = await self._prepare_data_loaders(dataset, config)
            
            job["progress"] = 20.0
            job["updated_at"] = datetime.now().isoformat()
            
            # Initialize model
            model = await self._create_model(config)
            
            job["progress"] = 30.0
            job["updated_at"] = datetime.now().isoformat()
            
            # Train model
            training_results = await self._train_model(model, train_loader, val_loader, config, job_id)
            
            # Update job with results
            job["status"] = "completed"
            job["progress"] = 100.0
            job["metrics"] = training_results["metrics"]
            job["model_path"] = training_results["model_path"]
            job["updated_at"] = datetime.now().isoformat()
            
            # Save final job state
            job_file = self.training_dir / f"{job_id}_config.json"
            with open(job_file, 'w') as f:
                json.dump(job, f, indent=2)
            
            logger.info(f"Training completed for job: {job_id}")
            
        except Exception as e:
            logger.error(f"Error in training job {job_id}: {e}")
            job = self.training_jobs[job_id]
            job["status"] = "failed"
            job["error_message"] = str(e)
            job["updated_at"] = datetime.now().isoformat()
    
    async def _load_dataset(self, dataset_path: str, task_type: str) -> Dataset:
        """Load dataset for training"""
        try:
            if task_type == "text_classification":
                return await self._load_text_classification_dataset(dataset_path)
            elif task_type == "document_classification":
                return await self._load_document_classification_dataset(dataset_path)
            elif task_type == "sentiment_analysis":
                return await self._load_sentiment_analysis_dataset(dataset_path)
            elif task_type == "named_entity_recognition":
                return await self._load_ner_dataset(dataset_path)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    async def _load_text_classification_dataset(self, dataset_path: str) -> Dataset:
        """Load text classification dataset"""
        try:
            # Load CSV dataset
            df = pd.read_csv(dataset_path)
            
            # Validate required columns
            if "text" not in df.columns or "label" not in df.columns:
                raise ValueError("Dataset must contain 'text' and 'label' columns")
            
            # Create dataset
            dataset = HFDataset.from_pandas(df)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading text classification dataset: {e}")
            raise
    
    async def _load_document_classification_dataset(self, dataset_path: str) -> Dataset:
        """Load document classification dataset"""
        try:
            # Load CSV dataset with document paths and labels
            df = pd.read_csv(dataset_path)
            
            # Validate required columns
            if "document_path" not in df.columns or "label" not in df.columns:
                raise ValueError("Dataset must contain 'document_path' and 'label' columns")
            
            # Load document contents
            texts = []
            for doc_path in df["document_path"]:
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                except:
                    texts.append("")  # Empty text for failed loads
            
            df["text"] = texts
            dataset = HFDataset.from_pandas(df)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading document classification dataset: {e}")
            raise
    
    async def _load_sentiment_analysis_dataset(self, dataset_path: str) -> Dataset:
        """Load sentiment analysis dataset"""
        try:
            # Load CSV dataset
            df = pd.read_csv(dataset_path)
            
            # Validate required columns
            if "text" not in df.columns or "sentiment" not in df.columns:
                raise ValueError("Dataset must contain 'text' and 'sentiment' columns")
            
            # Map sentiment labels to numbers
            sentiment_map = {"positive": 1, "negative": 0, "neutral": 2}
            df["label"] = df["sentiment"].map(sentiment_map)
            
            dataset = HFDataset.from_pandas(df)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading sentiment analysis dataset: {e}")
            raise
    
    async def _load_ner_dataset(self, dataset_path: str) -> Dataset:
        """Load named entity recognition dataset"""
        try:
            # Load JSON dataset with BIO tags
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            # Convert to HuggingFace dataset format
            dataset = HFDataset.from_list(data)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading NER dataset: {e}")
            raise
    
    async def _prepare_data_loaders(self, dataset: Dataset, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training"""
        try:
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Create data loaders
            batch_size = config.get("batch_size", 32)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Error preparing data loaders: {e}")
            raise
    
    async def _create_model(self, config: Dict[str, Any]) -> nn.Module:
        """Create model based on configuration"""
        try:
            task_type = config["task_type"]
            model_name = config.get("base_model", "distilbert-base-uncased")
            
            if task_type in ["text_classification", "document_classification", "sentiment_analysis"]:
                # Load pre-trained transformer model
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=config.get("num_labels", 2)
                )
                
                # Add custom layers if specified
                if config.get("custom_layers"):
                    model = self._add_custom_layers(model, config["custom_layers"])
                
            elif task_type == "named_entity_recognition":
                # Load NER model
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=config.get("num_labels", 9)  # BIO tags
                )
            
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def _add_custom_layers(self, model: nn.Module, custom_layers: List[Dict[str, Any]]) -> nn.Module:
        """Add custom layers to model"""
        try:
            # This is a simplified implementation
            # In practice, you'd implement more sophisticated layer addition
            
            for layer_config in custom_layers:
                layer_type = layer_config["type"]
                
                if layer_type == "dropout":
                    dropout = nn.Dropout(layer_config.get("rate", 0.1))
                    # Add dropout layer to model
                    
                elif layer_type == "dense":
                    dense = nn.Linear(
                        layer_config["input_size"],
                        layer_config["output_size"]
                    )
                    # Add dense layer to model
            
            return model
            
        except Exception as e:
            logger.error(f"Error adding custom layers: {e}")
            return model
    
    async def _train_model(self, model: nn.Module, train_loader: DataLoader, 
                          val_loader: DataLoader, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Train the model"""
        try:
            # Set up training parameters
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.get("learning_rate", 2e-5),
                weight_decay=config.get("weight_decay", 0.01)
            )
            
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            num_epochs = config.get("num_epochs", 3)
            best_val_loss = float('inf')
            training_history = []
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    # Update progress
                    progress = 30.0 + (epoch / num_epochs) * 60.0 + (batch_idx / len(train_loader)) * (60.0 / num_epochs)
                    self.training_jobs[job_id]["progress"] = progress
                    self.training_jobs[job_id]["updated_at"] = datetime.now().isoformat()
                    
                    # Training step
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = criterion(outputs.logits, batch["labels"])
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.logits.data, 1)
                    train_total += batch["labels"].size(0)
                    train_correct += (predicted == batch["labels"]).sum().item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        outputs = model(**batch)
                        loss = criterion(outputs.logits, batch["labels"])
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.logits.data, 1)
                        val_total += batch["labels"].size(0)
                        val_correct += (predicted == batch["labels"]).sum().item()
                
                # Calculate metrics
                train_acc = 100.0 * train_correct / train_total
                val_acc = 100.0 * val_correct / val_total
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                # Save training history
                epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc
                }
                training_history.append(epoch_metrics)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_path = self.models_dir / f"{job_id}_best_model.pth"
                    torch.save(model.state_dict(), model_path)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save final model
            final_model_path = self.models_dir / f"{job_id}_final_model.pth"
            torch.save(model.state_dict(), final_model_path)
            
            # Save model configuration
            model_config_path = self.models_dir / f"{job_id}_config.json"
            with open(model_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return {
                "model_path": str(final_model_path),
                "best_model_path": str(model_path),
                "metrics": {
                    "final_train_accuracy": train_acc,
                    "final_val_accuracy": val_acc,
                    "final_train_loss": avg_train_loss,
                    "final_val_loss": avg_val_loss,
                    "best_val_loss": best_val_loss,
                    "training_history": training_history
                }
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    async def deploy_model(self, job_id: str, deployment_config: Dict[str, Any]) -> str:
        """Deploy a trained model"""
        try:
            if job_id not in self.training_jobs:
                raise ValueError(f"Training job {job_id} not found")
            
            job = self.training_jobs[job_id]
            
            if job["status"] != "completed":
                raise ValueError(f"Training job {job_id} is not completed")
            
            # Create deployment
            deployment_id = str(uuid.uuid4())
            
            deployment = {
                "deployment_id": deployment_id,
                "job_id": job_id,
                "model_path": job["model_path"],
                "config": deployment_config,
                "status": "deployed",
                "created_at": datetime.now().isoformat(),
                "endpoint": f"/api/v1/models/{deployment_id}/predict"
            }
            
            self.deployed_models[deployment_id] = deployment
            
            # Save deployment configuration
            deployment_file = self.deployment_dir / f"{deployment_id}_deployment.json"
            with open(deployment_file, 'w') as f:
                json.dump(deployment, f, indent=2)
            
            logger.info(f"Model deployed: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    async def predict(self, deployment_id: str, input_data: Any) -> Dict[str, Any]:
        """Make prediction using deployed model"""
        try:
            if deployment_id not in self.deployed_models:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            deployment = self.deployed_models[deployment_id]
            job_id = deployment["job_id"]
            job = self.training_jobs[job_id]
            
            # Load model
            model_path = deployment["model_path"]
            config = job["config"]
            
            # Create model
            model = await self._create_model(config)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            # Make prediction
            with torch.no_grad():
                # This is a simplified implementation
                # In practice, you'd implement proper input preprocessing
                prediction = model(input_data)
                
                # Process prediction based on task type
                if config["task_type"] in ["text_classification", "document_classification", "sentiment_analysis"]:
                    probabilities = torch.softmax(prediction.logits, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1)
                    
                    return {
                        "predicted_class": predicted_class.item(),
                        "probabilities": probabilities.tolist(),
                        "confidence": torch.max(probabilities).item()
                    }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    async def get_training_job(self, job_id: str) -> Dict[str, Any]:
        """Get training job details"""
        try:
            if job_id not in self.training_jobs:
                raise ValueError(f"Training job {job_id} not found")
            
            return self.training_jobs[job_id]
            
        except Exception as e:
            logger.error(f"Error getting training job: {e}")
            raise
    
    async def list_training_jobs(self) -> List[Dict[str, Any]]:
        """List all training jobs"""
        try:
            return list(self.training_jobs.values())
            
        except Exception as e:
            logger.error(f"Error listing training jobs: {e}")
            raise
    
    async def list_deployed_models(self) -> List[Dict[str, Any]]:
        """List all deployed models"""
        try:
            return list(self.deployed_models.values())
            
        except Exception as e:
            logger.error(f"Error listing deployed models: {e}")
            raise
    
    def _validate_job_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training job configuration"""
        try:
            required_fields = ["task_type", "dataset_path"]
            
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Set default values
            config.setdefault("num_epochs", 3)
            config.setdefault("batch_size", 32)
            config.setdefault("learning_rate", 2e-5)
            config.setdefault("weight_decay", 0.01)
            config.setdefault("base_model", "distilbert-base-uncased")
            
            return config
            
        except Exception as e:
            logger.error(f"Error validating job config: {e}")
            raise


# Global custom ML training instance
custom_ml_training = CustomMLTraining()


async def initialize_custom_ml_training():
    """Initialize the custom ML training system"""
    await custom_ml_training.initialize()














