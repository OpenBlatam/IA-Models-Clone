"""
Custom Model Training and Fine-tuning Engine
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    pipeline, AutoConfig
)
from datasets import Dataset
import joblib
from pydantic import BaseModel, Field
import wandb
from config import settings

logger = logging.getLogger(__name__)


class TrainingDataset(BaseModel):
    """Model for training dataset"""
    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    data: List[Dict[str, Any]] = Field(..., description="Training data")
    labels: List[str] = Field(..., description="Available labels")
    task_type: str = Field(..., description="Task type: classification, regression, generation")
    language: str = Field(default="en", description="Dataset language")


class TrainingConfig(BaseModel):
    """Model for training configuration"""
    model_name: str = Field(..., description="Base model name")
    task_type: str = Field(..., description="Task type")
    num_labels: int = Field(..., description="Number of labels")
    max_length: int = Field(default=512, description="Maximum sequence length")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    batch_size: int = Field(default=16, description="Batch size")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    warmup_steps: int = Field(default=500, description="Warmup steps")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    save_steps: int = Field(default=500, description="Save model every N steps")
    eval_steps: int = Field(default=500, description="Evaluate model every N steps")
    logging_steps: int = Field(default=100, description="Log metrics every N steps")


class TrainingJob(BaseModel):
    """Model for training job"""
    job_id: str = Field(..., description="Unique job ID")
    dataset: TrainingDataset = Field(..., description="Training dataset")
    config: TrainingConfig = Field(..., description="Training configuration")
    status: str = Field(default="pending", description="Job status")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Training progress")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Training metrics")
    model_path: Optional[str] = Field(default=None, description="Path to trained model")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = Field(default=None)
    completed_at: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)


class ModelEvaluationResult(BaseModel):
    """Model for evaluation results"""
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    classification_report: Dict[str, Any] = Field(..., description="Detailed classification report")
    test_loss: float = Field(..., description="Test loss")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class CustomModelTrainingEngine:
    """Engine for custom model training and fine-tuning"""
    
    def __init__(self):
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.trained_models: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}
        self.wandb_initialized = False
    
    async def initialize(self):
        """Initialize the training engine"""
        try:
            # Initialize Weights & Biases for experiment tracking
            if hasattr(settings, 'wandb_project') and settings.wandb_project:
                wandb.init(project=settings.wandb_project)
                self.wandb_initialized = True
                logger.info("Initialized Weights & Biases tracking")
            
            # Create models directory
            os.makedirs("models", exist_ok=True)
            os.makedirs("datasets", exist_ok=True)
            
            logger.info("Custom model training engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing training engine: {e}")
    
    async def create_training_job(self, dataset: TrainingDataset, config: TrainingConfig) -> str:
        """Create a new training job"""
        job_id = str(uuid.uuid4())
        
        training_job = TrainingJob(
            job_id=job_id,
            dataset=dataset,
            config=config
        )
        
        self.training_jobs[job_id] = training_job
        
        # Save job to disk
        await self._save_training_job(training_job)
        
        logger.info(f"Created training job: {job_id}")
        return job_id
    
    async def start_training(self, job_id: str) -> bool:
        """Start training a model"""
        if job_id not in self.training_jobs:
            raise ValueError(f"Training job {job_id} not found")
        
        training_job = self.training_jobs[job_id]
        
        if training_job.status != "pending":
            raise ValueError(f"Training job {job_id} is not in pending status")
        
        try:
            # Update job status
            training_job.status = "running"
            training_job.started_at = datetime.now().isoformat()
            await self._save_training_job(training_job)
            
            # Start training in background
            asyncio.create_task(self._run_training(training_job))
            
            logger.info(f"Started training job: {job_id}")
            return True
            
        except Exception as e:
            training_job.status = "failed"
            training_job.error_message = str(e)
            await self._save_training_job(training_job)
            logger.error(f"Error starting training job {job_id}: {e}")
            return False
    
    async def _run_training(self, training_job: TrainingJob):
        """Run the actual training process"""
        try:
            # Prepare dataset
            train_dataset, eval_dataset = await self._prepare_dataset(training_job.dataset)
            
            # Load model and tokenizer
            model, tokenizer = await self._load_model_and_tokenizer(training_job.config)
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=f"models/{training_job.job_id}",
                num_train_epochs=training_job.config.num_epochs,
                per_device_train_batch_size=training_job.config.batch_size,
                per_device_eval_batch_size=training_job.config.batch_size,
                warmup_steps=training_job.config.warmup_steps,
                weight_decay=training_job.config.weight_decay,
                learning_rate=training_job.config.learning_rate,
                logging_dir=f"logs/{training_job.job_id}",
                logging_steps=training_job.config.logging_steps,
                save_steps=training_job.config.save_steps,
                eval_steps=training_job.config.eval_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="wandb" if self.wandb_initialized else None,
                run_name=f"training_{training_job.job_id}"
            )
            
            # Setup data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=self._compute_metrics
            )
            
            # Start training
            trainer.train()
            
            # Save final model
            model_path = f"models/{training_job.job_id}/final_model"
            trainer.save_model(model_path)
            tokenizer.save_pretrained(model_path)
            
            # Update job status
            training_job.status = "completed"
            training_job.completed_at = datetime.now().isoformat()
            training_job.model_path = model_path
            training_job.progress = 1.0
            
            # Get final metrics
            eval_results = trainer.evaluate()
            training_job.metrics = eval_results
            
            await self._save_training_job(training_job)
            
            # Cache the trained model
            self.trained_models[training_job.job_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": training_job.config,
                "metrics": eval_results
            }
            
            logger.info(f"Completed training job: {training_job.job_id}")
            
        except Exception as e:
            training_job.status = "failed"
            training_job.error_message = str(e)
            training_job.completed_at = datetime.now().isoformat()
            await self._save_training_job(training_job)
            logger.error(f"Error in training job {training_job.job_id}: {e}")
    
    async def _prepare_dataset(self, dataset: TrainingDataset) -> Tuple[Dataset, Dataset]:
        """Prepare dataset for training"""
        try:
            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset.data)
            
            # Split into train and eval
            train_df, eval_df = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42,
                stratify=df['label'] if 'label' in df.columns else None
            )
            
            # Convert to HuggingFace datasets
            train_dataset = Dataset.from_pandas(train_df)
            eval_dataset = Dataset.from_pandas(eval_df)
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise
    
    async def _load_model_and_tokenizer(self, config: TrainingConfig) -> Tuple[Any, Any]:
        """Load model and tokenizer"""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                config.model_name,
                num_labels=config.num_labels
            )
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                config=model_config
            )
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {e}")
            raise
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    async def evaluate_model(self, job_id: str, test_data: List[Dict[str, Any]]) -> ModelEvaluationResult:
        """Evaluate a trained model"""
        if job_id not in self.trained_models:
            raise ValueError(f"Trained model {job_id} not found")
        
        try:
            model_info = self.trained_models[job_id]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Create evaluation pipeline
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Prepare test data
            test_texts = [item["text"] for item in test_data]
            test_labels = [item["label"] for item in test_data]
            
            # Get predictions
            predictions = classifier(test_texts)
            predicted_labels = [pred["label"] for pred in predictions]
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predicted_labels)
            precision = precision_score(test_labels, predicted_labels, average='weighted')
            recall = recall_score(test_labels, predicted_labels, average='weighted')
            f1 = f1_score(test_labels, predicted_labels, average='weighted')
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(test_labels, predicted_labels)
            
            # Create classification report
            from sklearn.metrics import classification_report
            report = classification_report(test_labels, predicted_labels, output_dict=True)
            
            return ModelEvaluationResult(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=cm.tolist(),
                classification_report=report,
                test_loss=0.0  # Would need to compute this separately
            )
            
        except Exception as e:
            logger.error(f"Error evaluating model {job_id}: {e}")
            raise
    
    async def deploy_model(self, job_id: str, model_name: str) -> bool:
        """Deploy a trained model for inference"""
        if job_id not in self.trained_models:
            raise ValueError(f"Trained model {job_id} not found")
        
        try:
            model_info = self.trained_models[job_id]
            
            # Create deployment directory
            deploy_path = f"models/deployed/{model_name}"
            os.makedirs(deploy_path, exist_ok=True)
            
            # Copy model files
            import shutil
            source_path = model_info["model_path"]
            shutil.copytree(source_path, deploy_path, dirs_exist_ok=True)
            
            # Create model metadata
            metadata = {
                "model_name": model_name,
                "job_id": job_id,
                "deployed_at": datetime.now().isoformat(),
                "config": model_info["config"].model_dump(),
                "metrics": model_info["metrics"]
            }
            
            with open(f"{deploy_path}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Cache deployed model
            self.model_cache[model_name] = {
                "path": deploy_path,
                "metadata": metadata,
                "model_info": model_info
            }
            
            logger.info(f"Deployed model {model_name} from job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model {job_id}: {e}")
            return False
    
    async def load_deployed_model(self, model_name: str) -> Any:
        """Load a deployed model for inference"""
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        
        try:
            deploy_path = f"models/deployed/{model_name}"
            
            if not os.path.exists(deploy_path):
                raise ValueError(f"Deployed model {model_name} not found")
            
            # Load metadata
            with open(f"{deploy_path}/metadata.json", "r") as f:
                metadata = json.load(f)
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(deploy_path)
            model = AutoModelForSequenceClassification.from_pretrained(deploy_path)
            
            # Create pipeline
            pipeline_obj = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            model_info = {
                "pipeline": pipeline_obj,
                "metadata": metadata,
                "path": deploy_path
            }
            
            # Cache the model
            self.model_cache[model_name] = model_info
            
            logger.info(f"Loaded deployed model: {model_name}")
            return model_info
            
        except Exception as e:
            logger.error(f"Error loading deployed model {model_name}: {e}")
            raise
    
    async def predict_with_custom_model(self, model_name: str, text: str) -> Dict[str, Any]:
        """Make prediction with a custom trained model"""
        try:
            model_info = await self.load_deployed_model(model_name)
            pipeline_obj = model_info["pipeline"]
            
            # Make prediction
            result = pipeline_obj(text)
            
            return {
                "model_name": model_name,
                "prediction": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with model {model_name}: {e}")
            raise
    
    async def get_training_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get status of a training job"""
        return self.training_jobs.get(job_id)
    
    async def list_training_jobs(self) -> List[TrainingJob]:
        """List all training jobs"""
        return list(self.training_jobs.values())
    
    async def list_deployed_models(self) -> List[Dict[str, Any]]:
        """List all deployed models"""
        return [
            {
                "model_name": name,
                "metadata": info["metadata"]
            }
            for name, info in self.model_cache.items()
        ]
    
    async def delete_training_job(self, job_id: str) -> bool:
        """Delete a training job and its associated files"""
        if job_id not in self.training_jobs:
            return False
        
        try:
            training_job = self.training_jobs[job_id]
            
            # Delete model files
            if training_job.model_path and os.path.exists(training_job.model_path):
                import shutil
                shutil.rmtree(training_job.model_path)
            
            # Delete job file
            job_file = f"jobs/{job_id}.json"
            if os.path.exists(job_file):
                os.remove(job_file)
            
            # Remove from memory
            del self.training_jobs[job_id]
            
            logger.info(f"Deleted training job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting training job {job_id}: {e}")
            return False
    
    async def _save_training_job(self, training_job: TrainingJob):
        """Save training job to disk"""
        try:
            os.makedirs("jobs", exist_ok=True)
            job_file = f"jobs/{training_job.job_id}.json"
            
            with open(job_file, "w") as f:
                json.dump(training_job.model_dump(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving training job: {e}")


# Global custom model training engine
custom_training_engine = CustomModelTrainingEngine()


async def initialize_custom_training_engine():
    """Initialize the custom model training engine"""
    await custom_training_engine.initialize()
















