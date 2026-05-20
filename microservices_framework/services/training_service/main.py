"""
Training Service - Model Training & Fine-tuning
Provides endpoints for training transformer models, fine-tuning, and LoRA training.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import structlog
import json
from pathlib import Path

logger = structlog.get_logger()

_device = "cuda" if torch.cuda.is_available() else "cpu"


class TrainingJobRequest(BaseModel):
    """Request model for training job."""
    model_name: str = Field(..., description="Base model identifier")
    task_type: str = Field(..., description="Task type: causal_lm, classification, etc.")
    dataset_path: str = Field(..., description="Path to dataset or HuggingFace dataset name")
    output_dir: str = Field(default="./models", description="Output directory for trained model")
    num_epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=5e-5, ge=1e-6, le=1e-2)
    max_length: int = Field(default=512, ge=64, le=2048)
    use_lora: bool = Field(default=False, description="Use LoRA for efficient fine-tuning")
    lora_r: int = Field(default=8, description="LoRA rank")
    lora_alpha: int = Field(default=16, description="LoRA alpha")
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    warmup_steps: int = Field(default=100)
    save_steps: int = Field(default=500)
    eval_steps: Optional[int] = Field(default=None)
    logging_steps: int = Field(default=10)
    fp16: bool = Field(default=True, description="Use mixed precision training")


class TrainingJobResponse(BaseModel):
    """Response model for training job."""
    job_id: str
    status: str
    model_name: str
    output_dir: str
    estimated_time: Optional[str] = None


class TrainingJobStatus(BaseModel):
    """Training job status response."""
    job_id: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    logs: List[Dict[str, Any]] = []


# In-memory job tracking (use Redis in production)
_training_jobs: Dict[str, Dict[str, Any]] = {}


def load_dataset_from_path(dataset_path: str) -> Dataset:
    """Load dataset from path or HuggingFace."""
    try:
        if Path(dataset_path).exists():
            # Load from local file
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            return Dataset.from_dict(data)
        else:
            # Try loading from HuggingFace
            return load_dataset(dataset_path, split="train")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load dataset: {str(e)}"
        )


def create_lora_config(r: int = 8, alpha: int = 16, task_type: str = "CAUSAL_LM"):
    """Create LoRA configuration."""
    task_type_map = {
        "causal_lm": TaskType.CAUSAL_LM,
        "classification": TaskType.SEQ_CLS,
    }
    
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=task_type_map.get(task_type, TaskType.CAUSAL_LM),
    )


async def train_model_async(
    job_id: str,
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks
):
    """Async training function."""
    try:
        _training_jobs[job_id]["status"] = "running"
        
        logger.info("training_started", job_id=job_id, model=request.model_name)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(request.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if request.task_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                request.model_name,
                torch_dtype=torch.float16 if request.fp16 and _device == "cuda" else torch.float32,
            )
        elif request.task_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                request.model_name,
                num_labels=2,  # Default, should be configurable
                torch_dtype=torch.float16 if request.fp16 and _device == "cuda" else torch.float32,
            )
        else:
            raise ValueError(f"Unsupported task type: {request.task_type}")
        
        # Apply LoRA if requested
        if request.use_lora:
            lora_config = create_lora_config(
                r=request.lora_r,
                alpha=request.lora_alpha,
                task_type=request.task_type
            )
            model = get_peft_model(model, lora_config)
            logger.info("lora_applied", job_id=job_id)
        
        if _device == "cuda":
            model = model.to(_device)
        
        # Load dataset
        dataset = load_dataset_from_path(request.dataset_path)
        
        # Tokenize dataset
        def tokenize_function(examples):
            if request.task_type == "causal_lm":
                return tokenizer(
                    examples.get("text", examples.get("input", "")),
                    truncation=True,
                    max_length=request.max_length,
                    padding="max_length",
                )
            else:
                return tokenizer(
                    examples.get("text", examples.get("input", "")),
                    truncation=True,
                    max_length=request.max_length,
                    padding="max_length",
                )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        # Data collator
        if request.task_type == "causal_lm":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
        else:
            data_collator = None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=request.output_dir,
            num_train_epochs=request.num_epochs,
            per_device_train_batch_size=request.batch_size,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            learning_rate=request.learning_rate,
            warmup_steps=request.warmup_steps,
            logging_steps=request.logging_steps,
            save_steps=request.save_steps,
            eval_steps=request.eval_steps,
            fp16=request.fp16 and _device == "cuda",
            save_total_limit=3,
            load_best_model_at_end=True if request.eval_steps else False,
            report_to="none",  # Can be "tensorboard" or "wandb"
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Training callback for progress tracking
        class ProgressCallback:
            def __init__(self, job_id: str):
                self.job_id = job_id
                self.current_step = 0
                self.total_steps = len(tokenized_dataset) // (request.batch_size * request.gradient_accumulation_steps) * request.num_epochs
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    _training_jobs[self.job_id]["logs"].append(logs)
                    _training_jobs[self.job_id]["loss"] = logs.get("loss")
                    _training_jobs[self.job_id]["learning_rate"] = logs.get("learning_rate")
                    self.current_step = state.global_step
                    _training_jobs[self.job_id]["progress"] = min(
                        self.current_step / self.total_steps, 1.0
                    )
                    _training_jobs[self.job_id]["current_epoch"] = int(state.epoch)
        
        trainer.add_callback(ProgressCallback(job_id))
        
        # Run training in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, trainer.train)
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(request.output_dir)
        
        _training_jobs[job_id]["status"] = "completed"
        logger.info("training_completed", job_id=job_id, output_dir=request.output_dir)
        
    except Exception as e:
        _training_jobs[job_id]["status"] = "failed"
        _training_jobs[job_id]["error"] = str(e)
        logger.error("training_failed", job_id=job_id, error=str(e))
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    logger.info("training_service_starting", device=_device)
    yield
    logger.info("training_service_shutting_down")


app = FastAPI(
    title="Training Service",
    description="Model Training & Fine-tuning Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "training_service",
        "device": _device,
        "cuda_available": torch.cuda.is_available(),
        "active_jobs": len([j for j in _training_jobs.values() if j["status"] == "running"]),
    }


@app.post("/train", response_model=TrainingJobResponse)
async def start_training(
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a training job.
    
    Returns immediately with job ID. Training runs in background.
    """
    import uuid
    job_id = str(uuid.uuid4())
    
    _training_jobs[job_id] = {
        "status": "queued",
        "request": request.dict(),
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": request.num_epochs,
        "logs": [],
    }
    
    # Start training in background
    background_tasks.add_task(train_model_async, job_id, request, background_tasks)
    
    return TrainingJobResponse(
        job_id=job_id,
        status="queued",
        model_name=request.model_name,
        output_dir=request.output_dir,
    )


@app.get("/jobs/{job_id}/status", response_model=TrainingJobStatus)
async def get_training_status(job_id: str):
    """Get training job status."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _training_jobs[job_id]
    
    return TrainingJobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        current_epoch=job.get("current_epoch", 0),
        total_epochs=job.get("total_epochs", 0),
        loss=job.get("loss"),
        learning_rate=job.get("learning_rate"),
        logs=job.get("logs", [])[-10:],  # Last 10 log entries
    )


@app.get("/jobs")
async def list_jobs():
    """List all training jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "model_name": job["request"]["model_name"],
                "progress": job.get("progress", 0.0),
            }
            for job_id, job in _training_jobs.items()
        ]
    }


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a training job."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if _training_jobs[job_id]["status"] == "running":
        # In production, implement proper cancellation
        _training_jobs[job_id]["status"] = "cancelling"
        return {"status": "cancelling", "job_id": job_id}
    
    del _training_jobs[job_id]
    return {"status": "deleted", "job_id": job_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info",
    )



