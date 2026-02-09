"""
Fine-Tuning Service - Servicio de fine-tuning con mejores prácticas
====================================================================

Sistema profesional para fine-tuning de modelos usando:
- LoRA (Low-Rank Adaptation) con PEFT
- P-tuning
- Full fine-tuning
- Adapters

Sigue mejores prácticas de Transformers y PEFT.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False
    logger.warning("PyTorch/Transformers not available")

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. LoRA fine-tuning will not work.")


class FineTuningMethod(str, Enum):
    """Métodos de fine-tuning"""
    FULL = "full"
    LORA = "lora"
    P_TUNING = "p_tuning"
    ADAPTERS = "adapters"


class TrainingStatus(str, Enum):
    """Estados de entrenamiento"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LoRAConfig:
    """Configuración de LoRA"""
    r: int = 8  # Rank
    lora_alpha: int = 16  # LoRA alpha
    target_modules: Optional[List[str]] = None  # None = auto-detect
    lora_dropout: float = 0.1
    bias: str = "none"  # none, all, lora_only
    task_type: TaskType = TaskType.CAUSAL_LM


@dataclass
class FineTuningConfig:
    """Configuración completa de fine-tuning"""
    model_name: str
    method: FineTuningMethod = FineTuningMethod.LORA
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    max_seq_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    # LoRA specific
    lora_config: Optional[LoRAConfig] = None
    # Output
    output_dir: str = "./fine_tuned_models"
    save_steps: int = 500
    eval_steps: Optional[int] = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    # Evaluation
    evaluation_strategy: str = "steps"  # no, steps, epoch


@dataclass
class TrainingJob:
    """Job de entrenamiento"""
    id: str
    config: FineTuningConfig
    status: TrainingStatus
    dataset_path: str
    output_dir: str
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    model_path: Optional[str] = None


class FineTuningService:
    """Servicio de fine-tuning profesional"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if TORCH_AVAILABLE
            else None
        )
        logger.info(
            f"FineTuningService initialized "
            f"(PEFT: {PEFT_AVAILABLE}, Device: {self.device})"
        )
    
    def create_training_job(
        self,
        config: FineTuningConfig,
        dataset_path: str,
        output_dir: Optional[str] = None
    ) -> TrainingJob:
        """
        Crear job de entrenamiento.
        
        Args:
            config: Configuración de fine-tuning
            dataset_path: Ruta al dataset
            output_dir: Directorio de salida (usa config si None)
        
        Returns:
            TrainingJob creado
        """
        job_id = f"training_{int(datetime.now().timestamp())}"
        
        # Validate method availability
        if config.method == FineTuningMethod.LORA and not PEFT_AVAILABLE:
            raise ValueError("LoRA requires PEFT library. Install with: pip install peft")
        
        # Setup LoRA config if using LoRA
        if config.method == FineTuningMethod.LORA and config.lora_config is None:
            config.lora_config = LoRAConfig()
        
        # Create output directory
        final_output_dir = output_dir or config.output_dir
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)
        
        job = TrainingJob(
            id=job_id,
            config=config,
            status=TrainingStatus.PENDING,
            dataset_path=dataset_path,
            output_dir=final_output_dir,
        )
        
        self.training_jobs[job_id] = job
        
        logger.info(f"Training job created: {job_id} (Method: {config.method.value})")
        return job
    
    def _prepare_model_for_training(
        self,
        config: FineTuningConfig
    ) -> tuple:
        """
        Preparar modelo para entrenamiento.
        
        Returns:
            (model, tokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not available")
        
        logger.info(f"Loading model {config.model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        # Apply LoRA if using LoRA
        if config.method == FineTuningMethod.LORA:
            if not PEFT_AVAILABLE:
                raise RuntimeError("PEFT library required for LoRA")
            
            lora_config_obj = config.lora_config or LoRAConfig()
            
            # Auto-detect target modules if not specified
            target_modules = lora_config_obj.target_modules
            if target_modules is None:
                # Common transformer modules
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                # Try to detect actual modules in model
                if hasattr(model, "model"):
                    model_base = model.model
                else:
                    model_base = model
                
                # Check for different architectures
                if hasattr(model_base, "layers"):
                    first_layer = model_base.layers[0]
                    available_modules = [name for name, _ in first_layer.named_modules() if "proj" in name]
                    if available_modules:
                        target_modules = available_modules
            
            peft_config = LoraConfig(
                r=lora_config_obj.r,
                lora_alpha=lora_config_obj.lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_config_obj.lora_dropout,
                bias=lora_config_obj.bias,
                task_type=lora_config_obj.task_type,
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
            logger.info(f"LoRA applied with r={lora_config_obj.r}, alpha={lora_config_obj.lora_alpha}")
        
        return model, tokenizer
    
    def _create_training_arguments(
        self,
        config: FineTuningConfig,
        output_dir: str
    ) -> TrainingArguments:
        """Crear TrainingArguments"""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            fp16=config.fp16,
            bf16=config.bf16,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            evaluation_strategy=config.evaluation_strategy,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=config.load_best_model_at_end,
            report_to="none",  # Can be "tensorboard", "wandb", etc.
            remove_unused_columns=False,
        )
    
    async def start_training(
        self,
        job_id: str,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None
    ) -> TrainingJob:
        """
        Iniciar entrenamiento.
        
        Args:
            job_id: ID del job
            train_dataset: Dataset de entrenamiento (opcional)
            eval_dataset: Dataset de evaluación (opcional)
        
        Returns:
            TrainingJob actualizado
        """
        job = self.training_jobs.get(job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        if job.status != TrainingStatus.PENDING:
            raise ValueError(f"Training job {job_id} is not pending")
        
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers library not available")
            
            # Prepare model
            model, tokenizer = self._prepare_model_for_training(job.config)
            
            # Load datasets if not provided
            if train_dataset is None:
                # In production, load from dataset_path
                # For now, raise error
                raise NotImplementedError("Dataset loading from path not implemented yet")
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )
            
            # Create training arguments
            training_args = self._create_training_arguments(
                job.config,
                job.output_dir
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            # Train
            logger.info(f"Starting training for job {job_id}...")
            trainer.train()
            
            # Save final model
            final_model_path = Path(job.output_dir) / "final_model"
            trainer.save_model(str(final_model_path))
            tokenizer.save_pretrained(str(final_model_path))
            
            job.model_path = str(final_model_path)
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Get training metrics
            if hasattr(trainer.state, "log_history"):
                job.metrics = {
                    "train_loss": [
                        log.get("loss", 0) for log in trainer.state.log_history
                        if "loss" in log
                    ],
                    "eval_loss": [
                        log.get("eval_loss", 0) for log in trainer.state.log_history
                        if "eval_loss" in log
                    ],
                }
            
            logger.info(f"Training completed for job {job_id}")
            
        except Exception as e:
            logger.error(f"Training failed for job {job_id}: {e}", exc_info=True)
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            raise
        
        return job
    
    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Obtener estado de entrenamiento"""
        job = self.training_jobs.get(job_id)
        if not job:
            return {"exists": False}
        
        return {
            "id": job.id,
            "status": job.status.value,
            "model_name": job.config.model_name,
            "method": job.config.method.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "metrics": job.metrics,
            "error_message": job.error_message,
            "model_path": job.model_path,
        }
    
    def prepare_lora_config(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.1
    ) -> LoRAConfig:
        """
        Preparar configuración LoRA.
        
        Args:
            r: Rank de LoRA
            lora_alpha: Alpha de LoRA
            target_modules: Módulos objetivo (None = auto-detect)
            lora_dropout: Dropout de LoRA
        
        Returns:
            LoRAConfig
        """
        return LoRAConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
        )
