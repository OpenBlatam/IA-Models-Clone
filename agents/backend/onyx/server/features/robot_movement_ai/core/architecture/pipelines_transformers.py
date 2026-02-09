"""
Transformers Integration Module
================================

Integración profesional con Hugging Face Transformers.
Soporta fine-tuning, LoRA, y modelos pre-entrenados.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        DataCollatorWithPadding,
        PreTrainedModel,
        PreTrainedTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available.")

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT library not available for LoRA fine-tuning.")

logger = logging.getLogger(__name__)


class TransformerPipeline:
    """
    Pipeline para trabajar con modelos Transformer.
    
    Soporta:
    - Carga de modelos pre-entrenados
    - Fine-tuning completo
    - LoRA fine-tuning
    - Inferencia optimizada
    """
    
    def __init__(
        self,
        model_name: str,
        task: str = "causal_lm",
        use_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda"
    ):
        """
        Inicializar pipeline de Transformer.
        
        Args:
            model_name: Nombre del modelo (HuggingFace)
            task: Tipo de tarea ("causal_lm", "classification", etc.)
            use_lora: Usar LoRA para fine-tuning
            lora_config: Configuración de LoRA
            device: Dispositivo
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required")
        
        self.model_name = model_name
        self.task = task
        self.device = device
        self.use_lora = use_lora and PEFT_AVAILABLE
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo
        if task == "causal_lm":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto" if device == "cuda" else None
            )
        elif task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto" if device == "cuda" else None
            )
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Aplicar LoRA si se solicita
        if self.use_lora:
            self._apply_lora(lora_config)
        
        logger.info(f"TransformerPipeline initialized: {model_name} on {device}")
    
    def _apply_lora(self, lora_config: Optional[Dict[str, Any]]):
        """Aplicar LoRA al modelo."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required for LoRA")
        
        default_config = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM if self.task == "causal_lm" else TaskType.SEQ_CLS
        }
        
        config = lora_config or default_config
        peft_config = LoraConfig(**config)
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        logger.info("LoRA applied to model")
    
    def fine_tune(
        self,
        train_dataset,
        eval_dataset: Optional[Any] = None,
        output_dir: str = "./models/transformer",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        logging_steps: int = 10,
        save_steps: int = 500
    ) -> Trainer:
        """
        Fine-tune el modelo.
        
        Args:
            train_dataset: Dataset de entrenamiento
            eval_dataset: Dataset de evaluación (opcional)
            output_dir: Directorio de salida
            num_epochs: Número de épocas
            batch_size: Tamaño de batch
            learning_rate: Learning rate
            warmup_steps: Pasos de warmup
            logging_steps: Intervalo de logging
            save_steps: Intervalo de guardado
            
        Returns:
            Trainer configurado
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=True if self.device == "cuda" else False,
            gradient_accumulation_steps=1,
            report_to="wandb" if self._wandb_available() else "none"
        )
        
        # Data collator
        if self.task == "causal_lm":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        else:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        trainer.save_model()
        
        logger.info(f"Model fine-tuned and saved to {output_dir}")
        return trainer
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generar texto con el modelo.
        
        Args:
            prompt: Texto de entrada
            max_length: Longitud máxima
            temperature: Temperature para sampling
            top_p: Top-p sampling
            num_return_sequences: Número de secuencias a generar
            
        Returns:
            Lista de textos generados
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts
    
    def _wandb_available(self) -> bool:
        """Verificar si WandB está disponible."""
        try:
            import wandb
            return True
        except ImportError:
            return False

