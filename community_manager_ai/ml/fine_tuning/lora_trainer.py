"""
LoRA Trainer - Entrenador LoRA
===============================

Fine-tuning eficiente con LoRA (Low-Rank Adaptation).
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Entrenador para fine-tuning con LoRA"""
    
    def __init__(
        self,
        model_name: str,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        task_type: TaskType = TaskType.CAUSAL_LM,
        load_in_8bit: bool = False,
        device: Optional[str] = None
    ):
        """
        Inicializar entrenador LoRA
        
        Args:
            model_name: Nombre del modelo base
            r: Rango de LoRA
            lora_alpha: Alpha de LoRA
            lora_dropout: Dropout de LoRA
            target_modules: Módulos objetivo (auto-detecta si es None)
            task_type: Tipo de tarea
            load_in_8bit: Cargar en 8-bit
            device: Dispositivo
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo
        model_kwargs = {}
        if load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                logger.warning("bitsandbytes no disponible")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Preparar para entrenamiento
        if load_in_8bit:
            base_model = prepare_model_for_kbit_training(base_model)
        
        # Configurar LoRA
        if target_modules is None:
            # Auto-detectar módulos objetivo según arquitectura
            if "gpt" in model_name.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            else:
                target_modules = ["q_proj", "v_proj"]
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=task_type
        )
        
        # Aplicar LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info(f"LoRA Trainer inicializado para {model_name}")
    
    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        output_dir: str = "./lora_output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        use_amp: bool = True
    ):
        """
        Entrenar modelo con LoRA
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            output_dir: Directorio de salida
            num_train_epochs: Número de épocas
            per_device_train_batch_size: Batch size por dispositivo
            gradient_accumulation_steps: Pasos de acumulación
            learning_rate: Learning rate
            warmup_steps: Pasos de warmup
            logging_steps: Pasos de logging
            save_steps: Pasos de guardado
            use_amp: Usar mixed precision
        """
        # Argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=save_steps if val_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            fp16=use_amp and self.device == "cuda",
            bf16=False,
            optim="paged_adamw_8bit" if hasattr(self.model, "is_loaded_in_8bit") else "adamw_torch",
            report_to="wandb" if use_amp else None,
            run_name=f"lora_{self.model_name.split('/')[-1]}"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Entrenar
        trainer.train()
        
        # Guardar
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Modelo LoRA entrenado y guardado en {output_dir}")
    
    def merge_and_save(self, output_path: str):
        """
        Fusionar LoRA con modelo base y guardar
        
        Args:
            output_path: Ruta de salida
        """
        # Fusionar adaptadores
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Modelo fusionado guardado en {output_path}")




