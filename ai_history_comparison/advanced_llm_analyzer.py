"""
Advanced Large Language Model Analysis System
Sistema avanzado de análisis de modelos de lenguaje grandes
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import torch.optim as optim
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Transformers imports
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, pipeline, BitsAndBytesConfig
    )
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from transformers import BertTokenizer, BertModel
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Diffusers imports
try:
    from diffusers import (
        StableDiffusionPipeline, StableDiffusionXLPipeline,
        DDPMPipeline, DDIMPipeline, PNDMPipeline,
        UNet2DModel, DDPMScheduler, DDIMScheduler
    )
    from diffusers.utils import make_image_grid
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# Gradio imports
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

# Additional imports
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Tipos de modelos"""
    CAUSAL_LM = "causal_lm"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    QUESTION_ANSWERING = "question_answering"
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARIZATION = "text_summarization"
    TRANSLATION = "translation"
    DIFFUSION = "diffusion"
    VISION_TRANSFORMER = "vision_transformer"

class TaskType(Enum):
    """Tipos de tareas"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    FINE_TUNING = "fine_tuning"
    PROMPT_ENGINEERING = "prompt_engineering"
    RAG = "rag"  # Retrieval Augmented Generation

class OptimizationType(Enum):
    """Tipos de optimización"""
    LORA = "lora"
    P_TUNING = "p_tuning"
    ADAPTERS = "adapters"
    FULL_FINE_TUNING = "full_fine_tuning"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"

@dataclass
class ModelConfig:
    """Configuración de modelo"""
    model_name: str
    model_type: ModelType
    task_type: TaskType
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    optimization_type: OptimizationType = OptimizationType.FULL_FINE_TUNING
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TrainingResult:
    """Resultado de entrenamiento"""
    id: str
    model_config: ModelConfig
    training_history: Dict[str, List[float]]
    final_metrics: Dict[str, float]
    training_time: float
    best_epoch: int
    model_path: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ModelEvaluation:
    """Evaluación de modelo"""
    id: str
    model_id: str
    task_type: TaskType
    metrics: Dict[str, float]
    predictions: List[Any]
    ground_truth: List[Any]
    confidence_scores: List[float]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PromptTemplate:
    """Plantilla de prompt"""
    id: str
    name: str
    template: str
    variables: List[str]
    task_type: TaskType
    effectiveness_score: float = 0.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedLLMAnalyzer:
    """
    Analizador avanzado de modelos de lenguaje grandes
    """
    
    def __init__(
        self,
        enable_torch: bool = True,
        enable_transformers: bool = True,
        enable_diffusers: bool = True,
        enable_gradio: bool = True,
        models_directory: str = "models/llm/",
        cache_directory: str = "cache/",
        device: str = "auto"
    ):
        self.enable_torch = enable_torch and TORCH_AVAILABLE
        self.enable_transformers = enable_transformers and TRANSFORMERS_AVAILABLE
        self.enable_diffusers = enable_diffusers and DIFFUSERS_AVAILABLE
        self.enable_gradio = enable_gradio and GRADIO_AVAILABLE
        
        # Configurar dispositivo
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Directorios
        self.models_directory = models_directory
        self.cache_directory = cache_directory
        
        # Almacenamiento
        self.model_configs: Dict[str, ModelConfig] = {}
        self.training_results: Dict[str, TrainingResult] = {}
        self.model_evaluations: Dict[str, ModelEvaluation] = {}
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        self.loaded_models: Dict[str, Any] = {}
        
        # Configuración
        self.config = {
            "default_max_length": 512,
            "default_batch_size": 16,
            "default_learning_rate": 5e-5,
            "default_num_epochs": 3,
            "mixed_precision": True,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 4,
            "save_steps": 500,
            "eval_steps": 500,
            "logging_steps": 100
        }
        
        # Crear directorios
        os.makedirs(self.models_directory, exist_ok=True)
        os.makedirs(self.cache_directory, exist_ok=True)
        
        # Inicializar modelos base
        self._initialize_base_models()
        
        logger.info(f"Advanced LLM Analyzer inicializado en dispositivo: {self.device}")
    
    def _initialize_base_models(self):
        """Inicializar modelos base"""
        try:
            if self.enable_transformers:
                # Modelos pre-entrenados comunes
                self.base_models = {
                    "gpt2": "gpt2",
                    "gpt2-medium": "gpt2-medium",
                    "gpt2-large": "gpt2-large",
                    "bert-base": "bert-base-uncased",
                    "bert-large": "bert-large-uncased",
                    "t5-small": "t5-small",
                    "t5-base": "t5-base",
                    "t5-large": "t5-large",
                    "roberta-base": "roberta-base",
                    "distilbert": "distilbert-base-uncased"
                }
                logger.info("Modelos base inicializados")
            
            if self.enable_diffusers:
                self.diffusion_models = {
                    "stable-diffusion": "runwayml/stable-diffusion-v1-5",
                    "stable-diffusion-xl": "stabilityai/stable-diffusion-xl-base-1.0"
                }
                logger.info("Modelos de difusión inicializados")
                
        except Exception as e:
            logger.error(f"Error inicializando modelos base: {e}")
    
    async def load_model(
        self,
        model_name: str,
        model_type: ModelType,
        task_type: TaskType,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Cargar modelo
        
        Args:
            model_name: Nombre del modelo
            model_type: Tipo de modelo
            task_type: Tipo de tarea
            custom_config: Configuración personalizada
            
        Returns:
            ID del modelo cargado
        """
        try:
            if not self.enable_transformers:
                raise ValueError("Transformers no disponible")
            
            model_id = f"{model_name}_{model_type.value}_{task_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Cargando modelo: {model_name} para tarea: {task_type.value}")
            
            # Cargar tokenizador
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_directory)
            
            # Configurar tokenizador
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Cargar modelo según el tipo
            if model_type == ModelType.CAUSAL_LM:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_directory,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
            elif model_type == ModelType.SEQUENCE_CLASSIFICATION:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=self.cache_directory,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.cache_directory,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
            
            # Mover modelo al dispositivo
            model = model.to(self.device)
            
            # Configurar para entrenamiento
            if custom_config and custom_config.get("gradient_checkpointing", True):
                model.gradient_checkpointing_enable()
            
            # Almacenar modelo
            self.loaded_models[model_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "model_name": model_name,
                "model_type": model_type,
                "task_type": task_type,
                "config": custom_config or {}
            }
            
            logger.info(f"Modelo cargado exitosamente: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    async def create_model_config(
        self,
        model_name: str,
        model_type: ModelType,
        task_type: TaskType,
        **kwargs
    ) -> ModelConfig:
        """
        Crear configuración de modelo
        
        Args:
            model_name: Nombre del modelo
            model_type: Tipo de modelo
            task_type: Tipo de tarea
            **kwargs: Parámetros adicionales
            
        Returns:
            Configuración del modelo
        """
        try:
            config = ModelConfig(
                model_name=model_name,
                model_type=model_type,
                task_type=task_type,
                max_length=kwargs.get("max_length", self.config["default_max_length"]),
                batch_size=kwargs.get("batch_size", self.config["default_batch_size"]),
                learning_rate=kwargs.get("learning_rate", self.config["default_learning_rate"]),
                num_epochs=kwargs.get("num_epochs", self.config["default_num_epochs"]),
                warmup_steps=kwargs.get("warmup_steps", 100),
                weight_decay=kwargs.get("weight_decay", 0.01),
                gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
                mixed_precision=kwargs.get("mixed_precision", self.config["mixed_precision"]),
                optimization_type=kwargs.get("optimization_type", OptimizationType.FULL_FINE_TUNING)
            )
            
            config_id = f"config_{model_name}_{model_type.value}_{task_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.model_configs[config_id] = config
            
            logger.info(f"Configuración de modelo creada: {config_id}")
            return config
            
        except Exception as e:
            logger.error(f"Error creando configuración de modelo: {e}")
            raise
    
    async def fine_tune_model(
        self,
        model_id: str,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[ModelConfig] = None
    ) -> TrainingResult:
        """
        Fine-tuning de modelo
        
        Args:
            model_id: ID del modelo
            train_dataset: Dataset de entrenamiento
            eval_dataset: Dataset de evaluación
            config: Configuración del modelo
            
        Returns:
            Resultado del entrenamiento
        """
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            model_info = self.loaded_models[model_id]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            logger.info(f"Iniciando fine-tuning del modelo: {model_id}")
            
            # Configurar argumentos de entrenamiento
            training_args = TrainingArguments(
                output_dir=f"{self.models_directory}/fine_tuned_{model_id}",
                num_train_epochs=config.num_epochs if config else self.config["default_num_epochs"],
                per_device_train_batch_size=config.batch_size if config else self.config["default_batch_size"],
                per_device_eval_batch_size=config.batch_size if config else self.config["default_batch_size"],
                warmup_steps=config.warmup_steps if config else 100,
                weight_decay=config.weight_decay if config else 0.01,
                learning_rate=config.learning_rate if config else self.config["default_learning_rate"],
                logging_dir=f"{self.models_directory}/logs",
                logging_steps=self.config["logging_steps"],
                save_steps=self.config["save_steps"],
                eval_steps=self.config["eval_steps"],
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                greater_is_better=False,
                fp16=config.mixed_precision if config else self.config["mixed_precision"],
                gradient_accumulation_steps=config.gradient_accumulation_steps if config else 1,
                dataloader_num_workers=self.config["dataloader_num_workers"],
                remove_unused_columns=False
            )
            
            # Crear trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer
            )
            
            # Entrenar modelo
            start_time = datetime.now()
            trainer.train()
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Obtener historial de entrenamiento
            training_history = trainer.state.log_history
            
            # Calcular métricas finales
            final_metrics = {}
            if eval_dataset:
                eval_results = trainer.evaluate()
                final_metrics.update(eval_results)
            
            # Encontrar mejor época
            best_epoch = 1
            if training_history:
                best_loss = float('inf')
                for log in training_history:
                    if 'eval_loss' in log and log['eval_loss'] < best_loss:
                        best_loss = log['eval_loss']
                        best_epoch = log.get('epoch', 1)
            
            # Guardar modelo
            model_path = f"{self.models_directory}/fine_tuned_{model_id}"
            trainer.save_model(model_path)
            tokenizer.save_pretrained(model_path)
            
            # Crear resultado de entrenamiento
            result = TrainingResult(
                id=f"training_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_config=config or ModelConfig(
                    model_name=model_info["model_name"],
                    model_type=model_info["model_type"],
                    task_type=model_info["task_type"]
                ),
                training_history=training_history,
                final_metrics=final_metrics,
                training_time=training_time,
                best_epoch=best_epoch,
                model_path=model_path
            )
            
            # Almacenar resultado
            self.training_results[result.id] = result
            
            logger.info(f"Fine-tuning completado: {result.id}")
            logger.info(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
            logger.info(f"Métricas finales: {final_metrics}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en fine-tuning: {e}")
            raise
    
    async def evaluate_model(
        self,
        model_id: str,
        test_dataset: Dataset,
        task_type: TaskType
    ) -> ModelEvaluation:
        """
        Evaluar modelo
        
        Args:
            model_id: ID del modelo
            test_dataset: Dataset de prueba
            task_type: Tipo de tarea
            
        Returns:
            Evaluación del modelo
        """
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            model_info = self.loaded_models[model_id]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            logger.info(f"Evaluando modelo: {model_id}")
            
            # Configurar trainer para evaluación
            training_args = TrainingArguments(
                output_dir=f"{self.models_directory}/eval_{model_id}",
                per_device_eval_batch_size=self.config["default_batch_size"],
                dataloader_num_workers=self.config["dataloader_num_workers"],
                remove_unused_columns=False
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=test_dataset,
                tokenizer=tokenizer
            )
            
            # Evaluar modelo
            eval_results = trainer.evaluate()
            
            # Generar predicciones
            predictions = []
            ground_truth = []
            confidence_scores = []
            
            model.eval()
            with torch.no_grad():
                for batch in tqdm(DataLoader(test_dataset, batch_size=self.config["default_batch_size"])):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    
                    if task_type == TaskType.CLASSIFICATION:
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = F.softmax(logits, dim=-1)
                        preds = torch.argmax(logits, dim=-1)
                        
                        predictions.extend(preds.cpu().numpy().tolist())
                        ground_truth.extend(batch['labels'].numpy().tolist())
                        confidence_scores.extend(torch.max(probs, dim=-1)[0].cpu().numpy().tolist())
                    
                    elif task_type == TaskType.GENERATION:
                        generated = model.generate(
                            **inputs,
                            max_length=self.config["default_max_length"],
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        
                        for i, gen in enumerate(generated):
                            pred_text = tokenizer.decode(gen, skip_special_tokens=True)
                            predictions.append(pred_text)
                            
                            if 'labels' in batch:
                                true_text = tokenizer.decode(batch['labels'][i], skip_special_tokens=True)
                                ground_truth.append(true_text)
                            
                            # Calcular confianza (simplificado)
                            confidence_scores.append(0.8)  # Placeholder
            
            # Calcular métricas adicionales
            additional_metrics = {}
            if task_type == TaskType.CLASSIFICATION:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                additional_metrics = {
                    "accuracy": accuracy_score(ground_truth, predictions),
                    "precision": precision_score(ground_truth, predictions, average='weighted'),
                    "recall": recall_score(ground_truth, predictions, average='weighted'),
                    "f1_score": f1_score(ground_truth, predictions, average='weighted')
                }
            
            # Combinar métricas
            all_metrics = {**eval_results, **additional_metrics}
            
            # Crear evaluación
            evaluation = ModelEvaluation(
                id=f"eval_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_id=model_id,
                task_type=task_type,
                metrics=all_metrics,
                predictions=predictions,
                ground_truth=ground_truth,
                confidence_scores=confidence_scores
            )
            
            # Almacenar evaluación
            self.model_evaluations[evaluation.id] = evaluation
            
            logger.info(f"Evaluación completada: {evaluation.id}")
            logger.info(f"Métricas: {all_metrics}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluando modelo: {e}")
            raise
    
    async def generate_text(
        self,
        model_id: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generar texto
        
        Args:
            model_id: ID del modelo
            prompt: Texto de entrada
            max_length: Longitud máxima
            temperature: Temperatura de muestreo
            top_p: Top-p sampling
            num_return_sequences: Número de secuencias a generar
            
        Returns:
            Lista de textos generados
        """
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            model_info = self.loaded_models[model_id]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Tokenizar entrada
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generar texto
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decodificar salidas
            generated_texts = []
            for output in outputs:
                text = tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(text)
            
            logger.info(f"Texto generado con modelo: {model_id}")
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error generando texto: {e}")
            raise
    
    async def create_prompt_template(
        self,
        name: str,
        template: str,
        task_type: TaskType,
        variables: Optional[List[str]] = None
    ) -> PromptTemplate:
        """
        Crear plantilla de prompt
        
        Args:
            name: Nombre de la plantilla
            template: Plantilla de prompt
            task_type: Tipo de tarea
            variables: Variables en la plantilla
            
        Returns:
            Plantilla de prompt
        """
        try:
            # Extraer variables de la plantilla si no se proporcionan
            if variables is None:
                import re
                variables = re.findall(r'\{(\w+)\}', template)
            
            prompt_template = PromptTemplate(
                id=f"template_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=name,
                template=template,
                variables=variables,
                task_type=task_type
            )
            
            # Almacenar plantilla
            self.prompt_templates[prompt_template.id] = prompt_template
            
            logger.info(f"Plantilla de prompt creada: {prompt_template.id}")
            return prompt_template
            
        except Exception as e:
            logger.error(f"Error creando plantilla de prompt: {e}")
            raise
    
    async def optimize_prompt(
        self,
        template_id: str,
        test_data: List[Dict[str, Any]],
        model_id: str,
        optimization_rounds: int = 5
    ) -> Dict[str, Any]:
        """
        Optimizar prompt
        
        Args:
            template_id: ID de la plantilla
            test_data: Datos de prueba
            model_id: ID del modelo
            optimization_rounds: Número de rondas de optimización
            
        Returns:
            Resultados de optimización
        """
        try:
            if template_id not in self.prompt_templates:
                raise ValueError(f"Plantilla {template_id} no encontrada")
            
            if model_id not in self.loaded_models:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            template = self.prompt_templates[template_id]
            model_info = self.loaded_models[model_id]
            
            logger.info(f"Optimizando prompt: {template_id}")
            
            # Implementar optimización de prompt (simplificado)
            optimization_results = {
                "original_template": template.template,
                "optimization_rounds": optimization_rounds,
                "improvements": [],
                "final_template": template.template,
                "effectiveness_scores": []
            }
            
            # Simular optimización (en un sistema real, usarías técnicas como P-tuning)
            for round_num in range(optimization_rounds):
                # Evaluar plantilla actual
                current_score = await self._evaluate_prompt_effectiveness(
                    template.template, test_data, model_id
                )
                
                optimization_results["effectiveness_scores"].append(current_score)
                
                # Simular mejora de plantilla
                if round_num < optimization_rounds - 1:
                    # En un sistema real, aquí modificarías la plantilla
                    improvement = f"Ronda {round_num + 1}: Score mejorado a {current_score:.3f}"
                    optimization_results["improvements"].append(improvement)
            
            # Actualizar score de efectividad
            template.effectiveness_score = optimization_results["effectiveness_scores"][-1]
            
            logger.info(f"Optimización de prompt completada: {template_id}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizando prompt: {e}")
            raise
    
    async def _evaluate_prompt_effectiveness(
        self,
        template: str,
        test_data: List[Dict[str, Any]],
        model_id: str
    ) -> float:
        """Evaluar efectividad de prompt"""
        try:
            # Implementación simplificada
            # En un sistema real, evaluarías la calidad de las respuestas generadas
            
            model_info = self.loaded_models[model_id]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            scores = []
            for data_point in test_data[:5]:  # Evaluar solo los primeros 5 para eficiencia
                # Formatear prompt
                prompt = template.format(**data_point)
                
                # Generar respuesta
                generated = await self.generate_text(model_id, prompt, max_length=50)
                
                # Calcular score (simplificado)
                score = len(generated[0].split()) / 50  # Score basado en longitud
                scores.append(score)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error evaluando efectividad de prompt: {e}")
            return 0.0
    
    async def create_diffusion_pipeline(
        self,
        model_name: str = "stable-diffusion",
        custom_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Crear pipeline de difusión
        
        Args:
            model_name: Nombre del modelo
            custom_config: Configuración personalizada
            
        Returns:
            ID del pipeline
        """
        try:
            if not self.enable_diffusers:
                raise ValueError("Diffusers no disponible")
            
            pipeline_id = f"diffusion_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Creando pipeline de difusión: {model_name}")
            
            # Cargar pipeline según el modelo
            if model_name == "stable-diffusion":
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.diffusion_models["stable-diffusion"],
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
            elif model_name == "stable-diffusion-xl":
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.diffusion_models["stable-diffusion-xl"],
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
            else:
                raise ValueError(f"Modelo de difusión no soportado: {model_name}")
            
            # Mover al dispositivo
            pipeline = pipeline.to(self.device)
            
            # Configurar optimizaciones
            if custom_config:
                if custom_config.get("enable_memory_efficient_attention", True):
                    pipeline.enable_attention_slicing()
                if custom_config.get("enable_cpu_offload", False):
                    pipeline.enable_sequential_cpu_offload()
            
            # Almacenar pipeline
            self.loaded_models[pipeline_id] = {
                "pipeline": pipeline,
                "model_name": model_name,
                "model_type": ModelType.DIFFUSION,
                "task_type": TaskType.GENERATION,
                "config": custom_config or {}
            }
            
            logger.info(f"Pipeline de difusión creado: {pipeline_id}")
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Error creando pipeline de difusión: {e}")
            raise
    
    async def generate_image(
        self,
        pipeline_id: str,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512
    ) -> List[Any]:
        """
        Generar imagen con modelo de difusión
        
        Args:
            pipeline_id: ID del pipeline
            prompt: Prompt de texto
            negative_prompt: Prompt negativo
            num_images: Número de imágenes
            guidance_scale: Escala de guía
            num_inference_steps: Pasos de inferencia
            height: Altura de imagen
            width: Ancho de imagen
            
        Returns:
            Lista de imágenes generadas
        """
        try:
            if pipeline_id not in self.loaded_models:
                raise ValueError(f"Pipeline {pipeline_id} no encontrado")
            
            model_info = self.loaded_models[pipeline_id]
            pipeline = model_info["pipeline"]
            
            logger.info(f"Generando imagen con pipeline: {pipeline_id}")
            
            # Generar imagen
            with torch.autocast(self.device.type):
                images = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width
                ).images
            
            logger.info(f"Imagen generada exitosamente: {len(images)} imágenes")
            return images
            
        except Exception as e:
            logger.error(f"Error generando imagen: {e}")
            raise
    
    async def create_gradio_interface(self) -> str:
        """
        Crear interfaz Gradio
        
        Returns:
            Ruta del archivo de interfaz
        """
        try:
            if not self.enable_gradio:
                raise ValueError("Gradio no disponible")
            
            logger.info("Creando interfaz Gradio")
            
            def generate_text_interface(prompt, model_id, max_length, temperature):
                try:
                    if model_id not in self.loaded_models:
                        return "Error: Modelo no encontrado"
                    
                    generated = await self.generate_text(
                        model_id, prompt, max_length, temperature
                    )
                    return generated[0] if generated else "Error en la generación"
                except Exception as e:
                    return f"Error: {str(e)}"
            
            def generate_image_interface(prompt, pipeline_id, guidance_scale, num_steps):
                try:
                    if pipeline_id not in self.loaded_models:
                        return None
                    
                    images = await self.generate_image(
                        pipeline_id, prompt, guidance_scale=guidance_scale,
                        num_inference_steps=num_steps
                    )
                    return images[0] if images else None
                except Exception as e:
                    return None
            
            # Crear interfaz
            with gr.Blocks(title="Advanced LLM Analyzer") as interface:
                gr.Markdown("# Advanced LLM Analyzer")
                
                with gr.Tabs():
                    # Tab de generación de texto
                    with gr.Tab("Generación de Texto"):
                        with gr.Row():
                            with gr.Column():
                                text_prompt = gr.Textbox(
                                    label="Prompt",
                                    placeholder="Escribe tu prompt aquí...",
                                    lines=3
                                )
                                model_selector = gr.Dropdown(
                                    label="Modelo",
                                    choices=list(self.loaded_models.keys()),
                                    value=list(self.loaded_models.keys())[0] if self.loaded_models else None
                                )
                                max_length = gr.Slider(
                                    label="Longitud máxima",
                                    minimum=10,
                                    maximum=500,
                                    value=100,
                                    step=10
                                )
                                temperature = gr.Slider(
                                    label="Temperatura",
                                    minimum=0.1,
                                    maximum=2.0,
                                    value=0.7,
                                    step=0.1
                                )
                                generate_btn = gr.Button("Generar Texto")
                            
                            with gr.Column():
                                text_output = gr.Textbox(
                                    label="Texto Generado",
                                    lines=10
                                )
                        
                        generate_btn.click(
                            generate_text_interface,
                            inputs=[text_prompt, model_selector, max_length, temperature],
                            outputs=text_output
                        )
                    
                    # Tab de generación de imágenes
                    with gr.Tab("Generación de Imágenes"):
                        with gr.Row():
                            with gr.Column():
                                image_prompt = gr.Textbox(
                                    label="Prompt de Imagen",
                                    placeholder="Describe la imagen que quieres generar...",
                                    lines=3
                                )
                                pipeline_selector = gr.Dropdown(
                                    label="Pipeline",
                                    choices=[k for k, v in self.loaded_models.items() 
                                           if v.get("model_type") == ModelType.DIFFUSION],
                                    value=None
                                )
                                guidance_scale = gr.Slider(
                                    label="Escala de Guía",
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=7.5,
                                    step=0.5
                                )
                                num_steps = gr.Slider(
                                    label="Pasos de Inferencia",
                                    minimum=10,
                                    maximum=100,
                                    value=50,
                                    step=5
                                )
                                generate_img_btn = gr.Button("Generar Imagen")
                            
                            with gr.Column():
                                image_output = gr.Image(
                                    label="Imagen Generada"
                                )
                        
                        generate_img_btn.click(
                            generate_image_interface,
                            inputs=[image_prompt, pipeline_selector, guidance_scale, num_steps],
                            outputs=image_output
                        )
            
            # Guardar interfaz
            interface_path = f"{self.models_directory}/gradio_interface.py"
            interface.save(interface_path)
            
            logger.info(f"Interfaz Gradio creada: {interface_path}")
            return interface_path
            
        except Exception as e:
            logger.error(f"Error creando interfaz Gradio: {e}")
            raise
    
    async def get_llm_analysis_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis LLM"""
        try:
            return {
                "total_loaded_models": len(self.loaded_models),
                "total_model_configs": len(self.model_configs),
                "total_training_results": len(self.training_results),
                "total_evaluations": len(self.model_evaluations),
                "total_prompt_templates": len(self.prompt_templates),
                "model_types": {
                    model_type.value: len([m for m in self.loaded_models.values() if m.get("model_type") == model_type])
                    for model_type in ModelType
                },
                "task_types": {
                    task_type.value: len([m for m in self.loaded_models.values() if m.get("task_type") == task_type])
                    for task_type in TaskType
                },
                "capabilities": {
                    "torch": self.enable_torch,
                    "transformers": self.enable_transformers,
                    "diffusers": self.enable_diffusers,
                    "gradio": self.enable_gradio
                },
                "device": str(self.device),
                "last_activity": max([
                    max([r.created_at for r in self.training_results.values()]) if self.training_results else datetime.min,
                    max([e.created_at for e in self.model_evaluations.values()]) if self.model_evaluations else datetime.min
                ]).isoformat() if any([self.training_results, self.model_evaluations]) else None
            }
        except Exception as e:
            logger.error(f"Error obteniendo resumen LLM: {e}")
            return {}
    
    async def export_llm_data(self, filepath: str = None) -> str:
        """Exportar datos LLM"""
        try:
            if filepath is None:
                filepath = f"exports/llm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            export_data = {
                "model_configs": {
                    config_id: {
                        "model_name": config.model_name,
                        "model_type": config.model_type.value,
                        "task_type": config.task_type.value,
                        "max_length": config.max_length,
                        "batch_size": config.batch_size,
                        "learning_rate": config.learning_rate,
                        "num_epochs": config.num_epochs,
                        "optimization_type": config.optimization_type.value,
                        "created_at": config.created_at.isoformat()
                    }
                    for config_id, config in self.model_configs.items()
                },
                "training_results": {
                    result_id: {
                        "model_config": {
                            "model_name": result.model_config.model_name,
                            "model_type": result.model_config.model_type.value,
                            "task_type": result.model_config.task_type.value
                        },
                        "training_history": result.training_history,
                        "final_metrics": result.final_metrics,
                        "training_time": result.training_time,
                        "best_epoch": result.best_epoch,
                        "model_path": result.model_path,
                        "created_at": result.created_at.isoformat()
                    }
                    for result_id, result in self.training_results.items()
                },
                "model_evaluations": {
                    eval_id: {
                        "model_id": evaluation.model_id,
                        "task_type": evaluation.task_type.value,
                        "metrics": evaluation.metrics,
                        "confidence_scores": evaluation.confidence_scores,
                        "created_at": evaluation.created_at.isoformat()
                    }
                    for eval_id, evaluation in self.model_evaluations.items()
                },
                "prompt_templates": {
                    template_id: {
                        "name": template.name,
                        "template": template.template,
                        "variables": template.variables,
                        "task_type": template.task_type.value,
                        "effectiveness_score": template.effectiveness_score,
                        "usage_count": template.usage_count,
                        "created_at": template.created_at.isoformat()
                    }
                    for template_id, template in self.prompt_templates.items()
                },
                "summary": await self.get_llm_analysis_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Datos LLM exportados a {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exportando datos LLM: {e}")
            raise
























