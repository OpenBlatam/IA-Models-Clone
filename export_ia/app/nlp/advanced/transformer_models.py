"""
Transformer Models Manager - Gestor de modelos transformer avanzados
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, Pipeline
)

logger = logging.getLogger(__name__)


class TransformerModelManager:
    """Gestor de modelos transformer avanzados."""
    
    def __init__(self):
        self._initialized = False
        self.models = {}
        self.pipelines = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Modelos predefinidos
        self.model_configs = {
            "sentiment": {
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "task": "sentiment-analysis"
            },
            "emotion": {
                "model_name": "j-hartmann/emotion-english-distilroberta-base",
                "task": "text-classification"
            },
            "ner": {
                "model_name": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "task": "ner"
            },
            "summarization": {
                "model_name": "facebook/bart-large-cnn",
                "task": "summarization"
            },
            "translation": {
                "model_name": "Helsinki-NLP/opus-mt-en-es",
                "task": "translation"
            },
            "text_generation": {
                "model_name": "gpt2",
                "task": "text-generation"
            },
            "question_answering": {
                "model_name": "distilbert-base-cased-distilled-squad",
                "task": "question-answering"
            },
            "text_classification": {
                "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
                "task": "text-classification"
            }
        }
    
    async def initialize(self):
        """Inicializar el gestor de modelos."""
        if not self._initialized:
            try:
                logger.info(f"Inicializando TransformerModelManager en dispositivo: {self.device}")
                
                # Cargar modelos principales
                await self._load_essential_models()
                
                self._initialized = True
                logger.info("TransformerModelManager inicializado exitosamente")
                
            except Exception as e:
                logger.error(f"Error al inicializar TransformerModelManager: {e}")
                raise
    
    async def shutdown(self):
        """Cerrar el gestor de modelos."""
        if self._initialized:
            try:
                # Limpiar modelos de memoria
                for model_name in list(self.models.keys()):
                    del self.models[model_name]
                
                for pipeline_name in list(self.pipelines.keys()):
                    del self.pipelines[pipeline_name]
                
                # Limpiar cache de CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self._initialized = False
                logger.info("TransformerModelManager cerrado")
                
            except Exception as e:
                logger.error(f"Error al cerrar TransformerModelManager: {e}")
    
    async def _load_essential_models(self):
        """Cargar modelos esenciales."""
        essential_models = ["sentiment", "ner", "summarization"]
        
        for model_type in essential_models:
            try:
                await self.load_model(model_type)
                logger.info(f"Modelo {model_type} cargado exitosamente")
            except Exception as e:
                logger.warning(f"No se pudo cargar el modelo {model_type}: {e}")
    
    async def load_model(self, model_type: str, model_name: Optional[str] = None) -> bool:
        """Cargar un modelo específico."""
        try:
            if model_type not in self.model_configs:
                raise ValueError(f"Tipo de modelo no soportado: {model_type}")
            
            config = self.model_configs[model_type]
            model_name = model_name or config["model_name"]
            task = config["task"]
            
            logger.info(f"Cargando modelo {model_type}: {model_name}")
            
            # Crear pipeline
            pipeline_obj = pipeline(
                task,
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            self.pipelines[model_type] = pipeline_obj
            self.models[model_type] = {
                "name": model_name,
                "task": task,
                "loaded_at": datetime.now(),
                "device": self.device
            }
            
            logger.info(f"Modelo {model_type} cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar modelo {model_type}: {e}")
            return False
    
    async def unload_model(self, model_type: str) -> bool:
        """Descargar un modelo específico."""
        try:
            if model_type in self.pipelines:
                del self.pipelines[model_type]
            
            if model_type in self.models:
                del self.models[model_type]
            
            # Limpiar cache de CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Modelo {model_type} descargado")
            return True
            
        except Exception as e:
            logger.error(f"Error al descargar modelo {model_type}: {e}")
            return False
    
    async def analyze_sentiment_advanced(self, text: str) -> Dict[str, Any]:
        """Análisis de sentimiento avanzado."""
        if "sentiment" not in self.pipelines:
            await self.load_model("sentiment")
        
        try:
            pipeline_obj = self.pipelines["sentiment"]
            results = pipeline_obj(text)
            
            # Procesar resultados
            sentiment_scores = {}
            for result in results:
                label = result["label"].lower()
                score = result["score"]
                sentiment_scores[label] = score
            
            # Determinar sentimiento principal
            main_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[main_sentiment]
            
            return {
                "text": text,
                "sentiment": main_sentiment,
                "confidence": confidence,
                "scores": sentiment_scores,
                "model_used": self.models["sentiment"]["name"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de sentimiento avanzado: {e}")
            raise
    
    async def extract_entities_advanced(self, text: str) -> Dict[str, Any]:
        """Extracción de entidades avanzada."""
        if "ner" not in self.pipelines:
            await self.load_model("ner")
        
        try:
            pipeline_obj = self.pipelines["ner"]
            results = pipeline_obj(text)
            
            # Procesar entidades
            entities = []
            current_entity = None
            
            for result in results:
                word = result["word"]
                label = result["entity"]
                score = result["score"]
                
                if label.startswith("B-"):
                    # Inicio de nueva entidad
                    if current_entity:
                        entities.append(current_entity)
                    
                    current_entity = {
                        "text": word,
                        "label": label[2:],  # Remover prefijo B-
                        "start": result.get("start", 0),
                        "end": result.get("end", len(word)),
                        "confidence": score
                    }
                elif label.startswith("I-") and current_entity:
                    # Continuación de entidad
                    current_entity["text"] += " " + word
                    current_entity["end"] = result.get("end", current_entity["end"] + len(word) + 1)
                    current_entity["confidence"] = (current_entity["confidence"] + score) / 2
                else:
                    # Fin de entidad
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            # Agregar última entidad
            if current_entity:
                entities.append(current_entity)
            
            return {
                "text": text,
                "entities": entities,
                "entity_count": len(entities),
                "model_used": self.models["ner"]["name"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en extracción de entidades avanzada: {e}")
            raise
    
    async def summarize_advanced(self, text: str, max_length: int = 150, min_length: int = 30) -> Dict[str, Any]:
        """Resumen avanzado con modelos transformer."""
        if "summarization" not in self.pipelines:
            await self.load_model("summarization")
        
        try:
            pipeline_obj = self.pipelines["summarization"]
            
            # Configurar parámetros
            summary_params = {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False,
                "early_stopping": True
            }
            
            result = pipeline_obj(text, **summary_params)
            summary = result[0]["summary_text"]
            
            # Calcular métricas
            original_words = len(text.split())
            summary_words = len(summary.split())
            compression_ratio = summary_words / original_words if original_words > 0 else 0
            
            return {
                "original_text": text,
                "summary": summary,
                "compression_ratio": compression_ratio,
                "original_word_count": original_words,
                "summary_word_count": summary_words,
                "model_used": self.models["summarization"]["name"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en resumen avanzado: {e}")
            raise
    
    async def generate_text_advanced(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """Generación de texto avanzada."""
        if "text_generation" not in self.pipelines:
            await self.load_model("text_generation")
        
        try:
            pipeline_obj = self.pipelines["text_generation"]
            
            # Configurar parámetros
            generation_params = {
                "max_length": max_length,
                "temperature": temperature,
                "do_sample": True,
                "pad_token_id": pipeline_obj.tokenizer.eos_token_id
            }
            
            result = pipeline_obj(prompt, **generation_params)
            generated_text = result[0]["generated_text"]
            
            # Remover prompt del texto generado
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return {
                "prompt": prompt,
                "generated_text": generated_text,
                "full_text": result[0]["generated_text"],
                "model_used": self.models["text_generation"]["name"],
                "parameters": generation_params,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en generación de texto avanzada: {e}")
            raise
    
    async def classify_text_advanced(self, text: str) -> Dict[str, Any]:
        """Clasificación de texto avanzada."""
        if "text_classification" not in self.pipelines:
            await self.load_model("text_classification")
        
        try:
            pipeline_obj = self.pipelines["text_classification"]
            results = pipeline_obj(text)
            
            # Procesar resultados
            classifications = []
            for result in results:
                classifications.append({
                    "label": result["label"],
                    "score": result["score"]
                })
            
            # Obtener clasificación principal
            main_classification = max(classifications, key=lambda x: x["score"])
            
            return {
                "text": text,
                "predicted_class": main_classification["label"],
                "confidence": main_classification["score"],
                "all_classes": classifications,
                "model_used": self.models["text_classification"]["name"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en clasificación de texto avanzada: {e}")
            raise
    
    async def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Responder preguntas basadas en contexto."""
        if "question_answering" not in self.pipelines:
            await self.load_model("question_answering")
        
        try:
            pipeline_obj = self.pipelines["question_answering"]
            result = pipeline_obj(question=question, context=context)
            
            return {
                "question": question,
                "context": context,
                "answer": result["answer"],
                "confidence": result["score"],
                "start": result["start"],
                "end": result["end"],
                "model_used": self.models["question_answering"]["name"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en respuesta de preguntas: {e}")
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Obtener información de modelos cargados."""
        return {
            "loaded_models": list(self.models.keys()),
            "model_details": self.models,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "memory_usage": self._get_memory_usage(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Obtener uso de memoria."""
        try:
            if torch.cuda.is_available():
                return {
                    "cuda_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "cuda_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
                    "cuda_max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
                }
            else:
                return {"cuda_available": False}
        except Exception as e:
            logger.error(f"Error al obtener uso de memoria: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del gestor de modelos."""
        try:
            loaded_models = list(self.models.keys())
            available_pipelines = list(self.pipelines.keys())
            
            return {
                "status": "healthy" if self._initialized else "unhealthy",
                "initialized": self._initialized,
                "loaded_models": loaded_models,
                "available_pipelines": available_pipelines,
                "device": self.device,
                "cuda_available": torch.cuda.is_available(),
                "memory_usage": self._get_memory_usage(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




