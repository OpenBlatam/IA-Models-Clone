"""
LLM Processor System
====================

Sistema de procesamiento de lenguaje natural usando Transformers y LLMs.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        pipeline,
        Pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None
    pipeline = None
    Pipeline = None

logger = logging.getLogger(__name__)


class LLMTask(Enum):
    """Tipo de tarea LLM."""
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    COMMAND_PARSING = "command_parsing"
    INTENT_CLASSIFICATION = "intent_classification"


@dataclass
class LLMConfig:
    """Configuración de LLM."""
    model_id: str
    model_name: str
    task: LLMTask
    device: str = "auto"  # "auto", "cpu", "cuda"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    use_fp16: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMRequest:
    """Solicitud a LLM."""
    request_id: str
    prompt: str
    task: LLMTask
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LLMResponse:
    """Respuesta de LLM."""
    response_id: str
    request_id: str
    text: str
    confidence: Optional[float] = None
    tokens: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CommandIntent:
    """Intención de comando."""
    intent: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


class LLMProcessor:
    """
    Procesador de LLM.
    
    Maneja modelos de lenguaje para procesamiento de comandos y texto.
    """
    
    def __init__(self):
        """Inicializar procesador."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available. LLM features will be limited.")
        
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.configs: Dict[str, LLMConfig] = {}
        self.requests: List[LLMRequest] = []
        self.responses: List[LLMResponse] = []
        self.max_history = 10000
    
    def load_model(
        self,
        model_name: str,
        task: LLMTask,
        config: Optional[LLMConfig] = None
    ) -> str:
        """
        Cargar modelo.
        
        Args:
            model_name: Nombre del modelo (HuggingFace)
            task: Tipo de tarea
            config: Configuración opcional
            
        Returns:
            ID del modelo
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for LLM processing")
        
        model_id = str(uuid.uuid4())
        
        try:
            # Determinar tipo de pipeline
            if task == LLMTask.TEXT_GENERATION:
                pipeline_type = "text-generation"
            elif task == LLMTask.TEXT_CLASSIFICATION:
                pipeline_type = "text-classification"
            elif task == LLMTask.QUESTION_ANSWERING:
                pipeline_type = "question-answering"
            elif task == LLMTask.SUMMARIZATION:
                pipeline_type = "summarization"
            elif task == LLMTask.TRANSLATION:
                pipeline_type = "translation"
            else:
                pipeline_type = "text-generation"
            
            # Cargar pipeline
            device = config.device if config else "auto"
            if device == "auto":
                import torch
                device = 0 if torch.cuda.is_available() else -1
            
            pipe = pipeline(
                pipeline_type,
                model=model_name,
                device=device,
                model_kwargs={
                    "torch_dtype": "float16" if (config and config.use_fp16) else "float32"
                } if device != -1 else {}
            )
            
            self.pipelines[model_id] = pipe
            
            # Cargar tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizers[model_id] = tokenizer
            
            if config is None:
                config = LLMConfig(
                    model_id=model_id,
                    model_name=model_name,
                    task=task
                )
            
            self.configs[model_id] = config
            
            logger.info(f"Loaded model {model_name} for task {task.value}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
        
        return model_id
    
    def generate_text(
        self,
        model_id: str,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> LLMResponse:
        """
        Generar texto.
        
        Args:
            model_id: ID del modelo
            prompt: Prompt de entrada
            max_length: Longitud máxima (opcional)
            temperature: Temperatura (opcional)
            top_p: Top-p sampling (opcional)
            top_k: Top-k sampling (opcional)
            
        Returns:
            Respuesta generada
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required")
        
        if model_id not in self.pipelines:
            raise ValueError(f"Model not found: {model_id}")
        
        config = self.configs[model_id]
        pipe = self.pipelines[model_id]
        
        # Parámetros
        params = {
            "max_length": max_length or config.max_length,
            "temperature": temperature if temperature is not None else config.temperature,
            "top_p": top_p if top_p is not None else config.top_p,
            "top_k": top_k if top_k is not None else config.top_k,
            "do_sample": config.do_sample,
            "num_return_sequences": 1
        }
        
        # Crear request
        request = LLMRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            task=config.task,
            parameters=params
        )
        self.requests.append(request)
        if len(self.requests) > self.max_history:
            self.requests = self.requests[-self.max_history:]
        
        # Generar
        try:
            result = pipe(prompt, **params)
            
            # Procesar resultado según tipo
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    text = result[0].get("generated_text", result[0].get("text", ""))
                else:
                    text = str(result[0])
            else:
                text = str(result)
            
            # Remover prompt del inicio si está presente
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            
            response = LLMResponse(
                response_id=str(uuid.uuid4()),
                request_id=request.request_id,
                text=text,
                metadata={"raw_result": result}
            )
            
            self.responses.append(response)
            if len(self.responses) > self.max_history:
                self.responses = self.responses[-self.max_history:]
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def classify_text(
        self,
        model_id: str,
        text: str
    ) -> LLMResponse:
        """
        Clasificar texto.
        
        Args:
            model_id: ID del modelo
            text: Texto a clasificar
            
        Returns:
            Respuesta con clasificación
        """
        if model_id not in self.pipelines:
            raise ValueError(f"Model not found: {model_id}")
        
        pipe = self.pipelines[model_id]
        
        request = LLMRequest(
            request_id=str(uuid.uuid4()),
            prompt=text,
            task=LLMTask.TEXT_CLASSIFICATION
        )
        self.requests.append(request)
        
        try:
            result = pipe(text)
            
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            label = result.get("label", "unknown")
            score = result.get("score", 0.0)
            
            response = LLMResponse(
                response_id=str(uuid.uuid4()),
                request_id=request.request_id,
                text=label,
                confidence=score,
                metadata={"raw_result": result}
            )
            
            self.responses.append(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            raise
    
    def parse_command(
        self,
        model_id: str,
        command: str
    ) -> CommandIntent:
        """
        Parsear comando de robot.
        
        Args:
            model_id: ID del modelo
            command: Comando de texto
            
        Returns:
            Intención parseada
        """
        # Usar generación para extraer intención
        prompt = f"Parse this robot command and extract intent and parameters:\nCommand: {command}\nIntent:"
        
        response = self.generate_text(
            model_id,
            prompt,
            max_length=100,
            temperature=0.3
        )
        
        # Parsear respuesta (simplificado)
        intent_text = response.text.strip().lower()
        
        # Detectar intención común
        if "move" in intent_text or "go" in intent_text:
            intent = "move"
        elif "stop" in intent_text or "halt" in intent_text:
            intent = "stop"
        elif "rotate" in intent_text or "turn" in intent_text:
            intent = "rotate"
        elif "grab" in intent_text or "pick" in intent_text:
            intent = "grab"
        elif "release" in intent_text or "drop" in intent_text:
            intent = "release"
        else:
            intent = "unknown"
        
        # Extraer parámetros básicos (simplificado)
        parameters = {}
        words = command.lower().split()
        
        # Buscar números
        import re
        numbers = re.findall(r'-?\d+\.?\d*', command)
        if numbers:
            parameters["values"] = [float(n) for n in numbers]
        
        return CommandIntent(
            intent=intent,
            confidence=0.8,  # Simplificado
            parameters=parameters
        )
    
    def answer_question(
        self,
        model_id: str,
        question: str,
        context: Optional[str] = None
    ) -> LLMResponse:
        """
        Responder pregunta.
        
        Args:
            model_id: ID del modelo
            question: Pregunta
            context: Contexto opcional
            
        Returns:
            Respuesta
        """
        if model_id not in self.pipelines:
            raise ValueError(f"Model not found: {model_id}")
        
        config = self.configs[model_id]
        
        if config.task == LLMTask.QUESTION_ANSWERING:
            pipe = self.pipelines[model_id]
            result = pipe(question=question, context=context or "")
            
            answer = result.get("answer", "")
            score = result.get("score", 0.0)
            
            response = LLMResponse(
                response_id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4()),
                text=answer,
                confidence=score
            )
        else:
            # Usar generación de texto
            prompt = f"Question: {question}\n"
            if context:
                prompt += f"Context: {context}\n"
            prompt += "Answer:"
            
            response = self.generate_text(model_id, prompt, max_length=200, temperature=0.5)
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        task_counts = {}
        for config in self.configs.values():
            task_counts[config.task.value] = task_counts.get(config.task.value, 0) + 1
        
        return {
            "total_models": len(self.models),
            "task_counts": task_counts,
            "total_requests": len(self.requests),
            "total_responses": len(self.responses)
        }


# Instancia global
_llm_processor: Optional[LLMProcessor] = None


def get_llm_processor() -> LLMProcessor:
    """Obtener instancia global del procesador LLM."""
    global _llm_processor
    if _llm_processor is None:
        _llm_processor = LLMProcessor()
    return _llm_processor




