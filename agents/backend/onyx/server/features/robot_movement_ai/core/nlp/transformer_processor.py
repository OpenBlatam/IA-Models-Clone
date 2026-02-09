"""
Advanced Transformer-based NLP Processor
========================================

Procesador avanzado de lenguaje natural usando Transformers para interpretación
de comandos y generación de respuestas.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline
)

logger = logging.getLogger(__name__)


class TransformerCommandProcessor:
    """
    Procesador de comandos usando modelos Transformer.
    
    Usa modelos pre-entrenados para interpretar comandos de lenguaje natural
    y extraer intenciones y parámetros.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        task: str = "text-classification",
        device: Optional[torch.device] = None,
        use_fp16: bool = False
    ):
        """
        Inicializar procesador.
        
        Args:
            model_name: Nombre del modelo HuggingFace
            task: Tipo de tarea (text-classification, token-classification, etc.)
            device: Dispositivo (CPU/GPU)
            use_fp16: Usar precisión mixta
        """
        self.model_name = model_name
        self.task = task
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
        # Cargar tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Cargar pipeline
        self.pipeline = self._load_pipeline()
        
        logger.info(f"Transformer processor initialized with model: {model_name}")
    
    def _load_pipeline(self) -> Pipeline:
        """Cargar pipeline de Transformers."""
        model_kwargs = {}
        if self.use_fp16:
            model_kwargs['torch_dtype'] = torch.float16
        
        device_id = 0 if self.device.type == 'cuda' else -1
        
        return pipeline(
            self.task,
            model=self.model_name,
            tokenizer=self.tokenizer,
            device=device_id,
            model_kwargs=model_kwargs
        )
    
    def classify_intent(
        self,
        command: str,
        candidate_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Clasificar intención del comando.
        
        Args:
            command: Comando de texto
            candidate_labels: Etiquetas candidatas (para zero-shot)
            
        Returns:
            Diccionario con intención y confianza
        """
        if candidate_labels:
            # Zero-shot classification
            result = self.pipeline(command, candidate_labels=candidate_labels)
        else:
            # Standard classification
            result = self.pipeline(command)
        
        if isinstance(result, list):
            result = result[0]
        
        return {
            'intent': result.get('label', 'unknown'),
            'confidence': result.get('score', 0.0),
            'raw_result': result
        }
    
    def extract_entities(
        self,
        command: str
    ) -> List[Dict[str, Any]]:
        """
        Extraer entidades del comando.
        
        Args:
            command: Comando de texto
            
        Returns:
            Lista de entidades extraídas
        """
        # Usar NER si está disponible
        try:
            ner_pipeline = pipeline(
                "ner",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == 'cuda' else -1
            )
            entities = ner_pipeline(command)
            return entities
        except Exception as e:
            logger.warning(f"NER not available: {e}")
            # Fallback: extraer números
            import re
            numbers = re.findall(r'-?\d+\.?\d*', command)
            entities = [
                {
                    'entity': 'NUMBER',
                    'value': float(n),
                    'start': command.find(n),
                    'end': command.find(n) + len(n)
                }
                for n in numbers
            ]
            return entities
    
    def parse_command(
        self,
        command: str,
        intent_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Parsear comando completo.
        
        Args:
            command: Comando de texto
            intent_labels: Etiquetas de intención
            
        Returns:
            Diccionario con intención, entidades y parámetros
        """
        # Clasificar intención
        intent_result = self.classify_intent(command, intent_labels)
        
        # Extraer entidades
        entities = self.extract_entities(command)
        
        # Extraer parámetros numéricos
        parameters = self._extract_parameters(command, entities)
        
        return {
            'intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'entities': entities,
            'parameters': parameters,
            'raw_command': command
        }
    
    def _extract_parameters(
        self,
        command: str,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Extraer parámetros numéricos."""
        parameters = {}
        
        # Buscar coordenadas en formato (x, y, z)
        import re
        coord_pattern = r'\(?\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?'
        match = re.search(coord_pattern, command)
        if match:
            parameters['x'] = float(match.group(1))
            parameters['y'] = float(match.group(2))
            parameters['z'] = float(match.group(3))
        
        # Extraer números individuales
        numbers = [float(e['value']) for e in entities if e.get('entity') == 'NUMBER']
        if numbers and 'x' not in parameters:
            for i, num in enumerate(numbers[:3]):
                parameters[['x', 'y', 'z'][i]] = num
        
        return parameters


class TransformerChatGenerator:
    """
    Generador de respuestas conversacionales usando Transformers.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[torch.device] = None,
        use_fp16: bool = False,
        max_length: int = 100
    ):
        """
        Inicializar generador.
        
        Args:
            model_name: Nombre del modelo HuggingFace
            device: Dispositivo (CPU/GPU)
            use_fp16: Usar precisión mixta
            max_length: Longitud máxima de generación
        """
        self.model_name = model_name
        self.max_length = max_length
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
        # Cargar pipeline
        model_kwargs = {}
        if self.use_fp16:
            model_kwargs['torch_dtype'] = torch.float16
        
        device_id = 0 if self.device.type == 'cuda' else -1
        
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            device=device_id,
            model_kwargs=model_kwargs
        )
        
        logger.info(f"Chat generator initialized with model: {model_name}")
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generar respuesta conversacional.
        
        Args:
            prompt: Prompt del usuario
            context: Contexto adicional
            temperature: Temperatura para sampling
            top_p: Top-p sampling
            top_k: Top-k sampling
            
        Returns:
            Respuesta generada
        """
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {context}\n\nUser: {prompt}\n\nAssistant:"
        
        result = self.pipeline(
            full_prompt,
            max_length=self.max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=1
        )
        
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '')
            # Remover prompt del inicio
            if generated_text.startswith(full_prompt):
                generated_text = generated_text[len(full_prompt):].strip()
            return generated_text
        
        return "I'm sorry, I couldn't generate a response."


class TransformerEmbedder:
    """
    Generador de embeddings usando Transformers.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[torch.device] = None
    ):
        """
        Inicializar embedder.
        
        Args:
            model_name: Nombre del modelo
            device: Dispositivo (CPU/GPU)
        """
        self.model_name = model_name
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=str(self.device))
            logger.info(f"Embedder initialized with model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using transformers fallback")
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Generar embeddings.
        
        Args:
            texts: Texto(s) a codificar
            normalize: Normalizar embeddings
            
        Returns:
            Tensor de embeddings
        """
        if hasattr(self, 'model') and hasattr(self.model, 'encode'):
            # SentenceTransformers
            embeddings = self.model.encode(
                texts if isinstance(texts, list) else [texts],
                convert_to_tensor=True,
                normalize_embeddings=normalize
            )
            return embeddings
        else:
            # Fallback usando transformers
            if isinstance(texts, str):
                texts = [texts]
            
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Usar [CLS] token o mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings













