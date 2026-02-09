"""
Code Explanation Model - Implementación modular principal
=========================================================

Modelo principal que integra todos los módulos modulares:
- ModelLoader: Carga de modelo y tokenizer
- InputValidator: Validación de inputs
- PromptBuilder: Construcción de prompts
- ExplanationCache: Gestión de caché
- BatchProcessor: Procesamiento en batch
- ModelStats: Estadísticas
"""

import logging
from typing import Dict, Any, Optional, List
import torch
from transformers import GenerationConfig

from ..base import BaseModel
from .model_loader import ModelLoader
from .validator import InputValidator
from .prompt_builder import PromptBuilder
from .cache import ExplanationCache
from .batch_processor import BatchProcessor
from .stats import ModelStats

logger = logging.getLogger(__name__)


class CodeExplanationModel(BaseModel):
    """
    Modelo modular para explicar código usando modelos seq2seq.
    
    Utiliza una arquitectura modular con componentes separados para:
    - Carga de modelos
    - Validación
    - Construcción de prompts
    - Gestión de caché
    - Procesamiento en batch
    - Estadísticas
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Inicializar modelo de explicación de código.
        
        Args:
            config: Diccionario de configuración con:
                - model_name: Nombre del modelo (default: "t5-small")
                - max_length: Longitud máxima de entrada (default: 512)
                - max_target_length: Longitud máxima de salida (default: 128)
                - enable_cache: Habilitar caché (default: True)
                - cache_ttl: TTL del caché en segundos (default: 3600)
                - cache_instance: Instancia de caché externa (opcional)
                - prompt_template: Template del prompt (default: "Explain this code: {code}")
        
        Raises:
            ValueError: Si los parámetros de configuración son inválidos
        """
        super().__init__(config)
        
        # Validar configuración
        InputValidator.validate_config(config)
        
        # Configuración
        self.model_name: str = config.get("model_name", "t5-small").strip()
        self.max_length: int = config.get("max_length", 512)
        self.max_target_length: int = config.get("max_target_length", 128)
        
        # Inicializar componentes modulares
        self.model_loader = ModelLoader(
            model_name=self.model_name,
            device=self.device,
            max_length=self.max_length,
            max_target_length=self.max_target_length
        )
        
        self.validator = InputValidator()
        
        default_template = config.get(
            "prompt_template",
            PromptBuilder.DEFAULT_TEMPLATE
        )
        self.prompt_builder = PromptBuilder(default_template=default_template)
        
        self.stats = ModelStats()
        
        self.cache = ExplanationCache(
            enabled=config.get("enable_cache", True),
            ttl=float(config.get("cache_ttl", 3600.0)),
            cache_instance=config.get("cache_instance"),
            stats=self.stats  # Integrar stats con cache
        )
        
        # Batch processor se inicializa después de cargar el modelo
        self.batch_processor: Optional[BatchProcessor] = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Asegurar que el modelo esté inicializado."""
        if not self._initialized:
            self._initialize()
    
    def _initialize(self) -> None:
        """Inicializar modelo y componentes dependientes."""
        if self._initialized:
            return
        
        try:
            # Cargar modelo y tokenizer
            model, tokenizer = self.model_loader.load()
            
            # Inicializar batch processor
            self.batch_processor = BatchProcessor(
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                max_length=self.max_length,
                max_target_length=self.max_target_length,
                cache=self.cache
            )
            
            self._initialized = True
            logger.info("CodeExplanationModel fully initialized")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model: {e}") from e
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            input_ids: Tensor con IDs de tokens de entrada.
            attention_mask: Máscara de atención opcional.
            **kwargs: Argumentos adicionales para el modelo.
        
        Returns:
            Tensor con las salidas del modelo.
        
        Raises:
            RuntimeError: Si el modelo no está inicializado.
        """
        self._ensure_initialized()
        
        model, _ = self.model_loader.load()
        
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def _generate_explanation(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        num_beams: int,
        do_sample: bool = True,
        **kwargs: Any
    ) -> str:
        """
        Generar explicación desde un prompt (método interno).
        
        Args:
            prompt: Prompt de entrada
            max_length: Longitud máxima
            temperature: Temperatura
            num_beams: Número de beams
            do_sample: Si usar sampling
            **kwargs: Argumentos adicionales
        
        Returns:
            Explicación generada
        """
        self._ensure_initialized()
        
        model, tokenizer = self.model_loader.load()
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.device)
        
        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams if not do_sample else 1,
            do_sample=do_sample,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        explanation = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()
        
        return explanation
    
    def generate(
        self,
        code: str,
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        num_beams: int = 4,
        do_sample: bool = True,
        use_cache: Optional[bool] = None,
        **kwargs: Any
    ) -> str:
        """
        Generar explicación de código.
        
        Args:
            code: Código a explicar.
            max_length: Longitud máxima de la explicación (default: max_target_length).
            temperature: Temperatura para sampling (default: 0.7).
            num_beams: Número de beams para beam search (default: 4).
            do_sample: Si usar sampling (default: True).
            use_cache: Usar caché (default: self.cache.enabled).
            **kwargs: Argumentos adicionales para generate().
        
        Returns:
            Explicación del código en lenguaje natural.
        
        Raises:
            RuntimeError: Si el modelo no está inicializado.
            ValueError: Si el código o parámetros son inválidos.
        """
        # Validar inputs
        validated_code = self.validator.validate_code(code, max_length=self.max_length)
        self.validator.validate_generation_params(
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams
        )
        
        self.stats.increment_request()
        use_cache = use_cache if use_cache is not None else self.cache.enabled
        
        # Intentar obtener del caché (las estadísticas se actualizan automáticamente en cache.get)
        if use_cache:
            cached_result = self.cache.get(
                validated_code,
                max_length=max_length,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=do_sample,
                **kwargs
            )
            if cached_result is not None:
                logger.debug("Cache hit for code explanation")
                return cached_result
        
        # Generar explicación
        try:
            prompt = self.prompt_builder.build(validated_code)
            max_gen_length = max_length or self.max_target_length
            
            explanation = self._generate_explanation(
                prompt=prompt,
                max_length=max_gen_length,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=do_sample,
                **kwargs
            )
            
            # Guardar en caché
            if use_cache and explanation:
                self.cache.set(
                    validated_code,
                    explanation,
                    max_length=max_length,
                    temperature=temperature,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    **kwargs
                )
            
            logger.debug(
                f"Code explanation generated - code_len: {len(validated_code)}, "
                f"explanation_len: {len(explanation)}, max_length: {max_gen_length}"
            )
            
            return explanation
            
        except Exception as e:
            self.stats.increment_error()
            logger.error(f"Error generating explanation: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate explanation: {e}") from e
    
    def explain_code(
        self,
        code: str,
        language: Optional[str] = None,
        detail_level: str = "medium",
        **kwargs: Any
    ) -> str:
        """
        Explicar código con opciones adicionales.
        
        Args:
            code: Código a explicar.
            language: Lenguaje de programación (opcional).
            detail_level: Nivel de detalle ("brief", "medium", "detailed").
            **kwargs: Argumentos adicionales para generate().
        
        Returns:
            Explicación del código.
        
        Raises:
            ValueError: Si detail_level no es válido.
        """
        if detail_level not in PromptBuilder.TEMPLATES:
            raise ValueError(
                f"detail_level must be one of {list(PromptBuilder.TEMPLATES.keys())}, "
                f"got {detail_level}"
            )
        
        # Validar código
        validated_code = self.validator.validate_code(code, max_length=self.max_length)
        
        # Ajustar max_length según nivel de detalle
        max_length = self.prompt_builder.get_max_length_for_detail(
            self.max_target_length,
            detail_level
        )
        
        # Intentar caché primero (las estadísticas se actualizan automáticamente en cache.get)
        use_cache = kwargs.pop("use_cache", self.cache.enabled)
        if use_cache:
            cached_result = self.cache.get(
                validated_code,
                language=language,
                detail_level=detail_level,
                max_length=max_length,
                **kwargs
            )
            if cached_result is not None:
                logger.debug("Cache hit for code explanation")
                return cached_result
        
        # Construir prompt con nivel de detalle
        prompt = self.prompt_builder.build(
            code=validated_code,
            language=language,
            detail_level=detail_level
        )
        
        # Generar con el prompt construido
        try:
            max_gen_length = max_length
            explanation = self._generate_explanation(
                prompt=prompt,
                max_length=max_gen_length,
                temperature=kwargs.get("temperature", 0.7),
                num_beams=kwargs.get("num_beams", 4),
                do_sample=kwargs.get("do_sample", True),
                **{k: v for k, v in kwargs.items() 
                   if k not in ["temperature", "num_beams", "do_sample", "use_cache"]}
            )
            
            # Guardar en caché
            if use_cache and explanation:
                self.cache.set(
                    validated_code,
                    explanation,
                    language=language,
                    detail_level=detail_level,
                    max_length=max_length,
                    **{k: v for k, v in kwargs.items() 
                       if k not in ["use_cache"]}
                )
            
            return explanation
        except Exception as e:
            self.stats.increment_error()
            logger.error(f"Error explaining code: {e}", exc_info=True)
            raise RuntimeError(f"Failed to explain code: {e}") from e
    
    def explain_batch(
        self,
        codes: List[str],
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        num_beams: int = 4,
        do_sample: bool = True,
        **kwargs: Any
    ) -> List[str]:
        """
        Explicar múltiples códigos en batch (más eficiente).
        
        Args:
            codes: Lista de códigos a explicar.
            max_length: Longitud máxima de la explicación.
            temperature: Temperatura para sampling.
            num_beams: Número de beams para beam search.
            do_sample: Si usar sampling.
            **kwargs: Argumentos adicionales para generate().
        
        Returns:
            Lista de explicaciones en el mismo orden que los códigos.
        
        Raises:
            ValueError: Si codes está vacía o contiene elementos inválidos.
            RuntimeError: Si el modelo no está inicializado.
        """
        if not self.batch_processor:
            self._ensure_initialized()
        
        if self.batch_processor is None:
            raise RuntimeError("Batch processor not initialized")
        
        try:
            explanations = self.batch_processor.process(
                codes=codes,
                max_length=max_length,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=do_sample,
                **kwargs
            )
            
            # Actualizar estadísticas
            self.stats.increment_request()
            
            return explanations
            
        except Exception as e:
            self.stats.increment_error()
            logger.error(f"Error in batch processing: {e}", exc_info=True)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del modelo.
        
        Returns:
            Diccionario con estadísticas de uso.
        """
        return self.stats.get_stats(
            initialized=self._initialized,
            model_name=self.model_name,
            device=str(self.device)
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del caché.
        
        Returns:
            Diccionario con estadísticas del caché.
        """
        cache_stats = self.cache.get_stats()
        total_cache_ops = self.stats._stats["cache_hits"] + self.stats._stats["cache_misses"]
        
        return {
            **cache_stats,
            "cache_hits": self.stats._stats["cache_hits"],
            "cache_misses": self.stats._stats["cache_misses"],
            "cache_hit_rate": (
                (self.stats._stats["cache_hits"] / total_cache_ops * 100)
                if total_cache_ops > 0 else 0.0
            )
        }
    
    def clear_cache(self) -> int:
        """
        Limpiar caché de explicaciones.
        
        Returns:
            Número de entradas eliminadas.
        """
        return self.cache.clear()
    
    def save(self, path: str) -> None:
        """
        Guardar modelo y tokenizer.
        
        Args:
            path: Ruta donde guardar el modelo.
        
        Raises:
            RuntimeError: Si el modelo no está inicializado.
        """
        self._ensure_initialized()
        self.model_loader.save(path)
    
    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> "CodeExplanationModel":
        """
        Cargar modelo desde disco.
        
        Args:
            path: Ruta del modelo guardado.
            device: Dispositivo donde cargar el modelo (opcional).
            config: Configuración adicional (opcional).
        
        Returns:
            Instancia del modelo cargado.
        
        Raises:
            RuntimeError: Si hay error al cargar el modelo.
        """
        try:
            if config is None:
                config = {"model_name": path}
            else:
                config["model_name"] = path
            
            model = cls(config)
            if device:
                model.to_device(device)
            model._initialize()
            
            logger.info(f"CodeExplanationModel loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model: {e}") from e

