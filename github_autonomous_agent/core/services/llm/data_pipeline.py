"""
Data Pipeline - Pipeline de procesamiento de datos para LLMs.

Sigue principios de procesamiento funcional de datos.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import re

from config.logging_config import get_logger

logger = get_logger(__name__)


class ProcessingStage(str, Enum):
    """Etapas del pipeline."""
    RAW = "raw"
    CLEANED = "cleaned"
    NORMALIZED = "normalized"
    ENRICHED = "enriched"
    TOKENIZED = "tokenized"
    READY = "ready"


@dataclass
class ProcessedData:
    """Datos procesados."""
    original: str
    processed: str
    stage: ProcessingStage
    metadata: Dict[str, Any] = None
    transformations: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.transformations is None:
            self.transformations = []


class DataPipeline:
    """
    Pipeline de procesamiento de datos para LLMs.
    
    Sigue principios de procesamiento funcional.
    """
    
    def __init__(self):
        """Inicializar pipeline."""
        self.transformations: List[Callable] = []
        self.stages: Dict[ProcessingStage, List[Callable]] = {
            ProcessingStage.RAW: [],
            ProcessingStage.CLEANED: [],
            ProcessingStage.NORMALIZED: [],
            ProcessingStage.ENRICHED: [],
            ProcessingStage.TOKENIZED: [],
            ProcessingStage.READY: []
        }
    
    def add_transformation(
        self,
        func: Callable[[str], str],
        stage: ProcessingStage = ProcessingStage.CLEANED,
        name: Optional[str] = None
    ) -> 'DataPipeline':
        """
        Agregar transformación al pipeline.
        
        Args:
            func: Función de transformación
            stage: Etapa del pipeline
            name: Nombre de la transformación
            
        Returns:
            Self para chaining
        """
        if name is None:
            name = func.__name__
        
        wrapped_func = lambda data: self._apply_with_logging(func, data, name)
        self.stages[stage].append(wrapped_func)
        self.transformations.append((name, stage))
        
        return self
    
    def _apply_with_logging(
        self,
        func: Callable,
        data: Union[str, ProcessedData],
        name: str
    ) -> Union[str, ProcessedData]:
        """Aplicar transformación con logging."""
        if isinstance(data, ProcessedData):
            original = data.processed
            result = func(data.processed)
            data.processed = result
            data.transformations.append(name)
            return data
        else:
            return func(data)
    
    def process(
        self,
        data: str,
        stages: Optional[List[ProcessingStage]] = None
    ) -> ProcessedData:
        """
        Procesar datos a través del pipeline.
        
        Args:
            data: Datos a procesar
            stages: Etapas a ejecutar (todas si None)
            
        Returns:
            ProcessedData con datos procesados
        """
        if stages is None:
            stages = list(ProcessingStage)
        
        processed = ProcessedData(
            original=data,
            processed=data,
            stage=ProcessingStage.RAW
        )
        
        for stage in stages:
            if stage not in self.stages:
                continue
            
            for transform in self.stages[stage]:
                try:
                    processed = transform(processed)
                    if isinstance(processed, str):
                        processed = ProcessedData(
                            original=data,
                            processed=processed,
                            stage=stage
                        )
                except Exception as e:
                    logger.error(f"Error en transformación {transform.__name__}: {e}")
                    continue
            
            processed.stage = stage
        
        return processed
    
    @staticmethod
    def create_code_pipeline() -> 'DataPipeline':
        """Crear pipeline predefinido para código."""
        pipeline = DataPipeline()
        
        # Limpieza
        pipeline.add_transformation(
            lambda x: re.sub(r'\r\n', '\n', x),  # Normalizar line endings
            ProcessingStage.CLEANED,
            "normalize_line_endings"
        )
        
        pipeline.add_transformation(
            lambda x: re.sub(r'[ \t]+$', '', x, flags=re.MULTILINE),  # Eliminar trailing spaces
            ProcessingStage.CLEANED,
            "remove_trailing_spaces"
        )
        
        # Normalización
        pipeline.add_transformation(
            lambda x: re.sub(r'\n{3,}', '\n\n', x),  # Máximo 2 líneas vacías
            ProcessingStage.NORMALIZED,
            "normalize_blank_lines"
        )
        
        return pipeline
    
    @staticmethod
    def create_text_pipeline() -> 'DataPipeline':
        """Crear pipeline predefinido para texto."""
        pipeline = DataPipeline()
        
        # Limpieza
        pipeline.add_transformation(
            lambda x: re.sub(r'\s+', ' ', x),  # Normalizar espacios
            ProcessingStage.CLEANED,
            "normalize_whitespace"
        )
        
        pipeline.add_transformation(
            lambda x: x.strip(),  # Eliminar espacios al inicio/fin
            ProcessingStage.CLEANED,
            "strip"
        )
        
        # Normalización
        pipeline.add_transformation(
            lambda x: re.sub(r'[^\w\s.,!?;:()\[\]{}"\'-]', '', x),  # Eliminar caracteres especiales
            ProcessingStage.NORMALIZED,
            "remove_special_chars"
        )
        
        return pipeline


# Instancia global
_data_pipeline = DataPipeline()


def get_data_pipeline() -> DataPipeline:
    """Obtener instancia global del pipeline."""
    return _data_pipeline



