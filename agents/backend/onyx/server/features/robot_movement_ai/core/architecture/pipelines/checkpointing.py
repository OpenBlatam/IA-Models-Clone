"""
Pipeline Checkpointing
=====================

Sistema de checkpointing y persistencia para pipelines.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

from .pipeline import Pipeline
from .stages import PipelineStage

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass(frozen=True)
class Checkpoint:
    """
    Checkpoint de pipeline.
    Inmutable para mejor seguridad.
    """
    pipeline_name: str
    stage_index: int
    stage_name: str
    data: Any
    context: Dict[str, Any]
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con datos del checkpoint
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """
        Crear desde diccionario.
        
        Args:
            data: Datos del checkpoint
            
        Returns:
            Instancia de Checkpoint
        """
        return cls(**data)


def _generate_checkpoint_filename(
    pipeline_name: str,
    stage_index: int,
    stage_name: str
) -> str:
    """
    Generar nombre de archivo para checkpoint (función pura).
    
    Args:
        pipeline_name: Nombre del pipeline
        stage_index: Índice de la etapa
        stage_name: Nombre de la etapa
        
    Returns:
        Nombre de archivo
    """
    safe_stage_name = stage_name.replace('/', '_').replace('\\', '_')
    return f"{pipeline_name}_stage_{stage_index}_{safe_stage_name}.pkl"


class CheckpointManager:
    """
    Gestor de checkpoints para pipelines.
    Optimizado con mejor manejo de errores y validación.
    """
    
    def __init__(self, checkpoint_dir: Optional[str] = None) -> None:
        """
        Inicializar gestor de checkpoints.
        
        Args:
            checkpoint_dir: Directorio para checkpoints
            
        Raises:
            OSError: Si no se puede crear el directorio
        """
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path.cwd() / "checkpoints"
        
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create checkpoint directory: {e}")
            raise
    
    def save_checkpoint(
        self,
        pipeline_name: str,
        stage_index: int,
        stage_name: str,
        data: T,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guardar checkpoint.
        
        Args:
            pipeline_name: Nombre del pipeline
            stage_index: Índice de la etapa
            stage_name: Nombre de la etapa
            data: Datos
            context: Contexto
            metadata: Metadatos adicionales
            
        Returns:
            Ruta del checkpoint guardado
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se puede escribir el archivo
        """
        if not pipeline_name:
            raise ValueError("Pipeline name cannot be empty")
        
        if stage_index < 0:
            raise ValueError("Stage index cannot be negative")
        
        if not stage_name:
            raise ValueError("Stage name cannot be empty")
        
        checkpoint = Checkpoint(
            pipeline_name=pipeline_name,
            stage_index=stage_index,
            stage_name=stage_name,
            data=data,
            context=context,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {}
        )
        
        checkpoint_file = self.checkpoint_dir / _generate_checkpoint_filename(
            pipeline_name, stage_index, stage_name
        )
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            logger.info(f"Checkpoint saved: {checkpoint_file}")
            return str(checkpoint_file)
        except IOError as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Checkpoint:
        """
        Cargar checkpoint.
        
        Args:
            checkpoint_path: Ruta del checkpoint
            
        Returns:
            Checkpoint cargado
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el archivo está corrupto
        """
        checkpoint_file = Path(checkpoint_path)
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            if not isinstance(checkpoint, Checkpoint):
                raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise ValueError(f"Corrupted checkpoint file: {checkpoint_path}") from e
    
    def list_checkpoints(
        self,
        pipeline_name: Optional[str] = None
    ) -> list[str]:
        """
        Listar checkpoints.
        
        Args:
            pipeline_name: Filtrar por nombre de pipeline
            
        Returns:
            Lista de rutas de checkpoints
        """
        if not self.checkpoint_dir.exists():
            return []
        
        pattern = f"{pipeline_name}_*.pkl" if pipeline_name else "*.pkl"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        return [str(cp) for cp in sorted(checkpoints)]
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Eliminar checkpoint.
        
        Args:
            checkpoint_path: Ruta del checkpoint
            
        Returns:
            True si se eliminó, False si no existía
        """
        checkpoint_file = Path(checkpoint_path)
        
        if not checkpoint_file.exists():
            return False
        
        try:
            checkpoint_file.unlink()
            logger.info(f"Checkpoint deleted: {checkpoint_path}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False
    
    def cleanup_old_checkpoints(
        self,
        pipeline_name: Optional[str] = None,
        keep_latest: int = 5
    ) -> int:
        """
        Limpiar checkpoints antiguos.
        
        Args:
            pipeline_name: Filtrar por nombre de pipeline
            keep_latest: Mantener los N más recientes
            
        Returns:
            Número de checkpoints eliminados
        """
        if keep_latest < 0:
            raise ValueError("keep_latest must be non-negative")
        
        checkpoints = self.list_checkpoints(pipeline_name)
        
        if len(checkpoints) <= keep_latest:
            return 0
        
        deleted_count = 0
        for cp_path in checkpoints[:-keep_latest]:
            if self.delete_checkpoint(cp_path):
                deleted_count += 1
        
        return deleted_count


class CheckpointingMiddleware:
    """
    Middleware para checkpointing automático.
    Optimizado con mejor validación y manejo de errores.
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        checkpoint_interval: int = 1
    ) -> None:
        """
        Inicializar middleware de checkpointing.
        
        Args:
            checkpoint_manager: Gestor de checkpoints
            checkpoint_interval: Intervalo de checkpointing (cada N etapas)
        """
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be at least 1")
        
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_interval = checkpoint_interval
    
    def before_stage(
        self,
        stage: PipelineStage,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[T, Optional[Dict[str, Any]]]:
        """
        No hacer nada antes de la etapa.
        
        Args:
            stage: Etapa
            data: Datos
            context: Contexto
            
        Returns:
            Tupla (datos, contexto) sin modificar
        """
        return data, context
    
    def after_stage(
        self,
        stage: PipelineStage,
        data: T,
        context: Optional[Dict[str, Any]] = None,
        result: Optional[T] = None,
        error: Optional[Exception] = None
    ) -> Optional[T]:
        """
        Guardar checkpoint si es necesario.
        
        Args:
            stage: Etapa
            data: Datos de entrada
            context: Contexto
            result: Resultado de la etapa
            error: Error si hubo
            
        Returns:
            Resultado sin modificar
        """
        if result is None or error is not None:
            return result
        
        if not context:
            return result
        
        stage_index = context.get('_stage_index', 0)
        pipeline_name = context.get('_pipeline_name', 'pipeline')
        
        if stage_index % self.checkpoint_interval == 0:
            try:
                self.checkpoint_manager.save_checkpoint(
                    pipeline_name=pipeline_name,
                    stage_index=stage_index,
                    stage_name=stage.get_name(),
                    data=result,
                    context=context,
                    metadata={'error': False}
                )
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
        
        return result


class PipelineWithCheckpointing:
    """
    Pipeline con soporte para checkpointing.
    Optimizado con mejor validación y manejo de errores.
    """
    
    def __init__(
        self,
        name: str = "pipeline",
        checkpoint_manager: Optional[CheckpointManager] = None,
        enable_checkpointing: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Inicializar pipeline con checkpointing.
        
        Args:
            name: Nombre del pipeline
            checkpoint_manager: Gestor de checkpoints
            enable_checkpointing: Habilitar checkpointing
            **kwargs: Argumentos adicionales para Pipeline
        """
        from .pipeline import Pipeline
        
        if not name:
            raise ValueError("Pipeline name cannot be empty")
        
        self._pipeline = Pipeline(name, **kwargs)
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.enable_checkpointing = enable_checkpointing
        
        if enable_checkpointing:
            self._pipeline.add_middleware(
                CheckpointingMiddleware(self.checkpoint_manager)
            )
    
    def process(
        self,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Procesar con checkpointing.
        
        Args:
            data: Datos de entrada
            context: Contexto
            
        Returns:
            Resultado procesado
        """
        if context is None:
            context = {}
        
        context['_pipeline_name'] = self._pipeline.name
        
        return self._pipeline.process(data, context)
    
    def resume_from_checkpoint(
        self,
        checkpoint_path: str,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Reanudar desde checkpoint.
        
        Args:
            checkpoint_path: Ruta del checkpoint
            context: Contexto adicional
            
        Returns:
            Resultado final
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        start_index = checkpoint.stage_index + 1
        remaining_stages = self._pipeline.stages[start_index:]
        
        if not remaining_stages:
            logger.info("No remaining stages to execute")
            return checkpoint.data
        
        temp_pipeline = Pipeline(f"{self._pipeline.name}_resumed")
        temp_pipeline.stages = remaining_stages
        temp_pipeline.middleware = self._pipeline.middleware
        
        merged_context = {**checkpoint.context}
        if context:
            merged_context.update(context)
        
        return temp_pipeline.process(checkpoint.data, merged_context)
    
    @property
    def stages(self) -> list[PipelineStage]:
        """Obtener etapas del pipeline."""
        return self._pipeline.stages
