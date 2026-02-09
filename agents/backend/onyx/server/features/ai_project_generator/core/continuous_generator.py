"""
Continuous Generator - Generador Continuo
=========================================

Sistema que genera proyectos de forma continua sin parar,
procesando descripciones automáticamente.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

from .project_generator import ProjectGenerator
from ..utils.file_operations import read_json, write_json, FileOperationError

logger = logging.getLogger(__name__)


def _generate_project_id() -> str:
    """
    Genera un ID único para un proyecto (función pura).
    
    Returns:
        ID del proyecto
    """
    return f"project_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"


def _create_project_item(
    project_id: str,
    description: str,
    project_name: Optional[str],
    author: str,
    priority: int
) -> Dict[str, Any]:
    """
    Crea un item de proyecto (función pura).
    
    Args:
        project_id: ID del proyecto
        description: Descripción del proyecto
        project_name: Nombre del proyecto
        author: Autor del proyecto
        priority: Prioridad
        
    Returns:
        Diccionario con información del proyecto
    """
    return {
        'id': project_id,
        'description': description,
        'project_name': project_name,
        'author': author,
        'priority': priority,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'pending',
    }


def _insert_by_priority(queue: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    """
    Inserta un item en la cola según prioridad (función pura).
    
    Args:
        queue: Cola de proyectos
        item: Item a insertar
    """
    priority = item.get('priority', 0)
    inserted = False
    
    for i, existing_item in enumerate(queue):
        if priority > existing_item.get('priority', 0):
            queue.insert(i, item)
            inserted = True
            break
    
    if not inserted:
        queue.append(item)


def _load_queue_from_file(queue_file: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Carga la cola desde un archivo (función pura).
    
    Args:
        queue_file: Archivo de cola
        
    Returns:
        Tupla (cola, proyectos procesados)
    """
    if not queue_file.exists():
        return [], []
    
    try:
        data = read_json(queue_file, default={'queue': [], 'processed': []})
        return (
            data.get('queue', []),
            data.get('processed', [])
        )
    except FileOperationError as e:
        logger.warning(f"Error loading queue: {e}")
        return [], []


def _save_queue_to_file(
    queue_file: Path,
    queue: List[Dict[str, Any]],
    processed: List[Dict[str, Any]],
    max_processed: int = 100
) -> bool:
    """
    Guarda la cola en un archivo (función pura).
    
    Args:
        queue_file: Archivo de cola
        queue: Cola de proyectos
        processed: Proyectos procesados
        max_processed: Máximo de proyectos procesados a guardar
        
    Returns:
        True si se guardó exitosamente, False en caso contrario
    """
    try:
        data = {
            'queue': queue,
            'processed': processed[-max_processed:],
            'last_updated': datetime.now(timezone.utc).isoformat(),
        }
        write_json(queue_file, data)
        return True
    except FileOperationError as e:
        logger.error(f"Error saving queue: {e}")
        return False


def _truncate_description(description: str, max_length: int = 100) -> str:
    """
    Trunca una descripción (función pura).
    
    Args:
        description: Descripción a truncar
        max_length: Longitud máxima
        
    Returns:
        Descripción truncada
    """
    if len(description) <= max_length:
        return description
    return description[:max_length] + '...'


class ContinuousGenerator:
    """
    Generador continuo de proyectos.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(
        self,
        base_output_dir: str = "generated_projects",
        queue_file: Optional[str] = None,
    ) -> None:
        """
        Inicializa el generador continuo.

        Args:
            base_output_dir: Directorio base para proyectos generados
            queue_file: Archivo JSON para almacenar la cola de proyectos
        """
        if not base_output_dir:
            raise ValueError("base_output_dir cannot be empty")
        
        self.project_generator = ProjectGenerator(base_output_dir=base_output_dir)
        self.queue_file = Path(queue_file or "project_queue.json")
        self.is_running = False
        self.current_task: Optional[asyncio.Task] = None
        self.queue: List[Dict[str, Any]] = []
        self.processed_projects: List[Dict[str, Any]] = []
        self._load_queue()
    
    def _load_queue(self) -> None:
        """Carga la cola desde el archivo."""
        self.queue, self.processed_projects = _load_queue_from_file(self.queue_file)
        if self.queue:
            logger.info(f"Queue loaded with {len(self.queue)} pending projects")
    
    def _save_queue(self) -> None:
        """Guarda la cola en el archivo."""
        _save_queue_to_file(self.queue_file, self.queue, self.processed_projects)
    
    def add_project(
        self,
        description: str,
        project_name: Optional[str] = None,
        author: str = "Blatam Academy",
        priority: int = 0,
    ) -> str:
        """
        Agrega un proyecto a la cola.

        Args:
            description: Descripción del proyecto
            project_name: Nombre del proyecto
            author: Autor del proyecto
            priority: Prioridad (mayor número = mayor prioridad)

        Returns:
            ID del proyecto agregado
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not description:
            raise ValueError("description cannot be empty")
        
        if not author:
            raise ValueError("author cannot be empty")
        
        project_id = _generate_project_id()
        project_item = _create_project_item(
            project_id, description, project_name, author, priority
        )
        
        _insert_by_priority(self.queue, project_item)
        self._save_queue()
        
        logger.info(f"Project added to queue: {project_id}")
        
        # Si no está corriendo, iniciar procesamiento
        if not self.is_running:
            asyncio.create_task(self.start())
        
        return project_id
    
    async def start(self) -> None:
        """
        Inicia el procesamiento continuo.
        
        Raises:
            RuntimeError: Si ya está corriendo
        """
        if self.is_running:
            logger.warning("Generator is already running")
            return
        
        self.is_running = True
        logger.info("Starting continuous generator...")
        
        try:
            while self.is_running:
                if self.queue:
                    project_item = self.queue.pop(0)
                    project_item['status'] = 'processing'
                    project_item['started_at'] = datetime.now(timezone.utc).isoformat()
                    self._save_queue()
                    
                    try:
                        logger.info(f"Processing project: {project_item['id']}")
                        
                        result = await self.project_generator.generate_project(
                            description=project_item['description'],
                            project_name=project_item.get('project_name'),
                            author=project_item.get('author', 'Blatam Academy'),
                        )
                        
                        project_item['status'] = 'completed'
                        project_item['completed_at'] = datetime.now(timezone.utc).isoformat()
                        project_item['result'] = result
                        self.processed_projects.append(project_item)
                        
                        logger.info(f"Project completed: {project_item['id']}")
                    
                    except Exception as e:
                        logger.error(f"Error processing project {project_item['id']}: {e}", exc_info=True)
                        project_item['status'] = 'failed'
                        project_item['error'] = str(e)
                        project_item['failed_at'] = datetime.now(timezone.utc).isoformat()
                        self.processed_projects.append(project_item)
                    
                    self._save_queue()
                else:
                    await asyncio.sleep(1)
        
        except asyncio.CancelledError:
            logger.info("Continuous generator cancelled")
        except Exception as e:
            logger.error(f"Error in continuous generator: {e}", exc_info=True)
        finally:
            self.is_running = False
            logger.info("Continuous generator stopped")
    
    async def stop(self) -> None:
        """Detiene el procesamiento continuo."""
        logger.info("Stopping continuous generator...")
        self.is_running = False
        
        if self.current_task:
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del generador.
        
        Returns:
            Diccionario con estado del generador
        """
        return {
            'is_running': self.is_running,
            'queue_size': len(self.queue),
            'processed_count': len(self.processed_projects),
            'pending_projects': [
                {
                    'id': p['id'],
                    'description': _truncate_description(p['description']),
                    'status': p['status'],
                    'created_at': p['created_at'],
                }
                for p in self.queue[:10]
            ],
            'recent_completed': [
                {
                    'id': p['id'],
                    'status': p['status'],
                    'completed_at': p.get('completed_at'),
                }
                for p in self.processed_projects[-10:]
            ],
        }
    
    def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de un proyecto específico.
        
        Args:
            project_id: ID del proyecto
            
        Returns:
            Estado del proyecto o None si no se encuentra
        """
        if not project_id:
            raise ValueError("project_id cannot be empty")
        
        # Buscar en cola
        for project in self.queue:
            if project['id'] == project_id:
                return project
        
        # Buscar en procesados
        for project in self.processed_projects:
            if project['id'] == project_id:
                return project
        
        return None
