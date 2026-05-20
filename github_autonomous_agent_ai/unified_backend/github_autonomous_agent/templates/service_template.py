"""
Template para crear nuevos servicios de negocio.

Uso:
1. Copia este archivo a core/nombre_service.py
2. Reemplaza 'TemplateService' con el nombre de tu servicio
3. Implementa la lógica de negocio
4. Agrega tests en tests/test_nombre_service.py
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from core.exceptions import CustomException  # Ajustar según tus excepciones

logger = logging.getLogger(__name__)


class TemplateService:
    """
    Servicio para manejar lógica de negocio relacionada con templates.
    
    Este servicio encapsula la lógica de negocio y se comunica con
    repositorios, clientes externos, etc.
    """
    
    def __init__(
        self,
        # repository: ITemplateRepository,  # Inyectar dependencias
        # external_client: IExternalClient,
    ):
        """
        Inicializa el servicio.
        
        Args:
            repository: Repositorio para acceso a datos
            external_client: Cliente para servicios externos
        """
        # self.repository = repository
        # self.external_client = external_client
        pass
    
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea un nuevo template.
        
        Args:
            data: Datos del template a crear
            
        Returns:
            Template creado
            
        Raises:
            CustomException: Si hay un error en la creación
        """
        try:
            logger.info(f"Creando template: {data.get('name')}")
            
            # Validar datos
            self._validate_data(data)
            
            # Procesar datos
            processed_data = self._process_data(data)
            
            # Guardar (ejemplo)
            # template = await self.repository.create(processed_data)
            
            # Retornar
            return {
                "id": "generated-id",
                **processed_data,
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creando template: {e}")
            raise CustomException(f"Error creando template: {str(e)}")
    
    async def get_by_id(self, template_id: str) -> Dict[str, Any]:
        """
        Obtiene un template por ID.
        
        Args:
            template_id: ID del template
            
        Returns:
            Template encontrado
            
        Raises:
            CustomException: Si el template no existe
        """
        try:
            logger.info(f"Obteniendo template: {template_id}")
            
            # Buscar en repositorio
            # template = await self.repository.get_by_id(template_id)
            
            # if not template:
            #     raise CustomException(f"Template {template_id} no encontrado")
            
            # Retornar
            return {
                "id": template_id,
                "name": "Example",
                "created_at": datetime.now().isoformat()
            }
            
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"Error obteniendo template: {e}")
            raise CustomException(f"Error obteniendo template: {str(e)}")
    
    async def list_all(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Lista todos los templates con paginación y filtros.
        
        Args:
            skip: Número de items a saltar
            limit: Número máximo de items
            filters: Filtros opcionales
            
        Returns:
            Lista de templates
        """
        try:
            logger.info(f"Listando templates: skip={skip}, limit={limit}")
            
            # Aplicar filtros
            query_filters = self._build_filters(filters)
            
            # Buscar en repositorio
            # templates = await self.repository.list(skip=skip, limit=limit, filters=query_filters)
            
            # Retornar
            return []
            
        except Exception as e:
            logger.error(f"Error listando templates: {e}")
            raise CustomException(f"Error listando templates: {str(e)}")
    
    async def update(
        self,
        template_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Actualiza un template existente.
        
        Args:
            template_id: ID del template
            data: Datos a actualizar
            
        Returns:
            Template actualizado
            
        Raises:
            CustomException: Si el template no existe
        """
        try:
            logger.info(f"Actualizando template: {template_id}")
            
            # Verificar que existe
            await self.get_by_id(template_id)
            
            # Validar datos
            self._validate_data(data, is_update=True)
            
            # Procesar datos
            processed_data = self._process_data(data)
            
            # Actualizar
            # template = await self.repository.update(template_id, processed_data)
            
            # Retornar
            return {
                "id": template_id,
                **processed_data,
                "updated_at": datetime.now().isoformat()
            }
            
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"Error actualizando template: {e}")
            raise CustomException(f"Error actualizando template: {str(e)}")
    
    async def delete(self, template_id: str) -> bool:
        """
        Elimina un template.
        
        Args:
            template_id: ID del template a eliminar
            
        Returns:
            True si se eliminó correctamente
            
        Raises:
            CustomException: Si el template no existe
        """
        try:
            logger.info(f"Eliminando template: {template_id}")
            
            # Verificar que existe
            await self.get_by_id(template_id)
            
            # Eliminar
            # await self.repository.delete(template_id)
            
            return True
            
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"Error eliminando template: {e}")
            raise CustomException(f"Error eliminando template: {str(e)}")
    
    # ========================================================================
    # Métodos privados de ayuda
    # ========================================================================
    
    def _validate_data(self, data: Dict[str, Any], is_update: bool = False) -> None:
        """
        Valida los datos del template.
        
        Args:
            data: Datos a validar
            is_update: Si es una actualización
            
        Raises:
            CustomException: Si los datos son inválidos
        """
        if not is_update and not data.get('name'):
            raise CustomException("El nombre es requerido")
        
        # Agregar más validaciones según necesidad
    
    def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa los datos antes de guardar.
        
        Args:
            data: Datos a procesar
            
        Returns:
            Datos procesados
        """
        processed = data.copy()
        
        # Ejemplo: normalizar nombre
        if 'name' in processed:
            processed['name'] = processed['name'].strip()
        
        return processed
    
    def _build_filters(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Construye filtros para queries.
        
        Args:
            filters: Filtros del usuario
            
        Returns:
            Filtros procesados
        """
        if not filters:
            return {}
        
        # Procesar y validar filtros
        processed_filters = {}
        
        # Ejemplo: filtrar por nombre
        if 'name' in filters:
            processed_filters['name'] = filters['name']
        
        return processed_filters




