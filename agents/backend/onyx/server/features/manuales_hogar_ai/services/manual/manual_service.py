"""
Manual Service
=============

Servicio principal para gestión de manuales.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.base.service_base import BaseService
from ...database.models import Manual
from ...utils.category_detector import CategoryDetector
from ...utils.parsing.manual_parser import ManualParser
from .manual_repository import ManualRepository
from .manual_search_service import ManualSearchService
from .statistics_service import StatisticsService


class ManualService(BaseService):
    """Servicio principal para gestión de manuales."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar servicio.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.category_detector = CategoryDetector()
        self.parser = ManualParser()
        self.repository = ManualRepository(db)
        self.search_service = ManualSearchService(db)
        self.stats_service = StatisticsService(db)
    
    async def save_manual(
        self,
        problem_description: str,
        category: str,
        manual_content: str,
        model_used: Optional[str] = None,
        tokens_used: int = 0,
        image_analysis: Optional[str] = None,
        detected_category: Optional[str] = None,
        images_count: int = 0,
        user_id: Optional[str] = None
    ) -> Manual:
        """
        Guardar manual en base de datos.
        
        Args:
            problem_description: Descripción del problema
            category: Categoría del oficio
            manual_content: Contenido del manual
            model_used: Modelo usado
            tokens_used: Tokens utilizados
            image_analysis: Análisis de imagen (opcional)
            detected_category: Categoría detectada (opcional)
            images_count: Número de imágenes procesadas
        
        Returns:
            Manual guardado
        """
        try:
            parsed_info = self.parser.parse_manual(manual_content, category)
            
            manual = Manual(
                problem_description=problem_description,
                category=category,
                manual_content=manual_content,
                model_used=model_used,
                tokens_used=tokens_used,
                image_analysis=image_analysis,
                detected_category=detected_category,
                images_count=images_count,
                user_id=user_id,
                title=parsed_info.get("title"),
                difficulty=parsed_info.get("difficulty"),
                estimated_time=parsed_info.get("estimated_time"),
                tools_required=parsed_info.get("tools_required"),
                materials_required=parsed_info.get("materials_required"),
                safety_warnings=parsed_info.get("safety_warnings"),
                tags=parsed_info.get("tags")
            )
            
            saved_manual = await self.repository.save(manual)
            
            await self.stats_service.update_usage_stats(
                category, model_used, tokens_used, images_count
            )
            
            self.log_info(f"Manual guardado: ID {saved_manual.id}, Categoría: {category}")
            return saved_manual
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error guardando manual: {str(e)}")
            raise
    
    async def get_manual_by_id(self, manual_id: int) -> Optional[Manual]:
        """
        Obtener manual por ID.
        
        Args:
            manual_id: ID del manual
        
        Returns:
            Manual o None
        """
        return await self.repository.get_by_id(manual_id)
    
    async def search_manuals(
        self,
        category: Optional[str] = None,
        search_term: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        difficulty: Optional[str] = None,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        advanced_query: Optional[str] = None
    ) -> List[Manual]:
        """
        Buscar manuales con filtros avanzados.
        
        Args:
            category: Filtrar por categoría
            search_term: Término de búsqueda
            limit: Límite de resultados
            offset: Offset para paginación
            difficulty: Filtrar por dificultad
            min_rating: Rating mínimo
            max_rating: Rating máximo
            tags: Lista de tags
            date_from: Fecha desde
            date_to: Fecha hasta
            advanced_query: Query avanzada
        
        Returns:
            Lista de manuales
        """
        return await self.search_service.search(
            category=category,
            search_term=search_term,
            limit=limit,
            offset=offset,
            difficulty=difficulty,
            min_rating=min_rating,
            max_rating=max_rating,
            tags=tags,
            date_from=date_from,
            date_to=date_to,
            advanced_query=advanced_query
        )
    
    async def get_recent_manuals(
        self,
        limit: int = 10,
        category: Optional[str] = None
    ) -> List[Manual]:
        """
        Obtener manuales recientes.
        
        Args:
            limit: Número de manuales
            category: Filtrar por categoría (opcional)
        
        Returns:
            Lista de manuales recientes
        """
        return await self.search_service.search(category=category, limit=limit)
    
    async def get_manuals_by_category(
        self,
        category: str,
        limit: int = 20
    ) -> List[Manual]:
        """
        Obtener manuales por categoría.
        
        Args:
            category: Categoría
            limit: Límite de resultados
        
        Returns:
            Lista de manuales
        """
        return await self.repository.get_by_category(category, limit)
    
    async def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Obtener estadísticas de uso.
        
        Args:
            days: Número de días a considerar
        
        Returns:
            Diccionario con estadísticas
        """
        return await self.stats_service.get_statistics(days)

