"""
Servicio de Manuales
====================

Servicio para gestionar manuales en la base de datos.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, or_
from sqlalchemy.orm import selectinload

from ..core.base.service_base import BaseService
from ..database.models import Manual, ManualCache, UsageStats, ManualRating, ManualFavorite
from ..utils.category_detector import CategoryDetector
from ..utils.manual_parser import ManualParser
from ..utils.search.advanced_search import AdvancedSearch
from ..utils.validators import Validators


class ManualService(BaseService):
    """Servicio para gestionar manuales."""
    
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
        self.search = AdvancedSearch(db)
        self.validator = Validators()
    
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
            # Parsear manual para extraer información estructurada
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
                # Campos parseados
                title=parsed_info.get("title"),
                difficulty=parsed_info.get("difficulty"),
                estimated_time=parsed_info.get("estimated_time"),
                tools_required=parsed_info.get("tools_required"),
                materials_required=parsed_info.get("materials_required"),
                safety_warnings=parsed_info.get("safety_warnings"),
                tags=parsed_info.get("tags")
            )
            
            self.db.add(manual)
            await self.db.commit()
            await self.db.refresh(manual)
            
            # Actualizar estadísticas
            await self._update_usage_stats(category, model_used, tokens_used, images_count)
            
            self._logger.info(f"Manual guardado: ID {manual.id}, Categoría: {category}")
            return manual
        
        except Exception as e:
            await self.db.rollback()
            self._logger.error(f"Error guardando manual: {str(e)}")
            raise
    
    async def get_manual_by_id(self, manual_id: int) -> Optional[Manual]:
        """
        Obtener manual por ID.
        
        Args:
            manual_id: ID del manual
        
        Returns:
            Manual o None
        """
        try:
            result = await self.db.execute(
                select(Manual).where(Manual.id == manual_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self._logger.error(f"Error obteniendo manual {manual_id}: {str(e)}")
            return None
    
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
            advanced_query: Query avanzada con sintaxis especial
        
        Returns:
            Lista de manuales
        """
        try:
            query = select(Manual)
            
            # Si hay query avanzada, parsearla
            if advanced_query:
                filters = self.search.parse_search_query(advanced_query)
                conditions = self.search.build_search_conditions(filters, Manual)
            else:
                # Filtros básicos
                conditions = []
                if category:
                    is_valid, error = self.validator.validate_category(category)
                    if is_valid:
                        conditions.append(Manual.category == category.lower())
                if search_term:
                    conditions.append(
                        Manual.problem_description.ilike(f"%{search_term}%")
                    )
                if difficulty:
                    is_valid, error = self.validator.validate_difficulty(difficulty)
                    if is_valid:
                        conditions.append(Manual.difficulty == difficulty)
                if min_rating is not None:
                    conditions.append(Manual.average_rating >= min_rating)
                if max_rating is not None:
                    conditions.append(Manual.average_rating <= max_rating)
                if tags:
                    from sqlalchemy import or_
                    tag_conditions = [Manual.tags.ilike(f"%{tag}%") for tag in tags]
                    if tag_conditions:
                        conditions.append(or_(*tag_conditions))
                if date_from:
                    conditions.append(Manual.created_at >= date_from)
                if date_to:
                    conditions.append(Manual.created_at <= date_to)
            
            # Solo manuales públicos
            conditions.append(Manual.is_public == True)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            # Ordenar por relevancia (rating, vistas, fecha)
            query = query.order_by(
                desc(Manual.average_rating),
                desc(Manual.view_count),
                desc(Manual.created_at)
            )
            
            # Paginación
            query = query.limit(limit).offset(offset)
            
            result = await self.db.execute(query)
            return list(result.scalars().all())
        
        except Exception as e:
            self._logger.error(f"Error buscando manuales: {str(e)}")
            return []
    
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
        return await self.search_manuals(category=category, limit=limit)
    
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
        return await self.search_manuals(category=category, limit=limit)
    
    async def get_statistics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Obtener estadísticas de uso.
        
        Args:
            days: Número de días a considerar
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Total de manuales
            total_query = select(func.count(Manual.id))
            total_result = await self.db.execute(total_query)
            total_manuals = total_result.scalar() or 0
            
            # Manuales por categoría
            category_query = select(
                Manual.category,
                func.count(Manual.id).label('count')
            ).where(
                Manual.created_at >= start_date
            ).group_by(Manual.category)
            
            category_result = await self.db.execute(category_query)
            category_stats = {
                row.category: row.count
                for row in category_result.all()
            }
            
            # Total de tokens
            tokens_query = select(func.sum(Manual.tokens_used))
            tokens_result = await self.db.execute(tokens_query)
            total_tokens = tokens_result.scalar() or 0
            
            # Modelos más usados
            model_query = select(
                Manual.model_used,
                func.count(Manual.id).label('count')
            ).where(
                and_(
                    Manual.created_at >= start_date,
                    Manual.model_used.isnot(None)
                )
            ).group_by(Manual.model_used).order_by(desc('count')).limit(5)
            
            model_result = await self.db.execute(model_query)
            top_models = [
                {"model": row.model_used, "count": row.count}
                for row in model_result.all()
            ]
            
            return {
                "total_manuals": total_manuals,
                "category_stats": category_stats,
                "total_tokens": total_tokens,
                "top_models": top_models,
                "period_days": days
            }
        
        except Exception as e:
            self._logger.error(f"Error obteniendo estadísticas: {str(e)}")
            return {}
    
    async def _update_usage_stats(
        self,
        category: str,
        model_used: Optional[str],
        tokens_used: int,
        images_count: int
    ):
        """Actualizar estadísticas de uso."""
        try:
            today = datetime.now().date()
            
            # Buscar estadística del día
            stats_query = select(UsageStats).where(
                and_(
                    func.date(UsageStats.date) == today,
                    UsageStats.category == category,
                    UsageStats.model_used == model_used
                )
            )
            
            result = await self.db.execute(stats_query)
            stats = result.scalar_one_or_none()
            
            if stats:
                # Actualizar existente
                stats.total_requests += 1
                stats.total_tokens += tokens_used
                stats.total_images_processed += images_count
                stats.avg_tokens_per_request = stats.total_tokens / stats.total_requests
                stats.updated_at = datetime.now()
            else:
                # Crear nuevo
                stats = UsageStats(
                    date=datetime.now(),
                    category=category,
                    model_used=model_used,
                    total_requests=1,
                    total_tokens=tokens_used,
                    avg_tokens_per_request=tokens_used,
                    total_images_processed=images_count
                )
                self.db.add(stats)
            
            await self.db.commit()
        
        except Exception as e:
            self._logger.warning(f"Error actualizando estadísticas: {str(e)}")
            # No fallar si las estadísticas fallan
            await self.db.rollback()

