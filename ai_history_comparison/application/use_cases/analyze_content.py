"""
Analyze Content Use Case - Caso de Uso de Análisis de Contenido
Caso de uso para analizar contenido usando servicios LLM
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from ...domain.entities import Content, Analysis
from ...domain.services import AnalysisService
from ...domain.repositories import ContentRepository, AnalysisRepository
from ...infrastructure.external.llm import LLMService
from ...infrastructure.cache import CacheService
from ...infrastructure.messaging import EventBus
from ..dto import AnalysisDTO, ContentDTO
from ..validators import ContentValidator, AnalysisValidator
from ...core.exceptions import ValidationError, ExternalServiceError

class AnalyzeContentUseCase:
    """Caso de uso para analizar contenido"""
    
    def __init__(
        self,
        content_repository: ContentRepository,
        analysis_repository: AnalysisRepository,
        analysis_service: AnalysisService,
        llm_service: LLMService,
        cache_service: CacheService,
        event_bus: EventBus
    ):
        self.content_repository = content_repository
        self.analysis_repository = analysis_repository
        self.analysis_service = analysis_service
        self.llm_service = llm_service
        self.cache_service = cache_service
        self.event_bus = event_bus
        self.content_validator = ContentValidator()
        self.analysis_validator = AnalysisValidator()
    
    async def execute(
        self,
        content_data: Dict[str, Any],
        analysis_type: str = "comprehensive",
        force_refresh: bool = False
    ) -> AnalysisDTO:
        """
        Ejecutar análisis de contenido
        
        Args:
            content_data: Datos del contenido a analizar
            analysis_type: Tipo de análisis a realizar
            force_refresh: Forzar análisis sin usar caché
            
        Returns:
            AnalysisDTO: Resultado del análisis
            
        Raises:
            ValidationError: Si los datos de entrada son inválidos
            ExternalServiceError: Si falla el servicio LLM
        """
        # 1. Validar entrada
        self.content_validator.validate(content_data)
        
        # 2. Crear o obtener contenido
        content = await self._get_or_create_content(content_data)
        
        # 3. Verificar caché
        if not force_refresh:
            cached_analysis = await self._get_cached_analysis(content.id, analysis_type)
            if cached_analysis:
                return cached_analysis
        
        # 4. Realizar análisis
        analysis_result = await self._perform_analysis(content, analysis_type)
        
        # 5. Guardar análisis
        analysis = await self._save_analysis(content, analysis_result, analysis_type)
        
        # 6. Actualizar caché
        await self._update_cache(content.id, analysis_type, analysis)
        
        # 7. Publicar evento
        await self._publish_analysis_event(content, analysis)
        
        # 8. Retornar resultado
        return AnalysisDTO.from_entity(analysis)
    
    async def _get_or_create_content(self, content_data: Dict[str, Any]) -> Content:
        """Obtener o crear contenido"""
        content_hash = content_data.get("content_hash")
        
        if content_hash:
            # Buscar contenido existente por hash
            existing_content = await self.content_repository.find_by_hash(content_hash)
            if existing_content:
                return existing_content
        
        # Crear nuevo contenido
        content = Content(
            id=content_data["id"],
            content=content_data["content"],
            title=content_data.get("title"),
            description=content_data.get("description"),
            content_type=content_data.get("content_type", "text"),
            model_version=content_data.get("model_version"),
            model_provider=content_data.get("model_provider")
        )
        
        await self.content_repository.save(content)
        return content
    
    async def _get_cached_analysis(self, content_id: str, analysis_type: str) -> Optional[AnalysisDTO]:
        """Obtener análisis desde caché"""
        cache_key = f"analysis:{content_id}:{analysis_type}"
        cached_data = await self.cache_service.get(cache_key)
        
        if cached_data:
            return AnalysisDTO.from_dict(cached_data)
        return None
    
    async def _perform_analysis(self, content: Content, analysis_type: str) -> Dict[str, Any]:
        """Realizar análisis usando servicios LLM"""
        try:
            # Preparar prompt según tipo de análisis
            prompt = self._build_analysis_prompt(content, analysis_type)
            
            # Ejecutar análisis con LLM
            llm_result = await self.llm_service.analyze(
                prompt=prompt,
                model=content.model_version or "gpt-3.5-turbo",
                analysis_type=analysis_type
            )
            
            # Procesar resultado
            analysis_result = self._process_llm_result(llm_result, analysis_type)
            
            # Agregar análisis local
            local_analysis = self.analysis_service.analyze_local(content)
            analysis_result.update(local_analysis)
            
            return analysis_result
            
        except Exception as e:
            raise ExternalServiceError(f"LLM analysis failed: {str(e)}")
    
    def _build_analysis_prompt(self, content: Content, analysis_type: str) -> str:
        """Construir prompt para análisis"""
        base_prompt = f"""
Analyze the following content and provide a comprehensive assessment.

Content: "{content.content}"

Please provide analysis in JSON format with the following structure:
{{
    "readability_score": 0.0-1.0,
    "sentiment_score": -1.0 to 1.0,
    "complexity_score": 0.0-1.0,
    "topic_diversity": 0.0-1.0,
    "consistency_score": 0.0-1.0,
    "key_themes": ["theme1", "theme2"],
    "strengths": ["strength1", "strength2"],
    "improvements": ["improvement1", "improvement2"],
    "overall_quality": 0.0-1.0,
    "confidence": 0.0-1.0
}}

Analysis type: {analysis_type}
"""
        return base_prompt.strip()
    
    def _process_llm_result(self, llm_result: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Procesar resultado del LLM"""
        # Extraer datos del resultado del LLM
        if "content" in llm_result:
            content = llm_result["content"]
            if isinstance(content, dict):
                return content
            elif isinstance(content, str):
                # Intentar parsear JSON
                import json
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"raw_response": content}
        
        return llm_result
    
    async def _save_analysis(self, content: Content, analysis_result: Dict[str, Any], analysis_type: str) -> Analysis:
        """Guardar análisis en repositorio"""
        analysis = Analysis(
            id=f"analysis_{content.id}_{analysis_type}_{int(datetime.utcnow().timestamp())}",
            content_id=content.id,
            analysis_type=analysis_type,
            results=analysis_result,
            model_used=content.model_version,
            provider_used=content.model_provider,
            confidence_score=analysis_result.get("confidence", 0.0),
            processing_time=analysis_result.get("processing_time", 0.0)
        )
        
        await self.analysis_repository.save(analysis)
        return analysis
    
    async def _update_cache(self, content_id: str, analysis_type: str, analysis: Analysis) -> None:
        """Actualizar caché con resultado del análisis"""
        cache_key = f"analysis:{content_id}:{analysis_type}"
        cache_data = analysis.to_dict()
        
        # Cachear por 1 hora
        await self.cache_service.set(cache_key, cache_data, ttl=3600)
    
    async def _publish_analysis_event(self, content: Content, analysis: Analysis) -> None:
        """Publicar evento de análisis completado"""
        from ...domain.events import AnalysisCompletedEvent
        
        event = AnalysisCompletedEvent(
            content_id=content.id,
            analysis_id=analysis.id,
            analysis_type=analysis.analysis_type,
            timestamp=datetime.utcnow()
        )
        
        await self.event_bus.publish(event)
    
    async def get_analysis_history(self, content_id: str) -> List[AnalysisDTO]:
        """Obtener historial de análisis para un contenido"""
        analyses = await self.analysis_repository.find_by_content_id(content_id)
        return [AnalysisDTO.from_entity(analysis) for analysis in analyses]
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Eliminar análisis"""
        analysis = await self.analysis_repository.find_by_id(analysis_id)
        if not analysis:
            return False
        
        await self.analysis_repository.delete(analysis_id)
        
        # Limpiar caché
        cache_key = f"analysis:{analysis.content_id}:{analysis.analysis_type}"
        await self.cache_service.delete(cache_key)
        
        return True







