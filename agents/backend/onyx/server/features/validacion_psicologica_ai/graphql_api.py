"""
API GraphQL para Validación Psicológica AI
===========================================
API GraphQL alternativa a REST
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import structlog
from datetime import datetime

try:
    from strawberry import Schema, Query, Mutation, Field, type as strawberry_type
    from strawberry.fastapi import GraphQLRouter
    STRAWBERRY_AVAILABLE = True
except ImportError:
    STRAWBERRY_AVAILABLE = False
    logger = structlog.get_logger()
    logger.warning("strawberry not available, GraphQL API not available")

if STRAWBERRY_AVAILABLE:
    from .models import (
        PsychologicalValidation,
        PsychologicalProfile,
        ValidationReport,
        SocialMediaPlatform,
        ValidationStatus
    )
    from .service import PsychologicalValidationService
    
    logger = structlog.get_logger()
    
    @strawberry_type
    class SocialMediaPlatformType:
        """Tipo GraphQL para plataforma de red social"""
        value: str
    
    @strawberry_type
    class ValidationStatusType:
        """Tipo GraphQL para estado de validación"""
        value: str
    
    @strawberry_type
    class PersonalityTraitsType:
        """Tipo GraphQL para rasgos de personalidad"""
        openness: float
        conscientiousness: float
        extraversion: float
        agreeableness: float
        neuroticism: float
    
    @strawberry_type
    class EmotionalStateType:
        """Tipo GraphQL para estado emocional"""
        overall_sentiment: str
        emotional_stability: float
        stress_level: float
    
    @strawberry_type
    class PsychologicalProfileType:
        """Tipo GraphQL para perfil psicológico"""
        id: str
        user_id: str
        personality_traits: PersonalityTraitsType
        emotional_state: EmotionalStateType
        confidence_score: float
        strengths: List[str]
        risk_factors: List[str]
        recommendations: List[str]
        created_at: str
        updated_at: str
    
    @strawberry_type
    class ValidationReportType:
        """Tipo GraphQL para reporte de validación"""
        id: str
        validation_id: str
        summary: str
        generated_at: str
    
    @strawberry_type
    class PsychologicalValidationType:
        """Tipo GraphQL para validación psicológica"""
        id: str
        user_id: str
        status: ValidationStatusType
        connected_platforms: List[SocialMediaPlatformType]
        profile: Optional[PsychologicalProfileType]
        report: Optional[ValidationReportType]
        created_at: str
        updated_at: str
        completed_at: Optional[str]
    
    class Query:
        """Queries GraphQL"""
        
        @Field
        async def validation(
            self,
            validation_id: str,
            service: PsychologicalValidationService
        ) -> Optional[PsychologicalValidationType]:
            """Obtener validación por ID"""
            from uuid import UUID
            validation = await service.get_validation(UUID(validation_id))
            
            if not validation:
                return None
            
            # Convertir a tipo GraphQL
            return PsychologicalValidationType(
                id=str(validation.id),
                user_id=str(validation.user_id),
                status=ValidationStatusType(value=validation.status.value),
                connected_platforms=[
                    SocialMediaPlatformType(value=p.value)
                    for p in validation.connected_platforms
                ],
                profile=self._convert_profile(validation.profile) if validation.profile else None,
                report=self._convert_report(validation.report) if validation.report else None,
                created_at=validation.created_at.isoformat(),
                updated_at=validation.updated_at.isoformat(),
                completed_at=validation.completed_at.isoformat() if validation.completed_at else None
            )
        
        @Field
        async def validations(
            self,
            user_id: str,
            service: PsychologicalValidationService,
            limit: int = 100
        ) -> List[PsychologicalValidationType]:
            """Obtener validaciones de un usuario"""
            from uuid import UUID
            validations = await service.get_user_validations(UUID(user_id))
            
            return [
                PsychologicalValidationType(
                    id=str(v.id),
                    user_id=str(v.user_id),
                    status=ValidationStatusType(value=v.status.value),
                    connected_platforms=[
                        SocialMediaPlatformType(value=p.value)
                        for p in v.connected_platforms
                    ],
                    profile=self._convert_profile(v.profile) if v.profile else None,
                    report=self._convert_report(v.report) if v.report else None,
                    created_at=v.created_at.isoformat(),
                    updated_at=v.updated_at.isoformat(),
                    completed_at=v.completed_at.isoformat() if v.completed_at else None
                )
                for v in validations[:limit]
            ]
        
        def _convert_profile(
            self,
            profile: Optional[PsychologicalProfile]
        ) -> Optional[PsychologicalProfileType]:
            """Convertir perfil a tipo GraphQL"""
            if not profile:
                return None
            
            return PsychologicalProfileType(
                id=str(profile.id),
                user_id=str(profile.user_id),
                personality_traits=PersonalityTraitsType(
                    openness=profile.personality_traits.get("openness", 0.0),
                    conscientiousness=profile.personality_traits.get("conscientiousness", 0.0),
                    extraversion=profile.personality_traits.get("extraversion", 0.0),
                    agreeableness=profile.personality_traits.get("agreeableness", 0.0),
                    neuroticism=profile.personality_traits.get("neuroticism", 0.0)
                ),
                emotional_state=EmotionalStateType(
                    overall_sentiment=profile.emotional_state.get("overall_sentiment", "neutral"),
                    emotional_stability=profile.emotional_state.get("emotional_stability", 0.0),
                    stress_level=profile.emotional_state.get("stress_level", 0.0)
                ),
                confidence_score=profile.confidence_score,
                strengths=profile.strengths,
                risk_factors=profile.risk_factors,
                recommendations=profile.recommendations,
                created_at=profile.created_at.isoformat(),
                updated_at=profile.updated_at.isoformat()
            )
        
        def _convert_report(
            self,
            report: Optional[ValidationReport]
        ) -> Optional[ValidationReportType]:
            """Convertir reporte a tipo GraphQL"""
            if not report:
                return None
            
            return ValidationReportType(
                id=str(report.id),
                validation_id=str(report.validation_id),
                summary=report.summary,
                generated_at=report.generated_at.isoformat()
            )
    
    # Crear schema GraphQL
    schema = Schema(query=Query)
    
    def create_graphql_router(service: PsychologicalValidationService):
        """Crear router GraphQL"""
        return GraphQLRouter(schema)
else:
    def create_graphql_router(service):
        """Placeholder cuando strawberry no está disponible"""
        logger.warning("GraphQL not available, install strawberry")
        return None




