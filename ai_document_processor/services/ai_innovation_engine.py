"""
Motor de Innovación AI
======================

Motor para innovación, generación de ideas, análisis de tendencias y desarrollo de soluciones creativas.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path
import hashlib
import numpy as np
from collections import defaultdict, deque
import random
import itertools

logger = logging.getLogger(__name__)

class InnovationType(str, Enum):
    """Tipos de innovación"""
    PRODUCT_INNOVATION = "product_innovation"
    PROCESS_INNOVATION = "process_innovation"
    BUSINESS_MODEL_INNOVATION = "business_model_innovation"
    TECHNOLOGICAL_INNOVATION = "technological_innovation"
    SERVICE_INNOVATION = "service_innovation"
    MARKET_INNOVATION = "market_innovation"
    ORGANIZATIONAL_INNOVATION = "organizational_innovation"
    SUSTAINABLE_INNOVATION = "sustainable_innovation"

class IdeaStatus(str, Enum):
    """Estados de ideas"""
    DRAFT = "draft"
    EVALUATING = "evaluating"
    FEASIBLE = "feasible"
    IN_DEVELOPMENT = "in_development"
    PILOT = "pilot"
    IMPLEMENTED = "implemented"
    REJECTED = "rejected"
    ON_HOLD = "on_hold"

class TrendCategory(str, Enum):
    """Categorías de tendencias"""
    TECHNOLOGY = "technology"
    MARKET = "market"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    ECONOMIC = "economic"
    REGULATORY = "regulatory"
    CULTURAL = "cultural"
    DEMOGRAPHIC = "demographic"

@dataclass
class InnovationIdea:
    """Idea de innovación"""
    id: str
    title: str
    description: str
    innovation_type: InnovationType
    category: str
    author_id: str
    status: IdeaStatus = IdeaStatus.DRAFT
    feasibility_score: float = 0.0
    impact_score: float = 0.0
    novelty_score: float = 0.0
    implementation_effort: float = 0.0
    market_potential: float = 0.0
    tags: List[str] = field(default_factory=list)
    related_ideas: List[str] = field(default_factory=list)
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Trend:
    """Tendencia de innovación"""
    id: str
    name: str
    description: str
    category: TrendCategory
    impact_level: str  # low, medium, high, critical
    time_horizon: str  # short, medium, long
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    related_trends: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class InnovationProject:
    """Proyecto de innovación"""
    id: str
    name: str
    description: str
    idea_id: str
    team_members: List[str] = field(default_factory=list)
    budget: float = 0.0
    timeline: Dict[str, Any] = field(default_factory=dict)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    risks: List[Dict[str, Any]] = field(default_factory=list)
    success_metrics: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "planning"
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class InnovationChallenge:
    """Desafío de innovación"""
    id: str
    title: str
    description: str
    challenge_type: str
    target_audience: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    rewards: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    participants: List[str] = field(default_factory=list)
    submissions: List[str] = field(default_factory=list)
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class InnovationInsight:
    """Insight de innovación"""
    id: str
    title: str
    content: str
    insight_type: str
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    related_ideas: List[str] = field(default_factory=list)
    actionable_items: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

class AIInnovationEngine:
    """Motor de innovación AI"""
    
    def __init__(self):
        self.innovation_ideas: Dict[str, InnovationIdea] = {}
        self.trends: Dict[str, Trend] = {}
        self.innovation_projects: Dict[str, InnovationProject] = {}
        self.innovation_challenges: Dict[str, InnovationChallenge] = {}
        self.innovation_insights: List[InnovationInsight] = []
        
        # Configuración de innovación
        self.idea_evaluation_criteria = [
            "feasibility", "impact", "novelty", "market_potential", "implementation_effort"
        ]
        self.trend_analysis_interval = 3600  # 1 hora
        self.insight_generation_interval = 1800  # 30 minutos
        
        # Workers de innovación
        self.innovation_workers: Dict[str, asyncio.Task] = {}
        self.innovation_active = False
        
        # Base de conocimiento de innovación
        self.innovation_patterns: Dict[str, List[str]] = {}
        self.innovation_templates: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Inicializa el motor de innovación"""
        logger.info("Inicializando motor de innovación AI...")
        
        # Cargar datos existentes
        await self._load_innovation_data()
        
        # Cargar patrones y plantillas
        await self._load_innovation_patterns()
        
        # Iniciar workers de innovación
        await self._start_innovation_workers()
        
        logger.info("Motor de innovación AI inicializado")
    
    async def _load_innovation_data(self):
        """Carga datos de innovación"""
        try:
            # Cargar ideas de innovación
            ideas_file = Path("data/innovation_ideas.json")
            if ideas_file.exists():
                with open(ideas_file, 'r', encoding='utf-8') as f:
                    ideas_data = json.load(f)
                
                for idea_data in ideas_data:
                    idea = InnovationIdea(
                        id=idea_data["id"],
                        title=idea_data["title"],
                        description=idea_data["description"],
                        innovation_type=InnovationType(idea_data["innovation_type"]),
                        category=idea_data["category"],
                        author_id=idea_data["author_id"],
                        status=IdeaStatus(idea_data["status"]),
                        feasibility_score=idea_data["feasibility_score"],
                        impact_score=idea_data["impact_score"],
                        novelty_score=idea_data["novelty_score"],
                        implementation_effort=idea_data["implementation_effort"],
                        market_potential=idea_data["market_potential"],
                        tags=idea_data["tags"],
                        related_ideas=idea_data["related_ideas"],
                        feedback=idea_data["feedback"],
                        created_at=datetime.fromisoformat(idea_data["created_at"]),
                        updated_at=datetime.fromisoformat(idea_data["updated_at"])
                    )
                    self.innovation_ideas[idea.id] = idea
                
                logger.info(f"Cargadas {len(self.innovation_ideas)} ideas de innovación")
            
            # Cargar tendencias
            trends_file = Path("data/innovation_trends.json")
            if trends_file.exists():
                with open(trends_file, 'r', encoding='utf-8') as f:
                    trends_data = json.load(f)
                
                for trend_data in trends_data:
                    trend = Trend(
                        id=trend_data["id"],
                        name=trend_data["name"],
                        description=trend_data["description"],
                        category=TrendCategory(trend_data["category"]),
                        impact_level=trend_data["impact_level"],
                        time_horizon=trend_data["time_horizon"],
                        confidence=trend_data["confidence"],
                        sources=trend_data["sources"],
                        related_trends=trend_data["related_trends"],
                        opportunities=trend_data["opportunities"],
                        threats=trend_data["threats"],
                        created_at=datetime.fromisoformat(trend_data["created_at"]),
                        updated_at=datetime.fromisoformat(trend_data["updated_at"])
                    )
                    self.trends[trend.id] = trend
                
                logger.info(f"Cargadas {len(self.trends)} tendencias")
            
        except Exception as e:
            logger.error(f"Error cargando datos de innovación: {e}")
    
    async def _load_innovation_patterns(self):
        """Carga patrones de innovación"""
        try:
            # Patrones de innovación por tipo
            self.innovation_patterns = {
                InnovationType.PRODUCT_INNOVATION: [
                    "nuevas funcionalidades", "mejoras de diseño", "optimización de rendimiento",
                    "personalización", "sostenibilidad", "accesibilidad", "integración"
                ],
                InnovationType.PROCESS_INNOVATION: [
                    "automatización", "optimización", "digitalización", "agilidad",
                    "eficiencia", "calidad", "colaboración", "trazabilidad"
                ],
                InnovationType.BUSINESS_MODEL_INNOVATION: [
                    "nuevos canales", "modelos de suscripción", "economía colaborativa",
                    "plataformas", "servicios", "personalización", "sostenibilidad"
                ],
                InnovationType.TECHNOLOGICAL_INNOVATION: [
                    "inteligencia artificial", "machine learning", "blockchain",
                    "IoT", "realidad aumentada", "cloud computing", "edge computing"
                ],
                InnovationType.SERVICE_INNOVATION: [
                    "experiencia del usuario", "personalización", "automatización",
                    "proactividad", "accesibilidad", "sostenibilidad", "colaboración"
                ]
            }
            
            # Plantillas de innovación
            self.innovation_templates = {
                "product_innovation": {
                    "title_template": "Innovación en {producto}: {característica}",
                    "description_template": "Desarrollar {característica} para {producto} que {beneficio}",
                    "evaluation_criteria": ["feasibility", "market_potential", "novelty"]
                },
                "process_innovation": {
                    "title_template": "Optimización de {proceso}: {mejora}",
                    "description_template": "Implementar {mejora} en {proceso} para {beneficio}",
                    "evaluation_criteria": ["feasibility", "impact", "implementation_effort"]
                },
                "business_model_innovation": {
                    "title_template": "Nuevo modelo de negocio: {modelo}",
                    "description_template": "Explorar {modelo} para {mercado} con {ventaja}",
                    "evaluation_criteria": ["market_potential", "novelty", "feasibility"]
                }
            }
            
            logger.info("Patrones y plantillas de innovación cargados")
            
        except Exception as e:
            logger.error(f"Error cargando patrones de innovación: {e}")
    
    async def _start_innovation_workers(self):
        """Inicia workers de innovación"""
        try:
            self.innovation_active = True
            
            # Worker de análisis de tendencias
            asyncio.create_task(self._trend_analysis_worker())
            
            # Worker de generación de insights
            asyncio.create_task(self._insight_generation_worker())
            
            # Worker de evaluación de ideas
            asyncio.create_task(self._idea_evaluation_worker())
            
            logger.info("Workers de innovación iniciados")
            
        except Exception as e:
            logger.error(f"Error iniciando workers de innovación: {e}")
    
    async def _trend_analysis_worker(self):
        """Worker de análisis de tendencias"""
        while self.innovation_active:
            try:
                await asyncio.sleep(self.trend_analysis_interval)
                
                # Analizar nuevas tendencias
                await self._analyze_emerging_trends()
                
                # Actualizar tendencias existentes
                await self._update_existing_trends()
                
            except Exception as e:
                logger.error(f"Error en worker de análisis de tendencias: {e}")
                await asyncio.sleep(300)
    
    async def _insight_generation_worker(self):
        """Worker de generación de insights"""
        while self.innovation_active:
            try:
                await asyncio.sleep(self.insight_generation_interval)
                
                # Generar insights basados en tendencias
                await self._generate_trend_insights()
                
                # Generar insights basados en ideas
                await self._generate_idea_insights()
                
            except Exception as e:
                logger.error(f"Error en worker de generación de insights: {e}")
                await asyncio.sleep(300)
    
    async def _idea_evaluation_worker(self):
        """Worker de evaluación de ideas"""
        while self.innovation_active:
            try:
                await asyncio.sleep(600)  # Cada 10 minutos
                
                # Evaluar ideas pendientes
                await self._evaluate_pending_ideas()
                
            except Exception as e:
                logger.error(f"Error en worker de evaluación de ideas: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_emerging_trends(self):
        """Analiza tendencias emergentes"""
        try:
            # Simular análisis de tendencias emergentes
            emerging_trends = [
                {
                    "name": "AI Democratization",
                    "description": "Making AI accessible to non-technical users",
                    "category": TrendCategory.TECHNOLOGY,
                    "impact_level": "high",
                    "time_horizon": "medium"
                },
                {
                    "name": "Sustainable Technology",
                    "description": "Green tech and sustainable innovation",
                    "category": TrendCategory.ENVIRONMENTAL,
                    "impact_level": "critical",
                    "time_horizon": "long"
                },
                {
                    "name": "Remote Collaboration",
                    "description": "Advanced tools for distributed teams",
                    "category": TrendCategory.SOCIAL,
                    "impact_level": "medium",
                    "time_horizon": "short"
                }
            ]
            
            for trend_data in emerging_trends:
                # Verificar si ya existe
                existing_trend = None
                for trend in self.trends.values():
                    if trend.name.lower() == trend_data["name"].lower():
                        existing_trend = trend
                        break
                
                if not existing_trend:
                    # Crear nueva tendencia
                    trend_id = f"trend_{uuid.uuid4().hex[:8]}"
                    trend = Trend(
                        id=trend_id,
                        name=trend_data["name"],
                        description=trend_data["description"],
                        category=trend_data["category"],
                        impact_level=trend_data["impact_level"],
                        time_horizon=trend_data["time_horizon"],
                        confidence=0.7,  # Confianza inicial
                        sources=["AI Analysis", "Market Research"],
                        opportunities=await self._generate_trend_opportunities(trend_data),
                        threats=await self._generate_trend_threats(trend_data)
                    )
                    
                    self.trends[trend_id] = trend
                    logger.info(f"Nueva tendencia identificada: {trend.name}")
            
        except Exception as e:
            logger.error(f"Error analizando tendencias emergentes: {e}")
    
    async def _update_existing_trends(self):
        """Actualiza tendencias existentes"""
        try:
            for trend in self.trends.values():
                # Simular actualización de confianza
                trend.confidence = min(1.0, trend.confidence + np.random.uniform(0.01, 0.05))
                
                # Actualizar timestamp
                trend.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Error actualizando tendencias existentes: {e}")
    
    async def _generate_trend_opportunities(self, trend_data: Dict[str, Any]) -> List[str]:
        """Genera oportunidades basadas en tendencias"""
        try:
            opportunities = []
            
            if trend_data["category"] == TrendCategory.TECHNOLOGY:
                opportunities.extend([
                    "Desarrollo de nuevas herramientas",
                    "Mejora de procesos existentes",
                    "Creación de nuevos mercados",
                    "Optimización de recursos"
                ])
            elif trend_data["category"] == TrendCategory.ENVIRONMENTAL:
                opportunities.extend([
                    "Reducción de impacto ambiental",
                    "Eficiencia energética",
                    "Sostenibilidad a largo plazo",
                    "Cumplimiento regulatorio"
                ])
            elif trend_data["category"] == TrendCategory.SOCIAL:
                opportunities.extend([
                    "Mejora de experiencia del usuario",
                    "Accesibilidad universal",
                    "Inclusión digital",
                    "Colaboración global"
                ])
            
            return opportunities[:3]  # Limitar a 3 oportunidades
            
        except Exception as e:
            logger.error(f"Error generando oportunidades de tendencias: {e}")
            return []
    
    async def _generate_trend_threats(self, trend_data: Dict[str, Any]) -> List[str]:
        """Genera amenazas basadas en tendencias"""
        try:
            threats = []
            
            if trend_data["category"] == TrendCategory.TECHNOLOGY:
                threats.extend([
                    "Obsolescencia tecnológica",
                    "Dependencia excesiva",
                    "Brecha digital",
                    "Costos de implementación"
                ])
            elif trend_data["category"] == TrendCategory.ENVIRONMENTAL:
                threats.extend([
                    "Regulaciones estrictas",
                    "Costos de adaptación",
                    "Resistencia al cambio",
                    "Limitaciones tecnológicas"
                ])
            elif trend_data["category"] == TrendCategory.SOCIAL:
                threats.extend([
                    "Resistencia al cambio",
                    "Brecha generacional",
                    "Privacidad y seguridad",
                    "Dependencia tecnológica"
                ])
            
            return threats[:3]  # Limitar a 3 amenazas
            
        except Exception as e:
            logger.error(f"Error generando amenazas de tendencias: {e}")
            return []
    
    async def _generate_trend_insights(self):
        """Genera insights basados en tendencias"""
        try:
            for trend in self.trends.values():
                if trend.confidence > 0.8:  # Solo tendencias con alta confianza
                    insight_id = f"insight_{uuid.uuid4().hex[:8]}"
                    
                    insight = InnovationInsight(
                        id=insight_id,
                        title=f"Oportunidad de innovación: {trend.name}",
                        content=f"La tendencia '{trend.name}' presenta oportunidades significativas para innovación. {trend.description}",
                        insight_type="trend_opportunity",
                        confidence=trend.confidence,
                        sources=trend.sources,
                        actionable_items=[
                            f"Evaluar impacto de {trend.name} en productos actuales",
                            f"Desarrollar estrategia para aprovechar {trend.name}",
                            f"Identificar socios para {trend.name}"
                        ]
                    )
                    
                    self.innovation_insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generando insights de tendencias: {e}")
    
    async def _generate_idea_insights(self):
        """Genera insights basados en ideas"""
        try:
            # Analizar patrones en ideas existentes
            idea_patterns = defaultdict(list)
            
            for idea in self.innovation_ideas.values():
                idea_patterns[idea.innovation_type].append(idea)
            
            # Generar insights basados en patrones
            for innovation_type, ideas in idea_patterns.items():
                if len(ideas) >= 3:  # Mínimo 3 ideas para generar insight
                    insight_id = f"insight_{uuid.uuid4().hex[:8]}"
                    
                    insight = InnovationInsight(
                        id=insight_id,
                        title=f"Patrón de innovación: {innovation_type.value}",
                        content=f"Se han identificado {len(ideas)} ideas de {innovation_type.value} con patrones comunes que sugieren oportunidades de innovación sistemática.",
                        insight_type="idea_pattern",
                        confidence=0.8,
                        sources=["Análisis de ideas existentes"],
                        actionable_items=[
                            f"Desarrollar framework para {innovation_type.value}",
                            f"Crear plantillas de {innovation_type.value}",
                            f"Establecer métricas para {innovation_type.value}"
                        ]
                    )
                    
                    self.innovation_insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generando insights de ideas: {e}")
    
    async def _evaluate_pending_ideas(self):
        """Evalúa ideas pendientes"""
        try:
            for idea in self.innovation_ideas.values():
                if idea.status == IdeaStatus.DRAFT:
                    # Evaluar idea
                    await self._evaluate_idea(idea)
                    
                    # Cambiar estado
                    if idea.feasibility_score > 0.6 and idea.impact_score > 0.5:
                        idea.status = IdeaStatus.FEASIBLE
                    else:
                        idea.status = IdeaStatus.REJECTED
                    
                    idea.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Error evaluando ideas pendientes: {e}")
    
    async def _evaluate_idea(self, idea: InnovationIdea):
        """Evalúa una idea de innovación"""
        try:
            # Evaluar factibilidad
            idea.feasibility_score = await self._calculate_feasibility_score(idea)
            
            # Evaluar impacto
            idea.impact_score = await self._calculate_impact_score(idea)
            
            # Evaluar novedad
            idea.novelty_score = await self._calculate_novelty_score(idea)
            
            # Evaluar esfuerzo de implementación
            idea.implementation_effort = await self._calculate_implementation_effort(idea)
            
            # Evaluar potencial de mercado
            idea.market_potential = await self._calculate_market_potential(idea)
            
        except Exception as e:
            logger.error(f"Error evaluando idea: {e}")
    
    async def _calculate_feasibility_score(self, idea: InnovationIdea) -> float:
        """Calcula puntuación de factibilidad"""
        try:
            # Factores que afectan la factibilidad
            factors = {
                "technical_complexity": 0.3,
                "resource_requirements": 0.3,
                "timeline_realism": 0.2,
                "regulatory_compliance": 0.2
            }
            
            # Simular cálculo basado en descripción
            score = 0.5  # Puntuación base
            
            # Ajustar basado en tipo de innovación
            if idea.innovation_type == InnovationType.PROCESS_INNOVATION:
                score += 0.2  # Procesos suelen ser más factibles
            elif idea.innovation_type == InnovationType.TECHNOLOGICAL_INNOVATION:
                score -= 0.1  # Tecnología puede ser más compleja
            
            # Ajustar basado en palabras clave
            description_lower = idea.description.lower()
            if any(word in description_lower for word in ["simple", "fácil", "rápido", "básico"]):
                score += 0.2
            elif any(word in description_lower for word in ["complejo", "difícil", "avanzado", "revolucionario"]):
                score -= 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculando puntuación de factibilidad: {e}")
            return 0.5
    
    async def _calculate_impact_score(self, idea: InnovationIdea) -> float:
        """Calcula puntuación de impacto"""
        try:
            score = 0.5  # Puntuación base
            
            # Ajustar basado en tipo de innovación
            if idea.innovation_type == InnovationType.BUSINESS_MODEL_INNOVATION:
                score += 0.3  # Modelos de negocio tienen alto impacto
            elif idea.innovation_type == InnovationType.TECHNOLOGICAL_INNOVATION:
                score += 0.2  # Tecnología tiene impacto significativo
            
            # Ajustar basado en palabras clave
            description_lower = idea.description.lower()
            if any(word in description_lower for word in ["revolucionario", "transformador", "disruptivo", "innovador"]):
                score += 0.3
            elif any(word in description_lower for word in ["mejora", "optimización", "incremental"]):
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculando puntuación de impacto: {e}")
            return 0.5
    
    async def _calculate_novelty_score(self, idea: InnovationIdea) -> float:
        """Calcula puntuación de novedad"""
        try:
            score = 0.5  # Puntuación base
            
            # Comparar con ideas existentes
            similar_ideas = 0
            for other_idea in self.innovation_ideas.values():
                if other_idea.id != idea.id:
                    # Simular comparación de similitud
                    if idea.innovation_type == other_idea.innovation_type:
                        similar_ideas += 1
            
            # Ajustar basado en similitud
            if similar_ideas == 0:
                score += 0.3  # Muy novedoso
            elif similar_ideas < 3:
                score += 0.1  # Moderadamente novedoso
            else:
                score -= 0.2  # Poco novedoso
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculando puntuación de novedad: {e}")
            return 0.5
    
    async def _calculate_implementation_effort(self, idea: InnovationIdea) -> float:
        """Calcula esfuerzo de implementación"""
        try:
            effort = 0.5  # Esfuerzo base
            
            # Ajustar basado en tipo de innovación
            if idea.innovation_type == InnovationType.PROCESS_INNOVATION:
                effort -= 0.2  # Procesos requieren menos esfuerzo
            elif idea.innovation_type == InnovationType.TECHNOLOGICAL_INNOVATION:
                effort += 0.3  # Tecnología requiere más esfuerzo
            
            # Ajustar basado en palabras clave
            description_lower = idea.description.lower()
            if any(word in description_lower for word in ["rápido", "simple", "fácil", "básico"]):
                effort -= 0.2
            elif any(word in description_lower for word in ["complejo", "difícil", "largo plazo", "extenso"]):
                effort += 0.3
            
            return max(0.0, min(1.0, effort))
            
        except Exception as e:
            logger.error(f"Error calculando esfuerzo de implementación: {e}")
            return 0.5
    
    async def _calculate_market_potential(self, idea: InnovationIdea) -> float:
        """Calcula potencial de mercado"""
        try:
            potential = 0.5  # Potencial base
            
            # Ajustar basado en tipo de innovación
            if idea.innovation_type == InnovationType.BUSINESS_MODEL_INNOVATION:
                potential += 0.3  # Modelos de negocio tienen alto potencial
            elif idea.innovation_type == InnovationType.PRODUCT_INNOVATION:
                potential += 0.2  # Productos tienen buen potencial
            
            # Ajustar basado en palabras clave
            description_lower = idea.description.lower()
            if any(word in description_lower for word in ["mercado", "clientes", "demanda", "ventas"]):
                potential += 0.2
            elif any(word in description_lower for word in ["nicho", "específico", "limitado"]):
                potential -= 0.1
            
            return max(0.0, min(1.0, potential))
            
        except Exception as e:
            logger.error(f"Error calculando potencial de mercado: {e}")
            return 0.5
    
    async def create_innovation_idea(
        self,
        title: str,
        description: str,
        innovation_type: InnovationType,
        category: str,
        author_id: str,
        tags: List[str] = None
    ) -> str:
        """Crea nueva idea de innovación"""
        try:
            idea_id = f"idea_{uuid.uuid4().hex[:8]}"
            
            idea = InnovationIdea(
                id=idea_id,
                title=title,
                description=description,
                innovation_type=innovation_type,
                category=category,
                author_id=author_id,
                tags=tags or []
            )
            
            self.innovation_ideas[idea_id] = idea
            
            # Generar ideas relacionadas
            await self._suggest_related_ideas(idea)
            
            # Guardar datos
            await self._save_innovation_data()
            
            logger.info(f"Idea de innovación creada: {idea_id}")
            return idea_id
            
        except Exception as e:
            logger.error(f"Error creando idea de innovación: {e}")
            raise
    
    async def _suggest_related_ideas(self, idea: InnovationIdea):
        """Sugiere ideas relacionadas"""
        try:
            related_ideas = []
            
            # Buscar ideas similares
            for other_idea in self.innovation_ideas.values():
                if other_idea.id != idea.id:
                    # Simular cálculo de similitud
                    similarity = 0.0
                    
                    if idea.innovation_type == other_idea.innovation_type:
                        similarity += 0.3
                    
                    if idea.category == other_idea.category:
                        similarity += 0.2
                    
                    # Comparar tags
                    common_tags = set(idea.tags) & set(other_idea.tags)
                    similarity += len(common_tags) * 0.1
                    
                    if similarity > 0.4:  # Umbral de similitud
                        related_ideas.append(other_idea.id)
            
            idea.related_ideas = related_ideas[:5]  # Limitar a 5 ideas relacionadas
            
        except Exception as e:
            logger.error(f"Error sugiriendo ideas relacionadas: {e}")
    
    async def generate_innovation_ideas(
        self,
        innovation_type: InnovationType,
        category: str,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """Genera ideas de innovación automáticamente"""
        try:
            ideas = []
            patterns = self.innovation_patterns.get(innovation_type, [])
            
            for i in range(count):
                # Seleccionar patrón aleatorio
                pattern = random.choice(patterns) if patterns else "innovación general"
                
                # Generar idea basada en patrón
                idea_data = {
                    "title": f"Idea de {pattern} - {i+1}",
                    "description": f"Desarrollar una solución innovadora que incorpore {pattern} para {category}",
                    "innovation_type": innovation_type.value,
                    "category": category,
                    "tags": [pattern, category, "generada_automáticamente"],
                    "feasibility_score": np.random.uniform(0.3, 0.9),
                    "impact_score": np.random.uniform(0.4, 0.95),
                    "novelty_score": np.random.uniform(0.2, 0.8),
                    "implementation_effort": np.random.uniform(0.2, 0.8),
                    "market_potential": np.random.uniform(0.3, 0.9)
                }
                
                ideas.append(idea_data)
            
            return ideas
            
        except Exception as e:
            logger.error(f"Error generando ideas de innovación: {e}")
            return []
    
    async def create_innovation_challenge(
        self,
        title: str,
        description: str,
        challenge_type: str,
        target_audience: List[str],
        constraints: List[str],
        success_criteria: List[str],
        rewards: List[str],
        deadline: datetime = None
    ) -> str:
        """Crea nuevo desafío de innovación"""
        try:
            challenge_id = f"challenge_{uuid.uuid4().hex[:8]}"
            
            challenge = InnovationChallenge(
                id=challenge_id,
                title=title,
                description=description,
                challenge_type=challenge_type,
                target_audience=target_audience,
                constraints=constraints,
                success_criteria=success_criteria,
                rewards=rewards,
                deadline=deadline
            )
            
            self.innovation_challenges[challenge_id] = challenge
            
            # Guardar datos
            await self._save_innovation_data()
            
            logger.info(f"Desafío de innovación creado: {challenge_id}")
            return challenge_id
            
        except Exception as e:
            logger.error(f"Error creando desafío de innovación: {e}")
            raise
    
    async def _save_innovation_data(self):
        """Guarda datos de innovación"""
        try:
            # Crear directorio de datos
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar ideas de innovación
            ideas_data = []
            for idea in self.innovation_ideas.values():
                ideas_data.append({
                    "id": idea.id,
                    "title": idea.title,
                    "description": idea.description,
                    "innovation_type": idea.innovation_type.value,
                    "category": idea.category,
                    "author_id": idea.author_id,
                    "status": idea.status.value,
                    "feasibility_score": idea.feasibility_score,
                    "impact_score": idea.impact_score,
                    "novelty_score": idea.novelty_score,
                    "implementation_effort": idea.implementation_effort,
                    "market_potential": idea.market_potential,
                    "tags": idea.tags,
                    "related_ideas": idea.related_ideas,
                    "feedback": idea.feedback,
                    "created_at": idea.created_at.isoformat(),
                    "updated_at": idea.updated_at.isoformat()
                })
            
            ideas_file = data_dir / "innovation_ideas.json"
            with open(ideas_file, 'w', encoding='utf-8') as f:
                json.dump(ideas_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Guardar tendencias
            trends_data = []
            for trend in self.trends.values():
                trends_data.append({
                    "id": trend.id,
                    "name": trend.name,
                    "description": trend.description,
                    "category": trend.category.value,
                    "impact_level": trend.impact_level,
                    "time_horizon": trend.time_horizon,
                    "confidence": trend.confidence,
                    "sources": trend.sources,
                    "related_trends": trend.related_trends,
                    "opportunities": trend.opportunities,
                    "threats": trend.threats,
                    "created_at": trend.created_at.isoformat(),
                    "updated_at": trend.updated_at.isoformat()
                })
            
            trends_file = data_dir / "innovation_trends.json"
            with open(trends_file, 'w', encoding='utf-8') as f:
                json.dump(trends_data, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Error guardando datos de innovación: {e}")
    
    async def get_innovation_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard de innovación"""
        try:
            # Estadísticas generales
            total_ideas = len(self.innovation_ideas)
            feasible_ideas = len([i for i in self.innovation_ideas.values() if i.status == IdeaStatus.FEASIBLE])
            implemented_ideas = len([i for i in self.innovation_ideas.values() if i.status == IdeaStatus.IMPLEMENTED])
            
            total_trends = len(self.trends)
            high_impact_trends = len([t for t in self.trends.values() if t.impact_level == "high"])
            critical_trends = len([t for t in self.trends.values() if t.impact_level == "critical"])
            
            total_challenges = len(self.innovation_challenges)
            active_challenges = len([c for c in self.innovation_challenges.values() if c.status == "active"])
            
            total_insights = len(self.innovation_insights)
            
            # Distribución por tipo de innovación
            innovation_type_distribution = {}
            for idea in self.innovation_ideas.values():
                innovation_type = idea.innovation_type.value
                innovation_type_distribution[innovation_type] = innovation_type_distribution.get(innovation_type, 0) + 1
            
            # Distribución por categoría de tendencias
            trend_category_distribution = {}
            for trend in self.trends.values():
                category = trend.category.value
                trend_category_distribution[category] = trend_category_distribution.get(category, 0) + 1
            
            # Ideas con mejor puntuación
            best_ideas = sorted(
                self.innovation_ideas.values(),
                key=lambda x: (x.impact_score + x.feasibility_score + x.novelty_score) / 3,
                reverse=True
            )[:5]
            
            # Tendencias más confiables
            top_trends = sorted(
                self.trends.values(),
                key=lambda x: x.confidence,
                reverse=True
            )[:5]
            
            # Insights recientes
            recent_insights = sorted(
                self.innovation_insights,
                key=lambda x: x.created_at,
                reverse=True
            )[:5]
            
            return {
                "total_ideas": total_ideas,
                "feasible_ideas": feasible_ideas,
                "implemented_ideas": implemented_ideas,
                "total_trends": total_trends,
                "high_impact_trends": high_impact_trends,
                "critical_trends": critical_trends,
                "total_challenges": total_challenges,
                "active_challenges": active_challenges,
                "total_insights": total_insights,
                "innovation_type_distribution": innovation_type_distribution,
                "trend_category_distribution": trend_category_distribution,
                "best_ideas": [
                    {
                        "id": idea.id,
                        "title": idea.title,
                        "innovation_type": idea.innovation_type.value,
                        "impact_score": idea.impact_score,
                        "feasibility_score": idea.feasibility_score,
                        "novelty_score": idea.novelty_score,
                        "status": idea.status.value
                    }
                    for idea in best_ideas
                ],
                "top_trends": [
                    {
                        "id": trend.id,
                        "name": trend.name,
                        "category": trend.category.value,
                        "impact_level": trend.impact_level,
                        "confidence": trend.confidence,
                        "time_horizon": trend.time_horizon
                    }
                    for trend in top_trends
                ],
                "recent_insights": [
                    {
                        "id": insight.id,
                        "title": insight.title,
                        "insight_type": insight.insight_type,
                        "confidence": insight.confidence,
                        "created_at": insight.created_at.isoformat()
                    }
                    for insight in recent_insights
                ],
                "innovation_active": self.innovation_active,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard de innovación: {e}")
            return {"error": str(e)}

