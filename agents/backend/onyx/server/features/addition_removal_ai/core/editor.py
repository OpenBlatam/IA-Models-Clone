"""
Content Editor - Editor principal de contenido
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum

from .analyzer import ContextAnalyzer
from .validator import ChangeValidator
from .history import ChangeHistory
from .ai_engine import AIEngine
from .formatters import ContentFormatter, ContentFormat
from .metrics import MetricsCollector, PerformanceMonitor
from .diff import ContentDiff
from .undo_redo import UndoRedoManager
from .exceptions import (
    ContentValidationError,
    FormatNotSupportedError,
    PositionError,
    BatchOperationError
)

logger = logging.getLogger(__name__)


class Position(Enum):
    """Posiciones para agregar contenido"""
    START = "start"
    END = "end"
    BEFORE = "before"
    AFTER = "after"
    REPLACE = "replace"


class ContentEditor:
    """Editor principal de contenido con capacidades de IA"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el editor de contenido.

        Args:
            config: Configuración opcional del editor
        """
        self.config = config or {}
        self.analyzer = ContextAnalyzer(config)
        self.validator = ChangeValidator(config)
        self.history = ChangeHistory()
        self.ai_engine = AIEngine(config)
        self.formatter = ContentFormatter()
        self.metrics = MetricsCollector()
        self.monitor = PerformanceMonitor(self.metrics)
        self.diff = ContentDiff()
        self.undo_redo = UndoRedoManager()
        
        # Sistema de plugins
        try:
            from .plugins import PluginManager
            self.plugin_manager = PluginManager()
            # Cargar plugins por defecto
            from .plugins import SanitizerPlugin, LoggerPlugin
            self.plugin_manager.register_plugin(SanitizerPlugin())
            self.plugin_manager.register_plugin(LoggerPlugin())
        except Exception as e:
            logger.warning(f"Plugins no disponibles: {e}")
            self.plugin_manager = None
        
        # Sistema de persistencia
        try:
            from .database import DatabaseManager
            self.database = DatabaseManager()
        except Exception as e:
            logger.warning(f"Base de datos no disponible: {e}")
            self.database = None
        
        # Sistema de versionado
        try:
            from .versioning import ContentVersioning
            self.versioning = ContentVersioning()
        except Exception as e:
            logger.warning(f"Versionado no disponible: {e}")
            self.versioning = None
        
        # Sistema de backups
        try:
            from .backup import BackupManager
            self.backup_manager = BackupManager()
        except Exception as e:
            logger.warning(f"Backups no disponibles: {e}")
            self.backup_manager = None
        
        # Sistema de notificaciones
        try:
            from .notifications import NotificationManager, NotificationType
            self.notifications = NotificationManager()
            self.NotificationType = NotificationType
        except Exception as e:
            logger.warning(f"Notificaciones no disponibles: {e}")
            self.notifications = None
        
        # Sistema de integraciones
        try:
            from .integrations import IntegrationManager
            self.integrations = IntegrationManager()
        except Exception as e:
            logger.warning(f"Integraciones no disponibles: {e}")
            self.integrations = None
        
        # Sistema de tareas programadas
        try:
            from .scheduler import TaskScheduler
            self.scheduler = TaskScheduler()
            # Iniciar scheduler en background
            asyncio.create_task(self.scheduler.start())
        except Exception as e:
            logger.warning(f"Scheduler no disponible: {e}")
            self.scheduler = None
        
        # Sistema de exportación/importación
        try:
            from .export_import import ExportManager, ImportManager
            self.export_manager = ExportManager()
            self.import_manager = ImportManager()
        except Exception as e:
            logger.warning(f"Export/Import no disponible: {e}")
            self.export_manager = None
            self.import_manager = None
        
        # Sistema de búsqueda
        try:
            from .search import SearchEngine
            self.search_engine = SearchEngine()
        except Exception as e:
            logger.warning(f"Búsqueda no disponible: {e}")
            self.search_engine = None
        
        # Sistema de plantillas
        try:
            from .templates import TemplateManager
            self.templates = TemplateManager()
        except Exception as e:
            logger.warning(f"Plantillas no disponibles: {e}")
            self.templates = None
        
        # Sistema de eventos
        try:
            from .events import EventBus, EventType, Event
            self.event_bus = EventBus()
            self.EventType = EventType
            self.Event = Event
        except Exception as e:
            logger.warning(f"Eventos no disponibles: {e}")
            self.event_bus = None
        
        # Sistema de validación de esquemas
        try:
            from .schema_validator import SchemaValidator, ContentSchema
            self.schema_validator = SchemaValidator()
            self.ContentSchema = ContentSchema
        except Exception as e:
            logger.warning(f"Validación de esquemas no disponible: {e}")
            self.schema_validator = None
        
        # Sistema de transformaciones
        try:
            from .transformations import (
                TransformationPipeline,
                CaseTransformation,
                WhitespaceTransformation,
                LineTransformation
            )
            self.transformation_pipeline = TransformationPipeline()
            self.CaseTransformation = CaseTransformation
            self.WhitespaceTransformation = WhitespaceTransformation
            self.LineTransformation = LineTransformation
        except Exception as e:
            logger.warning(f"Transformaciones no disponibles: {e}")
            self.transformation_pipeline = None
        
        # Sistema de colas
        try:
            from .queue import TaskQueue, TaskPriority
            self.task_queue = TaskQueue()
            asyncio.create_task(self.task_queue.start())
            self.TaskPriority = TaskPriority
        except Exception as e:
            logger.warning(f"Cola de tareas no disponible: {e}")
            self.task_queue = None
        
        # Analizador avanzado de contenido
        try:
            from .content_analyzer import AdvancedContentAnalyzer
            self.advanced_analyzer = AdvancedContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador avanzado no disponible: {e}")
            self.advanced_analyzer = None
        
        # Sistema de colaboración
        try:
            from .collaboration import CollaborationManager
            self.collaboration = CollaborationManager()
        except Exception as e:
            logger.warning(f"Colaboración no disponible: {e}")
            self.collaboration = None
        
        # Sistema de permisos
        try:
            from .permissions import PermissionManager, Permission
            self.permissions = PermissionManager()
            self.Permission = Permission
        except Exception as e:
            logger.warning(f"Permisos no disponibles: {e}")
            self.permissions = None
        
        # Sistema de workflow
        try:
            from .workflow import WorkflowManager, WorkflowState
            self.workflow = WorkflowManager()
            self.WorkflowState = WorkflowState
        except Exception as e:
            logger.warning(f"Workflow no disponible: {e}")
            self.workflow = None
        
        # Sistema de etiquetas
        try:
            from .tags import TagManager
            self.tags = TagManager()
        except Exception as e:
            logger.warning(f"Etiquetas no disponibles: {e}")
            self.tags = None
        
        # Sistema de estadísticas avanzadas
        try:
            from .statistics import StatisticsCollector
            self.statistics = StatisticsCollector()
        except Exception as e:
            logger.warning(f"Estadísticas no disponibles: {e}")
            self.statistics = None
        
        # Sistema de reportes
        try:
            from .reports import ReportGenerator
            self.report_generator = ReportGenerator()
        except Exception as e:
            logger.warning(f"Reportes no disponibles: {e}")
            self.report_generator = None
        
        # Sistema de alertas
        try:
            from .alerts import AlertManager, AlertType, AlertSeverity
            self.alerts = AlertManager()
            self.AlertType = AlertType
            self.AlertSeverity = AlertSeverity
        except Exception as e:
            logger.warning(f"Alertas no disponibles: {e}")
            self.alerts = None
        
        # Motor de optimización
        try:
            from .optimization_engine import OptimizationEngine
            self.optimization_engine = OptimizationEngine()
        except Exception as e:
            logger.warning(f"Motor de optimización no disponible: {e}")
            self.optimization_engine = None
        
        # Sistema de recomendaciones
        try:
            from .recommendations import RecommendationEngine
            self.recommendations = RecommendationEngine()
        except Exception as e:
            logger.warning(f"Recomendaciones no disponibles: {e}")
            self.recommendations = None
        
        # Sistema de clustering
        try:
            from .clustering import ClusteringEngine
            self.clustering = ClusteringEngine()
        except Exception as e:
            logger.warning(f"Clustering no disponible: {e}")
            self.clustering = None
        
        # Análisis predictivo
        try:
            from .predictive_analysis import PredictiveAnalyzer
            self.predictive = PredictiveAnalyzer()
        except Exception as e:
            logger.warning(f"Análisis predictivo no disponible: {e}")
            self.predictive = None
        
        # Sistema de aprendizaje ML
        try:
            from .ml_learning import MLLearningEngine
            self.ml_learning = MLLearningEngine()
        except Exception as e:
            logger.warning(f"Aprendizaje ML no disponible: {e}")
            self.ml_learning = None
        
        # Sistema de sincronización
        try:
            from .sync import SyncManager
            self.sync_manager = SyncManager()
        except Exception as e:
            logger.warning(f"Sincronización no disponible: {e}")
            self.sync_manager = None
        
        # Sistema de reglas de negocio
        try:
            from .business_rules import BusinessRulesEngine, create_default_rules
            self.business_rules = BusinessRulesEngine()
            # Registrar reglas por defecto
            for rule in create_default_rules():
                self.business_rules.rules.append(rule)
        except Exception as e:
            logger.warning(f"Reglas de negocio no disponibles: {e}")
            self.business_rules = None
        
        # Sistema de auditoría
        try:
            from .audit import AuditManager, AuditEventType
            self.audit = AuditManager()
            self.AuditEventType = AuditEventType
        except Exception as e:
            logger.warning(f"Auditoría no disponible: {e}")
            self.audit = None
        
        # Comparación avanzada
        try:
            from .comparison import AdvancedComparison
            self.advanced_comparison = AdvancedComparison()
        except Exception as e:
            logger.warning(f"Comparación avanzada no disponible: {e}")
            self.advanced_comparison = None
        
        # Analizador de calidad
        try:
            from .quality_analyzer import QualityAnalyzer
            self.quality_analyzer = QualityAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de calidad no disponible: {e}")
            self.quality_analyzer = None
        
        # Generador de resúmenes
        try:
            from .summarizer import Summarizer
            self.summarizer = Summarizer()
        except Exception as e:
            logger.warning(f"Generador de resúmenes no disponible: {e}")
            self.summarizer = None
        
        # Búsqueda semántica
        try:
            from .semantic_search import SemanticSearch
            self.semantic_search = SemanticSearch()
        except Exception as e:
            logger.warning(f"Búsqueda semántica no disponible: {e}")
            self.semantic_search = None
        
        # Traductor
        try:
            from .translator import Translator
            self.translator = Translator()
        except Exception as e:
            logger.warning(f"Traductor no disponible: {e}")
            self.translator = None
        
        # Corrector ortográfico
        try:
            from .spell_checker import SpellChecker
            self.spell_checker = SpellChecker()
        except Exception as e:
            logger.warning(f"Corrector ortográfico no disponible: {e}")
            self.spell_checker = None
        
        # Validador de contenido mejorado
        try:
            from .content_validator import ContentValidator, ValidationLevel
            self.content_validator = ContentValidator(ValidationLevel.MODERATE)
            self.ValidationLevel = ValidationLevel
        except Exception as e:
            logger.warning(f"Validador de contenido no disponible: {e}")
            self.content_validator = None
        
        # Analizador de sentimientos avanzado
        try:
            from .sentiment_analyzer import AdvancedSentimentAnalyzer
            self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de sentimientos no disponible: {e}")
            self.sentiment_analyzer = None
        
        # Extractor de entidades
        try:
            from .entity_extractor import EntityExtractor
            self.entity_extractor = EntityExtractor()
        except Exception as e:
            logger.warning(f"Extractor de entidades no disponible: {e}")
            self.entity_extractor = None
        
        # Detector de plagio
        try:
            from .plagiarism_detector import PlagiarismDetector
            self.plagiarism_detector = PlagiarismDetector()
        except Exception as e:
            logger.warning(f"Detector de plagio no disponible: {e}")
            self.plagiarism_detector = None
        
        # Modelador de temas
        try:
            from .topic_modeler import TopicModeler
            self.topic_modeler = TopicModeler()
        except Exception as e:
            logger.warning(f"Modelador de temas no disponible: {e}")
            self.topic_modeler = None
        
        # Analizador de complejidad
        try:
            from .complexity_analyzer import ComplexityAnalyzer
            self.complexity_analyzer = ComplexityAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de complejidad no disponible: {e}")
            self.complexity_analyzer = None
        
        # Generador de contenido
        try:
            from .content_generator import ContentGenerator
            self.content_generator = ContentGenerator()
        except Exception as e:
            logger.warning(f"Generador de contenido no disponible: {e}")
            self.content_generator = None
        
        # Analizador de redundancia
        try:
            from .redundancy_analyzer import RedundancyAnalyzer
            self.redundancy_analyzer = RedundancyAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de redundancia no disponible: {e}")
            self.redundancy_analyzer = None
        
        # Analizador de estructura
        try:
            from .structure_analyzer import StructureAnalyzer
            self.structure_analyzer = StructureAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de estructura no disponible: {e}")
            self.structure_analyzer = None
        
        # Analizador de tono
        try:
            from .tone_analyzer import ToneAnalyzer
            self.tone_analyzer = ToneAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de tono no disponible: {e}")
            self.tone_analyzer = None
        
        # Analizador de coherencia
        try:
            from .coherence_analyzer import CoherenceAnalyzer
            self.coherence_analyzer = CoherenceAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de coherencia no disponible: {e}")
            self.coherence_analyzer = None
        
        # Analizador de accesibilidad
        try:
            from .accessibility_analyzer import AccessibilityAnalyzer
            self.accessibility_analyzer = AccessibilityAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de accesibilidad no disponible: {e}")
            self.accessibility_analyzer = None
        
        # Analizador SEO
        try:
            from .seo_analyzer import SEOAnalyzer
            self.seo_analyzer = SEOAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador SEO no disponible: {e}")
            self.seo_analyzer = None
        
        # Analizador de legibilidad avanzado
        try:
            from .readability_advanced import AdvancedReadabilityAnalyzer
            self.readability_advanced = AdvancedReadabilityAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de legibilidad avanzado no disponible: {e}")
            self.readability_advanced = None
        
        # Analizador de fluidez
        try:
            from .fluency_analyzer import FluencyAnalyzer
            self.fluency_analyzer = FluencyAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de fluidez no disponible: {e}")
            self.fluency_analyzer = None
        
        # Analizador de vocabulario
        try:
            from .vocabulary_analyzer import VocabularyAnalyzer
            self.vocabulary_analyzer = VocabularyAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de vocabulario no disponible: {e}")
            self.vocabulary_analyzer = None
        
        # Analizador de formato
        try:
            from .format_analyzer import FormatAnalyzer
            self.format_analyzer = FormatAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de formato no disponible: {e}")
            self.format_analyzer = None
        
        # Optimizador de longitud
        try:
            from .length_optimizer import LengthOptimizer
            self.length_optimizer = LengthOptimizer()
        except Exception as e:
            logger.warning(f"Optimizador de longitud no disponible: {e}")
            self.length_optimizer = None
        
        # Recomendador de mejoras
        try:
            from .improvement_recommender import ImprovementRecommender
            self.improvement_recommender = ImprovementRecommender()
        except Exception as e:
            logger.warning(f"Recomendador de mejoras no disponible: {e}")
            self.improvement_recommender = None
        
        # Analizador de engagement
        try:
            from .engagement_analyzer import EngagementAnalyzer
            self.engagement_analyzer = EngagementAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de engagement no disponible: {e}")
            self.engagement_analyzer = None
        
        # Métricas de contenido
        try:
            from .content_metrics import ContentMetrics
            self.content_metrics = ContentMetrics()
        except Exception as e:
            logger.warning(f"Métricas de contenido no disponibles: {e}")
            self.content_metrics = None
        
        # Analizador de performance
        try:
            from .performance_analyzer import PerformanceAnalyzer
            self.performance_analyzer = PerformanceAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de performance no disponible: {e}")
            self.performance_analyzer = None
        
        # Analizador de tendencias
        try:
            from .trend_analyzer import TrendAnalyzer
            self.trend_analyzer = TrendAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de tendencias no disponible: {e}")
            self.trend_analyzer = None
        
        # Analizador de competencia
        try:
            from .competitor_analyzer import CompetitorAnalyzer
            self.competitor_analyzer = CompetitorAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de competencia no disponible: {e}")
            self.competitor_analyzer = None
        
        # Analizador de ROI
        try:
            from .roi_analyzer import ROIAnalyzer
            self.roi_analyzer = ROIAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de ROI no disponible: {e}")
            self.roi_analyzer = None
        
        # Analizador de audiencia
        try:
            from .audience_analyzer import AudienceAnalyzer
            self.audience_analyzer = AudienceAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de audiencia no disponible: {e}")
            self.audience_analyzer = None
        
        # Analizador de conversión
        try:
            from .conversion_analyzer import ConversionAnalyzer
            self.conversion_analyzer = ConversionAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de conversión no disponible: {e}")
            self.conversion_analyzer = None
        
        # Gestor de A/B testing
        try:
            from .ab_testing import ABTestingManager
            self.ab_testing = ABTestingManager()
        except Exception as e:
            logger.warning(f"Gestor de A/B testing no disponible: {e}")
            self.ab_testing = None
        
        # Analizador de feedback
        try:
            from .feedback_analyzer import FeedbackAnalyzer
            self.feedback_analyzer = FeedbackAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de feedback no disponible: {e}")
            self.feedback_analyzer = None
        
        # Motor de personalización
        try:
            from .personalization_engine import PersonalizationEngine
            self.personalization_engine = PersonalizationEngine()
        except Exception as e:
            logger.warning(f"Motor de personalización no disponible: {e}")
            self.personalization_engine = None
        
        # Analizador de satisfacción
        try:
            from .satisfaction_analyzer import SatisfactionAnalyzer
            self.satisfaction_analyzer = SatisfactionAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de satisfacción no disponible: {e}")
            self.satisfaction_analyzer = None
        
        # Analizador de comportamiento
        try:
            from .behavior_analyzer import BehaviorAnalyzer
            self.behavior_analyzer = BehaviorAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de comportamiento no disponible: {e}")
            self.behavior_analyzer = None
        
        # Analizador de retención
        try:
            from .retention_analyzer import RetentionAnalyzer
            self.retention_analyzer = RetentionAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de retención no disponible: {e}")
            self.retention_analyzer = None
        
        # Analizador de viralidad
        try:
            from .virality_analyzer import ViralityAnalyzer
            self.virality_analyzer = ViralityAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de viralidad no disponible: {e}")
            self.virality_analyzer = None
        
        # Analizador predictivo de contenido
        try:
            from .predictive_content_analyzer import PredictiveContentAnalyzer
            self.predictive_analyzer = PredictiveContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador predictivo no disponible: {e}")
            self.predictive_analyzer = None
        
        # Analizador multiidioma
        try:
            from .multilanguage_analyzer import MultilanguageAnalyzer
            self.multilanguage_analyzer = MultilanguageAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador multiidioma no disponible: {e}")
            self.multilanguage_analyzer = None
        
        # Analizador de contenido generativo
        try:
            from .generative_content_analyzer import GenerativeContentAnalyzer
            self.generative_analyzer = GenerativeContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido generativo no disponible: {e}")
            self.generative_analyzer = None
        
        # Analizador en tiempo real
        try:
            from .realtime_analyzer import RealtimeAnalyzer
            self.realtime_analyzer = RealtimeAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador en tiempo real no disponible: {e}")
            self.realtime_analyzer = None
        
        # Analizador multimedia
        try:
            from .multimedia_analyzer import MultimediaAnalyzer
            self.multimedia_analyzer = MultimediaAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador multimedia no disponible: {e}")
            self.multimedia_analyzer = None
        
        # Analizador adaptativo
        try:
            from .adaptive_content_analyzer import AdaptiveContentAnalyzer
            self.adaptive_analyzer = AdaptiveContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador adaptativo no disponible: {e}")
            self.adaptive_analyzer = None
        
        # Analizador de contenido interactivo
        try:
            from .interactive_content_analyzer import InteractiveContentAnalyzer
            self.interactive_analyzer = InteractiveContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido interactivo no disponible: {e}")
            self.interactive_analyzer = None
        
        # Analizador contextual
        try:
            from .contextual_analyzer import ContextualAnalyzer
            self.contextual_analyzer = ContextualAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador contextual no disponible: {e}")
            self.contextual_analyzer = None
        
        # Analizador narrativo
        try:
            from .narrative_analyzer import NarrativeAnalyzer
            self.narrative_analyzer = NarrativeAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador narrativo no disponible: {e}")
            self.narrative_analyzer = None
        
        # Analizador de contenido emocional
        try:
            from .emotional_content_analyzer import EmotionalContentAnalyzer
            self.emotional_analyzer = EmotionalContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido emocional no disponible: {e}")
            self.emotional_analyzer = None
        
        # Analizador de contenido persuasivo
        try:
            from .persuasive_content_analyzer import PersuasiveContentAnalyzer
            self.persuasive_analyzer = PersuasiveContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido persuasivo no disponible: {e}")
            self.persuasive_analyzer = None
        
        # Analizador de contenido educativo
        try:
            from .educational_content_analyzer import EducationalContentAnalyzer
            self.educational_analyzer = EducationalContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido educativo no disponible: {e}")
            self.educational_analyzer = None
        
        # Analizador de contenido técnico
        try:
            from .technical_content_analyzer import TechnicalContentAnalyzer
            self.technical_analyzer = TechnicalContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido técnico no disponible: {e}")
            self.technical_analyzer = None
        
        # Analizador de contenido creativo
        try:
            from .creative_content_analyzer import CreativeContentAnalyzer
            self.creative_analyzer = CreativeContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido creativo no disponible: {e}")
            self.creative_analyzer = None
        
        # Analizador de contenido científico
        try:
            from .scientific_content_analyzer import ScientificContentAnalyzer
            self.scientific_analyzer = ScientificContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido científico no disponible: {e}")
            self.scientific_analyzer = None
        
        # Analizador de contenido legal
        try:
            from .legal_content_analyzer import LegalContentAnalyzer
            self.legal_analyzer = LegalContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido legal no disponible: {e}")
            self.legal_analyzer = None
        
        # Analizador de contenido financiero
        try:
            from .financial_content_analyzer import FinancialContentAnalyzer
            self.financial_analyzer = FinancialContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido financiero no disponible: {e}")
            self.financial_analyzer = None
        
        # Analizador de contenido periodístico
        try:
            from .journalistic_content_analyzer import JournalisticContentAnalyzer
            self.journalistic_analyzer = JournalisticContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido periodístico no disponible: {e}")
            self.journalistic_analyzer = None
        
        # Analizador de contenido médico
        try:
            from .medical_content_analyzer import MedicalContentAnalyzer
            self.medical_analyzer = MedicalContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido médico no disponible: {e}")
            self.medical_analyzer = None
        
        # Analizador de contenido de marketing
        try:
            from .marketing_content_analyzer import MarketingContentAnalyzer
            self.marketing_analyzer = MarketingContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de marketing no disponible: {e}")
            self.marketing_analyzer = None
        
        # Analizador de contenido de ventas
        try:
            from .sales_content_analyzer import SalesContentAnalyzer
            self.sales_analyzer = SalesContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de ventas no disponible: {e}")
            self.sales_analyzer = None
        
        # Analizador de contenido de recursos humanos
        try:
            from .hr_content_analyzer import HRContentAnalyzer
            self.hr_analyzer = HRContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de recursos humanos no disponible: {e}")
            self.hr_analyzer = None
        
        # Analizador de contenido de soporte técnico
        try:
            from .support_content_analyzer import SupportContentAnalyzer
            self.support_analyzer = SupportContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de soporte técnico no disponible: {e}")
            self.support_analyzer = None
        
        # Analizador de contenido de documentación técnica
        try:
            from .documentation_content_analyzer import DocumentationContentAnalyzer
            self.documentation_analyzer = DocumentationContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de documentación técnica no disponible: {e}")
            self.documentation_analyzer = None
        
        # Analizador de contenido de blog
        try:
            from .blog_content_analyzer import BlogContentAnalyzer
            self.blog_analyzer = BlogContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de blog no disponible: {e}")
            self.blog_analyzer = None
        
        # Analizador de contenido de email marketing
        try:
            from .email_marketing_analyzer import EmailMarketingAnalyzer
            self.email_marketing_analyzer = EmailMarketingAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de email marketing no disponible: {e}")
            self.email_marketing_analyzer = None
        
        # Analizador de contenido de redes sociales
        try:
            from .social_media_analyzer import SocialMediaAnalyzer
            self.social_media_analyzer = SocialMediaAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de redes sociales no disponible: {e}")
            self.social_media_analyzer = None
        
        # Analizador de contenido de e-learning
        try:
            from .elearning_content_analyzer import ELearningContentAnalyzer
            self.elearning_analyzer = ELearningContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de e-learning no disponible: {e}")
            self.elearning_analyzer = None
        
        # Analizador de contenido de podcast/audio
        try:
            from .podcast_content_analyzer import PodcastContentAnalyzer
            self.podcast_analyzer = PodcastContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de podcast no disponible: {e}")
            self.podcast_analyzer = None
        
        # Analizador de contenido de video/YouTube
        try:
            from .video_content_analyzer import VideoContentAnalyzer
            self.video_analyzer = VideoContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de video no disponible: {e}")
            self.video_analyzer = None
        
        # Analizador de contenido de noticias
        try:
            from .news_content_analyzer import NewsContentAnalyzer
            self.news_analyzer = NewsContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de noticias no disponible: {e}")
            self.news_analyzer = None
        
        # Analizador de contenido de reseñas
        try:
            from .review_content_analyzer import ReviewContentAnalyzer
            self.review_analyzer = ReviewContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de reseñas no disponible: {e}")
            self.review_analyzer = None
        
        # Analizador de contenido de landing pages
        try:
            from .landing_page_analyzer import LandingPageAnalyzer
            self.landing_page_analyzer = LandingPageAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de landing pages no disponible: {e}")
            self.landing_page_analyzer = None
        
        # Analizador de contenido de FAQ
        try:
            from .faq_content_analyzer import FAQContentAnalyzer
            self.faq_analyzer = FAQContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de FAQ no disponible: {e}")
            self.faq_analyzer = None
        
        # Analizador de contenido de newsletters
        try:
            from .newsletter_content_analyzer import NewsletterContentAnalyzer
            self.newsletter_analyzer = NewsletterContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de newsletters no disponible: {e}")
            self.newsletter_analyzer = None
        
        # Analizador de contenido de whitepapers
        try:
            from .whitepaper_content_analyzer import WhitepaperContentAnalyzer
            self.whitepaper_analyzer = WhitepaperContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de whitepapers no disponible: {e}")
            self.whitepaper_analyzer = None
        
        # Analizador de contenido de casos de estudio
        try:
            from .case_study_analyzer import CaseStudyAnalyzer
            self.case_study_analyzer = CaseStudyAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de casos de estudio no disponible: {e}")
            self.case_study_analyzer = None
        
        # Analizador de contenido de propuestas
        try:
            from .proposal_content_analyzer import ProposalContentAnalyzer
            self.proposal_analyzer = ProposalContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de propuestas no disponible: {e}")
            self.proposal_analyzer = None
        
        # Analizador de contenido de informes
        try:
            from .report_content_analyzer import ReportContentAnalyzer
            self.report_analyzer = ReportContentAnalyzer()
        except Exception as e:
            logger.warning(f"Analizador de contenido de informes no disponible: {e}")
            self.report_analyzer = None

    async def add(
        self,
        content: str,
        addition: str,
        position: str = "end",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Agregar contenido al texto original.

        Args:
            content: Contenido original
            addition: Contenido a agregar
            position: Posición donde agregar (start, end, before, after, replace)
            context: Contexto adicional opcional

        Returns:
            Diccionario con el resultado de la operación
        """
        import time
        start_time = time.time()
        
        try:
            # Guardar estado para undo
            self.undo_redo.save_state(content, "add", {"addition": addition, "position": position})
            
            # Ejecutar hooks de plugins
            if self.plugin_manager:
                modified_addition = self.plugin_manager.execute_hook("before_add", content, addition, position=position)
                if modified_addition:
                    addition = modified_addition
            
            # Analizar contexto
            analysis = await self.analyzer.analyze(content, context)
            
            # Determinar posición óptima si no se especifica
            if position == "auto":
                ai_suggestion = await self.ai_engine.suggest_optimal_position(content, addition, context)
                position = ai_suggestion.get("position", "end")
                logger.info(f"IA sugirió posición: {position} - {ai_suggestion.get('reason', '')}")
            
            # Validar posición
            if position not in [p.value for p in Position] and not position.startswith(("before_", "after_", "specific_")):
                raise PositionError(f"Posición inválida: {position}")
            
            # Detectar formato y realizar adición apropiada
            content_format = self.formatter.detect_format(content)
            
            if content_format == ContentFormat.MARKDOWN.value:
                result_content = self.formatter.add_to_markdown(content, addition, position)
            elif content_format == ContentFormat.JSON.value:
                try:
                    addition_dict = json.loads(addition) if isinstance(addition, str) else addition
                    result_content = self.formatter.add_to_json(content, addition_dict, position)
                except:
                    result_content = await self._perform_addition_intelligent(content, addition, position, context)
            else:
                # Realizar adición con posicionamiento inteligente
                result_content = await self._perform_addition_intelligent(content, addition, position, context)
            
            # Validar resultado
            validation = await self.validator.validate_change(
                original=content,
                modified=result_content,
                operation="add"
            )
            
            # Validación semántica con IA
            semantic_validation = await self.ai_engine.validate_semantic_coherence(
                original=content,
                modified=result_content,
                operation="add"
            )
            validation["semantic"] = semantic_validation
            
            if not validation["valid"]:
                logger.warning(f"Validación fallida: {validation.get('reason', 'Unknown')}")
            
            # Registrar en historial
            change_record = {
                "operation": "add",
                "position": position,
                "addition": addition,
                "original_length": len(content),
                "new_length": len(result_content)
            }
            change_id = self.history.record_change(change_record)
            
            # Guardar en base de datos si está disponible
            if self.database:
                try:
                    self.database.save_operation(
                        operation_id=change_id,
                        operation_type="add",
                        content_before=content,
                        content_after=result_content,
                        metadata={"position": position, "addition": addition}
                    )
                except Exception as e:
                    logger.error(f"Error guardando en BD: {e}")
            
            # Crear versión si está disponible
            if self.versioning:
                try:
                    self.versioning.create_version(
                        content=result_content,
                        operation_id=change_id,
                        metadata={"operation": "add", "position": position}
                    )
                except Exception as e:
                    logger.error(f"Error creando versión: {e}")
            
            # Registrar métricas
            duration = time.time() - start_time
            self.metrics.record_operation(
                operation_type="add",
                duration=duration,
                success=True,
                content_length=len(content),
                result_length=len(result_content)
            )
            
            result = {
                "success": True,
                "content": result_content,
                "validation": validation,
                "change_id": change_record.get("id"),
                "metrics": {
                    "duration": duration,
                    "length_change": len(result_content) - len(content)
                }
            }
            
            # Ejecutar hooks de plugins después
            if self.plugin_manager:
                result = self.plugin_manager.execute_hook("after_add", result) or result
            
            # Enviar notificación
            if self.notifications:
                self.notifications.notify(
                    title="Contenido Agregado",
                    message=f"Se agregó contenido exitosamente. Longitud: {len(result_content)} caracteres",
                    notification_type=self.NotificationType.SUCCESS,
                    metadata={"operation": "add", "change_id": change_id}
                )
            
            # Publicar evento
            if self.event_bus:
                event = self.Event(
                    type=self.EventType.CONTENT_ADDED,
                    data={
                        "change_id": change_id,
                        "original_length": len(content),
                        "new_length": len(result_content),
                        "position": position
                    },
                    source="editor"
                )
                await self.event_bus.publish(event)
            
            return result
            
        except Exception as e:
            logger.error(f"Error al agregar contenido: {str(e)}")
            duration = time.time() - start_time
            self.metrics.record_operation(
                operation_type="add",
                duration=duration,
                success=False,
                content_length=len(content),
                result_length=len(content),
                error=str(e)
            )
            raise

    async def remove(
        self,
        content: str,
        pattern: Optional[str] = None,
        selector: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Eliminar contenido del texto original.

        Args:
            content: Contenido original
            pattern: Patrón o texto a eliminar
            selector: Selector específico (si aplica)
            context: Contexto adicional opcional

        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            # Analizar contexto
            analysis = await self.analyzer.analyze(content, context)
            
            # Identificar qué eliminar si no se especifica patrón
            if not pattern:
                ai_suggestions = await self.ai_engine.suggest_removals(content, context)
                if ai_suggestions and len(ai_suggestions) > 0:
                    # Usar la primera sugerencia
                    suggestion = ai_suggestions[0]
                    pattern = suggestion.get("text") or suggestion.get("pattern")
                    logger.info(f"IA sugirió eliminar: {pattern}")
                else:
                    pattern = await self.analyzer.suggest_removal(content, analysis)
            
            # Detectar formato y realizar eliminación apropiada
            content_format = self.formatter.detect_format(content)
            
            if content_format == ContentFormat.MARKDOWN.value:
                result_content = self.formatter.remove_from_markdown(content, pattern)
            elif content_format == ContentFormat.JSON.value:
                # Intentar parsear pattern como lista de claves
                try:
                    keys = json.loads(pattern) if pattern.startswith('[') else [pattern]
                    result_content = self.formatter.remove_from_json(content, keys)
                except:
                    result_content = self._perform_removal(content, pattern)
            else:
                # Realizar eliminación estándar
                result_content = self._perform_removal(content, pattern)
            
            # Validar resultado
            validation = await self.validator.validate_change(
                original=content,
                modified=result_content,
                operation="remove"
            )
            
            # Validación semántica con IA
            semantic_validation = await self.ai_engine.validate_semantic_coherence(
                original=content,
                modified=result_content,
                operation="remove"
            )
            validation["semantic"] = semantic_validation
            
            if not validation["valid"]:
                logger.warning(f"Validación fallida: {validation.get('reason', 'Unknown')}")
            
            # Registrar en historial
            change_record = {
                "operation": "remove",
                "pattern": pattern,
                "original_length": len(content),
                "new_length": len(result_content)
            }
            self.history.record_change(change_record)
            
            return {
                "success": True,
                "content": result_content,
                "validation": validation,
                "change_id": change_record.get("id")
            }
            
        except Exception as e:
            logger.error(f"Error al eliminar contenido: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "content": content
            }

    async def _perform_addition_intelligent(
        self,
        content: str,
        addition: str,
        position: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Realizar la adición física del contenido con posicionamiento inteligente"""
        import re
        
        if position == Position.START.value:
            return addition + "\n\n" + content
        elif position == Position.END.value:
            return content + "\n\n" + addition
        elif position == Position.BEFORE.value or position.startswith("before_"):
            # Buscar sección específica
            if position.startswith("before_"):
                section = position.replace("before_", "")
                pattern = rf"(?i)({re.escape(section)}.*?)(?=\n\n|\n#|$)"
                match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
                if match:
                    pos = match.start()
                    return content[:pos] + addition + "\n\n" + content[pos:]
            # Fallback: agregar al inicio
            return addition + "\n\n" + content
        elif position == Position.AFTER.value or position.startswith("after_"):
            # Buscar sección específica
            if position.startswith("after_"):
                section = position.replace("after_", "")
                pattern = rf"(?i)({re.escape(section)}.*?)(?=\n\n|\n#|$)"
                match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
                if match:
                    pos = match.end()
                    return content[:pos] + "\n\n" + addition + content[pos:]
            # Fallback: agregar al final
            return content + "\n\n" + addition
        elif position == Position.REPLACE.value:
            return addition
        elif position == "specific_position" and context:
            # Posición específica basada en contexto
            specific_pos = context.get("position_index", len(content))
            return content[:specific_pos] + addition + content[specific_pos:]
        else:
            # Por defecto, agregar al final
            return content + "\n\n" + addition
    
    def _perform_addition(self, content: str, addition: str, position: str) -> str:
        """Realizar la adición física del contenido (método síncrono para compatibilidad)"""
        if position == Position.START.value:
            return addition + "\n\n" + content
        elif position == Position.END.value:
            return content + "\n\n" + addition
        else:
            return content + "\n\n" + addition

    def _perform_removal(self, content: str, pattern: str) -> str:
        """Realizar la eliminación física del contenido"""
        # Eliminación simple por patrón
        result = content.replace(pattern, "")
        # Limpiar espacios múltiples
        import re
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        return result.strip()

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener historial de cambios"""
        return self.history.get_recent_changes(limit)
    
    async def batch_add(
        self,
        content: str,
        additions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Agregar múltiples elementos en una operación batch.

        Args:
            content: Contenido original
            additions: Lista de adiciones con sus posiciones
            context: Contexto adicional

        Returns:
            Resultado de la operación batch
        """
        result_content = content
        results = []
        
        for addition_data in additions:
            addition = addition_data.get("addition", "")
            position = addition_data.get("position", "end")
            
            result = await self.add(result_content, addition, position, context)
            if result.get("success"):
                result_content = result["content"]
                results.append(result)
            else:
                results.append(result)
        
        success = all(r.get("success", False) for r in results)
        
        if not success:
            raise BatchOperationError(f"Algunas operaciones batch fallaron: {len([r for r in results if not r.get('success')])} de {len(results)}")
        
        return {
            "success": success,
            "content": result_content,
            "operations": results,
            "total_operations": len(results),
            "successful_operations": len([r for r in results if r.get("success")])
        }
    
    async def batch_remove(
        self,
        content: str,
        patterns: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Eliminar múltiples elementos en una operación batch.

        Args:
            content: Contenido original
            patterns: Lista de patrones a eliminar
            context: Contexto adicional

        Returns:
            Resultado de la operación batch
        """
        result_content = content
        results = []
        
        for pattern in patterns:
            result = await self.remove(result_content, pattern, None, context)
            if result.get("success"):
                result_content = result["content"]
                results.append(result)
            else:
                results.append(result)
        
        return {
            "success": all(r.get("success", False) for r in results),
            "content": result_content,
            "operations": results
        }

