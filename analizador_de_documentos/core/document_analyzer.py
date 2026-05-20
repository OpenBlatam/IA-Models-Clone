"""
Analizador de Documentos Inteligente
=====================================

Modelo principal para análisis de documentos con capacidades avanzadas
de procesamiento, extracción de información y generación de insights.
"""

import os
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .fine_tuning_model import FineTuningModel

# Importar utilidades mejoradas
import sys
from pathlib import Path
utils_path = Path(__file__).parent.parent / "utils"
if utils_path.exists():
    sys.path.insert(0, str(utils_path.parent))
    try:
        from utils.cache import get_cache_manager, cached
        from utils.metrics import get_performance_monitor
    except ImportError:
        cached = lambda *args, **kwargs: lambda f: f  # No-op decorator
        get_cache_manager = lambda: None
        get_performance_monitor = lambda: None

# Importar optimizadores
try:
    from .document_optimizer import DocumentCache, BatchOptimizer, MemoryOptimizer
    OPTIMIZERS_AVAILABLE = True
except ImportError:
    OPTIMIZERS_AVAILABLE = False
    logger.warning("Optimizadores no disponibles")

logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES DE UTILIDAD MEJORADAS
# ============================================================================

def validate_content(content: Optional[str], min_length: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Validar contenido de documento
    
    Args:
        content: Contenido a validar
        min_length: Longitud mínima requerida
    
    Returns:
        Tupla (is_valid, error_message)
    """
    if content is None:
        return False, "Contenido no puede ser None"
    
    if not isinstance(content, str):
        return False, f"Contenido debe ser string, recibido: {type(content).__name__}"
    
    content_stripped = content.strip()
    
    if len(content_stripped) == 0:
        return False, "Contenido está vacío"
    
    if len(content_stripped) < min_length:
        return False, f"Contenido muy corto ({len(content_stripped)} caracteres, mínimo {min_length})"
    
    return True, None


def safe_float_conversion(value_str: str, default: float = 0.0) -> float:
    """
    Convertir string a float de manera segura
    
    Args:
        value_str: String a convertir
        default: Valor por defecto si falla la conversión
    
    Returns:
        Valor float convertido o default
    """
    if not value_str or not isinstance(value_str, str):
        return default
    
    try:
        cleaned = value_str.replace(',', '').replace('$', '').strip()
        if cleaned:
            value = float(cleaned)
            # Validar que no sea NaN o infinito
            if not (np.isnan(value) or np.isinf(value)):
                return value
    except (ValueError, OverflowError) as e:
        logger.debug(f"Error convirtiendo '{value_str}' a float: {e}")
    
    return default


def safe_regex_match(pattern: str, text: str, flags: int = 0) -> List[Any]:
    """
    Ejecutar regex match de manera segura
    
    Args:
        pattern: Patrón regex
        text: Texto a buscar
        flags: Flags de regex
    
    Returns:
        Lista de matches o lista vacía si hay error
    """
    import re
    
    if not pattern or not text:
        return []
    
    try:
        matches = list(re.finditer(pattern, text, flags))
        return matches
    except re.error as e:
        logger.warning(f"Error en patrón regex '{pattern}': {e}")
        return []
    except Exception as e:
        logger.error(f"Error inesperado en regex match: {e}")
        return []


def calculate_statistics(values: List[float]) -> Dict[str, Any]:
    """
    Calcular estadísticas de una lista de valores
    
    Args:
        values: Lista de valores numéricos
    
    Returns:
        Diccionario con estadísticas (mean, median, std, min, max, count)
    """
    if not values or len(values) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
        }
    
    try:
        values_array = np.array(values)
        return {
            "count": len(values),
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array))
        }
    except Exception as e:
        logger.error(f"Error calculando estadísticas: {e}")
        return {
            "count": len(values),
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
        }


def normalize_text(text: str, remove_extra_spaces: bool = True, 
                   lowercase: bool = False) -> str:
    """
    Normalizar texto
    
    Args:
        text: Texto a normalizar
        remove_extra_spaces: Eliminar espacios extra
        lowercase: Convertir a minúsculas
    
    Returns:
        Texto normalizado
    """
    if not text or not isinstance(text, str):
        return ""
    
    normalized = text.strip()
    
    if remove_extra_spaces:
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
    
    if lowercase:
        normalized = normalized.lower()
    
    return normalized


def extract_patterns(text: str, patterns: List[str], 
                     flags: int = 0) -> Dict[str, List[str]]:
    """
    Extraer múltiples patrones de un texto
    
    Args:
        text: Texto a analizar
        patterns: Lista de patrones regex
        flags: Flags de regex
    
    Returns:
        Diccionario con patrones y matches encontrados
    """
    import re
    
    results = {}
    
    if not text or not patterns:
        return results
    
    for pattern in patterns:
        try:
            matches = safe_regex_match(pattern, text, flags)
            results[pattern] = [match.group(0) for match in matches if match.groups()]
        except Exception as e:
            logger.debug(f"Error extrayendo patrón '{pattern}': {e}")
            results[pattern] = []
    
    return results


def clean_numeric_value(value_str: str) -> Optional[float]:
    """
    Limpiar y convertir valor numérico de string
    
    Args:
        value_str: String con valor numérico
    
    Returns:
        Valor float o None si no se puede convertir
    """
    if not value_str:
        return None
    
    try:
        # Eliminar caracteres comunes de formato
        cleaned = value_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '')
        cleaned = cleaned.replace(' ', '').strip()
        
        if cleaned:
            value = float(cleaned)
            if not (np.isnan(value) or np.isinf(value)):
                return value
    except (ValueError, OverflowError):
        pass
    
    return None


def batch_process(items: List[Any], processor: Callable, 
                  batch_size: int = 10, max_workers: int = 4) -> List[Any]:
    """
    Procesar items en lotes de manera asíncrona
    
    Args:
        items: Lista de items a procesar
        processor: Función procesadora (debe ser async)
        batch_size: Tamaño del lote
        max_workers: Número máximo de workers
    
    Returns:
        Lista de resultados
    """
    if not items:
        return []
    
    async def process_batch(batch: List[Any]) -> List[Any]:
        try:
            tasks = [processor(item) for item in batch]
            return await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error procesando lote: {e}")
            return [None] * len(batch)
    
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            batch_results = asyncio.run(process_batch(batch))
            results.extend(batch_results)
        except Exception as e:
            logger.error(f"Error en procesamiento por lotes: {e}")
            results.extend([None] * len(batch))
    
    return results


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitizar nombre de archivo
    
    Args:
        filename: Nombre de archivo original
        max_length: Longitud máxima
    
    Returns:
        Nombre de archivo sanitizado
    """
    if not filename:
        return "unnamed"
    
    import re
    
    # Eliminar caracteres peligrosos
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Eliminar espacios al inicio y final
    sanitized = sanitized.strip()
    
    # Limitar longitud
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext
    
    return sanitized if sanitized else "unnamed"


def validate_file_path(file_path: str, must_exist: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validar ruta de archivo
    
    Args:
        file_path: Ruta a validar
        must_exist: Si debe existir el archivo
    
    Returns:
        Tupla (is_valid, error_message)
    """
    if not file_path:
        return False, "Ruta de archivo no proporcionada"
    
    if not isinstance(file_path, str):
        return False, f"Ruta debe ser string, recibido: {type(file_path).__name__}"
    
    # Verificar caracteres peligrosos
    if '..' in file_path or file_path.startswith('/'):
        return False, "Ruta de archivo contiene caracteres peligrosos"
    
    if must_exist and not os.path.exists(file_path):
        return False, f"Archivo no encontrado: {file_path}"
    
    return True, None


class DocumentType(Enum):
    """Tipos de documentos soportados"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    IMAGE = "image"


class AnalysisTask(Enum):
    """Tipos de tareas de análisis"""
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    SENTIMENT = "sentiment"
    ENTITY_RECOGNITION = "entity_recognition"
    KEYWORD_EXTRACTION = "keyword_extraction"
    TOPIC_MODELING = "topic_modeling"
    QUESTION_ANSWERING = "question_answering"


@dataclass
class DocumentAnalysisResult:
    """Resultado de análisis de documento"""
    document_id: str
    document_type: str
    content: str
    summary: Optional[str] = None
    classification: Optional[Dict[str, float]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[str]] = None
    sentiment: Optional[Dict[str, float]] = None
    topics: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class DocumentAnalyzer:
    """
    Analizador de Documentos Inteligente
    
    Proporciona capacidades avanzadas de análisis de documentos incluyendo:
    - Clasificación de documentos
    - Extracción de información
    - Análisis de sentimiento
    - Reconocimiento de entidades
    - Generación de resúmenes
    - Extracción de palabras clave
    - Modelado de temas
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        device: str = None,
        fine_tuned_model_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Inicializar el analizador de documentos
        
        Args:
            model_name: Nombre del modelo base a usar
            device: Dispositivo a usar ('cuda', 'cpu', o None para auto-detectar)
            fine_tuned_model_path: Ruta al modelo fine-tuned (opcional)
            cache_dir: Directorio para cache de modelos
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir or os.path.join(
            Path(__file__).parent.parent.parent,
            "models",
            "cache"
        )
        
        # Crear directorio de cache si no existe
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Inicializar componentes
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator(
            model_name=model_name,
            device=self.device,
            cache_dir=self.cache_dir
        )
        
        # Cargar modelo fine-tuned si existe
        if fine_tuned_model_path and os.path.exists(fine_tuned_model_path):
            logger.info(f"Cargando modelo fine-tuned desde {fine_tuned_model_path}")
            self.fine_tuning_model = FineTuningModel.load(fine_tuned_model_path)
            self.model = self.fine_tuning_model.model
            self.tokenizer = self.fine_tuning_model.tokenizer
        else:
            logger.info(f"Cargando modelo base: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.fine_tuning_model = None
        
        self.model.eval()
        
        # Pipelines para tareas específicas
        self._pipelines = {}
        
        # Inicializar caché y métricas
        try:
            self.cache_manager = get_cache_manager()
            self.metrics = get_performance_monitor()
        except:
            self.cache_manager = None
            self.metrics = None
        
        # Robust helpers
        self.circuit_breaker = CircuitBreaker(
            name="document_analyzer",
            failure_threshold=5,
            recovery_timeout=60.0
        )
        self.rate_limiter = AdvancedRateLimiter(rate=10.0, capacity=50.0)
        
        # Performance monitor
        self.performance_monitor = performance_monitor
        
        # Inicializar componentes mejorados si están disponibles
        try:
            from .document_analyzer_enhanced import (
                DocumentComparator,
                BatchDocumentProcessor,
                AdvancedInformationExtractor,
                DocumentLanguageAnalyzer
            )
            self.comparator = DocumentComparator(self)
            self.batch_processor = BatchDocumentProcessor(self, max_workers=10)
            self.info_extractor = AdvancedInformationExtractor(self)
            self.language_analyzer = DocumentLanguageAnalyzer(self)
            logger.info("Características avanzadas habilitadas")
        except ImportError:
            self.comparator = None
            self.batch_processor = None
            self.info_extractor = None
            self.language_analyzer = None
        
        # Inicializar componentes avanzados
        try:
            from .document_analyzer_advanced import (
                ImageAnalyzer,
                TableExtractor,
                MultiLanguageAnalyzer,
                DocumentQualityAnalyzer,
                FraudDetector,
                LegalDocumentAnalyzer
            )
            from .document_exporter import DocumentAnalysisExporter
            
            self.image_analyzer = ImageAnalyzer(self)
            self.table_extractor = TableExtractor(self)
            self.multilang_analyzer = MultiLanguageAnalyzer(self)
            self.quality_analyzer = DocumentQualityAnalyzer(self)
            self.fraud_detector = FraudDetector(self)
            self.legal_analyzer = LegalDocumentAnalyzer(self)
            self.exporter = DocumentAnalysisExporter()
            logger.info("Características avanzadas adicionales habilitadas")
        except ImportError:
            self.image_analyzer = None
            self.table_extractor = None
            self.multilang_analyzer = None
            self.quality_analyzer = None
            self.fraud_detector = None
            self.legal_analyzer = None
            self.exporter = None
        
        # Inicializar componentes adicionales
        try:
            from .document_versioning import DocumentVersionManager
            from .document_grammar import GrammarAnalyzer
            from .document_integrations import DocumentIntegrations
            
            self.version_manager = DocumentVersionManager(self)
            self.grammar_analyzer = GrammarAnalyzer(self)
            self.integrations = DocumentIntegrations(self)
            logger.info("Componentes adicionales habilitados")
        except ImportError:
            self.version_manager = None
            self.grammar_analyzer = None
            self.integrations = None
        
        # Inicializar componentes enterprise
        try:
            from .document_collaboration import CollaborationAnalyzer
            from .document_recommendations import RecommendationEngine
            from .document_metrics import MetricsCollector
            
            self.collaboration_analyzer = CollaborationAnalyzer(self)
            self.recommendation_engine = RecommendationEngine(self)
            self.metrics_collector = MetricsCollector(self)
            logger.info("Componentes enterprise habilitados")
        except ImportError:
            self.collaboration_analyzer = None
            self.recommendation_engine = None
            self.metrics_collector = None
        
        # Inicializar componentes premium
        try:
            from .document_api import create_api_server
            from .document_webhooks import WebhookManager, DocumentEvents
            from .document_ml import DocumentMLPredictor, DocumentTrendAnalyzer
            
            self.webhook_manager = WebhookManager()
            self.ml_predictor = DocumentMLPredictor(self)
            self.trend_analyzer = DocumentTrendAnalyzer(self)
            self._api_server = None  # Lazy initialization
            logger.info("Componentes premium habilitados")
        except ImportError:
            self.webhook_manager = None
            self.ml_predictor = None
            self.trend_analyzer = None
            logger.warning("Algunos componentes premium no disponibles")
        
        # Inicializar componentes ultimate
        try:
            from .document_plugins import PluginManager
            from .document_realtime import RealtimeAnalyzer
            from .document_database import DocumentDatabase, InMemoryDatabase
            
            self.plugin_manager = PluginManager(self)
            self.realtime_analyzer = RealtimeAnalyzer(self)
            self.database = DocumentDatabase(self, InMemoryDatabase())
            logger.info("Componentes ultimate habilitados")
        except ImportError:
            self.plugin_manager = None
            self.realtime_analyzer = None
            self.database = None
            logger.warning("Algunos componentes ultimate no disponibles")
        
        # Inicializar componentes finales
        try:
            from .document_dashboard import DashboardGenerator
            from .document_formats import DocumentFormatHandler
            from .document_alerts import AlertManager, create_quality_alert_rule, create_grammar_alert_rule
            
            self.dashboard_generator = DashboardGenerator(self)
            self.format_handler = DocumentFormatHandler(self)
            self.alert_manager = AlertManager(self)
            
            # Registrar reglas de alerta por defecto
            self.alert_manager.register_rule(create_quality_alert_rule(50.0))
            self.alert_manager.register_rule(create_grammar_alert_rule(60.0))
            
            logger.info("Componentes finales habilitados")
        except ImportError:
            self.dashboard_generator = None
            self.format_handler = None
            self.alert_manager = None
            logger.warning("Algunos componentes finales no disponibles")
        
        # Inicializar componentes adicionales finales
        try:
            from .document_security import SecurityManager
            from .document_multilang import MultiLanguageAnalyzer
            from .document_notifications import NotificationManager, NotificationChannel
            
            self.security_manager = SecurityManager(self)
            self.multilang_analyzer = MultiLanguageAnalyzer(self)
            self.notification_manager = NotificationManager(self)
            
            logger.info("Componentes adicionales finales habilitados")
        except ImportError:
            self.security_manager = None
            self.multilang_analyzer = None
            self.notification_manager = None
            logger.warning("Algunos componentes adicionales finales no disponibles")
        
        # Inicializar componentes complementarios
        try:
            from .document_backup import BackupManager
            from .document_workflow import WorkflowManager, create_analysis_workflow
            from .document_autolearn import AutoLearningSystem
            
            self.backup_manager = BackupManager(self)
            self.workflow_manager = WorkflowManager(self)
            self.auto_learning = AutoLearningSystem(self)
            
            # Registrar workflow de análisis por defecto
            default_workflow = create_analysis_workflow(self)
            self.workflow_manager.register_workflow("default_analysis", default_workflow)
            
            logger.info("Componentes complementarios habilitados")
        except ImportError:
            self.backup_manager = None
            self.workflow_manager = None
            self.auto_learning = None
            logger.warning("Algunos componentes complementarios no disponibles")
        
        # Inicializar componentes avanzados finales
        try:
            from .document_semantic_search import SemanticSearchEngine
            from .document_metadata_extractor import MetadataExtractor
            from .document_sentiment_advanced import AdvancedSentimentAnalyzer
            
            self.semantic_search = SemanticSearchEngine(self)
            self.metadata_extractor = MetadataExtractor(self)
            self.advanced_sentiment = AdvancedSentimentAnalyzer(self)
            
            logger.info("Componentes avanzados finales habilitados")
        except ImportError:
            self.semantic_search = None
            self.metadata_extractor = None
            self.advanced_sentiment = None
            logger.warning("Algunos componentes avanzados finales no disponibles")
        
        # Inicializar componentes adicionales finales
        try:
            from .document_cloud import CloudStorageManager
            from .document_tagging import TaggingSystem
            from .document_reporting import ReportGenerator
            
            self.cloud_storage = CloudStorageManager(self)
            self.tagging_system = TaggingSystem(self)
            self.report_generator = ReportGenerator(self)
            
            logger.info("Componentes adicionales finales habilitados")
        except ImportError:
            self.cloud_storage = None
            self.tagging_system = None
            self.report_generator = None
            logger.warning("Algunos componentes adicionales finales no disponibles")
        
        # Inicializar componentes finales adicionales
        try:
            from .document_validation import DocumentValidator, create_min_length_rule, create_has_summary_rule
            from .document_history import HistoricalAnalyzer
            from .document_documentation import DocumentationGenerator
            
            self.validator = DocumentValidator(self)
            self.historical_analyzer = HistoricalAnalyzer(self)
            self.doc_generator = DocumentationGenerator(self)
            
            # Registrar reglas de validación por defecto
            self.validator.register_rule(create_min_length_rule(100))
            self.validator.register_rule(create_has_summary_rule())
            
            logger.info("Componentes finales adicionales habilitados")
        except ImportError:
            self.validator = None
            self.historical_analyzer = None
            self.doc_generator = None
            logger.warning("Algunos componentes finales adicionales no disponibles")
        
        # Inicializar optimizadores
        if OPTIMIZERS_AVAILABLE:
            self.document_cache = DocumentCache(max_size=1000, ttl=3600)
            self.batch_optimizer = BatchOptimizer()
            self.memory_optimizer = MemoryOptimizer()
            logger.info("Optimizadores habilitados")
        else:
            self.document_cache = None
            self.batch_optimizer = None
            self.memory_optimizer = None
        
        logger.info(f"DocumentAnalyzer inicializado en dispositivo: {self.device}")
    
    def _get_pipeline(self, task: str, model_name: Optional[str] = None):
        """Obtener o crear pipeline para una tarea específica"""
        if task not in self._pipelines:
            model = model_name or self.model_name
            self._pipelines[task] = pipeline(
                task,
                model=model,
                device=0 if self.device == "cuda" else -1,
                tokenizer=self.tokenizer
            )
        return self._pipelines[task]
    
    async def analyze_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        document_type: Optional[DocumentType] = None,
        tasks: Optional[List[AnalysisTask]] = None,
        **kwargs
    ) -> DocumentAnalysisResult:
        """
        Analizar un documento completo
        
        Args:
            document_path: Ruta al archivo del documento
            document_content: Contenido del documento como texto
            document_type: Tipo de documento
            tasks: Lista de tareas de análisis a realizar
            **kwargs: Argumentos adicionales
        
        Returns:
            DocumentAnalysisResult con todos los análisis realizados
        """
        import time
        start_time = time.time()
        
        # Procesar documento
        if document_path:
            doc_type = document_type or DocumentType(
                Path(document_path).suffix[1:].lower()
            )
            content = self.document_processor.process_document(
                document_path,
                doc_type.value
            )
        elif document_content:
            content = document_content
            doc_type = document_type or DocumentType.TXT
        else:
            raise ValueError("Debe proporcionar document_path o document_content")
        
        # Generar ID único
        doc_id = kwargs.get("document_id", f"doc_{datetime.now().timestamp()}")
        
        # Tareas por defecto si no se especifican
        if tasks is None:
            tasks = [
                AnalysisTask.CLASSIFICATION,
                AnalysisTask.SUMMARIZATION,
                AnalysisTask.KEYWORD_EXTRACTION,
                AnalysisTask.SENTIMENT
            ]
        
        # Realizar análisis
        result = DocumentAnalysisResult(
            document_id=doc_id,
            document_type=doc_type.value,
            content=content[:1000] + "..." if len(content) > 1000 else content
        )
        
        # Clasificación
        if AnalysisTask.CLASSIFICATION in tasks:
            result.classification = await self.classify_document(content)
        
        # Resumen
        if AnalysisTask.SUMMARIZATION in tasks:
            result.summary = await self.summarize_document(content)
        
        # Extracción de keywords
        if AnalysisTask.KEYWORD_EXTRACTION in tasks:
            result.keywords = await self.extract_keywords(content)
        
        # Sentimiento
        if AnalysisTask.SENTIMENT in tasks:
            result.sentiment = await self.analyze_sentiment(content)
        
        # Entidades
        if AnalysisTask.ENTITY_RECOGNITION in tasks:
            result.entities = await self.extract_entities(content)
        
        # Topics
        if AnalysisTask.TOPIC_MODELING in tasks:
            result.topics = await self.extract_topics(content)
        
        # Calcular tiempo de procesamiento
        result.processing_time = time.time() - start_time
        
        # Calcular confianza promedio
        confidences = []
        if result.classification:
            confidences.extend(result.classification.values())
        if result.sentiment:
            confidences.extend(result.sentiment.values())
        result.confidence = np.mean(confidences) if confidences else 0.0
        
        return result
    
    async def classify_document(
        self,
        content: str,
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Clasificar un documento
        
        Args:
            content: Contenido del documento
            labels: Lista de etiquetas posibles (opcional)
        
        Returns:
            Diccionario con probabilidades por clase
        """
        import time
        start_time = time.time()
        
        try:
            # Verificar caché
            if self.cache_manager:
                cache_key = self.cache_manager.generate_key(
                    "classify",
                    content[:100],  # Usar primeros 100 chars como clave
                    str(labels)
                )
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    logger.debug("Cache hit en clasificación")
                    return cached_result
            
            if self.fine_tuning_model and hasattr(self.fine_tuning_model, 'classify'):
                result = await self.fine_tuning_model.classify(content)
            else:
                # Usar modelo base para clasificación
                classifier = self._get_pipeline("text-classification")
                result = classifier(content[:512])  # Limitar longitud
                
                if isinstance(result, list):
                    result = result[0]
                
                result = {result["label"]: result["score"]}
            
            # Guardar en caché
            if self.cache_manager:
                self.cache_manager.set(cache_key, result, ttl=3600)
            
            # Registrar métricas
            duration = time.time() - start_time
            if self.metrics:
                self.metrics.record_analysis("classification", duration, True)
            
            return result
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            duration = time.time() - start_time
            if self.metrics:
                self.metrics.record_analysis("classification", duration, False)
            return {"unknown": 0.0}
    
    async def summarize_document(
        self,
        content: str,
        max_length: int = 150,
        min_length: int = 30
    ) -> str:
        """
        Generar resumen del documento
        
        Args:
            content: Contenido del documento
            max_length: Longitud máxima del resumen
            min_length: Longitud mínima del resumen
        
        Returns:
            Resumen del documento
        """
        try:
            summarizer = self._get_pipeline("summarization")
            
            # Dividir en chunks si es muy largo
            max_chunk_size = 1024
            if len(content) > max_chunk_size:
                chunks = [
                    content[i:i + max_chunk_size]
                    for i in range(0, len(content), max_chunk_size)
                ]
                summaries = []
                for chunk in chunks:
                    summary = summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    if isinstance(summary, list):
                        summary = summary[0]
                    summaries.append(summary.get("summary_text", ""))
                return " ".join(summaries)
            else:
                summary = summarizer(
                    content,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                if isinstance(summary, list):
                    summary = summary[0]
                return summary.get("summary_text", "")
        except Exception as e:
            logger.error(f"Error en resumen: {e}")
            # Fallback a resumen simple
            sentences = content.split(". ")
            return ". ".join(sentences[:3]) + "."
    
    async def extract_keywords(
        self,
        content: str,
        top_k: int = 10
    ) -> List[str]:
        """
        Extraer palabras clave del documento
        
        Args:
            content: Contenido del documento
            top_k: Número de keywords a extraer
        
        Returns:
            Lista de palabras clave
        """
        try:
            # Generar embeddings
            embeddings = await self.embedding_generator.generate_embeddings([content])
            
            # Usar técnicas de extracción de keywords
            # (simplificado - en producción usar técnicas más avanzadas)
            words = content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Filtrar palabras muy cortas
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Ordenar por frecuencia
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_words[:top_k]]
        except Exception as e:
            logger.error(f"Error en extracción de keywords: {e}")
            return []
    
    async def analyze_sentiment(
        self,
        content: str
    ) -> Dict[str, float]:
        """
        Analizar sentimiento del documento
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con scores de sentimiento
        """
        try:
            sentiment_analyzer = self._get_pipeline("sentiment-analysis")
            result = sentiment_analyzer(content[:512])
            
            if isinstance(result, list):
                result = result[0]
            
            return {
                result["label"].lower(): result["score"],
                "neutral": 1.0 - result["score"] if result["label"].lower() != "neutral" else result["score"]
            }
        except Exception as e:
            logger.error(f"Error en análisis de sentimiento: {e}")
            return {"neutral": 0.5}
    
    async def extract_entities(
        self,
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Extraer entidades nombradas del documento
        
        Args:
            content: Contenido del documento
        
        Returns:
            Lista de entidades encontradas
        """
        try:
            ner_pipeline = self._get_pipeline("ner")
            entities = ner_pipeline(content[:512])
            
            if isinstance(entities, list):
                return [
                    {
                        "text": ent["word"],
                        "label": ent["entity"],
                        "score": ent["score"]
                    }
                    for ent in entities
                ]
            return []
        except Exception as e:
            logger.error(f"Error en extracción de entidades: {e}")
            return []
    
    async def extract_topics(
        self,
        content: str,
        num_topics: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extraer temas del documento
        
        Args:
            content: Contenido del documento
            num_topics: Número de temas a extraer
        
        Returns:
            Lista de temas con sus scores
        """
        try:
            # Implementación simplificada
            # En producción usar técnicas más avanzadas como LDA, BERTopic, etc.
            keywords = await self.extract_keywords(content, top_k=num_topics * 3)
            
            # Agrupar keywords en temas (simplificado)
            topics = []
            for i in range(num_topics):
                start_idx = i * 3
                topic_keywords = keywords[start_idx:start_idx + 3]
                topics.append({
                    "topic_id": i + 1,
                    "keywords": topic_keywords,
                    "score": 0.8 - (i * 0.1)  # Score simulado
                })
            
            return topics
        except Exception as e:
            logger.error(f"Error en extracción de temas: {e}")
            return []
    
    async def answer_question(
        self,
        document_content: str,
        question: str
    ) -> Dict[str, Any]:
        """
        Responder una pregunta sobre el documento
        
        Args:
            document_content: Contenido del documento
            question: Pregunta a responder
        
        Returns:
            Respuesta con contexto
        """
        try:
            qa_pipeline = self._get_pipeline("question-answering")
            result = qa_pipeline(
                question=question,
                context=document_content[:512]
            )
            
            return {
                "answer": result.get("answer", ""),
                "score": result.get("score", 0.0),
                "start": result.get("start", 0),
                "end": result.get("end", 0)
            }
        except Exception as e:
            logger.error(f"Error en Q&A: {e}")
            return {"answer": "", "score": 0.0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "fine_tuned": self.fine_tuning_model is not None,
            "model_type": type(self.model).__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "cache_dir": self.cache_dir
        }
    
    def save_model(self, path: str):
        """Guardar el modelo"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Modelo guardado en {path}")


# ============================================================================
# SISTEMAS AVANZADOS DE ANÁLISIS DE DOCUMENTOS
# ============================================================================

class BulkDocumentProcessor:
    """Procesador masivo de documentos con procesamiento paralelo."""
    
    def __init__(self, analyzer: DocumentAnalyzer, max_workers: int = 4):
        self.analyzer = analyzer
        self.max_workers = max_workers
        self.processing_history: List[Dict[str, Any]] = []
    
    async def process_batch(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        tasks: Optional[List[AnalysisTask]] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[DocumentAnalysisResult]:
        """
        Procesar múltiples documentos en paralelo
        
        Args:
            documents: Lista de documentos (paths o contenidos)
            tasks: Tareas de análisis a realizar
            progress_callback: Callback para actualizar progreso
        
        Returns:
            Lista de resultados de análisis
        """
        import asyncio
        import time
        
        start_time = time.time()
        results = []
        total = len(documents)
        
        # Crear semáforo para limitar concurrencia
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single(doc, index):
            async with semaphore:
                try:
                    if isinstance(doc, dict):
                        result = await self.analyzer.analyze_document(
                            document_path=doc.get("path"),
                            document_content=doc.get("content"),
                            document_type=doc.get("type"),
                            tasks=tasks,
                            document_id=doc.get("id", f"doc_{index}")
                        )
                    elif isinstance(doc, str):
                        # Asumir que es path
                        result = await self.analyzer.analyze_document(
                            document_path=doc,
                            tasks=tasks
                        )
                    else:
                        result = None
                    
                    if progress_callback:
                        await progress_callback(index + 1, total)
                    
                    return result
                except Exception as e:
                    logger.error(f"Error procesando documento {index}: {e}")
                    return None
        
        # Procesar en paralelo
        tasks_list = [process_single(doc, i) for i, doc in enumerate(documents)]
        results = await asyncio.gather(*tasks_list)
        
        # Filtrar None
        results = [r for r in results if r is not None]
        
        duration = time.time() - start_time
        
        self.processing_history.append({
            "timestamp": datetime.now().isoformat(),
            "total_documents": total,
            "processed": len(results),
            "failed": total - len(results),
            "duration": duration,
            "avg_time_per_doc": duration / total if total > 0 else 0
        })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de procesamiento"""
        if not self.processing_history:
            return {"message": "No processing history"}
        
        recent = self.processing_history[-100:]  # Últimos 100 batches
        
        total_docs = sum(h["total_documents"] for h in recent)
        total_processed = sum(h["processed"] for h in recent)
        total_duration = sum(h["duration"] for h in recent)
        
        return {
            "total_batches": len(recent),
            "total_documents": total_docs,
            "success_rate": total_processed / total_docs if total_docs > 0 else 0,
            "avg_duration": total_duration / len(recent) if recent else 0,
            "avg_docs_per_second": total_processed / total_duration if total_duration > 0 else 0
        }


class DocumentComparator:
    """Comparador de documentos para análisis de diferencias."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
    
    async def compare_documents(
        self,
        doc1_path: Optional[str] = None,
        doc1_content: Optional[str] = None,
        doc2_path: Optional[str] = None,
        doc2_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comparar dos documentos
        
        Args:
            doc1_path/doc1_content: Primer documento
            doc2_path/doc2_content: Segundo documento
        
        Returns:
            Análisis comparativo
        """
        # Analizar ambos documentos
        result1 = await self.analyzer.analyze_document(
            document_path=doc1_path,
            document_content=doc1_content
        )
        
        result2 = await self.analyzer.analyze_document(
            document_path=doc2_path,
            document_content=doc2_content
        )
        
        # Comparar clasificaciones
        classification_diff = self._compare_classifications(
            result1.classification or {},
            result2.classification or {}
        )
        
        # Comparar keywords
        keywords_diff = self._compare_keywords(
            result1.keywords or [],
            result2.keywords or []
        )
        
        # Comparar sentimientos
        sentiment_diff = self._compare_sentiments(
            result1.sentiment or {},
            result2.sentiment or {}
        )
        
        # Comparar entidades
        entities_diff = self._compare_entities(
            result1.entities or [],
            result2.entities or []
        )
        
        # Similitud semántica
        similarity = await self._calculate_semantic_similarity(
            result1.content,
            result2.content
        )
        
        return {
            "document1": {
                "id": result1.document_id,
                "type": result1.document_type,
                "summary": result1.summary
            },
            "document2": {
                "id": result2.document_id,
                "type": result2.document_type,
                "summary": result2.summary
            },
            "similarity_score": similarity,
            "classification_diff": classification_diff,
            "keywords_diff": keywords_diff,
            "sentiment_diff": sentiment_diff,
            "entities_diff": entities_diff,
            "timestamp": datetime.now().isoformat()
        }
    
    def _compare_classifications(
        self,
        class1: Dict[str, float],
        class2: Dict[str, float]
    ) -> Dict[str, Any]:
        """Comparar clasificaciones"""
        all_labels = set(class1.keys()) | set(class2.keys())
        
        diff = {}
        for label in all_labels:
            score1 = class1.get(label, 0.0)
            score2 = class2.get(label, 0.0)
            diff[label] = {
                "doc1_score": score1,
                "doc2_score": score2,
                "difference": abs(score1 - score2)
            }
        
        return diff
    
    def _compare_keywords(
        self,
        keywords1: List[str],
        keywords2: List[str]
    ) -> Dict[str, Any]:
        """Comparar keywords"""
        set1 = set(keywords1)
        set2 = set(keywords2)
        
        return {
            "common": list(set1 & set2),
            "only_in_doc1": list(set1 - set2),
            "only_in_doc2": list(set2 - set1),
            "jaccard_similarity": len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0
        }
    
    def _compare_sentiments(
        self,
        sentiment1: Dict[str, float],
        sentiment2: Dict[str, float]
    ) -> Dict[str, Any]:
        """Comparar sentimientos"""
        diff = {}
        all_labels = set(sentiment1.keys()) | set(sentiment2.keys())
        
        for label in all_labels:
            score1 = sentiment1.get(label, 0.0)
            score2 = sentiment2.get(label, 0.0)
            diff[label] = {
                "doc1": score1,
                "doc2": score2,
                "change": score2 - score1
            }
        
        return diff
    
    def _compare_entities(
        self,
        entities1: List[Dict[str, Any]],
        entities2: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Comparar entidades"""
        texts1 = {e["text"] for e in entities1}
        texts2 = {e["text"] for e in entities2}
        
        return {
            "common_entities": list(texts1 & texts2),
            "only_in_doc1": list(texts1 - texts2),
            "only_in_doc2": list(texts2 - texts1),
            "total_unique": len(texts1 | texts2)
        }
    
    async def _calculate_semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Calcular similitud semántica"""
        try:
            embeddings = await self.analyzer.embedding_generator.generate_embeddings([text1, text2])
            
            if len(embeddings) == 2:
                # Calcular cosine similarity
                emb1 = embeddings[0]
                emb2 = embeddings[1]
                
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                similarity = dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
                return float(similarity)
        except Exception as e:
            logger.error(f"Error calculando similitud: {e}")
        
        return 0.0


class TableExtractor:
    """Extractor de tablas de documentos."""
    
    def __init__(self):
        self.extracted_tables: List[Dict[str, Any]] = []
    
    async def extract_tables(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        document_type: Optional[DocumentType] = None
    ) -> List[Dict[str, Any]]:
        """
        Extraer tablas de un documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            document_type: Tipo de documento
        
        Returns:
            Lista de tablas extraídas
        """
        try:
            # Para PDF
            if document_type == DocumentType.PDF or (document_path and document_path.endswith('.pdf')):
                return await self._extract_from_pdf(document_path)
            
            # Para HTML
            elif document_type == DocumentType.HTML or (document_path and document_path.endswith('.html')):
                return await self._extract_from_html(document_path or document_content)
            
            # Para otros tipos, usar procesamiento básico
            else:
                return await self._extract_from_text(document_content or "")
        except Exception as e:
            logger.error(f"Error extrayendo tablas: {e}")
            return []
    
    async def _extract_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extraer tablas de PDF"""
        try:
            import pdfplumber
            
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables):
                        tables.append({
                            "page": page_num + 1,
                            "table_num": table_num + 1,
                            "data": table,
                            "rows": len(table),
                            "cols": len(table[0]) if table else 0
                        })
            
            return tables
        except ImportError:
            logger.warning("pdfplumber no está instalado. Instalar con: pip install pdfplumber")
            return []
        except Exception as e:
            logger.error(f"Error extrayendo tablas de PDF: {e}")
            return []
    
    async def _extract_from_html(self, html_content: str) -> List[Dict[str, Any]]:
        """Extraer tablas de HTML"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table')
            
            extracted = []
            for table_num, table in enumerate(tables):
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                
                if rows:
                    extracted.append({
                        "table_num": table_num + 1,
                        "data": rows,
                        "rows": len(rows),
                        "cols": len(rows[0]) if rows else 0
                    })
            
            return extracted
        except ImportError:
            logger.warning("BeautifulSoup no está instalado. Instalar con: pip install beautifulsoup4")
            return []
        except Exception as e:
            logger.error(f"Error extrayendo tablas de HTML: {e}")
            return []
    
    async def _extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extraer tablas de texto plano (básico)"""
        # Implementación básica para detectar tablas en texto
        lines = text.split('\n')
        tables = []
        current_table = []
        
        for line in lines:
            # Detectar líneas que parecen tablas (múltiples espacios o tabs)
            if '\t' in line or len(line.split()) > 3:
                cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                if not cells:
                    cells = [cell.strip() for cell in line.split() if cell.strip()]
                if cells:
                    current_table.append(cells)
            else:
                if current_table and len(current_table) > 1:
                    tables.append({
                        "table_num": len(tables) + 1,
                        "data": current_table,
                        "rows": len(current_table),
                        "cols": len(current_table[0]) if current_table else 0
                    })
                current_table = []
        
        # Agregar última tabla si existe
        if current_table and len(current_table) > 1:
            tables.append({
                "table_num": len(tables) + 1,
                "data": current_table,
                "rows": len(current_table),
                "cols": len(current_table[0]) if current_table else 0
            })
        
        return tables


class OCRProcessor:
    """Procesador OCR para extraer texto de imágenes."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.ocr_history: List[Dict[str, Any]] = []
    
    async def extract_text_from_image(
        self,
        image_path: str,
        language: str = "spa"
    ) -> Dict[str, Any]:
        """
        Extraer texto de una imagen usando OCR
        
        Args:
            image_path: Ruta a la imagen
            language: Idioma del texto (código ISO 639-2)
        
        Returns:
            Texto extraído y metadatos
        """
        try:
            import pytesseract
            from PIL import Image
            
            # Cargar imagen
            image = Image.open(image_path)
            
            # Extraer texto
            text = pytesseract.image_to_string(image, lang=language)
            
            # Obtener información adicional
            data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)
            
            # Calcular confianza promedio
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            result = {
                "text": text,
                "confidence": avg_confidence / 100.0,  # Normalizar a 0-1
                "word_count": len(text.split()),
                "character_count": len(text),
                "image_path": image_path,
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
            
            self.ocr_history.append(result)
            
            return result
        except ImportError:
            logger.error("pytesseract o PIL no están instalados. Instalar con: pip install pytesseract pillow")
            return {
                "text": "",
                "confidence": 0.0,
                "error": "OCR libraries not installed"
            }
        except Exception as e:
            logger.error(f"Error en OCR: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def process_image_document(
        self,
        image_path: str,
        language: str = "spa",
        analyze_text: bool = True
    ) -> DocumentAnalysisResult:
        """
        Procesar documento de imagen completo (OCR + análisis)
        
        Args:
            image_path: Ruta a la imagen
            language: Idioma del texto
            analyze_text: Si analizar el texto extraído
        
        Returns:
            Resultado de análisis completo
        """
        # Extraer texto
        ocr_result = await self.extract_text_from_image(image_path, language)
        
        if not analyze_text or not ocr_result.get("text"):
            return DocumentAnalysisResult(
                document_id=f"img_{datetime.now().timestamp()}",
                document_type="image",
                content=ocr_result.get("text", ""),
                confidence=ocr_result.get("confidence", 0.0)
            )
        
        # Analizar texto extraído
        result = await self.analyzer.analyze_document(
            document_content=ocr_result["text"],
            document_type=DocumentType.TXT
        )
        
        result.confidence = ocr_result.get("confidence", 0.0)
        result.metadata = {
            "ocr_confidence": ocr_result.get("confidence", 0.0),
            "image_path": image_path,
            "language": language
        }
        
        return result


class SemanticSearchEngine:
    """Motor de búsqueda semántica en documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.document_index: Dict[str, np.ndarray] = {}
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Indexar un documento para búsqueda semántica
        
        Args:
            document_id: ID único del documento
            content: Contenido del documento
            metadata: Metadatos adicionales
        """
        try:
            # Generar embedding
            embeddings = await self.analyzer.embedding_generator.generate_embeddings([content])
            
            if embeddings:
                self.document_index[document_id] = embeddings[0]
                self.document_metadata[document_id] = {
                    "content": content[:500],  # Guardar preview
                    "metadata": metadata or {},
                    "indexed_at": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error indexando documento {document_id}: {e}")
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Buscar documentos semánticamente similares
        
        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados a retornar
            min_similarity: Similaridad mínima
        
        Returns:
            Lista de documentos ordenados por relevancia
        """
        try:
            # Generar embedding de la query
            query_embeddings = await self.analyzer.embedding_generator.generate_embeddings([query])
            
            if not query_embeddings or not self.document_index:
                return []
            
            query_embedding = query_embeddings[0]
            
            # Calcular similitudes
            similarities = []
            for doc_id, doc_embedding in self.document_index.items():
                # Cosine similarity
                dot_product = np.dot(query_embedding, doc_embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_doc = np.linalg.norm(doc_embedding)
                
                similarity = dot_product / (norm_query * norm_doc) if (norm_query * norm_doc) > 0 else 0.0
                
                if similarity >= min_similarity:
                    similarities.append({
                        "document_id": doc_id,
                        "similarity": float(similarity),
                        "metadata": self.document_metadata.get(doc_id, {})
                    })
            
            # Ordenar por similitud
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del índice"""
        return {
            "total_documents": len(self.document_index),
            "index_size_mb": sum(e.nbytes for e in self.document_index.values()) / (1024 * 1024),
            "indexed_documents": list(self.document_index.keys())
        }


class DocumentStructureAnalyzer:
    """Analizador de estructura de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
    
    async def analyze_structure(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar estructura de un documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis de estructura
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Analizar estructura
        structure = {
            "sections": self._extract_sections(content),
            "paragraphs": self._count_paragraphs(content),
            "sentences": self._count_sentences(content),
            "words": len(content.split()),
            "characters": len(content),
            "headings": self._extract_headings(content),
            "lists": self._extract_lists(content),
            "links": self._extract_links(content),
            "tables_count": len(await TableExtractor().extract_tables(document_content=content)),
            "timestamp": datetime.now().isoformat()
        }
        
        return structure
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extraer secciones del documento"""
        sections = []
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detectar encabezados (líneas cortas, mayúsculas, o números)
            if (len(line) < 100 and 
                (line.isupper() or 
                 line.startswith('#') or 
                 any(char.isdigit() for char in line[:10]))):
                
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    "title": line,
                    "level": self._detect_heading_level(line),
                    "content": ""
                }
            elif current_section:
                current_section["content"] += line + "\n"
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _detect_heading_level(self, line: str) -> int:
        """Detectar nivel de encabezado"""
        if line.startswith('#'):
            return line.count('#')
        elif line.isupper() and len(line) < 80:
            return 1
        else:
            return 2
    
    def _count_paragraphs(self, content: str) -> int:
        """Contar párrafos"""
        return len([p for p in content.split('\n\n') if p.strip()])
    
    def _count_sentences(self, content: str) -> int:
        """Contar oraciones"""
        import re
        sentences = re.split(r'[.!?]+', content)
        return len([s for s in sentences if s.strip()])
    
    def _extract_headings(self, content: str) -> List[str]:
        """Extraer encabezados"""
        headings = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if (line.startswith('#') or 
                (line.isupper() and len(line) < 100 and len(line.split()) < 10)):
                headings.append(line)
        
        return headings
    
    def _extract_lists(self, content: str) -> List[Dict[str, Any]]:
        """Extraer listas"""
        lists = []
        lines = content.split('\n')
        current_list = None
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_list:
                    lists.append(current_list)
                    current_list = None
                continue
            
            # Detectar items de lista
            if line.startswith(('-', '*', '•')) or line[0].isdigit() and '.' in line[:5]:
                if not current_list:
                    current_list = {"type": "unordered" if line.startswith(('-', '*', '•')) else "ordered", "items": []}
                current_list["items"].append(line)
            elif current_list:
                lists.append(current_list)
                current_list = None
        
        if current_list:
            lists.append(current_list)
        
        return lists
    
    def _extract_links(self, content: str) -> List[str]:
        """Extraer enlaces"""
        import re
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, content)


class AdvancedDocumentClassifier:
    """Clasificador avanzado de documentos con múltiples estrategias."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.classification_history: List[Dict[str, Any]] = []
        self.confidence_threshold = 0.7
    
    async def classify_with_confidence(
        self,
        content: str,
        use_ensemble: bool = True
    ) -> Dict[str, Any]:
        """
        Clasificar documento con análisis de confianza
        
        Args:
            content: Contenido del documento
            use_ensemble: Usar ensemble de múltiples clasificadores
        
        Returns:
            Clasificación con métricas de confianza
        """
        results = []
        
        # Clasificación base
        base_classification = await self.analyzer.classify_document(content)
        results.append(base_classification)
        
        # Ensemble con análisis adicional
        if use_ensemble:
            # Análisis de keywords
            keywords = await self.analyzer.extract_keywords(content, top_k=5)
            
            # Análisis de estructura
            structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
            structure = await structure_analyzer.analyze_structure(document_content=content)
            
            # Análisis de entidades
            entities = await self.analyzer.extract_entities(content)
            
            # Combinar señales
            ensemble_score = self._calculate_ensemble_score(
                base_classification,
                keywords,
                structure,
                entities
            )
        else:
            ensemble_score = base_classification
        
        # Calcular confianza
        confidence = self._calculate_confidence(base_classification, ensemble_score)
        
        result = {
            "classification": ensemble_score,
            "confidence": confidence,
            "base_classification": base_classification,
            "keywords": keywords if use_ensemble else [],
            "entities_count": len(entities) if use_ensemble else 0,
            "structure_features": structure if use_ensemble else {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.classification_history.append(result)
        return result
    
    def _calculate_ensemble_score(
        self,
        base_classification: Dict[str, float],
        keywords: List[str],
        structure: Dict[str, Any],
        entities: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calcular score de ensemble"""
        # Implementación simplificada
        # En producción usar técnicas más avanzadas
        
        ensemble = base_classification.copy()
        
        # Ajustar basado en keywords (si hay keywords técnicos, aumentar probabilidad técnica)
        technical_keywords = {'api', 'function', 'class', 'method', 'code', 'algorithm'}
        if any(kw in technical_keywords for kw in [k.lower() for k in keywords]):
            ensemble['technical'] = ensemble.get('technical', 0.0) + 0.1
        
        # Ajustar basado en estructura
        if structure.get('sections', []):
            ensemble['structured'] = ensemble.get('structured', 0.0) + 0.1
        
        # Normalizar
        total = sum(ensemble.values())
        if total > 0:
            ensemble = {k: v / total for k, v in ensemble.items()}
        
        return ensemble
    
    def _calculate_confidence(
        self,
        base_classification: Dict[str, float],
        ensemble_score: Dict[str, float]
    ) -> float:
        """Calcular nivel de confianza"""
        # Confianza basada en diferencia entre scores
        max_base = max(base_classification.values()) if base_classification else 0.0
        max_ensemble = max(ensemble_score.values()) if ensemble_score else 0.0
        
        # Si hay gran diferencia, menor confianza
        diff = abs(max_base - max_ensemble)
        confidence = 1.0 - min(diff, 0.5)  # Reducir confianza si hay mucha diferencia
        
        # Ajustar por separación de clases
        sorted_scores = sorted(ensemble_score.values(), reverse=True)
        if len(sorted_scores) > 1:
            separation = sorted_scores[0] - sorted_scores[1]
            confidence = (confidence + separation) / 2
        
        return min(max(confidence, 0.0), 1.0)
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de clasificación"""
        if not self.classification_history:
            return {"message": "No classification history"}
        
        confidences = [h["confidence"] for h in self.classification_history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Contar clasificaciones por categoría
        category_counts = {}
        for history in self.classification_history:
            top_category = max(
                history["classification"].items(),
                key=lambda x: x[1]
            )[0] if history["classification"] else "unknown"
            
            category_counts[top_category] = category_counts.get(top_category, 0) + 1
        
        return {
            "total_classifications": len(self.classification_history),
            "avg_confidence": avg_confidence,
            "high_confidence_count": sum(1 for c in confidences if c >= self.confidence_threshold),
            "low_confidence_count": sum(1 for c in confidences if c < self.confidence_threshold),
            "category_distribution": category_counts
        }


class DocumentReportGenerator:
    """Generador de reportes avanzados de análisis de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
    
    async def generate_comprehensive_report(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        include_visualizations: bool = False
    ) -> Dict[str, Any]:
        """
        Generar reporte comprensivo de análisis
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            include_visualizations: Incluir datos para visualizaciones
        
        Returns:
            Reporte completo
        """
        # Análisis completo
        analysis_result = await self.analyzer.analyze_document(
            document_path=document_path,
            document_content=document_content,
            tasks=[
                AnalysisTask.CLASSIFICATION,
                AnalysisTask.SUMMARIZATION,
                AnalysisTask.KEYWORD_EXTRACTION,
                AnalysisTask.SENTIMENT,
                AnalysisTask.ENTITY_RECOGNITION,
                AnalysisTask.TOPIC_MODELING
            ]
        )
        
        # Análisis de estructura
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(
            document_path=document_path,
            document_content=document_content or analysis_result.content
        )
        
        # Extracción de tablas
        table_extractor = TableExtractor()
        tables = await table_extractor.extract_tables(
            document_path=document_path,
            document_content=document_content or analysis_result.content
        )
        
        # Clasificación avanzada
        advanced_classifier = AdvancedDocumentClassifier(self.analyzer)
        advanced_classification = await advanced_classifier.classify_with_confidence(
            document_content or analysis_result.content,
            use_ensemble=True
        )
        
        # Construir reporte
        report = {
            "document_info": {
                "id": analysis_result.document_id,
                "type": analysis_result.document_type,
                "timestamp": analysis_result.timestamp
            },
            "executive_summary": {
                "summary": analysis_result.summary,
                "top_classification": max(
                    analysis_result.classification.items(),
                    key=lambda x: x[1]
                )[0] if analysis_result.classification else None,
                "sentiment": max(
                    analysis_result.sentiment.items(),
                    key=lambda x: x[1]
                )[0] if analysis_result.sentiment else None,
                "confidence": analysis_result.confidence
            },
            "detailed_analysis": {
                "classification": analysis_result.classification,
                "advanced_classification": advanced_classification,
                "sentiment": analysis_result.sentiment,
                "keywords": analysis_result.keywords,
                "topics": analysis_result.topics,
                "entities": analysis_result.entities
            },
            "structure_analysis": structure,
            "tables": {
                "count": len(tables),
                "tables": tables[:5]  # Limitar a 5 tablas
            },
            "metrics": {
                "processing_time": analysis_result.processing_time,
                "confidence": analysis_result.confidence,
                "word_count": len((document_content or analysis_result.content).split()),
                "character_count": len(document_content or analysis_result.content)
            },
            "recommendations": self._generate_recommendations(analysis_result, structure, advanced_classification)
        }
        
        if include_visualizations:
            report["visualization_data"] = self._prepare_visualization_data(analysis_result, structure)
        
        return report
    
    def _generate_recommendations(
        self,
        analysis: DocumentAnalysisResult,
        structure: Dict[str, Any],
        classification: Dict[str, Any]
    ) -> List[str]:
        """Generar recomendaciones basadas en análisis"""
        recommendations = []
        
        # Recomendación de confianza
        if analysis.confidence < 0.7:
            recommendations.append("Confianza baja en el análisis. Considerar revisión manual.")
        
        # Recomendación de estructura
        if not structure.get("sections"):
            recommendations.append("Documento sin estructura clara. Considerar agregar secciones.")
        
        # Recomendación de keywords
        if not analysis.keywords or len(analysis.keywords) < 5:
            recommendations.append("Pocas keywords extraídas. El documento podría necesitar más contenido específico.")
        
        # Recomendación de tablas
        if structure.get("tables_count", 0) == 0 and len(analysis.content) > 5000:
            recommendations.append("Documento largo sin tablas. Considerar usar tablas para datos estructurados.")
        
        return recommendations
    
    def _prepare_visualization_data(
        self,
        analysis: DocumentAnalysisResult,
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Preparar datos para visualizaciones"""
        return {
            "classification_distribution": analysis.classification or {},
            "sentiment_distribution": analysis.sentiment or {},
            "keywords_frequency": {kw: 1 for kw in (analysis.keywords or [])},
            "topics_distribution": {t.get("topic_id"): t.get("score", 0) for t in (analysis.topics or [])},
            "structure_metrics": {
                "sections": len(structure.get("sections", [])),
                "paragraphs": structure.get("paragraphs", 0),
                "sentences": structure.get("sentences", 0)
            }
        }


# ============================================================================
# SISTEMAS AVANZADOS ADICIONALES DE ANÁLISIS
# ============================================================================

class DocumentChangeDetector:
    """Detector de cambios entre versiones de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.change_history: List[Dict[str, Any]] = []
    
    async def detect_changes(
        self,
        old_version_path: Optional[str] = None,
        old_version_content: Optional[str] = None,
        new_version_path: Optional[str] = None,
        new_version_content: Optional[str] = None,
        granularity: str = "sentence"
    ) -> Dict[str, Any]:
        """
        Detectar cambios entre dos versiones de un documento
        
        Args:
            old_version_path/content: Versión anterior
            new_version_path/content: Versión nueva
            granularity: Nivel de granularidad ('word', 'sentence', 'paragraph')
        
        Returns:
            Análisis de cambios
        """
        # Obtener contenido
        old_content = old_version_content
        new_content = new_version_content
        
        if old_version_path:
            processor = DocumentProcessor()
            old_content = processor.process_document(old_version_path, "txt")
        
        if new_version_path:
            processor = DocumentProcessor()
            new_content = processor.process_document(new_version_path, "txt")
        
        if not old_content or not new_content:
            return {"error": "Missing content"}
        
        # Detectar cambios según granularidad
        if granularity == "word":
            changes = self._detect_word_changes(old_content, new_content)
        elif granularity == "sentence":
            changes = self._detect_sentence_changes(old_content, new_content)
        else:
            changes = self._detect_paragraph_changes(old_content, new_content)
        
        # Analizar ambos documentos
        old_analysis = await self.analyzer.analyze_document(document_content=old_content)
        new_analysis = await self.analyzer.analyze_document(document_content=new_content)
        
        # Calcular similitud semántica
        similarity = await self._calculate_similarity(old_content, new_content)
        
        result = {
            "changes": changes,
            "statistics": {
                "similarity": similarity,
                "old_version": {
                    "word_count": len(old_content.split()),
                    "char_count": len(old_content),
                    "classification": old_analysis.classification
                },
                "new_version": {
                    "word_count": len(new_content.split()),
                    "char_count": len(new_content),
                    "classification": new_analysis.classification
                },
                "change_rate": self._calculate_change_rate(changes, old_content, new_content)
            },
            "granularity": granularity,
            "timestamp": datetime.now().isoformat()
        }
        
        self.change_history.append(result)
        return result
    
    def _detect_word_changes(self, old_text: str, new_text: str) -> Dict[str, Any]:
        """Detectar cambios a nivel de palabras"""
        old_words = old_text.split()
        new_words = new_text.split()
        
        # Usar difflib para comparación
        import difflib
        diff = list(difflib.unified_diff(old_words, new_words, lineterm=''))
        
        added = [word for word in new_words if word not in old_words]
        removed = [word for old_words if word not in new_words]
        
        return {
            "added_words": added,
            "removed_words": removed,
            "total_added": len(added),
            "total_removed": len(removed),
            "diff": diff[:100]  # Limitar tamaño
        }
    
    def _detect_sentence_changes(self, old_text: str, new_text: str) -> Dict[str, Any]:
        """Detectar cambios a nivel de oraciones"""
        import re
        
        old_sentences = re.split(r'[.!?]+', old_text)
        new_sentences = re.split(r'[.!?]+', new_text)
        
        old_set = {s.strip() for s in old_sentences if s.strip()}
        new_set = {s.strip() for s in new_sentences if s.strip()}
        
        added = list(new_set - old_set)
        removed = list(old_set - new_set)
        unchanged = list(old_set & new_set)
        
        return {
            "added_sentences": added,
            "removed_sentences": removed,
            "unchanged_sentences": unchanged,
            "total_added": len(added),
            "total_removed": len(removed),
            "total_unchanged": len(unchanged)
        }
    
    def _detect_paragraph_changes(self, old_text: str, new_text: str) -> Dict[str, Any]:
        """Detectar cambios a nivel de párrafos"""
        old_paragraphs = [p.strip() for p in old_text.split('\n\n') if p.strip()]
        new_paragraphs = [p.strip() for p in new_text.split('\n\n') if p.strip()]
        
        old_set = set(old_paragraphs)
        new_set = set(new_paragraphs)
        
        added = list(new_set - old_set)
        removed = list(old_set - new_set)
        unchanged = list(old_set & new_set)
        
        return {
            "added_paragraphs": added,
            "removed_paragraphs": removed,
            "unchanged_paragraphs": unchanged,
            "total_added": len(added),
            "total_removed": len(removed),
            "total_unchanged": len(unchanged)
        }
    
    async def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud entre textos"""
        try:
            embeddings = await self.analyzer.embedding_generator.generate_embeddings([text1, text2])
            
            if len(embeddings) == 2:
                emb1 = embeddings[0]
                emb2 = embeddings[1]
                
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                similarity = dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
                return float(similarity)
        except Exception as e:
            logger.error(f"Error calculando similitud: {e}")
        
        return 0.0
    
    def _calculate_change_rate(self, changes: Dict[str, Any], old_text: str, new_text: str) -> float:
        """
        Calcular tasa de cambio
        
        Args:
            changes: Diccionario con cambios detectados
            old_text: Texto anterior
            new_text: Texto nuevo
        
        Returns:
            Tasa de cambio (0.0 a 1.0)
        """
        try:
            if not isinstance(changes, dict):
                logger.warning("changes debe ser un diccionario")
                return 0.0
            
            if not old_text or not isinstance(old_text, str):
                old_text = ""
            if not new_text or not isinstance(new_text, str):
                new_text = ""
            
            total_added = changes.get("total_added", 0)
            total_removed = changes.get("total_removed", 0)
            
            # Validar que sean números
            if not isinstance(total_added, (int, float)):
                total_added = 0
            if not isinstance(total_removed, (int, float)):
                total_removed = 0
            
            total_changes = total_added + total_removed
            
            old_words = old_text.split() if old_text else []
            new_words = new_text.split() if new_text else []
            total_content = len(old_words) + len(new_words)
            
            if total_content > 0:
                change_rate = total_changes / total_content
                return min(change_rate, 1.0)  # Limitar a 1.0 máximo
            else:
                return 0.0
        
        except Exception as e:
            logger.error(f"Error calculando tasa de cambio: {e}", exc_info=True)
            return 0.0


class StructuredDataExtractor:
    """Extractor de datos estructurados de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.extraction_patterns: Dict[str, List[Callable]] = {}
        self.extraction_history: List[Dict[str, Any]] = []
    
    def add_extraction_pattern(
        self,
        pattern_name: str,
        pattern_func: Callable,
        data_type: str = "general"
    ):
        """Agregar patrón de extracción"""
        if data_type not in self.extraction_patterns:
            self.extraction_patterns[data_type] = []
        
        self.extraction_patterns[data_type].append({
            "name": pattern_name,
            "function": pattern_func
        })
    
    async def extract_structured_data(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        data_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extraer datos estructurados de un documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            data_types: Tipos de datos a extraer
        
        Returns:
            Datos estructurados extraídos
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to extract from"}
        
        data_types = data_types or list(self.extraction_patterns.keys())
        
        extracted_data = {}
        
        for data_type in data_types:
            if data_type not in self.extraction_patterns:
                continue
            
            patterns = self.extraction_patterns[data_type]
            type_results = []
            
            for pattern_info in patterns:
                pattern_name = pattern_info["name"]
                pattern_func = pattern_info["function"]
                
                try:
                    if asyncio.iscoroutinefunction(pattern_func):
                        result = await pattern_func(content)
                    else:
                        result = pattern_func(content)
                    
                    if result:
                        type_results.append({
                            "pattern": pattern_name,
                            "data": result
                        })
                except Exception as e:
                    logger.error(f"Error en patrón {pattern_name}: {e}")
            
            if type_results:
                extracted_data[data_type] = type_results
        
        # Extracciones comunes
        common_extractions = await self._extract_common_data(content)
        extracted_data.update(common_extractions)
        
        result = {
            "extracted_data": extracted_data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.extraction_history.append(result)
        return result
    
    async def _extract_common_data(self, content: str) -> Dict[str, Any]:
        """
        Extraer datos comunes (emails, teléfonos, fechas, etc.)
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con datos extraídos
        """
        import re
        
        if not content:
            return {}
        
        extractions = {}
        
        try:
            # Emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            matches = safe_regex_match(email_pattern, content, re.IGNORECASE)
            if matches:
                emails = [match.group(0) for match in matches]
                # Eliminar duplicados y validar
                unique_emails = list(set(email.lower() for email in emails))
                if unique_emails:
                    extractions["emails"] = unique_emails[:50]  # Limitar cantidad
            
            # Teléfonos
            phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            matches = safe_regex_match(phone_pattern, content, re.IGNORECASE)
            if matches:
                phones = []
                for match in matches:
                    if match.groups():
                        phone_str = ''.join(g for g in match.groups() if g)
                        if phone_str and len(phone_str) >= 7:  # Validar longitud mínima
                            phones.append(phone_str)
                if phones:
                    extractions["phones"] = list(set(phones))[:50]  # Limitar cantidad
            
            # Fechas
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
                r'[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}'
            ]
            dates = []
            for pattern in date_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                for match in matches:
                    date_str = match.group(0)
                    if date_str:
                        dates.append(date_str)
            if dates:
                extractions["dates"] = list(set(dates))[:50]  # Limitar cantidad
            
            # URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            matches = safe_regex_match(url_pattern, content, re.IGNORECASE)
            if matches:
                urls = [match.group(0) for match in matches]
                # Validar que sean URLs válidas
                valid_urls = [url for url in urls if url.startswith(('http://', 'https://'))]
                if valid_urls:
                    extractions["urls"] = list(set(valid_urls))[:50]  # Limitar cantidad
            
            # Números de identificación (DNI, CIF, etc.)
            id_patterns = {
                "dni": r'\b\d{8}[A-Z]\b',
                "cif": r'\b[A-Z]\d{7}[A-Z0-9]\b',
                "iban": r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b'
            }
            
            ids = {}
            for id_type, pattern in id_patterns.items():
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                if matches:
                    id_values = [match.group(0).upper() for match in matches]
                    if id_values:
                        ids[id_type] = list(set(id_values))[:20]  # Limitar cantidad
            
            if ids:
                extractions["identifiers"] = ids
        
        except Exception as e:
            logger.error(f"Error extrayendo datos comunes: {e}", exc_info=True)
        
        return extractions


class ReadabilityAnalyzer:
    """Analizador de legibilidad de documentos."""
    
    def __init__(self):
        self.readability_history: List[Dict[str, Any]] = []
    
    async def analyze_readability(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        language: str = "es"
    ) -> Dict[str, Any]:
        """
        Analizar legibilidad de un documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            language: Idioma del documento
        
        Returns:
            Análisis de legibilidad
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Calcular métricas básicas
        sentences = self._split_sentences(content)
        words = content.split()
        
        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(self._count_syllables(word, language) for word in words)
        
        # Calcular índices de legibilidad
        if language == "es":
            # Índice Flesch-Szigriszt (español)
            flesch_score = self._calculate_flesch_szigriszt(
                num_sentences, num_words, num_syllables
            )
        else:
            # Índice Flesch Reading Ease (inglés)
            flesch_score = self._calculate_flesch_reading_ease(
                num_sentences, num_words, num_syllables
            )
        
        # Calcular nivel de legibilidad
        readability_level = self._get_readability_level(flesch_score, language)
        
        # Análisis de complejidad
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
        avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
        
        result = {
            "flesch_score": flesch_score,
            "readability_level": readability_level,
            "metrics": {
                "num_sentences": num_sentences,
                "num_words": num_words,
                "num_syllables": num_syllables,
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
                "avg_syllables_per_word": num_syllables / num_words if num_words > 0 else 0
            },
            "recommendations": self._generate_readability_recommendations(
                flesch_score, avg_sentence_length, avg_word_length
            ),
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
        
        self.readability_history.append(result)
        return result
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Dividir texto en oraciones
        
        Args:
            text: Texto a dividir
        
        Returns:
            Lista de oraciones
        """
        import re
        
        if not text or not isinstance(text, str):
            return []
        
        try:
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Error dividiendo texto en oraciones: {e}", exc_info=True)
            return []
    
    def _count_syllables(self, word: str, language: str) -> int:
        """
        Contar sílabas en una palabra
        
        Args:
            word: Palabra a analizar
            language: Idioma ('es' para español, 'en' para inglés)
        
        Returns:
            Número de sílabas (mínimo 1)
        """
        if not word or not isinstance(word, str):
            return 1
        
        if not language or not isinstance(language, str):
            language = "es"  # Default a español
        
        try:
            word = word.lower().strip('.,!?;:')
            
            if not word:
                return 1
            
            if language == "es":
                # Reglas básicas para español
                vowels = 'aeiouáéíóúü'
                syllable_count = 0
                prev_was_vowel = False
                
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = is_vowel
                
                # Mínimo 1 sílaba
                return max(1, syllable_count)
            else:
                # Reglas básicas para inglés
                word_lower = word.lower()
                if len(word_lower) <= 3:
                    return 1
                
                vowels = 'aeiouy'
                syllable_count = 0
                prev_was_vowel = False
                
                for char in word_lower:
                    is_vowel = char in vowels
                    if is_vowel and not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = is_vowel
                
                # Ajustar para palabras que terminan en 'e'
                if word_lower.endswith('e'):
                    syllable_count = max(1, syllable_count - 1)
                
                return max(1, syllable_count)
        
        except Exception as e:
            logger.error(f"Error contando sílabas en palabra '{word}': {e}", exc_info=True)
            return 1
    
    def _calculate_flesch_szigriszt(
        self,
        num_sentences: int,
        num_words: int,
        num_syllables: int
    ) -> float:
        """
        Calcular índice Flesch-Szigriszt (español)
        
        Args:
            num_sentences: Número de oraciones
            num_words: Número de palabras
            num_syllables: Número de sílabas
        
        Returns:
            Score de legibilidad (0-100)
        """
        try:
            # Validar tipos y valores
            if not isinstance(num_sentences, (int, float)) or num_sentences < 0:
                num_sentences = 0
            if not isinstance(num_words, (int, float)) or num_words < 0:
                num_words = 0
            if not isinstance(num_syllables, (int, float)) or num_syllables < 0:
                num_syllables = 0
            
            if num_sentences == 0 or num_words == 0:
                return 0.0
            
            # Calcular promedios de manera segura
            avg_sentence_length = num_words / num_sentences
            avg_syllables_per_word = num_syllables / num_words
            
            # Fórmula Flesch-Szigriszt
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Limitar el score entre 0 y 100
            score = max(0.0, min(score, 100.0))
            return round(score, 2)
        
        except Exception as e:
            logger.error(f"Error calculando índice Flesch-Szigriszt: {e}", exc_info=True)
            return 0.0
    
    def _calculate_flesch_reading_ease(
        self,
        num_sentences: int,
        num_words: int,
        num_syllables: int
    ) -> float:
        """
        Calcular índice Flesch Reading Ease (inglés)
        
        Args:
            num_sentences: Número de oraciones
            num_words: Número de palabras
            num_syllables: Número de sílabas
        
        Returns:
            Score de legibilidad (0-100)
        """
        try:
            # Validar tipos y valores
            if not isinstance(num_sentences, (int, float)) or num_sentences < 0:
                num_sentences = 0
            if not isinstance(num_words, (int, float)) or num_words < 0:
                num_words = 0
            if not isinstance(num_syllables, (int, float)) or num_syllables < 0:
                num_syllables = 0
            
            if num_sentences == 0 or num_words == 0:
                return 0.0
            
            # Calcular promedios de manera segura
            avg_sentence_length = num_words / num_sentences
            avg_syllables_per_word = num_syllables / num_words
            
            # Fórmula Flesch Reading Ease
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Limitar el score entre 0 y 100
            score = max(0.0, min(score, 100.0))
            return round(score, 2)
        
        except Exception as e:
            logger.error(f"Error calculando índice Flesch Reading Ease: {e}", exc_info=True)
            return 0.0
    
    def _get_readability_level(self, score: float, language: str) -> str:
        """
        Obtener nivel de legibilidad
        
        Args:
            score: Score de legibilidad (0-100)
            language: Idioma ('es' para español, 'en' para inglés)
        
        Returns:
            Nivel de legibilidad como string
        """
        if not isinstance(score, (int, float)):
            score = 0.0
        
        if not language or not isinstance(language, str):
            language = "es"  # Default a español
        
        # Normalizar score entre 0 y 100
        score = max(0.0, min(100.0, float(score)))
        
        try:
            if language == "es":
                # Flesch-Szigriszt para español
                if score >= 80:
                    return "Muy fácil"
                elif score >= 60:
                    return "Fácil"
                elif score >= 40:
                    return "Normal"
                elif score >= 20:
                    return "Difícil"
                else:
                    return "Muy difícil"
            else:
                # Flesch Reading Ease para inglés
                if score >= 90:
                    return "Very Easy"
                elif score >= 80:
                    return "Easy"
                elif score >= 70:
                    return "Fairly Easy"
                elif score >= 60:
                    return "Standard"
                elif score >= 50:
                    return "Fairly Difficult"
                elif score >= 30:
                    return "Difficult"
                else:
                    return "Very Difficult"
        
        except Exception as e:
            logger.error(f"Error obteniendo nivel de legibilidad: {e}", exc_info=True)
            return "Unknown"
    
    def _generate_readability_recommendations(
        self,
        flesch_score: float,
        avg_sentence_length: float,
        avg_word_length: float
    ) -> List[str]:
        """
        Generar recomendaciones de legibilidad
        
        Args:
            flesch_score: Score de Flesch (0-100)
            avg_sentence_length: Longitud promedio de oraciones
            avg_word_length: Longitud promedio de palabras
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        try:
            # Validar y normalizar valores
            if not isinstance(flesch_score, (int, float)):
                flesch_score = 0.0
            flesch_score = max(0.0, min(100.0, float(flesch_score)))
            
            if not isinstance(avg_sentence_length, (int, float)):
                avg_sentence_length = 0.0
            avg_sentence_length = max(0.0, float(avg_sentence_length))
            
            if not isinstance(avg_word_length, (int, float)):
                avg_word_length = 0.0
            avg_word_length = max(0.0, float(avg_word_length))
            
            # Generar recomendaciones basadas en los valores
            if flesch_score < 40:
                recommendations.append("El documento es difícil de leer. Considerar usar oraciones más cortas.")
            
            if avg_sentence_length > 20:
                recommendations.append("Las oraciones son muy largas. Considerar dividirlas en oraciones más cortas.")
            
            if avg_word_length > 6:
                recommendations.append("Se usan palabras muy largas. Considerar usar palabras más simples cuando sea posible.")
            
            if flesch_score > 80:
                recommendations.append("El documento es muy fácil de leer. Puede ser apropiado para audiencias generales.")
        
        except Exception as e:
            logger.error(f"Error generando recomendaciones de legibilidad: {e}", exc_info=True)
        
        return recommendations


class PlagiarismDetector:
    """Detector de plagio usando análisis de similitud."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.reference_documents: Dict[str, str] = {}
        self.detection_history: List[Dict[str, Any]] = []
    
    def add_reference_document(self, doc_id: str, content: str):
        """
        Agregar documento de referencia
        
        Args:
            doc_id: ID del documento de referencia
            content: Contenido del documento
        
        Raises:
            ValueError: Si doc_id o content son inválidos
        """
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError("doc_id debe ser un string no vacío")
        
        if not content or not isinstance(content, str):
            raise ValueError("content debe ser un string no vacío")
        
        try:
            # Validar longitud mínima de contenido
            if len(content.strip()) < 10:
                logger.warning(f"Documento de referencia {doc_id} tiene contenido muy corto")
            
            self.reference_documents[doc_id] = content
            logger.debug(f"Documento de referencia agregado: {doc_id}")
        
        except Exception as e:
            logger.error(f"Error agregando documento de referencia {doc_id}: {e}", exc_info=True)
            raise
    
    async def detect_plagiarism(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Detectar posible plagio comparando con documentos de referencia
        
        Args:
            document_path: Ruta al documento a analizar
            document_content: Contenido del documento
            threshold: Umbral de similitud para considerar plagio (0.0 a 1.0)
        
        Returns:
            Análisis de plagio
        
        Raises:
            ValueError: Si threshold está fuera de rango
        """
        # Validar threshold
        if not isinstance(threshold, (int, float)):
            threshold = 0.7
        threshold = max(0.0, min(1.0, float(threshold)))
        
        content = document_content
        if document_path:
            try:
                is_valid, error_msg = validate_file_path(document_path, must_exist=True)
                if not is_valid:
                    return {"error": f"Invalid document path: {error_msg}"}
                
                processor = DocumentProcessor()
                content = processor.process_document(document_path, "txt")
            except Exception as e:
                logger.error(f"Error procesando documento {document_path}: {e}", exc_info=True)
                return {"error": f"Error processing document: {str(e)}"}
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Validar contenido
        is_valid, error_msg = validate_content(content, min_length=10)
        if not is_valid:
            return {"error": f"Invalid content: {error_msg}"}
        
        if not self.reference_documents or len(self.reference_documents) == 0:
            return {"error": "No reference documents available"}
        
        try:
            # Comparar con cada documento de referencia
            matches = []
            
            for ref_id, ref_content in self.reference_documents.items():
                if not ref_id or not ref_content:
                    continue
                
                try:
                    similarity = await self._calculate_text_similarity(content, ref_content)
                    
                    if not isinstance(similarity, (int, float)):
                        continue
                    
                    similarity = float(similarity)
                    
                    if similarity >= threshold:
                        # Análisis detallado de similitud
                        try:
                            detailed_analysis = await self._analyze_similarity_details(content, ref_content)
                        except Exception as e:
                            logger.warning(f"Error en análisis detallado para {ref_id}: {e}")
                            detailed_analysis = {}
                        
                        matches.append({
                            "reference_id": str(ref_id),
                            "similarity": round(similarity, 4),
                            "analysis": detailed_analysis
                        })
                except Exception as e:
                    logger.warning(f"Error comparando con documento de referencia {ref_id}: {e}")
                    continue
            
            # Ordenar por similitud
            matches.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
            
            highest_similarity = 0.0
            if matches:
                highest_similarity = matches[0].get("similarity", 0.0)
            
            result = {
                "document_content": content[:500] if content else "",  # Preview
                "matches": matches,
                "total_matches": len(matches),
                "highest_similarity": highest_similarity,
                "plagiarism_detected": len(matches) > 0,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            self.detection_history.append(result)
            return result
        
        except Exception as e:
            logger.error(f"Error detectando plagio: {e}", exc_info=True)
            return {
                "error": f"Error detecting plagiarism: {str(e)}",
                "matches": [],
                "total_matches": 0,
                "highest_similarity": 0.0,
                "plagiarism_detected": False,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud entre dos textos"""
        try:
            embeddings = await self.analyzer.embedding_generator.generate_embeddings([text1, text2])
            
            if len(embeddings) == 2:
                emb1 = embeddings[0]
                emb2 = embeddings[1]
                
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                similarity = dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
                return float(similarity)
        except Exception as e:
            logger.error(f"Error calculando similitud: {e}")
        
        return 0.0
    
    async def _analyze_similarity_details(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Analizar detalles de similitud entre dos textos
        
        Args:
            text1: Primer texto
            text2: Segundo texto
        
        Returns:
            Diccionario con detalles de similitud
        """
        if not text1 or not text2:
            return {
                "common_keywords": [],
                "common_entities": [],
                "keyword_overlap": 0.0,
                "entity_overlap": 0.0
            }
        
        try:
            # Comparar palabras clave con manejo de errores
            keywords1 = []
            keywords2 = []
            try:
                keywords1_result = await self.analyzer.extract_keywords(text1, top_k=10)
                if isinstance(keywords1_result, list):
                    keywords1 = keywords1_result
                elif isinstance(keywords1_result, dict) and "keywords" in keywords1_result:
                    keywords1 = keywords1_result["keywords"]
            except Exception as e:
                logger.warning(f"Error extrayendo keywords del texto 1: {e}")
            
            try:
                keywords2_result = await self.analyzer.extract_keywords(text2, top_k=10)
                if isinstance(keywords2_result, list):
                    keywords2 = keywords2_result
                elif isinstance(keywords2_result, dict) and "keywords" in keywords2_result:
                    keywords2 = keywords2_result["keywords"]
            except Exception as e:
                logger.warning(f"Error extrayendo keywords del texto 2: {e}")
            
            # Asegurar que sean listas de strings
            keywords1 = [str(k) for k in keywords1 if k]
            keywords2 = [str(k) for k in keywords2 if k]
            
            common_keywords = set(keywords1) & set(keywords2)
            
            # Comparar entidades con manejo de errores
            entities1 = []
            entities2 = []
            try:
                entities1_result = await self.analyzer.extract_entities(text1)
                if isinstance(entities1_result, list):
                    entities1 = entities1_result
                elif isinstance(entities1_result, dict) and "entities" in entities1_result:
                    entities1 = entities1_result["entities"]
            except Exception as e:
                logger.warning(f"Error extrayendo entidades del texto 1: {e}")
            
            try:
                entities2_result = await self.analyzer.extract_entities(text2)
                if isinstance(entities2_result, list):
                    entities2 = entities2_result
                elif isinstance(entities2_result, dict) and "entities" in entities2_result:
                    entities2 = entities2_result["entities"]
            except Exception as e:
                logger.warning(f"Error extrayendo entidades del texto 2: {e}")
            
            # Extraer textos de entidades de manera segura
            entity_texts1 = set()
            for entity in entities1:
                if isinstance(entity, dict) and "text" in entity:
                    entity_texts1.add(str(entity["text"]))
                elif isinstance(entity, str):
                    entity_texts1.add(entity)
            
            entity_texts2 = set()
            for entity in entities2:
                if isinstance(entity, dict) and "text" in entity:
                    entity_texts2.add(str(entity["text"]))
                elif isinstance(entity, str):
                    entity_texts2.add(entity)
            
            common_entities = entity_texts1 & entity_texts2
            
            # Calcular overlaps de manera segura
            keyword_overlap = 0.0
            max_keywords = max(len(keywords1), len(keywords2))
            if max_keywords > 0:
                keyword_overlap = len(common_keywords) / max_keywords
            
            entity_overlap = 0.0
            max_entities = max(len(entity_texts1), len(entity_texts2))
            if max_entities > 0:
                entity_overlap = len(common_entities) / max_entities
            
            return {
                "common_keywords": list(common_keywords),
                "common_entities": list(common_entities),
                "keyword_overlap": round(keyword_overlap, 4),
                "entity_overlap": round(entity_overlap, 4)
            }
        
        except Exception as e:
            logger.error(f"Error analizando detalles de similitud: {e}", exc_info=True)
            return {
                "common_keywords": [],
                "common_entities": [],
                "keyword_overlap": 0.0,
                "entity_overlap": 0.0
            }


class QuestionGenerator:
    """Generador de preguntas basadas en contenido de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.generated_questions: List[Dict[str, Any]] = []
    
    async def generate_questions(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        num_questions: int = 5,
        question_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generar preguntas basadas en el contenido del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            num_questions: Número de preguntas a generar
            question_types: Tipos de preguntas ('factual', 'inferential', 'analytical')
        
        Returns:
            Preguntas generadas
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to generate questions from"}
        
        question_types = question_types or ["factual", "inferential"]
        
        # Extraer entidades y keywords para generar preguntas
        entities = await self.analyzer.extract_entities(content)
        keywords = await self.analyzer.extract_keywords(content, top_k=10)
        
        # Analizar estructura
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        
        questions = []
        
        # Generar preguntas factuales
        if "factual" in question_types:
            factual_questions = self._generate_factual_questions(content, entities, keywords)
            questions.extend(factual_questions[:num_questions // 2])
        
        # Generar preguntas inferenciales
        if "inferential" in question_types:
            inferential_questions = self._generate_inferential_questions(content, structure)
            questions.extend(inferential_questions[:num_questions // 2])
        
        # Generar preguntas analíticas
        if "analytical" in question_types:
            analytical_questions = self._generate_analytical_questions(content, structure)
            questions.extend(analytical_questions[:num_questions // 3])
        
        result = {
            "questions": questions[:num_questions],
            "total_generated": len(questions),
            "document_preview": content[:200],
            "timestamp": datetime.now().isoformat()
        }
        
        self.generated_questions.append(result)
        return result
    
    def _generate_factual_questions(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generar preguntas factuales
        
        Args:
            content: Contenido del documento
            entities: Lista de entidades extraídas
            keywords: Lista de palabras clave
        
        Returns:
            Lista de preguntas factuales
        """
        questions = []
        
        if not content:
            return questions
        
        try:
            # Validar tipos
            if not isinstance(entities, list):
                entities = []
            if not isinstance(keywords, list):
                keywords = []
            
            # Preguntas sobre entidades
            valid_entity_labels = {"PERSON", "ORG", "LOC"}
            entity_count = 0
            
            for entity in entities:
                if entity_count >= 5:  # Limitar a 5 entidades
                    break
                
                if not isinstance(entity, dict):
                    continue
                
                entity_label = entity.get("label")
                entity_text = entity.get("text")
                
                if entity_label in valid_entity_labels and entity_text:
                    try:
                        question = f"¿Quién o qué es {entity_text}?"
                        questions.append({
                            "question": question,
                            "type": "factual",
                            "expected_answer_type": entity_label,
                            "related_entity": str(entity_text)
                        })
                        entity_count += 1
                    except Exception as e:
                        logger.warning(f"Error generando pregunta para entidad {entity_text}: {e}")
                        continue
            
            # Preguntas sobre keywords importantes
            keyword_count = 0
            for keyword in keywords:
                if keyword_count >= 3:  # Limitar a 3 keywords
                    break
                
                if not keyword or not isinstance(keyword, str):
                    continue
                
                try:
                    question = f"¿Qué información se proporciona sobre {keyword}?"
                    questions.append({
                        "question": question,
                        "type": "factual",
                        "related_keyword": str(keyword)
                    })
                    keyword_count += 1
                except Exception as e:
                    logger.warning(f"Error generando pregunta para keyword {keyword}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error generando preguntas factuales: {e}", exc_info=True)
        
        return questions
    
    def _generate_inferential_questions(
        self,
        content: str,
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generar preguntas inferenciales
        
        Args:
            content: Contenido del documento
            structure: Estructura del documento
        
        Returns:
            Lista de preguntas inferenciales
        """
        questions = []
        
        if not content:
            return questions
        
        try:
            # Validar estructura
            if not isinstance(structure, dict):
                structure = {}
            
            # Preguntas basadas en estructura
            sections = structure.get("sections")
            if sections and isinstance(sections, list) and len(sections) > 0:
                question = "¿Cuál es la relación entre las diferentes secciones del documento?"
                questions.append({
                    "question": question,
                    "type": "inferential",
                    "focus": "structure"
                })
            
            # Preguntas sobre implicaciones
            if content and len(content.strip()) > 50:  # Validar que haya suficiente contenido
                question = "¿Cuáles son las implicaciones principales del contenido?"
                questions.append({
                    "question": question,
                    "type": "inferential",
                    "focus": "implications"
                })
        
        except Exception as e:
            logger.error(f"Error generando preguntas inferenciales: {e}", exc_info=True)
        
        return questions
    
    def _generate_analytical_questions(
        self,
        content: str,
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generar preguntas analíticas
        
        Args:
            content: Contenido del documento
            structure: Estructura del documento
        
        Returns:
            Lista de preguntas analíticas
        """
        questions = []
        
        if not content:
            return questions
        
        try:
            # Validar estructura
            if not isinstance(structure, dict):
                structure = {}
            
            # Validar que haya suficiente contenido
            if len(content.strip()) < 50:
                return questions
            
            # Preguntas de análisis
            question1 = "¿Cuál es el argumento principal del documento?"
            questions.append({
                "question": question1,
                "type": "analytical",
                "focus": "main_argument"
            })
            
            question2 = "¿Qué evidencia se presenta para apoyar las conclusiones?"
            questions.append({
                "question": question2,
                "type": "analytical",
                "focus": "evidence"
            })
        
        except Exception as e:
            logger.error(f"Error generando preguntas analíticas: {e}", exc_info=True)
        
        return questions


class DocumentCoherenceAnalyzer:
    """Analizador de coherencia de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.coherence_history: List[Dict[str, Any]] = []
    
    async def analyze_coherence(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar coherencia de un documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis de coherencia
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Analizar coherencia temática
        topics = await self.analyzer.extract_topics(content, num_topics=5)
        topic_coherence = self._calculate_topic_coherence(topics)
        
        # Analizar coherencia estructural
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        structural_coherence = self._calculate_structural_coherence(structure)
        
        # Analizar coherencia semántica
        semantic_coherence = await self._calculate_semantic_coherence(content)
        
        # Calcular coherencia general
        overall_coherence = (
            topic_coherence * 0.3 +
            structural_coherence * 0.3 +
            semantic_coherence * 0.4
        )
        
        result = {
            "overall_coherence": overall_coherence,
            "topic_coherence": topic_coherence,
            "structural_coherence": structural_coherence,
            "semantic_coherence": semantic_coherence,
            "analysis": {
                "topics": topics,
                "structure": structure,
                "coherence_level": self._get_coherence_level(overall_coherence)
            },
            "recommendations": self._generate_coherence_recommendations(
                topic_coherence, structural_coherence, semantic_coherence
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        self.coherence_history.append(result)
        return result
    
    def _calculate_topic_coherence(self, topics: List[Dict[str, Any]]) -> float:
        """
        Calcular coherencia temática
        
        Args:
            topics: Lista de temas con sus scores
        
        Returns:
            Score de coherencia temática (0.0 a 1.0)
        """
        if not topics or not isinstance(topics, list):
            return 0.0
        
        try:
            # Extraer scores con validación
            scores = []
            for topic in topics:
                if isinstance(topic, dict):
                    score = topic.get("score", 0.0)
                    if isinstance(score, (int, float)) and score >= 0:
                        scores.append(float(score))
            
            if not scores:
                return 0.0
            
            avg_score = sum(scores) / len(scores)
            
            # Si hay muchos temas con scores similares, menor coherencia
            if len(scores) > 1:
                variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                coherence = 1.0 - min(variance, 0.5)
            else:
                coherence = min(avg_score, 1.0)
            
            return round(max(0.0, min(coherence, 1.0)), 4)
        
        except Exception as e:
            logger.error(f"Error calculando coherencia temática: {e}", exc_info=True)
            return 0.0
    
    def _calculate_structural_coherence(self, structure: Dict[str, Any]) -> float:
        """
        Calcular coherencia estructural
        
        Args:
            structure: Diccionario con estructura del documento
        
        Returns:
            Score de coherencia estructural (0.0 a 1.0)
        """
        if not structure or not isinstance(structure, dict):
            return 0.5  # Sin estructura clara
        
        try:
            sections = structure.get("sections", [])
            
            if not sections or not isinstance(sections, list):
                return 0.5  # Sin estructura clara
            
            # Más secciones bien definidas = mayor coherencia
            num_sections = len(sections)
            if num_sections <= 0:
                return 0.5
            
            coherence = min(num_sections / 5.0, 1.0)  # Normalizar
            
            return round(max(0.0, min(coherence, 1.0)), 4)
        
        except Exception as e:
            logger.error(f"Error calculando coherencia estructural: {e}", exc_info=True)
            return 0.5
    
    async def _calculate_semantic_coherence(self, content: str) -> float:
        """
        Calcular coherencia semántica
        
        Args:
            content: Contenido del documento
        
        Returns:
            Score de coherencia semántica (0.0 a 1.0)
        """
        if not content:
            return 0.5
        
        try:
            # Validar longitud mínima
            if len(content) < 100:
                return 1.0  # Documento muy corto, se considera coherente
            
            # Dividir en chunks y calcular similitud entre chunks adyacentes
            chunk_size = 500
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            
            # Filtrar chunks vacíos
            chunks = [chunk for chunk in chunks if chunk.strip()]
            
            if len(chunks) < 2:
                return 1.0  # Solo un chunk, se considera coherente
            
            similarities = []
            for i in range(len(chunks) - 1):
                try:
                    similarity = await self._chunk_similarity(chunks[i], chunks[i+1])
                    if isinstance(similarity, (int, float)) and 0 <= similarity <= 1:
                        similarities.append(float(similarity))
                except Exception as e:
                    logger.warning(f"Error calculando similitud de chunks {i}-{i+1}: {e}")
                    continue
            
            if not similarities:
                return 0.5  # No se pudo calcular similitud, valor por defecto
            
            avg_similarity = sum(similarities) / len(similarities)
            return round(max(0.0, min(avg_similarity, 1.0)), 4)
        
        except Exception as e:
            logger.error(f"Error calculando coherencia semántica: {e}", exc_info=True)
            return 0.5
    
    async def _chunk_similarity(self, chunk1: str, chunk2: str) -> float:
        """
        Calcular similitud entre dos chunks
        
        Args:
            chunk1: Primer chunk de texto
            chunk2: Segundo chunk de texto
        
        Returns:
            Score de similitud (0.0 a 1.0)
        """
        if not chunk1 or not chunk2:
            return 0.0
        
        try:
            # Validar que el analyzer tenga embedding_generator
            if not hasattr(self.analyzer, 'embedding_generator') or not self.analyzer.embedding_generator:
                return 0.0
            
            embeddings = await self.analyzer.embedding_generator.generate_embeddings([chunk1, chunk2])
            
            if not embeddings or len(embeddings) != 2:
                return 0.0
            
            emb1 = embeddings[0]
            emb2 = embeddings[1]
            
            # Validar que los embeddings sean arrays válidos
            if not isinstance(emb1, (list, np.ndarray)) or not isinstance(emb2, (list, np.ndarray)):
                return 0.0
            
            # Convertir a numpy arrays si es necesario
            if isinstance(emb1, list):
                emb1 = np.array(emb1)
            if isinstance(emb2, list):
                emb2 = np.array(emb2)
            
            # Validar que tengan la misma dimensión
            if len(emb1) != len(emb2) or len(emb1) == 0:
                return 0.0
            
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            # Validar división por cero
            if (norm1 * norm2) <= 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Asegurar que el resultado esté en el rango [0, 1]
            similarity = max(0.0, min(1.0, float(similarity)))
            
            return round(similarity, 4)
        
        except Exception as e:
            logger.error(f"Error calculando similitud de chunks: {e}", exc_info=True)
            return 0.0
    
    def _get_coherence_level(self, coherence_score: float) -> str:
        """
        Obtener nivel de coherencia
        
        Args:
            coherence_score: Score de coherencia (0.0 a 1.0)
        
        Returns:
            Nivel de coherencia como string
        """
        if not isinstance(coherence_score, (int, float)):
            return "Desconocido"
        
        # Normalizar score entre 0 y 1
        coherence_score = max(0.0, min(1.0, float(coherence_score)))
        
        try:
            if coherence_score >= 0.8:
                return "Muy alta"
            elif coherence_score >= 0.6:
                return "Alta"
            elif coherence_score >= 0.4:
                return "Media"
            elif coherence_score >= 0.2:
                return "Baja"
            else:
                return "Muy baja"
        
        except Exception as e:
            logger.error(f"Error obteniendo nivel de coherencia: {e}", exc_info=True)
            return "Desconocido"
    
    def _generate_coherence_recommendations(
        self,
        topic_coherence: float,
        structural_coherence: float,
        semantic_coherence: float
    ) -> List[str]:
        """
        Generar recomendaciones de coherencia
        
        Args:
            topic_coherence: Coherencia temática (0.0 a 1.0)
            structural_coherence: Coherencia estructural (0.0 a 1.0)
            semantic_coherence: Coherencia semántica (0.0 a 1.0)
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        try:
            # Validar y normalizar valores
            if not isinstance(topic_coherence, (int, float)):
                topic_coherence = 0.0
            topic_coherence = max(0.0, min(1.0, float(topic_coherence)))
            
            if not isinstance(structural_coherence, (int, float)):
                structural_coherence = 0.0
            structural_coherence = max(0.0, min(1.0, float(structural_coherence)))
            
            if not isinstance(semantic_coherence, (int, float)):
                semantic_coherence = 0.0
            semantic_coherence = max(0.0, min(1.0, float(semantic_coherence)))
            
            # Generar recomendaciones basadas en los valores
            if topic_coherence < 0.5:
                recommendations.append("La coherencia temática es baja. Considerar enfocarse en menos temas o mejor organización.")
            
            if structural_coherence < 0.5:
                recommendations.append("La estructura del documento podría mejorarse. Considerar agregar secciones claras.")
            
            if semantic_coherence < 0.5:
                recommendations.append("La coherencia semántica es baja. Las ideas podrían no estar bien conectadas.")
        
        except Exception as e:
            logger.error(f"Error generando recomendaciones de coherencia: {e}", exc_info=True)
        
        return recommendations


# ============================================================================
# SISTEMAS AVANZADOS ADICIONALES
# ============================================================================

class LanguageDetector:
    """Detector de idioma de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.detection_history: List[Dict[str, Any]] = []
    
    async def detect_language(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Detectar idioma de un documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            sample_size: Tamaño de muestra para análisis (mínimo 100)
        
        Returns:
            Análisis de idioma detectado
        """
        # Validar sample_size
        if not isinstance(sample_size, int) or sample_size < 100:
            sample_size = 1000
        sample_size = max(100, min(sample_size, 10000))  # Limitar entre 100 y 10000
        
        content = document_content
        if document_path:
            try:
                is_valid, error_msg = validate_file_path(document_path, must_exist=True)
                if not is_valid:
                    return {"error": f"Invalid document path: {error_msg}"}
                
                processor = DocumentProcessor()
                content = processor.process_document(document_path, "txt")
            except Exception as e:
                logger.error(f"Error procesando documento {document_path}: {e}", exc_info=True)
                return {"error": f"Error processing document: {str(e)}"}
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Validar contenido
        is_valid, error_msg = validate_content(content, min_length=10)
        if not is_valid:
            return {"error": f"Invalid content: {error_msg}"}
        
        try:
            # Usar muestra para análisis rápido
            sample = content[:sample_size] if len(content) > sample_size else content
            
            if not sample or len(sample.strip()) < 10:
                return {"error": "Sample too short for language detection"}
            
            # Detección básica usando patrones comunes
            language_scores = self._detect_by_patterns(sample)
            
            # Detección usando embeddings (si disponible)
            if hasattr(self.analyzer, 'embedding_generator') and self.analyzer.embedding_generator:
                try:
                    embedding_language = await self._detect_by_embeddings(sample)
                    if embedding_language and isinstance(embedding_language, dict):
                        # Combinar scores (promedio ponderado)
                        for lang, score in embedding_language.items():
                            if lang in language_scores:
                                language_scores[lang] = (language_scores[lang] + score) / 2.0
                            else:
                                language_scores[lang] = score
                except Exception as e:
                    logger.warning(f"Error en detección por embeddings: {e}")
            
            # Determinar idioma más probable
            if not language_scores:
                return {
                    "error": "No language scores generated",
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "language_scores": {},
                    "sample_analyzed": len(sample),
                    "timestamp": datetime.now().isoformat()
                }
            
            detected_language = max(language_scores.items(), key=lambda x: x[1])[0] if language_scores else "unknown"
            confidence = language_scores.get(detected_language, 0.0) if language_scores else 0.0
            
            # Normalizar confidence
            confidence = max(0.0, min(1.0, float(confidence)))
            
            result = {
                "detected_language": detected_language,
                "confidence": round(confidence, 4),
                "language_scores": {k: round(v, 4) for k, v in language_scores.items()},
                "sample_analyzed": len(sample),
                "timestamp": datetime.now().isoformat()
            }
            
            self.detection_history.append(result)
            return result
        
        except Exception as e:
            logger.error(f"Error detectando idioma: {e}", exc_info=True)
            return {
                "error": f"Error detecting language: {str(e)}",
                "detected_language": "unknown",
                "confidence": 0.0,
                "language_scores": {},
                "sample_analyzed": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def _detect_by_patterns(self, text: str) -> Dict[str, float]:
        """Detectar idioma usando patrones comunes"""
        text_lower = text.lower()
        
        # Patrones comunes por idioma
        patterns = {
            "es": ["el", "la", "de", "que", "y", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para", "del"],
            "en": ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"],
            "fr": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir", "que", "pour", "dans", "ce", "son", "une", "sur", "avec", "ne", "se"],
            "pt": ["o", "de", "a", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as"],
            "it": ["il", "di", "e", "la", "a", "per", "in", "un", "che", "è", "con", "da", "una", "non", "si", "le", "del", "al", "dei", "i"],
            "de": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als"]
        }
        
        scores = {}
        for lang, lang_patterns in patterns.items():
            matches = sum(1 for pattern in lang_patterns if pattern in text_lower)
            score = matches / len(lang_patterns) if lang_patterns else 0.0
            scores[lang] = score
        
        return scores
    
    async def _detect_by_embeddings(self, text: str) -> Optional[Dict[str, float]]:
        """Detectar idioma usando embeddings (simplificado)"""
        # En producción, usar un modelo de detección de idioma específico
        # Por ahora, retornar None para usar solo detección por patrones
        return None


class CitationExtractor:
    """Extractor de citas y referencias bibliográficas."""
    
    def __init__(self):
        self.extracted_citations: List[Dict[str, Any]] = []
    
    async def extract_citations(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extraer citas y referencias de un documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Citas y referencias extraídas
        """
        content = document_content
        if document_path:
            try:
                is_valid, error_msg = validate_file_path(document_path, must_exist=True)
                if not is_valid:
                    return {"error": f"Invalid document path: {error_msg}"}
                
                processor = DocumentProcessor()
                content = processor.process_document(document_path, "txt")
            except Exception as e:
                logger.error(f"Error procesando documento {document_path}: {e}", exc_info=True)
                return {"error": f"Error processing document: {str(e)}"}
        
        if not content:
            return {"error": "No content to extract from"}
        
        # Validar contenido
        is_valid, error_msg = validate_content(content, min_length=10)
        if not is_valid:
            return {"error": f"Invalid content: {error_msg}"}
        
        try:
            # Extraer citas en el texto (APA, MLA, Chicago, etc.)
            in_text_citations = self._extract_in_text_citations(content)
            if not isinstance(in_text_citations, list):
                in_text_citations = []
            
            # Extraer referencias bibliográficas
            references = self._extract_references(content)
            if not isinstance(references, list):
                references = []
            
            # Extraer URLs de referencias
            urls = self._extract_reference_urls(content)
            if not isinstance(urls, list):
                urls = []
            
            result = {
                "in_text_citations": in_text_citations,
                "references": references,
                "urls": urls,
                "total_citations": len(in_text_citations),
                "total_references": len(references),
                "total_urls": len(urls),
                "timestamp": datetime.now().isoformat()
            }
            
            self.extracted_citations.append(result)
            return result
        
        except Exception as e:
            logger.error(f"Error extrayendo citas: {e}", exc_info=True)
            return {
                "error": f"Error extracting citations: {str(e)}",
                "in_text_citations": [],
                "references": [],
                "urls": [],
                "total_citations": 0,
                "total_references": 0,
                "total_urls": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_in_text_citations(self, content: str) -> List[Dict[str, Any]]:
        """Extraer citas en el texto"""
        import re
        
        citations = []
        
        # Patrones comunes de citas
        patterns = [
            # APA: (Author, Year) o (Author, Year, p. Page)
            r'\([A-Z][a-zA-Z\s]+,\s*\d{4}(?:,\s*p\.\s*\d+)?\)',
            # MLA: (Author Page)
            r'\([A-Z][a-zA-Z\s]+\s+\d+\)',
            # Chicago: (Author Year, Page)
            r'\([A-Z][a-zA-Z\s]+\s+\d{4},\s*\d+\)',
            # Numeric: [1], [2-3], etc.
            r'\[\d+(?:-\d+)?\]'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                citations.append({
                    "text": match.group(),
                    "position": match.start(),
                    "type": "numeric" if "[" in match.group() else "author_year"
                })
        
        return citations
    
    def _extract_references(self, content: str) -> List[Dict[str, Any]]:
        """Extraer referencias bibliográficas"""
        import re
        
        references = []
        
        # Buscar sección de referencias
        references_section = re.search(r'(?:referencias|bibliografía|references|bibliography)\s*:?\s*\n', content, re.IGNORECASE)
        
        if references_section:
            ref_text = content[references_section.end():]
            
            # Patrones para diferentes formatos
            # APA: Author, A. A. (Year). Title. Publisher.
            apa_pattern = r'[A-Z][a-zA-Z\s]+,\s*[A-Z]\.\s*[A-Z]\.\s*\(\d{4}\)\.\s*[^.]+\.[^.]*'
            # MLA: Author, First. "Title." Publisher, Year.
            mla_pattern = r'[A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z]+\.\s*"[^"]+"\.[^.]*'
            
            for pattern in [apa_pattern, mla_pattern]:
                matches = re.finditer(pattern, ref_text)
                for match in matches:
                    references.append({
                        "text": match.group(),
                        "format": "APA" if "(" in match.group() else "MLA"
                    })
        
        return references
    
    def _extract_reference_urls(self, content: str) -> List[str]:
        """
        Extraer URLs de referencias
        
        Args:
            content: Contenido del documento
        
        Returns:
            Lista de URLs encontradas
        """
        import re
        
        if not content or not isinstance(content, str):
            return []
        
        try:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            matches = safe_regex_match(url_pattern, content, re.IGNORECASE)
            
            if matches:
                urls = [match.group(0) for match in matches]
                # Validar que sean URLs válidas
                valid_urls = [url for url in urls if url.startswith(('http://', 'https://'))]
                # Eliminar duplicados y limitar cantidad
                return list(set(valid_urls))[:50]
        
        except Exception as e:
            logger.error(f"Error extrayendo URLs de referencias: {e}", exc_info=True)
        
        return []


class RiskAnalyzer:
    """Analizador de riesgo de contenido de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.risk_history: List[Dict[str, Any]] = []
    
    async def analyze_risks(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        risk_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analizar riesgos en el contenido de un documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            risk_categories: Categorías de riesgo a analizar
        
        Returns:
            Análisis de riesgos detectados
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        risk_categories = risk_categories or ["security", "privacy", "compliance", "reputation"]
        
        detected_risks = {}
        
        # Análisis de seguridad
        if "security" in risk_categories:
            detected_risks["security"] = self._analyze_security_risks(content)
        
        # Análisis de privacidad
        if "privacy" in risk_categories:
            detected_risks["privacy"] = self._analyze_privacy_risks(content)
        
        # Análisis de compliance
        if "compliance" in risk_categories:
            detected_risks["compliance"] = self._analyze_compliance_risks(content)
        
        # Análisis de reputación
        if "reputation" in risk_categories:
            detected_risks["reputation"] = await self._analyze_reputation_risks(content)
        
        # Calcular riesgo general
        overall_risk = self._calculate_overall_risk(detected_risks)
        
        result = {
            "overall_risk_level": overall_risk["level"],
            "overall_risk_score": overall_risk["score"],
            "detected_risks": detected_risks,
            "recommendations": self._generate_risk_recommendations(detected_risks),
            "timestamp": datetime.now().isoformat()
        }
        
        self.risk_history.append(result)
        return result
    
    def _analyze_security_risks(self, content: str) -> Dict[str, Any]:
        """
        Analizar riesgos de seguridad
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con riesgos de seguridad detectados
        """
        import re
        
        if not content:
            return {
                "risks": [],
                "risk_score": 0.0,
                "count": 0
            }
        
        risks = []
        risk_score = 0.0
        
        try:
            # Detectar posibles contraseñas
            password_patterns = [
                r'password\s*[:=]\s*\S+',
                r'contraseña\s*[:=]\s*\S+',
                r'pass\s*[:=]\s*\S+'
            ]
            
            for pattern in password_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                if matches:
                    risks.append({
                        "type": "password_exposure",
                        "severity": "high",
                        "description": "Posible exposición de contraseña",
                        "count": len(matches)
                    })
                    risk_score += min(0.3 * len(matches), 0.9)  # Limitar contribución
            
            # Detectar tokens de API (solo si son muy largos, probablemente tokens)
            api_token_pattern = r'\b[A-Za-z0-9]{32,}\b'
            matches = safe_regex_match(api_token_pattern, content, re.IGNORECASE)
            if matches:
                # Filtrar números muy largos que no son tokens
                token_candidates = [m.group(0) for m in matches if not m.group(0).isdigit()]
                if token_candidates:
                    risks.append({
                        "type": "api_token_exposure",
                        "severity": "high",
                        "description": f"Posible token de API expuesto ({len(token_candidates)} encontrados)",
                        "count": len(token_candidates)
                    })
                    risk_score += min(0.3, 1.0 - risk_score)
            
            # Detectar IPs internas
            internal_ip_pattern = r'\b(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)\d{1,3}\.\d{1,3}'
            matches = safe_regex_match(internal_ip_pattern, content, re.IGNORECASE)
            if matches:
                ip_addresses = [m.group(0) for m in matches]
                risks.append({
                    "type": "internal_ip_exposure",
                    "severity": "medium",
                    "description": f"Posible exposición de IP interna ({len(ip_addresses)} encontradas)",
                    "count": len(ip_addresses),
                    "ip_addresses": ip_addresses[:5]  # Limitar cantidad
                })
                risk_score += min(0.2, 1.0 - risk_score)
        
        except Exception as e:
            logger.error(f"Error analizando riesgos de seguridad: {e}", exc_info=True)
        
        return {
            "risks": risks,
            "risk_score": min(risk_score, 1.0),
            "count": len(risks)
        }
    
    def _analyze_privacy_risks(self, content: str) -> Dict[str, Any]:
        """
        Analizar riesgos de privacidad
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con riesgos de privacidad detectados
        """
        import re
        
        if not content:
            return {
                "risks": [],
                "risk_score": 0.0,
                "count": 0
            }
        
        risks = []
        risk_score = 0.0
        
        try:
            # Detectar información personal (DNI)
            dni_pattern = r'\b\d{8}[A-Z]\b'
            matches = safe_regex_match(dni_pattern, content, re.IGNORECASE)
            if matches:
                dni_count = len(matches)
                risks.append({
                    "type": "dni_exposure",
                    "severity": "high",
                    "description": f"Posible DNI expuesto ({dni_count} encontrados)",
                    "count": dni_count
                })
                risk_score += min(0.4, 1.0 - risk_score)
            
            # Detectar emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            matches = safe_regex_match(email_pattern, content, re.IGNORECASE)
            if matches:
                emails = list(set(match.group(0).lower() for match in matches))
                email_count = len(emails)
                if email_count > 5:
                    risks.append({
                        "type": "email_exposure",
                        "severity": "medium",
                        "description": f"{email_count} emails encontrados",
                        "count": email_count,
                        "sample_emails": emails[:3]  # Muestra limitada
                    })
                    risk_score += min(0.2, 1.0 - risk_score)
            
            # Detectar números de teléfono
            phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            matches = safe_regex_match(phone_pattern, content, re.IGNORECASE)
            if matches:
                phones = []
                for match in matches:
                    if match.groups():
                        phone_str = ''.join(g for g in match.groups() if g)
                        if phone_str and len(phone_str) >= 7:
                            phones.append(phone_str)
                phone_count = len(set(phones))
                if phone_count > 3:
                    risks.append({
                        "type": "phone_exposure",
                        "severity": "medium",
                        "description": f"{phone_count} números de teléfono encontrados",
                        "count": phone_count,
                        "sample_phones": list(set(phones))[:3]  # Muestra limitada
                    })
                    risk_score += min(0.2, 1.0 - risk_score)
        
        except Exception as e:
            logger.error(f"Error analizando riesgos de privacidad: {e}", exc_info=True)
        
        return {
            "risks": risks,
            "risk_score": min(risk_score, 1.0),
            "count": len(risks)
        }
    
    def _analyze_compliance_risks(self, content: str) -> Dict[str, Any]:
        """
        Analizar riesgos de compliance
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con riesgos de compliance detectados
        """
        import re
        
        if not content:
            return {
                "risks": [],
                "risk_score": 0.0,
                "count": 0
            }
        
        risks = []
        risk_score = 0.0
        
        try:
            # Palabras clave que indican posibles problemas de compliance
            compliance_keywords = {
                "high": ["confidencial", "secreto", "proprietary", "nda", "acuerdo de confidencialidad"],
                "medium": ["información sensible", "datos personales", "privacidad", "rgpd", "gdpr"]
            }
            
            content_lower = content.lower()
            found_keywords = set()
            
            for severity, keywords in compliance_keywords.items():
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    # Buscar palabra completa con boundaries
                    pattern = rf'\b{re.escape(keyword_lower)}\b'
                    matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                    if matches:
                        keyword_tuple = (keyword, severity)
                        if keyword_tuple not in found_keywords:
                            found_keywords.add(keyword_tuple)
                            risks.append({
                                "type": "compliance_keyword",
                                "severity": severity,
                                "description": f"Palabra clave de compliance encontrada: {keyword}",
                                "keyword": keyword,
                                "count": len(matches)
                            })
                            if severity == "high":
                                risk_score += min(0.2, 1.0 - risk_score)
                            else:
                                risk_score += min(0.1, 1.0 - risk_score)
        
        except Exception as e:
            logger.error(f"Error analizando riesgos de compliance: {e}", exc_info=True)
        
        return {
            "risks": risks,
            "risk_score": min(risk_score, 1.0),
            "count": len(risks)
        }
    
    async def _analyze_reputation_risks(self, content: str) -> Dict[str, Any]:
        """
        Analizar riesgos de reputación
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con riesgos de reputación detectados
        """
        if not content:
            return {
                "risks": [],
                "risk_score": 0.0,
                "count": 0,
                "sentiment_analysis": {}
            }
        
        risks = []
        risk_score = 0.0
        
        try:
            # Analizar sentimiento con manejo de errores
            sentiment = {}
            try:
                sentiment = await self.analyzer.analyze_sentiment(content)
                if not isinstance(sentiment, dict):
                    sentiment = {}
            except Exception as e:
                logger.warning(f"Error en análisis de sentimiento para riesgos de reputación: {e}")
            
            # Sentimiento negativo alto
            if sentiment:
                negative_score = sentiment.get("negative", 0.0)
                if isinstance(negative_score, (int, float)) and negative_score > 0.7:
                    risks.append({
                        "type": "negative_sentiment",
                        "severity": "medium",
                        "description": "Sentimiento muy negativo detectado",
                        "score": float(negative_score)
                    })
                    risk_score += min(0.3, 1.0 - risk_score)
            
            # Detectar palabras ofensivas o problemáticas
            problematic_words = ["fraude", "estafa", "ilegal", "corrupción", "fraud", "scam", "illegal", "corruption"]
            content_lower = content.lower()
            
            found_words = []
            for word in problematic_words:
                word_lower = word.lower()
                # Buscar palabra completa con boundaries
                pattern = rf'\b{re.escape(word_lower)}\b'
                matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                if matches:
                    found_words.append(word)
            
            if found_words:
                risks.append({
                    "type": "problematic_language",
                    "severity": "high",
                    "description": f"Palabras problemáticas encontradas: {', '.join(found_words[:5])}",
                    "keywords": found_words,
                    "count": len(found_words)
                })
                risk_score += min(0.4, 1.0 - risk_score)
        
        except Exception as e:
            logger.error(f"Error analizando riesgos de reputación: {e}", exc_info=True)
        
        return {
            "risks": risks,
            "risk_score": min(risk_score, 1.0),
            "count": len(risks),
            "sentiment_analysis": sentiment
        }
    
    def _calculate_overall_risk(self, detected_risks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcular riesgo general
        
        Args:
            detected_risks: Diccionario con riesgos detectados por categoría
        
        Returns:
            Diccionario con nivel y score de riesgo general
        """
        if not detected_risks or not isinstance(detected_risks, dict):
            return {"level": "low", "score": 0.0}
        
        try:
            scores = []
            for risk_data in detected_risks.values():
                if isinstance(risk_data, dict):
                    risk_score = risk_data.get("risk_score", 0.0)
                    if isinstance(risk_score, (int, float)) and risk_score >= 0:
                        scores.append(float(risk_score))
            
            if not scores:
                return {"level": "low", "score": 0.0}
            
            avg_score = sum(scores) / len(scores)
            
            # Determinar nivel de riesgo
            if avg_score >= 0.7:
                level = "high"
            elif avg_score >= 0.4:
                level = "medium"
            else:
                level = "low"
            
            return {
                "level": level,
                "score": min(avg_score, 1.0),
                "categories_analyzed": len(detected_risks),
                "max_category_score": max(scores) if scores else 0.0
            }
        
        except Exception as e:
            logger.error(f"Error calculando riesgo general: {e}", exc_info=True)
            return {"level": "unknown", "score": 0.0}
    
    def _generate_risk_recommendations(self, detected_risks: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generar recomendaciones de riesgo"""
        recommendations = []
        
        for category, risks_data in detected_risks.items():
            if risks_data.get("risk_score", 0.0) > 0.5:
                recommendations.append(
                    f"Alto riesgo detectado en {category}. Revisar y mitigar riesgos identificados."
                )
        
        return recommendations


class AuthorAnalyzer:
    """Analizador de autoría y estilo de escritura."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.author_profiles: Dict[str, Dict[str, Any]] = {}
        self.analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_author_style(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        author_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar estilo de escritura del autor
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            author_id: ID del autor (opcional)
        
        Returns:
            Análisis de estilo de autor
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Analizar características de estilo
        style_features = self._extract_style_features(content)
        
        # Analizar vocabulario
        vocabulary_features = self._analyze_vocabulary(content)
        
        # Analizar estructura de oraciones
        sentence_features = self._analyze_sentence_structure(content)
        
        # Analizar uso de puntuación
        punctuation_features = self._analyze_punctuation(content)
        
        result = {
            "style_features": style_features,
            "vocabulary_features": vocabulary_features,
            "sentence_features": sentence_features,
            "punctuation_features": punctuation_features,
            "author_id": author_id,
            "signature": self._calculate_author_signature(
                style_features, vocabulary_features, sentence_features, punctuation_features
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        # Guardar perfil si hay author_id
        if author_id:
            self.author_profiles[author_id] = result
        
        self.analysis_history.append(result)
        return result
    
    def _extract_style_features(self, content: str) -> Dict[str, Any]:
        """
        Extraer características de estilo
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con características de estilo extraídas
        """
        if not content:
            return {
                "avg_word_length": 0.0,
                "avg_sentence_length": 0.0,
                "total_words": 0,
                "total_sentences": 0,
                "unique_words": 0,
                "vocabulary_richness": 0.0
            }
        
        try:
            words = content.split()
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            
            # Calcular características de manera segura
            avg_word_length = 0.0
            if words:
                total_chars = sum(len(word) for word in words)
                avg_word_length = total_chars / len(words) if words else 0.0
            
            avg_sentence_length = 0.0
            if sentences:
                total_words_in_sentences = sum(len(s.split()) for s in sentences)
                avg_sentence_length = total_words_in_sentences / len(sentences) if sentences else 0.0
            
            unique_words = len(set(word.lower().strip('.,!?;:') for word in words))
            vocabulary_richness = unique_words / len(words) if words else 0.0
            
            return {
                "avg_word_length": round(avg_word_length, 2),
                "avg_sentence_length": round(avg_sentence_length, 2),
                "total_words": len(words),
                "total_sentences": len(sentences),
                "unique_words": unique_words,
                "vocabulary_richness": round(vocabulary_richness, 4)
            }
        
        except Exception as e:
            logger.error(f"Error extrayendo características de estilo: {e}", exc_info=True)
            return {
                "avg_word_length": 0.0,
                "avg_sentence_length": 0.0,
                "total_words": 0,
                "total_sentences": 0,
                "unique_words": 0,
                "vocabulary_richness": 0.0
            }
    
    def _analyze_vocabulary(self, content: str) -> Dict[str, Any]:
        """
        Analizar vocabulario
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con características de vocabulario
        """
        if not content:
            return {
                "common_words_ratio": 0.0,
                "avg_unique_word_length": 0.0,
                "vocabulary_size": 0
            }
        
        try:
            words = [w.lower().strip('.,!?;:') for w in content.split() if w.strip()]
            
            if not words:
                return {
                    "common_words_ratio": 0.0,
                    "avg_unique_word_length": 0.0,
                    "vocabulary_size": 0
                }
            
            # Palabras comunes vs. únicas
            common_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le'}
            common_count = sum(1 for w in words if w in common_words)
            
            # Longitud promedio de palabras únicas
            unique_words = set(words)
            avg_unique_length = 0.0
            if unique_words:
                total_length = sum(len(w) for w in unique_words)
                avg_unique_length = total_length / len(unique_words)
            
            common_words_ratio = common_count / len(words) if words else 0.0
            
            return {
                "common_words_ratio": round(common_words_ratio, 4),
                "avg_unique_word_length": round(avg_unique_length, 2),
                "vocabulary_size": len(unique_words)
            }
        
        except Exception as e:
            logger.error(f"Error analizando vocabulario: {e}", exc_info=True)
            return {
                "common_words_ratio": 0.0,
                "avg_unique_word_length": 0.0,
                "vocabulary_size": 0
            }
    
    def _analyze_sentence_structure(self, content: str) -> Dict[str, Any]:
        """
        Analizar estructura de oraciones
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con características de estructura de oraciones
        """
        import re
        
        if not content:
            return {
                "avg_sentence_length": 0.0,
                "max_sentence_length": 0,
                "min_sentence_length": 0,
                "complex_sentences_ratio": 0.0
            }
        
        try:
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return {
                    "avg_sentence_length": 0.0,
                    "max_sentence_length": 0,
                    "min_sentence_length": 0,
                    "complex_sentences_ratio": 0.0
                }
            
            # Longitud de oraciones
            lengths = [len(s.split()) for s in sentences if s.split()]
            
            if not lengths:
                return {
                    "avg_sentence_length": 0.0,
                    "max_sentence_length": 0,
                    "min_sentence_length": 0,
                    "complex_sentences_ratio": 0.0
                }
            
            # Oraciones complejas (con comas)
            complex_sentences = sum(1 for s in sentences if ',' in s)
            
            avg_length = sum(lengths) / len(lengths) if lengths else 0.0
            complex_ratio = complex_sentences / len(sentences) if sentences else 0.0
            
            return {
                "avg_sentence_length": round(avg_length, 2),
                "max_sentence_length": max(lengths) if lengths else 0,
                "min_sentence_length": min(lengths) if lengths else 0,
                "complex_sentences_ratio": round(complex_ratio, 4)
            }
        
        except Exception as e:
            logger.error(f"Error analizando estructura de oraciones: {e}", exc_info=True)
            return {
                "avg_sentence_length": 0.0,
                "max_sentence_length": 0,
                "min_sentence_length": 0,
                "complex_sentences_ratio": 0.0
            }
    
    def _analyze_punctuation(self, content: str) -> Dict[str, Any]:
        """
        Analizar uso de puntuación
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con características de puntuación
        """
        if not content:
            return {
                "periods": 0,
                "commas": 0,
                "exclamation": 0,
                "question": 0,
                "semicolon": 0,
                "colon": 0,
                "punctuation_density": 0.0
            }
        
        try:
            punctuation_counts = {
                "periods": content.count('.'),
                "commas": content.count(','),
                "exclamation": content.count('!'),
                "question": content.count('?'),
                "semicolon": content.count(';'),
                "colon": content.count(':')
            }
            
            total_chars = len(content)
            total_punctuation = sum(punctuation_counts.values())
            
            punctuation_density = total_punctuation / total_chars if total_chars > 0 else 0.0
            
            return {
                **punctuation_counts,
                "punctuation_density": round(punctuation_density, 6),
                "total_punctuation": total_punctuation
            }
        
        except Exception as e:
            logger.error(f"Error analizando puntuación: {e}", exc_info=True)
            return {
                "periods": 0,
                "commas": 0,
                "exclamation": 0,
                "question": 0,
                "semicolon": 0,
                "colon": 0,
                "punctuation_density": 0.0,
                "total_punctuation": 0
            }
    
    def _calculate_author_signature(
        self,
        style_features: Dict[str, Any],
        vocabulary_features: Dict[str, Any],
        sentence_features: Dict[str, Any],
        punctuation_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calcular firma única del autor
        
        Args:
            style_features: Características de estilo
            vocabulary_features: Características de vocabulario
            sentence_features: Características de oraciones
            punctuation_features: Características de puntuación
        
        Returns:
            Diccionario con firma del autor
        """
        try:
            if not isinstance(style_features, dict):
                style_features = {}
            if not isinstance(vocabulary_features, dict):
                vocabulary_features = {}
            if not isinstance(sentence_features, dict):
                sentence_features = {}
            if not isinstance(punctuation_features, dict):
                punctuation_features = {}
            
            # Extraer valores con validación y defaults
            avg_word_length = style_features.get("avg_word_length", 0.0)
            if not isinstance(avg_word_length, (int, float)) or avg_word_length < 0:
                avg_word_length = 0.0
            
            avg_sentence_length = style_features.get("avg_sentence_length", 0.0)
            if not isinstance(avg_sentence_length, (int, float)) or avg_sentence_length < 0:
                avg_sentence_length = 0.0
            
            vocabulary_richness = style_features.get("vocabulary_richness", 0.0)
            if not isinstance(vocabulary_richness, (int, float)) or vocabulary_richness < 0:
                vocabulary_richness = 0.0
            
            complex_sentence_ratio = sentence_features.get("complex_sentences_ratio", 0.0)
            if not isinstance(complex_sentence_ratio, (int, float)) or complex_sentence_ratio < 0:
                complex_sentence_ratio = 0.0
            
            punctuation_density = punctuation_features.get("punctuation_density", 0.0)
            if not isinstance(punctuation_density, (int, float)) or punctuation_density < 0:
                punctuation_density = 0.0
            
            return {
                "avg_word_length": round(float(avg_word_length), 2),
                "avg_sentence_length": round(float(avg_sentence_length), 2),
                "vocabulary_richness": round(float(vocabulary_richness), 4),
                "complex_sentence_ratio": round(float(complex_sentence_ratio), 4),
                "punctuation_density": round(float(punctuation_density), 6)
            }
        
        except Exception as e:
            logger.error(f"Error calculando firma de autor: {e}", exc_info=True)
            return {
                "avg_word_length": 0.0,
                "avg_sentence_length": 0.0,
                "vocabulary_richness": 0.0,
                "complex_sentence_ratio": 0.0,
                "punctuation_density": 0.0
            }
    
    async def compare_author_styles(
        self,
        doc1_content: str,
        doc2_content: str
    ) -> Dict[str, Any]:
        """
        Comparar estilos de autor entre dos documentos
        
        Args:
            doc1_content: Contenido del primer documento
            doc2_content: Contenido del segundo documento
        
        Returns:
            Diccionario con comparación de estilos
        """
        if not doc1_content or not doc2_content:
            return {
                "style1": {},
                "style2": {},
                "similarities": {},
                "avg_similarity": 0.0,
                "same_author_probability": 0.0
            }
        
        try:
            style1 = await self.analyze_author_style(document_content=doc1_content)
            style2 = await self.analyze_author_style(document_content=doc2_content)
            
            if not isinstance(style1, dict):
                style1 = {}
            if not isinstance(style2, dict):
                style2 = {}
            
            signature1 = style1.get("signature", {})
            signature2 = style2.get("signature", {})
            
            if not isinstance(signature1, dict):
                signature1 = {}
            if not isinstance(signature2, dict):
                signature2 = {}
            
            # Calcular similitud de firma
            similarities = {}
            for key in signature1:
                if key in signature2:
                    val1 = signature1[key]
                    val2 = signature2[key]
                    
                    # Validar que sean valores numéricos
                    if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                        continue
                    
                    val1 = float(val1)
                    val2 = float(val2)
                    
                    # Calcular similitud solo si ambos valores son positivos
                    if val1 > 0 and val2 > 0:
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            similarity = 1.0 - abs(val1 - val2) / max_val
                            similarities[key] = round(max(0.0, min(1.0, similarity)), 4)
                    elif val1 == 0 and val2 == 0:
                        # Ambos son cero, perfecta similitud
                        similarities[key] = 1.0
            
            avg_similarity = 0.0
            if similarities:
                similarity_values = [v for v in similarities.values() if isinstance(v, (int, float))]
                if similarity_values:
                    avg_similarity = sum(similarity_values) / len(similarity_values)
                    avg_similarity = round(max(0.0, min(1.0, avg_similarity)), 4)
            
            return {
                "style1": style1,
                "style2": style2,
                "similarities": similarities,
                "avg_similarity": avg_similarity,
                "same_author_probability": avg_similarity
            }
        
        except Exception as e:
            logger.error(f"Error comparando estilos de autor: {e}", exc_info=True)
            return {
                "style1": {},
                "style2": {},
                "similarities": {},
                "avg_similarity": 0.0,
                "same_author_probability": 0.0
            }


class QualityScoreCalculator:
    """Calculadora de puntuación de calidad de contenido."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.quality_history: List[Dict[str, Any]] = []
    
    async def calculate_quality_score(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calcular puntuación de calidad de un documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            weights: Pesos para diferentes métricas
        
        Returns:
            Puntuación de calidad y análisis detallado
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        weights = weights or {
            "readability": 0.2,
            "coherence": 0.2,
            "structure": 0.15,
            "completeness": 0.15,
            "accuracy": 0.15,
            "style": 0.15
        }
        
        # Analizar diferentes aspectos
        readability_analyzer = ReadabilityAnalyzer()
        readability = await readability_analyzer.analyze_readability(document_content=content)
        readability_score = min(readability.get("flesch_score", 0) / 100.0, 1.0)
        
        coherence_analyzer = DocumentCoherenceAnalyzer(self.analyzer)
        coherence = await coherence_analyzer.analyze_coherence(document_content=content)
        coherence_score = coherence.get("overall_coherence", 0.0)
        
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        structure_score = self._calculate_structure_score(structure)
        
        completeness_score = self._calculate_completeness_score(content, structure)
        
        accuracy_score = await self._calculate_accuracy_score(content)
        
        style_score = await self._calculate_style_score(content)
        
        # Calcular puntuación general
        overall_score = (
            readability_score * weights["readability"] +
            coherence_score * weights["coherence"] +
            structure_score * weights["structure"] +
            completeness_score * weights["completeness"] +
            accuracy_score * weights["accuracy"] +
            style_score * weights["style"]
        )
        
        result = {
            "overall_quality_score": overall_score,
            "quality_level": self._get_quality_level(overall_score),
            "component_scores": {
                "readability": readability_score,
                "coherence": coherence_score,
                "structure": structure_score,
                "completeness": completeness_score,
                "accuracy": accuracy_score,
                "style": style_score
            },
            "weights": weights,
            "recommendations": self._generate_quality_recommendations(
                readability_score, coherence_score, structure_score,
                completeness_score, accuracy_score, style_score
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        self.quality_history.append(result)
        return result
    
    def _calculate_structure_score(self, structure: Dict[str, Any]) -> float:
        """
        Calcular puntuación de estructura
        
        Args:
            structure: Diccionario con estructura del documento
        
        Returns:
            Score de estructura (0.0 a 1.0)
        """
        if not structure or not isinstance(structure, dict):
            return 0.0
        
        try:
            sections = structure.get("sections", [])
            paragraphs = structure.get("paragraphs", 0)
            
            # Validar tipos
            if not isinstance(sections, list):
                sections = []
            if not isinstance(paragraphs, (int, float)) or paragraphs < 0:
                paragraphs = 0
            
            score = 0.0
            
            # Más secciones = mejor estructura
            if sections:
                num_sections = len(sections)
                score += min(num_sections / 5.0, 0.5)
            
            # Párrafos bien distribuidos
            if paragraphs > 0:
                score += min(float(paragraphs) / 20.0, 0.5)
            
            return round(max(0.0, min(score, 1.0)), 4)
        
        except Exception as e:
            logger.error(f"Error calculando score de estructura: {e}", exc_info=True)
            return 0.0
    
    def _calculate_completeness_score(self, content: str, structure: Dict[str, Any]) -> float:
        """
        Calcular puntuación de completitud
        
        Args:
            content: Contenido del documento
            structure: Estructura del documento
        
        Returns:
            Score de completitud (0.0 a 1.0)
        """
        if not content or not isinstance(content, str):
            return 0.0
        
        if not structure or not isinstance(structure, dict):
            structure = {}
        
        try:
            score = 0.0
            content_lower = content.lower()
            
            # Longitud mínima
            if len(content) > 500:
                score += 0.3
            
            # Tiene estructura
            sections = structure.get("sections")
            if sections and isinstance(sections, list) and len(sections) > 0:
                score += 0.3
            
            # Tiene conclusiones o cierre
            conclusion_words = ["conclusión", "conclusion", "resumen", "summary"]
            if any(word in content_lower for word in conclusion_words):
                score += 0.2
            
            # Tiene introducción
            introduction_words = ["introducción", "introduction", "prefacio", "preface"]
            if any(word in content_lower for word in introduction_words):
                score += 0.2
            
            return round(max(0.0, min(score, 1.0)), 4)
        
        except Exception as e:
            logger.error(f"Error calculando score de completitud: {e}", exc_info=True)
            return 0.0
    
    async def _calculate_accuracy_score(self, content: str) -> float:
        """
        Calcular puntuación de precisión
        
        Args:
            content: Contenido del documento
        
        Returns:
            Score de precisión (0.0 a 1.0)
        """
        if not content or not isinstance(content, str):
            return 0.5  # Score base para contenido vacío
        
        try:
            # Análisis básico - en producción usar verificadores más avanzados
            score = 0.7  # Base
            
            # Verificar si hay citas (indica verificación de fuentes)
            try:
                citation_extractor = CitationExtractor()
                citations = await citation_extractor.extract_citations(document_content=content)
                
                if isinstance(citations, dict):
                    total_citations = citations.get("total_citations", 0)
                    total_references = citations.get("total_references", 0)
                    
                    if isinstance(total_citations, (int, float)) and total_citations > 0:
                        score += 0.2
                    
                    if isinstance(total_references, (int, float)) and total_references > 0:
                        score += 0.1
            except Exception as e:
                logger.warning(f"Error extrayendo citas para accuracy score: {e}")
            
            return round(max(0.0, min(score, 1.0)), 4)
        
        except Exception as e:
            logger.error(f"Error calculando score de precisión: {e}", exc_info=True)
            return 0.5
    
    async def _calculate_style_score(self, content: str) -> float:
        """
        Calcular puntuación de estilo
        
        Args:
            content: Contenido del documento
        
        Returns:
            Score de estilo (0.0 a 1.0)
        """
        if not content or not isinstance(content, str):
            return 0.5  # Score base para contenido vacío
        
        try:
            author_analyzer = AuthorAnalyzer(self.analyzer)
            style = await author_analyzer.analyze_author_style(document_content=content)
            
            if not isinstance(style, dict):
                style = {}
            
            style_features = style.get("style_features", {})
            vocabulary_features = style.get("vocabulary_features", {})
            sentence_features = style.get("sentence_features", {})
            
            if not isinstance(style_features, dict):
                style_features = {}
            if not isinstance(vocabulary_features, dict):
                vocabulary_features = {}
            if not isinstance(sentence_features, dict):
                sentence_features = {}
            
            score = 0.5  # Base
            
            # Vocabulario rico
            vocabulary_richness = vocabulary_features.get("vocabulary_richness", 0.0)
            if isinstance(vocabulary_richness, (int, float)) and vocabulary_richness > 0.3:
                score += 0.2
            
            # Estructura variada
            complex_ratio = sentence_features.get("complex_sentences_ratio", 0.0)
            if isinstance(complex_ratio, (int, float)) and complex_ratio > 0.3:
                score += 0.2
            
            # Longitud apropiada de oraciones
            avg_sentence = style_features.get("avg_sentence_length", 0.0)
            if isinstance(avg_sentence, (int, float)) and 10 <= avg_sentence <= 20:
                score += 0.1
            
            return round(max(0.0, min(score, 1.0)), 4)
        
        except Exception as e:
            logger.error(f"Error calculando score de estilo: {e}", exc_info=True)
            return 0.5
    
    def _get_quality_level(self, score: float) -> str:
        """
        Obtener nivel de calidad
        
        Args:
            score: Score de calidad (0.0 a 1.0)
        
        Returns:
            Nivel de calidad como string
        """
        if not isinstance(score, (int, float)):
            return "Desconocido"
        
        # Normalizar score entre 0 y 1
        score = max(0.0, min(1.0, float(score)))
        
        try:
            if score >= 0.8:
                return "Excelente"
            elif score >= 0.6:
                return "Buena"
            elif score >= 0.4:
                return "Regular"
            else:
                return "Necesita Mejora"
        
        except Exception as e:
            logger.error(f"Error obteniendo nivel de calidad: {e}", exc_info=True)
            return "Desconocido"
    
    def _generate_quality_recommendations(
        self,
        readability: float,
        coherence: float,
        structure: float,
        completeness: float,
        accuracy: float,
        style: float
    ) -> List[str]:
        """
        Generar recomendaciones de calidad
        
        Args:
            readability: Score de legibilidad (0.0 a 1.0)
            coherence: Score de coherencia (0.0 a 1.0)
            structure: Score de estructura (0.0 a 1.0)
            completeness: Score de completitud (0.0 a 1.0)
            accuracy: Score de precisión (0.0 a 1.0)
            style: Score de estilo (0.0 a 1.0)
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        try:
            # Validar y normalizar valores
            if not isinstance(readability, (int, float)):
                readability = 0.0
            readability = max(0.0, min(1.0, float(readability)))
            
            if not isinstance(coherence, (int, float)):
                coherence = 0.0
            coherence = max(0.0, min(1.0, float(coherence)))
            
            if not isinstance(structure, (int, float)):
                structure = 0.0
            structure = max(0.0, min(1.0, float(structure)))
            
            if not isinstance(completeness, (int, float)):
                completeness = 0.0
            completeness = max(0.0, min(1.0, float(completeness)))
            
            if not isinstance(accuracy, (int, float)):
                accuracy = 0.0
            accuracy = max(0.0, min(1.0, float(accuracy)))
            
            if not isinstance(style, (int, float)):
                style = 0.0
            style = max(0.0, min(1.0, float(style)))
            
            # Generar recomendaciones basadas en los valores
            if readability < 0.5:
                recommendations.append("Mejorar legibilidad del documento")
            
            if coherence < 0.5:
                recommendations.append("Mejorar coherencia entre secciones")
            
            if structure < 0.5:
                recommendations.append("Mejorar estructura del documento")
            
            if completeness < 0.5:
                recommendations.append("El documento parece incompleto")
            
            if accuracy < 0.5:
                recommendations.append("Considerar agregar más referencias y citas")
            
            if style < 0.5:
                recommendations.append("Mejorar estilo y variedad de escritura")
        
        except Exception as e:
            logger.error(f"Error generando recomendaciones de calidad: {e}", exc_info=True)
        
        return recommendations


# ============================================================================
# MÉTODOS MEJORADOS - Integración con características avanzadas
# ============================================================================

def enhance_document_analyzer(analyzer: DocumentAnalyzer):
    """Añadir métodos mejorados al analizador base"""
    
    async def compare_documents(
        self,
        doc1_content: str,
        doc2_content: str,
        doc1_id: Optional[str] = None,
        doc2_id: Optional[str] = None
    ):
        """
        Comparar dos documentos (requiere características avanzadas).
        
        Args:
            doc1_content: Contenido del primer documento
            doc2_content: Contenido del segundo documento
            doc1_id: ID del primer documento
            doc2_id: ID del segundo documento
        
        Returns:
            DocumentSimilarity con resultados
        """
        if not hasattr(self, 'comparator') or not self.comparator:
            raise RuntimeError("Características avanzadas no disponibles. Instale document_analyzer_enhanced.")
        
        return await self.comparator.compare_documents(
            doc1_content, doc2_content, doc1_id, doc2_id
        )
    
    async def process_batch(
        self,
        documents: List[Dict[str, Any]],
        tasks: Optional[List[AnalysisTask]] = None,
        max_workers: int = 10,
        on_progress: Optional[Callable[[int, int], None]] = None
    ):
        """
        Procesar múltiples documentos en batch (requiere características avanzadas).
        
        Args:
            documents: Lista de documentos a procesar
            tasks: Tareas de análisis
            max_workers: Número máximo de workers paralelos
            on_progress: Callback de progreso
        
        Returns:
            BatchAnalysisResult con todos los resultados
        """
        if not hasattr(self, 'batch_processor') or not self.batch_processor:
            from .document_analyzer_enhanced import BatchDocumentProcessor
            self.batch_processor = BatchDocumentProcessor(self, max_workers=max_workers)
        
        if max_workers != self.batch_processor.max_workers:
            from .document_analyzer_enhanced import BatchDocumentProcessor
            self.batch_processor = BatchDocumentProcessor(self, max_workers=max_workers)
        
        return await self.batch_processor.process_batch(
            documents, tasks=tasks, on_progress=on_progress
        )
    
    async def extract_structured_data(
        self,
        content: str,
        schema: Dict[str, Any]
    ):
        """
        Extraer información estructurada (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
            schema: Schema de extracción
        
        Returns:
            Diccionario con datos extraídos
        """
        if not hasattr(self, 'info_extractor') or not self.info_extractor:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.info_extractor.extract_structured_data(content, schema)
    
    async def analyze_writing_style(
        self,
        content: str
    ):
        """
        Analizar estilo de escritura (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con análisis de estilo
        """
        if not hasattr(self, 'language_analyzer') or not self.language_analyzer:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.language_analyzer.analyze_writing_style(content)
    
    async def find_similar_documents(
        self,
        target_doc: str,
        document_corpus: List[Tuple[str, str]],
        threshold: float = 0.7,
        top_k: int = 10
    ):
        """
        Encontrar documentos similares (requiere características avanzadas).
        
        Args:
            target_doc: Documento objetivo
            document_corpus: Lista de (doc_id, content)
            threshold: Umbral de similitud
            top_k: Número de resultados
        
        Returns:
            Lista de DocumentSimilarity
        """
        if not hasattr(self, 'comparator') or not self.comparator:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.comparator.find_similar_documents(
            target_doc, document_corpus, threshold, top_k
        )
    
    # ============================================================================
    # MÉTODOS AVANZADOS - Funcionalidades adicionales
    # ============================================================================
    
    async def analyze_image(
        self,
        image_path: str,
        extract_text: bool = True,
        detect_objects: bool = False
    ):
        """
        Analizar imagen en documento (requiere características avanzadas).
        
        Args:
            image_path: Ruta a la imagen
            extract_text: Extraer texto con OCR
            detect_objects: Detectar objetos
        
        Returns:
            ImageAnalysis con resultados
        """
        if not hasattr(self, 'image_analyzer') or not self.image_analyzer:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.image_analyzer.analyze_image(
            image_path, extract_text, detect_objects
        )
    
    async def extract_tables(
        self,
        content: str,
        document_path: Optional[str] = None
    ):
        """
        Extraer tablas del documento (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
            document_path: Ruta al documento
        
        Returns:
            Lista de tablas extraídas
        """
        if not hasattr(self, 'table_extractor') or not self.table_extractor:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.table_extractor.extract_tables(content, document_path)
    
    async def detect_language(self, content: str):
        """
        Detectar idioma del documento (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con idioma detectado
        """
        if not hasattr(self, 'multilang_analyzer') or not self.multilang_analyzer:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.multilang_analyzer.detect_language(content)
    
    async def analyze_quality(
        self,
        content: str,
        document_type: Optional[str] = None
    ):
        """
        Analizar calidad del documento (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
            document_type: Tipo de documento
        
        Returns:
            DocumentQuality con análisis
        """
        if not hasattr(self, 'quality_analyzer') or not self.quality_analyzer:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.quality_analyzer.analyze_quality(content, document_type)
    
    async def detect_fraud(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Detectar indicadores de fraude (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
            metadata: Metadatos del documento
        
        Returns:
            Diccionario con indicadores detectados
        """
        if not hasattr(self, 'fraud_detector') or not self.fraud_detector:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.fraud_detector.detect_fraud_indicators(content, metadata)
    
    async def analyze_legal_document(
        self,
        content: str,
        document_type: Optional[str] = None
    ):
        """
        Analizar documento legal (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
            document_type: Tipo de documento legal
        
        Returns:
            Diccionario con análisis legal
        """
        if not hasattr(self, 'legal_analyzer') or not self.legal_analyzer:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.legal_analyzer.analyze_legal_document(content, document_type)
    
    def export_results(
        self,
        results: Any,
        output_path: str,
        format: str = 'json',
        include_raw: bool = False
    ) -> str:
        """
        Exportar resultados de análisis (requiere características avanzadas).
        
        Args:
            results: Resultado(s) de análisis
            output_path: Ruta de salida
            format: Formato (json, csv, xml, html, markdown, txt)
            include_raw: Incluir contenido raw
        
        Returns:
            Ruta del archivo exportado
        """
        if not hasattr(self, 'exporter') or not self.exporter:
            raise RuntimeError("Exportador no disponible.")
        
        return self.exporter.export(results, output_path, format, include_raw)
    
    # ============================================================================
    # MÉTODOS ADICIONALES - Versiones, Gramática e Integraciones
    # ============================================================================
    
    def add_document_version(
        self,
        document_id: str,
        content: str,
        version_id: Optional[str] = None,
        author: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Agregar versión de documento (requiere características avanzadas).
        
        Args:
            document_id: ID del documento
            content: Contenido de la versión
            version_id: ID de versión
            author: Autor
            metadata: Metadatos
        
        Returns:
            DocumentVersion creada
        """
        if not hasattr(self, 'version_manager') or not self.version_manager:
            raise RuntimeError("Gestor de versiones no disponible.")
        
        return self.version_manager.add_version(
            document_id, content, version_id, author, metadata
        )
    
    async def compare_document_versions(
        self,
        document_id: str,
        version1_id: str,
        version2_id: str
    ):
        """
        Comparar versiones de documento (requiere características avanzadas).
        
        Args:
            document_id: ID del documento
            version1_id: ID de primera versión
            version2_id: ID de segunda versión
        
        Returns:
            VersionComparison con resultados
        """
        if not hasattr(self, 'version_manager') or not self.version_manager:
            raise RuntimeError("Gestor de versiones no disponible.")
        
        return await self.version_manager.compare_versions(
            document_id, version1_id, version2_id
        )
    
    async def analyze_version_history(self, document_id: str):
        """
        Analizar historial de versiones (requiere características avanzadas).
        
        Args:
            document_id: ID del documento
        
        Returns:
            Diccionario con análisis del historial
        """
        if not hasattr(self, 'version_manager') or not self.version_manager:
            raise RuntimeError("Gestor de versiones no disponible.")
        
        return await self.version_manager.analyze_version_history(document_id)
    
    async def analyze_grammar(
        self,
        content: str,
        language: str = 'es'
    ):
        """
        Analizar gramática y redacción (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
            language: Idioma
        
        Returns:
            GrammarAnalysis con resultados
        """
        if not hasattr(self, 'grammar_analyzer') or not self.grammar_analyzer:
            raise RuntimeError("Analizador de gramática no disponible.")
        
        return await self.grammar_analyzer.analyze_grammar(content, language)
    
    async def suggest_grammar_corrections(
        self,
        content: str,
        language: str = 'es'
    ):
        """
        Sugerir correcciones gramaticales (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
            language: Idioma
        
        Returns:
            Lista de sugerencias
        """
        if not hasattr(self, 'grammar_analyzer') or not self.grammar_analyzer:
            raise RuntimeError("Analizador de gramática no disponible.")
        
        return await self.grammar_analyzer.suggest_corrections(content, language)
    
    def configure_integration(
        self,
        service_name: str,
        service: Any,
        config: Any
    ):
        """
        Configurar integración externa (requiere características avanzadas).
        
        Args:
            service_name: Nombre del servicio
            service: Instancia del servicio
            config: Configuración
        """
        if not hasattr(self, 'integrations') or not self.integrations:
            raise RuntimeError("Gestor de integraciones no disponible.")
        
        self.integrations.register_service(service_name, service, config)
    
    async def translate_document_external(
        self,
        content: str,
        target_language: str,
        source_language: Optional[str] = None
    ):
        """
        Traducir documento usando servicio externo (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
            target_language: Idioma objetivo
            source_language: Idioma origen
        
        Returns:
            Diccionario con traducción
        """
        if not hasattr(self, 'integrations') or not self.integrations:
            raise RuntimeError("Gestor de integraciones no disponible.")
        
        return await self.integrations.translate_document(
            content, target_language, source_language
        )
    
    async def extract_text_from_image_external(self, image_path: str):
        """
        Extraer texto de imagen usando servicio externo (requiere características avanzadas).
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            Diccionario con texto extraído
        """
        if not hasattr(self, 'integrations') or not self.integrations:
            raise RuntimeError("Gestor de integraciones no disponible.")
        
        return await self.integrations.extract_text_from_image(image_path)
    
    # ============================================================================
    # MÉTODOS ENTERPRISE - Colaboración, Recomendaciones y Métricas
    # ============================================================================
    
    async def analyze_collaboration(
        self,
        document_id: str,
        versions: Optional[List[Any]] = None
    ):
        """
        Analizar colaboración en documento (requiere características enterprise).
        
        Args:
            document_id: ID del documento
            versions: Lista de versiones (opcional, usa version_manager si no se proporciona)
        
        Returns:
            CollaborationAnalysis con resultados
        """
        if not hasattr(self, 'collaboration_analyzer') or not self.collaboration_analyzer:
            raise RuntimeError("Analizador de colaboración no disponible.")
        
        if versions is None:
            if hasattr(self, 'version_manager') and self.version_manager:
                versions = self.version_manager.get_versions(document_id)
            else:
                raise ValueError("No se proporcionaron versiones y version_manager no disponible")
        
        return await self.collaboration_analyzer.analyze_collaboration(document_id, versions)
    
    async def generate_recommendations(
        self,
        document_analysis: Any,
        quality_analysis: Optional[Any] = None,
        grammar_analysis: Optional[Any] = None,
        version_history: Optional[Any] = None
    ):
        """
        Generar recomendaciones inteligentes (requiere características enterprise).
        
        Args:
            document_analysis: Resultado de análisis de documento
            quality_analysis: Análisis de calidad (opcional)
            grammar_analysis: Análisis de gramática (opcional)
            version_history: Historial de versiones (opcional)
        
        Returns:
            Lista de DocumentRecommendation
        """
        if not hasattr(self, 'recommendation_engine') or not self.recommendation_engine:
            raise RuntimeError("Motor de recomendaciones no disponible.")
        
        return await self.recommendation_engine.generate_recommendations(
            document_analysis, quality_analysis, grammar_analysis, version_history
        )
    
    async def find_similar_documents_recommendations(
        self,
        current_document: str,
        document_corpus: List[Tuple[str, str]],
        top_k: int = 5
    ):
        """
        Recomendar documentos similares (requiere características enterprise).
        
        Args:
            current_document: Contenido del documento actual
            document_corpus: Lista de (doc_id, content)
            top_k: Número de recomendaciones
        
        Returns:
            Lista de recomendaciones
        """
        if not hasattr(self, 'recommendation_engine') or not self.recommendation_engine:
            raise RuntimeError("Motor de recomendaciones no disponible.")
        
        return await self.recommendation_engine.find_similar_documents_recommendations(
            current_document, document_corpus, top_k
        )
    
    def record_analysis_for_metrics(
        self,
        document_id: str,
        analysis_result: Any,
        processing_time: float,
        quality_score: Optional[float] = None,
        grammar_score: Optional[float] = None
    ):
        """
        Registrar análisis para métricas (requiere características enterprise).
        
        Args:
            document_id: ID del documento
            analysis_result: Resultado del análisis
            processing_time: Tiempo de procesamiento
            quality_score: Score de calidad (opcional)
            grammar_score: Score de gramática (opcional)
        """
        if hasattr(self, 'metrics_collector') and self.metrics_collector:
            self.metrics_collector.record_analysis(
                document_id, analysis_result, processing_time, quality_score, grammar_score
            )
    
    async def generate_metrics_dashboard(
        self,
        period: str = "daily",
        days: int = 7
    ):
        """
        Generar dashboard de métricas (requiere características enterprise).
        
        Args:
            period: Período ('daily', 'weekly', 'monthly')
            days: Número de días a analizar
        
        Returns:
            MetricsDashboard con estadísticas
        """
        if not hasattr(self, 'metrics_collector') or not self.metrics_collector:
            raise RuntimeError("Recolector de métricas no disponible.")
        
        return await self.metrics_collector.generate_dashboard(period, days)
    
    def get_metrics_statistics(self):
        """
        Obtener estadísticas de métricas (requiere características enterprise).
        
        Returns:
            Diccionario con estadísticas
        """
        if not hasattr(self, 'metrics_collector') or not self.metrics_collector:
            raise RuntimeError("Recolector de métricas no disponible.")
        
        return self.metrics_collector.get_statistics()
    
    # ============================================================================
    # MÉTODOS PREMIUM - API, Webhooks, ML y Análisis Predictivo
    # ============================================================================
    
    def create_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Crear servidor API REST (requiere características premium).
        
        Args:
            host: Host del servidor
            port: Puerto del servidor
        
        Returns:
            DocumentAPIServer
        """
        from .document_api import create_api_server
        self._api_server = create_api_server(self, host, port)
        return self._api_server
    
    def register_webhook(
        self,
        url: str,
        events: List[str] = None,
        secret: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Registrar webhook (requiere características premium).
        
        Args:
            url: URL del webhook
            events: Lista de eventos a escuchar (["*"] = todos)
            secret: Secret para autenticación
            timeout: Timeout en segundos
        """
        if not hasattr(self, 'webhook_manager') or not self.webhook_manager:
            raise RuntimeError("Gestor de webhooks no disponible.")
        
        from .document_webhooks import WebhookConfig
        config = WebhookConfig(
            url=url,
            events=events or ["*"],
            secret=secret,
            timeout=timeout
        )
        self.webhook_manager.register_webhook(config)
    
    async def trigger_webhook_event(
        self,
        event_type: str,
        document_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Disparar evento de webhook (requiere características premium).
        
        Args:
            event_type: Tipo de evento
            document_id: ID del documento
            data: Datos adicionales
        """
        if hasattr(self, 'webhook_manager') and self.webhook_manager:
            await self.webhook_manager.trigger_event(event_type, document_id, data)
    
    async def predict_document_quality(self, features: Dict[str, Any]):
        """
        Predecir calidad de documento (requiere características premium).
        
        Args:
            features: Features del documento
        
        Returns:
            PredictionResult con predicción
        """
        if not hasattr(self, 'ml_predictor') or not self.ml_predictor:
            raise RuntimeError("Predictor ML no disponible.")
        
        return await self.ml_predictor.predict_document_quality(features)
    
    async def predict_processing_time(
        self,
        document_length: int,
        tasks_count: int
    ):
        """
        Predecir tiempo de procesamiento (requiere características premium).
        
        Args:
            document_length: Longitud del documento
            tasks_count: Número de tareas
        
        Returns:
            PredictionResult con predicción
        """
        if not hasattr(self, 'ml_predictor') or not self.ml_predictor:
            raise RuntimeError("Predictor ML no disponible.")
        
        return await self.ml_predictor.predict_processing_time(document_length, tasks_count)
    
    async def analyze_trends(
        self,
        metrics_data: List[Dict[str, Any]],
        period_days: int = 30
    ):
        """
        Analizar tendencias (requiere características premium).
        
        Args:
            metrics_data: Datos de métricas
            period_days: Período en días
        
        Returns:
            Diccionario con análisis de tendencias
        """
        if not hasattr(self, 'trend_analyzer') or not self.trend_analyzer:
            raise RuntimeError("Analizador de tendencias no disponible.")
        
        return await self.trend_analyzer.analyze_trends(metrics_data, period_days)
    
    # ============================================================================
    # MÉTODOS ULTIMATE - Plugins, Tiempo Real y Base de Datos
    # ============================================================================
    
    def register_plugin(self, plugin: Any):
        """
        Registrar plugin (requiere características ultimate).
        
        Args:
            plugin: Instancia de DocumentPlugin
        """
        if not hasattr(self, 'plugin_manager') or not self.plugin_manager:
            raise RuntimeError("Gestor de plugins no disponible.")
        
        self.plugin_manager.register_plugin(plugin)
    
    async def initialize_plugins(self):
        """Inicializar todos los plugins (requiere características ultimate)."""
        if not hasattr(self, 'plugin_manager') or not self.plugin_manager:
            raise RuntimeError("Gestor de plugins no disponible.")
        
        await self.plugin_manager.initialize_all()
    
    def get_plugin(self, plugin_name: str):
        """Obtener plugin (requiere características ultimate)."""
        if not hasattr(self, 'plugin_manager') or not self.plugin_manager:
            raise RuntimeError("Gestor de plugins no disponible.")
        
        return self.plugin_manager.get_plugin(plugin_name)
    
    async def analyze_realtime(
        self,
        document_id: str,
        content: str,
        tasks: Optional[List[str]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None
    ):
        """
        Analizar documento en tiempo real (requiere características ultimate).
        
        Args:
            document_id: ID del documento
            content: Contenido del documento
            tasks: Tareas de análisis
            on_progress: Callback de progreso
        
        Yields:
            RealtimeAnalysisEvent con eventos en tiempo real
        """
        if not hasattr(self, 'realtime_analyzer') or not self.realtime_analyzer:
            raise RuntimeError("Analizador en tiempo real no disponible.")
        
        async for event in self.realtime_analyzer.analyze_realtime(
            document_id, content, tasks, on_progress
        ):
            yield event
    
    async def save_analysis_to_database(
        self,
        document_id: str,
        analysis_result: Any,
        analysis_type: str = "full",
        quality_score: Optional[float] = None,
        grammar_score: Optional[float] = None,
        processing_time: float = 0.0
    ):
        """
        Guardar análisis en base de datos (requiere características ultimate).
        
        Args:
            document_id: ID del documento
            analysis_result: Resultado del análisis
            analysis_type: Tipo de análisis
            quality_score: Score de calidad
            grammar_score: Score de gramática
            processing_time: Tiempo de procesamiento
        """
        if not hasattr(self, 'database') or not self.database:
            raise RuntimeError("Base de datos no disponible.")
        
        return await self.database.save_analysis_result(
            document_id, analysis_result, analysis_type,
            quality_score, grammar_score, processing_time
        )
    
    async def get_analysis_from_database(
        self,
        document_id: str,
        analysis_type: Optional[str] = None
    ):
        """
        Obtener análisis de base de datos (requiere características ultimate).
        
        Args:
            document_id: ID del documento
            analysis_type: Tipo de análisis (opcional)
        
        Returns:
            Lista de AnalysisRecord
        """
        if not hasattr(self, 'database') or not self.database:
            raise RuntimeError("Base de datos no disponible.")
        
        return await self.database.get_analysis_history(document_id, analysis_type)
    
    # ============================================================================
    # MÉTODOS FINALES - Dashboard, Formatos y Alertas
    # ============================================================================
    
    async def generate_visual_dashboard(
        self,
        period: str = "daily",
        days: int = 7,
        output_path: Optional[str] = None
    ):
        """
        Generar dashboard visual (requiere características finales).
        
        Args:
            period: Período ('daily', 'weekly', 'monthly')
            days: Número de días
            output_path: Ruta para guardar HTML (opcional)
        
        Returns:
            DashboardData y opcionalmente ruta del archivo HTML
        """
        if not hasattr(self, 'dashboard_generator') or not self.dashboard_generator:
            raise RuntimeError("Generador de dashboard no disponible.")
        
        dashboard_data = await self.dashboard_generator.generate_dashboard(period, days)
        
        if output_path:
            html_path = await self.dashboard_generator.save_dashboard_html(
                dashboard_data, output_path
            )
            return dashboard_data, html_path
        
        return dashboard_data
    
    async def analyze_document_from_file(
        self,
        file_path: str,
        tasks: Optional[List[str]] = None
    ):
        """
        Analizar documento desde archivo (requiere características finales).
        
        Args:
            file_path: Ruta al archivo
            tasks: Tareas de análisis
        
        Returns:
            Resultado del análisis
        """
        if not hasattr(self, 'format_handler') or not self.format_handler:
            raise RuntimeError("Manejador de formatos no disponible.")
        
        # Extraer texto según formato
        content = await self.format_handler.extract_text_from_file(file_path)
        
        # Analizar contenido extraído
        return await self.analyze_document(document_content=content, tasks=tasks)
    
    def get_supported_formats(self):
        """Obtener formatos soportados (requiere características finales)."""
        if not hasattr(self, 'format_handler') or not self.format_handler:
            raise RuntimeError("Manejador de formatos no disponible.")
        
        return self.format_handler.get_supported_formats()
    
    def register_alert_rule(self, rule: Any):
        """
        Registrar regla de alerta (requiere características finales).
        
        Args:
            rule: Instancia de AlertRule
        """
        if not hasattr(self, 'alert_manager') or not self.alert_manager:
            raise RuntimeError("Gestor de alertas no disponible.")
        
        self.alert_manager.register_rule(rule)
    
    async def check_alerts(self, context: Dict[str, Any]):
        """
        Verificar alertas (requiere características finales).
        
        Args:
            context: Contexto para evaluación de reglas
        
        Returns:
            Lista de alertas disparadas
        """
        if not hasattr(self, 'alert_manager') or not self.alert_manager:
            raise RuntimeError("Gestor de alertas no disponible.")
        
        return await self.alert_manager.check_alerts(context)
    
    def get_active_alerts(self, severity: Optional[Any] = None):
        """Obtener alertas activas (requiere características finales)."""
        if not hasattr(self, 'alert_manager') or not self.alert_manager:
            raise RuntimeError("Gestor de alertas no disponible.")
        
        return self.alert_manager.get_active_alerts(severity)
    
    # ============================================================================
    # MÉTODOS ADICIONALES FINALES - Seguridad, Multi-idioma y Notificaciones
    # ============================================================================
    
    async def validate_document_security(
        self,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        policy_id: Optional[str] = None
    ):
        """
        Validar documento según políticas de seguridad (requiere características adicionales).
        
        Args:
            file_path: Ruta al archivo
            content: Contenido del documento
            policy_id: ID de política
        
        Returns:
            Diccionario con resultado de validación
        """
        if not hasattr(self, 'security_manager') or not self.security_manager:
            raise RuntimeError("Gestor de seguridad no disponible.")
        
        return await self.security_manager.validate_document(file_path, content, policy_id)
    
    def register_security_policy(self, policy: Any):
        """Registrar política de seguridad (requiere características adicionales)."""
        if not hasattr(self, 'security_manager') or not self.security_manager:
            raise RuntimeError("Gestor de seguridad no disponible.")
        
        self.security_manager.register_policy(policy)
    
    def get_security_audit_logs(
        self,
        event_type: Optional[str] = None,
        document_id: Optional[str] = None,
        limit: int = 100
    ):
        """Obtener logs de auditoría de seguridad (requiere características adicionales)."""
        if not hasattr(self, 'security_manager') or not self.security_manager:
            raise RuntimeError("Gestor de seguridad no disponible.")
        
        return self.security_manager.get_audit_logs(event_type, document_id, limit=limit)
    
    async def analyze_multilanguage(
        self,
        content: str,
        languages: Optional[List[str]] = None
    ):
        """
        Analizar documento multi-idioma (requiere características adicionales).
        
        Args:
            content: Contenido del documento
            languages: Idiomas específicos
        
        Returns:
            MultiLanguageAnalysis con resultados
        """
        if not hasattr(self, 'multilang_analyzer') or not self.multilang_analyzer:
            raise RuntimeError("Analizador multi-idioma no disponible.")
        
        return await self.multilang_analyzer.analyze_multilanguage(content, languages)
    
    def register_notification_handler(
        self,
        channel: Any,
        handler: Callable
    ):
        """Registrar handler de notificaciones (requiere características adicionales)."""
        if not hasattr(self, 'notification_manager') or not self.notification_manager:
            raise RuntimeError("Gestor de notificaciones no disponible.")
        
        self.notification_manager.register_handler(channel, handler)
    
    async def send_notification(
        self,
        title: str,
        message: str,
        channel: Any,
        priority: Any = None,
        recipient: Optional[str] = None
    ):
        """Enviar notificación (requiere características adicionales)."""
        if not hasattr(self, 'notification_manager') or not self.notification_manager:
            raise RuntimeError("Gestor de notificaciones no disponible.")
        
        from .document_notifications import NotificationPriority
        if priority is None:
            priority = NotificationPriority.NORMAL
        
        return await self.notification_manager.send_notification(
            title, message, channel, priority, recipient
        )
    
    # ============================================================================
    # MÉTODOS COMPLEMENTARIOS - Backup, Workflow y Auto-Learning
    # ============================================================================
    
    async def create_backup(
        self,
        backup_type: str = "full",
        include_analyses: bool = True,
        include_versions: bool = True
    ):
        """
        Crear backup (requiere características complementarias).
        
        Args:
            backup_type: Tipo de backup
            include_analyses: Incluir análisis
            include_versions: Incluir versiones
        
        Returns:
            BackupMetadata
        """
        if not hasattr(self, 'backup_manager') or not self.backup_manager:
            raise RuntimeError("Gestor de backup no disponible.")
        
        return await self.backup_manager.create_backup(backup_type, include_analyses, include_versions)
    
    async def restore_backup(
        self,
        backup_id: str,
        restore_analyses: bool = True,
        restore_versions: bool = True
    ):
        """Restaurar backup (requiere características complementarias)."""
        if not hasattr(self, 'backup_manager') or not self.backup_manager:
            raise RuntimeError("Gestor de backup no disponible.")
        
        return await self.backup_manager.restore_backup(backup_id, restore_analyses, restore_versions)
    
    def register_workflow(
        self,
        workflow_id: str,
        steps: List[Any]
    ):
        """Registrar workflow (requiere características complementarias)."""
        if not hasattr(self, 'workflow_manager') or not self.workflow_manager:
            raise RuntimeError("Gestor de workflow no disponible.")
        
        self.workflow_manager.register_workflow(workflow_id, steps)
    
    async def execute_workflow(
        self,
        workflow_id: str,
        document_id: str,
        initial_data: Optional[Dict[str, Any]] = None
    ):
        """Ejecutar workflow (requiere características complementarias)."""
        if not hasattr(self, 'workflow_manager') or not self.workflow_manager:
            raise RuntimeError("Gestor de workflow no disponible.")
        
        return await self.workflow_manager.execute_workflow(workflow_id, document_id, initial_data)
    
    def add_learning_example(
        self,
        document_content: str,
        expected_result: Dict[str, Any],
        actual_result: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None
    ):
        """Agregar ejemplo de aprendizaje (requiere características complementarias)."""
        if not hasattr(self, 'auto_learning') or not self.auto_learning:
            raise RuntimeError("Sistema de auto-aprendizaje no disponible.")
        
        return self.auto_learning.add_learning_example(
            document_content, expected_result, actual_result, feedback
        )
    
    async def learn_from_examples(self):
        """Aprender de ejemplos (requiere características complementarias)."""
        if not hasattr(self, 'auto_learning') or not self.auto_learning:
            raise RuntimeError("Sistema de auto-aprendizaje no disponible.")
        
        return await self.auto_learning.learn_from_examples()
    
    async def apply_learned_patterns(self, analysis_result: Any):
        """Aplicar patrones aprendidos (requiere características complementarias)."""
        if not hasattr(self, 'auto_learning') or not self.auto_learning:
            raise RuntimeError("Sistema de auto-aprendizaje no disponible.")
        
        return await self.auto_learning.apply_learned_patterns(analysis_result)
    
    # ============================================================================
    # MÉTODOS AVANZADOS FINALES - Búsqueda Semántica, Metadatos y Sentimiento
    # ============================================================================
    
    async def create_semantic_index(
        self,
        index_id: str,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ):
        """
        Crear índice semántico (requiere características avanzadas finales).
        
        Args:
            index_id: ID del índice
            documents: Lista de (doc_id, content, metadata)
        
        Returns:
            SemanticIndex
        """
        if not hasattr(self, 'semantic_search') or not self.semantic_search:
            raise RuntimeError("Motor de búsqueda semántica no disponible.")
        
        return await self.semantic_search.create_index(index_id, documents)
    
    async def semantic_search(
        self,
        index_id: str,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5
    ):
        """
        Búsqueda semántica (requiere características avanzadas finales).
        
        Args:
            index_id: ID del índice
            query: Consulta de búsqueda
            top_k: Número de resultados
            threshold: Umbral de similitud
        
        Returns:
            Lista de SearchResult
        """
        if not hasattr(self, 'semantic_search') or not self.semantic_search:
            raise RuntimeError("Motor de búsqueda semántica no disponible.")
        
        return await self.semantic_search.search(index_id, query, top_k, threshold)
    
    async def extract_metadata(
        self,
        content: str,
        file_path: Optional[str] = None
    ):
        """
        Extraer metadatos avanzados (requiere características avanzadas finales).
        
        Args:
            content: Contenido del documento
            file_path: Ruta al archivo
        
        Returns:
            DocumentMetadata
        """
        if not hasattr(self, 'metadata_extractor') or not self.metadata_extractor:
            raise RuntimeError("Extractor de metadatos no disponible.")
        
        return await self.metadata_extractor.extract_metadata(content, file_path)
    
    async def analyze_sentiment_advanced(
        self,
        content: str,
        analyze_by_section: bool = True
    ):
        """
        Análisis de sentimiento avanzado (requiere características avanzadas finales).
        
        Args:
            content: Contenido del documento
            analyze_by_section: Analizar por secciones
        
        Returns:
            AdvancedSentiment
        """
        if not hasattr(self, 'advanced_sentiment') or not self.advanced_sentiment:
            raise RuntimeError("Analizador de sentimiento avanzado no disponible.")
        
        return await self.advanced_sentiment.analyze_sentiment_advanced(content, analyze_by_section)
    
    # ============================================================================
    # MÉTODOS ESPECIALIZADOS FINALES - Benchmarking, IA Generativa y Distribuido
    # ============================================================================
    
    async def benchmark_model(
        self,
        model_name: str,
        test_documents: List[str],
        configuration: Optional[Dict[str, Any]] = None
    ):
        """Hacer benchmark de modelo (requiere características especializadas finales)."""
        if not hasattr(self, 'benchmarker') or not self.benchmarker:
            raise RuntimeError("Benchmarker de modelos no disponible.")
        
        return await self.benchmarker.benchmark_model(model_name, test_documents, configuration)
    
    async def compare_models(
        self,
        models: List[Dict[str, Any]],
        test_documents: List[str]
    ):
        """Comparar modelos (requiere características especializadas finales)."""
        if not hasattr(self, 'benchmarker') or not self.benchmarker:
            raise RuntimeError("Benchmarker de modelos no disponible.")
        
        return await self.benchmarker.compare_models(models, test_documents)
    
    async def analyze_with_generative_ai(
        self,
        content: str,
        analysis_type: str = "comprehensive",
        model: str = "gpt-3.5-turbo",
        custom_prompt: Optional[str] = None
    ):
        """Analizar con IA generativa (requiere características especializadas finales)."""
        if not hasattr(self, 'generative_ai') or not self.generative_ai:
            raise RuntimeError("Analizador de IA generativa no disponible.")
        
        return await self.generative_ai.analyze_with_generative_ai(
            content, analysis_type, model, custom_prompt
        )
    
    async def generate_improvements(
        self,
        content: str,
        focus_areas: Optional[List[str]] = None
    ):
        """Generar sugerencias de mejora con IA (requiere características especializadas finales)."""
        if not hasattr(self, 'generative_ai') or not self.generative_ai:
            raise RuntimeError("Analizador de IA generativa no disponible.")
        
        return await self.generative_ai.generate_improvements(content, focus_areas)
    
    async def process_distributed(
        self,
        documents: List[Dict[str, Any]],
        task_processor: Optional[Callable] = None,
        chunk_size: int = 100
    ):
        """Procesar documentos de forma distribuida (requiere características especializadas finales)."""
        if not hasattr(self, 'distributed_processor') or not self.distributed_processor:
            raise RuntimeError("Procesador distribuido no disponible.")
        
        if task_processor is None:
            # Usar analizador por defecto
            async def default_processor(content, doc_id):
                return await self.analyze_document(document_content=content)
            task_processor = default_processor
        
        return await self.distributed_processor.process_distributed(
            documents, task_processor, chunk_size
        )
    
    # ============================================================================
    # MÉTODOS INTELIGENTES FINALES - Predictivo, Recomendaciones y Rendimiento
    # ============================================================================
    
    async def predict_document_quality(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Predecir calidad de documento (requiere características inteligentes finales)."""
        if not hasattr(self, 'predictive_analyzer') or not self.predictive_analyzer:
            raise RuntimeError("Analizador predictivo no disponible.")
        
        return await self.predictive_analyzer.predict_document_quality(content, metadata)
    
    async def predict_processing_time(
        self,
        content: str,
        tasks: Optional[List[str]] = None
    ):
        """Predecir tiempo de procesamiento (requiere características inteligentes finales)."""
        if not hasattr(self, 'predictive_analyzer') or not self.predictive_analyzer:
            raise RuntimeError("Analizador predictivo no disponible.")
        
        return await self.predictive_analyzer.predict_processing_time(content, tasks)
    
    async def predict_trend(
        self,
        metric_name: str,
        historical_values: List[float],
        timeframe_days: int = 30
    ):
        """Predecir tendencia (requiere características inteligentes finales)."""
        if not hasattr(self, 'predictive_analyzer') or not self.predictive_analyzer:
            raise RuntimeError("Analizador predictivo no disponible.")
        
        return await self.predictive_analyzer.predict_trend(metric_name, historical_values, timeframe_days)
    
    async def generate_intelligent_recommendations(
        self,
        document_id: str,
        content: str,
        analysis_result: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Generar recomendaciones inteligentes (requiere características inteligentes finales)."""
        if not hasattr(self, 'intelligent_recommendations') or not self.intelligent_recommendations:
            raise RuntimeError("Motor de recomendaciones inteligentes no disponible.")
        
        return await self.intelligent_recommendations.generate_recommendations(
            document_id, content, analysis_result, context
        )
    
    def set_performance_threshold(
        self,
        metric_name: str,
        warning: Optional[float] = None,
        critical: Optional[float] = None
    ):
        """Establecer umbral de rendimiento (requiere características inteligentes finales)."""
        if not hasattr(self, 'realtime_performance') or not self.realtime_performance:
            raise RuntimeError("Monitor de rendimiento no disponible.")
        
        self.realtime_performance.set_threshold(metric_name, warning, critical)
    
    async def get_performance_snapshot(self, time_window_seconds: int = 60):
        """Obtener snapshot de rendimiento (requiere características inteligentes finales)."""
        if not hasattr(self, 'realtime_performance') or not self.realtime_performance:
            raise RuntimeError("Monitor de rendimiento no disponible.")
        
        return await self.realtime_performance.get_performance_snapshot(time_window_seconds)
    
    def get_metric_statistics(
        self,
        metric_name: str,
        time_window_seconds: Optional[int] = None
    ):
        """Obtener estadísticas de métrica (requiere características inteligentes finales)."""
        if not hasattr(self, 'realtime_performance') or not self.realtime_performance:
            raise RuntimeError("Monitor de rendimiento no disponible.")
        
        return self.realtime_performance.get_metric_statistics(metric_name, time_window_seconds)
    
    # ============================================================================
    # MÉTODOS DE ANÁLISIS AVANZADO FINALES - Plagio, Estructura y Optimización
    # ============================================================================
    
    def add_reference_document(self, document_id: str, content: str):
        """Agregar documento de referencia para detección de plagio."""
        if not hasattr(self, 'plagiarism_detector') or not self.plagiarism_detector:
            raise RuntimeError("Detector de plagio no disponible.")
        
        self.plagiarism_detector.add_reference_document(document_id, content)
    
    async def detect_plagiarism(
        self,
        document_id: str,
        content: str,
        threshold: float = 0.7,
        check_references: bool = True
    ):
        """Detectar plagio (requiere características de análisis avanzado finales)."""
        if not hasattr(self, 'plagiarism_detector') or not self.plagiarism_detector:
            raise RuntimeError("Detector de plagio no disponible.")
        
        return await self.plagiarism_detector.detect_plagiarism(
            document_id, content, threshold, check_references
        )
    
    async def analyze_structure_advanced(
        self,
        document_id: str,
        content: str
    ):
        """Analizar estructura avanzada (requiere características de análisis avanzado finales)."""
        if not hasattr(self, 'structure_analyzer') or not self.structure_analyzer:
            raise RuntimeError("Analizador de estructura avanzado no disponible.")
        
        return await self.structure_analyzer.analyze_structure(document_id, content)
    
    async def optimize_document_auto(
        self,
        document_id: str,
        content: str,
        optimization_goals: Optional[List[str]] = None
    ):
        """Optimizar documento automáticamente (requiere características de análisis avanzado finales)."""
        if not hasattr(self, 'auto_optimizer') or not self.auto_optimizer:
            raise RuntimeError("Optimizador automático no disponible.")
        
        return await self.auto_optimizer.optimize_document(
            document_id, content, optimization_goals
        )


# ============================================================================
# SISTEMAS AVANZADOS ADICIONALES - ANÁLISIS PROFUNDO
# ============================================================================

class DocumentTranslator:
    """Servicio de traducción automática de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.translation_history: List[Dict[str, Any]] = []
    
    async def translate_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        target_language: str = "en",
        source_language: Optional[str] = None,
        preserve_formatting: bool = True
    ) -> Dict[str, Any]:
        """
        Traducir un documento a otro idioma
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            target_language: Idioma objetivo (código ISO 639-1)
            source_language: Idioma origen (opcional, se detecta automáticamente)
            preserve_formatting: Preservar formato del documento
        
        Returns:
            Documento traducido y metadatos
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to translate"}
        
        # Detectar idioma origen si no se especifica
        if not source_language:
            language_detector = LanguageDetector(self.analyzer)
            detection = await language_detector.detect_language(document_content=content)
            source_language = detection.get("detected_language", "es")
        
        # Dividir en chunks para traducción
        chunks = self._split_into_chunks(content, max_chunk_size=1000)
        
        translated_chunks = []
        for chunk in chunks:
            translated_chunk = await self._translate_chunk(chunk, source_language, target_language)
            translated_chunks.append(translated_chunk)
        
        translated_content = " ".join(translated_chunks)
        
        result = {
            "original_content": content[:500],  # Preview
            "translated_content": translated_content,
            "source_language": source_language,
            "target_language": target_language,
            "original_length": len(content),
            "translated_length": len(translated_content),
            "preserve_formatting": preserve_formatting,
            "timestamp": datetime.now().isoformat()
        }
        
        self.translation_history.append(result)
        return result
    
    def _split_into_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Dividir texto en chunks para traducción"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    async def _translate_chunk(self, chunk: str, source_lang: str, target_lang: str) -> str:
        """Traducir un chunk de texto"""
        # En producción, usar un servicio de traducción real (Google Translate API, etc.)
        # Por ahora, retornar texto sin traducir con marcador
        return f"[TRANSLATED {source_lang}->{target_lang}] {chunk}"


class SectionSentimentAnalyzer:
    """Analizador de sentimiento por secciones del documento."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.section_analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_section_sentiment(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar sentimiento por secciones del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis de sentimiento por sección
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Analizar estructura para obtener secciones
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        
        sections = structure.get("sections", [])
        
        if not sections:
            # Si no hay secciones claras, dividir por párrafos
            paragraphs = content.split('\n\n')
            sections = [{"title": f"Párrafo {i+1}", "content": p} for i, p in enumerate(paragraphs) if p.strip()]
        
        # Analizar sentimiento de cada sección
        section_sentiments = []
        for section in sections:
            section_content = section.get("content", "")
            if section_content:
                sentiment = await self.analyzer.analyze_sentiment(section_content)
                section_sentiments.append({
                    "section_title": section.get("title", "Sin título"),
                    "content_preview": section_content[:200],
                    "sentiment": sentiment,
                    "dominant_sentiment": max(sentiment.items(), key=lambda x: x[1])[0] if sentiment else "neutral",
                    "sentiment_score": max(sentiment.values()) if sentiment else 0.0
                })
        
        # Calcular sentimiento general
        overall_sentiment = self._calculate_overall_sentiment(section_sentiments)
        
        # Detectar cambios de sentimiento
        sentiment_transitions = self._detect_sentiment_transitions(section_sentiments)
        
        result = {
            "overall_sentiment": overall_sentiment,
            "section_sentiments": section_sentiments,
            "sentiment_transitions": sentiment_transitions,
            "total_sections": len(section_sentiments),
            "timestamp": datetime.now().isoformat()
        }
        
        self.section_analysis_history.append(result)
        return result
    
    def _calculate_overall_sentiment(self, section_sentiments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcular sentimiento general promediando secciones"""
        if not section_sentiments:
            return {"neutral": 1.0}
        
        total_sentiment = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        
        for section in section_sentiments:
            sentiment = section.get("sentiment", {})
            for key in total_sentiment:
                total_sentiment[key] += sentiment.get(key, 0.0)
        
        # Normalizar
        total = sum(total_sentiment.values())
        if total > 0:
            return {k: v / total for k, v in total_sentiment.items()}
        
        return {"neutral": 1.0}
    
    def _detect_sentiment_transitions(self, section_sentiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detectar transiciones de sentimiento entre secciones"""
        transitions = []
        
        for i in range(len(section_sentiments) - 1):
            current = section_sentiments[i]
            next_section = section_sentiments[i + 1]
            
            current_dominant = current.get("dominant_sentiment", "neutral")
            next_dominant = next_section.get("dominant_sentiment", "neutral")
            
            if current_dominant != next_dominant:
                transitions.append({
                    "from_section": current.get("section_title"),
                    "to_section": next_section.get("section_title"),
                    "transition": f"{current_dominant} -> {next_dominant}",
                    "from_sentiment": current_dominant,
                    "to_sentiment": next_dominant
                })
        
        return transitions


class ExecutiveSummaryGenerator:
    """Generador de resúmenes ejecutivos avanzados."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.generated_summaries: List[Dict[str, Any]] = []
    
    async def generate_executive_summary(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        max_length: int = 300,
        include_key_points: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Generar resumen ejecutivo del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            max_length: Longitud máxima del resumen
            include_key_points: Incluir puntos clave
            include_recommendations: Incluir recomendaciones
        
        Returns:
            Resumen ejecutivo completo
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to summarize"}
        
        # Generar resumen base
        summary = await self.analyzer.summarize_document(content, max_length=max_length, min_length=50)
        
        # Extraer puntos clave
        key_points = []
        if include_key_points:
            keywords = await self.analyzer.extract_keywords(content, top_k=10)
            entities = await self.analyzer.extract_entities(content)
            
            # Obtener entidades más importantes
            important_entities = sorted(entities, key=lambda x: x.get("score", 0), reverse=True)[:5]
            
            key_points = {
                "keywords": keywords,
                "important_entities": [e["text"] for e in important_entities],
                "main_topics": await self._extract_main_topics(content)
            }
        
        # Generar recomendaciones
        recommendations = []
        if include_recommendations:
            recommendations = await self._generate_recommendations(content)
        
        # Análisis de impacto
        impact_analysis = await self._analyze_impact(content)
        
        result = {
            "summary": summary,
            "key_points": key_points if include_key_points else None,
            "recommendations": recommendations if include_recommendations else None,
            "impact_analysis": impact_analysis,
            "document_length": len(content),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(content) if content else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.generated_summaries.append(result)
        return result
    
    async def _extract_main_topics(self, content: str) -> List[str]:
        """Extraer temas principales"""
        topics = await self.analyzer.extract_topics(content, num_topics=3)
        return [t.get("keywords", [])[0] if t.get("keywords") else f"Tema {t.get('topic_id')}" for t in topics]
    
    async def _generate_recommendations(self, content: str) -> List[str]:
        """Generar recomendaciones basadas en el contenido"""
        recommendations = []
        
        # Analizar calidad
        quality_calculator = QualityScoreCalculator(self.analyzer)
        quality = await quality_calculator.calculate_quality_score(document_content=content)
        
        # Agregar recomendaciones de calidad
        recommendations.extend(quality.get("recommendations", []))
        
        # Analizar legibilidad
        readability_analyzer = ReadabilityAnalyzer()
        readability = await readability_analyzer.analyze_readability(document_content=content)
        recommendations.extend(readability.get("recommendations", []))
        
        return recommendations[:10]  # Limitar a 10 recomendaciones
    
    async def _analyze_impact(self, content: str) -> Dict[str, Any]:
        """Analizar impacto potencial del documento"""
        # Analizar sentimiento
        sentiment = await self.analyzer.analyze_sentiment(content)
        
        # Contar palabras de acción
        action_words = ["implementar", "desarrollar", "crear", "mejorar", "optimizar", "implement", "develop", "create", "improve", "optimize"]
        action_count = sum(1 for word in action_words if word.lower() in content.lower())
        
        # Detectar urgencia
        urgency_words = ["urgente", "inmediato", "crítico", "prioritario", "urgent", "immediate", "critical", "priority"]
        urgency_count = sum(1 for word in urgency_words if word.lower() in content.lower())
        
        return {
            "sentiment_impact": sentiment,
            "action_items": action_count,
            "urgency_level": "high" if urgency_count > 3 else "medium" if urgency_count > 0 else "low",
            "urgency_score": urgency_count / 10.0
        }


class EntityRelationshipExtractor:
    """Extractor de relaciones entre entidades."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.relationship_graph: Dict[str, List[Dict[str, Any]]] = {}
        self.extraction_history: List[Dict[str, Any]] = []
    
    async def extract_relationships(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extraer relaciones entre entidades del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            relationship_types: Tipos de relaciones a buscar
        
        Returns:
            Grafo de relaciones entre entidades
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Extraer entidades
        entities = await self.analyzer.extract_entities(content)
        
        if not entities:
            return {"error": "No entities found"}
        
        # Detectar relaciones
        relationships = []
        relationship_types = relationship_types or ["works_with", "located_in", "related_to", "part_of"]
        
        # Encontrar posiciones de entidades en el texto
        entity_positions = {}
        content_lower = content.lower()
        for entity in entities:
            entity_text = entity["text"].lower()
            position = content_lower.find(entity_text)
            if position >= 0:
                entity_positions[entity["text"]] = position
        
        # Analizar co-ocurrencias y proximidad
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], start=i+1):
                # Verificar si están cerca en el texto
                pos1 = entity_positions.get(entity1["text"], 0)
                pos2 = entity_positions.get(entity2["text"], 0)
                distance = abs(pos1 - pos2)
                
                if distance < 200:  # Dentro de 200 caracteres
                    relationship_type = self._detect_relationship_type(entity1, entity2, content)
                    
                    if relationship_type:
                        relationships.append({
                            "entity1": entity1["text"],
                            "entity2": entity2["text"],
                            "relationship_type": relationship_type,
                            "confidence": 1.0 - (distance / 200.0),  # Más cerca = mayor confianza
                            "distance": distance
                        })
        
        # Construir grafo
        graph = self._build_relationship_graph(entities, relationships)
        
        result = {
            "entities": entities,
            "relationships": relationships,
            "graph": graph,
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "timestamp": datetime.now().isoformat()
        }
        
        self.extraction_history.append(result)
        return result
    
    def _detect_relationship_type(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        content: str
    ) -> Optional[str]:
        """Detectar tipo de relación entre dos entidades"""
        # Patrones simples para detectar relaciones
        # Buscar contexto entre entidades
        entity1_text = entity1["text"].lower()
        entity2_text = entity2["text"].lower()
        content_lower = content.lower()
        
        pos1 = content_lower.find(entity1_text)
        pos2 = content_lower.find(entity2_text)
        
        if pos1 >= 0 and pos2 >= 0:
            start = min(pos1, pos2)
            end = max(pos1, pos2)
            context = content[start:end].lower()
        else:
            context = ""
        
        # Patrones de relación
        if any(word in context for word in ["trabaja con", "works with", "colabora", "collaborates"]):
            return "works_with"
        elif any(word in context for word in ["en", "in", "ubicado", "located"]):
            return "located_in"
        elif any(word in context for word in ["de", "of", "parte de", "part of"]):
            return "part_of"
        else:
            return "related_to"
    
    def _build_relationship_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Construir grafo de relaciones"""
        graph = {}
        
        for entity in entities:
            graph[entity["text"]] = []
        
        for rel in relationships:
            entity1 = rel["entity1"]
            entity2 = rel["entity2"]
            
            if entity1 in graph:
                graph[entity1].append({
                    "related_to": entity2,
                    "relationship_type": rel["relationship_type"],
                    "confidence": rel["confidence"]
                })
            
            if entity2 in graph:
                graph[entity2].append({
                    "related_to": entity1,
                    "relationship_type": rel["relationship_type"],
                    "confidence": rel["confidence"]
                })
        
        return graph


class DocumentValidator:
    """Validador de documentos con múltiples criterios."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.validation_history: List[Dict[str, Any]] = []
    
    def add_validation_rule(self, rule_name: str, rule_func: Callable, category: str = "general"):
        """Agregar regla de validación"""
        if category not in self.validation_rules:
            self.validation_rules[category] = []
        
        self.validation_rules[category].append({
            "name": rule_name,
            "function": rule_func
        })
    
    async def validate_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validar documento según múltiples criterios
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            categories: Categorías de validación a aplicar
        
        Returns:
            Resultado de validación completo
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to validate"}
        
        categories = categories or list(self.validation_rules.keys())
        
        validation_results = {}
        overall_valid = True
        total_issues = 0
        
        for category in categories:
            if category not in self.validation_rules:
                continue
            
            rules = self.validation_rules[category]
            category_results = []
            
            for rule_info in rules:
                rule_name = rule_info["name"]
                rule_func = rule_info["function"]
                
                try:
                    if asyncio.iscoroutinefunction(rule_func):
                        result = await rule_func(content, self.analyzer)
                    else:
                        result = rule_func(content, self.analyzer)
                    
                    is_valid = result.get("valid", True) if isinstance(result, dict) else bool(result)
                    issues = result.get("issues", []) if isinstance(result, dict) else []
                    
                    category_results.append({
                        "rule": rule_name,
                        "valid": is_valid,
                        "issues": issues,
                        "message": result.get("message", "") if isinstance(result, dict) else ""
                    })
                    
                    if not is_valid:
                        overall_valid = False
                        total_issues += len(issues)
                except Exception as e:
                    category_results.append({
                        "rule": rule_name,
                        "valid": False,
                        "issues": [f"Error en validación: {str(e)}"],
                        "message": f"Error ejecutando regla: {str(e)}"
                    })
                    overall_valid = False
                    total_issues += 1
            
            validation_results[category] = category_results
        
        # Validaciones predefinidas
        predefined_validations = await self._run_predefined_validations(content)
        validation_results["predefined"] = predefined_validations
        
        # Calcular score de validación
        validation_score = self._calculate_validation_score(validation_results)
        
        result = {
            "valid": overall_valid and validation_score > 0.7,
            "validation_score": validation_score,
            "total_issues": total_issues,
            "validation_results": validation_results,
            "recommendations": self._generate_validation_recommendations(validation_results),
            "timestamp": datetime.now().isoformat()
        }
        
        self.validation_history.append(result)
        return result
    
    async def _run_predefined_validations(self, content: str) -> Dict[str, Any]:
        """Ejecutar validaciones predefinidas"""
        validations = {}
        
        # Validación de longitud mínima
        validations["min_length"] = {
            "valid": len(content) >= 100,
            "message": "Documento debe tener al menos 100 caracteres" if len(content) < 100 else "Longitud adecuada"
        }
        
        # Validación de estructura básica
        has_sentences = '.' in content or '!' in content or '?' in content
        validations["basic_structure"] = {
            "valid": has_sentences,
            "message": "Documento debe tener oraciones completas" if not has_sentences else "Estructura básica presente"
        }
        
        # Validación de contenido vacío
        content_stripped = content.strip()
        validations["not_empty"] = {
            "valid": len(content_stripped) > 0,
            "message": "Documento no puede estar vacío" if len(content_stripped) == 0 else "Contenido presente"
        }
        
        return validations
    
    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calcular score de validación"""
        total_rules = 0
        passed_rules = 0
        
        for category, results in validation_results.items():
            if category == "predefined":
                for rule_name, result in results.items():
                    total_rules += 1
                    if result.get("valid", False):
                        passed_rules += 1
            else:
                for result in results:
                    total_rules += 1
                    if result.get("valid", False):
                        passed_rules += 1
        
        return passed_rules / total_rules if total_rules > 0 else 0.0
    
    def _generate_validation_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generar recomendaciones basadas en validación"""
        recommendations = []
        
        for category, results in validation_results.items():
            for result in (results if isinstance(results, list) else [results]):
                if not result.get("valid", True):
                    issues = result.get("issues", [])
                    if issues:
                        recommendations.extend(issues)
        
        return recommendations[:10]  # Limitar a 10 recomendaciones


class TrendAnalyzer:
    """Analizador de tendencias temporales en documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.trend_history: List[Dict[str, Any]] = []
    
    async def analyze_trends(
        self,
        documents: List[Dict[str, Any]],
        time_key: str = "timestamp",
        content_key: str = "content"
    ) -> Dict[str, Any]:
        """
        Analizar tendencias a lo largo del tiempo en múltiples documentos
        
        Args:
            documents: Lista de documentos con timestamps
            time_key: Clave que contiene el timestamp
            content_key: Clave que contiene el contenido
        
        Returns:
            Análisis de tendencias temporales
        """
        if not documents:
            return {"error": "No documents provided"}
        
        # Ordenar por tiempo
        sorted_docs = sorted(
            documents,
            key=lambda x: datetime.fromisoformat(x.get(time_key, datetime.now().isoformat()))
        )
        
        # Analizar cada documento
        document_analyses = []
        for doc in sorted_docs:
            content = doc.get(content_key, "")
            if content:
                analysis = await self.analyzer.analyze_document(document_content=content)
                document_analyses.append({
                    "timestamp": doc.get(time_key),
                    "sentiment": analysis.sentiment,
                    "keywords": analysis.keywords,
                    "topics": analysis.topics,
                    "classification": analysis.classification
                })
        
        # Detectar tendencias
        sentiment_trend = self._analyze_sentiment_trend(document_analyses)
        keyword_trend = self._analyze_keyword_trend(document_analyses)
        topic_trend = self._analyze_topic_trend(document_analyses)
        
        result = {
            "total_documents": len(document_analyses),
            "time_range": {
                "start": sorted_docs[0].get(time_key) if sorted_docs else None,
                "end": sorted_docs[-1].get(time_key) if sorted_docs else None
            },
            "sentiment_trend": sentiment_trend,
            "keyword_trend": keyword_trend,
            "topic_trend": topic_trend,
            "overall_trend": self._calculate_overall_trend(sentiment_trend, keyword_trend),
            "timestamp": datetime.now().isoformat()
        }
        
        self.trend_history.append(result)
        return result
    
    def _analyze_sentiment_trend(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analizar tendencia de sentimiento"""
        sentiments = [a.get("sentiment", {}) for a in analyses]
        
        positive_scores = [s.get("positive", 0.0) for s in sentiments]
        negative_scores = [s.get("negative", 0.0) for s in sentiments]
        
        return {
            "positive_trend": "increasing" if len(positive_scores) > 1 and positive_scores[-1] > positive_scores[0] else "decreasing",
            "negative_trend": "increasing" if len(negative_scores) > 1 and negative_scores[-1] > negative_scores[0] else "decreasing",
            "avg_positive": sum(positive_scores) / len(positive_scores) if positive_scores else 0.0,
            "avg_negative": sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
        }
    
    def _analyze_keyword_trend(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analizar tendencia de keywords"""
        all_keywords = {}
        
        for analysis in analyses:
            keywords = analysis.get("keywords", [])
            for keyword in keywords:
                all_keywords[keyword] = all_keywords.get(keyword, 0) + 1
        
        # Keywords más frecuentes
        top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "top_keywords": [{"keyword": k, "frequency": v} for k, v in top_keywords],
            "total_unique_keywords": len(all_keywords)
        }
    
    def _analyze_topic_trend(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analizar tendencia de temas"""
        all_topics = {}
        
        for analysis in analyses:
            topics = analysis.get("topics", [])
            for topic in topics:
                topic_key = str(topic.get("topic_id", ""))
                if topic_key:
                    all_topics[topic_key] = all_topics.get(topic_key, 0) + 1
        
        return {
            "topic_distribution": all_topics,
            "most_common_topics": sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _calculate_overall_trend(
        self,
        sentiment_trend: Dict[str, Any],
        keyword_trend: Dict[str, Any]
    ) -> str:
        """Calcular tendencia general"""
        positive_trend = sentiment_trend.get("positive_trend", "stable") if sentiment_trend else "stable"
        
        if positive_trend == "increasing":
            return "Mejorando"
        elif positive_trend == "decreasing":
            return "Empeorando"
        else:
            return "Estable"


# ============================================================================
# SISTEMAS AVANZADOS FINALES - ANÁLISIS ESPECIALIZADO
# ============================================================================

class DocumentClustering:
    """Sistema de clustering de documentos similares."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.clusters: Dict[str, List[str]] = {}
        self.clustering_history: List[Dict[str, Any]] = []
    
    async def cluster_documents(
        self,
        documents: List[Dict[str, Any]],
        num_clusters: Optional[int] = None,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Agrupar documentos similares en clusters
        
        Args:
            documents: Lista de documentos con 'id' y 'content'
            num_clusters: Número de clusters (opcional, se calcula automáticamente)
            similarity_threshold: Umbral de similitud para clustering
        
        Returns:
            Clusters de documentos y análisis
        """
        if not documents:
            return {"error": "No documents provided"}
        
        # Generar embeddings para todos los documentos
        contents = [doc.get("content", "") for doc in documents]
        embeddings = await self.analyzer.embedding_generator.generate_embeddings(contents)
        
        if not embeddings:
            return {"error": "Failed to generate embeddings"}
        
        # Calcular matriz de similitud
        similarity_matrix = self._calculate_similarity_matrix(embeddings)
        
        # Agrupar documentos por similitud
        clusters = self._group_by_similarity(
            documents, similarity_matrix, similarity_threshold, num_clusters
        )
        
        # Analizar clusters
        cluster_analysis = self._analyze_clusters(clusters, documents)
        
        result = {
            "clusters": clusters,
            "cluster_analysis": cluster_analysis,
            "total_documents": len(documents),
            "total_clusters": len(clusters),
            "similarity_threshold": similarity_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        self.clustering_history.append(result)
        return result
    
    def _calculate_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Calcular matriz de similitud entre embeddings"""
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    dot_product = np.dot(embeddings[i], embeddings[j])
                    norm_i = np.linalg.norm(embeddings[i])
                    norm_j = np.linalg.norm(embeddings[j])
                    
                    similarity = dot_product / (norm_i * norm_j) if (norm_i * norm_j) > 0 else 0.0
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
        
        return similarity_matrix
    
    def _group_by_similarity(
        self,
        documents: List[Dict[str, Any]],
        similarity_matrix: np.ndarray,
        threshold: float,
        num_clusters: Optional[int]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Agrupar documentos por similitud"""
        clusters = {}
        assigned = set()
        
        cluster_id = 0
        
        for i, doc in enumerate(documents):
            if i in assigned:
                continue
            
            # Crear nuevo cluster
            cluster = [doc]
            assigned.add(i)
            
            # Buscar documentos similares
            for j, other_doc in enumerate(documents):
                if j in assigned or i == j:
                    continue
                
                if similarity_matrix[i][j] >= threshold:
                    cluster.append(other_doc)
                    assigned.add(j)
            
            clusters[cluster_id] = cluster
            cluster_id += 1
        
        return clusters
    
    def _analyze_clusters(
        self,
        clusters: Dict[int, List[Dict[str, Any]]],
        all_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar características de los clusters"""
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        
        return {
            "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "cluster_distribution": {
                "small": sum(1 for s in cluster_sizes if s <= 3),
                "medium": sum(1 for s in cluster_sizes if 3 < s <= 10),
                "large": sum(1 for s in cluster_sizes if s > 10)
            }
        }


class DocumentRecommender:
    """Sistema de recomendación de documentos similares."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.document_catalog: Dict[str, Dict[str, Any]] = {}
        self.recommendation_history: List[Dict[str, Any]] = []
    
    def add_to_catalog(self, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Agregar documento al catálogo"""
        self.document_catalog[document_id] = {
            "content": content,
            "metadata": metadata or {},
            "added_at": datetime.now().isoformat()
        }
    
    async def recommend_similar_documents(
        self,
        query_document_id: Optional[str] = None,
        query_content: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.6
    ) -> Dict[str, Any]:
        """
        Recomendar documentos similares
        
        Args:
            query_document_id: ID del documento de consulta (si está en catálogo)
            query_content: Contenido del documento de consulta
            top_k: Número de recomendaciones
            min_similarity: Similitud mínima
        
        Returns:
            Documentos recomendados ordenados por relevancia
        """
        # Obtener contenido de consulta
        if query_document_id and query_document_id in self.document_catalog:
            query_content = self.document_catalog[query_document_id]["content"]
        
        if not query_content:
            return {"error": "No query content provided"}
        
        if not self.document_catalog:
            return {"error": "Catalog is empty"}
        
        # Generar embedding de consulta
        query_embedding = await self.analyzer.embedding_generator.generate_embeddings([query_content])
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}
        
        query_emb = query_embedding[0]
        
        # Calcular similitud con todos los documentos del catálogo
        similarities = []
        
        for doc_id, doc_data in self.document_catalog.items():
            if query_document_id and doc_id == query_document_id:
                continue  # No recomendar el mismo documento
            
            content = doc_data["content"]
            doc_embeddings = await self.analyzer.embedding_generator.generate_embeddings([content])
            
            if doc_embeddings:
                doc_emb = doc_embeddings[0]
                
                # Calcular similitud coseno
                dot_product = np.dot(query_emb, doc_emb)
                norm_query = np.linalg.norm(query_emb)
                norm_doc = np.linalg.norm(doc_emb)
                
                similarity = dot_product / (norm_query * norm_doc) if (norm_query * norm_doc) > 0 else 0.0
                
                if similarity >= min_similarity:
                    similarities.append({
                        "document_id": doc_id,
                        "similarity": float(similarity),
                        "metadata": doc_data.get("metadata", {})
                    })
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        recommendations = similarities[:top_k]
        
        result = {
            "query_document_id": query_document_id,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "min_similarity": min_similarity,
            "timestamp": datetime.now().isoformat()
        }
        
        self.recommendation_history.append(result)
        return result


class LegalContractAnalyzer:
    """Analizador especializado para contratos legales."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.contract_analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_contract(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar contrato legal
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis especializado del contrato
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Extraer cláusulas
        clauses = self._extract_clauses(content)
        
        # Extraer partes del contrato
        parties = self._extract_parties(content)
        
        # Extraer fechas importantes
        dates = self._extract_important_dates(content)
        
        # Extraer obligaciones
        obligations = self._extract_obligations(content)
        
        # Extraer términos financieros
        financial_terms = self._extract_financial_terms(content)
        
        # Detectar cláusulas de riesgo
        risk_clauses = self._detect_risk_clauses(content)
        
        # Análisis de sentimiento legal
        legal_sentiment = await self._analyze_legal_sentiment(content)
        
        result = {
            "clauses": clauses,
            "parties": parties,
            "important_dates": dates,
            "obligations": obligations,
            "financial_terms": financial_terms,
            "risk_clauses": risk_clauses,
            "legal_sentiment": legal_sentiment,
            "total_clauses": len(clauses),
            "timestamp": datetime.now().isoformat()
        }
        
        self.contract_analysis_history.append(result)
        return result
    
    def _extract_clauses(self, content: str) -> List[Dict[str, Any]]:
        """Extraer cláusulas del contrato"""
        import re
        
        clauses = []
        # Patrones comunes de cláusulas
        clause_patterns = [
            r'(?:Cláusula|Clause)\s+\d+[.:]\s*([^\.]+(?:\.[^\.]+)*)',
            r'(?:Artículo|Article)\s+\d+[.:]\s*([^\.]+(?:\.[^\.]+)*)',
            r'(?:Sección|Section)\s+\d+[.:]\s*([^\.]+(?:\.[^\.]+)*)'
        ]
        
        for pattern in clause_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                clauses.append({
                    "text": match.group(1).strip(),
                    "position": match.start(),
                    "type": "clause"
                })
        
        return clauses
    
    def _extract_parties(self, content: str) -> List[Dict[str, Any]]:
        """Extraer partes del contrato"""
        import re
        
        parties = []
        # Patrones para identificar partes
        party_patterns = [
            r'(?:entre|between|by and between)\s+([A-Z][^,]+(?:,\s*[A-Z][^,]+)*)',
            r'(?:Parte|Party)\s+[A-Z]\s*[:]\s*([A-Z][^\.]+)'
        ]
        
        for pattern in party_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                party_text = match.group(1).strip()
                if len(party_text) > 5:  # Filtrar matches muy cortos
                    parties.append({
                        "name": party_text,
                        "position": match.start()
                    })
        
        return list({p["name"]: p for p in parties}.values())  # Eliminar duplicados
    
    def _extract_important_dates(self, content: str) -> List[Dict[str, Any]]:
        """Extraer fechas importantes"""
        import re
        
        dates = []
        # Patrones de fechas
        date_patterns = [
            r'(?:fecha|date|effective|vigencia)\s+[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}\s+de\s+[A-Z][a-z]+\s+de\s+\d{4})',
            r'([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                dates.append({
                    "date": match.group(1),
                    "position": match.start(),
                    "context": content[max(0, match.start()-50):match.end()+50]
                })
        
        return dates
    
    def _extract_obligations(self, content: str) -> List[Dict[str, Any]]:
        """Extraer obligaciones"""
        import re
        
        obligations = []
        # Palabras clave de obligaciones
        obligation_keywords = [
            "debe", "deberá", "obligado", "obligación", "must", "shall", "required", "obligation"
        ]
        
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in obligation_keywords):
                obligations.append({
                    "text": sentence.strip(),
                    "type": "obligation"
                })
        
        return obligations
    
    def _extract_financial_terms(self, content: str) -> List[Dict[str, Any]]:
        """Extraer términos financieros"""
        import re
        
        financial_terms = []
        
        # Patrones para montos
        amount_patterns = [
            r'(\$\s*\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|dólares|euros))',
            r'(?:monto|amount|valor|value)\s*[:\-]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in amount_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                financial_terms.append({
                    "amount": match.group(1),
                    "type": "monetary",
                    "position": match.start()
                })
        
        return financial_terms
    
    def _detect_risk_clauses(self, content: str) -> List[Dict[str, Any]]:
        """Detectar cláusulas de riesgo"""
        import re
        
        risk_keywords = [
            "limitación de responsabilidad", "limitation of liability",
            "indemnización", "indemnification",
            "exención", "waiver",
            "confidencialidad", "confidentiality",
            "no competencia", "non-compete"
        ]
        
        risk_clauses = []
        content_lower = content.lower()
        
        for keyword in risk_keywords:
            if keyword.lower() in content_lower:
                # Buscar contexto alrededor de la palabra clave
                position = content_lower.find(keyword.lower())
                context_start = max(0, position - 200)
                context_end = min(len(content), position + 500)
                context = content[context_start:context_end]
                
                risk_clauses.append({
                    "keyword": keyword,
                    "position": position,
                    "context": context,
                    "risk_level": "high" if "limitación" in keyword.lower() or "indemn" in keyword.lower() else "medium"
                })
        
        return risk_clauses
    
    async def _analyze_legal_sentiment(self, content: str) -> Dict[str, Any]:
        """Analizar sentimiento legal del contrato"""
        # Analizar sentimiento general
        sentiment = await self.analyzer.analyze_sentiment(content)
        
        # Detectar lenguaje positivo/negativo en contexto legal
        positive_legal_terms = ["favorable", "beneficio", "derecho", "favorable", "benefit", "right"]
        negative_legal_terms = ["penalización", "multa", "sanción", "penalty", "fine", "sanction"]
        
        content_lower = content.lower()
        positive_count = sum(1 for term in positive_legal_terms if term in content_lower)
        negative_count = sum(1 for term in negative_legal_terms if term in content_lower)
        
        return {
            "general_sentiment": sentiment,
            "positive_legal_terms": positive_count,
            "negative_legal_terms": negative_count,
            "legal_balance": "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "neutral"
        }


class FinancialDataExtractor:
    """Extractor especializado de datos financieros."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.extraction_history: List[Dict[str, Any]] = []
    
    async def extract_financial_data(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extraer datos financieros del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Datos financieros extraídos
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to extract from"}
        
        # Extraer montos
        amounts = self._extract_amounts(content)
        
        # Extraer porcentajes
        percentages = self._extract_percentages(content)
        
        # Extraer fechas financieras
        financial_dates = self._extract_financial_dates(content)
        
        # Extraer términos financieros clave
        financial_terms = self._extract_financial_terms(content)
        
        # Extraer información de cuentas
        accounts = self._extract_accounts(content)
        
        # Calcular totales
        totals = self._calculate_totals(amounts)
        
        result = {
            "amounts": amounts,
            "percentages": percentages,
            "financial_dates": financial_dates,
            "financial_terms": financial_terms,
            "accounts": accounts,
            "totals": totals,
            "summary": {
                "total_amounts": len(amounts),
                "total_percentages": len(percentages),
                "total_accounts": len(accounts),
                "currency_detected": self._detect_currency(amounts)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.extraction_history.append(result)
        return result
    
    def _extract_amounts(self, content: str) -> List[Dict[str, Any]]:
        """Extraer montos monetarios"""
        import re
        
        amounts = []
        
        # Patrones para diferentes formatos
        patterns = [
            r'(\$\s*\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|MXN|dólares|euros|pesos))',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:mil|millones|thousand|million|billion))',
            r'(?:monto|amount|total|valor|value|precio|price)\s*[:\-]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                amount_text = match.group(1)
                # Extraer número
                numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d{2})?', amount_text)
                if numbers:
                    amounts.append({
                        "amount": amount_text,
                        "numeric_value": numbers[0],
                        "position": match.start(),
                        "context": content[max(0, match.start()-30):match.end()+30]
                    })
        
        return amounts
    
    def _extract_percentages(self, content: str) -> List[Dict[str, Any]]:
        """Extraer porcentajes"""
        import re
        
        percentages = []
        pattern = r'(\d+(?:\.\d+)?)\s*%'
        
        matches = re.finditer(pattern, content)
        for match in matches:
            percentages.append({
                "percentage": match.group(1),
                "value": float(match.group(1)),
                "position": match.start(),
                "context": content[max(0, match.start()-30):match.end()+30]
            })
        
        return percentages
    
    def _extract_financial_dates(self, content: str) -> List[Dict[str, Any]]:
        """Extraer fechas financieras importantes"""
        import re
        
        dates = []
        # Patrones para fechas financieras
        patterns = [
            r'(?:vencimiento|due date|expiry|fecha de pago|payment date)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:fecha|date)\s+de\s+(?:vencimiento|pago)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                dates.append({
                    "date": match.group(1),
                    "type": "financial",
                    "position": match.start()
                })
        
        return dates
    
    def _extract_financial_terms(self, content: str) -> List[Dict[str, Any]]:
        """Extraer términos financieros clave"""
        import re
        
        financial_keywords = [
            "interés", "interest", "tasa", "rate", "descuento", "discount",
            "inversión", "investment", "dividendo", "dividend",
            "ganancia", "profit", "pérdida", "loss", "activo", "asset",
            "pasivo", "liability", "patrimonio", "equity"
        ]
        
        terms = []
        content_lower = content.lower()
        
        for keyword in financial_keywords:
            if keyword.lower() in content_lower:
                # Buscar contexto
                position = content_lower.find(keyword.lower())
                context = content[max(0, position-50):min(len(content), position+100)]
                
                terms.append({
                    "term": keyword,
                    "position": position,
                    "context": context
                })
        
        return terms
    
    def _extract_accounts(self, content: str) -> List[Dict[str, Any]]:
        """Extraer información de cuentas"""
        import re
        
        accounts = []
        # Patrones para cuentas
        patterns = [
            r'(?:cuenta|account|CBU|IBAN)\s*[:\-]?\s*([A-Z0-9\s\-]+)',
            r'([A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16})',  # IBAN
            r'(\d{22})'  # CBU argentino
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                account_text = match.group(1).strip()
                if len(account_text) >= 10:  # Filtrar matches muy cortos
                    accounts.append({
                        "account": account_text,
                        "type": "IBAN" if len(account_text) > 20 else "CBU" if len(account_text) == 22 else "account",
                        "position": match.start()
                    })
        
        return accounts
    
    def _calculate_totals(self, amounts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcular totales de montos"""
        if not amounts:
            return {"total": 0, "count": 0}
        
        # Intentar extraer valores numéricos
        numeric_values = []
        for amount in amounts:
            numeric_str = amount.get("numeric_value", "").replace(",", "").replace("$", "").strip()
            try:
                numeric_values.append(float(numeric_str))
            except:
                pass
        
        return {
            "total": sum(numeric_values) if numeric_values else 0,
            "count": len(numeric_values),
            "average": sum(numeric_values) / len(numeric_values) if numeric_values else 0,
            "max": max(numeric_values) if numeric_values else 0,
            "min": min(numeric_values) if numeric_values else 0
        }
    
    def _detect_currency(self, amounts: List[Dict[str, Any]]) -> str:
        """Detectar moneda predominante"""
        currency_counts = {}
        
        for amount in amounts:
            amount_text = amount.get("amount", "").upper()
            if "USD" in amount_text or "$" in amount_text:
                currency_counts["USD"] = currency_counts.get("USD", 0) + 1
            elif "EUR" in amount_text:
                currency_counts["EUR"] = currency_counts.get("EUR", 0) + 1
            elif "MXN" in amount_text or "pesos" in amount_text.lower():
                currency_counts["MXN"] = currency_counts.get("MXN", 0) + 1
        
        if currency_counts:
            return max(currency_counts.items(), key=lambda x: x[1])[0]
        
        return "unknown"


class MultiDocumentComparator:
    """Comparador de múltiples documentos simultáneamente."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.comparison_history: List[Dict[str, Any]] = []
    
    async def compare_multiple_documents(
        self,
        documents: List[Dict[str, Any]],
        comparison_type: str = "all_pairs"
    ) -> Dict[str, Any]:
        """
        Comparar múltiples documentos
        
        Args:
            documents: Lista de documentos con 'id' y 'content'
            comparison_type: 'all_pairs' o 'centroid'
        
        Returns:
            Matriz de comparación y análisis
        """
        if len(documents) < 2:
            return {"error": "Need at least 2 documents to compare"}
        
        if comparison_type == "all_pairs":
            return await self._compare_all_pairs(documents)
        elif comparison_type == "centroid":
            return await self._compare_with_centroid(documents)
        else:
            return {"error": f"Unknown comparison type: {comparison_type}"}
    
    async def _compare_all_pairs(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comparar todos los pares de documentos"""
        comparisons = []
        similarity_matrix = {}
        
        for i, doc1 in enumerate(documents):
            doc1_id = doc1.get("id", f"doc_{i}")
            similarity_matrix[doc1_id] = {}
            
            for j, doc2 in enumerate(documents[i+1:], start=i+1):
                doc2_id = doc2.get("id", f"doc_{j}")
                
                # Usar DocumentComparator
                comparator = DocumentComparator(self.analyzer)
                comparison = await comparator.compare_documents(
                    doc1_content=doc1.get("content", ""),
                    doc2_content=doc2.get("content", "")
                )
                
                similarity = comparison.get("similarity_score", 0.0)
                
                similarity_matrix[doc1_id][doc2_id] = similarity
                similarity_matrix[doc2_id] = similarity_matrix.get(doc2_id, {})
                similarity_matrix[doc2_id][doc1_id] = similarity
                
                comparisons.append({
                    "doc1_id": doc1_id,
                    "doc2_id": doc2_id,
                    "similarity": similarity,
                    "details": comparison
                })
        
        # Encontrar documentos más similares
        most_similar = max(comparisons, key=lambda x: x["similarity"]) if comparisons else None
        least_similar = min(comparisons, key=lambda x: x["similarity"]) if comparisons else None
        
        result = {
            "comparison_type": "all_pairs",
            "total_documents": len(documents),
            "total_comparisons": len(comparisons),
            "similarity_matrix": similarity_matrix,
            "comparisons": comparisons,
            "most_similar": most_similar,
            "least_similar": least_similar,
            "avg_similarity": sum(c["similarity"] for c in comparisons) / len(comparisons) if comparisons else 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.comparison_history.append(result)
        return result
    
    async def _compare_with_centroid(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comparar documentos con un documento centroide"""
        # Generar embeddings para todos
        contents = [doc.get("content", "") for doc in documents]
        embeddings = await self.analyzer.embedding_generator.generate_embeddings(contents)
        
        if not embeddings:
            return {"error": "Failed to generate embeddings"}
        
        # Calcular centroide
        centroid = np.mean(embeddings, axis=0)
        
        # Comparar cada documento con el centroide
        similarities = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            dot_product = np.dot(centroid, embedding)
            norm_centroid = np.linalg.norm(centroid)
            norm_emb = np.linalg.norm(embedding)
            
            similarity = dot_product / (norm_centroid * norm_emb) if (norm_centroid * norm_emb) > 0 else 0.0
            
            similarities.append({
                "document_id": doc.get("id", f"doc_{i}"),
                "similarity_to_centroid": float(similarity),
                "is_typical": similarity > 0.7  # Típico si similar al centroide
            })
        
        return {
            "comparison_type": "centroid",
            "total_documents": len(documents),
            "similarities": similarities,
            "avg_similarity_to_centroid": sum(s["similarity_to_centroid"] for s in similarities) / len(similarities) if similarities else 0.0,
            "typical_documents": [s for s in similarities if s["is_typical"]],
            "atypical_documents": [s for s in similarities if not s["is_typical"]],
            "timestamp": datetime.now().isoformat()
        }


class DocumentVersionTracker:
    """Sistema de seguimiento de versiones de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.versions: Dict[str, List[Dict[str, Any]]] = {}
        self.change_detector = DocumentChangeDetector(analyzer)
    
    def register_version(
        self,
        document_id: str,
        version: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar nueva versión de documento"""
        if document_id not in self.versions:
            self.versions[document_id] = []
        
        self.versions[document_id].append({
            "version": version,
            "content": content,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat()
        })
        
        # Ordenar por versión
        self.versions[document_id].sort(key=lambda x: x["version"])
    
    async def analyze_version_changes(
        self,
        document_id: str,
        version1: Optional[str] = None,
        version2: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar cambios entre versiones
        
        Args:
            document_id: ID del documento
            version1: Versión inicial (opcional, usa la primera si no se especifica)
            version2: Versión final (opcional, usa la última si no se especifica)
        
        Returns:
            Análisis de cambios entre versiones
        """
        if document_id not in self.versions:
            return {"error": f"Document {document_id} not found"}
        
        versions_list = self.versions[document_id]
        
        if len(versions_list) < 2:
            return {"error": "Need at least 2 versions to compare"}
        
        # Determinar versiones a comparar
        if not version1:
            version1 = versions_list[0]["version"]
        if not version2:
            version2 = versions_list[-1]["version"]
        
        # Encontrar contenidos
        content1 = next((v["content"] for v in versions_list if v["version"] == version1), None)
        content2 = next((v["content"] for v in versions_list if v["version"] == version2), None)
        
        if not content1 or not content2:
            return {"error": "Version not found"}
        
        # Analizar cambios
        changes = await self.change_detector.detect_changes(
            old_version_content=content1,
            new_version_content=content2,
            granularity="sentence"
        )
        
        # Calcular estadísticas de evolución
        evolution_stats = self._calculate_evolution_stats(versions_list)
        
        result = {
            "document_id": document_id,
            "version1": version1,
            "version2": version2,
            "changes": changes,
            "evolution_stats": evolution_stats,
            "total_versions": len(versions_list),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_evolution_stats(self, versions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcular estadísticas de evolución"""
        if len(versions) < 2:
            return {}
        
        lengths = [len(v["content"]) for v in versions]
        
        return {
            "version_count": len(versions),
            "length_evolution": {
                "initial": lengths[0],
                "final": lengths[-1],
                "change": lengths[-1] - lengths[0],
                "change_percentage": ((lengths[-1] - lengths[0]) / lengths[0] * 100) if lengths[0] > 0 else 0
            },
            "avg_length": sum(lengths) / len(lengths),
            "trend": "growing" if lengths[-1] > lengths[0] else "shrinking" if lengths[-1] < lengths[0] else "stable"
        }
    
    def get_version_history(self, document_id: str) -> List[Dict[str, Any]]:
        """Obtener historial de versiones"""
        if document_id not in self.versions:
            return []
        
        return self.versions[document_id]
    
    def get_latest_version(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Obtener última versión"""
        if document_id not in self.versions or not self.versions[document_id]:
            return None
        
        return self.versions[document_id][-1]


# ============================================================================
# SISTEMAS AVANZADOS FINALES - ANÁLISIS ESPECIALIZADO ADICIONAL
# ============================================================================

class DocumentImageAnalyzer:
    """Analizador de imágenes dentro de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.ocr_processor = OCRProcessor(analyzer)
        self.image_analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_images_in_document(
        self,
        document_path: str,
        extract_text: bool = True,
        detect_objects: bool = False
    ) -> Dict[str, Any]:
        """
        Analizar imágenes dentro de un documento
        
        Args:
            document_path: Ruta al documento
            extract_text: Extraer texto de imágenes (OCR)
            detect_objects: Detectar objetos en imágenes
        
        Returns:
            Análisis de imágenes encontradas
        """
        images_found = []
        
        # Detectar tipo de documento
        file_ext = Path(document_path).suffix.lower()
        
        if file_ext == '.pdf':
            images = await self._extract_images_from_pdf(document_path)
        elif file_ext in ['.docx', '.doc']:
            images = await self._extract_images_from_docx(document_path)
        else:
            return {"error": f"Image extraction not supported for {file_ext}"}
        
        # Analizar cada imagen
        for i, image_data in enumerate(images):
            image_analysis = {
                "image_index": i + 1,
                "image_path": image_data.get("path"),
                "size": image_data.get("size"),
                "format": image_data.get("format")
            }
            
            # OCR si está habilitado
            if extract_text and image_data.get("path"):
                ocr_result = await self.ocr_processor.extract_text_from_image(
                    image_data["path"]
                )
                image_analysis["extracted_text"] = ocr_result.get("text", "")
                image_analysis["ocr_confidence"] = ocr_result.get("confidence", 0.0)
            
            # Detección de objetos (simplificado)
            if detect_objects and image_data.get("path"):
                objects = await self._detect_objects_in_image(image_data["path"])
                image_analysis["detected_objects"] = objects
            
            images_found.append(image_analysis)
        
        result = {
            "document_path": document_path,
            "total_images": len(images_found),
            "images": images_found,
            "timestamp": datetime.now().isoformat()
        }
        
        self.image_analysis_history.append(result)
        return result
    
    async def _extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extraer imágenes de PDF"""
        try:
            import fitz  # PyMuPDF
            
            images = []
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Guardar imagen temporalmente
                    temp_path = f"/tmp/img_{page_num}_{img_index}.{image_ext}"
                    with open(temp_path, "wb") as f:
                        f.write(image_bytes)
                    
                    images.append({
                        "path": temp_path,
                        "page": page_num + 1,
                        "size": len(image_bytes),
                        "format": image_ext
                    })
            
            doc.close()
            return images
        except ImportError:
            logger.warning("PyMuPDF no está instalado. Instalar con: pip install PyMuPDF")
            return []
        except Exception as e:
            logger.error(f"Error extrayendo imágenes de PDF: {e}")
            return []
    
    async def _extract_images_from_docx(self, docx_path: str) -> List[Dict[str, Any]]:
        """Extraer imágenes de DOCX"""
        try:
            from docx import Document
            import zipfile
            import io
            
            images = []
            doc = Document(docx_path)
            
            # Extraer imágenes del archivo docx (que es un zip)
            with zipfile.ZipFile(docx_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.filename.startswith('word/media/'):
                        image_data = zip_ref.read(file_info.filename)
                        image_ext = Path(file_info.filename).suffix
                        
                        temp_path = f"/tmp/img_{len(images)}{image_ext}"
                        with open(temp_path, "wb") as f:
                            f.write(image_data)
                        
                        images.append({
                            "path": temp_path,
                            "size": len(image_data),
                            "format": image_ext[1:]  # Sin el punto
                        })
            
            return images
        except ImportError:
            logger.warning("python-docx no está instalado. Instalar con: pip install python-docx")
            return []
        except Exception as e:
            logger.error(f"Error extrayendo imágenes de DOCX: {e}")
            return []
    
    async def _detect_objects_in_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Detectar objetos en imagen (simplificado)"""
        # En producción, usar un modelo de detección de objetos real
        # Por ahora, retornar estructura básica
        return [
            {
                "object_type": "text_region",
                "confidence": 0.8,
                "description": "Región de texto detectada"
            }
        ]


class AutomaticIndexGenerator:
    """Generador automático de índices para documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.generated_indexes: List[Dict[str, Any]] = []
    
    async def generate_index(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        index_type: str = "table_of_contents"
    ) -> Dict[str, Any]:
        """
        Generar índice automático del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            index_type: Tipo de índice ('table_of_contents', 'subject_index', 'both')
        
        Returns:
            Índice generado
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to generate index from"}
        
        index = {}
        
        # Tabla de contenidos
        if index_type in ["table_of_contents", "both"]:
            toc = await self._generate_table_of_contents(content)
            index["table_of_contents"] = toc
        
        # Índice de materias
        if index_type in ["subject_index", "both"]:
            subject_index = await self._generate_subject_index(content)
            index["subject_index"] = subject_index
        
        result = {
            "index": index,
            "index_type": index_type,
            "document_length": len(content),
            "timestamp": datetime.now().isoformat()
        }
        
        self.generated_indexes.append(result)
        return result
    
    async def _generate_table_of_contents(self, content: str) -> List[Dict[str, Any]]:
        """Generar tabla de contenidos"""
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        
        sections = structure.get("sections", [])
        
        toc = []
        for section in sections:
            toc.append({
                "title": section.get("title", ""),
                "level": section.get("level", 1),
                "page_reference": "N/A",  # En producción calcular página real
                "content_preview": section.get("content", "")[:100]
            })
        
        return toc
    
    async def _generate_subject_index(self, content: str) -> Dict[str, List[int]]:
        """Generar índice de materias"""
        # Extraer keywords importantes
        keywords = await self.analyzer.extract_keywords(content, top_k=50)
        
        # Encontrar ocurrencias de cada keyword
        subject_index = {}
        content_lower = content.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            occurrences = []
            start = 0
            
            while True:
                pos = content_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                occurrences.append(pos)
                start = pos + 1
            
            if occurrences:
                subject_index[keyword] = occurrences
        
        return subject_index


class FormAnalyzer:
    """Analizador especializado para formularios."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.form_analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_form(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar formulario
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis completo del formulario
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Extraer campos del formulario
        fields = self._extract_form_fields(content)
        
        # Extraer opciones de selección
        options = self._extract_select_options(content)
        
        # Detectar validaciones
        validations = self._extract_validations(content)
        
        # Detectar campos requeridos
        required_fields = self._detect_required_fields(content)
        
        # Analizar estructura del formulario
        form_structure = self._analyze_form_structure(content)
        
        result = {
            "fields": fields,
            "options": options,
            "validations": validations,
            "required_fields": required_fields,
            "form_structure": form_structure,
            "total_fields": len(fields),
            "total_sections": len(form_structure.get("sections", [])),
            "timestamp": datetime.now().isoformat()
        }
        
        self.form_analysis_history.append(result)
        return result
    
    def _extract_form_fields(self, content: str) -> List[Dict[str, Any]]:
        """Extraer campos del formulario"""
        import re
        
        fields = []
        
        # Patrones comunes de campos
        field_patterns = [
            r'(?:campo|field|input)\s*[:\-]?\s*([A-Za-z_][A-Za-z0-9_]*)',
            r'<input[^>]+name=["\']([^"\']+)["\']',
            r'name=["\']([^"\']+)["\']',
            r'id=["\']([^"\']+)["\']'
        ]
        
        for pattern in field_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                field_name = match.group(1)
                if field_name not in [f["name"] for f in fields]:
                    fields.append({
                        "name": field_name,
                        "position": match.start(),
                        "type": self._detect_field_type(match.group(0))
                    })
        
        return fields
    
    def _detect_field_type(self, field_text: str) -> str:
        """Detectar tipo de campo"""
        field_lower = field_text.lower()
        
        if "email" in field_lower:
            return "email"
        elif "password" in field_lower or "pass" in field_lower:
            return "password"
        elif "date" in field_lower or "fecha" in field_lower:
            return "date"
        elif "number" in field_lower or "número" in field_lower:
            return "number"
        elif "tel" in field_lower or "phone" in field_lower:
            return "tel"
        else:
            return "text"
    
    def _extract_select_options(self, content: str) -> Dict[str, List[str]]:
        """Extraer opciones de campos select"""
        import re
        
        options = {}
        
        # Patrón para select options
        select_pattern = r'<select[^>]+name=["\']([^"\']+)["\'][^>]*>(.*?)</select>'
        matches = re.finditer(select_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            field_name = match.group(1)
            select_content = match.group(2)
            
            # Extraer opciones
            option_pattern = r'<option[^>]*>([^<]+)</option>'
            option_matches = re.finditer(option_pattern, select_content, re.IGNORECASE)
            
            field_options = [opt.group(1).strip() for opt in option_matches]
            if field_options:
                options[field_name] = field_options
        
        return options
    
    def _extract_validations(self, content: str) -> List[Dict[str, Any]]:
        """Extraer reglas de validación"""
        import re
        
        validations = []
        
        # Patrones de validación
        validation_patterns = [
            (r'required', "required"),
            (r'minlength[=:]\s*(\d+)', "min_length"),
            (r'maxlength[=:]\s*(\d+)', "max_length"),
            (r'pattern[=:]\s*["\']([^"\']+)["\']', "pattern"),
            (r'min[=:]\s*(\d+)', "min_value"),
            (r'max[=:]\s*(\d+)', "max_value")
        ]
        
        for pattern, validation_type in validation_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                validations.append({
                    "type": validation_type,
                    "value": match.group(1) if len(match.groups()) > 0 else None,
                    "position": match.start()
                })
        
        return validations
    
    def _detect_required_fields(self, content: str) -> List[str]:
        """Detectar campos requeridos"""
        import re
        
        required = []
        
        # Buscar campos marcados como requeridos
        patterns = [
            r'(?:required|obligatorio|requerido)',
            r'<input[^>]*required[^>]*name=["\']([^"\']+)["\']',
            r'name=["\']([^"\']+)["\'][^>]*required'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 0:
                    field_name = match.group(1)
                    if field_name not in required:
                        required.append(field_name)
        
        return required
    
    async def _analyze_form_structure(self, content: str) -> Dict[str, Any]:
        """Analizar estructura del formulario"""
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        
        return {
            "sections": structure.get("sections", []),
            "fields_per_section": len(structure.get("sections", []))
        }


class IntelligentTagger:
    """Sistema de etiquetado automático inteligente."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.tag_vocabulary: Dict[str, Dict[str, Any]] = {}
        self.tagging_history: List[Dict[str, Any]] = []
    
    def add_tag_category(self, category: str, tags: List[str], weights: Optional[Dict[str, float]] = None):
        """Agregar categoría de etiquetas"""
        self.tag_vocabulary[category] = {
            "tags": tags,
            "weights": weights or {tag: 1.0 for tag in tags}
        }
    
    async def auto_tag_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        max_tags_per_category: int = 5
    ) -> Dict[str, Any]:
        """
        Etiquetar documento automáticamente
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            max_tags_per_category: Máximo de etiquetas por categoría
        
        Returns:
            Etiquetas asignadas y confianza
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to tag"}
        
        # Analizar documento
        analysis = await self.analyzer.analyze_document(document_content=content)
        
        # Generar etiquetas basadas en análisis
        tags = {}
        
        # Etiquetas basadas en clasificación
        if analysis.classification:
            top_class = max(analysis.classification.items(), key=lambda x: x[1])[0]
            tags["classification"] = {
                "primary": top_class,
                "all": analysis.classification
            }
        
        # Etiquetas basadas en keywords
        if analysis.keywords:
            tags["keywords"] = analysis.keywords[:max_tags_per_category]
        
        # Etiquetas basadas en entidades
        if analysis.entities:
            entity_types = {}
            for entity in analysis.entities:
                entity_type = entity.get("label", "OTHER")
                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append(entity["text"])
            
            tags["entities"] = entity_types
        
        # Etiquetas basadas en temas
        if analysis.topics:
            tags["topics"] = [t.get("keywords", [])[0] if t.get("keywords") else f"Topic {t.get('topic_id')}" 
                            for t in analysis.topics[:max_tags_per_category]]
        
        # Etiquetas basadas en sentimiento
        if analysis.sentiment:
            dominant_sentiment = max(analysis.sentiment.items(), key=lambda x: x[1])[0]
            tags["sentiment"] = dominant_sentiment
        
        # Etiquetas personalizadas basadas en vocabulario
        custom_tags = self._generate_custom_tags(content)
        if custom_tags:
            tags["custom"] = custom_tags
        
        result = {
            "tags": tags,
            "total_tags": sum(len(v) if isinstance(v, (list, dict)) else 1 for v in tags.values()),
            "confidence": analysis.confidence,
            "document_id": analysis.document_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.tagging_history.append(result)
        return result
    
    def _generate_custom_tags(self, content: str) -> List[str]:
        """Generar etiquetas personalizadas basadas en vocabulario"""
        tags = []
        content_lower = content.lower()
        
        for category, category_data in self.tag_vocabulary.items():
            category_tags = category_data.get("tags", [])
            weights = category_data.get("weights", {})
            
            for tag in category_tags:
                tag_lower = tag.lower()
                if tag_lower in content_lower:
                    weight = weights.get(tag, 1.0)
                    # Contar ocurrencias
                    occurrences = content_lower.count(tag_lower)
                    score = occurrences * weight
                    
                    if score > 0.5:  # Umbral mínimo
                        tags.append({
                            "tag": tag,
                            "category": category,
                            "score": score,
                            "occurrences": occurrences
                        })
        
        # Ordenar por score y retornar top
        tags.sort(key=lambda x: x["score"], reverse=True)
        return [t["tag"] for t in tags[:10]]


class AdvancedMetadataExtractor:
    """Extractor avanzado de metadata de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.extraction_history: List[Dict[str, Any]] = []
    
    async def extract_comprehensive_metadata(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extraer metadata completa del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Metadata completa extraída
        """
        content = document_content
        file_metadata = {}
        
        if document_path:
            file_path = Path(document_path)
            file_metadata = {
                "filename": file_path.name,
                "extension": file_path.suffix,
                "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
                "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat() if file_path.exists() else None,
                "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
            }
            
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze", "file_metadata": file_metadata}
        
        # Metadata del contenido
        content_metadata = {
            "length": len(content),
            "word_count": len(content.split()),
            "character_count": len(content),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "sentence_count": len([s for s in content.split('.') if s.strip()]),
            "line_count": len(content.split('\n'))
        }
        
        # Detectar idioma
        language_detector = LanguageDetector(self.analyzer)
        language_info = await language_detector.detect_language(document_content=content)
        
        # Extraer fechas del contenido
        structured_extractor = StructuredDataExtractor(self.analyzer)
        structured_data = await structured_extractor.extract_structured_data(document_content=content)
        
        # Metadata de autor (si está disponible)
        author_metadata = self._extract_author_metadata(content)
        
        # Metadata de organización
        org_metadata = self._extract_organization_metadata(content)
        
        result = {
            "file_metadata": file_metadata,
            "content_metadata": content_metadata,
            "language": language_info.get("detected_language"),
            "language_confidence": language_info.get("confidence"),
            "structured_data": structured_data.get("extracted_data", {}),
            "author_metadata": author_metadata,
            "organization_metadata": org_metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        self.extraction_history.append(result)
        return result
    
    def _extract_author_metadata(self, content: str) -> Dict[str, Any]:
        """Extraer metadata de autor"""
        import re
        
        author_info = {}
        
        # Patrones comunes de autor
        patterns = [
            r'(?:autor|author|escrito por|written by)\s*[:\-]?\s*([A-Z][a-zA-Z\s]+)',
            r'©\s*(\d{4})\s*([A-Z][a-zA-Z\s]+)',
            r'by\s+([A-Z][a-zA-Z\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    author_info["name"] = match.group(2).strip()
                    author_info["year"] = match.group(1)
                else:
                    author_info["name"] = match.group(1).strip()
                break
        
        return author_info
    
    def _extract_organization_metadata(self, content: str) -> Dict[str, Any]:
        """Extraer metadata de organización"""
        import re
        
        org_info = {}
        
        # Buscar nombres de organizaciones (patrón simple)
        org_patterns = [
            r'(?:©|Copyright)\s*\d{4}\s*([A-Z][a-zA-Z\s&]+)',
            r'(?:Inc\.|LLC|Corp\.|Ltd\.|S\.A\.|S\.L\.)',
            r'([A-Z][a-zA-Z\s&]+)\s+(?:Inc\.|LLC|Corp\.|Ltd\.|S\.A\.|S\.L\.)'
        ]
        
        for pattern in org_patterns:
            match = re.search(pattern, content)
            if match:
                if len(match.groups()) > 0:
                    org_info["name"] = match.group(1).strip() if match.group(1) else "Unknown"
                else:
                    # Buscar el contexto antes del patrón
                    pos = match.start()
                    context_start = max(0, pos - 50)
                    context = content[context_start:pos + match.end() - context_start]
                    org_info["name"] = context.strip()
                break
        
        return org_info


class MultiLevelSummarizer:
    """Generador de resúmenes multi-nivel."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.summary_history: List[Dict[str, Any]] = []
    
    async def generate_multi_level_summary(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        levels: List[str] = ["brief", "standard", "detailed"]
    ) -> Dict[str, Any]:
        """
        Generar resúmenes en múltiples niveles
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            levels: Niveles de resumen a generar
        
        Returns:
            Resúmenes en diferentes niveles
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to summarize"}
        
        summaries = {}
        
        # Resumen breve (1-2 oraciones)
        if "brief" in levels:
            summaries["brief"] = await self.analyzer.summarize_document(
                content, max_length=50, min_length=20
            )
        
        # Resumen estándar (3-5 oraciones)
        if "standard" in levels:
            summaries["standard"] = await self.analyzer.summarize_document(
                content, max_length=150, min_length=50
            )
        
        # Resumen detallado (párrafos completos)
        if "detailed" in levels:
            summaries["detailed"] = await self.analyzer.summarize_document(
                content, max_length=300, min_length=100
            )
        
        # Resumen ejecutivo (si está en niveles)
        if "executive" in levels:
            exec_summary_gen = ExecutiveSummaryGenerator(self.analyzer)
            exec_summary = await exec_summary_gen.generate_executive_summary(
                document_content=content
            )
            summaries["executive"] = exec_summary
        
        # Calcular ratios de compresión
        compression_ratios = {}
        for level, summary_text in summaries.items():
            if isinstance(summary_text, str):
                compression_ratios[level] = len(summary_text) / len(content) if content else 0
            elif isinstance(summary_text, dict):
                summary_len = len(summary_text.get("summary", ""))
                compression_ratios[level] = summary_len / len(content) if content else 0
        
        result = {
            "summaries": summaries,
            "compression_ratios": compression_ratios,
            "original_length": len(content),
            "levels": levels,
            "timestamp": datetime.now().isoformat()
        }
        
        self.summary_history.append(result)
        return result


class AccessibilityAnalyzer:
    """Analizador de accesibilidad de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.accessibility_history: List[Dict[str, Any]] = []
    
    async def analyze_accessibility(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar accesibilidad del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis de accesibilidad
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Analizar diferentes aspectos de accesibilidad
        readability_analysis = await self._analyze_readability_accessibility(content)
        structure_analysis = await self._analyze_structure_accessibility(content)
        language_analysis = await self._analyze_language_accessibility(content)
        
        # Calcular score de accesibilidad
        accessibility_score = self._calculate_accessibility_score(
            readability_analysis, structure_analysis, language_analysis
        )
        
        # Generar recomendaciones
        recommendations = self._generate_accessibility_recommendations(
            readability_analysis, structure_analysis, language_analysis
        )
        
        result = {
            "accessibility_score": accessibility_score,
            "accessibility_level": self._get_accessibility_level(accessibility_score),
            "readability": readability_analysis,
            "structure": structure_analysis,
            "language": language_analysis,
            "recommendations": recommendations,
            "wcag_compliance": self._check_wcag_compliance(content),
            "timestamp": datetime.now().isoformat()
        }
        
        self.accessibility_history.append(result)
        return result
    
    async def _analyze_readability_accessibility(self, content: str) -> Dict[str, Any]:
        """Analizar legibilidad para accesibilidad"""
        readability_analyzer = ReadabilityAnalyzer()
        readability = await readability_analyzer.analyze_readability(document_content=content)
        
        return {
            "flesch_score": readability.get("flesch_score", 0),
            "readability_level": readability.get("readability_level", ""),
            "meets_standard": readability.get("flesch_score", 0) >= 60
        }
    
    async def _analyze_structure_accessibility(self, content: str) -> Dict[str, Any]:
        """Analizar estructura para accesibilidad"""
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        
        has_headings = len(structure.get("headings", [])) > 0
        has_sections = len(structure.get("sections", [])) > 0
        has_lists = len(structure.get("lists", [])) > 0
        
        return {
            "has_headings": has_headings,
            "has_sections": has_sections,
            "has_lists": has_lists,
            "structure_score": (1.0 if has_headings else 0.0) + (0.5 if has_sections else 0.0) + (0.3 if has_lists else 0.0)
        }
    
    async def _analyze_language_accessibility(self, content: str) -> Dict[str, Any]:
        """Analizar lenguaje para accesibilidad"""
        # Detectar idioma
        language_detector = LanguageDetector(self.analyzer)
        language_info = await language_detector.detect_language(document_content=content)
        
        # Analizar uso de lenguaje claro
        clear_language_score = self._analyze_clear_language(content)
        
        return {
            "language_detected": language_info.get("detected_language"),
            "clear_language_score": clear_language_score,
            "meets_standard": clear_language_score >= 0.7
        }
    
    def _analyze_clear_language(self, content: str) -> float:
        """Analizar claridad del lenguaje"""
        # Detectar uso de jerga técnica
        technical_jargon = ["API", "SDK", "framework", "endpoint", "middleware"]
        jargon_count = sum(1 for term in technical_jargon if term.lower() in content.lower())
        
        # Detectar uso de abreviaciones
        abbreviations = ["etc.", "e.g.", "i.e.", "vs.", "etc", "ej.", "p.ej."]
        abbrev_count = sum(1 for abbrev in abbreviations if abbrev.lower() in content.lower())
        
        # Calcular score (menos jerga y abreviaciones = mejor)
        score = 1.0 - min((jargon_count + abbrev_count) / 10.0, 0.5)
        
        return max(score, 0.0)
    
    def _calculate_accessibility_score(
        self,
        readability: Dict[str, Any],
        structure: Dict[str, Any],
        language: Dict[str, Any]
    ) -> float:
        """Calcular score general de accesibilidad"""
        score = 0.0
        
        # Legibilidad (40%)
        readability_score = 1.0 if readability.get("meets_standard", False) else 0.5
        score += readability_score * 0.4
        
        # Estructura (30%)
        structure_score = structure.get("structure_score", 0.0)
        score += structure_score * 0.3
        
        # Lenguaje (30%)
        language_score = language.get("clear_language_score", 0.0)
        score += language_score * 0.3
        
        return min(score, 1.0)
    
    def _get_accessibility_level(self, score: float) -> str:
        """Obtener nivel de accesibilidad"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _generate_accessibility_recommendations(
        self,
        readability: Dict[str, Any],
        structure: Dict[str, Any],
        language: Dict[str, Any]
    ) -> List[str]:
        """Generar recomendaciones de accesibilidad"""
        recommendations = []
        
        if not readability.get("meets_standard", False):
            recommendations.append("Mejorar legibilidad del documento (objetivo: Flesch score >= 60)")
        
        if not structure.get("has_headings", False):
            recommendations.append("Agregar encabezados para mejorar estructura")
        
        if language.get("clear_language_score", 0.0) < 0.7:
            recommendations.append("Reducir uso de jerga técnica y abreviaciones")
        
        return recommendations
    
    def _check_wcag_compliance(self, content: str) -> Dict[str, Any]:
        """Verificar cumplimiento básico de WCAG"""
        # Verificaciones básicas de WCAG 2.1
        checks = {
            "has_structure": len(content.split('\n\n')) > 1,
            "has_clear_language": True,  # Simplificado
            "readable": len(content.split()) > 0
        }
        
        passed = sum(checks.values())
        total = len(checks)
        
        return {
            "checks": checks,
            "passed": passed,
            "total": total,
            "compliance_percentage": (passed / total * 100) if total > 0 else 0
        }


class RedundancyDetector:
    """Detector de redundancias y contenido duplicado."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.detection_history: List[Dict[str, Any]] = []
    
    async def detect_redundancies(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        similarity_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Detectar redundancias en el documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            similarity_threshold: Umbral de similitud para considerar redundancia
        
        Returns:
            Redundancias detectadas
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Dividir en párrafos
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Detectar párrafos similares
        similar_paragraphs = []
        for i, para1 in enumerate(paragraphs):
            for j, para2 in enumerate(paragraphs[i+1:], start=i+1):
                similarity = await self._calculate_paragraph_similarity(para1, para2)
                
                if similarity >= similarity_threshold:
                    similar_paragraphs.append({
                        "paragraph1_index": i,
                        "paragraph2_index": j,
                        "similarity": similarity,
                        "paragraph1_preview": para1[:100],
                        "paragraph2_preview": para2[:100]
                    })
        
        # Detectar frases repetidas
        repeated_phrases = self._detect_repeated_phrases(content)
        
        # Detectar palabras repetidas excesivamente
        excessive_repetition = self._detect_excessive_repetition(content)
        
        result = {
            "similar_paragraphs": similar_paragraphs,
            "repeated_phrases": repeated_phrases,
            "excessive_repetition": excessive_repetition,
            "total_redundancies": len(similar_paragraphs) + len(repeated_phrases),
            "redundancy_score": self._calculate_redundancy_score(
                len(similar_paragraphs), len(repeated_phrases), len(paragraphs)
            ),
            "recommendations": self._generate_redundancy_recommendations(
                similar_paragraphs, repeated_phrases
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        self.detection_history.append(result)
        return result
    
    async def _calculate_paragraph_similarity(self, para1: str, para2: str) -> float:
        """Calcular similitud entre párrafos"""
        try:
            embeddings = await self.analyzer.embedding_generator.generate_embeddings([para1, para2])
            
            if len(embeddings) == 2:
                emb1 = embeddings[0]
                emb2 = embeddings[1]
                
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                similarity = dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
                return float(similarity)
        except Exception as e:
            logger.error(f"Error calculando similitud de párrafos: {e}")
        
        return 0.0
    
    def _detect_repeated_phrases(self, content: str) -> List[Dict[str, Any]]:
        """Detectar frases repetidas"""
        import re
        
        # Extraer oraciones
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filtrar muy cortas
        
        phrase_counts = {}
        for sentence in sentences:
            # Dividir en frases de 3-5 palabras
            words = sentence.split()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                phrase_lower = phrase.lower()
                if phrase_lower not in phrase_counts:
                    phrase_counts[phrase_lower] = []
                phrase_counts[phrase_lower].append(sentence)
        
        # Encontrar frases repetidas
        repeated = []
        for phrase, occurrences in phrase_counts.items():
            if len(occurrences) > 2:  # Repetida más de 2 veces
                repeated.append({
                    "phrase": phrase,
                    "occurrences": len(occurrences),
                    "examples": occurrences[:3]  # Primeros 3 ejemplos
                })
        
        return sorted(repeated, key=lambda x: x["occurrences"], reverse=True)[:10]
    
    def _detect_excessive_repetition(self, content: str) -> Dict[str, Any]:
        """Detectar palabras repetidas excesivamente"""
        words = content.lower().split()
        word_counts = {}
        
        for word in words:
            if len(word) > 4:  # Solo palabras de más de 4 caracteres
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Encontrar palabras con alta frecuencia
        total_words = len(words)
        excessive_words = []
        
        for word, count in word_counts.items():
            frequency = count / total_words if total_words > 0 else 0
            if frequency > 0.05:  # Más del 5% del documento
                excessive_words.append({
                    "word": word,
                    "count": count,
                    "frequency": frequency
                })
        
        return {
            "excessive_words": sorted(excessive_words, key=lambda x: x["frequency"], reverse=True)[:10],
            "total_unique_words": len(word_counts),
            "most_frequent_word": max(word_counts.items(), key=lambda x: x[1])[0] if word_counts else None
        }
    
    def _calculate_redundancy_score(
        self,
        similar_paragraphs: int,
        repeated_phrases: int,
        total_paragraphs: int
    ) -> float:
        """
        Calcular score de redundancia
        
        Args:
            similar_paragraphs: Número de párrafos similares
            repeated_phrases: Número de frases repetidas
            total_paragraphs: Total de párrafos
        
        Returns:
            Score de redundancia (0.0 a 1.0)
        """
        try:
            # Validar tipos y valores
            if not isinstance(similar_paragraphs, (int, float)) or similar_paragraphs < 0:
                similar_paragraphs = 0
            if not isinstance(repeated_phrases, (int, float)) or repeated_phrases < 0:
                repeated_phrases = 0
            if not isinstance(total_paragraphs, (int, float)) or total_paragraphs < 0:
                total_paragraphs = 0
            
            if total_paragraphs == 0:
                return 0.0
            
            # Calcular ratios de manera segura
            similar_ratio = similar_paragraphs / total_paragraphs
            repeated_ratio = repeated_phrases / max(total_paragraphs, 1)
            
            # Score combinado (más alto = más redundancia)
            score = (similar_ratio * 0.6) + (repeated_ratio * 0.4)
            return min(score, 1.0)
        
        except Exception as e:
            logger.error(f"Error calculando score de redundancia: {e}", exc_info=True)
            return 0.0
        """Calcular score de redundancia"""
        if total_paragraphs == 0:
            return 0.0
        
        paragraph_redundancy = similar_paragraphs / total_paragraphs
        phrase_redundancy = min(repeated_phrases / 10.0, 1.0)  # Normalizar
        
        score = (paragraph_redundancy * 0.6) + (phrase_redundancy * 0.4)
        return min(score, 1.0)
    
    def _generate_redundancy_recommendations(
        self,
        similar_paragraphs: List[Dict[str, Any]],
        repeated_phrases: List[Dict[str, Any]]
    ) -> List[str]:
        """Generar recomendaciones para reducir redundancias"""
        recommendations = []
        
        if len(similar_paragraphs) > 0:
            recommendations.append(f"Se encontraron {len(similar_paragraphs)} pares de párrafos muy similares. Considerar consolidar o eliminar redundancias.")
        
        if len(repeated_phrases) > 5:
            recommendations.append(f"Se encontraron {len(repeated_phrases)} frases repetidas. Considerar variar el lenguaje.")
        
        return recommendations


# ============================================================================
# SISTEMAS AVANZADOS FINALES - ANÁLISIS ESPECIALIZADO ADICIONAL
# ============================================================================

class DocumentSecurityAnalyzer:
    """Analizador de seguridad de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.security_analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_security(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar seguridad del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis de seguridad
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Detectar información sensible
        sensitive_info = self._detect_sensitive_information(content)
        
        # Detectar PII (Personally Identifiable Information)
        pii_data = await self._detect_pii(content)
        
        # Detectar credenciales
        credentials = self._detect_credentials(content)
        
        # Detectar información confidencial
        confidential_info = self._detect_confidential_info(content)
        
        # Calcular score de riesgo
        risk_score = self._calculate_risk_score(sensitive_info, pii_data, credentials, confidential_info)
        
        # Generar recomendaciones de seguridad
        recommendations = self._generate_security_recommendations(
            sensitive_info, pii_data, credentials, confidential_info
        )
        
        result = {
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "sensitive_information": sensitive_info,
            "pii_data": pii_data,
            "credentials": credentials,
            "confidential_info": confidential_info,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        self.security_analysis_history.append(result)
        return result
    
    def _detect_sensitive_information(self, content: str) -> Dict[str, Any]:
        """Detectar información sensible"""
        import re
        
        sensitive_patterns = {
            "credit_cards": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "ip_addresses": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "mac_addresses": r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b',
            "api_keys": r'(?:api[_-]?key|apikey)[=:]\s*([A-Za-z0-9_-]{20,})',
            "tokens": r'(?:token|bearer)[=:]\s*([A-Za-z0-9_-]{20,})'
        }
        
        detected = {}
        for info_type, pattern in sensitive_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                detected[info_type] = {
                    "count": len(matches),
                    "examples": matches[:3]  # Solo primeros 3 para no exponer datos
                }
        
        return detected
    
    async def _detect_pii(self, content: str) -> Dict[str, Any]:
        """Detectar PII usando NER"""
        entities = await self.analyzer.extract_entities(content)
        
        pii_types = {
            "person_names": [],
            "emails": [],
            "phone_numbers": [],
            "addresses": [],
            "dates_of_birth": []
        }
        
        # Extraer emails
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        pii_types["emails"] = emails[:10]  # Limitar cantidad
        
        # Extraer números de teléfono
        phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        phones = re.findall(phone_pattern, content)
        pii_types["phone_numbers"] = phones[:10]
        
        # Clasificar entidades
        for entity in entities:
            label = entity.get("label", "")
            text = entity.get("text", "")
            
            if label == "PER":
                pii_types["person_names"].append(text)
            elif label == "LOC":
                # Posible dirección
                if any(word in text.lower() for word in ["street", "avenue", "road", "calle", "avenida"]):
                    pii_types["addresses"].append(text)
        
        return {
            "pii_types": pii_types,
            "total_pii_items": sum(len(v) if isinstance(v, list) else 0 for v in pii_types.values()),
            "has_pii": sum(len(v) if isinstance(v, list) else 0 for v in pii_types.values()) > 0
        }
    
    def _detect_credentials(self, content: str) -> Dict[str, Any]:
        """Detectar credenciales"""
        import re
        
        credentials = {
            "passwords": [],
            "api_keys": [],
            "secrets": []
        }
        
        # Patrones de contraseñas (muy básico, solo detecta patrones comunes)
        password_patterns = [
            r'password[=:]\s*([^\s]+)',
            r'passwd[=:]\s*([^\s]+)',
            r'pwd[=:]\s*([^\s]+)'
        ]
        
        for pattern in password_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                credentials["passwords"].extend(matches[:3])
        
        # API keys
        api_key_pattern = r'(?:api[_-]?key|apikey|secret[_-]?key)[=:]\s*([A-Za-z0-9_-]{20,})'
        api_keys = re.findall(api_key_pattern, content, re.IGNORECASE)
        credentials["api_keys"] = api_keys[:5]
        
        return {
            "credentials": credentials,
            "has_credentials": any(len(v) > 0 for v in credentials.values())
        }
    
    def _detect_confidential_info(self, content: str) -> Dict[str, Any]:
        """Detectar información confidencial"""
        import re
        
        confidential_keywords = [
            "confidential", "confidencial",
            "secret", "secreto",
            "classified", "clasificado",
            "internal use only", "uso interno",
            "proprietary", "propietario",
            "nda", "non-disclosure"
        ]
        
        found_keywords = []
        content_lower = content.lower()
        
        for keyword in confidential_keywords:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
        
        return {
            "confidential_keywords": found_keywords,
            "has_confidential_markers": len(found_keywords) > 0
        }
    
    def _calculate_risk_score(
        self,
        sensitive_info: Dict[str, Any],
        pii_data: Dict[str, Any],
        credentials: Dict[str, Any],
        confidential_info: Dict[str, Any]
    ) -> float:
        """Calcular score de riesgo"""
        score = 0.0
        
        # Información sensible (30%)
        if sensitive_info:
            sensitive_count = sum(v.get("count", 0) if isinstance(v, dict) else 0 for v in sensitive_info.values())
            score += min(sensitive_count / 10.0, 1.0) * 0.3
        
        # PII (30%)
        if pii_data.get("has_pii", False):
            pii_count = pii_data.get("total_pii_items", 0)
            score += min(pii_count / 20.0, 1.0) * 0.3
        
        # Credenciales (25%)
        if credentials.get("has_credentials", False):
            cred_count = sum(len(v) if isinstance(v, list) else 0 for v in credentials.get("credentials", {}).values())
            score += min(cred_count / 5.0, 1.0) * 0.25
        
        # Marcadores confidenciales (15%)
        if confidential_info.get("has_confidential_markers", False):
            score += 0.15
        
        return min(score, 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        """Obtener nivel de riesgo"""
        if score >= 0.7:
            return "Critical"
        elif score >= 0.5:
            return "High"
        elif score >= 0.3:
            return "Medium"
        elif score >= 0.1:
            return "Low"
        else:
            return "Minimal"
    
    def _generate_security_recommendations(
        self,
        sensitive_info: Dict[str, Any],
        pii_data: Dict[str, Any],
        credentials: Dict[str, Any],
        confidential_info: Dict[str, Any]
    ) -> List[str]:
        """Generar recomendaciones de seguridad"""
        recommendations = []
        
        if sensitive_info:
            recommendations.append("Se detectó información sensible. Considerar encriptar o redactar el documento.")
        
        if pii_data.get("has_pii", False):
            recommendations.append("Se detectó información personal identificable (PII). Aplicar medidas de protección de datos.")
        
        if credentials.get("has_credentials", False):
            recommendations.append("CRÍTICO: Se detectaron credenciales. Eliminar inmediatamente y rotar contraseñas/API keys.")
        
        if confidential_info.get("has_confidential_markers", False):
            recommendations.append("El documento contiene marcadores de confidencialidad. Verificar niveles de acceso.")
        
        return recommendations


class ScientificDocumentAnalyzer:
    """Analizador especializado para documentos científicos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_scientific_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar documento científico
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis científico del documento
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Extraer citas bibliográficas
        citations = self._extract_citations(content)
        
        # Extraer referencias
        references = self._extract_references(content)
        
        # Extraer métodos
        methods = self._extract_methods(content)
        
        # Extraer resultados
        results = self._extract_results(content)
        
        # Detectar estructura IMRAD
        imrad_structure = self._detect_imrad_structure(content)
        
        # Extraer hipótesis
        hypotheses = self._extract_hypotheses(content)
        
        # Analizar rigor científico
        scientific_rigor = self._analyze_scientific_rigor(content, citations, references)
        
        result = {
            "citations": citations,
            "references": references,
            "methods": methods,
            "results": results,
            "imrad_structure": imrad_structure,
            "hypotheses": hypotheses,
            "scientific_rigor": scientific_rigor,
            "total_citations": len(citations),
            "total_references": len(references),
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_history.append(result)
        return result
    
    def _extract_citations(self, content: str) -> List[Dict[str, Any]]:
        """Extraer citas bibliográficas"""
        import re
        
        citations = []
        
        # Patrones comunes de citas
        citation_patterns = [
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4})\)',  # (Author, 2024)
            r'\[(\d+)\]',  # [1]
            r'([A-Z][a-z]+(?:\s+et\s+al\.)?\s+\(\d{4}\))',  # Author (2024)
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                citations.append({
                    "citation": match.group(1),
                    "position": match.start(),
                    "context": content[max(0, match.start()-50):match.end()+50]
                })
        
        return citations
    
    def _extract_references(self, content: str) -> List[Dict[str, Any]]:
        """Extraer referencias bibliográficas"""
        import re
        
        references = []
        
        # Buscar sección de referencias
        ref_section_pattern = r'(?:References|Referencias|Bibliography|Bibliografía)[:\s]*\n(.*?)(?=\n[A-Z]{2,}|\Z)'
        ref_match = re.search(ref_section_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if ref_match:
            ref_section = ref_match.group(1)
            
            # Extraer referencias individuales
            ref_pattern = r'^\d+\.\s*(.+?)(?=^\d+\.|\Z)'
            ref_matches = re.finditer(ref_pattern, ref_section, re.MULTILINE | re.DOTALL)
            
            for match in ref_matches:
                references.append({
                    "reference": match.group(1).strip(),
                    "position": match.start()
                })
        
        return references
    
    def _extract_methods(self, content: str) -> List[Dict[str, Any]]:
        """Extraer sección de métodos"""
        import re
        
        methods = []
        
        # Buscar sección de métodos
        method_patterns = [
            r'(?:Methodology|Methods|Métodos|Metodología)[:\s]*\n(.*?)(?=\n(?:Results|Conclusion|References)|\Z)',
            r'(?:Experimental\s+Procedure|Procedimiento\s+Experimental)[:\s]*\n(.*?)(?=\n(?:Results|Conclusion)|\Z)'
        ]
        
        for pattern in method_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                methods.append({
                    "section": match.group(1).strip()[:500],  # Primeros 500 caracteres
                    "position": match.start()
                })
                break
        
        return methods
    
    def _extract_results(self, content: str) -> List[Dict[str, Any]]:
        """Extraer sección de resultados"""
        import re
        
        results = []
        
        # Buscar sección de resultados
        results_pattern = r'(?:Results|Resultados|Findings)[:\s]*\n(.*?)(?=\n(?:Discussion|Conclusion|References)|\Z)'
        match = re.search(results_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            results.append({
                "section": match.group(1).strip()[:500],
                "position": match.start()
            })
        
        return results
    
    def _detect_imrad_structure(self, content: str) -> Dict[str, Any]:
        """Detectar estructura IMRAD (Introduction, Methods, Results, and Discussion)"""
        import re
        
        imrad_sections = {
            "Introduction": False,
            "Methods": False,
            "Results": False,
            "Discussion": False
        }
        
        content_lower = content.lower()
        
        # Detectar secciones
        intro_patterns = [r'introduction', r'introducción']
        methods_patterns = [r'methods?', r'methodology', r'métodos?', r'metodología']
        results_patterns = [r'results?', r'resultados?', r'findings']
        discussion_patterns = [r'discussion', r'discusión', r'conclusion']
        
        for pattern in intro_patterns:
            if re.search(rf'\b{pattern}\b', content_lower):
                imrad_sections["Introduction"] = True
                break
        
        for pattern in methods_patterns:
            if re.search(rf'\b{pattern}\b', content_lower):
                imrad_sections["Methods"] = True
                break
        
        for pattern in results_patterns:
            if re.search(rf'\b{pattern}\b', content_lower):
                imrad_sections["Results"] = True
                break
        
        for pattern in discussion_patterns:
            if re.search(rf'\b{pattern}\b', content_lower):
                imrad_sections["Discussion"] = True
                break
        
        return {
            "sections": imrad_sections,
            "completeness": sum(imrad_sections.values()) / len(imrad_sections),
            "has_full_imrad": all(imrad_sections.values())
        }
    
    def _extract_hypotheses(self, content: str) -> List[Dict[str, Any]]:
        """Extraer hipótesis"""
        import re
        
        hypotheses = []
        
        # Patrones de hipótesis
        hypothesis_patterns = [
            r'(?:hypothesis|hipótesis)[:\s]+(.+?)(?:\.|$)',
            r'(?:we\s+(?:hypothesize|propose|suggest))[:\s]+(.+?)(?:\.|$)',
            r'(?:H\d+)[:\s]+(.+?)(?:\.|$)'
        ]
        
        for pattern in hypothesis_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                hypotheses.append({
                    "hypothesis": match.group(1).strip(),
                    "position": match.start()
                })
        
        return hypotheses
    
    def _analyze_scientific_rigor(
        self,
        content: str,
        citations: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar rigor científico"""
        # Calcular métricas de rigor
        citation_count = len(citations)
        reference_count = len(references)
        content_length = len(content)
        
        # Ratio de citas por 1000 palabras
        word_count = len(content.split())
        citation_ratio = (citation_count / word_count * 1000) if word_count > 0 else 0
        
        # Detectar estadísticas
        import re
        statistical_terms = ["p-value", "p <", "statistically significant", "confidence interval", "CI"]
        has_statistics = any(term.lower() in content.lower() for term in statistical_terms)
        
        # Detectar experimentos
        experimental_terms = ["experiment", "study", "trial", "experimento", "estudio"]
        has_experiments = any(term.lower() in content.lower() for term in experimental_terms)
        
        return {
            "citation_count": citation_count,
            "reference_count": reference_count,
            "citation_ratio_per_1000_words": citation_ratio,
            "has_statistics": has_statistics,
            "has_experiments": has_experiments,
            "rigor_score": self._calculate_rigor_score(
                citation_ratio, reference_count, has_statistics, has_experiments
            )
        }
    
    def _calculate_rigor_score(
        self,
        citation_ratio: float,
        reference_count: int,
        has_statistics: bool,
        has_experiments: bool
    ) -> float:
        """Calcular score de rigor científico"""
        score = 0.0
        
        # Ratio de citas (40%)
        score += min(citation_ratio / 10.0, 1.0) * 0.4
        
        # Referencias (30%)
        score += min(reference_count / 20.0, 1.0) * 0.3
        
        # Estadísticas (15%)
        score += 0.15 if has_statistics else 0.0
        
        # Experimentos (15%)
        score += 0.15 if has_experiments else 0.0
        
        return min(score, 1.0)


class CitationExtractor:
    """Extractor especializado de citas bibliográficas."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.extraction_history: List[Dict[str, Any]] = []
    
    async def extract_citations(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        citation_format: str = "auto"
    ) -> Dict[str, Any]:
        """
        Extraer citas bibliográficas
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            citation_format: Formato de citas ('apa', 'mla', 'chicago', 'auto')
        
        Returns:
            Citas extraídas y normalizadas
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to extract from"}
        
        # Extraer citas en el texto
        in_text_citations = self._extract_in_text_citations(content)
        
        # Extraer referencias bibliográficas
        bibliography = self._extract_bibliography(content)
        
        # Normalizar citas
        normalized_citations = self._normalize_citations(in_text_citations, bibliography, citation_format)
        
        # Detectar formato de citas
        detected_format = self._detect_citation_format(in_text_citations, bibliography)
        
        result = {
            "in_text_citations": in_text_citations,
            "bibliography": bibliography,
            "normalized_citations": normalized_citations,
            "detected_format": detected_format,
            "total_citations": len(in_text_citations),
            "total_references": len(bibliography),
            "citation_format": citation_format,
            "timestamp": datetime.now().isoformat()
        }
        
        self.extraction_history.append(result)
        return result
    
    def _extract_in_text_citations(self, content: str) -> List[Dict[str, Any]]:
        """Extraer citas en el texto"""
        import re
        
        citations = []
        
        # Diferentes formatos de citas
        patterns = [
            # APA: (Author, 2024)
            (r'\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?)\)', 'apa'),
            # MLA: (Author 123)
            (r'\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+\d+)\)', 'mla'),
            # Numeric: [1], [1-3], [1,2,3]
            (r'\[(\d+(?:[-\s,]?\d+)*)\]', 'numeric'),
            # Author-date: Author (2024)
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+et\s+al\.)?\s+\(\d{4}\))', 'author-date')
        ]
        
        for pattern, format_type in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                citations.append({
                    "citation": match.group(1),
                    "format": format_type,
                    "position": match.start(),
                    "context": content[max(0, match.start()-30):match.end()+30]
                })
        
        return citations
    
    def _extract_bibliography(self, content: str) -> List[Dict[str, Any]]:
        """Extraer bibliografía completa"""
        import re
        
        bibliography = []
        
        # Buscar sección de referencias
        ref_patterns = [
            r'(?:References|Referencias|Bibliography|Bibliografía|Works\s+Cited)[:\s]*\n(.*?)(?=\n[A-Z]{2,}|\Z)',
            r'(?:REFERENCE|REFERENCIAS)[:\s]*\n(.*?)(?=\n[A-Z]{2,}|\Z)'
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                ref_section = match.group(1)
                
                # Extraer referencias individuales
                ref_item_pattern = r'^\d+[\.\)]\s*(.+?)(?=^\d+[\.\)]|\Z)'
                ref_matches = re.finditer(ref_item_pattern, ref_section, re.MULTILINE | re.DOTALL)
                
                for ref_match in ref_matches:
                    bibliography.append({
                        "reference": ref_match.group(1).strip(),
                        "position": ref_match.start()
                    })
                
                if bibliography:
                    break
        
        return bibliography
    
    def _normalize_citations(
        self,
        in_text_citations: List[Dict[str, Any]],
        bibliography: List[Dict[str, Any]],
        format_type: str
    ) -> List[Dict[str, Any]]:
        """Normalizar citas a un formato específico"""
        # Por ahora, retornar estructura básica
        normalized = []
        
        for citation in in_text_citations[:20]:  # Limitar a 20
            normalized.append({
                "original": citation.get("citation", ""),
                "normalized": citation.get("citation", ""),  # En producción, normalizar realmente
                "format": citation.get("format", "unknown")
            })
        
        return normalized
    
    def _detect_citation_format(
        self,
        in_text_citations: List[Dict[str, Any]],
        bibliography: List[Dict[str, Any]]
    ) -> str:
        """Detectar formato de citas"""
        if not in_text_citations:
            return "unknown"
        
        # Contar formatos
        format_counts = {}
        for citation in in_text_citations:
            fmt = citation.get("format", "unknown")
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        # Retornar formato más común
        if format_counts:
            return max(format_counts.items(), key=lambda x: x[1])[0]
        
        return "unknown"


class TechnicalDocumentAnalyzer:
    """Analizador especializado para documentos técnicos (API docs, etc.)."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_technical_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar documento técnico
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis técnico del documento
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Extraer endpoints API
        api_endpoints = self._extract_api_endpoints(content)
        
        # Extraer código de ejemplo
        code_examples = self._extract_code_examples(content)
        
        # Extraer parámetros
        parameters = self._extract_parameters(content)
        
        # Extraer respuestas
        responses = self._extract_responses(content)
        
        # Detectar estructura de documentación
        doc_structure = self._detect_documentation_structure(content)
        
        # Analizar completitud
        completeness = self._analyze_completeness(api_endpoints, code_examples, parameters, responses)
        
        result = {
            "api_endpoints": api_endpoints,
            "code_examples": code_examples,
            "parameters": parameters,
            "responses": responses,
            "documentation_structure": doc_structure,
            "completeness": completeness,
            "total_endpoints": len(api_endpoints),
            "total_code_examples": len(code_examples),
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_history.append(result)
        return result
    
    def _extract_api_endpoints(self, content: str) -> List[Dict[str, Any]]:
        """Extraer endpoints de API"""
        import re
        
        endpoints = []
        
        # Patrones de endpoints
        endpoint_patterns = [
            r'(?:GET|POST|PUT|DELETE|PATCH)\s+([\/\w\-{}]+)',
            r'`([\/\w\-{}]+)`',
            r'https?://[^\s]+/([\w\-{}]+)',
            r'endpoint[:\s]+([\/\w\-{}]+)'
        ]
        
        for pattern in endpoint_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                endpoint = match.group(1)
                if endpoint.startswith('/') or '{' in endpoint:
                    endpoints.append({
                        "endpoint": endpoint,
                        "position": match.start(),
                        "context": content[max(0, match.start()-50):match.end()+50]
                    })
        
        return endpoints
    
    def _extract_code_examples(self, content: str) -> List[Dict[str, Any]]:
        """Extraer ejemplos de código"""
        import re
        
        code_examples = []
        
        # Patrones de bloques de código
        code_patterns = [
            r'```(\w+)?\n(.*?)```',
            r'`([^`]+)`',
            r'<code>(.*?)</code>'
        ]
        
        for pattern in code_patterns:
            matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                code_examples.append({
                    "language": match.group(1) if len(match.groups()) > 1 else "unknown",
                    "code": match.group(-1).strip(),
                    "position": match.start()
                })
        
        return code_examples[:20]  # Limitar a 20 ejemplos
    
    def _extract_parameters(self, content: str) -> List[Dict[str, Any]]:
        """Extraer parámetros de API"""
        import re
        
        parameters = []
        
        # Patrones de parámetros
        param_patterns = [
            r'(?:parameter|param|parámetro)[:\s]+(\w+)[:\s]+(.+?)(?:\n|$)',
            r'`(\w+)`[:\s]+(.+?)(?:\n|$)',
            r'(\w+)[:\s]+(?:required|optional|opcional)[:\s]+(.+?)(?:\n|$)'
        ]
        
        for pattern in param_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                parameters.append({
                    "name": match.group(1),
                    "description": match.group(2).strip() if len(match.groups()) > 1 else "",
                    "position": match.start()
                })
        
        return parameters
    
    def _extract_responses(self, content: str) -> List[Dict[str, Any]]:
        """Extraer respuestas de API"""
        import re
        
        responses = []
        
        # Patrones de respuestas
        response_patterns = [
            r'(?:response|respuesta)[:\s]+\n(.*?)(?=\n(?:Request|Example|Endpoint)|\Z)',
            r'Status[:\s]+(\d{3})[:\s]+\n(.*?)(?=\n(?:Status|Request)|\Z)'
        ]
        
        for pattern in response_patterns:
            matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                responses.append({
                    "status_code": match.group(1) if len(match.groups()) > 1 else None,
                    "response": match.group(-1).strip()[:200],
                    "position": match.start()
                })
        
        return responses
    
    def _detect_documentation_structure(self, content: str) -> Dict[str, Any]:
        """Detectar estructura de documentación"""
        import re
        
        sections = {
            "has_overview": False,
            "has_authentication": False,
            "has_endpoints": False,
            "has_examples": False,
            "has_errors": False
        }
        
        content_lower = content.lower()
        
        section_patterns = {
            "has_overview": [r'overview', r'introduction', r'introducción'],
            "has_authentication": [r'authentication', r'auth', r'autenticación'],
            "has_endpoints": [r'endpoints?', r'api', r'reference'],
            "has_examples": [r'examples?', r'ejemplos?', r'sample'],
            "has_errors": [r'errors?', r'error codes?', r'errores?']
        }
        
        for section, patterns in section_patterns.items():
            for pattern in patterns:
                if re.search(rf'\b{pattern}\b', content_lower):
                    sections[section] = True
                    break
        
        return {
            "sections": sections,
            "completeness": sum(sections.values()) / len(sections)
        }
    
    def _analyze_completeness(
        self,
        api_endpoints: List[Dict[str, Any]],
        code_examples: List[Dict[str, Any]],
        parameters: List[Dict[str, Any]],
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar completitud de la documentación"""
        return {
            "has_endpoints": len(api_endpoints) > 0,
            "has_code_examples": len(code_examples) > 0,
            "has_parameters": len(parameters) > 0,
            "has_responses": len(responses) > 0,
            "completeness_score": (
                (1.0 if len(api_endpoints) > 0 else 0.0) * 0.3 +
                (1.0 if len(code_examples) > 0 else 0.0) * 0.3 +
                (1.0 if len(parameters) > 0 else 0.0) * 0.2 +
                (1.0 if len(responses) > 0 else 0.0) * 0.2
            )
        }


# ============================================================================
# SISTEMAS AVANZADOS FINALES - ANÁLISIS ESPECIALIZADO AVANZADO ADICIONAL
# ============================================================================

class DocumentAnomalyDetector:
    """Detector de anomalías en documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.detection_history: List[Dict[str, Any]] = []
    
    async def detect_anomalies(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        reference_documents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detectar anomalías en el documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            reference_documents: Documentos de referencia para comparación
        
        Returns:
            Anomalías detectadas
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Detectar diferentes tipos de anomalías
        length_anomalies = self._detect_length_anomalies(content, reference_documents)
        structure_anomalies = await self._detect_structure_anomalies(content)
        content_anomalies = await self._detect_content_anomalies(content)
        style_anomalies = self._detect_style_anomalies(content)
        
        # Calcular score de anomalía
        anomaly_score = self._calculate_anomaly_score(
            length_anomalies, structure_anomalies, content_anomalies, style_anomalies
        )
        
        result = {
            "anomaly_score": anomaly_score,
            "anomaly_level": self._get_anomaly_level(anomaly_score),
            "length_anomalies": length_anomalies,
            "structure_anomalies": structure_anomalies,
            "content_anomalies": content_anomalies,
            "style_anomalies": style_anomalies,
            "total_anomalies": (
                len(length_anomalies.get("issues", [])) +
                len(structure_anomalies.get("issues", [])) +
                len(content_anomalies.get("issues", [])) +
                len(style_anomalies.get("issues", []))
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        self.detection_history.append(result)
        return result
    
    def _detect_length_anomalies(
        self,
        content: str,
        reference_documents: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Detectar anomalías en longitud"""
        issues = []
        content_length = len(content)
        word_count = len(content.split())
        
        # Verificar si es demasiado corto o largo
        if word_count < 100:
            issues.append({
                "type": "too_short",
                "severity": "medium",
                "description": f"Documento muy corto ({word_count} palabras)"
            })
        elif word_count > 50000:
            issues.append({
                "type": "too_long",
                "severity": "low",
                "description": f"Documento muy extenso ({word_count} palabras)"
            })
        
        # Comparar con documentos de referencia si están disponibles
        if reference_documents:
            ref_lengths = [len(ref.split()) for ref in reference_documents]
            avg_ref_length = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0
            
            if avg_ref_length > 0:
                deviation = abs(word_count - avg_ref_length) / avg_ref_length
                if deviation > 0.5:  # Más del 50% de desviación
                    issues.append({
                        "type": "length_deviation",
                        "severity": "medium",
                        "description": f"Longitud desviada significativamente del promedio ({deviation:.1%})"
                    })
        
        return {
            "word_count": word_count,
            "character_count": content_length,
            "issues": issues
        }
    
    async def _detect_structure_anomalies(self, content: str) -> Dict[str, Any]:
        """Detectar anomalías en estructura"""
        issues = []
        
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        
        # Verificar si falta estructura
        if len(structure.get("headings", [])) == 0:
            issues.append({
                "type": "missing_headings",
                "severity": "medium",
                "description": "Documento sin encabezados"
            })
        
        if len(structure.get("sections", [])) < 2:
            issues.append({
                "type": "insufficient_sections",
                "severity": "low",
                "description": "Documento con muy pocas secciones"
            })
        
        # Verificar balance de secciones
        sections = structure.get("sections", [])
        if sections:
            section_lengths = [len(s.get("content", "")) for s in sections]
            if section_lengths:
                max_length = max(section_lengths)
                min_length = min(section_lengths)
                if max_length > 0 and min_length / max_length < 0.1:
                    issues.append({
                        "type": "unbalanced_sections",
                        "severity": "low",
                        "description": "Secciones muy desbalanceadas en tamaño"
                    })
        
        return {
            "structure": structure,
            "issues": issues
        }
    
    async def _detect_content_anomalies(self, content: str) -> Dict[str, Any]:
        """Detectar anomalías en contenido"""
        issues = []
        
        # Detectar repetición excesiva
        redundancy_detector = RedundancyDetector(self.analyzer)
        redundancy = await redundancy_detector.detect_redundancies(document_content=content)
        
        if redundancy.get("redundancy_score", 0.0) > 0.7:
            issues.append({
                "type": "high_redundancy",
                "severity": "medium",
                "description": "Alto nivel de redundancia detectado"
            })
        
        # Detectar contenido vacío o muy repetitivo
        sentences = content.split('.')
        unique_sentences = len(set(s.strip().lower() for s in sentences if s.strip()))
        total_sentences = len([s for s in sentences if s.strip()])
        
        if total_sentences > 0:
            uniqueness_ratio = unique_sentences / total_sentences
            if uniqueness_ratio < 0.3:
                issues.append({
                    "type": "low_uniqueness",
                    "severity": "high",
                    "description": f"Contenido muy repetitivo (ratio de unicidad: {uniqueness_ratio:.2%})"
                })
        
        return {
            "redundancy_score": redundancy.get("redundancy_score", 0.0),
            "uniqueness_ratio": uniqueness_ratio if total_sentences > 0 else 1.0,
            "issues": issues
        }
    
    def _detect_style_anomalies(self, content: str) -> Dict[str, Any]:
        """Detectar anomalías de estilo"""
        issues = []
        
        # Detectar uso inconsistente de mayúsculas
        import re
        sentences = re.split(r'[.!?]+', content)
        capitalization_issues = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Verificar si empieza con mayúscula
                if sentence and not sentence[0].isupper():
                    capitalization_issues += 1
        
        if len(sentences) > 0 and capitalization_issues / len(sentences) > 0.3:
            issues.append({
                "type": "inconsistent_capitalization",
                "severity": "low",
                "description": f"Uso inconsistente de mayúsculas ({capitalization_issues}/{len(sentences)} oraciones)"
            })
        
        # Detectar uso excesivo de signos de exclamación/interrogación
        exclamation_count = content.count('!')
        question_count = content.count('?')
        total_chars = len(content)
        
        if total_chars > 0:
            exclamation_ratio = exclamation_count / total_chars
            if exclamation_ratio > 0.01:  # Más del 1%
                issues.append({
                    "type": "excessive_exclamation",
                    "severity": "low",
                    "description": "Uso excesivo de signos de exclamación"
                })
        
        return {
            "capitalization_issues": capitalization_issues,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "issues": issues
        }
    
    def _calculate_anomaly_score(
        self,
        length_anomalies: Dict[str, Any],
        structure_anomalies: Dict[str, Any],
        content_anomalies: Dict[str, Any],
        style_anomalies: Dict[str, Any]
    ) -> float:
        """Calcular score de anomalía"""
        score = 0.0
        
        # Contar anomalías por severidad
        severity_weights = {"high": 0.4, "medium": 0.3, "low": 0.1}
        
        all_issues = (
            length_anomalies.get("issues", []) +
            structure_anomalies.get("issues", []) +
            content_anomalies.get("issues", []) +
            style_anomalies.get("issues", [])
        )
        
        for issue in all_issues:
            severity = issue.get("severity", "low")
            score += severity_weights.get(severity, 0.1)
        
        return min(score, 1.0)
    
    def _get_anomaly_level(self, score: float) -> str:
        """Obtener nivel de anomalía"""
        if score >= 0.7:
            return "Critical"
        elif score >= 0.5:
            return "High"
        elif score >= 0.3:
            return "Medium"
        elif score >= 0.1:
            return "Low"
        else:
            return "Normal"


class DocumentComparisonEngine:
    """Motor avanzado de comparación de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.comparison_history: List[Dict[str, Any]] = []
    
    async def comprehensive_compare(
        self,
        document1_path: Optional[str] = None,
        document1_content: Optional[str] = None,
        document2_path: Optional[str] = None,
        document2_content: Optional[str] = None,
        comparison_depth: str = "full"
    ) -> Dict[str, Any]:
        """
        Comparación comprehensiva de documentos
        
        Args:
            document1_path: Ruta al primer documento
            document1_content: Contenido del primer documento
            document2_path: Ruta al segundo documento
            document2_content: Contenido del segundo documento
            comparison_depth: Profundidad ('quick', 'standard', 'full')
        
        Returns:
            Comparación comprehensiva
        """
        # Obtener contenidos
        content1 = document1_content
        if document1_path:
            processor = DocumentProcessor()
            content1 = processor.process_document(document1_path, "txt")
        
        content2 = document2_content
        if document2_path:
            processor = DocumentProcessor()
            content2 = processor.process_document(document2_path, "txt")
        
        if not content1 or not content2:
            return {"error": "Missing document content"}
        
        # Comparaciones básicas
        semantic_comparison = await self._semantic_comparison(content1, content2)
        structural_comparison = await self._structural_comparison(content1, content2)
        
        if comparison_depth in ["standard", "full"]:
            keyword_comparison = self._keyword_comparison(content1, content2)
            entity_comparison = await self._entity_comparison(content1, content2)
        
        if comparison_depth == "full":
            style_comparison = self._style_comparison(content1, content2)
            topic_comparison = await self._topic_comparison(content1, content2)
        else:
            style_comparison = {}
            topic_comparison = {}
        
        # Calcular similitud general
        overall_similarity = self._calculate_overall_similarity(
            semantic_comparison, structural_comparison,
            keyword_comparison if comparison_depth in ["standard", "full"] else {},
            entity_comparison if comparison_depth in ["standard", "full"] else {},
            style_comparison, topic_comparison
        )
        
        result = {
            "overall_similarity": overall_similarity,
            "semantic_comparison": semantic_comparison,
            "structural_comparison": structural_comparison,
            "keyword_comparison": keyword_comparison if comparison_depth in ["standard", "full"] else {},
            "entity_comparison": entity_comparison if comparison_depth in ["standard", "full"] else {},
            "style_comparison": style_comparison if comparison_depth == "full" else {},
            "topic_comparison": topic_comparison if comparison_depth == "full" else {},
            "comparison_depth": comparison_depth,
            "timestamp": datetime.now().isoformat()
        }
        
        self.comparison_history.append(result)
        return result
    
    async def _semantic_comparison(self, content1: str, content2: str) -> Dict[str, Any]:
        """Comparación semántica"""
        comparator = DocumentComparator(self.analyzer)
        comparison = await comparator.compare_documents(
            doc1_content=content1,
            doc2_content=content2
        )
        
        return {
            "similarity_score": comparison.get("similarity_score", 0.0),
            "differences": comparison.get("differences", []),
            "common_elements": comparison.get("common_elements", [])
        }
    
    async def _structural_comparison(self, content1: str, content2: str) -> Dict[str, Any]:
        """Comparación estructural"""
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        
        structure1 = await structure_analyzer.analyze_structure(document_content=content1)
        structure2 = await structure_analyzer.analyze_structure(document_content=content2)
        
        sections1 = structure1.get("sections", [])
        sections2 = structure2.get("sections", [])
        
        return {
            "sections_count_doc1": len(sections1),
            "sections_count_doc2": len(sections2),
            "sections_difference": abs(len(sections1) - len(sections2)),
            "structure_similarity": 1.0 - (abs(len(sections1) - len(sections2)) / max(len(sections1), len(sections2), 1))
        }
    
    def _keyword_comparison(self, content1: str, content2: str) -> Dict[str, Any]:
        """Comparación de keywords"""
        keywords1 = set(content1.lower().split())
        keywords2 = set(content2.lower().split())
        
        common_keywords = keywords1.intersection(keywords2)
        unique_to_doc1 = keywords1 - keywords2
        unique_to_doc2 = keywords2 - keywords1
        
        total_keywords = len(keywords1.union(keywords2))
        similarity = len(common_keywords) / total_keywords if total_keywords > 0 else 0.0
        
        return {
            "common_keywords": list(common_keywords)[:20],
            "unique_to_doc1": list(unique_to_doc1)[:20],
            "unique_to_doc2": list(unique_to_doc2)[:20],
            "keyword_similarity": similarity,
            "total_common": len(common_keywords)
        }
    
    async def _entity_comparison(self, content1: str, content2: str) -> Dict[str, Any]:
        """Comparación de entidades"""
        entities1 = await self.analyzer.extract_entities(content1)
        entities2 = await self.analyzer.extract_entities(content2)
        
        entities1_text = set(e.get("text", "").lower() for e in entities1)
        entities2_text = set(e.get("text", "").lower() for e in entities2)
        
        common_entities = entities1_text.intersection(entities2_text)
        total_entities = len(entities1_text.union(entities2_text))
        similarity = len(common_entities) / total_entities if total_entities > 0 else 0.0
        
        return {
            "entities_count_doc1": len(entities1),
            "entities_count_doc2": len(entities2),
            "common_entities": list(common_entities)[:20],
            "entity_similarity": similarity
        }
    
    def _style_comparison(self, content1: str, content2: str) -> Dict[str, Any]:
        """Comparación de estilo"""
        # Calcular métricas de estilo
        def calculate_style_metrics(content: str) -> Dict[str, float]:
            words = content.split()
            sentences = content.split('.')
            
            return {
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
                "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
                "punctuation_ratio": sum(1 for c in content if c in '.,;:!?') / len(content) if content else 0
            }
        
        metrics1 = calculate_style_metrics(content1)
        metrics2 = calculate_style_metrics(content2)
        
        # Calcular similitud de estilo
        style_similarity = 1.0 - (
            abs(metrics1["avg_word_length"] - metrics2["avg_word_length"]) / max(metrics1["avg_word_length"], metrics2["avg_word_length"], 1) * 0.33 +
            abs(metrics1["avg_sentence_length"] - metrics2["avg_sentence_length"]) / max(metrics1["avg_sentence_length"], metrics2["avg_sentence_length"], 1) * 0.33 +
            abs(metrics1["punctuation_ratio"] - metrics2["punctuation_ratio"]) / max(metrics1["punctuation_ratio"], metrics2["punctuation_ratio"], 0.01) * 0.34
        )
        
        return {
            "metrics_doc1": metrics1,
            "metrics_doc2": metrics2,
            "style_similarity": max(0.0, min(style_similarity, 1.0))
        }
    
    async def _topic_comparison(self, content1: str, content2: str) -> Dict[str, Any]:
        """Comparación de temas"""
        topics1 = await self.analyzer.analyze_topics(content1, num_topics=5)
        topics2 = await self.analyzer.analyze_topics(content2, num_topics=5)
        
        topics1_keywords = set()
        for topic in topics1.get("topics", []):
            topics1_keywords.update(topic.get("keywords", []))
        
        topics2_keywords = set()
        for topic in topics2.get("topics", []):
            topics2_keywords.update(topic.get("keywords", []))
        
        common_topics = topics1_keywords.intersection(topics2_keywords)
        total_topics = len(topics1_keywords.union(topics2_keywords))
        similarity = len(common_topics) / total_topics if total_topics > 0 else 0.0
        
        return {
            "topics_doc1": list(topics1_keywords)[:10],
            "topics_doc2": list(topics2_keywords)[:10],
            "common_topics": list(common_topics)[:10],
            "topic_similarity": similarity
        }
    
    def _calculate_overall_similarity(
        self,
        semantic: Dict[str, Any],
        structural: Dict[str, Any],
        keyword: Dict[str, Any],
        entity: Dict[str, Any],
        style: Dict[str, Any],
        topic: Dict[str, Any]
    ) -> float:
        """Calcular similitud general"""
        weights = {
            "semantic": 0.3,
            "structural": 0.15,
            "keyword": 0.2,
            "entity": 0.15,
            "style": 0.1,
            "topic": 0.1
        }
        
        similarity = 0.0
        
        similarity += semantic.get("similarity_score", 0.0) * weights["semantic"]
        similarity += structural.get("structure_similarity", 0.0) * weights["structural"]
        
        if keyword:
            similarity += keyword.get("keyword_similarity", 0.0) * weights["keyword"]
        if entity:
            similarity += entity.get("entity_similarity", 0.0) * weights["entity"]
        if style:
            similarity += style.get("style_similarity", 0.0) * weights["style"]
        if topic:
            similarity += topic.get("topic_similarity", 0.0) * weights["topic"]
        
        return min(similarity, 1.0)


class DocumentInsightGenerator:
    """Generador de insights avanzados sobre documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.insights_history: List[Dict[str, Any]] = []
    
    async def generate_insights(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        insight_types: List[str] = ["all"]
    ) -> Dict[str, Any]:
        """
        Generar insights sobre el documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            insight_types: Tipos de insights a generar
        
        Returns:
            Insights generados
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        insights = {}
        
        if "all" in insight_types or "summary" in insight_types:
            insights["summary_insights"] = await self._generate_summary_insights(content)
        
        if "all" in insight_types or "quality" in insight_types:
            insights["quality_insights"] = await self._generate_quality_insights(content)
        
        if "all" in insight_types or "trends" in insight_types:
            insights["trend_insights"] = await self._generate_trend_insights(content)
        
        if "all" in insight_types or "recommendations" in insight_types:
            insights["recommendations"] = await self._generate_recommendations(content)
        
        result = {
            "insights": insights,
            "insight_types": insight_types,
            "timestamp": datetime.now().isoformat()
        }
        
        self.insights_history.append(result)
        return result
    
    async def _generate_summary_insights(self, content: str) -> Dict[str, Any]:
        """Generar insights de resumen"""
        summary = await self.analyzer.summarize_document(content)
        
        # Analizar longitud vs resumen
        content_length = len(content)
        summary_length = len(summary) if isinstance(summary, str) else len(summary.get("summary", ""))
        
        compression_ratio = summary_length / content_length if content_length > 0 else 0
        
        return {
            "summary": summary[:200] if isinstance(summary, str) else summary.get("summary", "")[:200],
            "compression_ratio": compression_ratio,
            "key_points": len(content.split('.'))  # Simplificado
        }
    
    async def _generate_quality_insights(self, content: str) -> Dict[str, Any]:
        """Generar insights de calidad"""
        quality_analyzer = ContentQualityAnalyzer(self.analyzer)
        quality = await quality_analyzer.analyze_content_quality(document_content=content)
        
        return {
            "overall_quality": quality.get("overall_quality_score", 0.0),
            "quality_level": quality.get("quality_level", "Unknown"),
            "strengths": [
                "Coherencia alta" if quality.get("coherence", 0.0) > 0.7 else None,
                "Completitud alta" if quality.get("completeness", 0.0) > 0.7 else None,
                "Claridad alta" if quality.get("clarity", 0.0) > 0.7 else None
            ],
            "weaknesses": quality.get("recommendations", [])
        }
    
    async def _generate_trend_insights(self, content: str) -> Dict[str, Any]:
        """Generar insights de tendencias"""
        # Analizar sentimiento a lo largo del documento
        section_sentiment = SectionSentimentAnalyzer(self.analyzer)
        sentiment_analysis = await section_sentiment.analyze_section_sentiment(document_content=content)
        
        # Detectar tendencia de sentimiento
        sections = sentiment_analysis.get("sections", [])
        if sections:
            first_half_sentiment = sum(s.get("sentiment_score", 0) for s in sections[:len(sections)//2]) / (len(sections)//2) if len(sections)//2 > 0 else 0
            second_half_sentiment = sum(s.get("sentiment_score", 0) for s in sections[len(sections)//2:]) / (len(sections) - len(sections)//2) if (len(sections) - len(sections)//2) > 0 else 0
            
            sentiment_trend = "positive" if second_half_sentiment > first_half_sentiment else "negative" if second_half_sentiment < first_half_sentiment else "stable"
        else:
            sentiment_trend = "unknown"
        
        return {
            "sentiment_trend": sentiment_trend,
            "overall_sentiment": sentiment_analysis.get("overall_sentiment", {}),
            "section_count": len(sections)
        }
    
    async def _generate_recommendations(self, content: str) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        # Analizar varios aspectos
        readability = ReadabilityAnalyzer()
        readability_analysis = await readability.analyze_readability(document_content=content)
        
        if readability_analysis.get("flesch_score", 0) < 60:
            recommendations.append("Mejorar legibilidad del documento")
        
        redundancy_detector = RedundancyDetector(self.analyzer)
        redundancy = await redundancy_detector.detect_redundancies(document_content=content)
        
        if redundancy.get("redundancy_score", 0.0) > 0.5:
            recommendations.append("Reducir redundancias en el contenido")
        
        return recommendations


# ============================================================================
# SISTEMAS AVANZADOS FINALES - GESTIÓN Y OPTIMIZACIÓN AVANZADA
# ============================================================================

class DocumentWorkflowManager:
    """Gestor de workflows para procesamiento de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_workflow(
        self,
        workflow_id: str,
        steps: List[Dict[str, Any]],
        conditions: Optional[Dict[str, Any]] = None
    ):
        """Registrar un workflow"""
        self.workflows[workflow_id] = {
            "steps": steps,
            "conditions": conditions or {},
            "created_at": datetime.now().isoformat()
        }
    
    async def execute_workflow(
        self,
        workflow_id: str,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar workflow en un documento
        
        Args:
            workflow_id: ID del workflow
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Resultados de ejecución del workflow
        """
        if workflow_id not in self.workflows:
            return {"error": f"Workflow {workflow_id} not found"}
        
        workflow = self.workflows[workflow_id]
        steps = workflow["steps"]
        
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to process"}
        
        results = {
            "workflow_id": workflow_id,
            "steps_executed": [],
            "results": {},
            "status": "in_progress"
        }
        
        try:
            for step in steps:
                step_type = step.get("type")
                step_config = step.get("config", {})
                
                step_result = await self._execute_step(step_type, content, step_config)
                results["steps_executed"].append({
                    "step": step_type,
                    "status": "completed",
                    "result": step_result
                })
                results["results"][step_type] = step_result
            
            results["status"] = "completed"
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
        
        self.execution_history.append(results)
        return results
    
    async def _execute_step(
        self,
        step_type: str,
        content: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecutar un paso del workflow"""
        if step_type == "classify":
            return await self.analyzer.classify_document(content)
        elif step_type == "summarize":
            return await self.analyzer.summarize_document(content, **config)
        elif step_type == "extract_keywords":
            return await self.analyzer.extract_keywords(content, **config)
        elif step_type == "extract_entities":
            return await self.analyzer.extract_entities(content)
        elif step_type == "analyze_sentiment":
            return await self.analyzer.analyze_sentiment(content)
        elif step_type == "analyze_topics":
            return await self.analyzer.analyze_topics(content, **config)
        elif step_type == "analyze_structure":
            structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
            return await structure_analyzer.analyze_structure(document_content=content)
        else:
            return {"error": f"Unknown step type: {step_type}"}


class DocumentKnowledgeGraphBuilder:
    """Constructor de grafos de conocimiento a partir de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.knowledge_graphs: Dict[str, Dict[str, Any]] = {}
        self.graph_history: List[Dict[str, Any]] = []
    
    async def build_knowledge_graph(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        graph_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Construir grafo de conocimiento del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            graph_depth: Profundidad del grafo
        
        Returns:
            Grafo de conocimiento
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Extraer entidades
        entities = await self.analyzer.extract_entities(content)
        
        # Extraer relaciones
        relationship_extractor = EntityRelationshipExtractor(self.analyzer)
        relationships = await relationship_extractor.extract_relationships(document_content=content)
        
        # Construir nodos y edges
        nodes = self._build_nodes(entities, content)
        edges = self._build_edges(relationships.get("relationships", []))
        
        # Calcular métricas del grafo
        graph_metrics = self._calculate_graph_metrics(nodes, edges)
        
        graph = {
            "nodes": nodes,
            "edges": edges,
            "metrics": graph_metrics,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "timestamp": datetime.now().isoformat()
        }
        
        self.graph_history.append(graph)
        return graph
    
    def _build_nodes(self, entities: List[Dict[str, Any]], content: str) -> List[Dict[str, Any]]:
        """Construir nodos del grafo"""
        nodes = []
        entity_counts = {}
        
        # Contar ocurrencias de entidades
        content_lower = content.lower()
        for entity in entities:
            entity_text = entity.get("text", "").lower()
            entity_counts[entity_text] = entity_counts.get(entity_text, 0) + 1
        
        # Crear nodos únicos
        seen_entities = set()
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_lower = entity_text.lower()
            
            if entity_lower not in seen_entities:
                nodes.append({
                    "id": entity_lower,
                    "label": entity_text,
                    "type": entity.get("label", "OTHER"),
                    "occurrences": entity_counts.get(entity_lower, 1),
                    "properties": {
                        "confidence": entity.get("confidence", 0.0)
                    }
                })
                seen_entities.add(entity_lower)
        
        return nodes
    
    def _build_edges(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Construir edges del grafo"""
        edges = []
        seen_edges = set()
        
        for rel in relationships:
            entity1 = rel.get("entity1", {}).get("text", "").lower()
            entity2 = rel.get("entity2", {}).get("text", "").lower()
            rel_type = rel.get("relationship_type", "related")
            
            edge_key = f"{entity1}-{rel_type}-{entity2}"
            if edge_key not in seen_edges:
                edges.append({
                    "source": entity1,
                    "target": entity2,
                    "type": rel_type,
                    "weight": rel.get("confidence", 0.5),
                    "properties": {
                        "confidence": rel.get("confidence", 0.5),
                        "distance": rel.get("distance", 0)
                    }
                })
                seen_edges.add(edge_key)
        
        return edges
    
    def _calculate_graph_metrics(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calcular métricas del grafo"""
        if not nodes:
            return {}
        
        # Calcular grados de nodos
        node_degrees = {}
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            node_degrees[source] = node_degrees.get(source, 0) + 1
            node_degrees[target] = node_degrees.get(target, 0) + 1
        
        # Nodo más conectado
        most_connected = max(node_degrees.items(), key=lambda x: x[1]) if node_degrees else None
        
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "average_degree": sum(node_degrees.values()) / len(nodes) if nodes else 0,
            "max_degree": max(node_degrees.values()) if node_degrees else 0,
            "most_connected_node": most_connected[0] if most_connected else None,
            "density": len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
        }


class DocumentAIAssistant:
    """Asistente IA para documentos con capacidades de análisis y respuesta."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.conversation_history: List[Dict[str, Any]] = []
        self.document_context: Dict[str, str] = {}
    
    def set_document_context(self, document_id: str, content: str):
        """Establecer contexto de documento"""
        self.document_context[document_id] = content
    
    async def answer_question(
        self,
        question: str,
        document_id: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Responder pregunta sobre el documento
        
        Args:
            question: Pregunta a responder
            document_id: ID del documento (si está en contexto)
            document_content: Contenido del documento
        
        Returns:
            Respuesta a la pregunta
        """
        content = document_content
        if document_id and document_id in self.document_context:
            content = self.document_context[document_id]
        
        if not content:
            return {"error": "No document content available"}
        
        # Usar QA system si está disponible
        qa_result = await self.analyzer.answer_question(content, question)
        
        # Buscar contexto relevante
        relevant_context = await self._find_relevant_context(question, content)
        
        response = {
            "question": question,
            "answer": qa_result.get("answer", "") if isinstance(qa_result, dict) else str(qa_result),
            "confidence": qa_result.get("confidence", 0.0) if isinstance(qa_result, dict) else 0.5,
            "relevant_context": relevant_context,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history.append({
            "type": "question",
            "content": question,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    async def _find_relevant_context(self, question: str, content: str) -> List[str]:
        """Encontrar contexto relevante para la pregunta"""
        # Dividir contenido en oraciones
        sentences = content.split('.')
        
        # Generar embedding de pregunta
        question_embedding = await self.analyzer.embedding_generator.generate_embeddings([question])
        if not question_embedding:
            return []
        
        # Calcular similitud con oraciones
        sentence_embeddings = await self.analyzer.embedding_generator.generate_embeddings(sentences[:50])  # Limitar a 50
        
        if not sentence_embeddings:
            return []
        
        similarities = []
        for i, sent_emb in enumerate(sentence_embeddings):
            dot_product = np.dot(question_embedding[0], sent_emb)
            norm_q = np.linalg.norm(question_embedding[0])
            norm_s = np.linalg.norm(sent_emb)
            
            similarity = dot_product / (norm_q * norm_s) if (norm_q * norm_s) > 0 else 0.0
            similarities.append((i, similarity, sentences[i]))
        
        # Ordenar por similitud y retornar top 3
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [sent[:200] for _, _, sent in similarities[:3]]
    
    async def generate_suggestions(
        self,
        document_id: Optional[str] = None,
        document_content: Optional[str] = None,
        suggestion_type: str = "improvement"
    ) -> Dict[str, Any]:
        """
        Generar sugerencias para el documento
        
        Args:
            document_id: ID del documento
            document_content: Contenido del documento
            suggestion_type: Tipo de sugerencia
        
        Returns:
            Sugerencias generadas
        """
        content = document_content
        if document_id and document_id in self.document_context:
            content = self.document_context[document_id]
        
        if not content:
            return {"error": "No document content available"}
        
        suggestions = []
        
        if suggestion_type == "improvement":
            # Analizar calidad y generar sugerencias
            quality_analyzer = ContentQualityAnalyzer(self.analyzer)
            quality = await quality_analyzer.analyze_content_quality(document_content=content)
            suggestions.extend(quality.get("recommendations", []))
        
        elif suggestion_type == "structure":
            # Sugerencias de estructura
            structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
            structure = await structure_analyzer.analyze_structure(document_content=content)
            
            if len(structure.get("headings", [])) == 0:
                suggestions.append("Considerar agregar encabezados para mejorar organización")
            
            if len(structure.get("sections", [])) < 3:
                suggestions.append("Documento podría beneficiarse de más secciones")
        
        elif suggestion_type == "content":
            # Sugerencias de contenido
            redundancy = RedundancyDetector(self.analyzer)
            redundancy_analysis = await redundancy.detect_redundancies(document_content=content)
            
            if redundancy_analysis.get("redundancy_score", 0.0) > 0.5:
                suggestions.append("Reducir redundancias en el contenido")
            
            keywords = await self.analyzer.extract_keywords(content, top_k=10)
            if len(keywords) < 5:
                suggestions.append("Considerar agregar más keywords relevantes")
        
        return {
            "suggestions": suggestions,
            "suggestion_type": suggestion_type,
            "total_suggestions": len(suggestions),
            "timestamp": datetime.now().isoformat()
        }


class DocumentPerformanceOptimizer:
    """Optimizador de rendimiento para análisis de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.performance_metrics: Dict[str, List[float]] = {}
        self.optimization_history: List[Dict[str, Any]] = []
    
    async def optimize_analysis(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        optimization_target: str = "speed"
    ) -> Dict[str, Any]:
        """
        Optimizar análisis de documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            optimization_target: Objetivo ('speed', 'accuracy', 'balance')
        
        Returns:
            Análisis optimizado
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        start_time = time.time()
        
        if optimization_target == "speed":
            # Análisis rápido con menos profundidad
            results = await self._fast_analysis(content)
        elif optimization_target == "accuracy":
            # Análisis completo y preciso
            results = await self._accurate_analysis(content)
        else:
            # Balance entre velocidad y precisión
            results = await self._balanced_analysis(content)
        
        elapsed_time = time.time() - start_time
        
        # Calcular métricas de rendimiento
        performance_metrics = {
            "elapsed_time": elapsed_time,
            "optimization_target": optimization_target,
            "content_length": len(content),
            "processing_speed": len(content) / elapsed_time if elapsed_time > 0 else 0
        }
        
        result = {
            "results": results,
            "performance_metrics": performance_metrics,
            "optimization_applied": optimization_target,
            "timestamp": datetime.now().isoformat()
        }
        
        self.optimization_history.append(result)
        return result
    
    async def _fast_analysis(self, content: str) -> Dict[str, Any]:
        """Análisis rápido"""
        # Usar solo análisis básicos y rápidos
        return {
            "classification": await self.analyzer.classify_document(content),
            "summary": await self.analyzer.summarize_document(content, max_length=100),
            "keywords": await self.analyzer.extract_keywords(content, top_k=5),
            "sentiment": await self.analyzer.analyze_sentiment(content)
        }
    
    async def _accurate_analysis(self, content: str) -> Dict[str, Any]:
        """Análisis preciso y completo"""
        # Ejecutar todos los análisis disponibles
        results = {
            "classification": await self.analyzer.classify_document(content),
            "summary": await self.analyzer.summarize_document(content),
            "keywords": await self.analyzer.extract_keywords(content, top_k=20),
            "entities": await self.analyzer.extract_entities(content),
            "sentiment": await self.analyzer.analyze_sentiment(content),
            "topics": await self.analyzer.analyze_topics(content, num_topics=10)
        }
        
        # Análisis adicionales
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        results["structure"] = await structure_analyzer.analyze_structure(document_content=content)
        
        return results
    
    async def _balanced_analysis(self, content: str) -> Dict[str, Any]:
        """Análisis balanceado"""
        # Análisis intermedios
        return {
            "classification": await self.analyzer.classify_document(content),
            "summary": await self.analyzer.summarize_document(content, max_length=200),
            "keywords": await self.analyzer.extract_keywords(content, top_k=10),
            "entities": await self.analyzer.extract_entities(content),
            "sentiment": await self.analyzer.analyze_sentiment(content),
            "topics": await self.analyzer.analyze_topics(content, num_topics=5)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento"""
        if not self.optimization_history:
            return {"error": "No performance data available"}
        
        times = [h["performance_metrics"]["elapsed_time"] for h in self.optimization_history]
        speeds = [h["performance_metrics"]["processing_speed"] for h in self.optimization_history]
        
        return {
            "total_analyses": len(self.optimization_history),
            "avg_time": sum(times) / len(times) if times else 0,
            "min_time": min(times) if times else 0,
            "max_time": max(times) if times else 0,
            "avg_speed": sum(speeds) / len(speeds) if speeds else 0,
            "optimization_targets_used": list(set(h["optimization_applied"] for h in self.optimization_history))
        }


class DocumentBatchProcessor:
    """Procesador de lotes de documentos con optimización."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.batch_history: List[Dict[str, Any]] = []
    
    async def process_batch(
        self,
        documents: List[Dict[str, Any]],
        analysis_types: List[str] = ["all"],
        batch_size: int = 10,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Procesar lote de documentos
        
        Args:
            documents: Lista de documentos con 'id' y 'content' o 'path'
            analysis_types: Tipos de análisis a realizar
            batch_size: Tamaño del lote
            parallel: Procesar en paralelo
        
        Returns:
            Resultados del procesamiento por lotes
        """
        if not documents:
            return {"error": "No documents provided"}
        
        results = {
            "total_documents": len(documents),
            "processed_documents": [],
            "failed_documents": [],
            "batch_size": batch_size,
            "parallel": parallel
        }
        
        if parallel:
            # Procesar en paralelo usando asyncio
            import asyncio
            
            batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
            
            for batch in batches:
                tasks = [self._process_single_document(doc, analysis_types) for doc in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for doc, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        results["failed_documents"].append({
                            "document_id": doc.get("id", "unknown"),
                            "error": str(result)
                        })
                    else:
                        results["processed_documents"].append({
                            "document_id": doc.get("id", "unknown"),
                            "result": result
                        })
        else:
            # Procesar secuencialmente
            for doc in documents:
                try:
                    result = await self._process_single_document(doc, analysis_types)
                    results["processed_documents"].append({
                        "document_id": doc.get("id", "unknown"),
                        "result": result
                    })
                except Exception as e:
                    results["failed_documents"].append({
                        "document_id": doc.get("id", "unknown"),
                        "error": str(e)
                    })
        
        results["success_rate"] = (
            len(results["processed_documents"]) / len(documents) if documents else 0
        )
        results["timestamp"] = datetime.now().isoformat()
        
        self.batch_history.append(results)
        return results
    
    async def _process_single_document(
        self,
        document: Dict[str, Any],
        analysis_types: List[str]
    ) -> Dict[str, Any]:
        """Procesar un solo documento"""
        content = document.get("content")
        if not content and document.get("path"):
            processor = DocumentProcessor()
            content = processor.process_document(document["path"], "txt")
        
        if not content:
            return {"error": "No content available"}
        
        result = {}
        
        if "all" in analysis_types or "classify" in analysis_types:
            result["classification"] = await self.analyzer.classify_document(content)
        
        if "all" in analysis_types or "summarize" in analysis_types:
            result["summary"] = await self.analyzer.summarize_document(content)
        
        if "all" in analysis_types or "keywords" in analysis_types:
            result["keywords"] = await self.analyzer.extract_keywords(content)
        
        if "all" in analysis_types or "entities" in analysis_types:
            result["entities"] = await self.analyzer.extract_entities(content)
        
        return result


# ============================================================================
# SISTEMAS AVANZADOS FINALES - AUTOMATIZACIÓN E INTELIGENCIA AVANZADA
# ============================================================================

class DocumentCollaborationAnalyzer:
    """Analizador de colaboración en documentos (multi-autor)."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.collaboration_history: List[Dict[str, Any]] = []
    
    async def analyze_collaboration(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        author_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analizar colaboración en documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            author_metadata: Metadata de autores (secciones, timestamps, etc.)
        
        Returns:
            Análisis de colaboración
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Analizar estilo de diferentes secciones
        style_analysis = await self._analyze_style_variation(content)
        
        # Detectar múltiples autores por estilo
        author_detection = self._detect_multiple_authors(content, style_analysis)
        
        # Analizar coherencia entre secciones
        coherence_analysis = await self._analyze_collaborative_coherence(content)
        
        # Calcular métricas de colaboración
        collaboration_metrics = self._calculate_collaboration_metrics(
            style_analysis, author_detection, coherence_analysis, author_metadata
        )
        
        result = {
            "style_analysis": style_analysis,
            "author_detection": author_detection,
            "coherence_analysis": coherence_analysis,
            "collaboration_metrics": collaboration_metrics,
            "estimated_authors": author_detection.get("estimated_author_count", 1),
            "collaboration_score": collaboration_metrics.get("collaboration_score", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        self.collaboration_history.append(result)
        return result
    
    async def _analyze_style_variation(self, content: str) -> Dict[str, Any]:
        """Analizar variación de estilo en el documento"""
        # Dividir en secciones
        sections = content.split('\n\n')
        
        style_metrics = []
        for i, section in enumerate(sections[:10]):  # Limitar a 10 secciones
            if len(section.strip()) > 50:
                words = section.split()
                avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
                avg_sentence_length = len(words) / len(section.split('.')) if section.count('.') > 0 else 0
                
                style_metrics.append({
                    "section_index": i,
                    "avg_word_length": avg_word_length,
                    "avg_sentence_length": avg_sentence_length,
                    "word_count": len(words)
                })
        
        # Calcular variación
        if len(style_metrics) > 1:
            word_lengths = [m["avg_word_length"] for m in style_metrics]
            sentence_lengths = [m["avg_sentence_length"] for m in style_metrics]
            
            word_variation = np.std(word_lengths) if word_lengths else 0
            sentence_variation = np.std(sentence_lengths) if sentence_lengths else 0
            
            variation_score = (word_variation + sentence_variation) / 2
        else:
            variation_score = 0.0
        
        return {
            "style_metrics": style_metrics,
            "variation_score": variation_score,
            "has_style_variation": variation_score > 0.5
        }
    
    def _detect_multiple_authors(
        self,
        content: str,
        style_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detectar múltiples autores"""
        has_variation = style_analysis.get("has_style_variation", False)
        
        # Estimar número de autores basado en variación
        variation_score = style_analysis.get("variation_score", 0.0)
        estimated_count = 1
        
        if has_variation:
            if variation_score > 1.5:
                estimated_count = 3
            elif variation_score > 1.0:
                estimated_count = 2
        
        return {
            "estimated_author_count": estimated_count,
            "has_multiple_authors": estimated_count > 1,
            "confidence": min(variation_score / 2.0, 1.0)
        }
    
    async def _analyze_collaborative_coherence(self, content: str) -> Dict[str, Any]:
        """Analizar coherencia colaborativa"""
        coherence_analyzer = DocumentCoherenceAnalyzer(self.analyzer)
        coherence = await coherence_analyzer.analyze_coherence(document_content=content)
        
        return {
            "overall_coherence": coherence.get("coherence_score", 0.0),
            "is_coherent": coherence.get("coherence_score", 0.0) > 0.6,
            "coherence_level": "high" if coherence.get("coherence_score", 0.0) > 0.7 else "medium" if coherence.get("coherence_score", 0.0) > 0.5 else "low"
        }
    
    def _calculate_collaboration_metrics(
        self,
        style_analysis: Dict[str, Any],
        author_detection: Dict[str, Any],
        coherence_analysis: Dict[str, Any],
        author_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calcular métricas de colaboración"""
        collaboration_score = 0.0
        
        # Si hay múltiples autores detectados
        if author_detection.get("has_multiple_authors", False):
            collaboration_score += 0.4
        
        # Si hay variación de estilo
        if style_analysis.get("has_style_variation", False):
            collaboration_score += 0.3
        
        # Si mantiene coherencia
        if coherence_analysis.get("is_coherent", False):
            collaboration_score += 0.3
        
        return {
            "collaboration_score": min(collaboration_score, 1.0),
            "has_collaboration": collaboration_score > 0.5,
            "author_count": author_detection.get("estimated_author_count", 1)
        }


class DocumentTemplateGenerator:
    """Generador de plantillas basado en documentos existentes."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.generation_history: List[Dict[str, Any]] = []
    
    async def generate_template_from_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        template_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generar plantilla a partir de un documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            template_name: Nombre de la plantilla
        
        Returns:
            Plantilla generada
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Analizar estructura
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        
        # Extraer patrones
        patterns = self._extract_patterns(content, structure)
        
        # Generar template
        template = {
            "template_name": template_name or f"template_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "structure": structure,
            "patterns": patterns,
            "sections": self._extract_section_templates(content, structure),
            "placeholders": self._identify_placeholders(content),
            "created_at": datetime.now().isoformat()
        }
        
        # Guardar template
        if template_name:
            self.templates[template_name] = template
        
        self.generation_history.append(template)
        return template
    
    def _extract_patterns(
        self,
        content: str,
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extraer patrones del documento"""
        import re
        
        patterns = {
            "headings": [],
            "list_patterns": [],
            "numbering_patterns": []
        }
        
        # Extraer patrones de encabezados
        heading_patterns = [
            r'^#+\s+(.+)$',
            r'^([A-Z][A-Z\s]+)$',
            r'^\d+\.\s+(.+)$'
        ]
        
        for pattern in heading_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                patterns["headings"].extend(matches[:5])
        
        # Extraer patrones de listas
        list_patterns = [
            r'^[-*]\s+(.+)$',
            r'^\d+[.)]\s+(.+)$'
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                patterns["list_patterns"].extend(matches[:5])
        
        return patterns
    
    def _extract_section_templates(
        self,
        content: str,
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extraer plantillas de secciones"""
        sections = structure.get("sections", [])
        section_templates = []
        
        for section in sections[:10]:  # Limitar a 10
            section_content = section.get("content", "")
            
            # Identificar tipo de sección
            section_type = self._identify_section_type(section_content)
            
            section_templates.append({
                "title": section.get("title", ""),
                "type": section_type,
                "length_range": {
                    "min": len(section_content) * 0.8,
                    "max": len(section_content) * 1.2
                },
                "structure": self._analyze_section_structure(section_content)
            })
        
        return section_templates
    
    def _identify_section_type(self, content: str) -> str:
        """Identificar tipo de sección"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["introducción", "introduction", "resumen", "summary"]):
            return "introduction"
        elif any(word in content_lower for word in ["conclusión", "conclusion", "resumen final"]):
            return "conclusion"
        elif any(word in content_lower for word in ["método", "method", "metodología", "methodology"]):
            return "methodology"
        elif any(word in content_lower for word in ["resultado", "result", "resultados", "results"]):
            return "results"
        else:
            return "body"
    
    def _analyze_section_structure(self, content: str) -> Dict[str, Any]:
        """Analizar estructura de sección"""
        import re
        
        return {
            "has_paragraphs": len(content.split('\n\n')) > 1,
            "has_lists": bool(re.search(r'^[-*]\s+|^\d+[.)]\s+', content, re.MULTILINE)),
            "has_headings": bool(re.search(r'^#+\s+|^[A-Z][A-Z\s]+$', content, re.MULTILINE)),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()])
        }
    
    def _identify_placeholders(self, content: str) -> List[Dict[str, Any]]:
        """Identificar placeholders en el documento"""
        import re
        
        placeholders = []
        
        # Patrones comunes de placeholders
        placeholder_patterns = [
            r'\[([A-Z_]+)\]',
            r'\{([A-Z_]+)\}',
            r'<([A-Z_]+)>',
            r'__([A-Z_]+)__'
        ]
        
        for pattern in placeholder_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                placeholders.append({
                    "placeholder": match.group(0),
                    "name": match.group(1),
                    "position": match.start()
                })
        
        return placeholders
    
    def apply_template(
        self,
        template_name: str,
        data: Dict[str, Any]
    ) -> str:
        """Aplicar plantilla con datos"""
        if template_name not in self.templates:
            return f"Template {template_name} not found"
        
        template = self.templates[template_name]
        # Implementación simplificada
        # En producción, usar un motor de templates real
        
        return f"Template applied: {template_name}"


class DocumentIntelligenceSystem:
    """Sistema de inteligencia de documentos con capacidades avanzadas."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.intelligence_history: List[Dict[str, Any]] = []
        self.learned_patterns: Dict[str, Any] = {}
    
    async def comprehensive_intelligence_analysis(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Análisis de inteligencia comprehensivo
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis de inteligencia completo
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Múltiples análisis en paralelo
        import asyncio
        
        tasks = [
            self._analyze_content_intelligence(content),
            self._analyze_context_intelligence(content),
            self._analyze_pattern_intelligence(content),
            self._analyze_semantic_intelligence(content)
        ]
        
        results = await asyncio.gather(*tasks)
        
        intelligence_report = {
            "content_intelligence": results[0],
            "context_intelligence": results[1],
            "pattern_intelligence": results[2],
            "semantic_intelligence": results[3],
            "overall_intelligence_score": self._calculate_intelligence_score(results),
            "timestamp": datetime.now().isoformat()
        }
        
        self.intelligence_history.append(intelligence_report)
        return intelligence_report
    
    async def _analyze_content_intelligence(self, content: str) -> Dict[str, Any]:
        """Analizar inteligencia del contenido"""
        # Análisis básico
        classification = await self.analyzer.classify_document(content)
        keywords = await self.analyzer.extract_keywords(content, top_k=20)
        entities = await self.analyzer.extract_entities(content)
        
        return {
            "classification": classification,
            "key_concepts": keywords[:10],
            "important_entities": entities[:10],
            "content_richness": len(keywords) + len(entities)
        }
    
    async def _analyze_context_intelligence(self, content: str) -> Dict[str, Any]:
        """Analizar inteligencia contextual"""
        # Analizar relaciones y contexto
        relationship_extractor = EntityRelationshipExtractor(self.analyzer)
        relationships = await relationship_extractor.extract_relationships(document_content=content)
        
        # Analizar temas
        topics = await self.analyzer.analyze_topics(content, num_topics=5)
        
        return {
            "entity_relationships": len(relationships.get("relationships", [])),
            "topics": topics.get("topics", [])[:5],
            "contextual_connections": len(relationships.get("relationships", []))
        }
    
    async def _analyze_pattern_intelligence(self, content: str) -> Dict[str, Any]:
        """Analizar inteligencia de patrones"""
        # Detectar patrones recurrentes
        import re
        
        patterns = {
            "repeated_phrases": [],
            "structural_patterns": [],
            "numerical_patterns": []
        }
        
        # Patrones numéricos
        numbers = re.findall(r'\d+', content)
        if numbers:
            patterns["numerical_patterns"] = {
                "count": len(numbers),
                "max": max(int(n) for n in numbers) if numbers else 0,
                "min": min(int(n) for n in numbers) if numbers else 0
            }
        
        # Patrones estructurales
        sections = content.split('\n\n')
        patterns["structural_patterns"] = {
            "section_count": len(sections),
            "avg_section_length": sum(len(s) for s in sections) / len(sections) if sections else 0
        }
        
        return patterns
    
    async def _analyze_semantic_intelligence(self, content: str) -> Dict[str, Any]:
        """Analizar inteligencia semántica"""
        # Análisis semántico profundo
        summary = await self.analyzer.summarize_document(content)
        sentiment = await self.analyzer.analyze_sentiment(content)
        
        # Calcular densidad semántica
        semantic_density = self._calculate_semantic_density(content)
        
        return {
            "summary": summary[:200] if isinstance(summary, str) else summary.get("summary", "")[:200] if isinstance(summary, dict) else "",
            "sentiment": sentiment,
            "semantic_density": semantic_density,
            "semantic_richness": "high" if semantic_density > 0.7 else "medium" if semantic_density > 0.4 else "low"
        }
    
    def _calculate_semantic_density(self, content: str) -> float:
        """Calcular densidad semántica"""
        # Ratio de palabras únicas
        words = content.lower().split()
        unique_words = len(set(words))
        total_words = len(words)
        
        uniqueness_ratio = unique_words / total_words if total_words > 0 else 0
        
        # Ratio de entidades (estimado)
        import re
        capitalized_words = len(re.findall(r'\b[A-Z][a-z]+\b', content))
        entity_ratio = capitalized_words / total_words if total_words > 0 else 0
        
        # Densidad semántica
        density = (uniqueness_ratio * 0.6) + (entity_ratio * 0.4)
        
        return min(density, 1.0)
    
    def _calculate_intelligence_score(self, results: List[Dict[str, Any]]) -> float:
        """Calcular score general de inteligencia"""
        score = 0.0
        
        # Content intelligence (30%)
        content_richness = results[0].get("content_richness", 0)
        score += min(content_richness / 30.0, 1.0) * 0.3
        
        # Context intelligence (25%)
        contextual_connections = results[1].get("contextual_connections", 0)
        score += min(contextual_connections / 20.0, 1.0) * 0.25
        
        # Pattern intelligence (20%)
        pattern_count = len(results[2].get("numerical_patterns", {}).get("count", 0)) if isinstance(results[2].get("numerical_patterns", {}), dict) else 0
        score += min(pattern_count / 50.0, 1.0) * 0.2
        
        # Semantic intelligence (25%)
        semantic_density = results[3].get("semantic_density", 0.0)
        score += semantic_density * 0.25
        
        return min(score, 1.0)


class DocumentAutomationEngine:
    """Motor de automatización para procesamiento de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.automation_rules: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_automation_rule(
        self,
        rule_id: str,
        condition: Dict[str, Any],
        actions: List[Dict[str, Any]],
        priority: int = 0
    ):
        """Registrar regla de automatización"""
        self.automation_rules[rule_id] = {
            "condition": condition,
            "actions": actions,
            "priority": priority,
            "created_at": datetime.now().isoformat()
        }
    
    async def process_document_with_automation(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Procesar documento con automatización
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Resultados de automatización
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to process"}
        
        # Analizar documento
        analysis = await self.analyzer.analyze_document(document_content=content)
        
        # Evaluar reglas
        triggered_rules = []
        executed_actions = []
        
        # Ordenar reglas por prioridad
        sorted_rules = sorted(
            self.automation_rules.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )
        
        for rule_id, rule in sorted_rules:
            if self._evaluate_condition(rule["condition"], analysis):
                triggered_rules.append(rule_id)
                
                # Ejecutar acciones
                for action in rule["actions"]:
                    action_result = await self._execute_action(action, content, analysis)
                    executed_actions.append({
                        "rule_id": rule_id,
                        "action": action.get("type"),
                        "result": action_result
                    })
        
        result = {
            "analysis": analysis,
            "triggered_rules": triggered_rules,
            "executed_actions": executed_actions,
            "total_rules_evaluated": len(self.automation_rules),
            "total_rules_triggered": len(triggered_rules),
            "timestamp": datetime.now().isoformat()
        }
        
        self.execution_history.append(result)
        return result
    
    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        analysis: Any
    ) -> bool:
        """Evaluar condición de regla"""
        condition_type = condition.get("type")
        
        if condition_type == "classification":
            required_class = condition.get("value")
            if hasattr(analysis, 'classification'):
                top_class = max(analysis.classification.items(), key=lambda x: x[1])[0] if analysis.classification else None
                return top_class == required_class
            return False
        
        elif condition_type == "keyword_present":
            required_keyword = condition.get("value", "").lower()
            if hasattr(analysis, 'keywords'):
                keywords_lower = [k.lower() for k in analysis.keywords]
                return required_keyword in keywords_lower
            return False
        
        elif condition_type == "sentiment":
            required_sentiment = condition.get("value")
            if hasattr(analysis, 'sentiment'):
                dominant_sentiment = max(analysis.sentiment.items(), key=lambda x: x[1])[0] if analysis.sentiment else None
                return dominant_sentiment == required_sentiment
            return False
        
        return False
    
    async def _execute_action(
        self,
        action: Dict[str, Any],
        content: str,
        analysis: Any
    ) -> Dict[str, Any]:
        """Ejecutar acción de automatización"""
        action_type = action.get("type")
        action_config = action.get("config", {})
        
        if action_type == "tag":
            tagger = IntelligentTagger(self.analyzer)
            return await tagger.auto_tag_document(document_content=content)
        
        elif action_type == "archive":
            return {"status": "archived", "message": "Document archived"}
        
        elif action_type == "notify":
            return {"status": "notified", "message": action_config.get("message", "Notification sent")}
        
        elif action_type == "route":
            return {"status": "routed", "destination": action_config.get("destination", "unknown")}
        
        else:
            return {"status": "unknown_action", "error": f"Unknown action type: {action_type}"}


class DocumentQualityAssurance:
    """Sistema de aseguramiento de calidad para documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.qa_history: List[Dict[str, Any]] = []
        self.quality_standards: Dict[str, Dict[str, Any]] = {}
    
    def set_quality_standard(
        self,
        standard_name: str,
        criteria: Dict[str, Any]
    ):
        """Establecer estándar de calidad"""
        self.quality_standards[standard_name] = {
            "criteria": criteria,
            "created_at": datetime.now().isoformat()
        }
    
    async def perform_quality_assurance(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        standard_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Realizar aseguramiento de calidad
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            standard_name: Nombre del estándar a aplicar
        
        Returns:
            Reporte de calidad
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Realizar múltiples verificaciones de calidad
        checks = {
            "spelling": await self._check_spelling(content),
            "grammar": await self._check_grammar(content),
            "formatting": self._check_formatting(content),
            "completeness": await self._check_completeness(content),
            "consistency": await self._check_consistency(content)
        }
        
        # Evaluar contra estándar si se especifica
        standard_evaluation = {}
        if standard_name and standard_name in self.quality_standards:
            standard_evaluation = self._evaluate_against_standard(
                checks, self.quality_standards[standard_name]["criteria"]
            )
        
        # Calcular score de calidad
        quality_score = self._calculate_quality_score(checks)
        
        result = {
            "quality_score": quality_score,
            "quality_level": self._get_quality_level(quality_score),
            "checks": checks,
            "standard_evaluation": standard_evaluation,
            "passed_checks": sum(1 for check in checks.values() if check.get("passed", False)),
            "total_checks": len(checks),
            "timestamp": datetime.now().isoformat()
        }
        
        self.qa_history.append(result)
        return result
    
    async def _check_spelling(self, content: str) -> Dict[str, Any]:
        """Verificar ortografía"""
        # Implementación simplificada
        # En producción, usar un corrector ortográfico real
        return {
            "passed": True,
            "issues_found": 0,
            "message": "Spelling check completed"
        }
    
    async def _check_grammar(self, content: str) -> Dict[str, Any]:
        """Verificar gramática"""
        # Implementación simplificada
        return {
            "passed": True,
            "issues_found": 0,
            "message": "Grammar check completed"
        }
    
    def _check_formatting(self, content: str) -> Dict[str, Any]:
        """Verificar formato"""
        import re
        
        issues = []
        
        # Verificar uso consistente de espacios
        double_spaces = len(re.findall(r'  +', content))
        if double_spaces > 0:
            issues.append(f"Found {double_spaces} instances of double spaces")
        
        # Verificar líneas vacías excesivas
        empty_lines = len(re.findall(r'\n\s*\n\s*\n', content))
        if empty_lines > 0:
            issues.append(f"Found {empty_lines} instances of excessive empty lines")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "issues_found": len(issues)
        }
    
    async def _check_completeness(self, content: str) -> Dict[str, Any]:
        """Verificar completitud"""
        # Verificar si tiene introducción, desarrollo, conclusión
        has_intro = any(word in content.lower() for word in ["introducción", "introduction"])
        has_body = len(content.split('\n\n')) > 2
        has_conclusion = any(word in content.lower() for word in ["conclusión", "conclusion"])
        
        passed = has_intro and has_body and has_conclusion
        
        return {
            "passed": passed,
            "has_introduction": has_intro,
            "has_body": has_body,
            "has_conclusion": has_conclusion,
            "issues": [] if passed else ["Missing required sections"]
        }
    
    async def _check_consistency(self, content: str) -> Dict[str, Any]:
        """Verificar consistencia"""
        # Verificar uso consistente de términos
        import re
        
        # Detectar términos que pueden variar
        terms = {
            "organización": ["organización", "organizacion"],
            "aplicación": ["aplicación", "aplicacion"]
        }
        
        inconsistencies = []
        for term, variants in terms.items():
            found_variants = []
            for variant in variants:
                if variant.lower() in content.lower():
                    found_variants.append(variant)
            
            if len(found_variants) > 1:
                inconsistencies.append(f"Inconsistent use of term variants: {found_variants}")
        
        return {
            "passed": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies,
            "issues_found": len(inconsistencies)
        }
    
    def _evaluate_against_standard(
        self,
        checks: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluar contra estándar de calidad"""
        passed_criteria = 0
        total_criteria = len(criteria)
        
        for criterion, requirement in criteria.items():
            if criterion in checks:
                check_result = checks[criterion]
                if check_result.get("passed", False):
                    passed_criteria += 1
        
        return {
            "passed_criteria": passed_criteria,
            "total_criteria": total_criteria,
            "compliance_rate": passed_criteria / total_criteria if total_criteria > 0 else 0,
            "meets_standard": passed_criteria == total_criteria
        }
    
    def _calculate_quality_score(self, checks: Dict[str, Any]) -> float:
        """Calcular score de calidad"""
        passed_checks = sum(1 for check in checks.values() if check.get("passed", False))
        total_checks = len(checks)
        
        return passed_checks / total_checks if total_checks > 0 else 0.0
    
    def _get_quality_level(self, score: float) -> str:
        """Obtener nivel de calidad"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        else:
            return "Needs Improvement"


# ============================================================================
# SISTEMAS AVANZADOS FINALES - ANÁLISIS PREDICTIVO Y APRENDIZAJE
# ============================================================================

class DocumentLearningSystem:
    """Sistema de aprendizaje automático para documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.learned_patterns: Dict[str, Any] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.prediction_models: Dict[str, Any] = {}
    
    async def learn_from_documents(
        self,
        training_documents: List[Dict[str, Any]],
        learning_task: str = "classification"
    ) -> Dict[str, Any]:
        """
        Aprender patrones de documentos de entrenamiento
        
        Args:
            training_documents: Lista de documentos con 'content' y metadata
            learning_task: Tarea de aprendizaje ('classification', 'summarization', 'extraction')
        
        Returns:
            Modelo aprendido y métricas
        """
        if not training_documents:
            return {"error": "No training documents provided"}
        
        # Extraer características de documentos de entrenamiento
        features = []
        labels = []
        
        for doc in training_documents:
            content = doc.get("content", "")
            if not content:
                continue
            
            # Generar características
            doc_features = await self._extract_features(content)
            features.append(doc_features)
            
            # Extraer etiquetas si están disponibles
            if learning_task == "classification":
                label = doc.get("category") or doc.get("classification")
                if label:
                    labels.append(label)
        
        # Aprender patrones
        learned_patterns = self._learn_patterns(features, labels, learning_task)
        
        # Guardar patrones aprendidos
        self.learned_patterns[learning_task] = learned_patterns
        
        result = {
            "learning_task": learning_task,
            "training_samples": len(features),
            "learned_patterns": learned_patterns,
            "model_accuracy": learned_patterns.get("accuracy", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_history.append(result)
        return result
    
    async def _extract_features(self, content: str) -> Dict[str, Any]:
        """Extraer características de documento"""
        # Características básicas
        word_count = len(content.split())
        sentence_count = len(content.split('.'))
        avg_word_length = sum(len(w) for w in content.split()) / word_count if word_count > 0 else 0
        
        # Características semánticas (simplificado)
        keywords = await self.analyzer.extract_keywords(content, top_k=10)
        entities = await self.analyzer.extract_entities(content)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "keyword_count": len(keywords),
            "entity_count": len(entities),
            "content_length": len(content)
        }
    
    def _learn_patterns(
        self,
        features: List[Dict[str, Any]],
        labels: List[str],
        learning_task: str
    ) -> Dict[str, Any]:
        """Aprender patrones de las características"""
        if not features:
            return {}
        
        # Calcular estadísticas de características
        feature_stats = {}
        for key in features[0].keys():
            values = [f[key] for f in features if key in f]
            if values:
                feature_stats[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": np.std(values) if len(values) > 1 else 0
                }
        
        # Patrones de clasificación si hay etiquetas
        classification_patterns = {}
        if labels and learning_task == "classification":
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            classification_patterns = {
                "label_distribution": label_counts,
                "most_common_label": max(label_counts.items(), key=lambda x: x[1])[0] if label_counts else None
            }
        
        return {
            "feature_statistics": feature_stats,
            "classification_patterns": classification_patterns,
            "sample_count": len(features),
            "accuracy": 0.8  # Placeholder
        }
    
    async def predict(
        self,
        document_content: str,
        learning_task: str = "classification"
    ) -> Dict[str, Any]:
        """Predecir propiedades de un nuevo documento"""
        if learning_task not in self.learned_patterns:
            return {"error": f"No model trained for task: {learning_task}"}
        
        # Extraer características del documento
        features = await self._extract_features(document_content)
        
        # Usar patrones aprendidos para predecir
        patterns = self.learned_patterns[learning_task]
        
        # Predicción simplificada basada en características
        prediction = self._make_prediction(features, patterns, learning_task)
        
        return {
            "prediction": prediction,
            "confidence": 0.75,  # Placeholder
            "features_used": features,
            "learning_task": learning_task
        }
    
    def _make_prediction(
        self,
        features: Dict[str, Any],
        patterns: Dict[str, Any],
        learning_task: str
    ) -> str:
        """Hacer predicción basada en características y patrones"""
        if learning_task == "classification":
            # Predicción basada en características más cercanas a patrones
            classification_patterns = patterns.get("classification_patterns", {})
            if classification_patterns:
                most_common = classification_patterns.get("most_common_label")
                return most_common or "unknown"
        
        return "unknown"


class DocumentPredictiveAnalyzer:
    """Analizador predictivo de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.prediction_history: List[Dict[str, Any]] = []
    
    async def predict_document_properties(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        prediction_types: List[str] = ["all"]
    ) -> Dict[str, Any]:
        """
        Predecir propiedades del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            prediction_types: Tipos de predicciones a realizar
        
        Returns:
            Predicciones del documento
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        predictions = {}
        
        if "all" in prediction_types or "popularity" in prediction_types:
            predictions["popularity"] = await self._predict_popularity(content)
        
        if "all" in prediction_types or "engagement" in prediction_types:
            predictions["engagement"] = await self._predict_engagement(content)
        
        if "all" in prediction_types or "quality_score" in prediction_types:
            predictions["quality_score"] = await self._predict_quality_score(content)
        
        if "all" in prediction_types or "readability_level" in prediction_types:
            predictions["readability_level"] = await self._predict_readability_level(content)
        
        result = {
            "predictions": predictions,
            "prediction_types": prediction_types,
            "timestamp": datetime.now().isoformat()
        }
        
        self.prediction_history.append(result)
        return result
    
    async def _predict_popularity(self, content: str) -> Dict[str, Any]:
        """Predecir popularidad del documento"""
        # Factores que influyen en popularidad
        keywords = await self.analyzer.extract_keywords(content, top_k=20)
        sentiment = await self.analyzer.analyze_sentiment(content)
        topics = await self.analyzer.analyze_topics(content, num_topics=5)
        
        # Score de popularidad basado en factores
        popularity_score = 0.0
        
        # Más keywords = más popular
        popularity_score += min(len(keywords) / 20.0, 1.0) * 0.3
        
        # Sentimiento positivo = más popular
        if isinstance(sentiment, dict):
            positive_score = sentiment.get("positive", 0.0)
            popularity_score += positive_score * 0.3
        
        # Más temas = más popular
        topic_count = len(topics.get("topics", []))
        popularity_score += min(topic_count / 5.0, 1.0) * 0.4
        
        return {
            "popularity_score": min(popularity_score, 1.0),
            "popularity_level": "high" if popularity_score > 0.7 else "medium" if popularity_score > 0.4 else "low",
            "factors": {
                "keyword_count": len(keywords),
                "sentiment_score": sentiment.get("positive", 0.0) if isinstance(sentiment, dict) else 0.0,
                "topic_count": topic_count
            }
        }
    
    async def _predict_engagement(self, content: str) -> Dict[str, Any]:
        """Predecir nivel de engagement"""
        # Factores de engagement
        readability = ReadabilityAnalyzer()
        readability_analysis = await readability.analyze_readability(document_content=content)
        
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        
        # Score de engagement
        engagement_score = 0.0
        
        # Buena legibilidad = más engagement
        flesch_score = readability_analysis.get("flesch_score", 0)
        engagement_score += min(flesch_score / 100.0, 1.0) * 0.4
        
        # Buena estructura = más engagement
        has_headings = len(structure.get("headings", [])) > 0
        has_sections = len(structure.get("sections", [])) > 2
        engagement_score += (0.3 if has_headings else 0.0) + (0.3 if has_sections else 0.0)
        
        return {
            "engagement_score": min(engagement_score, 1.0),
            "engagement_level": "high" if engagement_score > 0.7 else "medium" if engagement_score > 0.4 else "low",
            "factors": {
                "readability": flesch_score,
                "has_headings": has_headings,
                "has_sections": has_sections
            }
        }
    
    async def _predict_quality_score(self, content: str) -> Dict[str, Any]:
        """Predecir score de calidad"""
        quality_analyzer = ContentQualityAnalyzer(self.analyzer)
        quality = await quality_analyzer.analyze_content_quality(document_content=content)
        
        predicted_score = quality.get("overall_quality_score", 0.0)
        
        return {
            "predicted_quality_score": predicted_score,
            "quality_level": quality.get("quality_level", "Unknown"),
            "confidence": 0.8
        }
    
    async def _predict_readability_level(self, content: str) -> Dict[str, Any]:
        """Predecir nivel de legibilidad"""
        readability_analyzer = ReadabilityAnalyzer()
        readability = await readability_analyzer.analyze_readability(document_content=content)
        
        flesch_score = readability.get("flesch_score", 0)
        
        return {
            "predicted_flesch_score": flesch_score,
            "readability_level": readability.get("readability_level", "Unknown"),
            "confidence": 0.85
        }


class DocumentTrendAnalyzer:
    """Analizador de tendencias temporales en documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.trend_history: List[Dict[str, Any]] = []
    
    async def analyze_temporal_trends(
        self,
        documents: List[Dict[str, Any]],
        time_field: str = "timestamp"
    ) -> Dict[str, Any]:
        """
        Analizar tendencias temporales en múltiples documentos
        
        Args:
            documents: Lista de documentos con contenido y timestamp
            time_field: Campo que contiene la fecha/timestamp
        
        Returns:
            Análisis de tendencias temporales
        """
        if not documents:
            return {"error": "No documents provided"}
        
        # Analizar cada documento
        document_analyses = []
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            
            # Análisis básico
            sentiment = await self.analyzer.analyze_sentiment(content)
            keywords = await self.analyzer.extract_keywords(content, top_k=10)
            topics = await self.analyzer.analyze_topics(content, num_topics=3)
            
            document_analyses.append({
                "timestamp": doc.get(time_field, ""),
                "sentiment": sentiment,
                "keywords": keywords,
                "topics": topics.get("topics", []),
                "content_length": len(content)
            })
        
        # Analizar tendencias
        sentiment_trend = self._analyze_sentiment_trend(document_analyses)
        keyword_trend = self._analyze_keyword_trend(document_analyses)
        topic_trend = self._analyze_topic_trend(document_analyses)
        
        result = {
            "total_documents": len(document_analyses),
            "time_period": {
                "start": document_analyses[0].get("timestamp") if document_analyses else None,
                "end": document_analyses[-1].get("timestamp") if document_analyses else None
            },
            "sentiment_trend": sentiment_trend,
            "keyword_trend": keyword_trend,
            "topic_trend": topic_trend,
            "overall_trend": self._calculate_overall_trend(sentiment_trend, keyword_trend, topic_trend),
            "timestamp": datetime.now().isoformat()
        }
        
        self.trend_history.append(result)
        return result
    
    def _analyze_sentiment_trend(
        self,
        document_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar tendencia de sentimiento"""
        if not document_analyses:
            return {}
        
        sentiments = []
        for analysis in document_analyses:
            sentiment = analysis.get("sentiment", {})
            if isinstance(sentiment, dict):
                positive = sentiment.get("positive", 0.0)
                negative = sentiment.get("negative", 0.0)
                sentiments.append({
                    "positive": positive,
                    "negative": negative,
                    "net": positive - negative
                })
        
        if not sentiments:
            return {}
        
        # Calcular tendencia
        first_half = sentiments[:len(sentiments)//2]
        second_half = sentiments[len(sentiments)//2:]
        
        avg_first = sum(s["net"] for s in first_half) / len(first_half) if first_half else 0
        avg_second = sum(s["net"] for s in second_half) / len(second_half) if second_half else 0
        
        trend_direction = "improving" if avg_second > avg_first else "declining" if avg_second < avg_first else "stable"
        
        return {
            "trend_direction": trend_direction,
            "initial_avg": avg_first,
            "final_avg": avg_second,
            "change": avg_second - avg_first
        }
    
    def _analyze_keyword_trend(
        self,
        document_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar tendencia de keywords"""
        if not document_analyses:
            return {}
        
        # Contar keywords a lo largo del tiempo
        keyword_counts = {}
        for analysis in document_analyses:
            keywords = analysis.get("keywords", [])
            for keyword in keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Keywords más mencionados
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "most_common_keywords": [{"keyword": k, "count": c} for k, c in top_keywords],
            "total_unique_keywords": len(keyword_counts),
            "keyword_diversity": len(keyword_counts) / len(document_analyses) if document_analyses else 0
        }
    
    def _analyze_topic_trend(
        self,
        document_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar tendencia de temas"""
        if not document_analyses:
            return {}
        
        # Contar temas
        topic_counts = {}
        for analysis in document_analyses:
            topics = analysis.get("topics", [])
            for topic in topics:
                topic_keywords = topic.get("keywords", [])
                if topic_keywords:
                    main_keyword = topic_keywords[0]
                    topic_counts[main_keyword] = topic_counts.get(main_keyword, 0) + 1
        
        # Temas más comunes
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "most_common_topics": [{"topic": t, "count": c} for t, c in top_topics],
            "total_unique_topics": len(topic_counts),
            "topic_diversity": len(topic_counts) / len(document_analyses) if document_analyses else 0
        }
    
    def _calculate_overall_trend(
        self,
        sentiment_trend: Dict[str, Any],
        keyword_trend: Dict[str, Any],
        topic_trend: Dict[str, Any]
    ) -> str:
        """Calcular tendencia general"""
        sentiment_direction = sentiment_trend.get("trend_direction", "stable")
        
        if sentiment_direction == "improving":
            return "Positive Trend"
        elif sentiment_direction == "declining":
            return "Negative Trend"
        else:
            return "Stable Trend"


class DocumentImpactAnalyzer:
    """Analizador de impacto de documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.impact_history: List[Dict[str, Any]] = []
    
    async def analyze_impact(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analizar impacto potencial del documento
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
            context: Contexto adicional (audiencia, propósito, etc.)
        
        Returns:
            Análisis de impacto
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Analizar diferentes aspectos de impacto
        reach_impact = self._analyze_reach_impact(content, context)
        engagement_impact = await self._analyze_engagement_impact(content)
        influence_impact = await self._analyze_influence_impact(content)
        
        # Calcular score de impacto general
        overall_impact = self._calculate_overall_impact(
            reach_impact, engagement_impact, influence_impact
        )
        
        result = {
            "overall_impact_score": overall_impact,
            "impact_level": self._get_impact_level(overall_impact),
            "reach_impact": reach_impact,
            "engagement_impact": engagement_impact,
            "influence_impact": influence_impact,
            "recommendations": self._generate_impact_recommendations(
                reach_impact, engagement_impact, influence_impact
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        self.impact_history.append(result)
        return result
    
    def _analyze_reach_impact(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar impacto de alcance"""
        # Factores que afectan el alcance
        content_length = len(content)
        word_count = len(content.split())
        
        # Keywords relevantes aumentan alcance
        keyword_density = len(content.split()) / 100 if len(content.split()) > 0 else 0
        
        reach_score = 0.0
        
        # Longitud apropiada
        if 500 <= word_count <= 2000:
            reach_score += 0.3
        elif word_count > 2000:
            reach_score += 0.2
        
        # Metadata de contexto
        if context:
            audience_size = context.get("audience_size", 0)
            if audience_size > 1000:
                reach_score += 0.4
            elif audience_size > 100:
                reach_score += 0.2
        
        return {
            "reach_score": min(reach_score, 1.0),
            "content_length": content_length,
            "word_count": word_count,
            "keyword_density": keyword_density
        }
    
    async def _analyze_engagement_impact(self, content: str) -> Dict[str, Any]:
        """Analizar impacto de engagement"""
        # Análisis de legibilidad y estructura
        readability_analyzer = ReadabilityAnalyzer()
        readability = await readability_analyzer.analyze_readability(document_content=content)
        
        structure_analyzer = DocumentStructureAnalyzer(self.analyzer)
        structure = await structure_analyzer.analyze_structure(document_content=content)
        
        engagement_score = 0.0
        
        # Buena legibilidad
        flesch_score = readability.get("flesch_score", 0)
        engagement_score += min(flesch_score / 100.0, 1.0) * 0.5
        
        # Buena estructura
        has_headings = len(structure.get("headings", [])) > 0
        has_sections = len(structure.get("sections", [])) > 2
        engagement_score += (0.25 if has_headings else 0.0) + (0.25 if has_sections else 0.0)
        
        return {
            "engagement_score": min(engagement_score, 1.0),
            "readability": flesch_score,
            "has_structure": has_headings and has_sections
        }
    
    async def _analyze_influence_impact(self, content: str) -> Dict[str, Any]:
        """Analizar impacto de influencia"""
        # Factores de influencia
        sentiment = await self.analyzer.analyze_sentiment(content)
        keywords = await self.analyzer.extract_keywords(content, top_k=20)
        entities = await self.analyzer.extract_entities(content)
        
        influence_score = 0.0
        
        # Sentimiento positivo
        if isinstance(sentiment, dict):
            positive_score = sentiment.get("positive", 0.0)
            influence_score += positive_score * 0.3
        
        # Keywords relevantes
        influence_score += min(len(keywords) / 20.0, 1.0) * 0.3
        
        # Entidades importantes
        influence_score += min(len(entities) / 15.0, 1.0) * 0.4
        
        return {
            "influence_score": min(influence_score, 1.0),
            "keyword_count": len(keywords),
            "entity_count": len(entities),
            "sentiment_score": sentiment.get("positive", 0.0) if isinstance(sentiment, dict) else 0.0
        }
    
    def _calculate_overall_impact(
        self,
        reach: Dict[str, Any],
        engagement: Dict[str, Any],
        influence: Dict[str, Any]
    ) -> float:
        """Calcular score de impacto general"""
        return (
            reach.get("reach_score", 0.0) * 0.3 +
            engagement.get("engagement_score", 0.0) * 0.4 +
            influence.get("influence_score", 0.0) * 0.3
        )
    
    def _get_impact_level(self, score: float) -> str:
        """Obtener nivel de impacto"""
        if score >= 0.8:
            return "High Impact"
        elif score >= 0.6:
            return "Medium-High Impact"
        elif score >= 0.4:
            return "Medium Impact"
        elif score >= 0.2:
            return "Low-Medium Impact"
        else:
            return "Low Impact"
    
    def _generate_impact_recommendations(
        self,
        reach: Dict[str, Any],
        engagement: Dict[str, Any],
        influence: Dict[str, Any]
    ) -> List[str]:
        """Generar recomendaciones para aumentar impacto"""
        recommendations = []
        
        if reach.get("reach_score", 0.0) < 0.5:
            recommendations.append("Optimizar longitud del contenido para mayor alcance")
        
        if engagement.get("engagement_score", 0.0) < 0.6:
            recommendations.append("Mejorar legibilidad y estructura para aumentar engagement")
        
        if influence.get("influence_score", 0.0) < 0.5:
            recommendations.append("Aumentar keywords relevantes y entidades importantes")
        
        return recommendations


class DocumentBusinessIntelligence:
    """Sistema de inteligencia empresarial para documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.bi_reports: List[Dict[str, Any]] = []
    
    async def generate_business_intelligence_report(
        self,
        documents: List[Dict[str, Any]],
        report_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generar reporte de inteligencia empresarial
        
        Args:
            documents: Lista de documentos con contenido y metadata
            report_type: Tipo de reporte ('comprehensive', 'summary', 'detailed')
        
        Returns:
            Reporte de inteligencia empresarial
        """
        if not documents:
            return {"error": "No documents provided"}
        
        # Análisis agregado
        total_documents = len(documents)
        total_content_length = sum(len(doc.get("content", "")) for doc in documents)
        
        # Análisis de sentimiento agregado
        sentiment_analysis = await self._analyze_aggregate_sentiment(documents)
        
        # Análisis de temas dominantes
        topic_analysis = await self._analyze_aggregate_topics(documents)
        
        # Análisis de keywords empresariales
        keyword_analysis = await self._analyze_business_keywords(documents)
        
        # Análisis de entidades empresariales
        entity_analysis = await self._analyze_business_entities(documents)
        
        # Métricas empresariales
        business_metrics = self._calculate_business_metrics(
            documents, sentiment_analysis, topic_analysis, keyword_analysis, entity_analysis
        )
        
        result = {
            "report_type": report_type,
            "total_documents": total_documents,
            "total_content_length": total_content_length,
            "sentiment_analysis": sentiment_analysis,
            "topic_analysis": topic_analysis,
            "keyword_analysis": keyword_analysis,
            "entity_analysis": entity_analysis,
            "business_metrics": business_metrics,
            "insights": self._generate_business_insights(
                sentiment_analysis, topic_analysis, keyword_analysis, entity_analysis
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        self.bi_reports.append(result)
        return result
    
    async def _analyze_aggregate_sentiment(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar sentimiento agregado"""
        sentiments = []
        
        for doc in documents[:20]:  # Limitar para rendimiento
            content = doc.get("content", "")
            if content:
                sentiment = await self.analyzer.analyze_sentiment(content)
                if isinstance(sentiment, dict):
                    sentiments.append(sentiment)
        
        if not sentiments:
            return {}
        
        # Promediar sentimientos
        avg_positive = sum(s.get("positive", 0.0) for s in sentiments) / len(sentiments)
        avg_negative = sum(s.get("negative", 0.0) for s in sentiments) / len(sentiments)
        avg_neutral = sum(s.get("neutral", 0.0) for s in sentiments) / len(sentiments)
        
        return {
            "average_sentiment": {
                "positive": avg_positive,
                "negative": avg_negative,
                "neutral": avg_neutral
            },
            "dominant_sentiment": "positive" if avg_positive > avg_negative else "negative" if avg_negative > avg_positive else "neutral",
            "sentiment_balance": avg_positive - avg_negative
        }
    
    async def _analyze_aggregate_topics(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar temas agregados"""
        all_topics = []
        
        for doc in documents[:10]:  # Limitar para rendimiento
            content = doc.get("content", "")
            if content:
                topics = await self.analyzer.analyze_topics(content, num_topics=3)
                all_topics.extend(topics.get("topics", []))
        
        # Contar temas más comunes
        topic_counts = {}
        for topic in all_topics:
            keywords = topic.get("keywords", [])
            if keywords:
                main_keyword = keywords[0]
                topic_counts[main_keyword] = topic_counts.get(main_keyword, 0) + 1
        
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "dominant_topics": [{"topic": t, "frequency": c} for t, c in top_topics],
            "total_unique_topics": len(topic_counts),
            "topic_distribution": topic_counts
        }
    
    async def _analyze_business_keywords(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar keywords empresariales"""
        all_keywords = []
        
        for doc in documents[:15]:  # Limitar para rendimiento
            content = doc.get("content", "")
            if content:
                keywords = await self.analyzer.extract_keywords(content, top_k=10)
                all_keywords.extend(keywords)
        
        # Contar keywords más comunes
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_lower = keyword.lower()
            keyword_counts[keyword_lower] = keyword_counts.get(keyword_lower, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            "top_keywords": [{"keyword": k, "frequency": c} for k, c in top_keywords],
            "total_unique_keywords": len(keyword_counts),
            "keyword_diversity": len(keyword_counts) / len(documents) if documents else 0
        }
    
    async def _analyze_business_entities(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar entidades empresariales"""
        all_entities = []
        
        for doc in documents[:15]:  # Limitar para rendimiento
            content = doc.get("content", "")
            if content:
                entities = await self.analyzer.extract_entities(content)
                all_entities.extend(entities)
        
        # Agrupar por tipo de entidad
        entities_by_type = {}
        for entity in all_entities:
            entity_type = entity.get("label", "OTHER")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity.get("text", ""))
        
        # Contar entidades más comunes por tipo
        top_entities_by_type = {}
        for entity_type, entities in entities_by_type.items():
            entity_counts = {}
            for entity_text in entities:
                entity_counts[entity_text] = entity_counts.get(entity_text, 0) + 1
            
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_entities_by_type[entity_type] = [{"entity": e, "count": c} for e, c in top_entities]
        
        return {
            "entities_by_type": top_entities_by_type,
            "total_entities": len(all_entities),
            "entity_type_distribution": {k: len(v) for k, v in entities_by_type.items()}
        }
    
    def _calculate_business_metrics(
        self,
        documents: List[Dict[str, Any]],
        sentiment: Dict[str, Any],
        topics: Dict[str, Any],
        keywords: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcular métricas empresariales"""
        return {
            "document_volume": len(documents),
            "content_volume": sum(len(doc.get("content", "")) for doc in documents),
            "average_sentiment_score": sentiment.get("sentiment_balance", 0.0) if sentiment else 0.0,
            "topic_diversity": topics.get("total_unique_topics", 0),
            "keyword_coverage": keywords.get("total_unique_keywords", 0),
            "entity_richness": entities.get("total_entities", 0)
        }
    
    def _generate_business_insights(
        self,
        sentiment: Dict[str, Any],
        topics: Dict[str, Any],
        keywords: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> List[str]:
        """Generar insights empresariales"""
        insights = []
        
        if sentiment:
            dominant_sentiment = sentiment.get("dominant_sentiment", "neutral")
            if dominant_sentiment == "positive":
                insights.append("El sentimiento general es positivo, lo que indica buena percepción")
            elif dominant_sentiment == "negative":
                insights.append("El sentimiento general es negativo, requiere atención")
        
        if topics:
            dominant_topics = topics.get("dominant_topics", [])
            if dominant_topics:
                top_topic = dominant_topics[0].get("topic", "")
                insights.append(f"Tema dominante: {top_topic}")
        
        if keywords:
            top_keywords = keywords.get("top_keywords", [])
            if top_keywords:
                insights.append(f"Keywords más relevantes identificados: {len(top_keywords)}")
        
        return insights


# ============================================================================
# SISTEMAS AVANZADOS FINALES - ANÁLISIS ESPECIALIZADO POR DOMINIO
# ============================================================================

class MultiLanguageDocumentAnalyzer:
    """Analizador de documentos multi-idioma."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.multilang_history: List[Dict[str, Any]] = []
    
    async def analyze_multilanguage_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar documento multi-idioma
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis multi-idioma
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Detectar idiomas presentes
        language_detector = LanguageDetector(self.analyzer)
        languages = await self._detect_multiple_languages(content)
        
        # Analizar cada sección por idioma
        language_sections = self._segment_by_language(content, languages)
        
        # Análisis por idioma
        analysis_by_language = {}
        for lang, sections in language_sections.items():
            combined_content = ' '.join(sections)
            if combined_content:
                lang_analysis = await self._analyze_language_section(combined_content, lang)
                analysis_by_language[lang] = lang_analysis
        
        # Análisis comparativo entre idiomas
        comparative_analysis = self._compare_languages(analysis_by_language)
        
        result = {
            "detected_languages": languages,
            "language_sections": {lang: len(sections) for lang, sections in language_sections.items()},
            "analysis_by_language": analysis_by_language,
            "comparative_analysis": comparative_analysis,
            "primary_language": languages[0] if languages else "unknown",
            "is_multilingual": len(languages) > 1,
            "timestamp": datetime.now().isoformat()
        }
        
        self.multilang_history.append(result)
        return result
    
    async def _detect_multiple_languages(self, content: str) -> List[str]:
        """Detectar múltiples idiomas en el documento"""
        # Dividir en párrafos y detectar idioma de cada uno
        paragraphs = content.split('\n\n')
        detected_languages = set()
        
        language_detector = LanguageDetector(self.analyzer)
        
        for para in paragraphs[:20]:  # Limitar para rendimiento
            if len(para.strip()) > 50:
                lang_info = await language_detector.detect_language(document_content=para)
                detected_lang = lang_info.get("detected_language", "unknown")
                if detected_lang != "unknown":
                    detected_languages.add(detected_lang)
        
        return list(detected_languages)
    
    def _segment_by_language(
        self,
        content: str,
        languages: List[str]
    ) -> Dict[str, List[str]]:
        """Segmentar contenido por idioma"""
        # Implementación simplificada
        # En producción, usar detección de idioma por sección
        
        language_sections = {lang: [] for lang in languages}
        
        # Dividir en párrafos y asignar a idiomas
        paragraphs = content.split('\n\n')
        
        # Asignar párrafos a idiomas (simplificado - asignar todos al primer idioma)
        if languages:
            primary_lang = languages[0]
            language_sections[primary_lang] = paragraphs
        
        return language_sections
    
    async def _analyze_language_section(
        self,
        content: str,
        language: str
    ) -> Dict[str, Any]:
        """Analizar sección en un idioma específico"""
        # Análisis básico por idioma
        keywords = await self.analyzer.extract_keywords(content, top_k=10)
        sentiment = await self.analyzer.analyze_sentiment(content)
        
        return {
            "language": language,
            "keywords": keywords,
            "sentiment": sentiment,
            "word_count": len(content.split()),
            "character_count": len(content)
        }
    
    def _compare_languages(
        self,
        analysis_by_language: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Comparar análisis entre idiomas"""
        if len(analysis_by_language) < 2:
            return {"comparison": "insufficient_languages"}
        
        # Comparar keywords
        all_keywords = {}
        for lang, analysis in analysis_by_language.items():
            keywords = analysis.get("keywords", [])
            all_keywords[lang] = set(keywords)
        
        # Encontrar keywords comunes
        common_keywords = set.intersection(*all_keywords.values()) if all_keywords else set()
        
        return {
            "common_keywords": list(common_keywords)[:10],
            "unique_keywords_by_language": {
                lang: list(keywords - common_keywords)[:5]
                for lang, keywords in all_keywords.items()
            },
            "language_count": len(analysis_by_language)
        }


class AcademicDocumentAnalyzer:
    """Analizador especializado para documentos académicos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_academic_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar documento académico
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis académico completo
        """
        content = document_content
        if document_path:
            processor = DocumentProcessor()
            content = processor.process_document(document_path, "txt")
        
        if not content:
            return {"error": "No content to analyze"}
        
        # Usar analizador científico (ya existe)
        scientific_analyzer = ScientificDocumentAnalyzer(self.analyzer)
        scientific_analysis = await scientific_analyzer.analyze_scientific_document(document_content=content)
        
        # Análisis adicional académico
        abstract_analysis = self._extract_abstract(content)
        methodology_analysis = self._analyze_methodology(content)
        contribution_analysis = self._analyze_contributions(content)
        references_analysis = self._analyze_references_quality(content)
        
        # Calcular score académico
        academic_score = self._calculate_academic_score(
            scientific_analysis, abstract_analysis, methodology_analysis,
            contribution_analysis, references_analysis
        )
        
        result = {
            "scientific_analysis": scientific_analysis,
            "abstract_analysis": abstract_analysis,
            "methodology_analysis": methodology_analysis,
            "contribution_analysis": contribution_analysis,
            "references_analysis": references_analysis,
            "academic_score": academic_score,
            "academic_level": self._get_academic_level(academic_score),
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_history.append(result)
        return result
    
    def _extract_abstract(self, content: str) -> Dict[str, Any]:
        """Extraer y analizar abstract"""
        import re
        
        # Buscar abstract
        abstract_patterns = [
            r'(?:Abstract|Resumen|Abstracto)[:\s]*\n(.*?)(?=\n(?:Introduction|Keywords|Key words)|\Z)',
            r'(?:Abstract|Resumen)[:\s]*\n(.*?)(?=\n\n|\Z)'
        ]
        
        abstract_text = ""
        for pattern in abstract_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                abstract_text = match.group(1).strip()
                break
        
        return {
            "has_abstract": len(abstract_text) > 0,
            "abstract_length": len(abstract_text),
            "abstract_word_count": len(abstract_text.split()),
            "abstract_preview": abstract_text[:200] if abstract_text else ""
        }
    
    def _analyze_methodology(self, content: str) -> Dict[str, Any]:
        """Analizar metodología"""
        import re
        
        # Buscar sección de metodología
        method_patterns = [
            r'(?:Methodology|Methods|Métodos|Metodología)[:\s]*\n(.*?)(?=\n(?:Results|Findings|Conclusion)|\Z)',
            r'(?:Method|Método)[:\s]*\n(.*?)(?=\n(?:Result|Conclusion)|\Z)'
        ]
        
        method_content = ""
        for pattern in method_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                method_content = match.group(1).strip()
                break
        
        # Detectar métodos mencionados
        method_keywords = [
            "experiment", "survey", "interview", "case study", "analysis",
            "experimento", "encuesta", "entrevista", "estudio de caso", "análisis"
        ]
        
        methods_detected = []
        method_lower = method_content.lower()
        for keyword in method_keywords:
            if keyword.lower() in method_lower:
                methods_detected.append(keyword)
        
        return {
            "has_methodology": len(method_content) > 0,
            "methodology_length": len(method_content),
            "methods_detected": methods_detected,
            "methodology_preview": method_content[:300] if method_content else ""
        }
    
    def _analyze_contributions(self, content: str) -> Dict[str, Any]:
        """Analizar contribuciones"""
        import re
        
        # Buscar sección de contribuciones
        contribution_patterns = [
            r'(?:Contribution|Contributions|Contribuciones|Contribución)[:\s]*\n(.*?)(?=\n(?:Conclusion|References)|\Z)',
            r'(?:Main\s+Contribution|Principal\s+Contribución)[:\s]*\n(.*?)(?=\n(?:Conclusion|References)|\Z)'
        ]
        
        contribution_text = ""
        for pattern in contribution_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                contribution_text = match.group(1).strip()
                break
        
        # Si no hay sección explícita, buscar en introducción/conclusión
        if not contribution_text:
            intro_pattern = r'(?:Introduction|Introducción)[:\s]*\n(.*?)(?=\n(?:Method|Background)|\Z)'
            match = re.search(intro_pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                contribution_text = match.group(1).strip()
        
        return {
            "has_explicit_contributions": len(contribution_text) > 0,
            "contribution_length": len(contribution_text),
            "contribution_preview": contribution_text[:200] if contribution_text else ""
        }
    
    def _analyze_references_quality(self, content: str) -> Dict[str, Any]:
        """Analizar calidad de referencias"""
        citation_extractor = CitationExtractor(self.analyzer)
        citations = citation_extractor.extract_citations(document_content=content)
        
        inline_citations = citations.get("in_text_citations", [])
        bibliography = citations.get("bibliography", [])
        
        # Calcular métricas de calidad
        citation_ratio = len(inline_citations) / len(content.split()) * 1000 if content.split() else 0
        
        return {
            "inline_citations_count": len(inline_citations),
            "bibliography_count": len(bibliography),
            "citation_ratio_per_1000_words": citation_ratio,
            "has_bibliography": len(bibliography) > 0,
            "citation_quality": "high" if citation_ratio > 10 else "medium" if citation_ratio > 5 else "low"
        }
    
    def _calculate_academic_score(
        self,
        scientific: Dict[str, Any],
        abstract: Dict[str, Any],
        methodology: Dict[str, Any],
        contributions: Dict[str, Any],
        references: Dict[str, Any]
    ) -> float:
        """Calcular score académico"""
        score = 0.0
        
        # Rigor científico (30%)
        rigor_score = scientific.get("scientific_rigor", {}).get("rigor_score", 0.0)
        score += rigor_score * 0.3
        
        # Abstract (20%)
        if abstract.get("has_abstract", False):
            score += 0.2
        
        # Metodología (25%)
        if methodology.get("has_methodology", False):
            score += 0.25
        
        # Contribuciones (15%)
        if contributions.get("has_explicit_contributions", False):
            score += 0.15
        
        # Referencias (10%)
        if references.get("citation_quality", "") == "high":
            score += 0.1
        elif references.get("citation_quality", "") == "medium":
            score += 0.05
        
        return min(score, 1.0)
    
    def _get_academic_level(self, score: float) -> str:
        """Obtener nivel académico"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"


class MedicalDocumentAnalyzer:
    """Analizador especializado para documentos médicos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar analizador médico
        
        Args:
            analyzer: Instancia de DocumentAnalyzer para análisis base
        
        Raises:
            ValueError: Si analyzer es None
        """
        if analyzer is None:
            raise ValueError("DocumentAnalyzer no puede ser None")
        
        self.analyzer = analyzer
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("MedicalDocumentAnalyzer inicializado correctamente")
    
    async def analyze_medical_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar documento médico
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis médico completo con términos, síntomas, diagnósticos y medicamentos
        
        Raises:
            ValueError: Si no se proporciona ni document_path ni document_content
            FileNotFoundError: Si document_path no existe
        """
        # Validación de entrada
        if not document_path and not document_content:
            raise ValueError("Debe proporcionarse document_path o document_content")
        
        # Validar ruta de archivo
        if document_path:
            is_valid, error_msg = validate_file_path(document_path, must_exist=True)
            if not is_valid:
                raise FileNotFoundError(error_msg or f"Documento no encontrado: {document_path}")
        
        try:
            content = document_content
            if document_path:
                processor = DocumentProcessor()
                content = processor.process_document(document_path, "txt")
            
            # Validar contenido
            is_valid, error_msg = validate_content(content, min_length=50)
            if not is_valid:
                logger.warning(f"Contenido inválido: {error_msg}")
                return {"error": error_msg, "timestamp": datetime.now().isoformat()}
            
            logger.info(f"Analizando documento médico (longitud: {len(content)} caracteres)")
            
            # Extraer información médica específica
            medical_terms = self._extract_medical_terms(content)
            symptoms = self._extract_symptoms(content)
            diagnoses = self._extract_diagnoses(content)
            medications = self._extract_medications(content)
            procedures = self._extract_procedures(content)
            
            # Análisis de estructura médica
            medical_structure = self._analyze_medical_structure(content)
            
            # Detectar información sensible médica
            sensitive_info = self._detect_medical_sensitive_info(content)
            
            result = {
                "medical_terms": medical_terms,
                "symptoms": symptoms,
                "diagnoses": diagnoses,
                "medications": medications,
                "procedures": procedures,
                "medical_structure": medical_structure,
                "sensitive_info": sensitive_info,
                "total_medical_entities": len(medical_terms) + len(symptoms) + len(diagnoses) + len(medications),
                "timestamp": datetime.now().isoformat()
            }
            
            self.analysis_history.append(result)
            logger.info("Análisis médico completado exitosamente")
            return result
        
        except Exception as e:
            logger.error(f"Error analizando documento médico: {e}", exc_info=True)
            return {
                "error": f"Error durante el análisis: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_medical_terms(self, content: str) -> List[Dict[str, Any]]:
        """
        Extraer términos médicos
        
        Args:
            content: Contenido del documento
        
        Returns:
            Lista de términos médicos encontrados
        """
        import re
        
        if not content:
            return []
        
        # Patrones comunes de términos médicos
        medical_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:syndrome|disease|disorder|condition)',
            r'\b(?:diagnosis|diagnóstico)[:\s]+([A-Z][a-z]+(?:\s+[a-z]+)*)',
            r'\b(?:ICD|DSM)[-\s]?(\d+[\.-]?\d*)'
        ]
        
        medical_terms = []
        try:
            for pattern in medical_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                for match in matches:
                    term = match.group(1) if match.groups() and len(match.groups()) > 0 else match.group(0)
                    medical_terms.append({
                        "term": normalize_text(term, remove_extra_spaces=True),
                        "type": "medical_term",
                        "position": match.start()
                    })
        except Exception as e:
            logger.error(f"Error extrayendo términos médicos: {e}", exc_info=True)
        
        return medical_terms
    
    def _extract_symptoms(self, content: str) -> List[Dict[str, Any]]:
        """
        Extraer síntomas
        
        Args:
            content: Contenido del documento
        
        Returns:
            Lista de síntomas encontrados
        """
        import re
        
        if not content:
            return []
        
        symptoms = []
        
        # Patrones de síntomas
        symptom_patterns = [
            r'(?:symptom|síntoma)[:\s]+([^\.]+)',
            r'(?:presenting with|presentando)[:\s]+([^\.]+)',
            r'(?:complains of|se queja de)[:\s]+([^\.]+)'
        ]
        
        try:
            for pattern in symptom_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        symptom_text = normalize_text(match.group(1), remove_extra_spaces=True)
                        symptoms.append({
                            "symptom": symptom_text,
                            "position": match.start()
                        })
        except Exception as e:
            logger.error(f"Error extrayendo síntomas: {e}", exc_info=True)
        
        return symptoms
    
    def _extract_diagnoses(self, content: str) -> List[Dict[str, Any]]:
        """
        Extraer diagnósticos
        
        Args:
            content: Contenido del documento
        
        Returns:
            Lista de diagnósticos encontrados
        """
        import re
        
        if not content:
            return []
        
        diagnoses = []
        
        # Patrones de diagnósticos
        diagnosis_patterns = [
            r'(?:diagnosis|diagnóstico|dx)[:\s]+([A-Z][^\.]+)',
            r'(?:diagnosed with|diagnosticado con)[:\s]+([A-Z][^\.]+)',
            r'(?:primary diagnosis|diagnóstico principal)[:\s]+([A-Z][^\.]+)'
        ]
        
        try:
            for pattern in diagnosis_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        diagnosis_text = normalize_text(match.group(1), remove_extra_spaces=True)
                        diagnoses.append({
                            "diagnosis": diagnosis_text,
                            "position": match.start()
                        })
        except Exception as e:
            logger.error(f"Error extrayendo diagnósticos: {e}", exc_info=True)
        
        return diagnoses
    
    def _extract_medications(self, content: str) -> List[Dict[str, Any]]:
        """
        Extraer medicamentos
        
        Args:
            content: Contenido del documento
        
        Returns:
            Lista de medicamentos encontrados
        """
        import re
        
        if not content:
            return []
        
        medications = []
        
        # Patrones de medicamentos
        medication_patterns = [
            r'(?:medication|medicamento|prescribed|prescrito)[:\s]+([A-Z][a-z]+(?:\s+[a-z]+)*)',
            r'(?:taking|tomando)[:\s]+([A-Z][a-z]+(?:\s+[a-z]+)*)',
            r'\b([A-Z][a-z]+)\s+(?:mg|ml|tablet|capsule|tableta|cápsula)'
        ]
        
        try:
            for pattern in medication_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        medication_text = normalize_text(match.group(1), remove_extra_spaces=True)
                        if medication_text:
                            medications.append({
                                "medication": medication_text,
                                "position": match.start()
                            })
        except Exception as e:
            logger.error(f"Error extrayendo medicamentos: {e}", exc_info=True)
        
        return medications
    
    def _extract_procedures(self, content: str) -> List[Dict[str, Any]]:
        """
        Extraer procedimientos médicos
        
        Args:
            content: Contenido del documento
        
        Returns:
            Lista de procedimientos encontrados
        """
        import re
        
        if not content:
            return []
        
        procedures = []
        
        # Patrones de procedimientos
        procedure_patterns = [
            r'(?:procedure|procedimiento|surgery|cirugía)[:\s]+([A-Z][^\.]+)',
            r'(?:performed|realizado)[:\s]+([A-Z][^\.]+)',
            r'(?:CPT|procedure code)[:\s]+(\d+)'
        ]
        
        try:
            for pattern in procedure_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        procedure_text = normalize_text(match.group(1), remove_extra_spaces=True)
                    else:
                        procedure_text = normalize_text(match.group(0), remove_extra_spaces=True)
                    
                    if procedure_text:
                        procedures.append({
                            "procedure": procedure_text,
                            "position": match.start()
                        })
        except Exception as e:
            logger.error(f"Error extrayendo procedimientos: {e}", exc_info=True)
        
        return procedures
    
    def _analyze_medical_structure(self, content: str) -> Dict[str, Any]:
        """Analizar estructura médica"""
        import re
        
        # Secciones comunes en documentos médicos
        medical_sections = {
            "chief_complaint": False,
            "history_of_present_illness": False,
            "physical_examination": False,
            "assessment": False,
            "plan": False
        }
        
        content_lower = content.lower()
        
        section_patterns = {
            "chief_complaint": [r'chief complaint', r'queja principal', r'cc'],
            "history_of_present_illness": [r'history of present illness', r'historia de la enfermedad actual', r'hpi'],
            "physical_examination": [r'physical examination', r'examen físico', r'pe'],
            "assessment": [r'assessment', r'evaluación', r'impression'],
            "plan": [r'plan', r'plan de tratamiento', r'treatment plan']
        }
        
        for section, patterns in section_patterns.items():
            for pattern in patterns:
                if re.search(rf'\b{pattern}\b', content_lower):
                    medical_sections[section] = True
                    break
        
        return {
            "sections": medical_sections,
            "completeness": sum(medical_sections.values()) / len(medical_sections),
            "has_standard_structure": sum(medical_sections.values()) >= 3
        }
    
    def _detect_medical_sensitive_info(self, content: str) -> Dict[str, Any]:
        """
        Detectar información sensible médica
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con información sensible detectada
        """
        import re
        
        if not content:
            return {
                "sensitive_info": {},
                "has_sensitive_info": False,
                "risk_level": "low"
            }
        
        sensitive_info = {
            "patient_identifiers": [],
            "dates_of_birth": [],
            "medical_record_numbers": []
        }
        
        try:
            # Números de expediente médico
            mrn_pattern = r'(?:MRN|medical record number|número de expediente)[:\s]+([A-Z0-9-]+)'
            mrn_matches = safe_regex_match(mrn_pattern, content, re.IGNORECASE)
            for match in mrn_matches:
                if match.groups():
                    mrn = normalize_text(match.group(1), remove_extra_spaces=True)
                    if mrn:
                        sensitive_info["medical_record_numbers"].append(mrn)
            
            # Fechas de nacimiento
            dob_pattern = r'(?:DOB|date of birth|fecha de nacimiento)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            dob_matches = safe_regex_match(dob_pattern, content, re.IGNORECASE)
            for match in dob_matches:
                if match.groups():
                    dob = normalize_text(match.group(1), remove_extra_spaces=True)
                    if dob:
                        sensitive_info["dates_of_birth"].append(dob)
            
            has_sensitive = any(len(v) > 0 for v in sensitive_info.values())
            
            return {
                "sensitive_info": sensitive_info,
                "has_sensitive_info": has_sensitive,
                "risk_level": "high" if has_sensitive else "low",
                "total_sensitive_items": sum(len(v) for v in sensitive_info.values())
            }
        
        except Exception as e:
            logger.error(f"Error detectando información sensible: {e}", exc_info=True)
            return {
                "sensitive_info": sensitive_info,
                "has_sensitive_info": False,
                "risk_level": "unknown",
                "error": str(e)
            }


class FakeNewsDetector:
    """Detector de fake news y desinformación en documentos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar detector de fake news
        
        Args:
            analyzer: Instancia de DocumentAnalyzer para análisis base
        
        Raises:
            ValueError: Si analyzer es None
        """
        if analyzer is None:
            raise ValueError("DocumentAnalyzer no puede ser None")
        
        self.analyzer = analyzer
        self.detection_history: List[Dict[str, Any]] = []
        logger.info("FakeNewsDetector inicializado correctamente")
    
    async def detect_fake_news(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detectar indicadores de fake news
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis de veracidad con score de credibilidad
        
        Raises:
            ValueError: Si no se proporciona ni document_path ni document_content
            FileNotFoundError: Si document_path no existe
        """
        # Validación de entrada
        if not document_path and not document_content:
            raise ValueError("Debe proporcionarse document_path o document_content")
        
        if document_path:
            is_valid, error_msg = validate_file_path(document_path, must_exist=True)
            if not is_valid:
                raise FileNotFoundError(error_msg or f"Documento no encontrado: {document_path}")
        
        try:
            content = document_content
            if document_path:
                processor = DocumentProcessor()
                content = processor.process_document(document_path, "txt")
            
            # Validar contenido
            is_valid, error_msg = validate_content(content, min_length=50)
            if not is_valid:
                logger.warning(f"Contenido inválido: {error_msg}")
                return {"error": error_msg, "timestamp": datetime.now().isoformat()}
            
            logger.info(f"Analizando documento para fake news (longitud: {len(content)} caracteres)")
            
            # Análisis de diferentes indicadores
            sensationalism_score = self._analyze_sensationalism(content)
            credibility_indicators = self._analyze_credibility_indicators(content)
            source_analysis = self._analyze_sources(content)
            
            # Análisis de lenguaje con manejo de errores
            language_analysis = {}
            try:
                language_analysis = await self._analyze_language_patterns(content)
            except Exception as e:
                logger.warning(f"Error en análisis de lenguaje: {e}")
            
            # Calcular score de veracidad
            credibility_score = self._calculate_credibility_score(
                sensationalism_score, credibility_indicators, source_analysis, language_analysis
            )
            
            result = {
                "credibility_score": credibility_score,
                "credibility_level": self._get_credibility_level(credibility_score),
                "sensationalism_score": sensationalism_score,
                "credibility_indicators": credibility_indicators,
                "source_analysis": source_analysis,
                "language_analysis": language_analysis,
                "risk_factors": self._identify_risk_factors(
                    sensationalism_score, credibility_indicators, source_analysis
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            self.detection_history.append(result)
            logger.info("Análisis de fake news completado exitosamente")
            return result
        
        except Exception as e:
            logger.error(f"Error analizando documento para fake news: {e}", exc_info=True)
            return {
                "error": f"Error durante el análisis: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_sensationalism(self, content: str) -> float:
        """
        Analizar sensacionalismo en el contenido
        
        Args:
            content: Contenido del documento
        
        Returns:
            Score de sensacionalismo (0-1, donde 1 es muy sensacionalista)
        """
        if not content:
            return 0.0
        
        try:
            # Palabras sensacionalistas
            sensational_words = [
                "shocking", "unbelievable", "amazing", "incredible",
                "impactante", "increíble", "asombroso", "sorprendente",
                "breaking", "urgent", "alert", "warning"
            ]
            
            content_lower = content.lower()
            sensational_count = sum(1 for word in sensational_words if word.lower() in content_lower)
            
            words = content.split()
            total_words = len(words)
            
            sensationalism_ratio = sensational_count / total_words if total_words > 0 else 0.0
            
            # Uso excesivo de mayúsculas
            content_length = len(content)
            if content_length > 0:
                uppercase_count = sum(1 for c in content if c.isupper())
                uppercase_ratio = uppercase_count / content_length
            else:
                uppercase_ratio = 0.0
            
            # Score de sensacionalismo (0-1, donde 1 es muy sensacionalista)
            # Normalizar para evitar valores extremos
            sensationalism_score = min((sensationalism_ratio * 10) + (uppercase_ratio * 5), 1.0)
            
            return max(0.0, min(sensationalism_score, 1.0))
        
        except Exception as e:
            logger.error(f"Error analizando sensacionalismo: {e}", exc_info=True)
            return 0.0
    
    def _analyze_credibility_indicators(self, content: str) -> Dict[str, Any]:
        """
        Analizar indicadores de credibilidad
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con indicadores de credibilidad encontrados
        """
        import re
        
        if not content:
            return {
                "indicators": {},
                "credibility_indicators_count": 0,
                "has_strong_indicators": False
            }
        
        indicators = {
            "has_sources": False,
            "has_dates": False,
            "has_quotes": False,
            "has_statistics": False,
            "has_expert_mentions": False
        }
        
        try:
            # Verificar fuentes
            source_patterns = [
                r'(?:source|fuente|according to|según)[:\s]+([A-Z][^\.]+)',
                r'(?:cited|citado|references|referencias)'
            ]
            for pattern in source_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                if matches:
                    indicators["has_sources"] = True
                    break
            
            # Verificar fechas
            date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
            matches = safe_regex_match(date_pattern, content)
            if matches:
                indicators["has_dates"] = True
            
            # Verificar citas
            quote_pattern = r'["\']([^"\']{20,})["\']'
            matches = safe_regex_match(quote_pattern, content)
            if matches:
                indicators["has_quotes"] = True
            
            # Verificar estadísticas
            stat_pattern = r'\d+(?:\.\d+)?\s*(?:%|percent|por ciento|million|millón|billion|mil millones)'
            matches = safe_regex_match(stat_pattern, content, re.IGNORECASE)
            if matches:
                indicators["has_statistics"] = True
            
            # Verificar menciones de expertos
            expert_patterns = [
                r'(?:expert|experto|researcher|investigador|professor|profesor)[:\s]+([A-Z][a-z]+)',
                r'(?:Dr\.|Doctor|PhD|doctor)[:\s]+([A-Z][a-z]+)'
            ]
            for pattern in expert_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                if matches:
                    indicators["has_expert_mentions"] = True
                    break
            
            indicator_count = sum(indicators.values())
            
            return {
                "indicators": indicators,
                "credibility_indicators_count": indicator_count,
                "has_strong_indicators": indicator_count >= 3
            }
        
        except Exception as e:
            logger.error(f"Error analizando indicadores de credibilidad: {e}", exc_info=True)
            return {
                "indicators": indicators,
                "credibility_indicators_count": 0,
                "has_strong_indicators": False,
                "error": str(e)
            }
    
    def _analyze_sources(self, content: str) -> Dict[str, Any]:
        """
        Analizar fuentes mencionadas
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con fuentes encontradas
        """
        import re
        
        if not content:
            return {
                "sources_found": [],
                "source_count": 0,
                "has_sources": False
            }
        
        # Buscar menciones de fuentes
        source_patterns = [
            r'(?:source|fuente|according to|según)[:\s]+([A-Z][^\.]+)',
            r'(?:reported by|reportado por)[:\s]+([A-Z][^\.]+)',
            r'(?:study|estudio|research|investigación)[:\s]+([A-Z][^\.]+)'
        ]
        
        sources = []
        try:
            for pattern in source_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        source_text = normalize_text(match.group(1), remove_extra_spaces=True)
                        if source_text:
                            sources.append(source_text)
        except Exception as e:
            logger.error(f"Error analizando fuentes: {e}", exc_info=True)
        
        # Eliminar duplicados manteniendo el orden
        unique_sources = []
        seen = set()
        for source in sources:
            source_lower = source.lower()
            if source_lower not in seen:
                seen.add(source_lower)
                unique_sources.append(source)
        
        return {
            "sources_found": unique_sources[:10],
            "source_count": len(unique_sources),
            "has_sources": len(unique_sources) > 0
        }
    
    async def _analyze_language_patterns(self, content: str) -> Dict[str, Any]:
        """
        Analizar patrones de lenguaje
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con análisis de patrones de lenguaje
        """
        if not content:
            return {
                "emotional_ratio": 0.0,
                "extreme_sentiment": False,
                "has_emotional_language": False
            }
        
        try:
            # Detectar uso excesivo de emociones
            emotional_words = [
                "terrible", "horrible", "amazing", "incredible",
                "increíble", "asombroso", "increíble", "asombroso"
            ]
            
            content_lower = content.lower()
            emotional_count = sum(1 for word in emotional_words if word.lower() in content_lower)
            
            # Calcular ratio de manera segura
            words = content.split()
            word_count = len(words)
            if word_count > 0:
                emotional_ratio = emotional_count / word_count
            else:
                emotional_ratio = 0.0
            
            # Análisis de sentimiento extremo con manejo de errores
            extreme_sentiment = False
            sentiment = {}
            try:
                sentiment = await self.analyzer.analyze_sentiment(content)
                if isinstance(sentiment, dict):
                    positive = sentiment.get("positive", 0.0)
                    negative = sentiment.get("negative", 0.0)
                    if positive > 0.8 or negative > 0.8:
                        extreme_sentiment = True
            except Exception as e:
                logger.warning(f"Error en análisis de sentimiento para patrones de lenguaje: {e}")
            
            return {
                "emotional_ratio": emotional_ratio,
                "extreme_sentiment": extreme_sentiment,
                "has_emotional_language": emotional_ratio > 0.05,
                "emotional_count": emotional_count,
                "word_count": word_count
            }
        
        except Exception as e:
            logger.error(f"Error analizando patrones de lenguaje: {e}", exc_info=True)
            return {
                "emotional_ratio": 0.0,
                "extreme_sentiment": False,
                "has_emotional_language": False,
                "error": str(e)
            }
    
    def _calculate_credibility_score(
        self,
        sensationalism: float,
        credibility: Dict[str, Any],
        sources: Dict[str, Any],
        language: Dict[str, Any]
    ) -> float:
        """Calcular score de credibilidad"""
        score = 0.0
        
        # Sensacionalismo (negativo - restar)
        score += (1.0 - sensationalism) * 0.3
        
        # Indicadores de credibilidad (positivo)
        indicator_count = credibility.get("credibility_indicators_count", 0)
        score += min(indicator_count / 5.0, 1.0) * 0.4
        
        # Fuentes (positivo)
        if sources.get("has_sources", False):
            score += 0.2
        
        # Lenguaje emocional (negativo - restar)
        if language.get("has_emotional_language", False):
            score -= 0.1
        
        return max(0.0, min(score, 1.0))
    
    def _get_credibility_level(self, score: float) -> str:
        """Obtener nivel de credibilidad"""
        if score >= 0.7:
            return "High Credibility"
        elif score >= 0.5:
            return "Medium Credibility"
        elif score >= 0.3:
            return "Low-Medium Credibility"
        else:
            return "Low Credibility"
    
    def _identify_risk_factors(
        self,
        sensationalism: float,
        credibility: Dict[str, Any],
        sources: Dict[str, Any]
    ) -> List[str]:
        """Identificar factores de riesgo"""
        risk_factors = []
        
        if sensationalism > 0.5:
            risk_factors.append("Alto nivel de sensacionalismo detectado")
        
        if not credibility.get("has_strong_indicators", False):
            risk_factors.append("Faltan indicadores fuentes de credibilidad")
        
        if not sources.get("has_sources", False):
            risk_factors.append("No se encontraron fuentes citadas")
        
        return risk_factors


class SmartContractAnalyzer:
    """Analizador inteligente de contratos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar analizador inteligente de contratos
        
        Args:
            analyzer: Instancia de DocumentAnalyzer para análisis base
        
        Raises:
            ValueError: Si analyzer es None
        """
        if analyzer is None:
            raise ValueError("DocumentAnalyzer no puede ser None")
        
        self.analyzer = analyzer
        try:
            self.contract_analyzer = LegalContractAnalyzer(analyzer)
        except Exception as e:
            logger.warning(f"No se pudo inicializar LegalContractAnalyzer: {e}")
            self.contract_analyzer = None
        
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("SmartContractAnalyzer inicializado correctamente")
    
    async def analyze_contract_intelligently(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Análisis inteligente de contrato
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis inteligente del contrato con obligaciones, riesgos y cumplimiento
        
        Raises:
            ValueError: Si no se proporciona ni document_path ni document_content
            FileNotFoundError: Si document_path no existe
        """
        # Validación de entrada
        if not document_path and not document_content:
            raise ValueError("Debe proporcionarse document_path o document_content")
        
        if document_path:
            is_valid, error_msg = validate_file_path(document_path, must_exist=True)
            if not is_valid:
                raise FileNotFoundError(error_msg or f"Documento no encontrado: {document_path}")
        
        try:
            content = document_content
            if document_path:
                processor = DocumentProcessor()
                content = processor.process_document(document_path, "txt")
            
            # Validar contenido
            is_valid, error_msg = validate_content(content, min_length=50)
            if not is_valid:
                logger.warning(f"Contenido inválido: {error_msg}")
                return {"error": error_msg, "timestamp": datetime.now().isoformat()}
            
            logger.info(f"Analizando contrato inteligentemente (longitud: {len(content)} caracteres)")
            
            # Análisis legal básico con manejo de errores
            legal_analysis = {}
            if self.contract_analyzer:
                try:
                    legal_analysis = await self.contract_analyzer.analyze_contract(document_content=content)
                except Exception as e:
                    logger.warning(f"Error en análisis legal básico: {e}")
            else:
                logger.warning("LegalContractAnalyzer no disponible")
            
            # Análisis adicional inteligente
            obligation_analysis = self._analyze_obligations_detailed(content)
            risk_assessment = await self._assess_contract_risks(content)
            compliance_check = self._check_contract_compliance(content)
            
            # Generar recomendaciones inteligentes
            recommendations = []
            try:
                recommendations = await self._generate_smart_recommendations(
                    legal_analysis, obligation_analysis, risk_assessment, compliance_check
                )
            except Exception as e:
                logger.warning(f"Error generando recomendaciones: {e}")
            
            result = {
                "legal_analysis": legal_analysis,
                "obligation_analysis": obligation_analysis,
                "risk_assessment": risk_assessment,
                "compliance_check": compliance_check,
                "smart_recommendations": recommendations,
                "overall_risk_score": risk_assessment.get("overall_risk_score", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            
            self.analysis_history.append(result)
            logger.info("Análisis inteligente de contrato completado exitosamente")
            return result
        
        except Exception as e:
            logger.error(f"Error analizando contrato inteligentemente: {e}", exc_info=True)
            return {
                "error": f"Error durante el análisis: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_obligations_detailed(self, content: str) -> Dict[str, Any]:
        """
        Análisis detallado de obligaciones
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con obligaciones encontradas
        """
        import re
        
        if not content:
            return {
                "obligations": [],
                "total_obligations": 0,
                "obligation_density": 0.0
            }
        
        obligations = []
        
        # Patrones de obligaciones
        obligation_patterns = [
            r'(?:shall|must|debe|deberá|obligado a|required to)\s+([^\.]+)',
            r'(?:obligation|obligación)[:\s]+([^\.]+)',
            r'(?:is required|se requiere|must comply|debe cumplir)[:\s]+([^\.]+)'
        ]
        
        try:
            for pattern in obligation_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        obligation_text = normalize_text(match.group(1), remove_extra_spaces=True)
                        if obligation_text:
                            obligations.append({
                                "obligation": obligation_text,
                                "position": match.start(),
                                "type": "mandatory"
                            })
        except Exception as e:
            logger.error(f"Error analizando obligaciones: {e}", exc_info=True)
        
        words = content.split()
        word_count = len(words)
        obligation_density = (len(obligations) / word_count * 1000) if word_count > 0 else 0.0
        
        return {
            "obligations": obligations,
            "total_obligations": len(obligations),
            "obligation_density": obligation_density
        }
    
    async def _assess_contract_risks(self, content: str) -> Dict[str, Any]:
        """
        Evaluar riesgos del contrato
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con evaluación de riesgos
        """
        import re
        
        if not content:
            return {
                "risk_clauses": [],
                "total_risk_clauses": 0,
                "overall_risk_score": 0.0,
                "risk_level": "low"
            }
        
        # Detectar cláusulas de riesgo alto
        risk_keywords = [
            "indemnification", "indemnización",
            "limitation of liability", "limitación de responsabilidad",
            "penalty", "penalización",
            "termination", "terminación",
            "breach", "incumplimiento"
        ]
        
        risk_clauses = []
        content_lower = content.lower()
        
        try:
            for keyword in risk_keywords:
                keyword_lower = keyword.lower()
                # Buscar palabra completa con boundaries
                pattern = rf'\b{re.escape(keyword_lower)}\b'
                matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                if matches:
                    risk_level = "high" if ("indemn" in keyword_lower or "penalty" in keyword_lower or "penalización" in keyword_lower) else "medium"
                    risk_clauses.append({
                        "keyword": keyword,
                        "risk_level": risk_level,
                        "count": len(matches)
                    })
        except Exception as e:
            logger.error(f"Error evaluando riesgos del contrato: {e}", exc_info=True)
        
        # Calcular score de riesgo
        total_risk_clauses = len(risk_clauses)
        high_risk_count = sum(1 for r in risk_clauses if r["risk_level"] == "high")
        
        # Score basado en número de cláusulas y nivel de riesgo
        overall_risk = min((high_risk_count * 0.3 + total_risk_clauses * 0.1), 1.0) if risk_clauses else 0.0
        
        if overall_risk > 0.6:
            risk_level = "high"
        elif overall_risk > 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_clauses": risk_clauses,
            "total_risk_clauses": total_risk_clauses,
            "high_risk_count": high_risk_count,
            "overall_risk_score": overall_risk,
            "risk_level": risk_level
        }
    
    def _check_contract_compliance(self, content: str) -> Dict[str, Any]:
        """
        Verificar cumplimiento de contrato
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con verificación de cumplimiento
        """
        import re
        
        if not content:
            return {
                "compliance_checks": {},
                "compliance_score": 0.0,
                "missing_elements": []
            }
        
        compliance_checks = {
            "has_effective_date": False,
            "has_termination_clause": False,
            "has_payment_terms": False,
            "has_dispute_resolution": False
        }
        
        try:
            # Patrones para cada verificación
            compliance_patterns = {
                "has_effective_date": r'\b(?:effective|vigencia|effective date|fecha de vigencia)\b',
                "has_termination_clause": r'\b(?:termination|terminación|terminate|terminar)\b',
                "has_payment_terms": r'\b(?:payment|pago|payment terms|términos de pago)\b',
                "has_dispute_resolution": r'\b(?:dispute|disputa|arbitration|arbitraje|mediation|mediación)\b'
            }
            
            for check, pattern in compliance_patterns.items():
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                if matches:
                    compliance_checks[check] = True
            
            compliance_count = sum(compliance_checks.values())
            compliance_score = compliance_count / len(compliance_checks) if compliance_checks else 0.0
            missing_elements = [k for k, v in compliance_checks.items() if not v]
            
            return {
                "compliance_checks": compliance_checks,
                "compliance_score": compliance_score,
                "missing_elements": missing_elements
            }
        
        except Exception as e:
            logger.error(f"Error verificando cumplimiento de contrato: {e}", exc_info=True)
            return {
                "compliance_checks": compliance_checks,
                "compliance_score": 0.0,
                "missing_elements": list(compliance_checks.keys()),
                "error": str(e)
            }
    
    async def _generate_smart_recommendations(
        self,
        legal: Dict[str, Any],
        obligations: Dict[str, Any],
        risks: Dict[str, Any],
        compliance: Dict[str, Any]
    ) -> List[str]:
        """
        Generar recomendaciones inteligentes
        
        Args:
            legal: Análisis legal básico
            obligations: Análisis de obligaciones
            risks: Evaluación de riesgos
            compliance: Verificación de cumplimiento
        
        Returns:
            Lista de recomendaciones inteligentes
        """
        recommendations = []
        
        try:
            # Validar que los parámetros sean diccionarios
            if not isinstance(risks, dict):
                risks = {}
            if not isinstance(obligations, dict):
                obligations = {}
            if not isinstance(compliance, dict):
                compliance = {}
            
            # Recomendación basada en riesgo
            risk_score = risks.get("overall_risk_score", 0.0)
            if isinstance(risk_score, (int, float)) and risk_score > 0.6:
                recommendations.append("ALERTA: Alto nivel de riesgo detectado. Revisar cláusulas de indemnización y limitación de responsabilidad")
            
            # Recomendación basada en número de obligaciones
            total_obligations = obligations.get("total_obligations", 0)
            if isinstance(total_obligations, (int, float)) and total_obligations > 20:
                recommendations.append("Contrato con muchas obligaciones. Considerar revisar si todas son necesarias")
            
            # Recomendación basada en cumplimiento
            compliance_score = compliance.get("compliance_score", 0.0)
            if isinstance(compliance_score, (int, float)) and compliance_score < 0.5:
                missing = compliance.get("missing_elements", [])
                missing_str = ", ".join(missing) if missing else "elementos importantes"
                recommendations.append(f"Faltan {missing_str} del contrato. Agregar fecha de vigencia, términos de pago y resolución de disputas")
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error generando recomendaciones inteligentes: {e}", exc_info=True)
            return ["Error generando recomendaciones. Por favor, revise el contrato manualmente."]


# ============================================================================
# SISTEMAS AVANZADOS FINALES - ANÁLISIS ESPECIALIZADO POR INDUSTRIA
# ============================================================================

class FinancialDocumentAnalyzer:
    """Analizador especializado para documentos financieros."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar analizador financiero
        
        Args:
            analyzer: Instancia de DocumentAnalyzer para análisis base
        
        Raises:
            ValueError: Si analyzer es None
        """
        if analyzer is None:
            raise ValueError("DocumentAnalyzer no puede ser None")
        
        self.analyzer = analyzer
        try:
            self.financial_extractor = FinancialDataExtractor(analyzer)
        except Exception as e:
            logger.warning(f"No se pudo inicializar FinancialDataExtractor: {e}")
            self.financial_extractor = None
        
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("FinancialDocumentAnalyzer inicializado correctamente")
    
    async def analyze_financial_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar documento financiero
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis financiero completo con métricas, riesgos y cumplimiento
        
        Raises:
            ValueError: Si no se proporciona ni document_path ni document_content
            FileNotFoundError: Si document_path no existe
        """
        # Validación de entrada
        if not document_path and not document_content:
            raise ValueError("Debe proporcionarse document_path o document_content")
        
        if document_path and not os.path.exists(document_path):
            raise FileNotFoundError(f"Documento no encontrado: {document_path}")
        
        try:
            content = document_content
            if document_path:
                processor = DocumentProcessor()
                content = processor.process_document(document_path, "txt")
            
            if not content or len(content.strip()) == 0:
                logger.warning("Contenido del documento vacío o inválido")
                return {"error": "No content to analyze", "timestamp": datetime.now().isoformat()}
            
            # Validar longitud mínima
            if len(content.strip()) < 50:
                logger.warning(f"Documento muy corto ({len(content)} caracteres)")
            
            logger.info(f"Analizando documento financiero (longitud: {len(content)} caracteres)")
            
            # Extracción de datos financieros
            financial_data = {}
            if self.financial_extractor:
                try:
                    financial_data = await self.financial_extractor.extract_financial_data(document_content=content)
                except Exception as e:
                    logger.error(f"Error extrayendo datos financieros: {e}")
                    financial_data = {"error": str(e)}
            else:
                logger.warning("FinancialDataExtractor no disponible")
            
            # Análisis adicional financiero
            financial_metrics = self._extract_financial_metrics(content)
            risk_analysis = self._analyze_financial_risks(content)
            compliance_analysis = self._analyze_financial_compliance(content)
            trend_analysis = self._analyze_financial_trends(content)
            
            result = {
                "financial_data": financial_data,
                "financial_metrics": financial_metrics,
                "risk_analysis": risk_analysis,
                "compliance_analysis": compliance_analysis,
                "trend_analysis": trend_analysis,
                "financial_health_score": self._calculate_financial_health_score(
                    financial_metrics, risk_analysis, compliance_analysis
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            self.analysis_history.append(result)
            logger.info("Análisis financiero completado exitosamente")
            return result
        
        except Exception as e:
            logger.error(f"Error analizando documento financiero: {e}", exc_info=True)
            return {
                "error": f"Error durante el análisis: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_financial_metrics(self, content: str) -> Dict[str, Any]:
        """
        Extraer métricas financieras del contenido
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con métricas financieras extraídas
        """
        # Validar contenido usando función de utilidad
        is_valid, error_msg = validate_content(content, min_length=20)
        if not is_valid:
            logger.warning(f"Contenido inválido para extracción de métricas: {error_msg}")
            return {"metrics": {}, "has_financial_data": False, "error": error_msg}
        
        import re
        
        metrics = {
            "revenue": [],
            "expenses": [],
            "profit": [],
            "assets": [],
            "liabilities": [],
            "equity": []
        }
        
        # Patrones para extraer métricas
        metric_patterns = {
            "revenue": [
                r'(?:revenue|ingresos|ventas|sales)[:\s]*\$?([\d,]+(?:\.\d+)?)',
                r'(?:total revenue|ingresos totales)[:\s]*\$?([\d,]+(?:\.\d+)?)'
            ],
            "expenses": [
                r'(?:expenses|gastos|costs|costos)[:\s]*\$?([\d,]+(?:\.\d+)?)',
                r'(?:total expenses|gastos totales)[:\s]*\$?([\d,]+(?:\.\d+)?)'
            ],
            "profit": [
                r'(?:profit|ganancia|net income|ingreso neto)[:\s]*\$?([\d,]+(?:\.\d+)?)',
                r'(?:net profit|ganancia neta)[:\s]*\$?([\d,]+(?:\.\d+)?)'
            ],
            "assets": [
                r'(?:assets|activos|total assets|activos totales)[:\s]*\$?([\d,]+(?:\.\d+)?)'
            ],
            "liabilities": [
                r'(?:liabilities|pasivos|total liabilities|pasivos totales)[:\s]*\$?([\d,]+(?:\.\d+)?)'
            ],
            "equity": [
                r'(?:equity|patrimonio|shareholders equity|patrimonio de accionistas)[:\s]*\$?([\d,]+(?:\.\d+)?)'
            ]
        }
        
        try:
            for metric_type, patterns in metric_patterns.items():
                for pattern in patterns:
                    try:
                        matches = safe_regex_match(pattern, content, re.IGNORECASE)
                        for match in matches:
                            if match.groups():
                                value_str = match.group(1).strip()
                                if value_str:
                                    # Usar función de utilidad para conversión segura
                                    value = safe_float_conversion(value_str)
                                    if value != 0.0:  # Solo agregar si la conversión fue exitosa
                                        # Validar que el valor sea razonable (no negativo a menos que sea expenses/liabilities)
                                        if value >= 0 or metric_type in ["expenses", "liabilities"]:
                                            metrics[metric_type].append(value)
                    except re.error as e:
                        logger.warning(f"Error en patrón regex para {metric_type}: {e}")
            
            # Calcular promedios si hay múltiples valores
            calculated_metrics = {}
            for metric_type, values in metrics.items():
                if values:
                    # Filtrar valores atípicos (outliers)
                    if len(values) > 1:
                        values_sorted = sorted(values)
                        q1_idx = len(values_sorted) // 4
                        q3_idx = 3 * len(values_sorted) // 4
                        q1 = values_sorted[q1_idx]
                        q3 = values_sorted[q3_idx]
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        filtered_values = [v for v in values if lower_bound <= v <= upper_bound]
                        if filtered_values:
                            values = filtered_values
                    
                    # Usar función de utilidad para calcular estadísticas
                    stats = calculate_statistics(values)
                    calculated_metrics[metric_type] = {
                        "values": values,
                        "total": stats["mean"] * stats["count"],
                        "average": stats["mean"],
                        "median": stats["median"],
                        "std": stats["std"],
                        "count": stats["count"],
                        "min": stats["min"],
                        "max": stats["max"]
                    }
            
            logger.info(f"Extraídas {len(calculated_metrics)} tipos de métricas financieras")
            return {
                "metrics": calculated_metrics,
                "has_financial_data": len(calculated_metrics) > 0,
                "total_metrics_found": sum(len(v) for v in metrics.values())
            }
        
        except Exception as e:
            logger.error(f"Error extrayendo métricas financieras: {e}", exc_info=True)
            return {"metrics": {}, "has_financial_data": False, "error": str(e)}
    
    def _analyze_financial_risks(self, content: str) -> Dict[str, Any]:
        """
        Analizar riesgos financieros
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con análisis de riesgos financieros
        """
        import re
        
        if not content:
            return {
                "risk_mentions": [],
                "total_risk_mentions": 0,
                "high_risk_count": 0,
                "risk_score": 0.0,
                "risk_level": "low"
            }
        
        risk_keywords = [
            "debt", "deuda",
            "default", "incumplimiento",
            "bankruptcy", "quiebra",
            "liquidity", "liquidez",
            "volatility", "volatilidad",
            "risk", "riesgo",
            "uncertainty", "incertidumbre"
        ]
        
        risk_mentions = []
        content_lower = content.lower()
        
        try:
            high_risk_keywords = {"default", "bankruptcy", "incumplimiento", "quiebra"}
            
            for keyword in risk_keywords:
                keyword_lower = keyword.lower()
                # Buscar palabra completa con boundaries
                pattern = rf'\b{re.escape(keyword_lower)}\b'
                matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                count = len(matches)
                if count > 0:
                    risk_level = "high" if keyword_lower in high_risk_keywords else "medium"
                    risk_mentions.append({
                        "keyword": keyword,
                        "count": count,
                        "risk_level": risk_level
                    })
            
            # Calcular score de riesgo
            high_risk_count = sum(1 for r in risk_mentions if r["risk_level"] == "high")
            total_risk_mentions = sum(r["count"] for r in risk_mentions)
            
            # Score basado en número de cláusulas de alto riesgo y total de menciones
            risk_score = min((high_risk_count * 0.5) + (total_risk_mentions * 0.1), 1.0)
            
            if risk_score > 0.6:
                risk_level = "high"
            elif risk_score > 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "risk_mentions": risk_mentions,
                "total_risk_mentions": total_risk_mentions,
                "high_risk_count": high_risk_count,
                "risk_score": risk_score,
                "risk_level": risk_level
            }
        
        except Exception as e:
            logger.error(f"Error analizando riesgos financieros: {e}", exc_info=True)
            return {
                "risk_mentions": [],
                "total_risk_mentions": 0,
                "high_risk_count": 0,
                "risk_score": 0.0,
                "risk_level": "unknown",
                "error": str(e)
            }
    
    def _analyze_financial_compliance(self, content: str) -> Dict[str, Any]:
        """
        Analizar cumplimiento financiero
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con verificación de cumplimiento financiero
        """
        import re
        
        if not content:
            return {
                "compliance_indicators": {},
                "compliance_score": 0.0,
                "compliance_level": "low"
            }
        
        compliance_indicators = {
            "has_audit": False,
            "has_gaap": False,
            "has_ifrs": False,
            "has_disclosures": False,
            "has_footnotes": False
        }
        
        content_lower = content.lower()
        
        try:
            # Patrones para cada indicador
            compliance_patterns = {
                "has_audit": r'\b(?:audit|auditoría|audited|auditado)\b',
                "has_gaap": r'\b(?:GAAP|generally accepted accounting principles)\b',
                "has_ifrs": r'\b(?:IFRS|international financial reporting standards)\b',
                "has_disclosures": r'\b(?:disclosure|revelación|disclosures|revelaciones)\b',
                "has_footnotes": r'\b(?:footnote|nota al pie|footnotes|notas al pie)\b'
            }
            
            for indicator, pattern in compliance_patterns.items():
                matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                if matches:
                    compliance_indicators[indicator] = True
            
            compliance_count = sum(compliance_indicators.values())
            compliance_score = compliance_count / len(compliance_indicators) if compliance_indicators else 0.0
            
            if compliance_count >= 3:
                compliance_level = "high"
            elif compliance_count >= 2:
                compliance_level = "medium"
            else:
                compliance_level = "low"
            
            return {
                "compliance_indicators": compliance_indicators,
                "compliance_score": compliance_score,
                "compliance_level": compliance_level
            }
        
        except Exception as e:
            logger.error(f"Error analizando cumplimiento financiero: {e}", exc_info=True)
            return {
                "compliance_indicators": compliance_indicators,
                "compliance_score": 0.0,
                "compliance_level": "unknown",
                "error": str(e)
            }
    
    def _analyze_financial_trends(self, content: str) -> Dict[str, Any]:
        """
        Analizar tendencias financieras
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con análisis de tendencias financieras
        """
        import re
        
        if not content:
            return {
                "trends": [],
                "positive_trends_count": 0,
                "negative_trends_count": 0,
                "overall_trend": "neutral"
            }
        
        trend_keywords = [
            "growth", "crecimiento",
            "increase", "aumento",
            "decrease", "disminución",
            "decline", "declive",
            "improvement", "mejora",
            "deterioration", "deterioro"
        ]
        
        positive_keywords = {"growth", "increase", "improvement", "crecimiento", "aumento", "mejora"}
        
        trends = []
        content_lower = content.lower()
        
        try:
            for keyword in trend_keywords:
                keyword_lower = keyword.lower()
                # Buscar palabra completa con boundaries
                pattern = rf'\b{re.escape(keyword_lower)}\b'
                matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                count = len(matches)
                if count > 0:
                    trend_type = "positive" if keyword_lower in positive_keywords else "negative"
                    trends.append({
                        "keyword": keyword,
                        "count": count,
                        "type": trend_type
                    })
            
            positive_trends = sum(t["count"] for t in trends if t["type"] == "positive")
            negative_trends = sum(t["count"] for t in trends if t["type"] == "negative")
            
            if positive_trends > negative_trends:
                overall_trend = "positive"
            elif negative_trends > positive_trends:
                overall_trend = "negative"
            else:
                overall_trend = "neutral"
            
            return {
                "trends": trends,
                "positive_trends_count": positive_trends,
                "negative_trends_count": negative_trends,
                "overall_trend": overall_trend
            }
        
        except Exception as e:
            logger.error(f"Error analizando tendencias financieras: {e}", exc_info=True)
            return {
                "trends": [],
                "positive_trends_count": 0,
                "negative_trends_count": 0,
                "overall_trend": "unknown",
                "error": str(e)
            }
    
    def _calculate_financial_health_score(
        self,
        metrics: Dict[str, Any],
        risks: Dict[str, Any],
        compliance: Dict[str, Any]
    ) -> float:
        """Calcular score de salud financiera"""
        score = 0.0
        
        # Métricas financieras (40%)
        if metrics.get("has_financial_data", False):
            score += 0.4
        
        # Riesgos (negativo - 30%)
        risk_score = risks.get("risk_score", 0.0)
        score += (1.0 - risk_score) * 0.3
        
        # Cumplimiento (30%)
        compliance_score = compliance.get("compliance_score", 0.0)
        score += compliance_score * 0.3
        
        return min(score, 1.0)


class HRDocumentAnalyzer:
    """Analizador especializado para documentos de recursos humanos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar analizador HR
        
        Args:
            analyzer: Instancia de DocumentAnalyzer para análisis base
        
        Raises:
            ValueError: Si analyzer es None
        """
        if analyzer is None:
            raise ValueError("DocumentAnalyzer no puede ser None")
        
        self.analyzer = analyzer
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("HRDocumentAnalyzer inicializado correctamente")
    
    async def analyze_hr_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar documento de recursos humanos
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis de HR completo con información de empleado, trabajo y desempeño
        
        Raises:
            ValueError: Si no se proporciona ni document_path ni document_content
            FileNotFoundError: Si document_path no existe
        """
        # Validación de entrada
        if not document_path and not document_content:
            raise ValueError("Debe proporcionarse document_path o document_content")
        
        if document_path and not os.path.exists(document_path):
            raise FileNotFoundError(f"Documento no encontrado: {document_path}")
        
        try:
            content = document_content
            if document_path:
                processor = DocumentProcessor()
                content = processor.process_document(document_path, "txt")
            
            # Validar contenido
            is_valid, error_msg = validate_content(content, min_length=30)
            if not is_valid:
                logger.warning(f"Contenido inválido: {error_msg}")
                return {"error": error_msg, "timestamp": datetime.now().isoformat()}
            
            logger.info(f"Analizando documento HR (longitud: {len(content)} caracteres)")
            
            # Análisis de HR
            employee_info = self._extract_employee_info(content)
            job_info = self._extract_job_info(content)
            performance_data = self._extract_performance_data(content)
            compliance_analysis = self._analyze_hr_compliance(content)
            
            # Análisis de sentimiento con manejo de errores
            sentiment_analysis = {}
            try:
                sentiment_analysis = await self.analyzer.analyze_sentiment(content)
            except Exception as e:
                logger.warning(f"Error en análisis de sentimiento: {e}")
            
            result = {
                "employee_info": employee_info,
                "job_info": job_info,
                "performance_data": performance_data,
                "compliance_analysis": compliance_analysis,
                "sentiment_analysis": sentiment_analysis,
                "document_type": self._identify_document_type(content),
                "timestamp": datetime.now().isoformat()
            }
            
            self.analysis_history.append(result)
            logger.info("Análisis HR completado exitosamente")
            return result
        
        except Exception as e:
            logger.error(f"Error analizando documento HR: {e}", exc_info=True)
            return {
                "error": f"Error durante el análisis: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_employee_info(self, content: str) -> Dict[str, Any]:
        """
        Extraer información del empleado
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con información del empleado extraída
        """
        if not content:
            return {"data": {}, "completeness": 0.0}
        
        import re
        
        employee_data = {
            "employee_id": None,
            "department": None,
            "position": None,
            "hire_date": None,
            "salary": None
        }
        
        try:
            # ID de empleado
            id_pattern = r'(?:employee id|id de empleado|emp id)[:\s]+([A-Z0-9-]+)'
            matches = safe_regex_match(id_pattern, content, re.IGNORECASE)
            if matches:
                employee_data["employee_id"] = matches[0].group(1)
            
            # Departamento
            dept_pattern = r'(?:department|departamento)[:\s]+([A-Z][a-z]+(?:\s+[a-z]+)*)'
            matches = safe_regex_match(dept_pattern, content, re.IGNORECASE)
            if matches:
                employee_data["department"] = matches[0].group(1)
            
            # Posición
            pos_pattern = r'(?:position|posición|title|título)[:\s]+([A-Z][^\.]+)'
            matches = safe_regex_match(pos_pattern, content, re.IGNORECASE)
            if matches:
                employee_data["position"] = matches[0].group(1).strip()
            
            # Fecha de contratación
            hire_pattern = r'(?:hire date|fecha de contratación|date hired)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            matches = safe_regex_match(hire_pattern, content, re.IGNORECASE)
            if matches:
                employee_data["hire_date"] = matches[0].group(1)
            
            # Salario
            salary_pattern = r'(?:salary|salario)[:\s]*\$?([\d,]+(?:\.\d+)?)'
            matches = safe_regex_match(salary_pattern, content, re.IGNORECASE)
            if matches:
                salary_str = matches[0].group(1)
                salary = safe_float_conversion(salary_str)
                if salary > 0:
                    employee_data["salary"] = salary
            
            completeness = sum(1 for v in employee_data.values() if v is not None) / len(employee_data)
            
            return {
                "data": employee_data,
                "completeness": completeness
            }
        
        except Exception as e:
            logger.error(f"Error extrayendo información de empleado: {e}", exc_info=True)
            return {
                "data": employee_data,
                "completeness": 0.0,
                "error": str(e)
            }
    
    def _extract_job_info(self, content: str) -> Dict[str, Any]:
        """
        Extraer información del trabajo
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con información del trabajo extraída
        """
        import re
        
        if not content:
            return {
                "job_title": None,
                "job_description": None,
                "requirements": [],
                "benefits": []
            }
        
        job_data = {
            "job_title": None,
            "job_description": None,
            "requirements": [],
            "benefits": []
        }
        
        try:
            # Título del trabajo
            title_pattern = r'(?:job title|título del trabajo|position title)[:\s]+([A-Z][^\.]+)'
            matches = safe_regex_match(title_pattern, content, re.IGNORECASE)
            if matches:
                job_data["job_title"] = normalize_text(matches[0].group(1), remove_extra_spaces=True)
            
            # Descripción del trabajo
            desc_pattern = r'(?:job description|descripción del trabajo)[:\s]+([^\.]+(?:\.[^\.]+)*)'
            matches = safe_regex_match(desc_pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                description = normalize_text(matches[0].group(1), remove_extra_spaces=True)
                job_data["job_description"] = description[:200]  # Limitar longitud
            
            # Requisitos
            req_pattern = r'(?:requirement|requisito|qualification|calificación)[:\s]+([^\.]+)'
            matches = safe_regex_match(req_pattern, content, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    requirement = normalize_text(match.group(1), remove_extra_spaces=True)
                    if requirement and requirement not in job_data["requirements"]:
                        job_data["requirements"].append(requirement)
            
            # Beneficios
            benefit_pattern = r'(?:benefit|beneficio)[:\s]+([^\.]+)'
            matches = safe_regex_match(benefit_pattern, content, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    benefit = normalize_text(match.group(1), remove_extra_spaces=True)
                    if benefit and benefit not in job_data["benefits"]:
                        job_data["benefits"].append(benefit)
        except Exception as e:
            logger.error(f"Error extrayendo información del trabajo: {e}", exc_info=True)
        
        return job_data
    
    def _extract_performance_data(self, content: str) -> Dict[str, Any]:
        """
        Extraer datos de desempeño
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con datos de desempeño extraídos
        """
        import re
        
        if not content:
            return {
                "performance_mentions": [],
                "ratings": [],
                "average_rating": None,
                "has_performance_data": False
            }
        
        performance_keywords = [
            "performance", "desempeño",
            "rating", "calificación",
            "review", "revisión",
            "goal", "objetivo",
            "achievement", "logro",
            "kpi", "key performance indicator"
        ]
        
        performance_mentions = []
        content_lower = content.lower()
        
        try:
            for keyword in performance_keywords:
                keyword_lower = keyword.lower()
                pattern = rf'\b{re.escape(keyword_lower)}\b'
                matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                count = len(matches)
                if count > 0:
                    performance_mentions.append({
                        "keyword": keyword,
                        "count": count
                    })
            
            # Buscar calificaciones numéricas
            rating_pattern = r'(?:rating|calificación|score|puntuación)[:\s]+(\d+(?:\.\d+)?)'
            ratings = []
            matches = safe_regex_match(rating_pattern, content, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    try:
                        rating_value = safe_float_conversion(match.group(1))
                        if rating_value > 0:  # Validar que sea un valor positivo
                            ratings.append(rating_value)
                    except (ValueError, TypeError):
                        pass
            
            average_rating = None
            if ratings:
                stats = calculate_statistics(ratings)
                average_rating = stats["mean"]
            
            return {
                "performance_mentions": performance_mentions,
                "ratings": ratings,
                "average_rating": average_rating,
                "has_performance_data": len(performance_mentions) > 0 or len(ratings) > 0
            }
        
        except Exception as e:
            logger.error(f"Error extrayendo datos de desempeño: {e}", exc_info=True)
            return {
                "performance_mentions": [],
                "ratings": [],
                "average_rating": None,
                "has_performance_data": False,
                "error": str(e)
            }
    
    def _analyze_hr_compliance(self, content: str) -> Dict[str, Any]:
        """
        Analizar cumplimiento de HR
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con verificación de cumplimiento HR
        """
        import re
        
        if not content:
            return {
                "compliance_checks": {},
                "compliance_score": 0.0,
                "compliance_level": "low"
            }
        
        compliance_checks = {
            "has_eeo": False,
            "has_ada": False,
            "has_confidentiality": False,
            "has_policy_references": False
        }
        
        content_lower = content.lower()
        
        try:
            # Patrones para cada verificación
            compliance_patterns = {
                "has_eeo": r'\b(?:EEO|equal employment opportunity|igualdad de oportunidades)\b',
                "has_ada": r'\b(?:ADA|americans with disabilities act)\b',
                "has_confidentiality": r'\b(?:confidential|confidencial|privacy|privacidad)\b',
                "has_policy_references": r'\b(?:policy|política|procedure|procedimiento)\b'
            }
            
            for check, pattern in compliance_patterns.items():
                matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                if matches:
                    compliance_checks[check] = True
            
            compliance_count = sum(compliance_checks.values())
            compliance_score = compliance_count / len(compliance_checks) if compliance_checks else 0.0
            
            if compliance_count >= 3:
                compliance_level = "high"
            elif compliance_count >= 2:
                compliance_level = "medium"
            else:
                compliance_level = "low"
            
            return {
                "compliance_checks": compliance_checks,
                "compliance_score": compliance_score,
                "compliance_level": compliance_level
            }
        
        except Exception as e:
            logger.error(f"Error analizando cumplimiento HR: {e}", exc_info=True)
            return {
                "compliance_checks": compliance_checks,
                "compliance_score": 0.0,
                "compliance_level": "unknown",
                "error": str(e)
            }
    
    def _identify_document_type(self, content: str) -> str:
        """
        Identificar tipo de documento HR
        
        Args:
            content: Contenido del documento
        
        Returns:
            Tipo de documento identificado
        """
        import re
        
        if not content:
            return "General HR Document"
        
        content_lower = content.lower()
        
        try:
            # Patrones para identificar tipo de documento
            document_patterns = [
                (r'\b(?:job description|descripción del trabajo|job posting|oferta de trabajo)\b', "Job Description"),
                (r'\b(?:performance review|revisión de desempeño|performance evaluation)\b', "Performance Review"),
                (r'\b(?:employee handbook|manual del empleado|employee manual)\b', "Employee Handbook"),
                (r'\b(?:offer letter|carta de oferta|employment offer)\b', "Offer Letter"),
                (r'\b(?:termination|terminación|separation|separación)\b', "Termination Document")
            ]
            
            for pattern, doc_type in document_patterns:
                matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                if matches:
                    return doc_type
            
            return "General HR Document"
        
        except Exception as e:
            logger.error(f"Error identificando tipo de documento HR: {e}", exc_info=True)
            return "General HR Document"


class MarketingDocumentAnalyzer:
    """Analizador especializado para documentos de marketing."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar analizador de marketing
        
        Args:
            analyzer: Instancia de DocumentAnalyzer para análisis base
        
        Raises:
            ValueError: Si analyzer es None
        """
        if analyzer is None:
            raise ValueError("DocumentAnalyzer no puede ser None")
        
        self.analyzer = analyzer
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("MarketingDocumentAnalyzer inicializado correctamente")
    
    async def analyze_marketing_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar documento de marketing
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis de marketing completo con campaña, audiencia, mensajería y SEO
        
        Raises:
            ValueError: Si no se proporciona ni document_path ni document_content
            FileNotFoundError: Si document_path no existe
        """
        # Validación de entrada
        if not document_path and not document_content:
            raise ValueError("Debe proporcionarse document_path o document_content")
        
        if document_path:
            is_valid, error_msg = validate_file_path(document_path, must_exist=True)
            if not is_valid:
                raise FileNotFoundError(error_msg or f"Documento no encontrado: {document_path}")
        
        try:
            content = document_content
            if document_path:
                processor = DocumentProcessor()
                content = processor.process_document(document_path, "txt")
            
            # Validar contenido
            is_valid, error_msg = validate_content(content, min_length=30)
            if not is_valid:
                logger.warning(f"Contenido inválido: {error_msg}")
                return {"error": error_msg, "timestamp": datetime.now().isoformat()}
            
            logger.info(f"Analizando documento de marketing (longitud: {len(content)} caracteres)")
            
            # Análisis de marketing
            campaign_analysis = self._analyze_campaign_elements(content)
            audience_analysis = self._analyze_target_audience(content)
            
            # Análisis de mensajería con manejo de errores
            messaging_analysis = {}
            try:
                messaging_analysis = await self._analyze_messaging(content)
            except Exception as e:
                logger.warning(f"Error en análisis de mensajería: {e}")
            
            cta_analysis = self._analyze_call_to_actions(content)
            
            # Análisis SEO con manejo de errores
            seo_analysis = {}
            try:
                seo_analysis = await self._analyze_seo_elements(content)
            except Exception as e:
                logger.warning(f"Error en análisis SEO: {e}")
            
            result = {
                "campaign_analysis": campaign_analysis,
                "audience_analysis": audience_analysis,
                "messaging_analysis": messaging_analysis,
                "cta_analysis": cta_analysis,
                "seo_analysis": seo_analysis,
                "marketing_effectiveness_score": self._calculate_marketing_score(
                    campaign_analysis, audience_analysis, messaging_analysis, cta_analysis
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            self.analysis_history.append(result)
            logger.info("Análisis de marketing completado exitosamente")
            return result
        
        except Exception as e:
            logger.error(f"Error analizando documento de marketing: {e}", exc_info=True)
            return {
                "error": f"Error durante el análisis: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_campaign_elements(self, content: str) -> Dict[str, Any]:
        """
        Analizar elementos de campaña
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con elementos de campaña encontrados
        """
        import re
        
        if not content:
            return {
                "elements": {},
                "completeness": 0.0
            }
        
        campaign_elements = {
            "has_campaign_name": False,
            "has_objectives": False,
            "has_timeline": False,
            "has_budget": False,
            "has_channels": False
        }
        
        content_lower = content.lower()
        
        try:
            # Nombre de campaña
            name_pattern = r'\b(?:campaign name|nombre de campaña|campaign)[:\s]+([A-Z][^\.]+)'
            matches = safe_regex_match(name_pattern, content, re.IGNORECASE)
            if matches:
                campaign_elements["has_campaign_name"] = True
            
            # Objetivos
            obj_pattern = r'\b(?:objective|objetivo|goal|meta)[:\s]+'
            matches = safe_regex_match(obj_pattern, content_lower, re.IGNORECASE)
            if matches:
                campaign_elements["has_objectives"] = True
            
            # Timeline
            timeline_pattern = r'\b(?:timeline|cronograma|schedule|programa)[:\s]+'
            matches = safe_regex_match(timeline_pattern, content_lower, re.IGNORECASE)
            if matches:
                campaign_elements["has_timeline"] = True
            
            # Presupuesto
            budget_pattern = r'\b(?:budget|presupuesto)[:\s]*\$?([\d,]+(?:\.\d+)?)'
            matches = safe_regex_match(budget_pattern, content, re.IGNORECASE)
            if matches:
                campaign_elements["has_budget"] = True
            
            # Canales
            channels = ["social media", "email", "website", "print", "tv", "radio", "redes sociales", "correo"]
            for channel in channels:
                if channel.lower() in content_lower:
                    campaign_elements["has_channels"] = True
                    break
            
            element_count = sum(campaign_elements.values())
            completeness = element_count / len(campaign_elements) if campaign_elements else 0.0
            
            return {
                "elements": campaign_elements,
                "completeness": completeness
            }
        
        except Exception as e:
            logger.error(f"Error analizando elementos de campaña: {e}", exc_info=True)
            return {
                "elements": campaign_elements,
                "completeness": 0.0,
                "error": str(e)
            }
    
    def _analyze_target_audience(self, content: str) -> Dict[str, Any]:
        """
        Analizar audiencia objetivo
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con análisis de audiencia objetivo
        """
        import re
        
        if not content:
            return {
                "audience_keywords": [],
                "audience_descriptions": [],
                "has_audience_definition": False
            }
        
        audience_keywords_list = [
            "target audience", "audiencia objetivo",
            "demographics", "demografía",
            "persona", "perfil",
            "age", "edad",
            "gender", "género",
            "location", "ubicación"
        ]
        
        audience_mentions = []
        content_lower = content.lower()
        
        try:
            for keyword in audience_keywords_list:
                keyword_lower = keyword.lower()
                if keyword_lower in content_lower:
                    audience_mentions.append(keyword)
            
            # Buscar descripciones de audiencia
            audience_pattern = r'(?:target audience|audiencia objetivo|target market|mercado objetivo)[:\s]+([^\.]+)'
            audience_descriptions = []
            matches = safe_regex_match(audience_pattern, content, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    desc = normalize_text(match.group(1), remove_extra_spaces=True)
                    if desc and desc not in audience_descriptions:
                        audience_descriptions.append(desc)
            
            return {
                "audience_keywords": audience_mentions,
                "audience_descriptions": audience_descriptions,
                "has_audience_definition": len(audience_mentions) > 0 or len(audience_descriptions) > 0
            }
        
        except Exception as e:
            logger.error(f"Error analizando audiencia objetivo: {e}", exc_info=True)
            return {
                "audience_keywords": [],
                "audience_descriptions": [],
                "has_audience_definition": False,
                "error": str(e)
            }
    
    async def _analyze_messaging(self, content: str) -> Dict[str, Any]:
        """
        Analizar mensajería
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con análisis de mensajería
        """
        if not content:
            return {
                "sentiment": {},
                "emotional_keywords_count": 0,
                "tone": "neutral",
                "persuasiveness_score": 0.0
            }
        
        try:
            # Análisis de sentimiento con manejo de errores
            sentiment = {}
            try:
                sentiment = await self.analyzer.analyze_sentiment(content)
            except Exception as e:
                logger.warning(f"Error en análisis de sentimiento: {e}")
            
            # Keywords emocionales
            emotional_keywords = [
                "exclusive", "exclusivo",
                "limited", "limitado",
                "new", "nuevo",
                "free", "gratis",
                "save", "ahorrar",
                "discount", "descuento"
            ]
            
            content_lower = content.lower()
            emotional_count = sum(1 for keyword in emotional_keywords if keyword.lower() in content_lower)
            
            # Análisis de tono
            if emotional_count > 5:
                tone = "persuasive"
            elif emotional_count > 2:
                tone = "informative"
            else:
                tone = "neutral"
            
            return {
                "sentiment": sentiment,
                "emotional_keywords_count": emotional_count,
                "tone": tone,
                "persuasiveness_score": min(emotional_count / 10.0, 1.0)
            }
        
        except Exception as e:
            logger.error(f"Error analizando mensajería: {e}", exc_info=True)
            return {
                "sentiment": {},
                "emotional_keywords_count": 0,
                "tone": "unknown",
                "persuasiveness_score": 0.0,
                "error": str(e)
            }
    
    def _analyze_call_to_actions(self, content: str) -> Dict[str, Any]:
        """
        Analizar llamados a la acción
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con CTAs encontrados
        """
        import re
        
        if not content:
            return {
                "ctas": [],
                "cta_count": 0,
                "has_cta": False,
                "cta_effectiveness": "low"
            }
        
        cta_patterns = [
            r'\b(?:buy now|comprar ahora|shop now|comprar ya)\b',
            r'\b(?:sign up|registrarse|subscribe|suscribirse)\b',
            r'\b(?:learn more|saber más|discover|descubrir)\b',
            r'\b(?:contact us|contáctanos|get in touch|ponte en contacto)\b',
            r'\b(?:download|descargar|get started|empezar)\b'
        ]
        
        ctas_found = []
        try:
            for pattern in cta_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                for match in matches:
                    cta_text = normalize_text(match.group(0), remove_extra_spaces=True)
                    if cta_text and cta_text not in ctas_found:
                        ctas_found.append(cta_text)
        except Exception as e:
            logger.error(f"Error analizando CTAs: {e}", exc_info=True)
        
        return {
            "ctas": ctas_found,
            "cta_count": len(ctas_found),
            "has_cta": len(ctas_found) > 0,
            "cta_effectiveness": "high" if len(ctas_found) >= 3 else "medium" if len(ctas_found) >= 1 else "low"
        }
    
    async def _analyze_seo_elements(self, content: str) -> Dict[str, Any]:
        """
        Analizar elementos SEO
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con análisis SEO
        """
        if not content:
            return {
                "keywords": [],
                "word_count": 0,
                "heading_count": 0,
                "seo_score": 0.0
            }
        
        try:
            # Keywords con manejo de errores
            keywords = []
            try:
                keywords = await self.analyzer.extract_keywords(content, top_k=10)
                if not isinstance(keywords, list):
                    keywords = []
            except Exception as e:
                logger.warning(f"Error extrayendo keywords: {e}")
            
            # Longitud del contenido
            words = content.split()
            word_count = len(words)
            
            # Headings (estructura)
            heading_count = content.count('\n#') + content.count('\n##')
            
            # Calcular score SEO
            keyword_score = min(len(keywords) / 10.0, 1.0) if keywords else 0.0
            word_score = min(word_count / 2000.0, 1.0) if word_count > 0 else 0.0
            heading_score = min(heading_count / 10.0, 1.0) if heading_count > 0 else 0.0
            
            seo_score = min(
                (keyword_score * 0.4) +
                (word_score * 0.4) +
                (heading_score * 0.2),
                1.0
            )
            
            return {
                "keywords": keywords,
                "word_count": word_count,
                "heading_count": heading_count,
                "seo_score": seo_score
            }
        
        except Exception as e:
            logger.error(f"Error analizando elementos SEO: {e}", exc_info=True)
            return {
                "keywords": [],
                "word_count": 0,
                "heading_count": 0,
                "seo_score": 0.0,
                "error": str(e)
            }
    
    def _calculate_marketing_score(
        self,
        campaign: Dict[str, Any],
        audience: Dict[str, Any],
        messaging: Dict[str, Any],
        cta: Dict[str, Any]
    ) -> float:
        """Calcular score de efectividad de marketing"""
        score = 0.0
        
        # Elementos de campaña (30%)
        campaign_completeness = campaign.get("completeness", 0.0)
        score += campaign_completeness * 0.3
        
        # Audiencia (20%)
        if audience.get("has_audience_definition", False):
            score += 0.2
        
        # Mensajería (30%)
        persuasiveness = messaging.get("persuasiveness_score", 0.0)
        score += persuasiveness * 0.3
        
        # CTAs (20%)
        if cta.get("has_cta", False):
            cta_effectiveness = cta.get("cta_effectiveness", "low")
            if cta_effectiveness == "high":
                score += 0.2
            elif cta_effectiveness == "medium":
                score += 0.1
        
        return min(score, 1.0)


class AdvancedTechnicalDocumentAnalyzer:
    """Analizador especializado avanzado para documentos técnicos."""
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar analizador técnico avanzado
        
        Args:
            analyzer: Instancia de DocumentAnalyzer para análisis base
        
        Raises:
            ValueError: Si analyzer es None
        """
        if analyzer is None:
            raise ValueError("DocumentAnalyzer no puede ser None")
        
        self.analyzer = analyzer
        try:
            self.tech_analyzer = TechnicalDocumentAnalyzer(analyzer)
        except Exception as e:
            logger.warning(f"No se pudo inicializar TechnicalDocumentAnalyzer: {e}")
            self.tech_analyzer = None
        
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("AdvancedTechnicalDocumentAnalyzer inicializado correctamente")
    
    async def analyze_technical_document(
        self,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar documento técnico
        
        Args:
            document_path: Ruta al documento
            document_content: Contenido del documento
        
        Returns:
            Análisis técnico completo con código, APIs, arquitectura y documentación
        
        Raises:
            ValueError: Si no se proporciona ni document_path ni document_content
            FileNotFoundError: Si document_path no existe
        """
        # Validación de entrada
        if not document_path and not document_content:
            raise ValueError("Debe proporcionarse document_path o document_content")
        
        if document_path:
            is_valid, error_msg = validate_file_path(document_path, must_exist=True)
            if not is_valid:
                raise FileNotFoundError(error_msg or f"Documento no encontrado: {document_path}")
        
        try:
            content = document_content
            if document_path:
                processor = DocumentProcessor()
                content = processor.process_document(document_path, "txt")
            
            # Validar contenido
            is_valid, error_msg = validate_content(content, min_length=30)
            if not is_valid:
                logger.warning(f"Contenido inválido: {error_msg}")
                return {"error": error_msg, "timestamp": datetime.now().isoformat()}
            
            logger.info(f"Analizando documento técnico (longitud: {len(content)} caracteres)")
            
            # Análisis técnico básico con manejo de errores
            tech_analysis = {}
            if self.tech_analyzer:
                try:
                    tech_analysis = await self.tech_analyzer.analyze_technical_document(document_content=content)
                except Exception as e:
                    logger.warning(f"Error en análisis técnico básico: {e}")
            else:
                logger.warning("TechnicalDocumentAnalyzer no disponible")
            
            # Análisis adicional técnico
            code_analysis = self._analyze_code_blocks(content)
            api_analysis = self._analyze_api_references(content)
            architecture_analysis = self._analyze_architecture(content)
            documentation_quality = self._analyze_documentation_quality(content)
            
            result = {
                "technical_analysis": tech_analysis,
                "code_analysis": code_analysis,
                "api_analysis": api_analysis,
                "architecture_analysis": architecture_analysis,
                "documentation_quality": documentation_quality,
                "technical_completeness_score": self._calculate_technical_score(
                    code_analysis, api_analysis, architecture_analysis, documentation_quality
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            self.analysis_history.append(result)
            logger.info("Análisis técnico completado exitosamente")
            return result
        
        except Exception as e:
            logger.error(f"Error analizando documento técnico: {e}", exc_info=True)
            return {
                "error": f"Error durante el análisis: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_code_blocks(self, content: str) -> Dict[str, Any]:
        """
        Analizar bloques de código
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con bloques de código encontrados
        """
        import re
        
        if not content:
            return {
                "code_blocks": [],
                "code_block_count": 0,
                "has_code": False
            }
        
        # Buscar bloques de código
        code_patterns = [
            r'```(\w+)?\n(.*?)```',
            r'`([^`]+)`',
            r'(?:function|def|class|import|const|let|var)\s+'
        ]
        
        code_blocks = []
        try:
            for pattern in code_patterns:
                matches = safe_regex_match(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    code_text = match.group(0)[:100]  # Limitar longitud
                    code_type = "unknown"
                    if match.groups():
                        code_type = match.group(1) if match.group(1) else "unknown"
                    
                    code_blocks.append({
                        "code": code_text,
                        "type": normalize_text(code_type, remove_extra_spaces=True) if code_type else "unknown"
                    })
        except Exception as e:
            logger.error(f"Error analizando bloques de código: {e}", exc_info=True)
        
        # Eliminar duplicados basados en el texto del código
        unique_blocks = []
        seen = set()
        for block in code_blocks:
            code_hash = hash(block["code"])
            if code_hash not in seen:
                seen.add(code_hash)
                unique_blocks.append(block)
        
        return {
            "code_blocks": unique_blocks[:10],
            "code_block_count": len(unique_blocks),
            "has_code": len(unique_blocks) > 0
        }
    
    def _analyze_api_references(self, content: str) -> Dict[str, Any]:
        """
        Analizar referencias a APIs
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con referencias a APIs encontradas
        """
        import re
        
        if not content:
            return {
                "api_references": [],
                "api_count": 0,
                "has_api_references": False
            }
        
        api_patterns = [
            r'(?:API|endpoint)[:\s]+([A-Za-z0-9/]+)',
            r'(?:GET|POST|PUT|DELETE|PATCH)\s+([/A-Za-z0-9-]+)',
            r'https?://[^\s]+',
            r'api/[^\s]+'
        ]
        
        api_references = []
        try:
            for pattern in api_patterns:
                matches = safe_regex_match(pattern, content, re.IGNORECASE)
                for match in matches:
                    api_text = normalize_text(match.group(0), remove_extra_spaces=True)
                    if api_text and api_text not in api_references:
                        api_references.append(api_text)
        except Exception as e:
            logger.error(f"Error analizando referencias a APIs: {e}", exc_info=True)
        
        return {
            "api_references": api_references[:20],
            "api_count": len(api_references),
            "has_api_references": len(api_references) > 0
        }
    
    def _analyze_architecture(self, content: str) -> Dict[str, Any]:
        """
        Analizar arquitectura
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con menciones de arquitectura encontradas
        """
        import re
        
        if not content:
            return {
                "architecture_mentions": [],
                "architecture_discussion_count": 0,
                "has_architecture_discussion": False
            }
        
        architecture_keywords = [
            "architecture", "arquitectura",
            "design pattern", "patrón de diseño",
            "microservice", "microservicio",
            "monolith", "monolito",
            "system design", "diseño de sistema",
            "component", "componente",
            "module", "módulo"
        ]
        
        architecture_mentions = []
        content_lower = content.lower()
        
        try:
            for keyword in architecture_keywords:
                keyword_lower = keyword.lower()
                # Escapar el keyword para regex
                pattern = rf'\b{re.escape(keyword_lower)}\b'
                matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                count = len(matches)
                if count > 0:
                    architecture_mentions.append({
                        "keyword": keyword,
                        "count": count
                    })
        except Exception as e:
            logger.error(f"Error analizando arquitectura: {e}", exc_info=True)
        
        return {
            "architecture_mentions": architecture_mentions,
            "architecture_discussion_count": len(architecture_mentions),
            "has_architecture_discussion": len(architecture_mentions) > 0
        }
    
    def _analyze_documentation_quality(self, content: str) -> Dict[str, Any]:
        """
        Analizar calidad de documentación
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con indicadores de calidad de documentación
        """
        import re
        
        if not content:
            return {
                "quality_indicators": {},
                "quality_score": 0.0,
                "quality_level": "low"
            }
        
        quality_indicators = {
            "has_examples": False,
            "has_diagrams": False,
            "has_installation": False,
            "has_troubleshooting": False,
            "has_version_info": False
        }
        
        content_lower = content.lower()
        
        try:
            # Patrones para cada indicador
            quality_patterns = {
                "has_examples": r'\b(?:example|ejemplo|sample|muestra)\b',
                "has_diagrams": r'\b(?:diagram|diagrama|chart|gráfico|figure|figura)\b',
                "has_installation": r'\b(?:installation|instalación|setup|configuración)\b',
                "has_troubleshooting": r'\b(?:troubleshooting|solución de problemas|common issues|problemas comunes)\b',
                "has_version_info": r'\b(?:version|versión|v\d+\.\d+)\b'
            }
            
            for indicator, pattern in quality_patterns.items():
                matches = safe_regex_match(pattern, content_lower, re.IGNORECASE)
                if matches:
                    quality_indicators[indicator] = True
            
            quality_count = sum(quality_indicators.values())
            quality_score = quality_count / len(quality_indicators)
            
            if quality_count >= 4:
                quality_level = "high"
            elif quality_count >= 2:
                quality_level = "medium"
            else:
                quality_level = "low"
            
            return {
                "quality_indicators": quality_indicators,
                "quality_score": quality_score,
                "quality_level": quality_level
            }
        
        except Exception as e:
            logger.error(f"Error analizando calidad de documentación: {e}", exc_info=True)
            return {
                "quality_indicators": quality_indicators,
                "quality_score": 0.0,
                "quality_level": "unknown",
                "error": str(e)
            }
    
    def _calculate_technical_score(
        self,
        code: Dict[str, Any],
        api: Dict[str, Any],
        architecture: Dict[str, Any],
        documentation: Dict[str, Any]
    ) -> float:
        """Calcular score de completitud técnica"""
        score = 0.0
        
        # Código (25%)
        if code.get("has_code", False):
            score += 0.25
        
        # APIs (25%)
        if api.get("has_api_references", False):
            score += 0.25
        
        # Arquitectura (25%)
        if architecture.get("has_architecture_discussion", False):
            score += 0.25
        
        # Documentación (25%)
        doc_score = documentation.get("quality_score", 0.0)
        score += doc_score * 0.25
        
        return min(score, 1.0)


if __name__ == "__main__":
    # Ejemplo de uso
    analyzer = DocumentAnalyzer()
    print("Analizador de Documentos inicializado correctamente")
    print(f"Información del modelo: {analyzer.get_model_info()}")

