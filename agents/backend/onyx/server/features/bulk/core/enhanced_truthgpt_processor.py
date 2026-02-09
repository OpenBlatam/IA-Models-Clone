"""
Enhanced TruthGPT Bulk Document Processor
=========================================

Versión mejorada del procesador TruthGPT con características avanzadas:
- Caching inteligente
- Optimización de prompts
- Balanceo de carga
- Métricas avanzadas
- Recuperación de errores mejorada
- Optimización de modelos AI
"""

import asyncio
import logging
import uuid
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import redis
from collections import defaultdict, deque
import statistics

from langchain_community.callbacks import get_openai_callback

from ..config.openrouter_config import OpenRouterConfig
from ..config.bul_config import BULConfig
from ..utils.document_processor import DocumentProcessor
from .base_processor import BaseBulkProcessor
from .langchain_setup import LangChainSetup
from .task_creator import TaskCreator
from .prompt_templates import PromptTemplates
from .task_error_handler import TaskErrorHandler
from .stats_helper import StatsHelper
from .request_validator import RequestValidator
from .request_submitter import RequestSubmitter
from .request_query_helper import RequestQueryHelper
from .task_queue_helper import TaskQueueHelper
from .metrics_updater import MetricsUpdater
from .processing_loop import ProcessingLoop
from .content_generator import ContentGenerator
from .constants import ENHANCED_RETRY_DELAY_BASE, DEFAULT_LOOP_SLEEP_SECONDS

logger = logging.getLogger(__name__)

@dataclass
class EnhancedBulkDocumentRequest:
    """Enhanced request for bulk document generation with advanced features."""
    id: str
    query: str
    document_types: List[str]
    business_areas: List[str]
    max_documents: int = 100
    continuous_mode: bool = True
    priority: int = 1
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    
    enable_caching: bool = True
    enable_optimization: bool = True
    quality_threshold: float = 0.85
    enable_variations: bool = True
    max_variations: int = 5
    enable_cross_referencing: bool = True
    enable_evolution: bool = True
    target_audience: Optional[str] = None
    language: str = "es"
    tone: str = "professional"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class EnhancedDocumentTask:
    """Enhanced document generation task with advanced tracking."""
    id: str
    request_id: str
    document_type: str
    business_area: str
    query: str
    priority: int
    status: str = "pending"
    content: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    quality_score: float = 0.0
    processing_time: float = 0.0
    model_used: Optional[str] = None
    tokens_used: int = 0
    cost_estimate: float = 0.0
    cache_hit: bool = False
    optimization_applied: bool = False
    variations_generated: int = 0
    cross_references: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.cross_references is None:
            self.cross_references = []

@dataclass
class EnhancedProcessingMetrics:
    """Advanced processing metrics for enhanced processor."""
    total_requests: int = 0
    total_documents_generated: int = 0
    total_documents_failed: int = 0
    average_processing_time: float = 0.0
    average_quality_score: float = 0.0
    cache_hit_rate: float = 0.0
    optimization_success_rate: float = 0.0
    model_usage_stats: Dict[str, int] = None
    cost_tracking: Dict[str, float] = None
    error_analysis: Dict[str, int] = None
    performance_trends: List[float] = None
    
    def __post_init__(self):
        if self.model_usage_stats is None:
            self.model_usage_stats = defaultdict(int)
        if self.cost_tracking is None:
            self.cost_tracking = defaultdict(float)
        if self.error_analysis is None:
            self.error_analysis = defaultdict(int)
        if self.performance_trends is None:
            self.performance_trends = deque(maxlen=100)

class IntelligentCache:
    """Intelligent caching system for document generation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            self.enabled = True
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis cache unavailable: {e}")
            self.enabled = False
            self.memory_cache = {}
    
    def _generate_cache_key(self, query: str, doc_type: str, business_area: str, 
                          language: str = "es", tone: str = "professional") -> str:
        """Generate cache key for document request."""
        key_data = f"{query}:{doc_type}:{business_area}:{language}:{tone}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, query: str, doc_type: str, business_area: str, 
                 language: str = "es", tone: str = "professional") -> Optional[str]:
        """Get cached document content."""
        if not self.enabled:
            cache_key = self._generate_cache_key(query, doc_type, business_area, language, tone)
            return self.memory_cache.get(cache_key)
        
        try:
            cache_key = self._generate_cache_key(query, doc_type, business_area, language, tone)
            cached_content = self.redis_client.get(f"truthgpt:{cache_key}")
            if cached_content:
                logger.info(f"🎯 Cache hit for {doc_type} in {business_area}")
                return cached_content
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def set(self, query: str, doc_type: str, business_area: str, 
                 content: str, language: str = "es", tone: str = "professional", 
                 ttl: int = 3600):
        """Cache document content."""
        if not self.enabled:
            cache_key = self._generate_cache_key(query, doc_type, business_area, language, tone)
            self.memory_cache[cache_key] = content
            return
        
        try:
            cache_key = self._generate_cache_key(query, doc_type, business_area, language, tone)
            self.redis_client.setex(f"truthgpt:{cache_key}", ttl, content)
            logger.info(f"💾 Cached {doc_type} for {business_area}")
        except Exception as e:
            logger.error(f"Cache set error: {e}")

class PromptOptimizer:
    """Advanced prompt optimization system."""
    
    def __init__(self):
        self.optimization_rules = {
            "business_plan": {
                "keywords": ["estrategia", "objetivos", "mercado", "financiero", "competencia"],
                "structure": ["resumen ejecutivo", "análisis de mercado", "plan financiero", "estrategia de marketing"],
                "tone_modifiers": ["profesional", "convincente", "detallado"]
            },
            "marketing_strategy": {
                "keywords": ["audiencia", "canales", "contenido", "métricas", "ROI"],
                "structure": ["análisis de audiencia", "estrategia de canales", "plan de contenido", "métricas y KPIs"],
                "tone_modifiers": ["creativo", "estratégico", "orientado a resultados"]
            },
            "technical_documentation": {
                "keywords": ["implementación", "arquitectura", "código", "configuración", "troubleshooting"],
                "structure": ["introducción", "arquitectura", "implementación", "ejemplos", "solución de problemas"],
                "tone_modifiers": ["técnico", "preciso", "completo"]
            }
        }
    
    def optimize_prompt(self, base_prompt: str, doc_type: str, business_area: str, 
                       target_audience: Optional[str] = None, language: str = "es") -> str:
        """Optimize prompt based on document type and context."""
        
        if doc_type not in self.optimization_rules:
            return base_prompt
        
        rules = self.optimization_rules[doc_type]
        
        keywords = ", ".join(rules["keywords"])
        structure = "\n".join([f"- {item}" for item in rules["structure"]])
        tone_modifiers = ", ".join(rules["tone_modifiers"])
        
        optimized_prompt = f"""{base_prompt}

OPTIMIZACIONES ESPECÍFICAS PARA {doc_type.upper()}:
- Palabras clave importantes: {keywords}
- Estructura recomendada:
{structure}
- Tono: {tone_modifiers}
- Idioma: {language}"""

        if target_audience:
            optimized_prompt += f"\n- Audiencia objetivo: {target_audience}"
        
        return optimized_prompt

class ModelLoadBalancer:
    """Intelligent model load balancer."""
    
    def __init__(self, models: List[str]):
        self.models = models
        self.model_stats = defaultdict(lambda: {
            "requests": 0,
            "success_rate": 1.0,
            "average_time": 0.0,
            "last_used": datetime.now(),
            "errors": 0
        })
        self.current_model_index = 0
    
    def get_best_model(self, priority: int = 1) -> str:
        """Get the best model based on current stats and priority."""
        
        if priority == 1:
            best_model = max(self.models, key=lambda m: self.model_stats[m]["success_rate"])
            return best_model
        
        available_models = [m for m in self.models if self.model_stats[m]["success_rate"] > 0.7]
        
        if not available_models:
            available_models = self.models
        
        model = available_models[self.current_model_index % len(available_models)]
        self.current_model_index += 1
        
        return model
    
    def update_model_stats(self, model: str, success: bool, processing_time: float):
        """Update model statistics."""
        stats = self.model_stats[model]
        stats["requests"] += 1
        stats["last_used"] = datetime.now()
        
        if success:
            stats["success_rate"] = (stats["success_rate"] * (stats["requests"] - 1) + 1.0) / stats["requests"]
            stats["average_time"] = (stats["average_time"] * (stats["requests"] - 1) + processing_time) / stats["requests"]
        else:
            stats["errors"] += 1
            stats["success_rate"] = (stats["success_rate"] * (stats["requests"] - 1) + 0.0) / stats["requests"]

class QualityAssessor:
    """Document quality assessment system."""
    
    def __init__(self):
        self.quality_criteria = {
            "length": {"min": 500, "max": 10000, "weight": 0.2},
            "structure": {"min_sections": 3, "weight": 0.3},
            "keywords": {"min_density": 0.02, "weight": 0.2},
            "readability": {"min_score": 0.6, "weight": 0.3}
        }
    
    def assess_quality(self, content: str, doc_type: str, business_area: str) -> float:
        """Assess document quality and return score (0-1)."""
        
        if not content or len(content.strip()) < 100:
            return 0.0
        
        scores = []
        
        length = len(content)
        length_score = min(1.0, length / self.quality_criteria["length"]["max"])
        scores.append(length_score * self.quality_criteria["length"]["weight"])
        
        headings = content.count('#') + content.count('##') + content.count('###')
        structure_score = min(1.0, headings / self.quality_criteria["structure"]["min_sections"])
        scores.append(structure_score * self.quality_criteria["structure"]["weight"])
        
        words = content.lower().split()
        total_words = len(words)
        if total_words > 0:
            keyword_count = sum(1 for word in words if len(word) > 4)
            keyword_score = min(1.0, keyword_count / (total_words * self.quality_criteria["keywords"]["min_density"]))
            scores.append(keyword_score * self.quality_criteria["keywords"]["weight"])
        else:
            scores.append(0.0)
        
        sentences = content.split('.')
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            readability_score = max(0.0, 1.0 - (avg_sentence_length - 15) / 20)
            scores.append(readability_score * self.quality_criteria["readability"]["weight"])
        else:
            scores.append(0.0)
        
        return sum(scores)

class EnhancedTruthGPTProcessor(BaseBulkProcessor):
    """
    Enhanced TruthGPT bulk document processor with advanced features.
    """
    
    def __init__(self, config: Optional[BULConfig] = None):
        super().__init__(config)
        
        self.cache = IntelligentCache()
        self.prompt_optimizer = PromptOptimizer()
        self.model_balancer = ModelLoadBalancer([
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-sonnet"
        ])
        self.quality_assessor = QualityAssessor()
        
        self.active_requests: Dict[str, EnhancedBulkDocumentRequest] = {}
        self.active_tasks: Dict[str, EnhancedDocumentTask] = {}
        self.completed_tasks: Dict[str, EnhancedDocumentTask] = {}
        self.results: Dict[str, Any] = {}
        
        self.task_queue: List[EnhancedDocumentTask] = []
        self.metrics = EnhancedProcessingMetrics()
        
        self.on_quality_assessed: Optional[Callable] = None
        
        self._setup_langchain()
        
        logger.info("Enhanced TruthGPT Bulk Processor initialized")
    
    def _setup_langchain(self):
        """Setup LangChain with multiple models."""
        self._setup_langchain_base()
        self.models = LangChainSetup.create_multiple_llms(
            self.openrouter_config,
            self.model_balancer.models
        )
        self._create_enhanced_prompt_templates()
        
        logger.info(f"LangChain configured with {len(self.models)} models")
    
    def _create_enhanced_prompt_templates(self):
        """Create enhanced prompt templates with optimization."""
        self.enhanced_document_prompt = PromptTemplates.create_enhanced_document_prompt()
    
    async def start_enhanced_processing(self):
        """Start the enhanced continuous processing loop."""
        await ProcessingLoop.run_processing_loop(
            processor=self,
            process_func=self._process_enhanced_tasks,
            update_func=self._update_metrics,
            sleep_seconds=DEFAULT_LOOP_SLEEP_SECONDS,
            loop_name="Enhanced TruthGPT continuous processing"
        )
    
    async def _process_enhanced_tasks(self):
        """Process tasks with enhanced features."""
        if not self.task_queue:
            return
        
        def sort_key(task: Any) -> tuple:
            """Sort by priority (ascending) and quality_score (descending)."""
            quality = getattr(task, 'quality_score', 0.0)
            return (task.priority, -quality)
        
        tasks_to_process, self.task_queue = TaskQueueHelper.get_batch_from_queue(
            task_queue=self.task_queue,
            batch_size=self.config.processing.max_concurrent_tasks,
            sort_key=sort_key
        )
        
        if tasks_to_process:
            processing_tasks = [
                self._process_enhanced_single_task(task) for task in tasks_to_process
            ]
            await asyncio.gather(*processing_tasks, return_exceptions=True)
    
    async def _process_enhanced_single_task(self, task: EnhancedDocumentTask):
        """Process a single task with enhanced features."""
        start_time = time.time()
        
        try:
            task.status = "processing"
            self.active_tasks[task.id] = task
            
            cached_content = None
            if hasattr(task, 'enable_caching') and task.enable_caching:
                cached_content = await self.cache.get(
                    task.query, task.document_type, task.business_area
                )
            
            if cached_content:
                task.content = cached_content
                task.cache_hit = True
                logger.info(f"🎯 Cache hit for task {task.id}")
            else:
                content = await self._generate_enhanced_content(task)
                if content:
                    task.content = content
                    
                    if hasattr(task, 'enable_caching') and task.enable_caching:
                        await self.cache.set(
                            task.query, task.document_type, task.business_area, content
                        )
                else:
                    raise Exception("Failed to generate content")
            
            quality_score = self.quality_assessor.assess_quality(
                task.content, task.document_type, task.business_area
            )
            task.quality_score = quality_score
            
            request = self.active_requests.get(task.request_id)
            if request and quality_score < request.quality_threshold:
                logger.warning(f"⚠️ Quality below threshold for task {task.id}: {quality_score}")
            
            task.processing_time = time.time() - start_time
            
            TaskErrorHandler.mark_task_completed(
                task=task,
                completed_tasks=self.completed_tasks,
                processing_stats=None
            )
            
            self.metrics.total_documents_generated += 1
            self.metrics.performance_trends.append(task.processing_time)
            
            await self.callbacks.execute_document_callback(task, None)
            
            if self.on_quality_assessed:
                await self.callbacks.execute_callback(self.on_quality_assessed, task, quality_score)
            
            logger.info(f"✅ Enhanced document generated: {task.id} - Quality: {quality_score:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Enhanced task failed: {task.id} - {e}")
            task.processing_time = time.time() - start_time
            
            if task.model_used:
                self.model_balancer.update_model_stats(task.model_used, False, task.processing_time)
            
            async def error_callback(t, err):
                if self.on_error:
                    await self.callbacks.execute_error_callback(t, err)
            
            should_retry = await TaskErrorHandler.handle_task_error(
                task=task,
                error=e,
                task_queue=self.task_queue,
                max_retries=task.max_retries,
                retry_delay_base=ENHANCED_RETRY_DELAY_BASE,
                on_error_callback=error_callback
            )
            
            if not should_retry:
                TaskErrorHandler.mark_task_failed(
                    task=task,
                    completed_tasks=self.completed_tasks,
                    processing_stats=None,
                    error_analysis=self.metrics.error_analysis,
                    error=e
                )
        
        finally:
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    async def _generate_enhanced_content(self, task: EnhancedDocumentTask) -> Optional[str]:
        """Generate enhanced document content."""
        try:
            model_name = self.model_balancer.get_best_model(task.priority)
            model = self.models.get(model_name)
            
            if not model:
                logger.error(f"Model {model_name} not available")
                return None
            
            task.model_used = model_name
            
            request = self.active_requests.get(task.request_id)
            if not request:
                logger.error(f"Request {task.request_id} not found")
                return None
            
            base_prompt = self.enhanced_document_prompt
            if request.enable_optimization:
                optimized_prompt = self.prompt_optimizer.optimize_prompt(
                    str(base_prompt), task.document_type, task.business_area,
                    request.target_audience, request.language
                )
            
            context = ContentGenerator.build_task_context(
                task_id=task.id,
                priority=task.priority,
                quality_target=request.quality_threshold
            )
            
            content = await ContentGenerator.generate_content(
                prompt_template=base_prompt,
                llm=model,
                output_parser=self.output_parser,
                task_data={
                    "business_area": task.business_area,
                    "document_type": task.document_type,
                    "query": task.query,
                    "context": context,
                    "target_audience": request.target_audience or "empresarios y profesionales",
                    "language": request.language,
                    "tone": request.tone
                }
            )
            
            self.model_balancer.update_model_stats(model_name, True, 0.0)
            
            return content
            
        except Exception as e:
            logger.error(f"Enhanced content generation failed for task {task.id}: {e}")
            return None
    
    async def submit_enhanced_bulk_request(self, 
                                         query: str,
                                         document_types: List[str],
                                         business_areas: List[str],
                                         max_documents: int = 100,
                                         continuous_mode: bool = True,
                                         priority: int = 1,
                                         metadata: Optional[Dict[str, Any]] = None,
                                         **enhanced_kwargs) -> str:
        """
        Submit an enhanced bulk document generation request.
        
        Raises:
            ValueError: If validation fails
        """
        is_valid, error_msg, request_id = RequestSubmitter.validate_and_prepare_request(
            query=query,
            document_types=document_types,
            business_areas=business_areas,
            max_documents=max_documents,
            priority=priority
        )
        
        if not is_valid:
            raise ValueError(error_msg)
        
        request = RequestSubmitter.create_request_object(
            request_class=EnhancedBulkDocumentRequest,
            request_id=request_id,
            query=query,
            document_types=document_types,
            business_areas=business_areas,
            max_documents=max_documents,
            continuous_mode=continuous_mode,
            priority=priority,
            metadata=metadata,
            **enhanced_kwargs
        )
        
        def update_stats():
            self.metrics.total_requests += 1
        
        await RequestSubmitter.register_request_and_start_processing(
            request=request,
            active_requests=self.active_requests,
            stats_updater=update_stats,
            task_creator=self._create_enhanced_tasks,
            processor=self,
            start_processing_func=self.start_enhanced_processing
        )
        
        logger.info(f"🚀 Enhanced bulk request submitted: {request_id} - {max_documents} documents")
        
        return request_id
    
    async def _create_enhanced_tasks(self, request: EnhancedBulkDocumentRequest):
        """Create enhanced tasks for a bulk request."""
        initial_tasks = TaskCreator.create_initial_tasks(
            request_id=request.id,
            query=request.query,
            document_types=request.document_types,
            business_areas=request.business_areas,
            max_documents=request.max_documents,
            priority=request.priority,
            task_class=EnhancedDocumentTask
        )
        
        for task in initial_tasks:
            task.enable_caching = request.enable_caching
            task.optimization_applied = request.enable_optimization
        
        self.task_queue.extend(initial_tasks)
        tasks_created = len(initial_tasks)
        
        if request.enable_variations and tasks_created < request.max_documents:
            remaining = min(request.max_documents - tasks_created, request.max_variations)
            variation_tasks = TaskCreator.create_additional_tasks(
                request_id=request.id,
                query=request.query,
                document_types=request.document_types,
                business_areas=request.business_areas,
                current_count=tasks_created,
                max_documents=tasks_created + remaining,
                priority=request.priority,
                task_class=EnhancedDocumentTask
            )
            
            for i, task in enumerate(variation_tasks):
                task.variations_generated = 1
                if not hasattr(task, 'metadata') or task.metadata is None:
                    task.metadata = {}
                task.metadata["variation"] = True
                task.metadata["variation_number"] = i + 1
            
            self.task_queue.extend(variation_tasks)
            tasks_created += len(variation_tasks)
        
        logger.info(f"📝 Created {tasks_created} enhanced tasks for request {request.id}")
    
    async def _update_metrics(self):
        """Update enhanced processing metrics."""
        MetricsUpdater.update_enhanced_metrics(
            metrics=self.metrics,
            completed_tasks=self.completed_tasks
        )
    
    async def get_enhanced_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get enhanced status of a bulk request."""
        request = RequestQueryHelper.find_request(request_id, self.active_requests)
        if not request:
            return None
        
        status = StatsHelper.get_request_status_base(
            request_id=request_id,
            request=request,
            completed_tasks=self.completed_tasks,
            active_tasks=self.active_tasks,
            task_queue=self.task_queue
        )
        
        metrics = RequestQueryHelper.calculate_request_metrics(
            request_id=request_id,
            completed_tasks=self.completed_tasks,
            active_tasks=self.active_tasks,
            task_queue=self.task_queue
        )
        
        status.update({
            "continuous_mode": request.continuous_mode,
            "average_quality_score": metrics.get("average_quality_score", 0.0),
            "cache_hit_rate": metrics.get("cache_hit_rate", 0.0),
            "optimization_enabled": request.enable_optimization,
            "variations_enabled": request.enable_variations,
            "target_audience": request.target_audience,
            "language": request.language,
            "tone": request.tone
        })
        
        return status
    
    def get_enhanced_processing_stats(self) -> Dict[str, Any]:
        """Get enhanced processing statistics."""
        base_stats = StatsHelper.get_base_stats(
            active_requests=self.active_requests,
            active_tasks=self.active_tasks,
            task_queue=self.task_queue,
            completed_tasks=self.completed_tasks,
            is_running=self.is_running
        )
        
        return {
            "total_requests": self.metrics.total_requests,
            "total_documents_generated": self.metrics.total_documents_generated,
            "total_documents_failed": self.metrics.total_documents_failed,
            "average_processing_time": self.metrics.average_processing_time,
            "average_quality_score": self.metrics.average_quality_score,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "optimization_success_rate": self.metrics.optimization_success_rate,
            "model_usage_stats": dict(self.metrics.model_usage_stats),
            "error_analysis": dict(self.metrics.error_analysis),
            **base_stats,
            "cache_enabled": self.cache.enabled,
            "models_available": len(self.models)
        }
    
    def set_enhanced_callbacks(self, 
                             document_callback: Optional[Callable] = None,
                             request_callback: Optional[Callable] = None,
                             error_callback: Optional[Callable] = None,
                             quality_callback: Optional[Callable] = None):
        """Set enhanced callbacks."""
        if document_callback:
            self.callbacks.set_document_callback(document_callback)
        if request_callback:
            self.callbacks.set_task_callback(request_callback)
        if error_callback:
            self.callbacks.set_error_callback(error_callback)
        if quality_callback:
            self.on_quality_assessed = quality_callback

_global_enhanced_processor: Optional[EnhancedTruthGPTProcessor] = None

def get_global_enhanced_processor() -> EnhancedTruthGPTProcessor:
    """Get the global enhanced TruthGPT processor instance."""
    global _global_enhanced_processor
    if _global_enhanced_processor is None:
        _global_enhanced_processor = EnhancedTruthGPTProcessor()
    return _global_enhanced_processor





























