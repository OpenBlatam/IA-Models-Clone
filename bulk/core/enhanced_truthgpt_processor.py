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
import json
import uuid
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import redis
import pickle
from collections import defaultdict, deque
import statistics

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

from ..config.openrouter_config import OpenRouterConfig
from ..config.bul_config import BULConfig
from ..utils.document_processor import DocumentProcessor

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
    
    # Enhanced features
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
    
    # Enhanced features
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
class ProcessingMetrics:
    """Advanced processing metrics."""
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
        
        # Add specific keywords
        keywords = ", ".join(rules["keywords"])
        
        # Add structure guidance
        structure = "\n".join([f"- {item}" for item in rules["structure"]])
        
        # Add tone modifiers
        tone_modifiers = ", ".join(rules["tone_modifiers"])
        
        # Build optimized prompt
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
        
        # For high priority requests, use the most reliable model
        if priority == 1:
            best_model = max(self.models, key=lambda m: self.model_stats[m]["success_rate"])
            return best_model
        
        # For normal requests, use round-robin with success rate consideration
        available_models = [m for m in self.models if self.model_stats[m]["success_rate"] > 0.7]
        
        if not available_models:
            available_models = self.models
        
        # Round-robin selection
        model = available_models[self.current_model_index % len(available_models)]
        self.current_model_index += 1
        
        return model
    
    def update_model_stats(self, model: str, success: bool, processing_time: float):
        """Update model statistics."""
        stats = self.model_stats[model]
        stats["requests"] += 1
        stats["last_used"] = datetime.now()
        
        if success:
            # Update success rate
            stats["success_rate"] = (stats["success_rate"] * (stats["requests"] - 1) + 1.0) / stats["requests"]
            # Update average time
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
        
        # Length score
        length = len(content)
        length_score = min(1.0, length / self.quality_criteria["length"]["max"])
        scores.append(length_score * self.quality_criteria["length"]["weight"])
        
        # Structure score (count of headings)
        headings = content.count('#') + content.count('##') + content.count('###')
        structure_score = min(1.0, headings / self.quality_criteria["structure"]["min_sections"])
        scores.append(structure_score * self.quality_criteria["structure"]["weight"])
        
        # Keywords score (basic keyword density)
        words = content.lower().split()
        total_words = len(words)
        if total_words > 0:
            keyword_count = sum(1 for word in words if len(word) > 4)
            keyword_score = min(1.0, keyword_count / (total_words * self.quality_criteria["keywords"]["min_density"]))
            scores.append(keyword_score * self.quality_criteria["keywords"]["weight"])
        else:
            scores.append(0.0)
        
        # Readability score (simple metric based on sentence length)
        sentences = content.split('.')
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            readability_score = max(0.0, 1.0 - (avg_sentence_length - 15) / 20)
            scores.append(readability_score * self.quality_criteria["readability"]["weight"])
        else:
            scores.append(0.0)
        
        return sum(scores)

class EnhancedTruthGPTProcessor:
    """
    Enhanced TruthGPT bulk document processor with advanced features.
    """
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.openrouter_config = OpenRouterConfig()
        self.document_processor = DocumentProcessor(self.config)
        
        # Enhanced components
        self.cache = IntelligentCache()
        self.prompt_optimizer = PromptOptimizer()
        self.model_balancer = ModelLoadBalancer([
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-sonnet"
        ])
        self.quality_assessor = QualityAssessor()
        
        # Processing state
        self.is_running = False
        self.active_requests: Dict[str, EnhancedBulkDocumentRequest] = {}
        self.active_tasks: Dict[str, EnhancedDocumentTask] = {}
        self.completed_tasks: Dict[str, EnhancedDocumentTask] = {}
        self.results: Dict[str, Any] = {}
        
        # Task queue with priority
        self.task_queue: List[EnhancedDocumentTask] = []
        self.metrics = ProcessingMetrics()
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.processing.max_concurrent_tasks)
        
        # Callbacks
        self.on_document_generated: Optional[Callable] = None
        self.on_request_completed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_quality_assessed: Optional[Callable] = None
        
        # Initialize LangChain
        self._setup_langchain()
        
        logger.info("Enhanced TruthGPT Bulk Processor initialized")
    
    def _setup_langchain(self):
        """Setup LangChain with multiple models."""
        if not self.openrouter_config.is_configured():
            raise ValueError("OpenRouter API key not configured")
        
        # Create multiple model instances
        self.models = {}
        for model_name in self.model_balancer.models:
            try:
                self.models[model_name] = ChatOpenAI(
                    model=model_name,
                    openai_api_key=self.openrouter_config.api_key,
                    openai_api_base=self.openrouter_config.base_url,
                    temperature=0.7,
                    max_tokens=4096,
                    headers=self.openrouter_config.get_headers()
                )
                logger.info(f"✅ Model {model_name} initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize model {model_name}: {e}")
        
        if not self.models:
            raise ValueError("No models could be initialized")
        
        # Create output parser
        self.output_parser = StrOutputParser()
        
        # Create enhanced prompt templates
        self._create_enhanced_prompt_templates()
        
        logger.info(f"LangChain configured with {len(self.models)} models")
    
    def _create_enhanced_prompt_templates(self):
        """Create enhanced prompt templates with optimization."""
        
        # Enhanced TruthGPT system prompt
        self.enhanced_system_prompt = """Eres TruthGPT Enhanced, un sistema de IA avanzado especializado en generar documentos empresariales de alta calidad. Tu misión es crear contenido comprehensivo, preciso y valioso.

Principios Fundamentales:
1. VERDAD: Siempre proporciona información precisa, factual y verificable
2. COMPREHENSIVIDAD: Crea documentos detallados que cubran todos los aspectos
3. PRACTICIDAD: Enfócate en insights accionables y aplicaciones del mundo real
4. CALIDAD: Mantén altos estándares en estructura y presentación del contenido
5. CONTINUIDAD: Genera contenido que fluya naturalmente y se construya sobre sí mismo
6. OPTIMIZACIÓN: Adapta el contenido al tipo de documento y audiencia específica

Características Avanzadas:
- Análisis profundo del contexto empresarial
- Adaptación automática al tipo de documento
- Optimización para diferentes audiencias
- Integración de mejores prácticas de la industria
- Referencias cruzadas y evolución del contenido
- Métricas de calidad integradas

Genera documentos que sean inmediatamente útiles y implementables por las empresas."""

        # Enhanced document generation prompt
        self.enhanced_document_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.enhanced_system_prompt),
            HumanMessage(content="""ÁREA DE NEGOCIO: {business_area}
TIPO DE DOCUMENTO: {document_type}
CONSULTA/TEMA: {query}
CONTEXTO: {context}
AUDIENCIA OBJETIVO: {target_audience}
IDIOMA: {language}
TONO: {tone}

Genera un documento comprehensivo de {document_type} para el área de {business_area} basado en la consulta: "{query}"

REQUISITOS AVANZADOS:
1. Crea un documento detallado y profesional
2. Incluye ejemplos prácticos y casos de estudio relevantes
3. Proporciona recomendaciones accionables y específicas
4. Estructura el contenido con encabezados claros y secciones lógicas
5. Asegúrate de que el documento sea comprehensivo y valioso
6. Enfócate en aplicabilidad del mundo real
7. Incluye guías de implementación donde sea relevante
8. Adapta el contenido a la audiencia objetivo: {target_audience}
9. Mantén un tono {tone} consistente
10. Optimiza para el idioma {language}

Haz de este documento un recurso valioso que las empresas puedan usar e implementar inmediatamente.""")
        ])
    
    async def start_enhanced_processing(self):
        """Start the enhanced continuous processing loop."""
        if self.is_running:
            logger.warning("Enhanced processing is already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting Enhanced TruthGPT continuous processing...")
        
        try:
            while self.is_running:
                await self._process_enhanced_tasks()
                await self._update_metrics()
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in enhanced processing: {e}")
            if self.on_error:
                await self._safe_callback(self.on_error, e)
        finally:
            self.is_running = False
            logger.info("Enhanced processing stopped")
    
    async def _process_enhanced_tasks(self):
        """Process tasks with enhanced features."""
        if not self.task_queue:
            return
        
        # Sort by priority and quality requirements
        self.task_queue.sort(key=lambda x: (x.priority, -x.quality_score))
        
        # Process multiple tasks concurrently
        batch_size = min(self.config.processing.max_concurrent_tasks, len(self.task_queue))
        tasks_to_process = self.task_queue[:batch_size]
        
        # Remove tasks from queue
        self.task_queue = self.task_queue[batch_size:]
        
        # Process tasks concurrently
        processing_tasks = []
        for task in tasks_to_process:
            processing_tasks.append(self._process_enhanced_single_task(task))
        
        if processing_tasks:
            await asyncio.gather(*processing_tasks, return_exceptions=True)
    
    async def _process_enhanced_single_task(self, task: EnhancedDocumentTask):
        """Process a single task with enhanced features."""
        start_time = time.time()
        
        try:
            task.status = "processing"
            self.active_tasks[task.id] = task
            
            # Check cache first
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
                # Generate new content
                content = await self._generate_enhanced_content(task)
                if content:
                    task.content = content
                    
                    # Cache the content
                    if hasattr(task, 'enable_caching') and task.enable_caching:
                        await self.cache.set(
                            task.query, task.document_type, task.business_area, content
                        )
                else:
                    raise Exception("Failed to generate content")
            
            # Assess quality
            quality_score = self.quality_assessor.assess_quality(
                task.content, task.document_type, task.business_area
            )
            task.quality_score = quality_score
            
            # Check quality threshold
            request = self.active_requests.get(task.request_id)
            if request and quality_score < request.quality_threshold:
                logger.warning(f"⚠️ Quality below threshold for task {task.id}: {quality_score}")
                # Could implement retry with different approach here
            
            # Update task completion
            task.status = "completed"
            task.completed_at = datetime.now()
            task.processing_time = time.time() - start_time
            
            # Store completed task
            self.completed_tasks[task.id] = task
            
            # Update metrics
            self.metrics.total_documents_generated += 1
            self.metrics.performance_trends.append(task.processing_time)
            
            # Callback for document generated
            if self.on_document_generated:
                await self._safe_callback(self.on_document_generated, task)
            
            # Callback for quality assessment
            if self.on_quality_assessed:
                await self._safe_callback(self.on_quality_assessed, task, quality_score)
            
            logger.info(f"✅ Enhanced document generated: {task.id} - Quality: {quality_score:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Enhanced task failed: {task.id} - {e}")
            task.status = "failed"
            task.error = str(e)
            task.retry_count += 1
            task.processing_time = time.time() - start_time
            
            # Update model stats
            if task.model_used:
                self.model_balancer.update_model_stats(task.model_used, False, task.processing_time)
            
            # Retry logic with exponential backoff
            if task.retry_count < task.max_retries:
                delay = min(60, 2 ** task.retry_count)  # Exponential backoff, max 60s
                await asyncio.sleep(delay)
                self.task_queue.append(task)
                logger.info(f"🔄 Retrying task: {task.id} (attempt {task.retry_count + 1})")
            else:
                self.metrics.total_documents_failed += 1
                self.metrics.error_analysis[type(e).__name__] += 1
                logger.error(f"💥 Task failed permanently: {task.id}")
            
            if self.on_error:
                await self._safe_callback(self.on_error, task, e)
        
        finally:
            # Remove from active tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    async def _generate_enhanced_content(self, task: EnhancedDocumentTask) -> Optional[str]:
        """Generate enhanced document content."""
        try:
            # Select best model
            model_name = self.model_balancer.get_best_model(task.priority)
            model = self.models.get(model_name)
            
            if not model:
                logger.error(f"Model {model_name} not available")
                return None
            
            task.model_used = model_name
            
            # Get request context
            request = self.active_requests.get(task.request_id)
            if not request:
                logger.error(f"Request {task.request_id} not found")
                return None
            
            # Optimize prompt
            base_prompt = self.enhanced_document_prompt
            if request.enable_optimization:
                # Apply prompt optimization
                optimized_prompt = self.prompt_optimizer.optimize_prompt(
                    str(base_prompt), task.document_type, task.business_area,
                    request.target_audience, request.language
                )
                # For now, we'll use the base prompt, but in a real implementation,
                # we'd create a new ChatPromptTemplate with the optimized content
            
            # Create the processing chain
            chain = base_prompt | model | self.output_parser
            
            # Generate content with enhanced context
            content = await chain.ainvoke({
                "business_area": task.business_area,
                "document_type": task.document_type,
                "query": task.query,
                "context": f"Task ID: {task.id}, Priority: {task.priority}, Quality Target: {request.quality_threshold}",
                "target_audience": request.target_audience or "empresarios y profesionales",
                "language": request.language,
                "tone": request.tone
            })
            
            # Update model stats
            self.model_balancer.update_model_stats(model_name, True, 0.0)  # We don't have exact time here
            
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
        """
        request_id = str(uuid.uuid4())
        
        # Create enhanced request
        request = EnhancedBulkDocumentRequest(
            id=request_id,
            query=query,
            document_types=document_types,
            business_areas=business_areas,
            max_documents=max_documents,
            continuous_mode=continuous_mode,
            priority=priority,
            metadata=metadata or {},
            **enhanced_kwargs
        )
        
        # Store request
        self.active_requests[request_id] = request
        self.metrics.total_requests += 1
        
        # Create enhanced tasks
        await self._create_enhanced_tasks(request)
        
        # Start enhanced processing if not running
        if not self.is_running:
            asyncio.create_task(self.start_enhanced_processing())
        
        logger.info(f"🚀 Enhanced bulk request submitted: {request_id} - {max_documents} documents")
        
        return request_id
    
    async def _create_enhanced_tasks(self, request: EnhancedBulkDocumentRequest):
        """Create enhanced tasks for a bulk request."""
        tasks_created = 0
        
        # Create tasks for each combination
        for doc_type in request.document_types:
            for business_area in request.business_areas:
                if tasks_created >= request.max_documents:
                    break
                
                task = EnhancedDocumentTask(
                    id=str(uuid.uuid4()),
                    request_id=request.id,
                    document_type=doc_type,
                    business_area=business_area,
                    query=request.query,
                    priority=request.priority
                )
                
                # Add enhanced features
                task.enable_caching = request.enable_caching
                task.optimization_applied = request.enable_optimization
                
                self.task_queue.append(task)
                tasks_created += 1
        
        # Create variations if enabled
        if request.enable_variations and tasks_created < request.max_documents:
            await self._create_variation_tasks(request, tasks_created)
        
        logger.info(f"📝 Created {tasks_created} enhanced tasks for request {request.id}")
    
    async def _create_variation_tasks(self, request: EnhancedBulkDocumentRequest, current_count: int):
        """Create variation tasks for enhanced generation."""
        remaining = min(request.max_documents - current_count, request.max_variations)
        
        for i in range(remaining):
            doc_type = request.document_types[i % len(request.document_types)]
            business_area = request.business_areas[i % len(request.business_areas)]
            
            task = EnhancedDocumentTask(
                id=str(uuid.uuid4()),
                request_id=request.id,
                document_type=doc_type,
                business_area=business_area,
                query=request.query,
                priority=request.priority
            )
            
            # Mark as variation
            task.variations_generated = 1
            task.metadata = {"variation": True, "variation_number": i + 1}
            
            self.task_queue.append(task)
    
    async def _update_metrics(self):
        """Update enhanced processing metrics."""
        if not self.completed_tasks:
            return
        
        # Calculate average quality score
        quality_scores = [task.quality_score for task in self.completed_tasks.values() 
                         if task.quality_score > 0]
        if quality_scores:
            self.metrics.average_quality_score = statistics.mean(quality_scores)
        
        # Calculate cache hit rate
        cache_hits = sum(1 for task in self.completed_tasks.values() if task.cache_hit)
        total_completed = len(self.completed_tasks)
        if total_completed > 0:
            self.metrics.cache_hit_rate = cache_hits / total_completed
        
        # Calculate average processing time
        processing_times = [task.processing_time for task in self.completed_tasks.values() 
                           if task.processing_time > 0]
        if processing_times:
            self.metrics.average_processing_time = statistics.mean(processing_times)
    
    async def get_enhanced_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get enhanced status of a bulk request."""
        if request_id not in self.active_requests:
            return None
        
        request = self.active_requests[request_id]
        
        # Count tasks with enhanced metrics
        completed_tasks = [task for task in self.completed_tasks.values() 
                          if task.request_id == request_id and task.status == "completed"]
        failed_tasks = [task for task in self.completed_tasks.values() 
                       if task.request_id == request_id and task.status == "failed"]
        active_tasks = [task for task in self.active_tasks.values() 
                       if task.request_id == request_id]
        queued_tasks = [task for task in self.task_queue 
                       if task.request_id == request_id]
        
        # Calculate enhanced metrics
        avg_quality = 0.0
        if completed_tasks:
            avg_quality = sum(task.quality_score for task in completed_tasks) / len(completed_tasks)
        
        cache_hits = sum(1 for task in completed_tasks if task.cache_hit)
        cache_hit_rate = cache_hits / len(completed_tasks) if completed_tasks else 0.0
        
        return {
            "request_id": request_id,
            "status": "active" if request_id in self.active_requests else "completed",
            "query": request.query,
            "max_documents": request.max_documents,
            "documents_generated": len(completed_tasks),
            "documents_failed": len(failed_tasks),
            "active_tasks": len(active_tasks),
            "queued_tasks": len(queued_tasks),
            "progress_percentage": (len(completed_tasks) / request.max_documents) * 100,
            "created_at": request.created_at.isoformat(),
            "continuous_mode": request.continuous_mode,
            # Enhanced metrics
            "average_quality_score": avg_quality,
            "cache_hit_rate": cache_hit_rate,
            "optimization_enabled": request.enable_optimization,
            "variations_enabled": request.enable_variations,
            "target_audience": request.target_audience,
            "language": request.language,
            "tone": request.tone
        }
    
    def get_enhanced_processing_stats(self) -> Dict[str, Any]:
        """Get enhanced processing statistics."""
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
            "active_requests": len(self.active_requests),
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "is_running": self.is_running,
            "cache_enabled": self.cache.enabled,
            "models_available": len(self.models)
        }
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute a callback function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
    
    def set_enhanced_callbacks(self, 
                             document_callback: Optional[Callable] = None,
                             request_callback: Optional[Callable] = None,
                             error_callback: Optional[Callable] = None,
                             quality_callback: Optional[Callable] = None):
        """Set enhanced callbacks."""
        if document_callback:
            self.on_document_generated = document_callback
        if request_callback:
            self.on_request_completed = request_callback
        if error_callback:
            self.on_error = error_callback
        if quality_callback:
            self.on_quality_assessed = quality_callback

# Global enhanced processor instance
_global_enhanced_processor: Optional[EnhancedTruthGPTProcessor] = None

def get_global_enhanced_processor() -> EnhancedTruthGPTProcessor:
    """Get the global enhanced TruthGPT processor instance."""
    global _global_enhanced_processor
    if _global_enhanced_processor is None:
        _global_enhanced_processor = EnhancedTruthGPTProcessor()
    return _global_enhanced_processor



























