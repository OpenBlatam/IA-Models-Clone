"""
BUL (Business Universal Language) Engine
========================================

A realistic and functional document generation system for SMEs using OpenRouter and LangChain.
This system generates comprehensive business documents based on queries and business areas.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
import json
import aiohttp
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import httpx
import time
from contextlib import asynccontextmanager
from ..utils import get_cache_manager, cached, get_logger, monitor_performance

# Configure logging
logger = get_logger(__name__)

class BusinessArea(Enum):
    """Business areas for SME document generation"""
    MARKETING = "marketing"
    SALES = "sales"
    OPERATIONS = "operations"
    HR = "hr"
    FINANCE = "finance"
    LEGAL = "legal"
    TECHNICAL = "technical"
    CONTENT = "content"
    STRATEGY = "strategy"
    CUSTOMER_SERVICE = "customer_service"

class DocumentType(Enum):
    """Types of documents that can be generated"""
    BUSINESS_PLAN = "business_plan"
    MARKETING_STRATEGY = "marketing_strategy"
    SALES_PROPOSAL = "sales_proposal"
    OPERATIONAL_MANUAL = "operational_manual"
    HR_POLICY = "hr_policy"
    FINANCIAL_REPORT = "financial_report"
    LEGAL_CONTRACT = "legal_contract"
    TECHNICAL_SPECIFICATION = "technical_specification"
    CONTENT_STRATEGY = "content_strategy"
    STRATEGIC_PLAN = "strategic_plan"
    CUSTOMER_SERVICE_GUIDE = "customer_service_guide"

@dataclass
class DocumentRequest:
    """Request for document generation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    business_area: BusinessArea = BusinessArea.STRATEGY
    document_type: DocumentType = DocumentType.BUSINESS_PLAN
    company_name: str = ""
    industry: str = ""
    company_size: str = ""
    target_audience: str = ""
    language: str = "es"
    format: str = "markdown"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentResponse:
    """Response from document generation"""
    id: str = ""
    request_id: str = ""
    content: str = ""
    title: str = ""
    summary: str = ""
    business_area: BusinessArea = BusinessArea.STRATEGY
    document_type: DocumentType = DocumentType.BUSINESS_PLAN
    word_count: int = 0
    processing_time: float = 0.0
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BULEngine:
    """
    Enhanced BUL Engine - Main document generation system
    
    This engine processes business queries and generates comprehensive documents
    for SMEs using OpenRouter and LangChain integration with advanced features.
    """
    
    def __init__(self, openrouter_api_key: str, openai_api_key: str = None):
        self.openrouter_api_key = openrouter_api_key
        self.openai_api_key = openai_api_key
        self.llm = None
        self.memory = ConversationBufferMemory()
        self.session = None
        self.http_client = None
        self.cache = get_cache_manager()
        self.is_initialized = False
        self.start_time = time.time()
        
        # Enhanced document generation statistics
        self.stats = {
            "documents_generated": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "business_areas_used": {},
            "document_types_generated": {},
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls_successful": 0,
            "api_calls_failed": 0,
            "fallback_usage": 0,
            "error_count": 0,
            "last_error": None,
            "uptime": 0.0
        }
        
        # Performance tracking
        self.performance_metrics = {
            "avg_response_time": 0.0,
            "max_response_time": 0.0,
            "min_response_time": float('inf'),
            "response_times": []
        }
        
        # Rate limiting and retry configuration
        self.rate_limit_config = {
            "max_requests_per_minute": 60,
            "max_requests_per_hour": 1000,
            "retry_attempts": 3,
            "retry_delay": 1.0
        }
        
        logger.info("Enhanced BUL Engine initialized")
    
    async def initialize(self) -> bool:
        """Initialize the BUL engine with enhanced API connections and error handling"""
        try:
            # Initialize HTTP clients with enhanced configuration
            timeout = httpx.Timeout(30.0, connect=10.0, read=60.0, write=30.0)
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
            
            self.http_client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                headers={
                    "User-Agent": "BUL-Engine/2.0.0",
                    "Accept": "application/json"
                }
            )
            
            # Initialize aiohttp session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60, connect=10),
                connector=connector,
                headers={
                    "User-Agent": "BUL-Engine/2.0.0"
                }
            )
            
            # Initialize LangChain with OpenAI (fallback)
            if self.openai_api_key:
                self.llm = ChatOpenAI(
                    openai_api_key=self.openai_api_key,
                    temperature=0.7,
                    max_tokens=4000,
                    model_name="gpt-3.5-turbo",
                    request_timeout=60
                )
            
            # Test API connectivity
            await self._test_api_connectivity()
            
            self.is_initialized = True
            logger.info("Enhanced BUL Engine fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BUL Engine: {e}")
            self.stats["error_count"] += 1
            self.stats["last_error"] = str(e)
            return False
    
    async def _test_api_connectivity(self):
        """Test API connectivity and configuration"""
        try:
            # Test OpenRouter API
            test_headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            test_payload = {
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            
            response = await self.http_client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=test_headers,
                json=test_payload
            )
            
            if response.status_code == 200:
                logger.info("OpenRouter API connectivity test passed")
            else:
                logger.warning(f"OpenRouter API test returned status: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"API connectivity test failed: {e}")
            # Don't fail initialization for connectivity test
    
    @monitor_performance("document_generation")
    async def generate_document(self, request: DocumentRequest) -> DocumentResponse:
        """Generate a document based on the request with enhanced error handling"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Check cache first
            cache_key = f"doc_gen:{hash(request.query + str(request.business_area.value) + str(request.document_type.value))}"
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                logger.info(f"Document served from cache: {request_id}")
                return DocumentResponse(**cached_result)
            
            self.stats["cache_misses"] += 1
            
            # Analyze the query to determine business area and document type
            analysis = await self._analyze_query_with_retry(request.query)
            
            # Update request with analysis results
            if analysis.get("business_area"):
                request.business_area = BusinessArea(analysis["business_area"])
            if analysis.get("document_type"):
                request.document_type = DocumentType(analysis["document_type"])
            
            # Generate document content with retry logic
            content = await self._generate_content_with_retry(request)
            
            # Create response
            processing_time = time.time() - start_time
            response = DocumentResponse(
                id=request_id,
                request_id=request.id,
                content=content,
                title=self._generate_title(request, content),
                summary=self._generate_summary(content),
                business_area=request.business_area,
                document_type=request.document_type,
                word_count=len(content.split()),
                processing_time=processing_time,
                confidence_score=analysis.get("confidence", 0.8),
                metadata={
                    "analysis": analysis,
                    "generation_method": "openrouter_langchain",
                    "model_used": "gpt-4",
                    "language": request.language,
                    "cache_key": cache_key,
                    "processing_time_breakdown": {
                        "analysis_time": analysis.get("processing_time", 0),
                        "generation_time": analysis.get("generation_time", 0)
                    }
                }
            )
            
            # Cache the result
            await self.cache.set(cache_key, response.__dict__, ttl=1800)
            
            # Update statistics and performance metrics
            self._update_stats(response)
            self._update_performance_metrics(processing_time)
            
            logger.info(f"Document generated successfully: {response.id} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["error_count"] += 1
            self.stats["last_error"] = str(e)
            logger.error(f"Failed to generate document {request_id}: {e} (took {processing_time:.2f}s)")
            raise
    
    async def _analyze_query_with_retry(self, query: str) -> Dict[str, Any]:
        """Analyze query with retry logic"""
        for attempt in range(self.rate_limit_config["retry_attempts"]):
            try:
                return await self._analyze_query(query)
            except Exception as e:
                if attempt == self.rate_limit_config["retry_attempts"] - 1:
                    raise
                await asyncio.sleep(self.rate_limit_config["retry_delay"] * (attempt + 1))
                logger.warning(f"Query analysis attempt {attempt + 1} failed, retrying: {e}")
    
    async def _generate_content_with_retry(self, request: DocumentRequest) -> str:
        """Generate content with retry logic"""
        for attempt in range(self.rate_limit_config["retry_attempts"]):
            try:
                return await self._generate_content(request)
            except Exception as e:
                if attempt == self.rate_limit_config["retry_attempts"] - 1:
                    # Try fallback if all retries failed
                    if self.llm:
                        logger.warning("Using OpenAI fallback after retries failed")
                        self.stats["fallback_usage"] += 1
                        return await self._fallback_to_openai(self._create_prompt(request))
                    raise
                await asyncio.sleep(self.rate_limit_config["retry_delay"] * (attempt + 1))
                logger.warning(f"Content generation attempt {attempt + 1} failed, retrying: {e}")
    
    @cached(ttl=1800)  # Cache for 30 minutes
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine business area and document type"""
        # Simple keyword-based analysis (can be enhanced with ML)
        query_lower = query.lower()
        
        # Business area detection
        business_area = BusinessArea.STRATEGY
        if any(word in query_lower for word in ["marketing", "publicidad", "promoción", "ventas"]):
            business_area = BusinessArea.MARKETING
        elif any(word in query_lower for word in ["ventas", "sales", "vender", "cliente"]):
            business_area = BusinessArea.SALES
        elif any(word in query_lower for word in ["operaciones", "proceso", "producción", "logística"]):
            business_area = BusinessArea.OPERATIONS
        elif any(word in query_lower for word in ["recursos humanos", "hr", "empleados", "personal"]):
            business_area = BusinessArea.HR
        elif any(word in query_lower for word in ["finanzas", "financiero", "presupuesto", "dinero"]):
            business_area = BusinessArea.FINANCE
        elif any(word in query_lower for word in ["legal", "contrato", "ley", "jurídico"]):
            business_area = BusinessArea.LEGAL
        elif any(word in query_lower for word in ["técnico", "tecnología", "software", "sistema"]):
            business_area = BusinessArea.TECHNICAL
        elif any(word in query_lower for word in ["contenido", "blog", "artículo", "redes sociales"]):
            business_area = BusinessArea.CONTENT
        elif any(word in query_lower for word in ["atención al cliente", "soporte", "servicio"]):
            business_area = BusinessArea.CUSTOMER_SERVICE
        
        # Document type detection
        document_type = DocumentType.BUSINESS_PLAN
        if any(word in query_lower for word in ["plan de negocio", "business plan", "estrategia"]):
            document_type = DocumentType.BUSINESS_PLAN
        elif any(word in query_lower for word in ["estrategia de marketing", "marketing strategy"]):
            document_type = DocumentType.MARKETING_STRATEGY
        elif any(word in query_lower for word in ["propuesta de ventas", "sales proposal"]):
            document_type = DocumentType.SALES_PROPOSAL
        elif any(word in query_lower for word in ["manual operativo", "operational manual"]):
            document_type = DocumentType.OPERATIONAL_MANUAL
        elif any(word in query_lower for word in ["política de hr", "hr policy"]):
            document_type = DocumentType.HR_POLICY
        elif any(word in query_lower for word in ["reporte financiero", "financial report"]):
            document_type = DocumentType.FINANCIAL_REPORT
        elif any(word in query_lower for word in ["contrato", "contract", "legal"]):
            document_type = DocumentType.LEGAL_CONTRACT
        elif any(word in query_lower for word in ["especificación técnica", "technical spec"]):
            document_type = DocumentType.TECHNICAL_SPECIFICATION
        elif any(word in query_lower for word in ["estrategia de contenido", "content strategy"]):
            document_type = DocumentType.CONTENT_STRATEGY
        elif any(word in query_lower for word in ["plan estratégico", "strategic plan"]):
            document_type = DocumentType.STRATEGIC_PLAN
        elif any(word in query_lower for word in ["guía de atención", "customer service guide"]):
            document_type = DocumentType.CUSTOMER_SERVICE_GUIDE
        
        return {
            "business_area": business_area.value,
            "document_type": document_type.value,
            "confidence": 0.8,
            "keywords_found": len([word for word in query_lower.split() if len(word) > 3])
        }
    
    async def _generate_content(self, request: DocumentRequest) -> str:
        """Generate the actual document content"""
        # Create a comprehensive prompt based on the request
        prompt = self._create_prompt(request)
        
        # Use OpenRouter API for content generation
        content = await self._call_openrouter_api(prompt, request)
        
        return content
    
    def _create_prompt(self, request: DocumentRequest) -> str:
        """Create a detailed prompt for document generation"""
        prompt = f"""
        Eres un experto consultor de negocios especializado en PYMEs. 
        Genera un documento profesional y completo basado en la siguiente información:

        CONSULTA: {request.query}
        ÁREA DE NEGOCIO: {request.business_area.value}
        TIPO DE DOCUMENTO: {request.document_type.value}
        NOMBRE DE LA EMPRESA: {request.company_name or 'Empresa PYME'}
        INDUSTRIA: {request.industry or 'No especificada'}
        TAMAÑO DE EMPRESA: {request.company_size or 'Pequeña/Mediana'}
        AUDIENCIA OBJETIVO: {request.target_audience or 'Clientes generales'}
        IDIOMA: {request.language}

        INSTRUCCIONES:
        1. Genera un documento profesional, estructurado y detallado
        2. Incluye secciones relevantes para el tipo de documento solicitado
        3. Proporciona ejemplos prácticos y casos de uso
        4. Incluye métricas y KPIs cuando sea apropiado
        5. Usa un tono profesional pero accesible
        6. Adapta el contenido al tamaño de empresa (PYME)
        7. Incluye recomendaciones específicas y accionables

        ESTRUCTURA DEL DOCUMENTO:
        - Título claro y descriptivo
        - Resumen ejecutivo
        - Secciones principales con subtítulos
        - Conclusiones y recomendaciones
        - Anexos si es necesario

        Genera el documento completo en formato {request.format.upper()}.
        """
        
        return prompt
    
    async def _call_openrouter_api(self, prompt: str, request: DocumentRequest) -> str:
        """Call OpenRouter API for content generation with enhanced error handling and tracking"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://bul-system.com",
                "X-Title": "BUL Document Generator",
                "User-Agent": "BUL-Engine/2.0.0"
            }
            
            payload = {
                "model": "openai/gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": "Eres un experto consultor de negocios especializado en PYMEs. Genera documentos profesionales, detallados y accionables."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.7,
                "stream": False
            }
            
            # Try httpx first (more efficient)
            try:
                response = await self.http_client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                # Track successful API call
                self.stats["api_calls_successful"] += 1
                
                return result["choices"][0]["message"]["content"]
                
            except httpx.HTTPError as e:
                self.stats["api_calls_failed"] += 1
                logger.warning(f"httpx request failed, trying aiohttp: {e}")
                
                # Fallback to aiohttp
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.stats["api_calls_successful"] += 1
                        return result["choices"][0]["message"]["content"]
                    else:
                        self.stats["api_calls_failed"] += 1
                        raise Exception(f"OpenRouter API error: {response.status} - {await response.text()}")
                        
        except Exception as e:
            self.stats["api_calls_failed"] += 1
            logger.error(f"Error calling OpenRouter API: {e}")
            # Fallback to OpenAI if available
            if self.llm:
                logger.info("Attempting OpenAI fallback")
                return await self._fallback_to_openai(prompt)
            else:
                raise
    
    async def _fallback_to_openai(self, prompt: str) -> str:
        """Fallback to OpenAI API if OpenRouter fails"""
        try:
            # Use LangChain with OpenAI
            template = PromptTemplate(
                input_variables=["prompt"],
                template="{prompt}"
            )
            
            chain = LLMChain(llm=self.llm, prompt=template, memory=self.memory)
            result = await chain.arun(prompt=prompt)
            return result
            
        except Exception as e:
            logger.error(f"Error with OpenAI fallback: {e}")
            raise
    
    def _generate_title(self, request: DocumentRequest, content: str) -> str:
        """Generate a title for the document"""
        # Extract first line or create based on document type
        lines = content.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                return line.strip()[:100]
        
        # Fallback title
        return f"{request.document_type.value.replace('_', ' ').title()} - {request.company_name or 'Empresa PYME'}"
    
    def _generate_summary(self, content: str) -> str:
        """Generate a summary of the document"""
        # Simple summary - first 200 characters
        summary = content.replace('\n', ' ').strip()
        if len(summary) > 200:
            summary = summary[:200] + "..."
        return summary
    
    def _update_stats(self, response: DocumentResponse):
        """Update generation statistics"""
        self.stats["documents_generated"] += 1
        self.stats["total_processing_time"] += response.processing_time
        self.stats["average_confidence"] = (
            (self.stats["average_confidence"] * (self.stats["documents_generated"] - 1) + response.confidence_score) 
            / self.stats["documents_generated"]
        )
        
        # Update business area stats
        area = response.business_area.value
        self.stats["business_areas_used"][area] = self.stats["business_areas_used"].get(area, 0) + 1
        
        # Update document type stats
        doc_type = response.document_type.value
        self.stats["document_types_generated"][doc_type] = self.stats["document_types_generated"].get(doc_type, 0) + 1
        
        # Update uptime
        self.stats["uptime"] = time.time() - self.start_time
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        self.performance_metrics["response_times"].append(processing_time)
        
        # Keep only last 100 response times for memory efficiency
        if len(self.performance_metrics["response_times"]) > 100:
            self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-100:]
        
        # Update min/max
        if processing_time < self.performance_metrics["min_response_time"]:
            self.performance_metrics["min_response_time"] = processing_time
        if processing_time > self.performance_metrics["max_response_time"]:
            self.performance_metrics["max_response_time"] = processing_time
        
        # Update average
        self.performance_metrics["avg_response_time"] = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics"""
        return {
            "documents_generated": self.stats["documents_generated"],
            "total_processing_time": self.stats["total_processing_time"],
            "average_processing_time": (
                self.stats["total_processing_time"] / self.stats["documents_generated"] 
                if self.stats["documents_generated"] > 0 else 0
            ),
            "average_confidence": self.stats["average_confidence"],
            "business_areas_used": self.stats["business_areas_used"],
            "document_types_generated": self.stats["document_types_generated"],
            "cache_performance": {
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "hit_rate": (
                    self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
                    if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
                )
            },
            "api_performance": {
                "successful_calls": self.stats["api_calls_successful"],
                "failed_calls": self.stats["api_calls_failed"],
                "fallback_usage": self.stats["fallback_usage"],
                "error_count": self.stats["error_count"],
                "last_error": self.stats["last_error"]
            },
            "performance_metrics": {
                "avg_response_time": self.performance_metrics["avg_response_time"],
                "min_response_time": self.performance_metrics["min_response_time"] if self.performance_metrics["min_response_time"] != float('inf') else 0,
                "max_response_time": self.performance_metrics["max_response_time"],
                "total_requests_tracked": len(self.performance_metrics["response_times"])
            },
            "system_info": {
                "is_initialized": self.is_initialized,
                "uptime_seconds": self.stats["uptime"],
                "start_time": self.start_time
            }
        }
    
    async def close(self):
        """Close the engine and cleanup resources"""
        if self.session:
            await self.session.close()
        if self.http_client:
            await self.http_client.aclose()
        logger.info("BUL Engine closed")

# Global engine instance
_bul_engine: Optional[BULEngine] = None

async def get_global_bul_engine() -> BULEngine:
    """Get the global BUL engine instance"""
    global _bul_engine
    if _bul_engine is None:
        # This should be configured via environment variables
        import os
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        _bul_engine = BULEngine(openrouter_key, openai_key)
        await _bul_engine.initialize()
    
    return _bul_engine



