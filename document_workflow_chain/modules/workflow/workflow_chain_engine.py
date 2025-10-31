"""
Document Workflow Chain Engine
==============================

This module implements an AI-powered document generation system that uses prompt chaining
to create continuous document workflows. Each document's output becomes the input for
the next document in the chain, enabling seamless content generation flows.

Features:
- Continuous document generation with chaining
- Workflow management and state tracking
- Title generation for blog posts
- Content flow optimization
- Multiple AI model support
"""

import asyncio
import json
import logging
import re
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from enum import Enum
from collections import deque

# Import our custom modules
from ai_clients import AIClientFactory, AIClientManager, AIClient
from database import DatabaseManager, create_database_manager
from config import settings, get_ai_client_config, get_workflow_settings
from content_analyzer import ContentAnalyzer, ContentMetrics
from content_templates import ContentTemplateManager, template_manager
from multilang_support import MultiLanguageManager, multilang_manager
from advanced_analytics import AdvancedAnalytics, PerformanceMetrics, advanced_analytics
from ai_optimization import PromptOptimizer, ContextOptimizer, prompt_optimizer, context_optimizer
from intelligent_generation import IntelligentGenerator, GenerationContext, GenerationResult, intelligent_generator
from trend_analysis import TrendAnalyzer, TrendData, ContentPrediction, MarketIntelligence, trend_analyzer
from external_integrations import ExternalIntegrations, IntegrationConfig, external_integrations
from workflow_scheduler import WorkflowScheduler, ScheduleType, workflow_scheduler
from content_quality_control import ContentQualityController, QualityLevel, content_quality_controller
from content_versioning import ContentVersionManager, VersionType, content_version_manager

# AI Model Context Limits (tokens)
class ModelContextLimits(Enum):
    """Context limits for different AI models"""
    CLAUDE_3_5_SONNET = 200000
    CLAUDE_3_OPUS = 200000
    CLAUDE_3_HAIKU = 200000
    GPT_4_TURBO = 128000
    GPT_4 = 8192
    GPT_3_5_TURBO = 16384
    GEMINI_1_5_PRO = 1000000
    GEMINI_1_5_FLASH = 1000000
    GEMINI_PRO = 32768
    
    @classmethod
    def get_limit(cls, model_name: str) -> int:
        """Get context limit for a specific model"""
        model_mapping = {
            'claude-3-5-sonnet-20241022': cls.CLAUDE_3_5_SONNET,
            'claude-3-opus-20240229': cls.CLAUDE_3_OPUS,
            'claude-3-haiku-20240307': cls.CLAUDE_3_HAIKU,
            'gpt-4-turbo': cls.GPT_4_TURBO,
            'gpt-4': cls.GPT_4,
            'gpt-3.5-turbo': cls.GPT_3_5_TURBO,
            'gemini-1.5-pro': cls.GEMINI_1_5_PRO,
            'gemini-1.5-flash': cls.GEMINI_1_5_FLASH,
            'gemini-pro': cls.GEMINI_PRO
        }
        return model_mapping.get(model_name.lower(), cls.GPT_4_TURBO).value

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of a document for context management"""
    id: str
    content: str
    chunk_index: int
    total_chunks: int
    token_count: int
    summary: Optional[str] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

@dataclass
class DocumentNode:
    """Represents a single document in the workflow chain"""
    id: str
    title: str
    content: str
    prompt: str
    generated_at: datetime
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    metadata: Dict[str, Any] = None
    chunks: List[DocumentChunk] = None
    context_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}
        if self.chunks is None:
            self.chunks = []
        if self.context_hash is None:
            self.context_hash = self._generate_context_hash()
    
    def _generate_context_hash(self) -> str:
        """Generate a hash for context deduplication"""
        content_to_hash = f"{self.title}:{self.content[:1000]}"
        return hashlib.md5(content_to_hash.encode()).hexdigest()

@dataclass
class ContextWindow:
    """Manages context window for AI models"""
    model_name: str
    max_tokens: int
    current_tokens: int = 0
    documents: deque = None
    summaries: Dict[str, str] = None
    
    def __post_init__(self):
        if self.documents is None:
            self.documents = deque()
        if self.summaries is None:
            self.summaries = {}
    
    def can_add_document(self, token_count: int) -> bool:
        """Check if document can be added to context"""
        return self.current_tokens + token_count <= self.max_tokens
    
    def add_document(self, doc_id: str, token_count: int, content: str):
        """Add document to context window"""
        if not self.can_add_document(token_count):
            self._compress_context()
        
        self.documents.append((doc_id, token_count, content))
        self.current_tokens += token_count
    
    def _compress_context(self):
        """Compress context by summarizing older documents"""
        while self.documents and self.current_tokens > self.max_tokens * 0.8:
            doc_id, token_count, content = self.documents.popleft()
            self.current_tokens -= token_count
            # Create summary for removed document
            self.summaries[doc_id] = content[:200] + "..."

@dataclass
class WorkflowChain:
    """Represents a complete workflow chain"""
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    root_node_id: Optional[str] = None
    nodes: Dict[str, DocumentNode] = None
    status: str = "active"  # active, paused, completed, error
    settings: Dict[str, Any] = None
    context_window: Optional[ContextWindow] = None
    
    def __post_init__(self):
        if self.nodes is None:
            self.nodes = {}
        if self.settings is None:
            self.settings = {}
        if self.context_window is None:
            self.context_window = ContextWindow(
                model_name=self.settings.get('ai_model', 'gpt-4-turbo'),
                max_tokens=ModelContextLimits.get_limit(self.settings.get('ai_model', 'gpt-4-turbo'))
            )

class WorkflowChainEngine:
    """
    Main engine for managing document workflow chains with advanced context management
    """
    
    def __init__(self, ai_client=None, database_manager=None):
        self.ai_client = ai_client
        self.database_manager = database_manager
        self.active_chains: Dict[str, WorkflowChain] = {}
        self.chain_history: List[WorkflowChain] = []
        self._initialized = False
        
        # Initialize advanced components
        self.content_analyzer = ContentAnalyzer()
        self.template_manager = template_manager
        self.multilang_manager = multilang_manager
        self.analytics = advanced_analytics
        self.token_estimator = TokenEstimator()
        self.context_compressor = ContextCompressor()
        
            # Initialize new advanced features
            self.prompt_optimizer = prompt_optimizer
            self.context_optimizer = context_optimizer
            self.intelligent_generator = intelligent_generator
            self.trend_analyzer = trend_analyzer
            self.external_integrations = external_integrations
            
            # Initialize workflow management features
            self.workflow_scheduler = workflow_scheduler
            self.quality_controller = content_quality_controller
            self.version_manager = content_version_manager
    
    async def initialize(self):
        """Initialize the engine with AI client and database"""
        try:
            # Initialize AI client if not provided
            if not self.ai_client:
                ai_config = get_ai_client_config()
                self.ai_client = AIClientFactory.create_client(
                    client_type=settings.ai_client_type.value,
                    api_key=settings.ai_api_key or "mock",
                    model=settings.ai_model,
                    **ai_config
                )
                logger.info(f"Initialized AI client: {settings.ai_client_type.value}")
            
            # Initialize database manager if not provided
            if not self.database_manager and settings.database_url:
                self.database_manager = await create_database_manager(settings.database_url)
                logger.info("Initialized database manager")
            
            # Initialize external integrations
            await self.external_integrations.initialize()
            
            # Initialize workflow management features
            await self.workflow_scheduler.start()
            
            self._initialized = True
            logger.info("WorkflowChainEngine initialized successfully with all advanced features")
            
        except Exception as e:
            logger.error(f"Failed to initialize WorkflowChainEngine: {str(e)}")
            raise
        
    async def create_workflow_chain(
        self, 
        name: str, 
        description: str,
        initial_prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> WorkflowChain:
        """
        Create a new workflow chain with an initial document
        
        Args:
            name: Name of the workflow chain
            description: Description of the workflow purpose
            initial_prompt: Initial prompt to start the chain
            settings: Optional settings for the workflow
            user_id: Optional user ID for multi-user support
            
        Returns:
            WorkflowChain: The created workflow chain
        """
        try:
            # Ensure engine is initialized
            if not self._initialized:
                await self.initialize()
            
            chain_id = str(uuid.uuid4())
            now = datetime.now()
            
            # Merge with default settings
            workflow_settings = get_workflow_settings(settings)
            
            # Create initial document node
            initial_node = await self._generate_document_node(
                prompt=initial_prompt,
                parent_id=None,
                chain_id=chain_id
            )
            
            # Create workflow chain in memory
            chain = WorkflowChain(
                id=chain_id,
                name=name,
                description=description,
                created_at=now,
                updated_at=now,
                root_node_id=initial_node.id,
                nodes={initial_node.id: initial_node},
                settings=workflow_settings
            )
            
            # Save to database if available
            if self.database_manager:
                try:
                    # Save workflow chain to database
                    db_chain = await self.database_manager.create_workflow_chain(
                        name=name,
                        description=description,
                        root_node_id=initial_node.id,
                        status="active",
                        settings=workflow_settings,
                        user_id=user_id,
                        metadata={"created_by": "workflow_engine"}
                    )
                    
                    # Save initial document node to database
                    await self.database_manager.create_document_node(
                        chain_id=chain_id,
                        title=initial_node.title,
                        content=initial_node.content,
                        prompt=initial_node.prompt,
                        parent_id=initial_node.parent_id,
                        children_ids=initial_node.children_ids,
                        metadata=initial_node.metadata,
                        ai_model_used=initial_node.metadata.get("ai_model_used"),
                        tokens_used=initial_node.metadata.get("tokens_used", 0),
                        generation_time=initial_node.metadata.get("generation_time", 0.0),
                        quality_score=initial_node.metadata.get("quality_score")
                    )
                    
                    logger.info(f"Saved workflow chain to database: {chain_id}")
                    
                except Exception as db_error:
                    logger.warning(f"Failed to save to database: {str(db_error)}")
                    # Continue without database persistence
            
            self.active_chains[chain_id] = chain
            logger.info(f"Created workflow chain: {chain_id}")
            
            return chain
            
        except Exception as e:
            logger.error(f"Error creating workflow chain: {str(e)}")
            raise
    
    async def continue_workflow_chain(
        self, 
        chain_id: str, 
        continuation_prompt: Optional[str] = None
    ) -> DocumentNode:
        """
        Continue a workflow chain by generating the next document
        
        Args:
            chain_id: ID of the workflow chain to continue
            continuation_prompt: Optional custom prompt, uses auto-generated if None
            
        Returns:
            DocumentNode: The newly generated document node
        """
        try:
            if chain_id not in self.active_chains:
                raise ValueError(f"Workflow chain {chain_id} not found")
            
            chain = self.active_chains[chain_id]
            
            # Get the latest document in the chain
            latest_node = self._get_latest_node(chain)
            
            # Generate continuation prompt if not provided
            if not continuation_prompt:
                continuation_prompt = await self._generate_continuation_prompt(latest_node)
            
            # Create new document node
            new_node = await self._generate_document_node(
                prompt=continuation_prompt,
                parent_id=latest_node.id,
                chain_id=chain_id
            )
            
            # Update chain
            chain.nodes[new_node.id] = new_node
            latest_node.children_ids.append(new_node.id)
            chain.updated_at = datetime.now()
            
            logger.info(f"Continued workflow chain {chain_id} with new document: {new_node.id}")
            
            return new_node
            
        except Exception as e:
            logger.error(f"Error continuing workflow chain: {str(e)}")
            raise
    
    async def generate_blog_title(self, content: str) -> str:
        """
        Generate a compelling blog title from document content
        
        Args:
            content: The document content to generate title from
            
        Returns:
            str: Generated blog title
        """
        try:
            if self.ai_client:
                try:
                    # Use the AI client's dedicated title generation method
                    ai_response = await self.ai_client.generate_title(content)
                    title = ai_response.content.strip()
                    
                    # Clean up the title
                    title = title.replace('"', '').replace("'", "").strip()
                    
                    # Ensure title is within reasonable length
                    if len(title) > 80:
                        title = title[:77] + "..."
                    
                    logger.info(f"Generated title using {ai_response.model}: {title}")
                    return title
                    
                except Exception as ai_error:
                    logger.warning(f"AI title generation failed: {str(ai_error)}")
                    # Fallback to text generation
                    title_prompt = f"""
                    Based on the following content, generate a compelling, SEO-friendly blog title that:
                    1. Captures the main topic and value proposition
                    2. Is between 50-60 characters
                    3. Uses engaging language
                    4. Includes relevant keywords
                    
                    Content:
                    {content[:1000]}...
                    
                    Generate only the title, no additional text.
                    """
                    
                    ai_response = await self.ai_client.generate_text(title_prompt)
                    title = ai_response.content.strip()
                    title = title.replace('"', '').replace("'", "").strip()
                    
                    if len(title) > 80:
                        title = title[:77] + "..."
                    
                    return title
            else:
                # Fallback title generation
                words = content.split()[:8]
                title = " ".join(words).title() + "..."
                
                if len(title) > 80:
                    title = title[:77] + "..."
                
                return title
                
        except Exception as e:
            logger.error(f"Error generating blog title: {str(e)}")
            return "Generated Blog Post"
    
    async def _generate_document_node(
        self, 
        prompt: str, 
        parent_id: Optional[str], 
        chain_id: str
    ) -> DocumentNode:
        """
        Generate a new document node using AI
        
        Args:
            prompt: The prompt to generate content from
            parent_id: ID of parent document (None for root)
            chain_id: ID of the workflow chain
            
        Returns:
            DocumentNode: The generated document node
        """
        try:
            node_id = str(uuid.uuid4())
            now = datetime.now()
            
            # Optimize prompt for context management
            optimized_prompt = await self._optimize_prompt_for_context(prompt, chain_id)
            
            # Generate content using AI
            if self.ai_client:
                try:
                    ai_response = await self.ai_client.generate_text(optimized_prompt)
                    content = ai_response.content
                    ai_model_used = ai_response.model
                    tokens_used = ai_response.tokens_used
                    generation_time = ai_response.response_time
                    
                    logger.info(f"Generated content using {ai_model_used}, tokens: {tokens_used}")
                    
                except Exception as ai_error:
                    logger.error(f"AI generation failed: {str(ai_error)}")
                    # Fallback content generation
                    content = f"Generated content for prompt: {prompt[:100]}..."
                    ai_model_used = "fallback"
                    tokens_used = 0
                    generation_time = 0.0
            else:
                # Fallback content generation
                content = f"Generated content for prompt: {prompt[:100]}..."
                ai_model_used = "fallback"
                tokens_used = 0
                generation_time = 0.0
            
            # Generate title
            title = await self.generate_blog_title(content)
            
            # Calculate quality score (enhanced heuristic)
            quality_score = self._calculate_enhanced_quality_score(content, prompt)
            
            # Create document chunks for context management
            chunks = self._create_document_chunks(content, node_id)
            
            # Create document node
            node = DocumentNode(
                id=node_id,
                title=title,
                content=content,
                prompt=prompt,
                generated_at=now,
                parent_id=parent_id,
                chunks=chunks,
                metadata={
                    "chain_id": chain_id,
                    "word_count": len(content.split()),
                    "token_count": self.token_estimator.estimate_tokens(content),
                    "generation_method": "ai" if self.ai_client else "fallback",
                    "ai_model_used": ai_model_used,
                    "tokens_used": tokens_used,
                    "generation_time": generation_time,
                    "quality_score": quality_score,
                    "readability_score": self._calculate_readability_score(content),
                    "coherence_score": self._calculate_coherence_score(content),
                    "relevance_score": self._calculate_relevance_score(content, prompt)
                }
            )
            
            # Update context window
            if chain_id in self.active_chains:
                chain = self.active_chains[chain_id]
                chain.context_window.add_document(
                    node_id, 
                    node.metadata["token_count"], 
                    content
                )
            
            return node
            
        except Exception as e:
            logger.error(f"Error generating document node: {str(e)}")
            raise
    
    def _calculate_quality_score(self, content: str, prompt: str) -> float:
        """
        Calculate a simple quality score for generated content
        
        Args:
            content: Generated content
            prompt: Original prompt
            
        Returns:
            float: Quality score between 0 and 1
        """
        try:
            score = 0.0
            
            # Length score (optimal length between 200-2000 words)
            word_count = len(content.split())
            if 200 <= word_count <= 2000:
                score += 0.3
            elif 100 <= word_count < 200 or 2000 < word_count <= 3000:
                score += 0.2
            else:
                score += 0.1
            
            # Structure score (paragraphs, sentences)
            paragraphs = content.count('\n\n') + 1
            sentences = content.count('.') + content.count('!') + content.count('?')
            
            if paragraphs >= 3 and sentences >= 5:
                score += 0.3
            elif paragraphs >= 2 and sentences >= 3:
                score += 0.2
            else:
                score += 0.1
            
            # Relevance score (keyword overlap)
            prompt_words = set(prompt.lower().split())
            content_words = set(content.lower().split())
            overlap = len(prompt_words.intersection(content_words))
            
            if overlap >= 3:
                score += 0.2
            elif overlap >= 1:
                score += 0.1
            
            # Readability score (simple heuristic)
            avg_sentence_length = word_count / max(sentences, 1)
            if 10 <= avg_sentence_length <= 25:
                score += 0.2
            elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 35:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {str(e)}")
            return 0.5  # Default score
    
    async def _generate_continuation_prompt(self, previous_node: DocumentNode) -> str:
        """
        Generate a continuation prompt based on the previous document
        
        Args:
            previous_node: The previous document node
            
        Returns:
            str: Generated continuation prompt
        """
        try:
            continuation_prompt = f"""
            Based on the previous document titled "{previous_node.title}", 
            create a natural continuation that:
            1. Builds upon the ideas presented
            2. Introduces new related concepts
            3. Maintains the same tone and style
            4. Provides additional value to the reader
            
            Previous content summary:
            {previous_node.content[:500]}...
            
            Generate a new, related document that flows naturally from this content.
            """
            
            return continuation_prompt
            
        except Exception as e:
            logger.error(f"Error generating continuation prompt: {str(e)}")
            return "Continue the previous topic with new insights and information."
    
    def _get_latest_node(self, chain: WorkflowChain) -> DocumentNode:
        """
        Get the latest (most recent) node in a workflow chain
        
        Args:
            chain: The workflow chain
            
        Returns:
            DocumentNode: The latest document node
        """
        if not chain.nodes:
            raise ValueError("Chain has no nodes")
        
        # Find the node with the most recent timestamp
        latest_node = max(chain.nodes.values(), key=lambda n: n.generated_at)
        return latest_node
    
    def get_workflow_chain(self, chain_id: str) -> Optional[WorkflowChain]:
        """
        Get a workflow chain by ID
        
        Args:
            chain_id: ID of the workflow chain
            
        Returns:
            WorkflowChain or None if not found
        """
        return self.active_chains.get(chain_id)
    
    def get_chain_history(self, chain_id: str) -> List[DocumentNode]:
        """
        Get the complete history of a workflow chain
        
        Args:
            chain_id: ID of the workflow chain
            
        Returns:
            List[DocumentNode]: Ordered list of all documents in the chain
        """
        if chain_id not in self.active_chains:
            return []
        
        chain = self.active_chains[chain_id]
        nodes = list(chain.nodes.values())
        
        # Sort by generation time
        nodes.sort(key=lambda n: n.generated_at)
        
        return nodes
    
    def pause_workflow_chain(self, chain_id: str) -> bool:
        """
        Pause a workflow chain
        
        Args:
            chain_id: ID of the workflow chain
            
        Returns:
            bool: True if successfully paused
        """
        if chain_id in self.active_chains:
            self.active_chains[chain_id].status = "paused"
            self.active_chains[chain_id].updated_at = datetime.now()
            logger.info(f"Paused workflow chain: {chain_id}")
            return True
        return False
    
    def resume_workflow_chain(self, chain_id: str) -> bool:
        """
        Resume a paused workflow chain
        
        Args:
            chain_id: ID of the workflow chain
            
        Returns:
            bool: True if successfully resumed
        """
        if chain_id in self.active_chains and self.active_chains[chain_id].status == "paused":
            self.active_chains[chain_id].status = "active"
            self.active_chains[chain_id].updated_at = datetime.now()
            logger.info(f"Resumed workflow chain: {chain_id}")
            return True
        return False
    
    def complete_workflow_chain(self, chain_id: str) -> bool:
        """
        Mark a workflow chain as completed
        
        Args:
            chain_id: ID of the workflow chain
            
        Returns:
            bool: True if successfully completed
        """
        if chain_id in self.active_chains:
            chain = self.active_chains[chain_id]
            chain.status = "completed"
            chain.updated_at = datetime.now()
            
            # Move to history
            self.chain_history.append(chain)
            del self.active_chains[chain_id]
            
            logger.info(f"Completed workflow chain: {chain_id}")
            return True
        return False
    
    def get_all_active_chains(self) -> List[WorkflowChain]:
        """
        Get all active workflow chains
        
        Returns:
            List[WorkflowChain]: List of all active chains
        """
        return list(self.active_chains.values())
    
    def export_workflow_chain(self, chain_id: str) -> Dict[str, Any]:
        """
        Export a workflow chain to a dictionary format
        
        Args:
            chain_id: ID of the workflow chain
            
        Returns:
            Dict: Exported workflow chain data
        """
        chain = self.get_workflow_chain(chain_id)
        if not chain:
            return {}
        
        # Convert to serializable format
        export_data = asdict(chain)
        
        # Convert datetime objects to strings
        export_data['created_at'] = chain.created_at.isoformat()
        export_data['updated_at'] = chain.updated_at.isoformat()
        
        for node_id, node in export_data['nodes'].items():
            node['generated_at'] = node['generated_at'].isoformat()
        
        return export_data
    
    async def create_workflow_with_template(
        self,
        template_id: str,
        topic: str,
        name: str,
        description: str,
        language_code: str = "en",
        **kwargs
    ) -> WorkflowChain:
        """
        Create a workflow chain using a content template
        
        Args:
            template_id: ID of the content template to use
            topic: Main topic for the content
            name: Name of the workflow chain
            description: Description of the workflow
            language_code: Language code for content generation
            **kwargs: Additional parameters for template
            
        Returns:
            WorkflowChain: The created workflow chain
        """
        try:
            # Get template
            template = self.template_manager.get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Generate prompt from template
            prompt = self.template_manager.generate_prompt_from_template(
                template_id, topic, **kwargs
            )
            
            # Adapt prompt for language
            if language_code != "en":
                prompt = self.multilang_manager.adapt_prompt_for_language(
                    prompt, language_code, template_id
                )
            
            # Create workflow chain
            chain = await self.create_workflow_chain(
                name=name,
                description=description,
                initial_prompt=prompt,
                settings={
                    "template_id": template_id,
                    "language_code": language_code,
                    "target_word_count": template.target_word_count,
                    "seo_keywords": template.seo_keywords
                }
            )
            
            logger.info(f"Created workflow chain with template {template_id} in {language_code}")
            return chain
            
        except Exception as e:
            logger.error(f"Error creating workflow with template: {str(e)}")
            raise
    
    async def analyze_workflow_performance(self, chain_id: str) -> Dict[str, Any]:
        """
        Analyze workflow performance using advanced analytics
        
        Args:
            chain_id: ID of the workflow chain
            
        Returns:
            Dict: Performance analysis results
        """
        try:
            chain = self.get_workflow_chain(chain_id)
            if not chain:
                return {"error": "Workflow chain not found"}
            
            # Get performance summary
            performance_summary = await self.analytics.get_performance_summary(chain_id, "30d")
            
            # Get predictive insights
            insights = await self.analytics.generate_predictive_insights(chain_id)
            
            # Get optimization recommendations
            recommendations = await self.analytics.get_optimization_recommendations(chain_id)
            
            # Get trend analysis for key metrics
            trends = {}
            key_metrics = ["quality_score", "generation_time", "tokens_used", "engagement_score"]
            for metric in key_metrics:
                trend = await self.analytics.analyze_trends(metric, "7d", chain_id)
                trends[metric] = asdict(trend)
            
            return {
                "chain_id": chain_id,
                "performance_summary": performance_summary,
                "predictive_insights": [asdict(insight) for insight in insights],
                "optimization_recommendations": recommendations,
                "trend_analysis": trends,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing workflow performance: {str(e)}")
            return {"error": str(e)}
    
    async def enhance_document_with_analysis(
        self,
        content: str,
        title: str,
        language_code: str = "en"
    ) -> Dict[str, Any]:
        """
        Enhance document using content analysis and optimization
        
        Args:
            content: Document content
            title: Document title
            language_code: Language code
            
        Returns:
            Dict: Enhanced content with analysis
        """
        try:
            # Analyze content
            metrics = await self.content_analyzer.analyze_content(content, title)
            
            # Get improvement suggestions
            suggestions = await self.content_analyzer.suggest_improvements(metrics, content)
            
            # Localize metadata
            localized_metadata = self.multilang_manager.localize_content_metadata(
                {"original_analysis": asdict(metrics)}, language_code
            )
            
            return {
                "original_content": content,
                "enhanced_content": content,  # Could be enhanced based on suggestions
                "title": title,
                "content_metrics": asdict(metrics),
                "improvement_suggestions": suggestions,
                "localized_metadata": localized_metadata,
                "language_code": language_code,
                "enhancement_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error enhancing document: {str(e)}")
            return {"error": str(e)}
    
    async def get_available_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available content templates
        
        Args:
            category: Optional category filter
            
        Returns:
            List: Available templates
        """
        try:
            if category:
                templates = self.template_manager.get_templates_by_category(category)
            else:
                templates = self.template_manager.get_all_templates()
            
            return [
                {
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "category": template.category,
                    "target_word_count": template.target_word_count,
                    "seo_keywords": template.seo_keywords,
                    "metadata": template.metadata
                }
                for template in templates
            ]
            
        except Exception as e:
            logger.error(f"Error getting templates: {str(e)}")
            return []
    
    async def get_supported_languages(self) -> List[Dict[str, Any]]:
        """
        Get supported languages for content generation
        
        Returns:
            List: Supported languages with information
        """
        try:
            languages = []
            for lang_code in self.multilang_manager.get_supported_languages():
                lang_info = self.multilang_manager.get_language_info(lang_code)
                languages.append(lang_info)
            
            return languages
            
        except Exception as e:
            logger.error(f"Error getting supported languages: {str(e)}")
            return []
    
    async def record_document_metrics(
        self,
        chain_id: str,
        document_id: str,
        content: str,
        generation_time: float,
        tokens_used: int
    ):
        """
        Record document metrics for analytics
        
        Args:
            chain_id: Workflow chain ID
            document_id: Document ID
            content: Document content
            generation_time: Time taken to generate
            tokens_used: Tokens used for generation
        """
        try:
            # Analyze content for metrics
            metrics = await self.content_analyzer.analyze_content(content)
            
            # Create performance metrics
            performance_metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                chain_id=chain_id,
                document_id=document_id,
                quality_score=metrics.overall_quality,
                generation_time=generation_time,
                tokens_used=tokens_used,
                word_count=metrics.word_count,
                engagement_score=metrics.engagement_score,
                seo_score=metrics.seo_score,
                readability_score=metrics.readability_score
            )
            
            # Record metrics
            await self.analytics.record_metrics(performance_metrics)
            
            logger.info(f"Recorded metrics for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error recording document metrics: {str(e)}")
    
    async def get_workflow_insights(self, chain_id: str) -> Dict[str, Any]:
        """
        Get comprehensive insights for a workflow chain
        
        Args:
            chain_id: Workflow chain ID
            
        Returns:
            Dict: Comprehensive insights
        """
        try:
            chain = self.get_workflow_chain(chain_id)
            if not chain:
                return {"error": "Workflow chain not found"}
            
            # Get performance analysis
            performance_analysis = await self.analyze_workflow_performance(chain_id)
            
            # Get chain history
            history = self.get_chain_history(chain_id)
            
            # Get template information if available
            template_info = None
            if "template_id" in chain.settings:
                template = self.template_manager.get_template(chain.settings["template_id"])
                if template:
                    template_info = {
                        "id": template.id,
                        "name": template.name,
                        "category": template.category
                    }
            
            # Get language information if available
            language_info = None
            if "language_code" in chain.settings:
                language_info = self.multilang_manager.get_language_info(
                    chain.settings["language_code"]
                )
            
            return {
                "chain_info": {
                    "id": chain.id,
                    "name": chain.name,
                    "description": chain.description,
                    "status": chain.status,
                    "created_at": chain.created_at.isoformat(),
                    "updated_at": chain.updated_at.isoformat(),
                    "document_count": len(chain.nodes),
                    "template_info": template_info,
                    "language_info": language_info
                },
                "performance_analysis": performance_analysis,
                "document_history": [
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "generated_at": doc.generated_at.isoformat(),
                        "metadata": doc.metadata
                    }
                    for doc in history
                ],
                "insights_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow insights: {str(e)}")
            return {"error": str(e)}
    
    async def _optimize_prompt_for_context(self, prompt: str, chain_id: str) -> str:
        """
        Optimize prompt to fit within context limits
        
        Args:
            prompt: Original prompt
            chain_id: ID of the workflow chain
            
        Returns:
            str: Optimized prompt
        """
        try:
            if chain_id not in self.active_chains:
                return prompt
            
            chain = self.active_chains[chain_id]
            context_window = chain.context_window
            
            # Estimate tokens for current prompt
            prompt_tokens = self.token_estimator.estimate_tokens(prompt)
            
            # If prompt is too long, compress it
            if prompt_tokens > context_window.max_tokens * 0.3:  # Use max 30% for prompt
                compressed_prompt = await self.context_compressor.compress_prompt(prompt)
                logger.info(f"Compressed prompt from {prompt_tokens} to {self.token_estimator.estimate_tokens(compressed_prompt)} tokens")
                return compressed_prompt
            
            return prompt
            
        except Exception as e:
            logger.warning(f"Error optimizing prompt: {str(e)}")
            return prompt
    
    def _create_document_chunks(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """
        Create document chunks for context management
        
        Args:
            content: Document content
            doc_id: Document ID
            
        Returns:
            List[DocumentChunk]: List of document chunks
        """
        try:
            # Split content into sentences
            sentences = re.split(r'[.!?]+\s*', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = ""
            chunk_index = 0
            max_chunk_tokens = 1000  # Optimal chunk size
            
            for sentence in sentences:
                sentence_tokens = self.token_estimator.estimate_tokens(sentence)
                
                if self.token_estimator.estimate_tokens(current_chunk + sentence) > max_chunk_tokens and current_chunk:
                    # Create chunk
                    chunk = DocumentChunk(
                        id=f"{doc_id}_chunk_{chunk_index}",
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        total_chunks=0,  # Will be updated later
                        token_count=self.token_estimator.estimate_tokens(current_chunk),
                        keywords=self._extract_keywords(current_chunk)
                    )
                    chunks.append(chunk)
                    current_chunk = sentence
                    chunk_index += 1
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add final chunk
            if current_chunk:
                chunk = DocumentChunk(
                    id=f"{doc_id}_chunk_{chunk_index}",
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    total_chunks=chunk_index + 1,
                    token_count=self.token_estimator.estimate_tokens(current_chunk),
                    keywords=self._extract_keywords(current_chunk)
                )
                chunks.append(chunk)
            
            # Update total_chunks for all chunks
            for chunk in chunks:
                chunk.total_chunks = len(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating document chunks: {str(e)}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        try:
            # Simple keyword extraction (can be enhanced with NLP libraries)
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return top 5 keywords
            return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            
        except Exception as e:
            logger.warning(f"Error extracting keywords: {str(e)}")
            return []
    
    def _calculate_enhanced_quality_score(self, content: str, prompt: str) -> float:
        """
        Calculate enhanced quality score for generated content
        
        Args:
            content: Generated content
            prompt: Original prompt
            
        Returns:
            float: Quality score between 0 and 1
        """
        try:
            scores = {
                'length': self._calculate_length_score(content),
                'structure': self._calculate_structure_score(content),
                'relevance': self._calculate_relevance_score(content, prompt),
                'readability': self._calculate_readability_score(content),
                'coherence': self._calculate_coherence_score(content)
            }
            
            # Weighted average
            weights = {'length': 0.2, 'structure': 0.2, 'relevance': 0.3, 'readability': 0.15, 'coherence': 0.15}
            weighted_score = sum(scores[key] * weights[key] for key in scores)
            
            return min(weighted_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating enhanced quality score: {str(e)}")
            return 0.5
    
    def _calculate_length_score(self, content: str) -> float:
        """Calculate length-based quality score"""
        word_count = len(content.split())
        if 200 <= word_count <= 2000:
            return 1.0
        elif 100 <= word_count < 200 or 2000 < word_count <= 3000:
            return 0.7
        elif 50 <= word_count < 100 or 3000 < word_count <= 5000:
            return 0.4
        else:
            return 0.1
    
    def _calculate_structure_score(self, content: str) -> float:
        """Calculate structure-based quality score"""
        paragraphs = content.count('\n\n') + 1
        sentences = content.count('.') + content.count('!') + content.count('?')
        
        if paragraphs >= 3 and sentences >= 5:
            return 1.0
        elif paragraphs >= 2 and sentences >= 3:
            return 0.7
        elif paragraphs >= 1 and sentences >= 2:
            return 0.4
        else:
            return 0.1
    
    def _calculate_relevance_score(self, content: str, prompt: str) -> float:
        """Calculate relevance-based quality score"""
        prompt_words = set(prompt.lower().split())
        content_words = set(content.lower().split())
        overlap = len(prompt_words.intersection(content_words))
        
        if overlap >= 5:
            return 1.0
        elif overlap >= 3:
            return 0.7
        elif overlap >= 1:
            return 0.4
        else:
            return 0.1
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability-based quality score"""
        try:
            words = content.split()
            sentences = content.count('.') + content.count('!') + content.count('?')
            
            if sentences == 0:
                return 0.1
            
            avg_sentence_length = len(words) / sentences
            
            if 10 <= avg_sentence_length <= 25:
                return 1.0
            elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 35:
                return 0.7
            elif 3 <= avg_sentence_length < 5 or 35 < avg_sentence_length <= 50:
                return 0.4
            else:
                return 0.1
                
        except Exception as e:
            logger.warning(f"Error calculating readability score: {str(e)}")
            return 0.5
    
    def _calculate_coherence_score(self, content: str) -> float:
        """Calculate coherence-based quality score"""
        try:
            # Simple coherence check based on transition words and sentence flow
            transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 'additionally', 'in contrast', 'on the other hand']
            
            content_lower = content.lower()
            transition_count = sum(1 for word in transition_words if word in content_lower)
            
            # Check for repetitive sentence starts
            sentences = re.split(r'[.!?]+\s*', content)
            sentence_starts = [s.split()[0].lower() if s.split() else '' for s in sentences if s.strip()]
            unique_starts = len(set(sentence_starts))
            
            # Calculate coherence score
            transition_score = min(transition_count / 3, 1.0)  # Max 3 transitions
            variety_score = min(unique_starts / len(sentence_starts), 1.0) if sentence_starts else 0
            
            return (transition_score + variety_score) / 2
            
        except Exception as e:
            logger.warning(f"Error calculating coherence score: {str(e)}")
            return 0.5


class TokenEstimator:
    """Estimates token count for text"""
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        try:
            # Simple estimation: ~4 characters per token for English
            # This is a rough approximation
            return len(text) // 4
        except Exception:
            return 0


class ContextCompressor:
    """Compresses text to fit within context limits"""
    
    async def compress_prompt(self, prompt: str) -> str:
        """
        Compress a prompt to reduce token count
        
        Args:
            prompt: Original prompt
            
        Returns:
            str: Compressed prompt
        """
        try:
            # Simple compression: remove redundant words and phrases
            compressed = re.sub(r'\s+', ' ', prompt)  # Remove extra spaces
            compressed = re.sub(r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', '', compressed)  # Remove common words
            compressed = compressed.strip()
            
            # If still too long, truncate intelligently
            if len(compressed) > 2000:
                sentences = re.split(r'[.!?]+\s*', compressed)
                compressed = '. '.join(sentences[:3]) + '.'
            
            return compressed
            
        except Exception as e:
            logger.warning(f"Error compressing prompt: {str(e)}")
            return prompt[:2000]  # Fallback truncation

# Example usage and testing
if __name__ == "__main__":
    async def test_workflow_engine():
        """Test the workflow chain engine"""
        engine = WorkflowChainEngine()
        
        # Create a new workflow chain
        chain = await engine.create_workflow_chain(
            name="Blog Series: AI and Content Creation",
            description="A series of blog posts about AI-powered content creation",
            initial_prompt="Write an introduction to AI-powered content creation, covering the basics and benefits."
        )
        
        print(f"Created chain: {chain.id}")
        print(f"Initial document: {chain.nodes[chain.root_node_id].title}")
        
        # Continue the chain
        next_doc = await engine.continue_workflow_chain(chain.id)
        print(f"Next document: {next_doc.title}")
        
        # Get chain history
        history = engine.get_chain_history(chain.id)
        print(f"Chain has {len(history)} documents")
    
    # Run test
    asyncio.run(test_workflow_engine())


# Additional utility functions for document processing
class DocumentProcessor:
    """Utility class for advanced document processing"""
    
    @staticmethod
    def estimate_document_pages(content: str) -> int:
        """Estimate number of pages in a document"""
        # Average: 250-500 words per page
        word_count = len(content.split())
        return max(1, word_count // 375)  # 375 words per page average
    
    @staticmethod
    def get_document_statistics(content: str) -> Dict[str, Any]:
        """Get comprehensive document statistics"""
        words = content.split()
        sentences = re.split(r'[.!?]+\\s*', content)
        paragraphs = content.count('\\n\\n') + 1
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'paragraph_count': paragraphs,
            'estimated_pages': DocumentProcessor.estimate_document_pages(content),
            'estimated_tokens': len(content) // 4,
            'average_sentence_length': len(words) / max(len(sentences), 1),
            'reading_time_minutes': len(words) / 200  # 200 words per minute
        }
    
    @staticmethod
    def can_process_with_model(content: str, model_name: str) -> Tuple[bool, str]:
        """Check if content can be processed with a specific model"""
        estimated_tokens = len(content) // 4
        model_limit = ModelContextLimits.get_limit(model_name)
        
        if estimated_tokens <= model_limit:
            return True, f"Content fits within {model_name} context limit ({estimated_tokens}/{model_limit} tokens)"
        else:
            return False, f"Content exceeds {model_name} context limit ({estimated_tokens}/{model_limit} tokens). Consider chunking or using a model with larger context."


# Enhanced workflow chain with multi-model support
class MultiModelWorkflowChain(WorkflowChain):
    """Enhanced workflow chain with multi-model support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_switching_strategy = "auto"  # auto, manual, hybrid
        self.available_models = [
            "claude-3-5-sonnet-20241022",
            "gpt-4-turbo", 
            "gemini-1.5-pro"
        ]
        self.current_model_index = 0
    
    def get_optimal_model(self, content_size: int) -> str:
        """Get the optimal model for content of given size"""
        for model in self.available_models:
            if content_size <= ModelContextLimits.get_limit(model):
                return model
        
        # If no model can handle the content, return the one with largest context
        return max(self.available_models, key=lambda m: ModelContextLimits.get_limit(m))
    
    def switch_model_if_needed(self, content_size: int) -> bool:
        """Switch to optimal model if needed"""
        optimal_model = self.get_optimal_model(content_size)
        if optimal_model != self.available_models[self.current_model_index]:
            self.current_model_index = self.available_models.index(optimal_model)
            logger.info(f"Switched to optimal model: {optimal_model}")
            return True
        return False


# Enhanced WorkflowChainEngine with all advanced features
class EnhancedWorkflowChainEngine(WorkflowChainEngine):
    """Enhanced workflow chain engine with all advanced features"""
    
    async def optimize_prompt_for_workflow(
        self,
        prompt: str,
        workflow_id: str,
        optimization_goals: List[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize prompt for a specific workflow using AI optimization
        
        Args:
            prompt: Original prompt to optimize
            workflow_id: ID of the workflow
            optimization_goals: List of optimization goals
            
        Returns:
            Dict containing optimization results
        """
        try:
            # Get workflow context
            workflow = self.active_chains.get(workflow_id)
            if not workflow:
                return {"error": "Workflow not found"}
            
            # Optimize prompt
            optimization_result = await self.prompt_optimizer.optimize_prompt(
                prompt=prompt,
                target_length=None,
                optimization_goals=optimization_goals or ["improve_clarity", "add_structure"]
            )
            
            return {
                "success": True,
                "original_prompt": optimization_result.original_prompt,
                "optimized_prompt": optimization_result.optimized_prompt,
                "improvement_score": optimization_result.improvement_score,
                "tokens_saved": optimization_result.tokens_saved,
                "quality_improvement": optimization_result.expected_quality_improvement
            }
            
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            return {"error": str(e)}
    
    async def generate_intelligent_content(
        self,
        topic: str,
        content_type: str = "blog",
        target_audience: str = "general",
        tone: str = "professional",
        length_preference: str = "medium",
        quality_requirements: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate content using intelligent generation system
        
        Args:
            topic: Content topic
            content_type: Type of content
            target_audience: Target audience
            tone: Content tone
            length_preference: Length preference
            quality_requirements: Quality requirements
            
        Returns:
            Dict containing generated content and metadata
        """
        try:
            # Create generation context
            context = GenerationContext(
                topic=topic,
                target_audience=target_audience,
                content_type=content_type,
                tone=tone,
                length_preference=length_preference,
                quality_requirements=quality_requirements or ["comprehensive", "engaging"]
            )
            
            # Generate content
            result = await self.intelligent_generator.generate_intelligent_content(
                context=context,
                ai_client=self.ai_client,
                optimization_level="advanced"
            )
            
            return {
                "success": True,
                "content": result.content,
                "title": result.title,
                "quality_score": result.quality_score,
                "engagement_prediction": result.engagement_prediction,
                "seo_score": result.seo_score,
                "readability_score": result.readability_score,
                "confidence_score": result.confidence_score,
                "suggestions": result.suggestions,
                "generation_time": result.generation_time
            }
            
        except Exception as e:
            logger.error(f"Error generating intelligent content: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_trends_for_workflow(
        self,
        workflow_id: str,
        time_period: int = 30,
        categories: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze trends for a specific workflow
        
        Args:
            workflow_id: ID of the workflow
            time_period: Time period for analysis
            categories: Categories to analyze
            
        Returns:
            Dict containing trend analysis results
        """
        try:
            # Get workflow
            workflow = self.active_chains.get(workflow_id)
            if not workflow:
                return {"error": "Workflow not found"}
            
            # Analyze trends
            trend_analysis = await self.trend_analyzer.analyze_trends(
                time_period=time_period,
                categories=categories
            )
            
            # Predict content success for workflow topics
            predictions = {}
            for node in workflow.nodes:
                if node.content_type == "document":
                    prediction = await self.trend_analyzer.predict_content_success(
                        topic=node.topic,
                        content_type="blog",
                        target_audience="general"
                    )
                    predictions[node.id] = {
                        "topic": prediction.topic,
                        "predicted_popularity": prediction.predicted_popularity,
                        "predicted_engagement": prediction.predicted_engagement,
                        "confidence_score": prediction.confidence_score,
                        "recommendations": prediction.content_suggestions
                    }
            
            return {
                "success": True,
                "trend_analysis": trend_analysis,
                "workflow_predictions": predictions
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {"error": str(e)}
    
    async def get_market_intelligence(
        self,
        market_segment: str,
        time_period: int = 90
    ) -> Dict[str, Any]:
        """
        Get market intelligence for content strategy
        
        Args:
            market_segment: Market segment to analyze
            time_period: Time period for analysis
            
        Returns:
            Dict containing market intelligence data
        """
        try:
            market_intel = await self.trend_analyzer.get_market_intelligence(
                market_segment=market_segment,
                time_period=time_period
            )
            
            return {
                "success": True,
                "market_size": market_intel.market_size,
                "competition_level": market_intel.competition_level,
                "opportunity_score": market_intel.opportunity_score,
                "trending_topics": market_intel.trending_topics,
                "emerging_keywords": market_intel.emerging_keywords,
                "audience_insights": market_intel.audience_insights,
                "content_gaps": market_intel.content_gaps,
                "recommendations": market_intel.recommendations
            }
            
        except Exception as e:
            logger.error(f"Error getting market intelligence: {str(e)}")
            return {"error": str(e)}
    
    async def integrate_external_service(
        self,
        service_name: str,
        action: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate with external services
        
        Args:
            service_name: Name of the external service
            action: Action to perform
            data: Data for the action
            
        Returns:
            Dict containing integration results
        """
        try:
            if action == "generate_content":
                result = await self.external_integrations.generate_content_with_ai(
                    service_name=service_name,
                    prompt=data.get("prompt", ""),
                    model=data.get("model"),
                    parameters=data.get("parameters")
                )
            elif action == "publish_content":
                result = await self.external_integrations.publish_content(
                    service_name=service_name,
                    content=data.get("content", ""),
                    title=data.get("title", ""),
                    metadata=data.get("metadata")
                )
            elif action == "send_notification":
                result = await self.external_integrations.send_notification(
                    service_name=service_name,
                    message=data.get("message", ""),
                    recipients=data.get("recipients", []),
                    metadata=data.get("metadata")
                )
            elif action == "analyze_content":
                result = await self.external_integrations.analyze_content(
                    service_name=service_name,
                    content=data.get("content", ""),
                    analysis_type=data.get("analysis_type", "sentiment")
                )
            elif action == "translate_content":
                result = await self.external_integrations.translate_content(
                    service_name=service_name,
                    content=data.get("content", ""),
                    target_language=data.get("target_language", "en"),
                    source_language=data.get("source_language")
                )
            elif action == "generate_media":
                result = await self.external_integrations.generate_media(
                    service_name=service_name,
                    prompt=data.get("prompt", ""),
                    media_type=data.get("media_type", "image"),
                    parameters=data.get("parameters")
                )
            else:
                return {"error": f"Action {action} not supported"}
            
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "response_time": result.response_time,
                "rate_limit_remaining": result.rate_limit_remaining
            }
            
        except Exception as e:
            logger.error(f"Error integrating with external service: {str(e)}")
            return {"error": str(e)}
    
    async def optimize_context_window(
        self,
        workflow_id: str,
        target_tokens: int = 8000
    ) -> Dict[str, Any]:
        """
        Optimize context window for a workflow
        
        Args:
            workflow_id: ID of the workflow
            target_tokens: Target token count
            
        Returns:
            Dict containing optimization results
        """
        try:
            # Get workflow
            workflow = self.active_chains.get(workflow_id)
            if not workflow:
                return {"error": "Workflow not found"}
            
            # Prepare content history
            content_history = []
            for node in workflow.nodes:
                if node.content:
                    content_history.append({
                        "content": node.content,
                        "timestamp": node.created_at.isoformat(),
                        "topic": node.topic,
                        "type": node.content_type
                    })
            
            # Optimize context
            context_window = await self.context_optimizer.optimize_context(
                content_history=content_history,
                current_prompt="Continue workflow",
                target_tokens=target_tokens
            )
            
            return {
                "success": True,
                "max_tokens": context_window.max_tokens,
                "used_tokens": context_window.used_tokens,
                "available_tokens": context_window.available_tokens,
                "compression_ratio": context_window.compression_ratio,
                "priority_sections": context_window.priority_sections
            }
            
        except Exception as e:
            logger.error(f"Error optimizing context window: {str(e)}")
            return {"error": str(e)}
    
    async def get_comprehensive_workflow_insights(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive insights for a workflow using all advanced features
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Dict containing comprehensive insights
        """
        try:
            # Get workflow
            workflow = self.active_chains.get(workflow_id)
            if not workflow:
                return {"error": "Workflow not found"}
            
            # Get basic workflow insights
            basic_insights = await self.get_workflow_insights(workflow_id)
            
            # Get performance analysis
            performance_analysis = await self.analyze_workflow_performance(workflow_id)
            
            # Get trend analysis
            trend_analysis = await self.analyze_trends_for_workflow(workflow_id)
            
            # Get context optimization
            context_optimization = await self.optimize_context_window(workflow_id)
            
            # Combine all insights
            comprehensive_insights = {
                "success": True,
                "workflow_id": workflow_id,
                "basic_insights": basic_insights,
                "performance_analysis": performance_analysis,
                "trend_analysis": trend_analysis,
                "context_optimization": context_optimization,
                "recommendations": [],
                "next_steps": []
            }
            
            # Generate recommendations based on all insights
            recommendations = []
            
            # Performance-based recommendations
            if performance_analysis.get("success"):
                perf_data = performance_analysis.get("performance_summary", {})
                if perf_data.get("average_quality_score", 0) < 0.7:
                    recommendations.append("Improve content quality through better prompts and templates")
                if perf_data.get("average_generation_time", 0) > 30:
                    recommendations.append("Optimize generation speed by using faster models or simpler prompts")
            
            # Trend-based recommendations
            if trend_analysis.get("success"):
                trend_data = trend_analysis.get("trend_analysis", {})
                if trend_data.get("overall_trends", {}).get("trend_momentum", 0) > 0.7:
                    recommendations.append("High trend momentum detected - consider increasing content production")
            
            # Context-based recommendations
            if context_optimization.get("success"):
                context_data = context_optimization
                if context_data.get("compression_ratio", 1.0) < 0.8:
                    recommendations.append("Consider optimizing context usage to improve efficiency")
            
            comprehensive_insights["recommendations"] = recommendations
            
            # Generate next steps
            next_steps = [
                "Monitor workflow performance regularly",
                "Update content templates based on trends",
                "Optimize prompts for better quality",
                "Consider external integrations for enhanced functionality"
            ]
            comprehensive_insights["next_steps"] = next_steps
            
            return comprehensive_insights
            
        except Exception as e:
            logger.error(f"Error getting comprehensive workflow insights: {str(e)}")
            return {"error": str(e)}
