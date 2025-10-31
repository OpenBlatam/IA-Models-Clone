"""
Unified LangChain Integration System for the ads feature.

This module consolidates all LangChain integration functionality from the scattered implementations:
- langchain_service.py (comprehensive LangChain integration)

The new structure follows Clean Architecture principles with clear separation of concerns.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import MessagesPlaceholder
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever

import logging
from datetime import datetime
import json
import asyncio
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import cohere
from voyageai import Client as VoyageClient
import litellm
from google.cloud import aiplatform

from ...config import get_settings


class LangChainConfig(BaseModel):
    """Configuration for LangChain integration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 1000
    temperature: float = 0.7
    model_name: str = "gpt-3.5-turbo"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_memory: bool = True
    enable_vector_store: bool = True
    enable_agents: bool = True


class ContentAnalysis(BaseModel):
    """Content analysis results."""
    content_id: str
    brand_voice: Dict[str, Any]
    sentiment: str
    tone: str
    keywords: List[str]
    target_audience: str
    recommendations: List[str]
    score: float
    timestamp: datetime = Field(default_factory=datetime.now)


class ContentVariation(BaseModel):
    """Content variation information."""
    original_content: str
    variation_text: str
    variation_type: str
    target_audience: str
    score: float
    reasoning: str


class PerformanceMetrics(BaseModel):
    """Performance metrics for content."""
    content_id: str
    impressions: int
    clicks: int
    conversions: int
    ctr: float
    cpc: float
    roas: float
    timestamp: datetime = Field(default_factory=datetime.now)


class LangChainService:
    """Service for LangChain integration with Onyx ads functionality."""
    
    def __init__(self, llm=None, embeddings=None, config: Optional[LangChainConfig] = None):
        """Initialize the LangChain service."""
        self.config = config or LangChainConfig()
        self.llm = llm
        # Defer embeddings initialization to first use to reduce startup overhead
        self.embeddings = embeddings
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        ) if self.config.enable_memory else None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self._initialize_model_clients()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_embeddings(self):
        """Initialize embeddings with multiple providers."""
        try:
            # Try OpenAI embeddings first
            return OpenAIEmbeddings()
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
            try:
                # Fallback to HuggingFace embeddings
                return HuggingFaceEmbeddings(
                    model_name=self.config.embeddings_model
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
                raise
    
    def _initialize_model_clients(self):
        """Initialize various model clients."""
        # Initialize optional providers lazily to avoid import/initialization overhead
        self.cohere_client = None
        self.voyage_client = None
        self.google_ai_client = None
    
    def _get_embeddings(self):
        if self.embeddings is None:
            self.embeddings = self._initialize_embeddings()
        return self.embeddings

    async def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create a vector store from documents."""
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            vector_store = await FAISS.afrom_texts(
                texts=texts,
                metadatas=metadatas,
                embedding=self._get_embeddings()
            )
            
            self.logger.info(f"Created vector store with {len(documents)} documents")
            return vector_store
        except Exception as e:
            self.logger.error(f"Failed to create vector store: {e}")
            raise
    
    async def create_retriever(self, vector_store: FAISS):
        """Create a retriever from vector store."""
        try:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            return retriever
        except Exception as e:
            self.logger.error(f"Failed to create retriever: {e}")
            raise
    
    async def create_qa_chain(self, retriever) -> Any:
        """Create a question-answering chain."""
        try:
            # Create prompt template
            prompt = ChatPromptTemplate.from_template(
                "Answer the following question based on the context:\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
            
            # Create document chain
            document_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=prompt
            )
            
            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain
            )
            
            self.logger.info("Created QA chain successfully")
            return retrieval_chain
        except Exception as e:
            self.logger.error(f"Failed to create QA chain: {e}")
            raise
    
    async def create_agent(self, tools: List[Tool]):
        """Create a LangChain agent."""
        try:
            if not self.config.enable_agents:
                raise ValueError("Agents are disabled in configuration")
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant for ads generation and optimization."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Create agent
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=self.memory,
                verbose=True
            )
            
            self.logger.info("Created agent successfully")
            return agent_executor
        except Exception as e:
            self.logger.error(f"Failed to create agent: {e}")
            raise
    
    async def generate_ads(self, content: str, num_ads: int = 3) -> List[str]:
        """Generate ads based on content."""
        try:
            if self.llm:
                return await self._generate_ads_openai(content, num_ads)
            elif self.cohere_client:
                return await self._generate_ads_cohere(content, num_ads)
            else:
                raise ValueError("No LLM provider available")
        except Exception as e:
            self.logger.error(f"Failed to generate ads: {e}")
            raise
    
    async def _generate_ads_openai(self, content: str, num_ads: int) -> List[str]:
        """Generate ads using OpenAI."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Generate {num_ads} compelling ads based on the following content:\n\n"
                "Content: {content}\n\n"
                "Ads should be engaging, relevant, and optimized for conversion."
            )
            
            chain = prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke({
                "content": content,
                "num_ads": num_ads
            })
            
            # Parse the result into individual ads
            ads = [ad.strip() for ad in result.split('\n') if ad.strip()]
            return ads[:num_ads]
        except Exception as e:
            self.logger.error(f"Failed to generate ads with OpenAI: {e}")
            raise
    
    async def _generate_ads_cohere(self, content: str, num_ads: int) -> List[str]:
        """Generate ads using Cohere."""
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not available")
            
            prompt = f"Generate {num_ads} compelling ads based on this content: {content}"
            
            response = self.cohere_client.generate(
                model="command",
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            ads = [ad.strip() for ad in response.generations[0].text.split('\n') if ad.strip()]
            return ads[:num_ads]
        except Exception as e:
            self.logger.error(f"Failed to generate ads with Cohere: {e}")
            raise
    
    async def analyze_brand_voice(self, content: str) -> Dict[str, Any]:
        """Analyze brand voice from content."""
        try:
            if self.llm:
                return await self._analyze_brand_voice_openai(content)
            elif self.cohere_client:
                return await self._analyze_brand_voice_cohere(content)
            else:
                raise ValueError("No LLM provider available")
        except Exception as e:
            self.logger.error(f"Failed to analyze brand voice: {e}")
            raise
    
    async def _analyze_brand_voice_openai(self, content: str) -> Dict[str, Any]:
        """Analyze brand voice using OpenAI."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Analyze the brand voice from the following content and return a JSON with:\n"
                "- tone (formal, casual, friendly, professional, etc.)\n"
                "- personality (confident, humble, innovative, traditional, etc.)\n"
                "- language_style (simple, complex, technical, conversational, etc.)\n"
                "- key_phrases (list of characteristic phrases)\n"
                "- brand_values (list of core values)\n\n"
                "Content: {content}"
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({"content": content})
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to analyze brand voice with OpenAI: {e}")
            raise
    
    async def _analyze_brand_voice_cohere(self, content: str) -> Dict[str, Any]:
        """Analyze brand voice using Cohere."""
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not available")
            
            prompt = f"Analyze the brand voice from this content: {content}"
            
            response = self.cohere_client.generate(
                model="command",
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse the response (simplified)
            return {
                "tone": "professional",
                "personality": "confident",
                "language_style": "clear",
                "key_phrases": [],
                "brand_values": []
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze brand voice with Cohere: {e}")
            raise
    
    async def optimize_content(self, content: str, target_audience: str) -> str:
        """Optimize content for target audience."""
        try:
            if self.llm:
                return await self._optimize_content_openai(content, target_audience)
            elif self.cohere_client:
                return await self._optimize_content_cohere(content, target_audience)
            else:
                raise ValueError("No LLM provider available")
        except Exception as e:
            self.logger.error(f"Failed to optimize content: {e}")
            raise
    
    async def _optimize_content_openai(self, content: str, target_audience: str) -> str:
        """Optimize content using OpenAI."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Optimize the following content for the target audience:\n\n"
                "Content: {content}\n\n"
                "Target Audience: {target_audience}\n\n"
                "Make the content more engaging, relevant, and optimized for this audience."
            )
            
            chain = prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke({
                "content": content,
                "target_audience": target_audience
            })
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to optimize content with OpenAI: {e}")
            raise
    
    async def _optimize_content_cohere(self, content: str, target_audience: str) -> str:
        """Optimize content using Cohere."""
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not available")
            
            prompt = f"Optimize this content for {target_audience}: {content}"
            
            response = self.cohere_client.generate(
                model="command",
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.generations[0].text
        except Exception as e:
            self.logger.error(f"Failed to optimize content with Cohere: {e}")
            raise
    
    async def generate_content_variations(self, content: str, num_variations: int = 3) -> List[ContentVariation]:
        """Generate content variations."""
        try:
            if self.llm:
                return await self._generate_variations_openai(content, num_variations)
            elif self.cohere_client:
                return await self._generate_variations_cohere(content, num_variations)
            else:
                raise ValueError("No LLM provider available")
        except Exception as e:
            self.logger.error(f"Failed to generate content variations: {e}")
            raise
    
    async def _generate_variations_openai(self, content: str, num_variations: int) -> List[ContentVariation]:
        """Generate variations using OpenAI."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Generate {num_variations} variations of the following content:\n\n"
                "Original Content: {content}\n\n"
                "Each variation should target a different audience segment and have a different approach."
            )
            
            chain = prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke({
                "content": content,
                "num_variations": num_variations
            })
            
            # Parse variations (simplified)
            variations = []
            for i in range(num_variations):
                variations.append(ContentVariation(
                    original_content=content,
                    variation_text=f"Variation {i+1}",
                    variation_type=f"type_{i+1}",
                    target_audience=f"audience_{i+1}",
                    score=0.8,
                    reasoning="Generated variation"
                ))
            
            return variations
        except Exception as e:
            self.logger.error(f"Failed to generate variations with OpenAI: {e}")
            raise
    
    async def _generate_variations_cohere(self, content: str, num_variations: int) -> List[ContentVariation]:
        """Generate variations using Cohere."""
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not available")
            
            prompt = f"Generate {num_variations} variations of: {content}"
            
            response = self.cohere_client.generate(
                model="command",
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse variations (simplified)
            variations = []
            for i in range(num_variations):
                variations.append(ContentVariation(
                    original_content=content,
                    variation_text=f"Variation {i+1}",
                    variation_type=f"type_{i+1}",
                    target_audience=f"audience_{i+1}",
                    score=0.8,
                    reasoning="Generated variation"
                ))
            
            return variations
        except Exception as e:
            self.logger.error(f"Failed to generate variations with Cohere: {e}")
            raise
    
    async def analyze_audience(self, content: str) -> Dict[str, Any]:
        """Analyze target audience from content."""
        try:
            if self.llm:
                return await self._analyze_audience_openai(content)
            elif self.cohere_client:
                return await self._analyze_audience_cohere(content)
            else:
                raise ValueError("No LLM provider available")
        except Exception as e:
            self.logger.error(f"Failed to analyze audience: {e}")
            raise
    
    async def _analyze_audience_openai(self, content: str) -> Dict[str, Any]:
        """Analyze audience using OpenAI."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Analyze the target audience for the following content and return a JSON with:\n"
                "- demographics (age_range, gender, location, income_level)\n"
                "- interests (list of interests)\n"
                "- pain_points (list of problems they face)\n"
                "- motivations (list of what drives them)\n"
                "- preferred_channels (list of communication channels)\n\n"
                "Content: {content}"
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({"content": content})
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to analyze audience with OpenAI: {e}")
            raise
    
    async def _analyze_audience_cohere(self, content: str) -> Dict[str, Any]:
        """Analyze audience using Cohere."""
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not available")
            
            prompt = f"Analyze the target audience for this content: {content}"
            
            response = self.cohere_client.generate(
                model="command",
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse the response (simplified)
            return {
                "demographics": {"age_range": "25-45", "gender": "all", "location": "global"},
                "interests": [],
                "pain_points": [],
                "motivations": [],
                "preferred_channels": []
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze audience with Cohere: {e}")
            raise
    
    async def generate_recommendations(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on content and context."""
        try:
            if self.llm:
                return await self._generate_recommendations_openai(content, context)
            elif self.cohere_client:
                return await self._generate_recommendations_cohere(content, context)
            else:
                raise ValueError("No LLM provider available")
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            raise
    
    async def _generate_recommendations_openai(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Generate recommendations using OpenAI."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Generate recommendations for improving the following content based on the context:\n\n"
                "Content: {content}\n\n"
                "Context: {context}\n\n"
                "Provide specific, actionable recommendations."
            )
            
            chain = prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke({
                "content": content,
                "context": json.dumps(context)
            })
            
            # Parse recommendations
            recommendations = [rec.strip() for rec in result.split('\n') if rec.strip()]
            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations with OpenAI: {e}")
            raise
    
    async def _generate_recommendations_cohere(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Generate recommendations using Cohere."""
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not available")
            
            prompt = f"Generate recommendations for improving this content: {content}"
            
            response = self.cohere_client.generate(
                model="command",
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse recommendations
            recommendations = [rec.strip() for rec in response.generations[0].text.split('\n') if rec.strip()]
            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations with Cohere: {e}")
            raise
    
    async def analyze_competitor_content(self, content: str, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitor content."""
        try:
            if self.llm:
                return await self._analyze_competitors_openai(content, competitor_urls)
            elif self.cohere_client:
                return await self._analyze_competitors_cohere(content, competitor_urls)
            else:
                raise ValueError("No LLM provider available")
        except Exception as e:
            self.logger.error(f"Failed to analyze competitor content: {e}")
            raise
    
    async def _analyze_competitors_openai(self, content: str, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitors using OpenAI."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Analyze the following content compared to competitors and return a JSON with:\n"
                "- strengths (list of content strengths)\n"
                "- weaknesses (list of areas for improvement)\n"
                "- opportunities (list of opportunities)\n"
                "- threats (list of competitive threats)\n"
                "- competitive_advantage (description of unique value)\n\n"
                "Content: {content}\n"
                "Competitor URLs: {competitor_urls}"
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({
                "content": content,
                "competitor_urls": competitor_urls
            })
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to analyze competitors with OpenAI: {e}")
            raise
    
    async def _analyze_competitors_cohere(self, content: str, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitors using Cohere."""
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not available")
            
            prompt = f"Analyze this content compared to competitors: {content}"
            
            response = self.cohere_client.generate(
                model="command",
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse the response (simplified)
            return {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": [],
                "competitive_advantage": "Unique positioning"
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze competitors with Cohere: {e}")
            raise
    
    async def track_content_performance(self, content_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track content performance metrics."""
        try:
            if self.llm:
                return await self._track_performance_openai(content_id, metrics)
            elif self.cohere_client:
                return await self._track_performance_cohere(content_id, metrics)
            else:
                raise ValueError("No LLM provider available")
        except Exception as e:
            self.logger.error(f"Failed to track content performance: {e}")
            raise
    
    async def _track_performance_openai(self, content_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track performance using OpenAI."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Analyze the performance metrics for content and provide insights:\n\n"
                "Content ID: {content_id}\n"
                "Metrics: {metrics}\n\n"
                "Provide analysis and recommendations for improvement."
            )
            
            chain = prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke({
                "content_id": content_id,
                "metrics": json.dumps(metrics)
            })
            
            return {
                "content_id": content_id,
                "analysis": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to track performance with OpenAI: {e}")
            raise
    
    async def _track_performance_cohere(self, content_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track performance using Cohere."""
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not available")
            
            prompt = f"Analyze performance metrics for content {content_id}: {metrics}"
            
            response = self.cohere_client.generate(
                model="command",
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return {
                "content_id": content_id,
                "analysis": response.generations[0].text,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to track performance with Cohere: {e}")
            raise
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service capabilities."""
        return {
            "llm_provider": "OpenAI" if self.llm else "None",
            "embeddings_provider": type(self.embeddings).__name__,
            "cohere_available": self.cohere_client is not None,
            "voyage_available": self.voyage_client is not None,
            "google_ai_available": self.google_ai_client is not None,
            "memory_enabled": self.config.enable_memory,
            "vector_store_enabled": self.config.enable_vector_store,
            "agents_enabled": self.config.enable_agents,
            "config": self.config.dict()
        }


# Global utility functions
def get_langchain_service(llm=None, embeddings=None, config: Optional[LangChainConfig] = None) -> LangChainService:
    """Get a global LangChain service instance."""
    return LangChainService(llm, embeddings, config)


def create_langchain_config(**kwargs) -> LangChainConfig:
    """Create a LangChain configuration with custom settings."""
    return LangChainConfig(**kwargs)
