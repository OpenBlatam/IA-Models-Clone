from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
from typing import Any, List, Dict, Optional
"""
LangChain integration service for Onyx ads functionality.
"""


logger = logging.getLogger(__name__)

class LangChainService:
    """Service for LangChain integration with Onyx."""
    
    def __init__(self, llm, embeddings=None) -> Any:
        self.llm = llm
        self.embeddings = embeddings or self._initialize_embeddings()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self._initialize_model_clients()
    
    def _initialize_embeddings(self) -> Any:
        """Initialize embeddings with multiple providers."""
        try:
            # Try OpenAI embeddings first
            return OpenAIEmbeddings()
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
            try:
                # Fallback to HuggingFace embeddings
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
                raise
    
    def _initialize_model_clients(self) -> Any:
        """Initialize various model clients."""
        try:
            # Initialize Cohere client
            self.cohere_client = cohere.Client()
            
            # Initialize Voyage client
            self.voyage_client = VoyageClient()
            
            # Initialize Google AI Platform
            aiplatform.init()
            
            # Initialize LiteLLM
            litellm.set_verbose = True
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
        except Exception as e:
            logger.error(f"Error initializing model clients: {e}")
            raise
    
    async def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create a vector store from documents."""
        texts = self.text_splitter.split_documents(documents)
        return await FAISS.afrom_documents(texts, self.embeddings)
    
    async def create_retriever(self, vector_store: FAISS):
        """Create a retriever from vector store."""
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
    
    async def create_qa_chain(self, retriever) -> Any:
        """Create a QA chain with retriever."""
        # Create the base prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in marketing and advertising. Use the following context to answer the question. If you don't know the answer, just say that you don't know."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )
        
        # Create the retriever chain
        retriever_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )
        
        return retriever_chain
    
    async def create_agent(self, tools: List[Tool]):
        """Create an agent with tools."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in marketing and advertising. Use the following tools to help answer the question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True
        )
    
    async def generate_ads(self, content: str, num_ads: int = 3) -> List[str]:
        """Generate ads using multiple models."""
        try:
            # Try OpenAI first
            ads = await self._generate_ads_openai(content, num_ads)
        except Exception as e:
            logger.warning(f"OpenAI generation failed: {e}")
            try:
                # Fallback to Cohere
                ads = await self._generate_ads_cohere(content, num_ads)
            except Exception as e:
                logger.error(f"Cohere generation failed: {e}")
                raise
        
        return ads
    
    async def _generate_ads_openai(self, content: str, num_ads: int) -> List[str]:
        """Generate ads using OpenAI."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert copywriter for social media ads. Generate concise, engaging ads based on the content."),
            ("human", "Generate {num_ads} ads based on this content:\n{content}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = await chain.ainvoke({
            "num_ads": num_ads,
            "content": content
        })
        
        return [ad.strip() for ad in result.split("\n") if ad.strip()]
    
    async def _generate_ads_cohere(self, content: str, num_ads: int) -> List[str]:
        """Generate ads using Cohere."""
        response = self.cohere_client.generate(
            prompt=f"Generate {num_ads} social media ads based on this content:\n{content}",
            max_tokens=200,
            temperature=0.7,
            k=0,
            p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_sequences=["\n\n"],
            return_likelihoods="NONE"
        )
        
        return [ad.strip() for ad in response.generations[0].text.split("\n") if ad.strip()]
    
    async def analyze_brand_voice(self, content: str) -> Dict[str, Any]:
        """Analyze brand voice using multiple models."""
        try:
            # Try OpenAI first
            analysis = await self._analyze_brand_voice_openai(content)
        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {e}")
            try:
                # Fallback to Cohere
                analysis = await self._analyze_brand_voice_cohere(content)
            except Exception as e:
                logger.error(f"Cohere analysis failed: {e}")
                raise
        
        return analysis
    
    async def _analyze_brand_voice_openai(self, content: str) -> Dict[str, Any]:
        """Analyze brand voice using OpenAI."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in brand voice analysis. Analyze the content and provide insights about the brand voice."),
            ("human", "Analyze the brand voice in this content:\n{content}")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        
        return await chain.ainvoke({"content": content})
    
    async def _analyze_brand_voice_cohere(self, content: str) -> Dict[str, Any]:
        """Analyze brand voice using Cohere."""
        response = self.cohere_client.generate(
            prompt=f"Analyze the brand voice in this content and provide insights in JSON format:\n{content}",
            max_tokens=300,
            temperature=0.3,
            k=0,
            p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_sequences=["\n\n"],
            return_likelihoods="NONE"
        )
        
        return json.loads(response.generations[0].text)
    
    async def optimize_content(self, content: str, target_audience: str) -> str:
        """Optimize content for target audience using multiple models."""
        try:
            # Try OpenAI first
            optimized = await self._optimize_content_openai(content, target_audience)
        except Exception as e:
            logger.warning(f"OpenAI optimization failed: {e}")
            try:
                # Fallback to Cohere
                optimized = await self._optimize_content_cohere(content, target_audience)
            except Exception as e:
                logger.error(f"Cohere optimization failed: {e}")
                raise
        
        return optimized
    
    async def _optimize_content_openai(self, content: str, target_audience: str) -> str:
        """Optimize content using OpenAI."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in content optimization. Optimize the content for the target audience."),
            ("human", "Optimize this content for {target_audience}:\n{content}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        return await chain.ainvoke({
            "content": content,
            "target_audience": target_audience
        })
    
    async def _optimize_content_cohere(self, content: str, target_audience: str) -> str:
        """Optimize content using Cohere."""
        response = self.cohere_client.generate(
            prompt=f"Optimize this content for {target_audience}:\n{content}",
            max_tokens=500,
            temperature=0.7,
            k=0,
            p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_sequences=["\n\n"],
            return_likelihoods="NONE"
        )
        
        return response.generations[0].text.strip()
    
    async def generate_content_variations(self, content: str, num_variations: int = 3) -> List[str]:
        """Generate content variations using multiple models."""
        try:
            # Try OpenAI first
            variations = await self._generate_variations_openai(content, num_variations)
        except Exception as e:
            logger.warning(f"OpenAI variations failed: {e}")
            try:
                # Fallback to Cohere
                variations = await self._generate_variations_cohere(content, num_variations)
            except Exception as e:
                logger.error(f"Cohere variations failed: {e}")
                raise
        
        return variations
    
    async def _generate_variations_openai(self, content: str, num_variations: int) -> List[str]:
        """Generate variations using OpenAI."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in content creation. Generate variations of the content while maintaining the core message."),
            ("human", "Generate {num_variations} variations of this content:\n{content}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = await chain.ainvoke({
            "num_variations": num_variations,
            "content": content
        })
        
        return [variation.strip() for variation in result.split("\n") if variation.strip()]
    
    async def _generate_variations_cohere(self, content: str, num_variations: int) -> List[str]:
        """Generate variations using Cohere."""
        response = self.cohere_client.generate(
            prompt=f"Generate {num_variations} variations of this content while maintaining the core message:\n{content}",
            max_tokens=500,
            temperature=0.8,
            k=0,
            p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            stop_sequences=["\n\n"],
            return_likelihoods="NONE"
        )
        
        return [variation.strip() for variation in response.generations[0].text.split("\n") if variation.strip()]
    
    async def analyze_audience(self, content: str) -> Dict[str, Any]:
        """Analyze audience from content using multiple models."""
        try:
            # Try OpenAI first
            analysis = await self._analyze_audience_openai(content)
        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {e}")
            try:
                # Fallback to Cohere
                analysis = await self._analyze_audience_cohere(content)
            except Exception as e:
                logger.error(f"Cohere analysis failed: {e}")
                raise
        
        return analysis
    
    async def _analyze_audience_openai(self, content: str) -> Dict[str, Any]:
        """Analyze audience using OpenAI."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in audience analysis. Analyze the content and provide insights about the target audience."),
            ("human", "Analyze the target audience from this content:\n{content}")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        
        return await chain.ainvoke({"content": content})
    
    async def _analyze_audience_cohere(self, content: str) -> Dict[str, Any]:
        """Analyze audience using Cohere."""
        response = self.cohere_client.generate(
            prompt=f"Analyze the target audience from this content and provide insights in JSON format:\n{content}",
            max_tokens=300,
            temperature=0.3,
            k=0,
            p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_sequences=["\n\n"],
            return_likelihoods="NONE"
        )
        
        return json.loads(response.generations[0].text)
    
    async def generate_recommendations(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Generate recommendations using multiple models."""
        try:
            # Try OpenAI first
            recommendations = await self._generate_recommendations_openai(content, context)
        except Exception as e:
            logger.warning(f"OpenAI recommendations failed: {e}")
            try:
                # Fallback to Cohere
                recommendations = await self._generate_recommendations_cohere(content, context)
            except Exception as e:
                logger.error(f"Cohere recommendations failed: {e}")
                raise
        
        return recommendations
    
    async def _generate_recommendations_openai(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Generate recommendations using OpenAI."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in marketing strategy. Generate recommendations based on the content and context."),
            ("human", "Generate recommendations based on this content and context:\nContent: {content}\nContext: {context}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = await chain.ainvoke({
            "content": content,
            "context": json.dumps(context)
        })
        
        return [rec.strip() for rec in result.split("\n") if rec.strip()]
    
    async def _generate_recommendations_cohere(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Generate recommendations using Cohere."""
        response = self.cohere_client.generate(
            prompt=f"Generate recommendations based on this content and context:\nContent: {content}\nContext: {json.dumps(context)}",
            max_tokens=500,
            temperature=0.7,
            k=0,
            p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_sequences=["\n\n"],
            return_likelihoods="NONE"
        )
        
        return [rec.strip() for rec in response.generations[0].text.split("\n") if rec.strip()]
    
    async def analyze_competitor_content(self, content: str, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitor content using multiple models."""
        try:
            # Try OpenAI first
            analysis = await self._analyze_competitors_openai(content, competitor_urls)
        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {e}")
            try:
                # Fallback to Cohere
                analysis = await self._analyze_competitors_cohere(content, competitor_urls)
            except Exception as e:
                logger.error(f"Cohere analysis failed: {e}")
                raise
        
        return analysis
    
    async def _analyze_competitors_openai(self, content: str, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitors using OpenAI."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in competitive analysis. Analyze the content and provide insights about competitors."),
            ("human", "Analyze this content and compare it with competitors:\nContent: {content}\nCompetitor URLs: {competitor_urls}")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        
        return await chain.ainvoke({
            "content": content,
            "competitor_urls": json.dumps(competitor_urls)
        })
    
    async def _analyze_competitors_cohere(self, content: str, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitors using Cohere."""
        response = self.cohere_client.generate(
            prompt=f"Analyze this content and compare it with competitors, provide insights in JSON format:\nContent: {content}\nCompetitor URLs: {json.dumps(competitor_urls)}",
            max_tokens=500,
            temperature=0.3,
            k=0,
            p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_sequences=["\n\n"],
            return_likelihoods="NONE"
        )
        
        return json.loads(response.generations[0].text)
    
    async def track_content_performance(self, content_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track content performance using multiple models."""
        try:
            # Try OpenAI first
            analysis = await self._track_performance_openai(content_id, metrics)
        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {e}")
            try:
                # Fallback to Cohere
                analysis = await self._track_performance_cohere(content_id, metrics)
            except Exception as e:
                logger.error(f"Cohere analysis failed: {e}")
                raise
        
        return analysis
    
    async def _track_performance_openai(self, content_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track performance using OpenAI."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in content performance analysis. Analyze the metrics and provide insights."),
            ("human", "Analyze these performance metrics for content {content_id}:\n{metrics}")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        
        return await chain.ainvoke({
            "content_id": content_id,
            "metrics": json.dumps(metrics)
        })
    
    async def _track_performance_cohere(self, content_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track performance using Cohere."""
        response = self.cohere_client.generate(
            prompt=f"Analyze these performance metrics for content {content_id} and provide insights in JSON format:\n{json.dumps(metrics)}",
            max_tokens=300,
            temperature=0.3,
            k=0,
            p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_sequences=["\n\n"],
            return_likelihoods="NONE"
        )
        
        return json.loads(response.generations[0].text) 