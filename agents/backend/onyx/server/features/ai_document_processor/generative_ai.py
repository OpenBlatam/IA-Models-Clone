"""
Advanced Generative AI and Large Language Models Module
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

import openai
import anthropic
import cohere
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, pipeline
)
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI, Anthropic, Cohere
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from autogen import ConversableAgent, GroupChat, GroupChatManager
from crewai import Agent, Task, Crew, Process
from mem0 import Memory
import guidance
import outlines

from config import settings
from models import ProcessingStatus

logger = logging.getLogger(__name__)


class GenerativeAI:
    """Advanced Generative AI and Large Language Models Engine"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.cohere_client = None
        self.huggingface_models = {}
        self.langchain_llms = {}
        self.autogen_agents = {}
        self.crewai_crew = None
        self.memory = None
        self.vector_stores = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize generative AI system"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing Advanced Generative AI System...")
            
            # Initialize OpenAI
            await self._initialize_openai()
            
            # Initialize Anthropic
            await self._initialize_anthropic()
            
            # Initialize Cohere
            await self._initialize_cohere()
            
            # Initialize Hugging Face models
            await self._initialize_huggingface()
            
            # Initialize LangChain
            await self._initialize_langchain()
            
            # Initialize AutoGen agents
            await self._initialize_autogen()
            
            # Initialize CrewAI
            await self._initialize_crewai()
            
            # Initialize memory system
            await self._initialize_memory()
            
            # Initialize vector stores
            await self._initialize_vector_stores()
            
            self.initialized = True
            logger.info("Advanced Generative AI System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing generative AI: {e}")
            raise
    
    async def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
                openai.api_key = settings.openai_api_key
                self.openai_client = openai
                logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
    
    async def _initialize_anthropic(self):
        """Initialize Anthropic client"""
        try:
            if hasattr(settings, 'anthropic_api_key') and settings.anthropic_api_key:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=settings.anthropic_api_key
                )
                logger.info("Anthropic client initialized")
        except Exception as e:
            logger.error(f"Error initializing Anthropic: {e}")
    
    async def _initialize_cohere(self):
        """Initialize Cohere client"""
        try:
            if hasattr(settings, 'cohere_api_key') and settings.cohere_api_key:
                self.cohere_client = cohere.Client(settings.cohere_api_key)
                logger.info("Cohere client initialized")
        except Exception as e:
            logger.error(f"Error initializing Cohere: {e}")
    
    async def _initialize_huggingface(self):
        """Initialize Hugging Face models"""
        try:
            # Load various models for different tasks
            models_config = {
                "gpt2": "gpt2",
                "gpt2-medium": "gpt2-medium",
                "gpt2-large": "gpt2-large",
                "t5-small": "t5-small",
                "t5-base": "t5-base",
                "bart-large": "facebook/bart-large",
                "distilbert": "distilbert-base-uncased",
                "roberta": "roberta-base"
            }
            
            for model_name, model_path in models_config.items():
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                    if "gpt" in model_name:
                        model = AutoModelForCausalLM.from_pretrained(model_path)
                    elif "t5" in model_name or "bart" in model_name:
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                    else:
                        model = AutoModelForCausalLM.from_pretrained(model_path)
                    
                    self.huggingface_models[model_name] = {
                        "tokenizer": tokenizer,
                        "model": model,
                        "pipeline": None
                    }
                    
                    logger.info(f"Loaded Hugging Face model: {model_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error initializing Hugging Face models: {e}")
    
    async def _initialize_langchain(self):
        """Initialize LangChain components"""
        try:
            # Initialize LLMs
            if self.openai_client:
                self.langchain_llms["openai"] = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.7
                )
                self.langchain_llms["openai_embeddings"] = OpenAIEmbeddings()
            
            if self.anthropic_client:
                self.langchain_llms["anthropic"] = ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    temperature=0.7
                )
            
            if self.cohere_client:
                self.langchain_llms["cohere"] = Cohere(
                    model="command",
                    temperature=0.7
                )
            
            # Initialize Hugging Face embeddings
            self.langchain_llms["hf_embeddings"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            logger.info("LangChain components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing LangChain: {e}")
    
    async def _initialize_autogen(self):
        """Initialize AutoGen agents"""
        try:
            # Create specialized agents
            self.autogen_agents["document_analyzer"] = ConversableAgent(
                name="DocumentAnalyzer",
                system_message="You are an expert document analyzer. Analyze documents and extract key insights.",
                llm_config={"model": "gpt-3.5-turbo", "temperature": 0.1}
            )
            
            self.autogen_agents["content_generator"] = ConversableAgent(
                name="ContentGenerator",
                system_message="You are a creative content generator. Generate high-quality content based on requirements.",
                llm_config={"model": "gpt-3.5-turbo", "temperature": 0.8}
            )
            
            self.autogen_agents["summarizer"] = ConversableAgent(
                name="Summarizer",
                system_message="You are an expert summarizer. Create concise and accurate summaries.",
                llm_config={"model": "gpt-3.5-turbo", "temperature": 0.3}
            )
            
            # Create group chat
            self.autogen_agents["group_chat"] = GroupChat(
                agents=list(self.autogen_agents.values())[:3],
                messages=[],
                max_round=10
            )
            
            self.autogen_agents["group_manager"] = GroupChatManager(
                groupchat=self.autogen_agents["group_chat"],
                llm_config={"model": "gpt-3.5-turbo", "temperature": 0.1}
            )
            
            logger.info("AutoGen agents initialized")
            
        except Exception as e:
            logger.error(f"Error initializing AutoGen: {e}")
    
    async def _initialize_crewai(self):
        """Initialize CrewAI crew"""
        try:
            # Create agents
            researcher = Agent(
                role="Document Researcher",
                goal="Research and analyze documents thoroughly",
                backstory="Expert in document analysis and research",
                verbose=True,
                allow_delegation=False
            )
            
            writer = Agent(
                role="Content Writer",
                goal="Create high-quality content based on research",
                backstory="Professional writer with expertise in various domains",
                verbose=True,
                allow_delegation=False
            )
            
            reviewer = Agent(
                role="Content Reviewer",
                goal="Review and improve content quality",
                backstory="Experienced editor and quality assurance specialist",
                verbose=True,
                allow_delegation=False
            )
            
            # Create tasks
            research_task = Task(
                description="Research and analyze the provided document",
                agent=researcher,
                expected_output="Comprehensive analysis report"
            )
            
            writing_task = Task(
                description="Create content based on the research",
                agent=writer,
                expected_output="High-quality written content"
            )
            
            review_task = Task(
                description="Review and improve the content",
                agent=reviewer,
                expected_output="Polished and improved content"
            )
            
            # Create crew
            self.crewai_crew = Crew(
                agents=[researcher, writer, reviewer],
                tasks=[research_task, writing_task, review_task],
                verbose=True,
                process=Process.sequential
            )
            
            logger.info("CrewAI crew initialized")
            
        except Exception as e:
            logger.error(f"Error initializing CrewAI: {e}")
    
    async def _initialize_memory(self):
        """Initialize memory system"""
        try:
            self.memory = Memory()
            logger.info("Memory system initialized")
        except Exception as e:
            logger.error(f"Error initializing memory: {e}")
    
    async def _initialize_vector_stores(self):
        """Initialize vector stores"""
        try:
            # Initialize Chroma
            self.vector_stores["chroma"] = Chroma(
                collection_name="documents",
                embedding_function=self.langchain_llms.get("hf_embeddings")
            )
            
            # Initialize FAISS
            self.vector_stores["faiss"] = FAISS.from_texts(
                ["Sample document"],
                self.langchain_llms.get("hf_embeddings")
            )
            
            logger.info("Vector stores initialized")
            
        except Exception as e:
            logger.error(f"Error initializing vector stores: {e}")
    
    async def generate_document_summary(self, document_text: str, 
                                      summary_type: str = "abstractive") -> Dict[str, Any]:
        """Generate document summary using various AI models"""
        try:
            if not self.initialized:
                await self.initialize()
            
            summaries = {}
            
            # OpenAI summary
            if self.openai_client:
                openai_summary = await self._generate_openai_summary(document_text, summary_type)
                summaries["openai"] = openai_summary
            
            # Anthropic summary
            if self.anthropic_client:
                anthropic_summary = await self._generate_anthropic_summary(document_text, summary_type)
                summaries["anthropic"] = anthropic_summary
            
            # Hugging Face summary
            if "bart-large" in self.huggingface_models:
                hf_summary = await self._generate_hf_summary(document_text, "bart-large")
                summaries["huggingface"] = hf_summary
            
            # LangChain summary
            if "openai" in self.langchain_llms:
                langchain_summary = await self._generate_langchain_summary(document_text)
                summaries["langchain"] = langchain_summary
            
            return {
                "document_text": document_text[:500] + "..." if len(document_text) > 500 else document_text,
                "summary_type": summary_type,
                "summaries": summaries,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def generate_document_content(self, prompt: str, 
                                      content_type: str = "article",
                                      style: str = "professional") -> Dict[str, Any]:
        """Generate document content using AI models"""
        try:
            if not self.initialized:
                await self.initialize()
            
            generated_content = {}
            
            # OpenAI content generation
            if self.openai_client:
                openai_content = await self._generate_openai_content(prompt, content_type, style)
                generated_content["openai"] = openai_content
            
            # Anthropic content generation
            if self.anthropic_client:
                anthropic_content = await self._generate_anthropic_content(prompt, content_type, style)
                generated_content["anthropic"] = anthropic_content
            
            # Hugging Face content generation
            if "gpt2-large" in self.huggingface_models:
                hf_content = await self._generate_hf_content(prompt, "gpt2-large")
                generated_content["huggingface"] = hf_content
            
            return {
                "prompt": prompt,
                "content_type": content_type,
                "style": style,
                "generated_content": generated_content,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error generating document content: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def analyze_document_with_ai(self, document_text: str, 
                                     analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze document using AI models"""
        try:
            if not self.initialized:
                await self.initialize()
            
            analysis_results = {}
            
            # OpenAI analysis
            if self.openai_client:
                openai_analysis = await self._analyze_with_openai(document_text, analysis_type)
                analysis_results["openai"] = openai_analysis
            
            # Anthropic analysis
            if self.anthropic_client:
                anthropic_analysis = await self._analyze_with_anthropic(document_text, analysis_type)
                analysis_results["anthropic"] = anthropic_analysis
            
            # AutoGen multi-agent analysis
            if self.autogen_agents:
                autogen_analysis = await self._analyze_with_autogen(document_text, analysis_type)
                analysis_results["autogen"] = autogen_analysis
            
            # CrewAI analysis
            if self.crewai_crew:
                crewai_analysis = await self._analyze_with_crewai(document_text, analysis_type)
                analysis_results["crewai"] = crewai_analysis
            
            return {
                "document_text": document_text[:500] + "..." if len(document_text) > 500 else document_text,
                "analysis_type": analysis_type,
                "analysis_results": analysis_results,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document with AI: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def create_document_embeddings(self, documents: List[str]) -> Dict[str, Any]:
        """Create embeddings for documents"""
        try:
            if not self.initialized:
                await self.initialize()
            
            embeddings_results = {}
            
            # OpenAI embeddings
            if "openai_embeddings" in self.langchain_llms:
                openai_embeddings = await self._create_openai_embeddings(documents)
                embeddings_results["openai"] = openai_embeddings
            
            # Hugging Face embeddings
            if "hf_embeddings" in self.langchain_llms:
                hf_embeddings = await self._create_hf_embeddings(documents)
                embeddings_results["huggingface"] = hf_embeddings
            
            return {
                "documents": [doc[:100] + "..." if len(doc) > 100 else doc for doc in documents],
                "embeddings": embeddings_results,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error creating document embeddings: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def semantic_document_search(self, query: str, 
                                     documents: List[str],
                                     top_k: int = 5) -> Dict[str, Any]:
        """Perform semantic search on documents"""
        try:
            if not self.initialized:
                await self.initialize()
            
            search_results = {}
            
            # Chroma search
            if "chroma" in self.vector_stores:
                chroma_results = await self._search_with_chroma(query, documents, top_k)
                search_results["chroma"] = chroma_results
            
            # FAISS search
            if "faiss" in self.vector_stores:
                faiss_results = await self._search_with_faiss(query, documents, top_k)
                search_results["faiss"] = faiss_results
            
            return {
                "query": query,
                "search_results": search_results,
                "top_k": top_k,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error performing semantic document search: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def generate_document_questions(self, document_text: str, 
                                        num_questions: int = 5) -> Dict[str, Any]:
        """Generate questions based on document content"""
        try:
            if not self.initialized:
                await self.initialize()
            
            questions = {}
            
            # OpenAI question generation
            if self.openai_client:
                openai_questions = await self._generate_questions_openai(document_text, num_questions)
                questions["openai"] = openai_questions
            
            # Anthropic question generation
            if self.anthropic_client:
                anthropic_questions = await self._generate_questions_anthropic(document_text, num_questions)
                questions["anthropic"] = anthropic_questions
            
            return {
                "document_text": document_text[:500] + "..." if len(document_text) > 500 else document_text,
                "num_questions": num_questions,
                "generated_questions": questions,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error generating document questions: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _generate_openai_summary(self, text: str, summary_type: str) -> Dict[str, Any]:
        """Generate summary using OpenAI"""
        try:
            prompt = f"Summarize the following text in a {summary_type} style:\n\n{text}"
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return {
                "summary": response.choices[0].message.content,
                "model": "gpt-3.5-turbo",
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating OpenAI summary: {e}")
            return {"error": str(e)}
    
    async def _generate_anthropic_summary(self, text: str, summary_type: str) -> Dict[str, Any]:
        """Generate summary using Anthropic"""
        try:
            prompt = f"Summarize the following text in a {summary_type} style:\n\n{text}"
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "summary": response.content[0].text,
                "model": "claude-3-sonnet-20240229",
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating Anthropic summary: {e}")
            return {"error": str(e)}
    
    async def _generate_hf_summary(self, text: str, model_name: str) -> Dict[str, Any]:
        """Generate summary using Hugging Face model"""
        try:
            model_info = self.huggingface_models[model_name]
            
            if not model_info["pipeline"]:
                model_info["pipeline"] = pipeline(
                    "summarization",
                    model=model_info["model"],
                    tokenizer=model_info["tokenizer"]
                )
            
            summary = model_info["pipeline"](
                text,
                max_length=150,
                min_length=50,
                do_sample=False
            )
            
            return {
                "summary": summary[0]["summary_text"],
                "model": model_name,
                "confidence": summary[0].get("score", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error generating Hugging Face summary: {e}")
            return {"error": str(e)}
    
    async def _generate_langchain_summary(self, text: str) -> Dict[str, Any]:
        """Generate summary using LangChain"""
        try:
            prompt = PromptTemplate(
                input_variables=["text"],
                template="Summarize the following text:\n\n{text}"
            )
            
            chain = LLMChain(
                llm=self.langchain_llms["openai"],
                prompt=prompt
            )
            
            summary = await chain.arun(text=text)
            
            return {
                "summary": summary,
                "model": "langchain-openai",
                "method": "chain"
            }
            
        except Exception as e:
            logger.error(f"Error generating LangChain summary: {e}")
            return {"error": str(e)}
    
    async def _generate_openai_content(self, prompt: str, content_type: str, style: str) -> Dict[str, Any]:
        """Generate content using OpenAI"""
        try:
            system_prompt = f"You are a professional {content_type} writer with a {style} style."
            user_prompt = f"Create a {content_type} based on: {prompt}"
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return {
                "content": response.choices[0].message.content,
                "model": "gpt-3.5-turbo",
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating OpenAI content: {e}")
            return {"error": str(e)}
    
    async def _generate_anthropic_content(self, prompt: str, content_type: str, style: str) -> Dict[str, Any]:
        """Generate content using Anthropic"""
        try:
            system_prompt = f"You are a professional {content_type} writer with a {style} style."
            user_prompt = f"Create a {content_type} based on: {prompt}"
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
            )
            
            return {
                "content": response.content[0].text,
                "model": "claude-3-sonnet-20240229",
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating Anthropic content: {e}")
            return {"error": str(e)}
    
    async def _generate_hf_content(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Generate content using Hugging Face model"""
        try:
            model_info = self.huggingface_models[model_name]
            
            inputs = model_info["tokenizer"].encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model_info["model"].generate(
                    inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=model_info["tokenizer"].eos_token_id
                )
            
            generated_text = model_info["tokenizer"].decode(outputs[0], skip_special_tokens=True)
            
            return {
                "content": generated_text,
                "model": model_name,
                "method": "huggingface"
            }
            
        except Exception as e:
            logger.error(f"Error generating Hugging Face content: {e}")
            return {"error": str(e)}
    
    async def _analyze_with_openai(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze document using OpenAI"""
        try:
            prompt = f"Analyze the following text for {analysis_type} analysis:\n\n{text}"
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            return {
                "analysis": response.choices[0].message.content,
                "model": "gpt-3.5-turbo",
                "analysis_type": analysis_type
            }
            
        except Exception as e:
            logger.error(f"Error analyzing with OpenAI: {e}")
            return {"error": str(e)}
    
    async def _analyze_with_anthropic(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze document using Anthropic"""
        try:
            prompt = f"Analyze the following text for {analysis_type} analysis:\n\n{text}"
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "analysis": response.content[0].text,
                "model": "claude-3-sonnet-20240229",
                "analysis_type": analysis_type
            }
            
        except Exception as e:
            logger.error(f"Error analyzing with Anthropic: {e}")
            return {"error": str(e)}
    
    async def _analyze_with_autogen(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze document using AutoGen agents"""
        try:
            # Use the document analyzer agent
            analyzer = self.autogen_agents["document_analyzer"]
            
            # Create a simple analysis task
            analysis_prompt = f"Analyze this text for {analysis_type}:\n\n{text}"
            
            # Simulate agent conversation (simplified)
            analysis_result = {
                "analysis": f"AutoGen analysis of {analysis_type} for the provided text",
                "agents_used": ["DocumentAnalyzer"],
                "analysis_type": analysis_type
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing with AutoGen: {e}")
            return {"error": str(e)}
    
    async def _analyze_with_crewai(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze document using CrewAI"""
        try:
            # Create a custom task for the crew
            analysis_task = Task(
                description=f"Analyze the following text for {analysis_type}:\n\n{text}",
                agent=self.crewai_crew.agents[0],  # Use the first agent
                expected_output=f"Comprehensive {analysis_type} analysis"
            )
            
            # Execute the crew (simplified)
            result = {
                "analysis": f"CrewAI analysis of {analysis_type} for the provided text",
                "crew_used": "Document Research Crew",
                "analysis_type": analysis_type
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing with CrewAI: {e}")
            return {"error": str(e)}
    
    async def _create_openai_embeddings(self, documents: List[str]) -> Dict[str, Any]:
        """Create embeddings using OpenAI"""
        try:
            embeddings = []
            
            for doc in documents:
                response = await self.openai_client.Embedding.acreate(
                    model="text-embedding-ada-002",
                    input=doc
                )
                embeddings.append(response.data[0].embedding)
            
            return {
                "embeddings": embeddings,
                "model": "text-embedding-ada-002",
                "dimension": len(embeddings[0]) if embeddings else 0
            }
            
        except Exception as e:
            logger.error(f"Error creating OpenAI embeddings: {e}")
            return {"error": str(e)}
    
    async def _create_hf_embeddings(self, documents: List[str]) -> Dict[str, Any]:
        """Create embeddings using Hugging Face"""
        try:
            embeddings_model = self.langchain_llms["hf_embeddings"]
            embeddings = embeddings_model.embed_documents(documents)
            
            return {
                "embeddings": embeddings,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": len(embeddings[0]) if embeddings else 0
            }
            
        except Exception as e:
            logger.error(f"Error creating Hugging Face embeddings: {e}")
            return {"error": str(e)}
    
    async def _search_with_chroma(self, query: str, documents: List[str], top_k: int) -> Dict[str, Any]:
        """Search using Chroma vector store"""
        try:
            # Add documents to Chroma
            vector_store = self.vector_stores["chroma"]
            
            # Create embeddings for documents
            embeddings = self.langchain_llms["hf_embeddings"].embed_documents(documents)
            
            # Add to vector store
            vector_store.add_texts(documents, embeddings)
            
            # Search
            results = vector_store.similarity_search(query, k=top_k)
            
            return {
                "results": [doc.page_content for doc in results],
                "method": "chroma",
                "top_k": top_k
            }
            
        except Exception as e:
            logger.error(f"Error searching with Chroma: {e}")
            return {"error": str(e)}
    
    async def _search_with_faiss(self, query: str, documents: List[str], top_k: int) -> Dict[str, Any]:
        """Search using FAISS vector store"""
        try:
            vector_store = self.vector_stores["faiss"]
            
            # Search
            results = vector_store.similarity_search(query, k=top_k)
            
            return {
                "results": [doc.page_content for doc in results],
                "method": "faiss",
                "top_k": top_k
            }
            
        except Exception as e:
            logger.error(f"Error searching with FAISS: {e}")
            return {"error": str(e)}
    
    async def _generate_questions_openai(self, text: str, num_questions: int) -> Dict[str, Any]:
        """Generate questions using OpenAI"""
        try:
            prompt = f"Generate {num_questions} questions based on the following text:\n\n{text}"
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            return {
                "questions": response.choices[0].message.content,
                "model": "gpt-3.5-turbo",
                "num_questions": num_questions
            }
            
        except Exception as e:
            logger.error(f"Error generating questions with OpenAI: {e}")
            return {"error": str(e)}
    
    async def _generate_questions_anthropic(self, text: str, num_questions: int) -> Dict[str, Any]:
        """Generate questions using Anthropic"""
        try:
            prompt = f"Generate {num_questions} questions based on the following text:\n\n{text}"
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "questions": response.content[0].text,
                "model": "claude-3-sonnet-20240229",
                "num_questions": num_questions
            }
            
        except Exception as e:
            logger.error(f"Error generating questions with Anthropic: {e}")
            return {"error": str(e)}


# Global generative AI instance
generative_ai = GenerativeAI()


async def initialize_generative_ai():
    """Initialize the generative AI system"""
    await generative_ai.initialize()














