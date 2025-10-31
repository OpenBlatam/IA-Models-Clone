from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import List, Dict, Any, Optional
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.schema import Document
from .ai_service import AIService
from ..config.providers import ProvidersConfig
from typing import Any, List, Dict, Optional
import asyncio
"""
Integrated service combining Onyx capabilities with ads system.
"""


logger = logging.getLogger(__name__)

class IntegratedService:
    """Service that integrates Onyx capabilities with ads system."""
    
    def __init__(self, config: Optional[ProvidersConfig] = None):
        
    """__init__ function."""
self.ai_service = AIService(config)
        self.config = config or ProvidersConfig()
        self.logger = logger
        self._initialize_components()
    
    def _initialize_components(self) -> Any:
        """Initialize LangChain components."""
        try:
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.config.openai.api_key,
                openai_organization=self.config.openai.organization
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            self.logger.info("LangChain components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain components: {e}")
            raise
    
    async def process_content(self, content: str) -> Dict[str, Any]:
        """Process content using Onyx capabilities."""
        try:
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Create vector store
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Create retrieval chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.ai_service._get_provider(self.config.default_provider).chat_model,
                retriever=vectorstore.as_retriever(),
                memory=self.memory
            )
            
            # Analyze content
            analysis = await qa_chain.ainvoke({
                "question": "Analyze this content and provide insights about its structure, tone, and key messages."
            })
            
            return {
                "analysis": analysis,
                "chunks": len(chunks),
                "vectorstore": vectorstore
            }
        except Exception as e:
            self.logger.error(f"Content processing failed: {e}")
            raise
    
    async def generate_ads_with_context(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Generate ads using content and context."""
        try:
            # Process content first
            processed = await self.process_content(content)
            
            # Create tools for the agent
            tools = [
                Tool(
                    name="Generate Ads",
                    func=lambda x: self.ai_service.generate_ads(x),
                    description="Generate ads based on content"
                ),
                Tool(
                    name="Analyze Brand Voice",
                    func=lambda x: self.ai_service.analyze_brand_voice(x),
                    description="Analyze brand voice from content"
                ),
                Tool(
                    name="Optimize Content",
                    func=lambda x: self.ai_service.optimize_content(x, context.get("target_audience", "")),
                    description="Optimize content for target audience"
                )
            ]
            
            # Initialize agent
            agent = initialize_agent(
                tools,
                self.ai_service._get_provider(self.config.default_provider).chat_model,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            # Generate ads with context
            prompt = f"""
            Generate ads based on the following content and context:
            
            Content: {content}
            Context: {context}
            Analysis: {processed['analysis']}
            
            Consider the brand voice, target audience, and content analysis when generating ads.
            """
            
            result = await agent.arun(prompt)
            return result.split("\n")
        except Exception as e:
            self.logger.error(f"Ad generation with context failed: {e}")
            raise
    
    async def analyze_competitors_with_onyx(self, content: str, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitors using Onyx capabilities."""
        try:
            # Process main content
            main_content = await self.process_content(content)
            
            # Process competitor content
            competitor_analyses = []
            for url in competitor_urls:
                # Here you would typically fetch content from URL
                # For now, we'll use a placeholder
                competitor_content = f"Content from {url}"
                analysis = await self.process_content(competitor_content)
                competitor_analyses.append(analysis)
            
            # Create comparison chain
            comparison_chain = LLMChain(
                llm=self.ai_service._get_provider(self.config.default_provider).chat_model,
                prompt=PromptTemplate(
                    input_variables=["main_content", "competitor_analyses"],
                    template="""
                    Compare the main content with competitor content:
                    
                    Main Content Analysis: {main_content}
                    Competitor Analyses: {competitor_analyses}
                    
                    Provide detailed comparison and recommendations.
                    """
                )
            )
            
            # Generate comparison
            comparison = await comparison_chain.arun(
                main_content=main_content,
                competitor_analyses=competitor_analyses
            )
            
            return {
                "main_content_analysis": main_content,
                "competitor_analyses": competitor_analyses,
                "comparison": comparison
            }
        except Exception as e:
            self.logger.error(f"Competitor analysis failed: {e}")
            raise
    
    async def track_performance_with_onyx(self, content_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track performance using Onyx capabilities."""
        try:
            # Create performance analysis chain
            analysis_chain = LLMChain(
                llm=self.ai_service._get_provider(self.config.default_provider).chat_model,
                prompt=PromptTemplate(
                    input_variables=["metrics"],
                    template="""
                    Analyze the following performance metrics and provide insights:
                    
                    Metrics: {metrics}
                    
                    Consider trends, patterns, and provide actionable recommendations.
                    """
                )
            )
            
            # Generate analysis
            analysis = await analysis_chain.arun(metrics=metrics)
            
            # Create recommendations chain
            recommendations_chain = LLMChain(
                llm=self.ai_service._get_provider(self.config.default_provider).chat_model,
                prompt=PromptTemplate(
                    input_variables=["analysis", "metrics"],
                    template="""
                    Based on the analysis and metrics, provide specific recommendations:
                    
                    Analysis: {analysis}
                    Metrics: {metrics}
                    
                    Focus on actionable improvements and optimization strategies.
                    """
                )
            )
            
            # Generate recommendations
            recommendations = await recommendations_chain.arun(
                analysis=analysis,
                metrics=metrics
            )
            
            return {
                "analysis": analysis,
                "recommendations": recommendations,
                "metrics": metrics
            }
        except Exception as e:
            self.logger.error(f"Performance tracking failed: {e}")
            raise 