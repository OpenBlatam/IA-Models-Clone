from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser, JsonOutputParser
from .base import BaseProvider
from ..config.providers import OpenAIConfig
from typing import Any, List, Dict, Optional
import asyncio
"""
OpenAI provider implementation.
"""

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: OpenAIConfig):
        
    """__init__ function."""
super().__init__(config)
        self.chat_model = None
        self.embeddings = None
    
    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        try:
            self.chat_model = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=self.config.api_key,
                openai_organization=self.config.organization
            )
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.config.api_key,
                openai_organization=self.config.organization
            )
            self._initialized = True
            self.logger.info("OpenAI provider initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        self._validate_initialization()
        self._log_operation("generate_text", prompt=prompt)
        
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", kwargs.get("system_prompt", "You are a helpful assistant.")),
                ("user", prompt)
            ])
            chain = chat_prompt | self.chat_model | StrOutputParser()
            return await chain.ainvoke({})
        except Exception as e:
            self._log_error("generate_text", e)
            raise
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON using OpenAI."""
        self._validate_initialization()
        self._log_operation("generate_json", prompt=prompt)
        
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that responds in valid JSON format."),
                ("user", prompt)
            ])
            chain = chat_prompt | self.chat_model | JsonOutputParser()
            return await chain.ainvoke({})
        except Exception as e:
            self._log_error("generate_json", e)
            raise
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        self._validate_initialization()
        self._log_operation("generate_embeddings", text=text)
        
        try:
            return await self.embeddings.aembed_query(text)
        except Exception as e:
            self._log_error("generate_embeddings", e)
            raise
    
    async def analyze_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Analyze text using OpenAI."""
        self._validate_initialization()
        self._log_operation("analyze_text", text=text)
        
        try:
            system_prompt = kwargs.get("system_prompt", "You are an expert in text analysis.")
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", f"Analyze the following text and provide insights:\n\n{text}")
            ])
            chain = chat_prompt | self.chat_model | JsonOutputParser()
            return await chain.ainvoke({})
        except Exception as e:
            self._log_error("analyze_text", e)
            raise
    
    async def optimize_text(self, text: str, target: str, **kwargs) -> str:
        """Optimize text using OpenAI."""
        self._validate_initialization()
        self._log_operation("optimize_text", text=text, target=target)
        
        try:
            system_prompt = kwargs.get("system_prompt", "You are an expert in text optimization.")
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", f"Optimize the following text for the target: {target}\n\nText: {text}")
            ])
            chain = chat_prompt | self.chat_model | StrOutputParser()
            return await chain.ainvoke({})
        except Exception as e:
            self._log_error("optimize_text", e)
            raise
    
    async def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare texts using OpenAI."""
        self._validate_initialization()
        self._log_operation("compare_texts", text1=text1, text2=text2)
        
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert in text comparison."),
                ("user", f"Compare the following texts and provide insights:\n\nText 1: {text1}\n\nText 2: {text2}")
            ])
            chain = chat_prompt | self.chat_model | JsonOutputParser()
            return await chain.ainvoke({})
        except Exception as e:
            self._log_error("compare_texts", e)
            raise
    
    async def generate_variations(self, text: str, num_variations: int = 3, **kwargs) -> List[str]:
        """Generate variations using OpenAI."""
        self._validate_initialization()
        self._log_operation("generate_variations", text=text, num_variations=num_variations)
        
        try:
            system_prompt = kwargs.get("system_prompt", "You are an expert in content creation.")
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", f"Generate {num_variations} variations of the following text while maintaining the core message:\n\n{text}")
            ])
            chain = chat_prompt | self.chat_model | JsonOutputParser()
            result = await chain.ainvoke({})
            return result.get("variations", [])
        except Exception as e:
            self._log_error("generate_variations", e)
            raise
    
    async def analyze_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics using OpenAI."""
        self._validate_initialization()
        self._log_operation("analyze_metrics", metrics=metrics)
        
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert in metrics analysis."),
                ("user", f"Analyze the following metrics and provide insights:\n\n{metrics}")
            ])
            chain = chat_prompt | self.chat_model | JsonOutputParser()
            return await chain.ainvoke({})
        except Exception as e:
            self._log_error("analyze_metrics", e)
            raise 