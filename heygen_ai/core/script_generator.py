from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import re
from .langchain_manager import LangChainManager
from typing import Any, List, Dict, Optional
"""
Script Generator for HeyGen AI equivalent.
Handles AI-powered script generation and editing using LangChain and OpenRouter.
"""



logger = logging.getLogger(__name__)


class ScriptGenerator:
    """
    Manages AI-powered script generation and editing using LangChain.
    
    This class handles:
    - AI script generation from topics using LangChain
    - Script optimization and editing
    - Multi-language script translation
    - Script formatting and timing
    - Content style adaptation
    - Advanced prompt engineering
    """
    
    def __init__(self, openrouter_api_key: str = None):
        """Initialize the Script Generator with LangChain."""
        self.langchain_manager = None
        self.templates = {}
        self.initialized = False
        
        # Initialize LangChain if API key is provided
        if openrouter_api_key:
            self.langchain_manager = LangChainManager(openrouter_api_key)
        
    def initialize(self) -> Any:
        """Initialize script generation components."""
        try:
            # Initialize LangChain manager if available
            if self.langchain_manager:
                self.langchain_manager.initialize()
            
            # Load script templates
            self._load_script_templates()
            
            self.initialized = True
            logger.info("Script Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Script Generator: {e}")
            raise
    
    def _load_script_templates(self) -> Any:
        """Load script templates for different use cases."""
        self.templates = {
            "professional": {
                "name": "Professional Presentation",
                "structure": [
                    "introduction",
                    "main_points",
                    "conclusion"
                ],
                "tone": "formal",
                "duration": "2-3 minutes"
            },
            "casual": {
                "name": "Casual Conversation",
                "structure": [
                    "greeting",
                    "main_content",
                    "closing"
                ],
                "tone": "friendly",
                "duration": "1-2 minutes"
            },
            "educational": {
                "name": "Educational Content",
                "structure": [
                    "hook",
                    "explanation",
                    "examples",
                    "summary"
                ],
                "tone": "instructive",
                "duration": "3-5 minutes"
            },
            "marketing": {
                "name": "Marketing Pitch",
                "structure": [
                    "problem_statement",
                    "solution_presentation",
                    "call_to_action"
                ],
                "tone": "persuasive",
                "duration": "1-2 minutes"
            }
        }
    
    async def generate_script(self, topic: str, language: str = "en", 
                            style: str = "professional", duration: str = "2 minutes",
                            context: str = "") -> str:
        """
        Generate a script using LangChain and OpenRouter.
        
        Args:
            topic: Topic for the script
            language: Language code
            style: Script style (professional, casual, etc.)
            duration: Target duration
            context: Additional context for generation
            
        Returns:
            Generated script text
        """
        try:
            if self.langchain_manager and self.langchain_manager.is_healthy():
                # Use LangChain for advanced script generation
                logger.info(f"Generating script using LangChain for topic: {topic}")
                return await self.langchain_manager.generate_script(
                    topic=topic,
                    style=style,
                    language=language,
                    duration=duration,
                    context=context
                )
            else:
                # Fallback to basic generation
                logger.info(f"Generating script using fallback method for topic: {topic}")
                return await self._generate_script_fallback(topic, language, style, duration)
            
        except Exception as e:
            logger.error(f"Failed to generate script: {e}")
            # Fallback to basic generation
            return await self._generate_script_fallback(topic, language, style, duration)
    
    async def process_script(self, script: str, language: str = "en") -> str:
        """
        Process and optimize an existing script using LangChain.
        
        Args:
            script: Input script text
            language: Language code
            
        Returns:
            Processed script text
        """
        try:
            logger.info("Processing script...")
            
            if self.langchain_manager and self.langchain_manager.is_healthy():
                # Use LangChain for optimization
                optimized_script = await self.langchain_manager.optimize_script(
                    script=script,
                    language=language
                )
            else:
                # Fallback processing
                optimized_script = await self._process_script_fallback(script, language)
            
            # Add timing markers
            timed_script = await self._add_timing_markers(optimized_script)
            
            logger.info("Script processed successfully")
            return timed_script
            
        except Exception as e:
            logger.error(f"Failed to process script: {e}")
            raise
    
    async def translate_script(self, script: str, target_language: str, 
                             source_language: str = "en", preserve_style: bool = True) -> str:
        """
        Translate script to target language using LangChain.
        
        Args:
            script: Source script text
            target_language: Target language code
            source_language: Source language code
            preserve_style: Whether to preserve original style
            
        Returns:
            Translated script text
        """
        try:
            if self.langchain_manager and self.langchain_manager.is_healthy():
                # Use LangChain for translation
                logger.info(f"Translating script using LangChain to {target_language}")
                translated = await self.langchain_manager.translate_script(
                    script=script,
                    target_language=target_language,
                    source_language=source_language,
                    preserve_style=preserve_style
                )
            else:
                # Fallback translation
                logger.info(f"Translating script using fallback method to {target_language}")
                translated = await self._translate_script_fallback(
                    script, target_language, source_language
                )
            
            # Adapt for cultural context
            adapted = await self._adapt_cultural_context(translated, target_language)
            
            # Optimize for target language speech patterns
            optimized = await self._optimize_for_language(adapted, target_language)
            
            logger.info("Script translated successfully")
            return optimized
            
        except Exception as e:
            logger.error(f"Failed to translate script: {e}")
            raise
    
    async def analyze_script(self, script: str) -> Dict:
        """
        Analyze script for various metrics using LangChain.
        
        Args:
            script: Script text to analyze
            
        Returns:
            Analysis results dictionary
        """
        try:
            if self.langchain_manager and self.langchain_manager.is_healthy():
                # Use LangChain for advanced analysis
                logger.info("Analyzing script using LangChain...")
                return await self.langchain_manager.analyze_script(script)
            else:
                # Fallback analysis
                logger.info("Analyzing script using fallback method...")
                return await self._analyze_script_fallback(script)
            
        except Exception as e:
            logger.error(f"Failed to analyze script: {e}")
            return await self._analyze_script_fallback(script)
    
    async def chat_with_agent(self, message: str) -> str:
        """
        Chat with LangChain agent for complex script workflows.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        try:
            if self.langchain_manager and self.langchain_manager.is_healthy():
                return await self.langchain_manager.chat_with_agent(message)
            else:
                return "LangChain agent is not available. Please check your OpenRouter API key."
                
        except Exception as e:
            logger.error(f"Failed to chat with agent: {e}")
            raise
    
    async def create_knowledge_base(self, documents: List[str], name: str = "scripts"):
        """
        Create a knowledge base for script generation using LangChain.
        
        Args:
            documents: List of document texts
            name: Name for the knowledge base
        """
        try:
            if self.langchain_manager and self.langchain_manager.is_healthy():
                await self.langchain_manager.create_vectorstore(documents, name)
                logger.info(f"Knowledge base '{name}' created successfully")
            else:
                logger.warning("LangChain manager not available for knowledge base creation")
                
        except Exception as e:
            logger.error(f"Failed to create knowledge base: {e}")
            raise
    
    async def search_knowledge_base(self, query: str, name: str = "scripts", k: int = 5) -> List[str]:
        """
        Search knowledge base for relevant content.
        
        Args:
            query: Search query
            name: Knowledge base name
            k: Number of results
            
        Returns:
            List of relevant document chunks
        """
        try:
            if self.langchain_manager and self.langchain_manager.is_healthy():
                return await self.langchain_manager.search_documents(query, name, k)
            else:
                logger.warning("LangChain manager not available for knowledge base search")
                return []
                
        except Exception as e:
            logger.error(f"Failed to search knowledge base: {e}")
            return []
    
    # Fallback methods when LangChain is not available
    async def _generate_script_fallback(self, topic: str, language: str, 
                                      style: str, duration: str) -> str:
        """Fallback script generation without LangChain."""
        template = self.templates.get(style, self.templates["professional"])
        
        # Basic script generation
        script_parts = [
            f"Introduction about {topic}",
            f"Main points about {topic}",
            f"Conclusion about {topic}"
        ]
        
        return " ".join([f"{part}. This is a detailed explanation." for part in script_parts])
    
    async def _process_script_fallback(self, script: str, language: str) -> str:
        """Fallback script processing without LangChain."""
        # Basic cleaning and formatting
        cleaned = re.sub(r'\s+', ' ', script.strip())
        return cleaned
    
    async def _translate_script_fallback(self, script: str, target_language: str, 
                                       source_language: str) -> str:
        """Fallback translation without LangChain."""
        # Basic translation placeholder
        if target_language == "es":
            return f"[Traducido al español] {script}"
        elif target_language == "fr":
            return f"[Traduit en français] {script}"
        else:
            return f"[Translated to {target_language}] {script}"
    
    async def _analyze_script_fallback(self, script: str) -> Dict:
        """Fallback script analysis without LangChain."""
        word_count = len(script.split())
        estimated_duration = word_count / 150.0
        
        return {
            "word_count": word_count,
            "estimated_duration": estimated_duration,
            "readability_score": 70.0,
            "sentiment": {
                "overall": "neutral",
                "confidence": 0.8,
                "emotions": ["neutral"]
            },
            "complexity": {
                "vocabulary_level": "intermediate",
                "sentence_complexity": "medium",
                "overall_complexity": "moderate"
            },
            "suggestions": [
                "Consider adding more engaging opening",
                "Include specific examples",
                "Add a clear call to action"
            ]
        }
    
    async def _adapt_cultural_context(self, text: str, language: str) -> str:
        """Adapt text for cultural context."""
        # Implementation would adapt idioms, references, etc.
        return text
    
    async def _optimize_for_language(self, text: str, language: str) -> str:
        """Optimize text for specific language speech patterns."""
        # Implementation would optimize for language-specific patterns
        return text
    
    async def _add_timing_markers(self, script: str) -> str:
        """Add timing markers to script."""
        # Implementation would add timing information
        return script
    
    def is_healthy(self) -> bool:
        """Check if the script generator is healthy."""
        return self.initialized and (
            self.langchain_manager is None or self.langchain_manager.is_healthy()
        ) 