"""
LLM Providers
Integration with Claude, Gemini, and other LLMs for script processing
"""

from typing import Optional, Dict, Any, List
import logging
import os
import httpx

logger = logging.getLogger(__name__)


class LLMProvider:
    """Base class for LLM providers"""
    
    async def enhance_script(self, script: str, language: str = "es") -> str:
        """
        Enhance script with LLM
        
        Args:
            script: Original script
            language: Language code
            
        Returns:
            Enhanced script
        """
        raise NotImplementedError


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1"
    
    async def enhance_script(self, script: str, language: str = "es") -> str:
        """Enhance script using Claude"""
        if not self.api_key:
            logger.warning("Claude API key not configured")
            return script
        
        prompt = f"""You are a professional video script enhancer. 
Enhance the following script to make it more engaging, clear, and suitable for video narration.
Keep the original meaning and style.
Language: {language}

Script:
{script}

Enhanced script:"""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-3-sonnet-20240229",
                        "max_tokens": 2000,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                enhanced = result.get("content", [{}])[0].get("text", script)
                logger.info("Script enhanced with Claude")
                return enhanced
                
        except Exception as e:
            logger.error(f"Claude enhancement failed: {str(e)}")
            return script


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    async def enhance_script(self, script: str, language: str = "es") -> str:
        """Enhance script using Gemini"""
        if not self.api_key:
            logger.warning("Gemini API key not configured")
            return script
        
        prompt = f"""Enhance this video script to make it more engaging and clear.
Keep the original meaning. Language: {language}

Script:
{script}"""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/models/gemini-pro:generateContent?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [{
                                "text": prompt
                            }]
                        }]
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                enhanced = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", script)
                logger.info("Script enhanced with Gemini")
                return enhanced
                
        except Exception as e:
            logger.error(f"Gemini enhancement failed: {str(e)}")
            return script


class OpenAILLMProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
    
    async def enhance_script(self, script: str, language: str = "es") -> str:
        """Enhance script using GPT"""
        if not self.api_key:
            logger.warning("OpenAI API key not configured")
            return script
        
        prompt = f"""Enhance this video script to make it more engaging and professional.
Keep the original meaning. Language: {language}

Script:
{script}"""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a professional video script enhancer."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 2000,
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                enhanced = result.get("choices", [{}])[0].get("message", {}).get("content", script)
                logger.info("Script enhanced with GPT")
                return enhanced
                
        except Exception as e:
            logger.error(f"GPT enhancement failed: {str(e)}")
            return script


def get_llm_provider(provider: str = "openai") -> LLMProvider:
    """
    Get LLM provider instance
    
    Args:
        provider: Provider name (claude, gemini, openai)
        
    Returns:
        LLM provider instance
    """
    providers = {
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        "openai": OpenAILLMProvider,
    }
    
    provider_class = providers.get(provider.lower(), OpenAILLMProvider)
    return provider_class()

