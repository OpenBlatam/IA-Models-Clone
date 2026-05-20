"""
API handler for Contador AI.

Refactored to consolidate API call handling with metrics into a dedicated class.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class APIHandler:
    """
    Handles API calls with metrics and error handling.
    
    Responsibilities:
    - Execute API calls with timing
    - Handle errors consistently
    - Format responses
    - Extract content from responses
    
    Single Responsibility: Handle all API call operations with metrics.
    """
    
    def __init__(self, client, config):
        """
        Initialize API handler.
        
        Args:
            client: OpenRouter client
            config: Contador config
        """
        self.client = client
        self.config = config
    
    async def call_with_metrics(
        self,
        messages: List[Dict[str, Any]],
        service_name: str,
        service_data: Dict[str, Any],
        temperature: Optional[float] = None,
        extract_key: str = "resultado"
    ) -> Dict[str, Any]:
        """
        Call API with metrics and error handling.
        
        Args:
            messages: Conversation messages
            service_name: Name of the service (for error messages)
            service_data: Service-specific data to include in response
            temperature: Optional temperature override
            extract_key: Key to use for extracted content in response
            
        Returns:
            Response dictionary with success status, metrics, and content
        """
        start_time = time.time()
        
        try:
            response = await self.client.generate_completion(
                messages=messages,
                model=self.config.openrouter.default_model,
                temperature=temperature or self.config.openrouter.temperature,
                max_tokens=self.config.openrouter.max_tokens
            )
            
            response_time = time.time() - start_time
            content = self._extract_content(response)
            
            return {
                "success": True,
                **service_data,
                extract_key: content,
                "tiempo_respuesta": response_time,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in {service_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                **service_data
            }
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract content from OpenRouter response.
        
        Args:
            response: API response dictionary
            
        Returns:
            Extracted content string
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting content from response: {e}")
            return "Error al procesar la respuesta"

