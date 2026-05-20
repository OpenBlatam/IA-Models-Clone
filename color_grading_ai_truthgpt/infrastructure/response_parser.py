"""
Response parsing utilities for OpenRouter Client.

Refactored to consolidate response parsing logic.
"""

from typing import Dict, Any


class OpenRouterResponseParser:
    """
    Parses OpenRouter API responses.
    
    Responsibilities:
    - Extract content from API responses
    - Extract usage information
    - Format response dictionaries
    """
    
    @staticmethod
    def parse_chat_completion(data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Parse OpenRouter chat completion response.
        
        Args:
            data: Raw API response data
            model: Model identifier used in request
            
        Returns:
            Parsed response dictionary with:
            - response: Content text
            - tokens_used: Total tokens used
            - model: Model identifier
            - finish_reason: Finish reason
            - usage: Usage details
            - id: Response ID
            - created: Creation timestamp
        """
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        
        return {
            "response": content,
            "tokens_used": tokens_used,
            "model": data.get("model", model),
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "id": data.get("id"),
            "created": data.get("created")
        }




