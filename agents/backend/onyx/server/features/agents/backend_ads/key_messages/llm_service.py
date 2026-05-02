import logging
from typing import Dict, Any, Optional
from llm_interface import call_deepseek_api
from .models import MessageType, MessageTone

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMKeyMessageService:
    def __init__(self):
        self.system_prompt = """You are an expert copywriter and marketing strategist. 
        Your task is to generate compelling key messages that resonate with the target audience.
        Focus on clarity, impact, and alignment with the brand's voice."""

    async def generate_message(
        self,
        message: str,
        message_type: MessageType,
        tone: MessageTone,
        target_audience: Optional[str] = None,
        context: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            # Construct the prompt
            prompt = f"""Generate a {message_type.value} message with a {tone.value} tone.
            Original message: {message}
            """
            
            if target_audience:
                prompt += f"\nTarget audience: {target_audience}"
            if context:
                prompt += f"\nContext: {context}"
            if keywords:
                prompt += f"\nKeywords to include: {', '.join(keywords)}"
            if max_length:
                prompt += f"\nMaximum length: {max_length} characters"

            # Call the LLM
            response = await call_deepseek_api(
                prompt=prompt,
                system_prompt=self.system_prompt
            )

            # Process and return the response
            return {
                "success": True,
                "data": {
                    "id": str(uuid.uuid4()),
                    "original_message": message,
                    "response": response,
                    "message_type": message_type,
                    "tone": tone,
                    "created_at": datetime.utcnow().isoformat(),
                    "word_count": len(response.split()),
                    "character_count": len(response),
                    "keywords_used": keywords or [],
                    "sentiment_score": 0.8,  # Placeholder
                    "readability_score": 0.9  # Placeholder
                },
                "processing_time": 0.5,  # Placeholder
                "suggestions": []  # Placeholder
            }

        except Exception as e:
            logger.error(f"Error generating message: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": 0,
                "suggestions": []
            } 