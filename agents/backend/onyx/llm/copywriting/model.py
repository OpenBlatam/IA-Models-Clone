from typing import List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field
from langchain.schema.messages import BaseMessage, HumanMessage

from onyx.llm.factory import get_default_llms
from onyx.llm.exceptions import GenAIDisabledException
from onyx.llm.utils import dict_based_prompt_to_langchain_prompt, message_to_string
from onyx.utils.logger import setup_logger

if TYPE_CHECKING:
    from onyx.db.models import ChatMessage

logger = setup_logger()

class CopywritingInput(BaseModel):
    """Input model for copywriting generation."""
    product_description: str = Field(..., description="Description of the product or service")
    target_platform: str = Field(..., description="Platform where the copy will be used")
    tone: str = Field(..., description="Desired tone of the copy")
    target_audience: Optional[str] = Field(None, description="Target audience for the copy")
    key_points: Optional[List[str]] = Field(None, description="Key points to highlight")

class CopywritingOutput(BaseModel):
    """Output model for copywriting generation."""
    headline: str = Field(..., description="Main headline for the copy")
    primary_text: str = Field(..., description="Main body text of the copy")
    hashtags: Optional[List[str]] = Field(None, description="Relevant hashtags")
    platform_tips: Optional[str] = Field(None, description="Platform-specific tips")

    def to_langchain_msg(self) -> BaseMessage:
        """Convert the output to a LangChain message."""
        content = f"""Headline: {self.headline}

Primary Text: {self.primary_text}

Hashtags: {' '.join(self.hashtags) if self.hashtags else 'None'}

Platform Tips: {self.platform_tips if self.platform_tips else 'None'}"""
        return HumanMessage(content=content)

class CopywritingModel:
    """Model for generating copywriting content."""
    
    @staticmethod
    async def generate(input_data: CopywritingInput) -> CopywritingOutput:
        """
        Generate copywriting content.
        
        Args:
            input_data: CopywritingInput containing generation parameters
            
        Returns:
            CopywritingOutput with generated content
        """
        try:
            _, fast_llm = get_default_llms(timeout=10)
        except GenAIDisabledException:
            logger.error("Unable to generate copywriting - Gen AI disabled")
            raise

        prompt = f"""Generate advertising copy for {input_data.target_platform} with the following specifications:
        
        Product Description: {input_data.product_description}
        Target Platform: {input_data.target_platform}
        Tone: {input_data.tone}
        Target Audience: {input_data.target_audience or 'General audience'}
        Key Points: {', '.join(input_data.key_points) if input_data.key_points else 'None specified'}
        
        Please provide:
        1. A compelling headline
        2. Primary advertising text
        3. Relevant hashtags
        4. Platform-specific tips for optimization
        """

        messages = [{"role": "user", "content": prompt}]
        filled_llm_prompt = dict_based_prompt_to_langchain_prompt(messages)
        
        try:
            model_output = message_to_string(fast_llm.invoke(filled_llm_prompt))
            logger.debug(f"Generated copywriting: {model_output}")
            
            sections = model_output.split('\n\n')
            
            return CopywritingOutput(
                headline=sections[0].strip(),
                primary_text=sections[1].strip(),
                hashtags=[tag.strip() for tag in sections[2].split() if tag.startswith('#')] if len(sections) > 2 else None,
                platform_tips=sections[3].strip() if len(sections) > 3 else None
            )
            
        except Exception as e:
            logger.error(f"Error generating copywriting: {str(e)}")
            raise



