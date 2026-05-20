from typing import List, Optional
from pydantic import BaseModel
from onyx.llm.factory import get_default_llms
from onyx.llm.exceptions import GenAIDisabledException
from onyx.llm.utils import dict_based_prompt_to_langchain_prompt, message_to_string
from onyx.utils.logger import setup_logger

logger = setup_logger()

class CopywritingRequest(BaseModel):
    """Request model for copywriting generation."""
    product_description: str
    target_platform: str
    tone: str
    target_audience: Optional[str] = None
    key_points: Optional[List[str]] = None

class CopywritingResponse(BaseModel):
    """Response model for copywriting generation."""
    headline: str
    primary_text: str
    hashtags: Optional[List[str]] = None
    platform_tips: Optional[str] = None

def generate_copywriting(request: CopywritingRequest) -> CopywritingResponse:
    """
    Generate copywriting content using a secondary LLM.
    
    Args:
        request: CopywritingRequest containing generation parameters
        
    Returns:
        CopywritingResponse with generated content
    """
    try:
        _, fast_llm = get_default_llms(timeout=10)
    except GenAIDisabledException:
        logger.error("Unable to generate copywriting - Gen AI disabled")
        raise

    prompt = f"""Generate advertising copy for {request.target_platform} with the following specifications:
    
    Product Description: {request.product_description}
    Target Platform: {request.target_platform}
    Tone: {request.tone}
    Target Audience: {request.target_audience or 'General audience'}
    Key Points: {', '.join(request.key_points) if request.key_points else 'None specified'}
    
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
        
        # Parse the model output into structured response
        # This is a simplified parsing - you may want to enhance this based on your needs
        sections = model_output.split('\n\n')
        
        return CopywritingResponse(
            headline=sections[0].strip(),
            primary_text=sections[1].strip(),
            hashtags=[tag.strip() for tag in sections[2].split() if tag.startswith('#')] if len(sections) > 2 else None,
            platform_tips=sections[3].strip() if len(sections) > 3 else None
        )
        
    except Exception as e:
        logger.error(f"Error generating copywriting: {str(e)}")
        raise
