from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from agents.copywriting.model import CopywritingRequest, CopywritingResponse, CopywritingError
from agents.copywriting.propmts.facebook import FACEBOOK_AD_PROMPT
from agents.copywriting.propmts.youtube import YOUTUBE_AD_PROMPT

class CopywritingLLM:
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        self.model = ChatOpenAI(model_name=model_name)
        self.prompts = {
            "facebook": FACEBOOK_AD_PROMPT,
            "youtube": YOUTUBE_AD_PROMPT
        }

    def _create_prompt(self, platform: str) -> ChatPromptTemplate:
        if platform not in self.prompts:
            raise ValueError(f"Unsupported platform: {platform}")
        
        return ChatPromptTemplate.from_template(self.prompts[platform])

    def _extract_hashtags(self, text: str) -> List[str]:
        # Simple hashtag extraction - can be enhanced with more sophisticated logic
        words = text.split()
        return [word for word in words if word.startswith("#")]

    def _get_platform_tips(self, platform: str) -> str:
        tips = {
            "facebook": "Remember to keep your copy concise and scannable. Use clear, conversational language and focus on benefits.",
            "youtube": "Keep your ad between 20-30 seconds. Make the first 5 seconds count to prevent skipping."
        }
        return tips.get(platform, "Optimize your content for the specific platform's best practices.")

    def generate_ad_copy(self, request: CopywritingRequest) -> CopywritingResponse:
        try:
            # Create the prompt template
            prompt = self._create_prompt(request.platform.lower())
            
            # Create the chain
            chain = (
                {"message": RunnablePassthrough(), 
                 "target_audience": RunnablePassthrough(),
                 "tone": RunnablePassthrough(),
                 "call_to_action": RunnablePassthrough()}
                | prompt
                | self.model
                | StrOutputParser()
            )

            # Generate the ad copy
            ad_copy = chain.invoke({
                "message": request.message,
                "target_audience": request.target_audience,
                "tone": request.tone,
                "call_to_action": request.call_to_action
            })

            # Extract hashtags and get platform tips
            hashtags = self._extract_hashtags(ad_copy)
            platform_tips = self._get_platform_tips(request.platform.lower())

            return CopywritingResponse(
                ad_copy=ad_copy,
                platform_specific_tips=platform_tips,
                suggested_hashtags=hashtags,
                character_count=len(ad_copy)
            )

        except Exception as e:
            raise CopywritingError(
                error="Failed to generate ad copy",
                details=str(e)
            )

