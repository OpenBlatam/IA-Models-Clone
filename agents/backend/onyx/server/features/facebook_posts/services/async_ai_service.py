"""
Async AI Service for Facebook Posts
Following functional programming principles and async patterns
"""

from typing import Dict, Any, Optional, List
import asyncio
import aiohttp
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


# Pure functions for AI service

def build_ai_request_data(
    topic: str,
    content_type: str,
    audience_type: str,
    tone: str,
    language: str,
    max_length: int,
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """Build AI request data - pure function"""
    base_prompt = f"Generate a {content_type} Facebook post about '{topic}' for {audience_type} audience in {tone} tone, maximum {max_length} characters in {language}."
    
    if custom_instructions:
        base_prompt += f" Additional instructions: {custom_instructions}"
    
    return {
        "prompt": base_prompt,
        "max_tokens": max_length,
        "temperature": 0.7,
        "content_type": content_type,
        "audience_type": audience_type,
        "tone": tone,
        "language": language
    }


def parse_ai_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse AI response - pure function"""
    if not response_data or "choices" not in response_data:
        return {"content": "", "error": "Invalid AI response"}
    
    try:
        choice = response_data["choices"][0]
        content = choice.get("text", "").strip()
        
        return {
            "content": content,
            "usage": response_data.get("usage", {}),
            "model": response_data.get("model", "unknown"),
            "created": response_data.get("created", int(datetime.now().timestamp()))
        }
    except (KeyError, IndexError) as e:
        return {"content": "", "error": f"Failed to parse AI response: {str(e)}"}


def validate_ai_content(content: str, max_length: int) -> bool:
    """Validate AI generated content - pure function"""
    if not content or not content.strip():
        return False
    
    if len(content) > max_length:
        return False
    
    return True


def extract_hashtags(content: str) -> List[str]:
    """Extract hashtags from content - pure function"""
    import re
    hashtag_pattern = r'#\w+'
    hashtags = re.findall(hashtag_pattern, content)
    return [tag.lower() for tag in hashtags]


def extract_emojis(content: str) -> List[str]:
    """Extract emojis from content - pure function"""
    import re
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
    emojis = re.findall(emoji_pattern, content)
    return emojis


def analyze_content_sentiment(content: str) -> float:
    """Analyze content sentiment - pure function (mock implementation)"""
    positive_words = ["great", "amazing", "wonderful", "excellent", "fantastic", "love", "best", "awesome"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "horrible", "disappointing"]
    
    content_lower = content.lower()
    positive_count = sum(1 for word in positive_words if word in content_lower)
    negative_count = sum(1 for word in negative_words if word in content_lower)
    
    if positive_count + negative_count == 0:
        return 0.0
    
    return (positive_count - negative_count) / (positive_count + negative_count)


def calculate_engagement_score(content: str, hashtags: List[str], emojis: List[str]) -> float:
    """Calculate engagement score - pure function"""
    base_score = 0.5
    
    # Length factor
    if 100 <= len(content) <= 280:
        base_score += 0.2
    elif 50 <= len(content) < 100:
        base_score += 0.1
    
    # Hashtag factor
    if 1 <= len(hashtags) <= 5:
        base_score += 0.2
    elif len(hashtags) > 5:
        base_score += 0.1
    
    # Emoji factor
    if 1 <= len(emojis) <= 3:
        base_score += 0.1
    
    return min(1.0, max(0.0, base_score))


# Async AI Service Class

class AsyncAIService:
    """Async AI Service following functional principles"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.error_count = 0
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize AI service"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        logger.info("AI service initialized")
    
    async def cleanup(self) -> None:
        """Cleanup AI service"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("AI service cleaned up")
    
    async def generate_content(self, request_data: Dict[str, Any]) -> str:
        """Generate content using AI service"""
        if not self.session:
            await self.initialize()
        
        try:
            self.request_count += 1
            
            # Build request data
            ai_request = build_ai_request_data(**request_data)
            
            # Make API call
            response = await self._make_ai_request(ai_request)
            
            # Parse response
            parsed_response = parse_ai_response(response)
            
            if parsed_response.get("error"):
                raise Exception(parsed_response["error"])
            
            content = parsed_response["content"]
            
            # Validate content
            if not validate_ai_content(content, request_data.get("max_length", 280)):
                raise Exception("Generated content validation failed")
            
            return content
            
        except Exception as e:
            self.error_count += 1
            logger.error("Error generating content", error=str(e))
            raise
    
    async def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content for metrics"""
        try:
            # Extract features
            hashtags = extract_hashtags(content)
            emojis = extract_emojis(content)
            
            # Calculate metrics
            sentiment = analyze_content_sentiment(content)
            engagement_score = calculate_engagement_score(content, hashtags, emojis)
            
            return {
                "content_length": len(content),
                "hashtags": hashtags,
                "emojis": emojis,
                "sentiment_score": sentiment,
                "engagement_score": engagement_score,
                "readability_score": 0.8,  # Mock implementation
                "viral_potential": engagement_score * 0.9
            }
            
        except Exception as e:
            logger.error("Error analyzing content", error=str(e))
            return {
                "content_length": len(content),
                "hashtags": [],
                "emojis": [],
                "sentiment_score": 0.0,
                "engagement_score": 0.5,
                "readability_score": 0.5,
                "viral_potential": 0.5
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check AI service health"""
        try:
            if not self.session:
                return {"status": "unhealthy", "error": "Session not initialized"}
            
            # Simple health check
            test_request = {
                "prompt": "Test",
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            start_time = asyncio.get_event_loop().time()
            response = await self._make_ai_request(test_request)
            response_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(1, self.request_count)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "request_count": self.request_count,
                "error_count": self.error_count
            }
    
    async def _make_ai_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make AI API request"""
        if not self.session:
            raise Exception("AI service not initialized")
        
        try:
            url = f"{self.base_url}/chat/completions"
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": request_data["prompt"]}
                ],
                "max_tokens": request_data["max_tokens"],
                "temperature": request_data["temperature"]
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"AI API error {response.status}: {error_text}")
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise Exception(f"AI API connection error: {str(e)}")
        except Exception as e:
            raise Exception(f"AI API request failed: {str(e)}")


# Factory functions for service creation

def create_ai_service(api_key: str, model: str = "gpt-3.5-turbo") -> AsyncAIService:
    """Create AI service instance - pure function"""
    return AsyncAIService(api_key, model)


async def get_ai_service(api_key: str, model: str = "gpt-3.5-turbo") -> AsyncAIService:
    """Get AI service instance with initialization"""
    service = create_ai_service(api_key, model)
    await service.initialize()
    return service


# Utility functions for AI operations

async def generate_post_content_async(
    topic: str,
    content_type: str,
    audience_type: str,
    tone: str,
    language: str,
    max_length: int,
    ai_service: AsyncAIService,
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """Generate post content asynchronously"""
    try:
        request_data = {
            "topic": topic,
            "content_type": content_type,
            "audience_type": audience_type,
            "tone": tone,
            "language": language,
            "max_length": max_length,
            "custom_instructions": custom_instructions
        }
        
        content = await ai_service.generate_content(request_data)
        analysis = await ai_service.analyze_content(content)
        
        return {
            "content": content,
            "analysis": analysis,
            "success": True
        }
        
    except Exception as e:
        return {
            "content": "",
            "analysis": {},
            "success": False,
            "error": str(e)
        }


async def batch_generate_content_async(
    requests: List[Dict[str, Any]],
    ai_service: AsyncAIService
) -> List[Dict[str, Any]]:
    """Generate content for multiple requests asynchronously"""
    tasks = []
    
    for request in requests:
        task = generate_post_content_async(
            topic=request["topic"],
            content_type=request["content_type"],
            audience_type=request["audience_type"],
            tone=request.get("tone", "professional"),
            language=request.get("language", "en"),
            max_length=request.get("max_length", 280),
            ai_service=ai_service,
            custom_instructions=request.get("custom_instructions")
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "content": "",
                "analysis": {},
                "success": False,
                "error": str(result)
            })
        else:
            processed_results.append(result)
    
    return processed_results