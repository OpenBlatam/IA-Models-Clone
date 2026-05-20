from typing import List, Optional, Dict
import numpy as np
from threading import Lock
import mmh3
import orjson
from .models import (
    AdditionalContentRequest,
    AdditionalContentResponse,
    Hashtag,
    CallToAction,
    Link,
    ErrorResponse
)

class AdditionalContentService:
    """Service for generating additional content like hashtags, CTAs, and links."""
    
    def __init__(self):
        """Initialize the service with caches and locks."""
        self.hashtag_cache = {}
        self.cta_cache = {}
        self.link_cache = {}
        self.hashtag_lock = Lock()
        self.cta_lock = Lock()
        self.link_lock = Lock()
        
    def _generate_cache_key(self, text: str, platform: str, content_type: str) -> str:
        """Generate a cache key using MurmurHash3."""
        key = f"{text}:{platform}:{content_type}"
        return str(mmh3.hash(key))
    
    async def generate_additional_content(
        self,
        request: AdditionalContentRequest
    ) -> AdditionalContentResponse:
        """Generate additional content based on the request."""
        try:
            cache_key = self._generate_cache_key(
                request.text,
                request.platform,
                request.content_type
            )
            
            # Generate hashtags
            hashtags = await self._generate_hashtags(
                request.text,
                request.platform,
                request.max_hashtags,
                cache_key
            )
            
            # Generate CTA if requested
            cta = None
            if request.include_cta:
                cta = await self._generate_cta(
                    request.text,
                    request.platform,
                    request.tone,
                    cache_key
                )
            
            # Generate suggested links
            links = await self._generate_links(
                request.text,
                request.platform,
                request.content_type,
                cache_key
            )
            
            # Combine everything into the full text
            full_text = self._combine_content(
                request.text,
                hashtags,
                cta,
                links
            )
            
            return AdditionalContentResponse(
                hashtags=hashtags,
                call_to_action=cta,
                suggested_links=links,
                full_text=full_text,
                metadata={
                    "platform": request.platform,
                    "content_type": request.content_type,
                    "tone": request.tone
                }
            )
            
        except Exception as e:
            raise ErrorResponse(
                error="Failed to generate additional content",
                details={"error": str(e)}
            )
    
    async def _generate_hashtags(
        self,
        text: str,
        platform: str,
        max_hashtags: int,
        cache_key: str
    ) -> List[Hashtag]:
        """Generate relevant hashtags for the content."""
        with self.hashtag_lock:
            if cache_key in self.hashtag_cache:
                return self.hashtag_cache[cache_key]
            
            # TODO: Implement actual hashtag generation logic
            # This is a placeholder implementation
            hashtags = [
                Hashtag(
                    tag=f"#{word.lower()}",
                    relevance_score=np.random.random(),
                    popularity_score=np.random.random()
                )
                for word in text.split()[:max_hashtags]
            ]
            
            self.hashtag_cache[cache_key] = hashtags
            return hashtags
    
    async def _generate_cta(
        self,
        text: str,
        platform: str,
        tone: str,
        cache_key: str
    ) -> Optional[CallToAction]:
        """Generate a call to action for the content."""
        with self.cta_lock:
            if cache_key in self.cta_cache:
                return self.cta_cache[cache_key]
            
            # TODO: Implement actual CTA generation logic
            # This is a placeholder implementation
            cta = CallToAction(
                text="Click here to learn more!",
                type="click",
                relevance_score=np.random.random()
            )
            
            self.cta_cache[cache_key] = cta
            return cta
    
    async def _generate_links(
        self,
        text: str,
        platform: str,
        content_type: str,
        cache_key: str
    ) -> List[Link]:
        """Generate relevant links for the content."""
        with self.link_lock:
            if cache_key in self.link_cache:
                return self.link_cache[cache_key]
            
            # TODO: Implement actual link generation logic
            # This is a placeholder implementation
            links = [
                Link(
                    text="Learn more",
                    url="https://example.com",
                    relevance_score=np.random.random()
                )
            ]
            
            self.link_cache[cache_key] = links
            return links
    
    def _combine_content(
        self,
        text: str,
        hashtags: List[Hashtag],
        cta: Optional[CallToAction],
        links: List[Link]
    ) -> str:
        """Combine all content elements into a single text."""
        result = [text]
        
        if cta:
            result.append(f"\n\n{cta.text}")
        
        if links:
            result.append("\n\nLinks:")
            for link in links:
                result.append(f"- {link.text}: {link.url}")
        
        if hashtags:
            result.append("\n\n" + " ".join(h.tag for h in hashtags))
        
        return "\n".join(result)
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        with self.hashtag_lock:
            self.hashtag_cache.clear()
        with self.cta_lock:
            self.cta_cache.clear()
        with self.link_lock:
            self.link_cache.clear() 