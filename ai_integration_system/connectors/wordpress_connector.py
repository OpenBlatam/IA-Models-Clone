"""
WordPress Connector for AI Integration System
Handles integration with WordPress CMS via REST API
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import base64

from ..integration_engine import PlatformConnector, IntegrationResult, IntegrationStatus

logger = logging.getLogger(__name__)

class WordPressConnector(PlatformConnector):
    """WordPress platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url")  # e.g., "https://yoursite.com"
        self.username = config.get("username")
        self.password = config.get("password")
        self.application_password = config.get("application_password")
        self.api_url = f"{self.base_url}/wp-json/wp/v2"
        
    async def authenticate(self) -> bool:
        """Authenticate with WordPress REST API"""
        try:
            if not all([self.base_url, self.username, (self.password or self.application_password)]):
                logger.error("Missing required WordPress credentials")
                return False
            
            # Test authentication by getting current user
            auth_url = f"{self.api_url}/users/me"
            
            # Use application password if available, otherwise use basic auth
            if self.application_password:
                auth_string = f"{self.username}:{self.application_password}"
                auth_header = base64.b64encode(auth_string.encode()).decode()
                headers = {"Authorization": f"Basic {auth_header}"}
            else:
                auth_string = f"{self.username}:{self.password}"
                auth_header = base64.b64encode(auth_string.encode()).decode()
                headers = {"Authorization": f"Basic {auth_header}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(auth_url, headers=headers) as response:
                    if response.status == 200:
                        logger.info("WordPress authentication successful")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"WordPress authentication failed: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"WordPress authentication error: {str(e)}")
            return False
    
    async def create_content(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create content in WordPress (post, page, etc.)"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id=content_data.get("content_id", "unknown"),
                    platform="wordpress",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Determine content type
            content_type = content_data.get("content_type", "").lower()
            
            if "page" in content_type:
                return await self._create_page(content_data)
            else:
                # Default to creating a post
                return await self._create_post(content_data)
                
        except Exception as e:
            logger.error(f"Error creating WordPress content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="wordpress",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_post(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create a WordPress post"""
        try:
            # Prepare post data
            post_data = {
                "title": content_data.get("title", "AI Generated Post"),
                "content": self._format_content(content_data.get("content", "")),
                "status": content_data.get("status", "draft"),  # draft, publish, private
                "excerpt": content_data.get("excerpt", content_data.get("content", "")[:160]),
                "categories": await self._get_or_create_categories(content_data.get("category", "AI Generated")),
                "tags": await self._get_or_create_tags(content_data.get("tags", [])),
                "meta": {
                    "ai_generated": True,
                    "generated_by": content_data.get("author", "AI Assistant"),
                    "original_content_id": content_data.get("content_id", "unknown")
                }
            }
            
            # Create post
            create_url = f"{self.api_url}/posts"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=post_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        post_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="wordpress",
                            status=IntegrationStatus.COMPLETED,
                            external_id=str(post_id),
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="wordpress",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create post: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating WordPress post: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="wordpress",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_page(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create a WordPress page"""
        try:
            # Prepare page data
            page_data = {
                "title": content_data.get("title", "AI Generated Page"),
                "content": self._format_content(content_data.get("content", "")),
                "status": content_data.get("status", "draft"),
                "excerpt": content_data.get("excerpt", content_data.get("content", "")[:160]),
                "meta": {
                    "ai_generated": True,
                    "generated_by": content_data.get("author", "AI Assistant"),
                    "original_content_id": content_data.get("content_id", "unknown")
                }
            }
            
            # Create page
            create_url = f"{self.api_url}/pages"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=page_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        page_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="wordpress",
                            status=IntegrationStatus.COMPLETED,
                            external_id=str(page_id),
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="wordpress",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create page: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating WordPress page: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="wordpress",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    def _format_content(self, content: str) -> str:
        """Format content for WordPress"""
        # Convert line breaks to HTML
        formatted_content = content.replace('\n', '<br>')
        
        # Add basic HTML structure if needed
        if not formatted_content.startswith('<'):
            formatted_content = f"<p>{formatted_content}</p>"
        
        return formatted_content
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if self.application_password:
            auth_string = f"{self.username}:{self.application_password}"
        else:
            auth_string = f"{self.username}:{self.password}"
        
        auth_header = base64.b64encode(auth_string.encode()).decode()
        return {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/json"
        }
    
    async def _get_or_create_categories(self, category_name: str) -> List[int]:
        """Get or create WordPress categories"""
        try:
            # First, try to find existing category
            search_url = f"{self.api_url}/categories?search={category_name}"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        categories = await response.json()
                        if categories:
                            return [cat["id"] for cat in categories]
            
            # Create new category if not found
            create_url = f"{self.api_url}/categories"
            category_data = {
                "name": category_name,
                "description": f"AI Generated category: {category_name}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=category_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        return [result["id"]]
            
            return []
            
        except Exception as e:
            logger.error(f"Error handling categories: {str(e)}")
            return []
    
    async def _get_or_create_tags(self, tag_names: List[str]) -> List[int]:
        """Get or create WordPress tags"""
        try:
            tag_ids = []
            
            for tag_name in tag_names:
                # First, try to find existing tag
                search_url = f"{self.api_url}/tags?search={tag_name}"
                headers = self._get_auth_headers()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(search_url, headers=headers) as response:
                        if response.status == 200:
                            tags = await response.json()
                            if tags:
                                tag_ids.append(tags[0]["id"])
                                continue
                
                # Create new tag if not found
                create_url = f"{self.api_url}/tags"
                tag_data = {
                    "name": tag_name,
                    "description": f"AI Generated tag: {tag_name}"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(create_url, json=tag_data, headers=headers) as response:
                        if response.status in [200, 201]:
                            result = await response.json()
                            tag_ids.append(result["id"])
            
            return tag_ids
            
        except Exception as e:
            logger.error(f"Error handling tags: {str(e)}")
            return []
    
    async def update_content(self, external_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update existing content in WordPress"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id=content_data.get("content_id", "unknown"),
                    platform="wordpress",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Determine if it's a post or page
            content_type = content_data.get("content_type", "").lower()
            endpoint = "pages" if "page" in content_type else "posts"
            
            # Prepare update data
            update_data = {
                "title": content_data.get("title", "Updated AI Generated Content"),
                "content": self._format_content(content_data.get("content", "")),
                "excerpt": content_data.get("excerpt", content_data.get("content", "")[:160])
            }
            
            # Update content
            update_url = f"{self.api_url}/{endpoint}/{external_id}"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(update_url, json=update_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="wordpress",
                            status=IntegrationStatus.COMPLETED,
                            external_id=external_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="wordpress",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to update content: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error updating WordPress content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="wordpress",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def delete_content(self, external_id: str) -> IntegrationResult:
        """Delete content from WordPress"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="wordpress",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Delete content (try posts first, then pages)
            for endpoint in ["posts", "pages"]:
                delete_url = f"{self.api_url}/{endpoint}/{external_id}"
                headers = self._get_auth_headers()
                
                async with aiohttp.ClientSession() as session:
                    async with session.delete(delete_url, headers=headers) as response:
                        if response.status == 200:
                            result_data = await response.json()
                            return IntegrationResult(
                                request_id="unknown",
                                platform="wordpress",
                                status=IntegrationStatus.COMPLETED,
                                external_id=external_id,
                                response_data=result_data
                            )
            
            return IntegrationResult(
                request_id="unknown",
                platform="wordpress",
                status=IntegrationStatus.FAILED,
                error_message="Content not found"
            )
                        
        except Exception as e:
            logger.error(f"Error deleting WordPress content: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="wordpress",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def get_content_status(self, external_id: str) -> IntegrationResult:
        """Get content status from WordPress"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="wordpress",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Get content (try posts first, then pages)
            for endpoint in ["posts", "pages"]:
                get_url = f"{self.api_url}/{endpoint}/{external_id}"
                headers = self._get_auth_headers()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(get_url, headers=headers) as response:
                        if response.status == 200:
                            result_data = await response.json()
                            return IntegrationResult(
                                request_id="unknown",
                                platform="wordpress",
                                status=IntegrationStatus.COMPLETED,
                                external_id=external_id,
                                response_data=result_data
                            )
            
            return IntegrationResult(
                request_id="unknown",
                platform="wordpress",
                status=IntegrationStatus.FAILED,
                error_message="Content not found"
            )
                        
        except Exception as e:
            logger.error(f"Error getting WordPress content status: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="wordpress",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def publish_content(self, external_id: str) -> IntegrationResult:
        """Publish content in WordPress"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="wordpress",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Update status to published
            update_data = {"status": "publish"}
            
            # Try posts first, then pages
            for endpoint in ["posts", "pages"]:
                update_url = f"{self.api_url}/{endpoint}/{external_id}"
                headers = self._get_auth_headers()
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(update_url, json=update_data, headers=headers) as response:
                        if response.status in [200, 201]:
                            result_data = await response.json()
                            return IntegrationResult(
                                request_id="unknown",
                                platform="wordpress",
                                status=IntegrationStatus.COMPLETED,
                                external_id=external_id,
                                response_data=result_data
                            )
            
            return IntegrationResult(
                request_id="unknown",
                platform="wordpress",
                status=IntegrationStatus.FAILED,
                error_message="Content not found"
            )
                        
        except Exception as e:
            logger.error(f"Error publishing WordPress content: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="wordpress",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )



























