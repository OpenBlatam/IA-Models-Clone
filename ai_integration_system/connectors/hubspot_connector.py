"""
HubSpot Connector for AI Integration System
Handles integration with HubSpot CRM and Marketing Hub
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..integration_engine import PlatformConnector, IntegrationResult, IntegrationStatus

logger = logging.getLogger(__name__)

class HubSpotConnector(PlatformConnector):
    """HubSpot platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.access_token = config.get("access_token")
        self.base_url = "https://api.hubapi.com"
        self.portal_id = config.get("portal_id")
        
    async def authenticate(self) -> bool:
        """Authenticate with HubSpot API"""
        try:
            if not (self.api_key or self.access_token):
                logger.error("Missing required HubSpot credentials")
                return False
            
            # Test authentication by getting account info
            if self.access_token:
                auth_url = f"{self.base_url}/crm/v3/objects/contacts"
                headers = {"Authorization": f"Bearer {self.access_token}"}
            else:
                auth_url = f"{self.base_url}/crm/v3/objects/contacts"
                headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(auth_url, headers=headers) as response:
                    if response.status == 200:
                        logger.info("HubSpot authentication successful")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"HubSpot authentication failed: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"HubSpot authentication error: {str(e)}")
            return False
    
    async def create_content(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create content in HubSpot"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id=content_data.get("content_id", "unknown"),
                    platform="hubspot",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Determine content type and create accordingly
            content_type = content_data.get("content_type", "").lower()
            
            if "blog" in content_type or "post" in content_type:
                return await self._create_blog_post(content_data)
            elif "email" in content_type or "campaign" in content_type:
                return await self._create_email_campaign(content_data)
            elif "landing" in content_type or "page" in content_type:
                return await self._create_landing_page(content_data)
            elif "contact" in content_type or "lead" in content_type:
                return await self._create_contact(content_data)
            else:
                # Default to creating a blog post
                return await self._create_blog_post(content_data)
                
        except Exception as e:
            logger.error(f"Error creating HubSpot content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_blog_post(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create a blog post in HubSpot"""
        try:
            # Prepare blog post data
            blog_data = {
                "name": content_data.get("title", "AI Generated Blog Post"),
                "content_group_id": content_data.get("content_group_id", "default"),
                "post_body": self._format_blog_content(content_data.get("content", "")),
                "meta_description": content_data.get("excerpt", content_data.get("content", "")[:160]),
                "publish_date": int(datetime.now().timestamp() * 1000),
                "state": "DRAFT"  # Start as draft
            }
            
            # Create blog post
            create_url = f"{self.base_url}/content/api/v2/blog-posts"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=blog_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        blog_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.COMPLETED,
                            external_id=str(blog_id),
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create blog post: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating HubSpot blog post: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_email_campaign(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create an email campaign in HubSpot"""
        try:
            # Prepare email campaign data
            campaign_data = {
                "name": content_data.get("title", "AI Generated Email Campaign"),
                "type": "REGULAR",
                "subject": content_data.get("title", "AI Generated Email"),
                "html_content": self._generate_email_html(content_data),
                "text_content": content_data.get("content", ""),
                "from_name": content_data.get("author", "AI Assistant"),
                "from_email": content_data.get("from_email", "noreply@example.com"),
                "reply_to": content_data.get("reply_to", "noreply@example.com")
            }
            
            # Create email campaign
            create_url = f"{self.base_url}/email/public/v1/campaigns"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=campaign_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        campaign_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.COMPLETED,
                            external_id=str(campaign_id),
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create email campaign: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating HubSpot email campaign: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_landing_page(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create a landing page in HubSpot"""
        try:
            # Prepare landing page data
            page_data = {
                "name": content_data.get("title", "AI Generated Landing Page"),
                "html_title": content_data.get("title", "AI Generated Landing Page"),
                "meta_description": content_data.get("excerpt", content_data.get("content", "")[:160]),
                "html_content": self._generate_landing_page_html(content_data),
                "state": "DRAFT"
            }
            
            # Create landing page
            create_url = f"{self.base_url}/content/api/v2/pages"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=page_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        page_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.COMPLETED,
                            external_id=str(page_id),
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create landing page: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating HubSpot landing page: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_contact(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create a contact in HubSpot"""
        try:
            # Prepare contact data
            contact_data = {
                "properties": {
                    "firstname": content_data.get("first_name", "AI"),
                    "lastname": content_data.get("last_name", "Generated"),
                    "email": content_data.get("email", "ai@example.com"),
                    "company": content_data.get("company", "AI Generated Company"),
                    "jobtitle": content_data.get("job_title", "AI Generated"),
                    "phone": content_data.get("phone", ""),
                    "website": content_data.get("website", ""),
                    "notes_last_contacted": content_data.get("content", ""),
                    "lifecyclestage": "lead"
                }
            }
            
            # Create contact
            create_url = f"{self.base_url}/crm/v3/objects/contacts"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=contact_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        contact_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.COMPLETED,
                            external_id=str(contact_id),
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create contact: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating HubSpot contact: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    def _format_blog_content(self, content: str) -> str:
        """Format content for HubSpot blog posts"""
        # Convert line breaks to HTML
        formatted_content = content.replace('\n', '<br>')
        
        # Add basic HTML structure
        if not formatted_content.startswith('<'):
            formatted_content = f"<p>{formatted_content}</p>"
        
        return formatted_content
    
    def _generate_email_html(self, content_data: Dict[str, Any]) -> str:
        """Generate HTML content for email campaigns"""
        title = content_data.get("title", "AI Generated Email")
        content = content_data.get("content", "")
        author = content_data.get("author", "AI Assistant")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; }}
                .header {{ background-color: #f4f4f4; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ background-color: #f4f4f4; padding: 10px; text-align: center; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>By {author}</p>
            </div>
            <div class="content">
                {content.replace(chr(10), '<br>')}
            </div>
            <div class="footer">
                <p>Generated by AI Integration System</p>
            </div>
        </body>
        </html>
        """
        return html
    
    def _generate_landing_page_html(self, content_data: Dict[str, Any]) -> str:
        """Generate HTML content for landing pages"""
        title = content_data.get("title", "AI Generated Landing Page")
        content = content_data.get("content", "")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .content {{ margin-bottom: 30px; }}
                .cta {{ text-align: center; background-color: #007cba; color: white; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                </div>
                <div class="content">
                    {content.replace(chr(10), '<br>')}
                </div>
                <div class="cta">
                    <h2>Ready to Get Started?</h2>
                    <p>Contact us today to learn more!</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if self.access_token:
            return {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
        else:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
    
    async def update_content(self, external_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update existing content in HubSpot"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id=content_data.get("content_id", "unknown"),
                    platform="hubspot",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Determine content type and update accordingly
            content_type = content_data.get("content_type", "").lower()
            
            if "blog" in content_type or "post" in content_type:
                return await self._update_blog_post(external_id, content_data)
            elif "email" in content_type or "campaign" in content_type:
                return await self._update_email_campaign(external_id, content_data)
            elif "landing" in content_type or "page" in content_type:
                return await self._update_landing_page(external_id, content_data)
            else:
                return await self._update_blog_post(external_id, content_data)
                
        except Exception as e:
            logger.error(f"Error updating HubSpot content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _update_blog_post(self, blog_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update a blog post in HubSpot"""
        try:
            update_data = {
                "name": content_data.get("title", "Updated AI Generated Blog Post"),
                "post_body": self._format_blog_content(content_data.get("content", "")),
                "meta_description": content_data.get("excerpt", content_data.get("content", "")[:160])
            }
            
            update_url = f"{self.base_url}/content/api/v2/blog-posts/{blog_id}"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(update_url, json=update_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.COMPLETED,
                            external_id=blog_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to update blog post: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error updating HubSpot blog post: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _update_email_campaign(self, campaign_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update an email campaign in HubSpot"""
        try:
            update_data = {
                "name": content_data.get("title", "Updated AI Generated Email Campaign"),
                "subject": content_data.get("title", "Updated AI Generated Email"),
                "html_content": self._generate_email_html(content_data),
                "text_content": content_data.get("content", "")
            }
            
            update_url = f"{self.base_url}/email/public/v1/campaigns/{campaign_id}"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(update_url, json=update_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.COMPLETED,
                            external_id=campaign_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to update email campaign: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error updating HubSpot email campaign: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _update_landing_page(self, page_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update a landing page in HubSpot"""
        try:
            update_data = {
                "name": content_data.get("title", "Updated AI Generated Landing Page"),
                "html_title": content_data.get("title", "Updated AI Generated Landing Page"),
                "meta_description": content_data.get("excerpt", content_data.get("content", "")[:160]),
                "html_content": self._generate_landing_page_html(content_data)
            }
            
            update_url = f"{self.base_url}/content/api/v2/pages/{page_id}"
            headers = self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(update_url, json=update_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.COMPLETED,
                            external_id=page_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="hubspot",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to update landing page: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error updating HubSpot landing page: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def delete_content(self, external_id: str) -> IntegrationResult:
        """Delete content from HubSpot"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="hubspot",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Try to delete from different endpoints
            endpoints = [
                f"{self.base_url}/content/api/v2/blog-posts/{external_id}",
                f"{self.base_url}/content/api/v2/pages/{external_id}",
                f"{self.base_url}/email/public/v1/campaigns/{external_id}"
            ]
            
            headers = self._get_auth_headers()
            
            for delete_url in endpoints:
                async with aiohttp.ClientSession() as session:
                    async with session.delete(delete_url, headers=headers) as response:
                        if response.status == 204:
                            return IntegrationResult(
                                request_id="unknown",
                                platform="hubspot",
                                status=IntegrationStatus.COMPLETED,
                                external_id=external_id,
                                response_data={"status": "deleted"}
                            )
            
            return IntegrationResult(
                request_id="unknown",
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message="Content not found"
            )
                        
        except Exception as e:
            logger.error(f"Error deleting HubSpot content: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def get_content_status(self, external_id: str) -> IntegrationResult:
        """Get content status from HubSpot"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="hubspot",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Try to get from different endpoints
            endpoints = [
                f"{self.base_url}/content/api/v2/blog-posts/{external_id}",
                f"{self.base_url}/content/api/v2/pages/{external_id}",
                f"{self.base_url}/email/public/v1/campaigns/{external_id}"
            ]
            
            headers = self._get_auth_headers()
            
            for get_url in endpoints:
                async with aiohttp.ClientSession() as session:
                    async with session.get(get_url, headers=headers) as response:
                        if response.status == 200:
                            result_data = await response.json()
                            return IntegrationResult(
                                request_id="unknown",
                                platform="hubspot",
                                status=IntegrationStatus.COMPLETED,
                                external_id=external_id,
                                response_data=result_data
                            )
            
            return IntegrationResult(
                request_id="unknown",
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message="Content not found"
            )
                        
        except Exception as e:
            logger.error(f"Error getting HubSpot content status: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="hubspot",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )



























