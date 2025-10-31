"""
Mailchimp Connector for AI Integration System
Handles integration with Mailchimp email marketing platform
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import base64

from ..integration_engine import PlatformConnector, IntegrationResult, IntegrationStatus

logger = logging.getLogger(__name__)

class MailchimpConnector(PlatformConnector):
    """Mailchimp platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.server_prefix = config.get("server_prefix")  # e.g., "us1", "us2", etc.
        self.base_url = f"https://{self.server_prefix}.api.mailchimp.com/3.0"
        self.list_id = config.get("list_id")
        self.audience_id = config.get("audience_id")
        
    async def authenticate(self) -> bool:
        """Authenticate with Mailchimp API"""
        try:
            if not self.api_key or not self.server_prefix:
                logger.error("Missing required Mailchimp credentials")
                return False
            
            # Test authentication by getting account info
            auth_url = f"{self.base_url}/"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(auth_url, headers=headers) as response:
                    if response.status == 200:
                        logger.info("Mailchimp authentication successful")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Mailchimp authentication failed: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Mailchimp authentication error: {str(e)}")
            return False
    
    async def create_content(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create content in Mailchimp (campaign, template, etc.)"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id=content_data.get("content_id", "unknown"),
                    platform="mailchimp",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Determine content type and create accordingly
            content_type = content_data.get("content_type", "").lower()
            
            if "email" in content_type or "campaign" in content_type:
                return await self._create_campaign(content_data)
            elif "template" in content_type:
                return await self._create_template(content_data)
            elif "automation" in content_type:
                return await self._create_automation(content_data)
            else:
                # Default to creating a campaign
                return await self._create_campaign(content_data)
                
        except Exception as e:
            logger.error(f"Error creating Mailchimp content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="mailchimp",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_campaign(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create an email campaign in Mailchimp"""
        try:
            # Prepare campaign data
            campaign_data = {
                "type": "regular",
                "recipients": {
                    "list_id": self.list_id or self.audience_id
                },
                "settings": {
                    "subject_line": content_data.get("title", "AI Generated Email"),
                    "from_name": content_data.get("author", "AI Assistant"),
                    "reply_to": content_data.get("reply_to", "noreply@example.com"),
                    "title": content_data.get("title", "AI Generated Campaign")
                }
            }
            
            # Create campaign
            create_url = f"{self.base_url}/campaigns"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=campaign_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        campaign_id = result_data.get("id")
                        
                        # Set campaign content
                        content_result = await self._set_campaign_content(campaign_id, content_data)
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="mailchimp",
                            status=IntegrationStatus.COMPLETED,
                            external_id=campaign_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="mailchimp",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create campaign: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating Mailchimp campaign: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="mailchimp",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _set_campaign_content(self, campaign_id: str, content_data: Dict[str, Any]) -> bool:
        """Set the content for a Mailchimp campaign"""
        try:
            # Prepare HTML content
            html_content = self._generate_html_content(content_data)
            
            content_payload = {
                "html": html_content,
                "plain_text": content_data.get("content", "")
            }
            
            # Set campaign content
            content_url = f"{self.base_url}/campaigns/{campaign_id}/content"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.put(content_url, json=content_payload, headers=headers) as response:
                    return response.status in [200, 204]
                    
        except Exception as e:
            logger.error(f"Error setting campaign content: {str(e)}")
            return False
    
    async def _create_template(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create a template in Mailchimp"""
        try:
            # Prepare template data
            template_data = {
                "name": content_data.get("title", "AI Generated Template"),
                "html": self._generate_html_content(content_data)
            }
            
            # Create template
            create_url = f"{self.base_url}/templates"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=template_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        template_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="mailchimp",
                            status=IntegrationStatus.COMPLETED,
                            external_id=template_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="mailchimp",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create template: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating Mailchimp template: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="mailchimp",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_automation(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create an automation workflow in Mailchimp"""
        try:
            # Prepare automation data
            automation_data = {
                "recipients": {
                    "list_id": self.list_id or self.audience_id
                },
                "settings": {
                    "title": content_data.get("title", "AI Generated Automation"),
                    "from_name": content_data.get("author", "AI Assistant"),
                    "reply_to": content_data.get("reply_to", "noreply@example.com")
                },
                "trigger_settings": {
                    "workflow_type": "abandonedBrowse"
                }
            }
            
            # Create automation
            create_url = f"{self.base_url}/automations"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=automation_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        automation_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="mailchimp",
                            status=IntegrationStatus.COMPLETED,
                            external_id=automation_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="mailchimp",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create automation: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating Mailchimp automation: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="mailchimp",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    def _generate_html_content(self, content_data: Dict[str, Any]) -> str:
        """Generate HTML content from content data"""
        title = content_data.get("title", "AI Generated Content")
        content = content_data.get("content", "")
        author = content_data.get("author", "AI Assistant")
        tags = content_data.get("tags", [])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #f4f4f4; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ background-color: #f4f4f4; padding: 10px; text-align: center; font-size: 12px; }}
                .tags {{ margin-top: 20px; }}
                .tag {{ display: inline-block; background-color: #007cba; color: white; padding: 5px 10px; margin: 5px; border-radius: 3px; }}
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
                <div class="tags">
                    {''.join([f'<span class="tag">{tag}</span>' for tag in tags])}
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    async def update_content(self, external_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update existing content in Mailchimp"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id=content_data.get("content_id", "unknown"),
                    platform="mailchimp",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Update campaign content
            content_payload = {
                "html": self._generate_html_content(content_data),
                "plain_text": content_data.get("content", "")
            }
            
            update_url = f"{self.base_url}/campaigns/{external_id}/content"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.put(update_url, json=content_payload, headers=headers) as response:
                    if response.status in [200, 204]:
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="mailchimp",
                            status=IntegrationStatus.COMPLETED,
                            external_id=external_id,
                            response_data={"status": "updated"}
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="mailchimp",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to update content: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error updating Mailchimp content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="mailchimp",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def delete_content(self, external_id: str) -> IntegrationResult:
        """Delete content from Mailchimp"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="mailchimp",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Delete campaign
            delete_url = f"{self.base_url}/campaigns/{external_id}"
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(delete_url, headers=headers) as response:
                    if response.status == 204:
                        return IntegrationResult(
                            request_id="unknown",
                            platform="mailchimp",
                            status=IntegrationStatus.COMPLETED,
                            external_id=external_id,
                            response_data={"status": "deleted"}
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="mailchimp",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to delete content: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error deleting Mailchimp content: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="mailchimp",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def get_content_status(self, external_id: str) -> IntegrationResult:
        """Get content status from Mailchimp"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="mailchimp",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Get campaign info
            get_url = f"{self.base_url}/campaigns/{external_id}"
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(get_url, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="mailchimp",
                            status=IntegrationStatus.COMPLETED,
                            external_id=external_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="mailchimp",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to get content status: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error getting Mailchimp content status: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="mailchimp",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def send_campaign(self, campaign_id: str) -> IntegrationResult:
        """Send a campaign in Mailchimp"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="mailchimp",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Send campaign
            send_url = f"{self.base_url}/campaigns/{campaign_id}/actions/send"
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(send_url, headers=headers) as response:
                    if response.status == 204:
                        return IntegrationResult(
                            request_id="unknown",
                            platform="mailchimp",
                            status=IntegrationStatus.COMPLETED,
                            external_id=campaign_id,
                            response_data={"status": "sent"}
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="mailchimp",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to send campaign: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error sending Mailchimp campaign: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="mailchimp",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )



























