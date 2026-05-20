"""
Salesforce Connector for AI Integration System
Handles integration with Salesforce CRM and Marketing Cloud
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

class SalesforceConnector(PlatformConnector):
    """Salesforce platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://your-instance.salesforce.com")
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.username = config.get("username")
        self.password = config.get("password")
        self.security_token = config.get("security_token")
        self.access_token = None
        self.instance_url = None
        
    async def authenticate(self) -> bool:
        """Authenticate with Salesforce using OAuth2"""
        try:
            if not all([self.client_id, self.client_secret, self.username, self.password]):
                logger.error("Missing required Salesforce credentials")
                return False
            
            # Prepare authentication data
            auth_data = {
                "grant_type": "password",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "username": self.username,
                "password": f"{self.password}{self.security_token or ''}"
            }
            
            auth_url = f"{self.base_url}/services/oauth2/token"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(auth_url, data=auth_data) as response:
                    if response.status == 200:
                        auth_response = await response.json()
                        self.access_token = auth_response.get("access_token")
                        self.instance_url = auth_response.get("instance_url")
                        logger.info("Salesforce authentication successful")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Salesforce authentication failed: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Salesforce authentication error: {str(e)}")
            return False
    
    async def create_content(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create content in Salesforce"""
        try:
            if not self.access_token:
                if not await self.authenticate():
                    return IntegrationResult(
                        request_id=content_data.get("content_id", "unknown"),
                        platform="salesforce",
                        status=IntegrationStatus.FAILED,
                        error_message="Authentication failed"
                    )
            
            # Determine Salesforce object type based on content type
            object_type = self._determine_salesforce_object(content_data)
            
            # Prepare Salesforce record data
            salesforce_data = self._prepare_salesforce_data(content_data, object_type)
            
            # Create record in Salesforce
            create_url = f"{self.instance_url}/services/data/v58.0/sobjects/{object_type}/"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=salesforce_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        external_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="salesforce",
                            status=IntegrationStatus.COMPLETED,
                            external_id=external_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="salesforce",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create record: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating Salesforce content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="salesforce",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def update_content(self, external_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update existing content in Salesforce"""
        try:
            if not self.access_token:
                if not await self.authenticate():
                    return IntegrationResult(
                        request_id=content_data.get("content_id", "unknown"),
                        platform="salesforce",
                        status=IntegrationStatus.FAILED,
                        error_message="Authentication failed"
                    )
            
            # Determine Salesforce object type
            object_type = self._determine_salesforce_object(content_data)
            
            # Prepare update data
            update_data = self._prepare_salesforce_data(content_data, object_type)
            
            # Update record in Salesforce
            update_url = f"{self.instance_url}/services/data/v58.0/sobjects/{object_type}/{external_id}"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.patch(update_url, json=update_data, headers=headers) as response:
                    if response.status == 204:
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="salesforce",
                            status=IntegrationStatus.COMPLETED,
                            external_id=external_id,
                            response_data={"status": "updated"}
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="salesforce",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to update record: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error updating Salesforce content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="salesforce",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def delete_content(self, external_id: str) -> IntegrationResult:
        """Delete content from Salesforce"""
        try:
            if not self.access_token:
                if not await self.authenticate():
                    return IntegrationResult(
                        request_id="unknown",
                        platform="salesforce",
                        status=IntegrationStatus.FAILED,
                        error_message="Authentication failed"
                    )
            
            # Delete record from Salesforce
            delete_url = f"{self.instance_url}/services/data/v58.0/sobjects/ContentDocument/{external_id}"
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(delete_url, headers=headers) as response:
                    if response.status == 204:
                        return IntegrationResult(
                            request_id="unknown",
                            platform="salesforce",
                            status=IntegrationStatus.COMPLETED,
                            external_id=external_id,
                            response_data={"status": "deleted"}
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="salesforce",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to delete record: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error deleting Salesforce content: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="salesforce",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def get_content_status(self, external_id: str) -> IntegrationResult:
        """Get content status from Salesforce"""
        try:
            if not self.access_token:
                if not await self.authenticate():
                    return IntegrationResult(
                        request_id="unknown",
                        platform="salesforce",
                        status=IntegrationStatus.FAILED,
                        error_message="Authentication failed"
                    )
            
            # Get record from Salesforce
            get_url = f"{self.instance_url}/services/data/v58.0/sobjects/ContentDocument/{external_id}"
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(get_url, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="salesforce",
                            status=IntegrationStatus.COMPLETED,
                            external_id=external_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="salesforce",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to get record: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error getting Salesforce content status: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="salesforce",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    def _determine_salesforce_object(self, content_data: Dict[str, Any]) -> str:
        """Determine the appropriate Salesforce object type based on content"""
        content_type = content_data.get("content_type", "").lower()
        
        if "blog" in content_type or "article" in content_type:
            return "ContentDocument"  # For blog posts/articles
        elif "email" in content_type or "campaign" in content_type:
            return "Campaign"  # For email campaigns
        elif "lead" in content_type or "contact" in content_type:
            return "Lead"  # For lead generation content
        elif "opportunity" in content_type:
            return "Opportunity"  # For sales opportunities
        else:
            return "ContentDocument"  # Default to content document
    
    def _prepare_salesforce_data(self, content_data: Dict[str, Any], object_type: str) -> Dict[str, Any]:
        """Prepare data for Salesforce API based on object type"""
        if object_type == "ContentDocument":
            return {
                "Title": content_data.get("title", "AI Generated Content"),
                "Description": content_data.get("content", "")[:255],  # Limit description length
                "ContentType": "text/plain",
                "Body": content_data.get("content", ""),
                "Tags": ",".join(content_data.get("tags", [])),
                "Author": content_data.get("author", "AI Assistant")
            }
        elif object_type == "Campaign":
            return {
                "Name": content_data.get("title", "AI Generated Campaign"),
                "Description": content_data.get("content", ""),
                "Type": "Email",
                "Status": "Planned",
                "StartDate": datetime.now().strftime("%Y-%m-%d")
            }
        elif object_type == "Lead":
            return {
                "FirstName": content_data.get("author", "AI"),
                "LastName": "Generated",
                "Company": content_data.get("category", "AI Content"),
                "Email": content_data.get("email", "ai@example.com"),
                "Description": content_data.get("content", "")
            }
        else:
            return {
                "Name": content_data.get("title", "AI Generated Record"),
                "Description": content_data.get("content", "")
            }
    
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> IntegrationResult:
        """Create a marketing campaign in Salesforce"""
        try:
            if not self.access_token:
                if not await self.authenticate():
                    return IntegrationResult(
                        request_id=campaign_data.get("content_id", "unknown"),
                        platform="salesforce",
                        status=IntegrationStatus.FAILED,
                        error_message="Authentication failed"
                    )
            
            # Prepare campaign data
            salesforce_campaign = {
                "Name": campaign_data.get("title", "AI Generated Campaign"),
                "Description": campaign_data.get("content", ""),
                "Type": campaign_data.get("campaign_type", "Email"),
                "Status": "Planned",
                "StartDate": datetime.now().strftime("%Y-%m-%d"),
                "EndDate": (datetime.now().replace(day=28) + timedelta(days=4)).strftime("%Y-%m-%d")
            }
            
            # Create campaign
            create_url = f"{self.instance_url}/services/data/v58.0/sobjects/Campaign/"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=salesforce_campaign, headers=headers) as response:
                    if response.status in [200, 201]:
                        result_data = await response.json()
                        campaign_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=campaign_data.get("content_id", "unknown"),
                            platform="salesforce",
                            status=IntegrationStatus.COMPLETED,
                            external_id=campaign_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=campaign_data.get("content_id", "unknown"),
                            platform="salesforce",
                            status=IntegrationStatus.FAILED,
                            error_message=f"Failed to create campaign: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating Salesforce campaign: {str(e)}")
            return IntegrationResult(
                request_id=campaign_data.get("content_id", "unknown"),
                platform="salesforce",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
