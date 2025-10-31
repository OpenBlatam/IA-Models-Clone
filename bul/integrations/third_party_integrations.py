"""
Third-Party Integrations for BUL System
Integrates with Google Docs, Office 365, CRM systems, and other external services
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import json
import httpx
import aiohttp
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import base64
import io

logger = logging.getLogger(__name__)


class IntegrationType(str, Enum):
    """Integration types"""
    GOOGLE_DOCS = "google_docs"
    OFFICE_365 = "office_365"
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"
    SLACK = "slack"
    MICROSOFT_TEAMS = "microsoft_teams"
    ZOOM = "zoom"
    DROPBOX = "dropbox"
    ONEDRIVE = "onedrive"
    GITHUB = "github"
    JIRA = "jira"
    TRELLO = "trello"


class IntegrationStatus(str, Enum):
    """Integration status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    EXPIRED = "expired"


class DocumentFormat(str, Enum):
    """Document formats"""
    GOOGLE_DOC = "google_doc"
    WORD_DOCX = "word_docx"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    RTF = "rtf"


@dataclass
class IntegrationCredentials:
    """Integration credentials"""
    client_id: str
    client_secret: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    scope: List[str] = None


class IntegrationConfig(BaseModel):
    """Integration configuration"""
    id: str = Field(..., description="Integration ID")
    name: str = Field(..., description="Integration name")
    type: IntegrationType = Field(..., description="Integration type")
    status: IntegrationStatus = Field(default=IntegrationStatus.INACTIVE, description="Integration status")
    credentials: Dict[str, Any] = Field(..., description="Integration credentials")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Integration settings")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for notifications")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_sync: Optional[datetime] = Field(None, description="Last synchronization time")


class DocumentSync(BaseModel):
    """Document synchronization data"""
    document_id: str = Field(..., description="BUL document ID")
    external_id: str = Field(..., description="External document ID")
    integration_id: str = Field(..., description="Integration ID")
    format: DocumentFormat = Field(..., description="Document format")
    status: str = Field(default="synced", description="Sync status")
    last_sync: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sync metadata")


class GoogleDocsIntegration:
    """Google Docs integration"""
    
    def __init__(self, credentials: IntegrationCredentials):
        self.credentials = credentials
        self.base_url = "https://docs.googleapis.com/v1"
        self.drive_url = "https://www.googleapis.com/drive/v3"
        self.http_client = httpx.AsyncClient()
    
    async def create_document(self, title: str, content: str, folder_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new Google Doc"""
        try:
            # Create document
            doc_data = {
                "title": title,
                "mimeType": "application/vnd.google-apps.document"
            }
            
            if folder_id:
                doc_data["parents"] = [folder_id]
            
            # Upload to Google Drive
            response = await self.http_client.post(
                f"{self.drive_url}/files",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/json"
                },
                json=doc_data
            )
            response.raise_for_status()
            
            doc_info = response.json()
            doc_id = doc_info["id"]
            
            # Add content to document
            await self._update_document_content(doc_id, content)
            
            return {
                "document_id": doc_id,
                "title": title,
                "url": f"https://docs.google.com/document/d/{doc_id}/edit",
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating Google Doc: {e}")
            raise
    
    async def _update_document_content(self, doc_id: str, content: str):
        """Update document content"""
        try:
            # Convert content to Google Docs format
            formatted_content = self._format_content_for_google_docs(content)
            
            # Update document
            response = await self.http_client.patch(
                f"{self.base_url}/documents/{doc_id}",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/json"
                },
                json=formatted_content
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error updating Google Doc content: {e}")
            raise
    
    def _format_content_for_google_docs(self, content: str) -> Dict[str, Any]:
        """Format content for Google Docs API"""
        # Simple formatting - in production, this would be more sophisticated
        return {
            "requests": [
                {
                    "insertText": {
                        "location": {
                            "index": 1
                        },
                        "text": content
                    }
                }
            ]
        }
    
    async def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get document from Google Docs"""
        try:
            response = await self.http_client.get(
                f"{self.base_url}/documents/{doc_id}",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}"
                }
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting Google Doc: {e}")
            raise
    
    async def share_document(self, doc_id: str, email: str, role: str = "reader") -> Dict[str, Any]:
        """Share document with specific user"""
        try:
            share_data = {
                "role": role,
                "type": "user",
                "emailAddress": email
            }
            
            response = await self.http_client.post(
                f"{self.drive_url}/files/{doc_id}/permissions",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/json"
                },
                json=share_data
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error sharing Google Doc: {e}")
            raise


class Office365Integration:
    """Office 365 integration"""
    
    def __init__(self, credentials: IntegrationCredentials):
        self.credentials = credentials
        self.base_url = "https://graph.microsoft.com/v1.0"
        self.http_client = httpx.AsyncClient()
    
    async def create_document(self, title: str, content: str, folder_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new Word document in OneDrive"""
        try:
            # Create document
            doc_data = {
                "name": f"{title}.docx",
                "file": {
                    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                }
            }
            
            # Upload to OneDrive
            endpoint = f"{self.base_url}/me/drive/items"
            if folder_id:
                endpoint = f"{self.base_url}/me/drive/items/{folder_id}/children"
            
            response = await self.http_client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/json"
                },
                json=doc_data
            )
            response.raise_for_status()
            
            doc_info = response.json()
            doc_id = doc_info["id"]
            
            # Add content to document
            await self._update_document_content(doc_id, content)
            
            return {
                "document_id": doc_id,
                "title": title,
                "url": doc_info.get("webUrl"),
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating Office 365 document: {e}")
            raise
    
    async def _update_document_content(self, doc_id: str, content: str):
        """Update document content"""
        try:
            # Convert content to Word format
            formatted_content = self._format_content_for_word(content)
            
            # Update document
            response = await self.http_client.patch(
                f"{self.base_url}/me/drive/items/{doc_id}/content",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                },
                content=formatted_content
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error updating Office 365 document content: {e}")
            raise
    
    def _format_content_for_word(self, content: str) -> bytes:
        """Format content for Word document"""
        # Simple formatting - in production, this would use python-docx
        # For now, return the content as bytes
        return content.encode('utf-8')
    
    async def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get document from OneDrive"""
        try:
            response = await self.http_client.get(
                f"{self.base_url}/me/drive/items/{doc_id}",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}"
                }
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting Office 365 document: {e}")
            raise


class SalesforceIntegration:
    """Salesforce CRM integration"""
    
    def __init__(self, credentials: IntegrationCredentials):
        self.credentials = credentials
        self.base_url = f"https://{credentials.client_id}.my.salesforce.com/services/data/v58.0"
        self.http_client = httpx.AsyncClient()
    
    async def create_lead(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new lead in Salesforce"""
        try:
            response = await self.http_client.post(
                f"{self.base_url}/sobjects/Lead",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/json"
                },
                json=lead_data
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error creating Salesforce lead: {e}")
            raise
    
    async def create_opportunity(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new opportunity in Salesforce"""
        try:
            response = await self.http_client.post(
                f"{self.base_url}/sobjects/Opportunity",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/json"
                },
                json=opportunity_data
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error creating Salesforce opportunity: {e}")
            raise
    
    async def get_accounts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get accounts from Salesforce"""
        try:
            response = await self.http_client.get(
                f"{self.base_url}/query/?q=SELECT Id, Name, Industry, AnnualRevenue FROM Account LIMIT {limit}",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}"
                }
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("records", [])
            
        except Exception as e:
            logger.error(f"Error getting Salesforce accounts: {e}")
            raise


class HubSpotIntegration:
    """HubSpot CRM integration"""
    
    def __init__(self, credentials: IntegrationCredentials):
        self.credentials = credentials
        self.base_url = "https://api.hubapi.com"
        self.http_client = httpx.AsyncClient()
    
    async def create_contact(self, contact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new contact in HubSpot"""
        try:
            response = await self.http_client.post(
                f"{self.base_url}/crm/v3/objects/contacts",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/json"
                },
                json={"properties": contact_data}
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error creating HubSpot contact: {e}")
            raise
    
    async def create_deal(self, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new deal in HubSpot"""
        try:
            response = await self.http_client.post(
                f"{self.base_url}/crm/v3/objects/deals",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/json"
                },
                json={"properties": deal_data}
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error creating HubSpot deal: {e}")
            raise
    
    async def get_contacts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get contacts from HubSpot"""
        try:
            response = await self.http_client.get(
                f"{self.base_url}/crm/v3/objects/contacts?limit={limit}",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}"
                }
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])
            
        except Exception as e:
            logger.error(f"Error getting HubSpot contacts: {e}")
            raise


class SlackIntegration:
    """Slack integration"""
    
    def __init__(self, credentials: IntegrationCredentials):
        self.credentials = credentials
        self.base_url = "https://slack.com/api"
        self.http_client = httpx.AsyncClient()
    
    async def send_message(self, channel: str, message: str, blocks: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Send message to Slack channel"""
        try:
            payload = {
                "channel": channel,
                "text": message
            }
            
            if blocks:
                payload["blocks"] = blocks
            
            response = await self.http_client.post(
                f"{self.base_url}/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            raise
    
    async def create_document_notification(self, document_title: str, document_url: str, channel: str) -> Dict[str, Any]:
        """Send document creation notification to Slack"""
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📄 *New Document Created*\n*Title:* {document_title}\n*URL:* {document_url}"
                }
            }
        ]
        
        return await self.send_message(
            channel=channel,
            message=f"New document created: {document_title}",
            blocks=blocks
        )


class ThirdPartyIntegrationManager:
    """Manager for all third-party integrations"""
    
    def __init__(self):
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.document_syncs: Dict[str, DocumentSync] = {}
        self._initialize_default_integrations()
    
    def _initialize_default_integrations(self):
        """Initialize default integration configurations"""
        default_integrations = [
            IntegrationConfig(
                id="google_docs_default",
                name="Google Docs",
                type=IntegrationType.GOOGLE_DOCS,
                credentials={
                    "client_id": "",
                    "client_secret": "",
                    "access_token": "",
                    "refresh_token": "",
                    "scope": ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"]
                },
                settings={
                    "default_folder": "",
                    "auto_sync": True,
                    "sync_interval": 300
                }
            ),
            IntegrationConfig(
                id="office_365_default",
                name="Office 365",
                type=IntegrationType.OFFICE_365,
                credentials={
                    "client_id": "",
                    "client_secret": "",
                    "access_token": "",
                    "refresh_token": "",
                    "scope": ["https://graph.microsoft.com/Files.ReadWrite", "https://graph.microsoft.com/Sites.ReadWrite.All"]
                },
                settings={
                    "default_folder": "",
                    "auto_sync": True,
                    "sync_interval": 300
                }
            ),
            IntegrationConfig(
                id="salesforce_default",
                name="Salesforce CRM",
                type=IntegrationType.SALESFORCE,
                credentials={
                    "client_id": "",
                    "client_secret": "",
                    "access_token": "",
                    "refresh_token": "",
                    "scope": ["api", "refresh_token"]
                },
                settings={
                    "auto_sync": False,
                    "sync_interval": 600
                }
            ),
            IntegrationConfig(
                id="hubspot_default",
                name="HubSpot CRM",
                type=IntegrationType.HUBSPOT,
                credentials={
                    "client_id": "",
                    "client_secret": "",
                    "access_token": "",
                    "refresh_token": "",
                    "scope": ["crm.objects.contacts.read", "crm.objects.contacts.write", "crm.objects.deals.read", "crm.objects.deals.write"]
                },
                settings={
                    "auto_sync": False,
                    "sync_interval": 600
                }
            ),
            IntegrationConfig(
                id="slack_default",
                name="Slack",
                type=IntegrationType.SLACK,
                credentials={
                    "client_id": "",
                    "client_secret": "",
                    "access_token": "",
                    "refresh_token": "",
                    "scope": ["chat:write", "channels:read"]
                },
                settings={
                    "default_channel": "",
                    "notifications_enabled": True
                }
            )
        ]
        
        for integration in default_integrations:
            self.integrations[integration.id] = integration
        
        logger.info(f"Initialized {len(default_integrations)} default integrations")
    
    async def create_integration(self, integration_data: Dict[str, Any]) -> IntegrationConfig:
        """Create a new integration"""
        integration = IntegrationConfig(**integration_data)
        self.integrations[integration.id] = integration
        
        logger.info(f"Created integration {integration.id}")
        return integration
    
    async def update_integration(self, integration_id: str, updates: Dict[str, Any]) -> Optional[IntegrationConfig]:
        """Update an existing integration"""
        integration = self.integrations.get(integration_id)
        if not integration:
            return None
        
        for key, value in updates.items():
            if hasattr(integration, key):
                setattr(integration, key, value)
        
        integration.updated_at = datetime.utcnow()
        logger.info(f"Updated integration {integration_id}")
        
        return integration
    
    async def get_integration(self, integration_id: str) -> Optional[IntegrationConfig]:
        """Get integration by ID"""
        return self.integrations.get(integration_id)
    
    async def list_integrations(self, integration_type: Optional[IntegrationType] = None) -> List[IntegrationConfig]:
        """List integrations with optional filtering"""
        integrations = list(self.integrations.values())
        
        if integration_type:
            integrations = [i for i in integrations if i.type == integration_type]
        
        return integrations
    
    async def sync_document_to_external(
        self,
        integration_id: str,
        document_id: str,
        document_title: str,
        document_content: str,
        format: DocumentFormat = DocumentFormat.GOOGLE_DOC
    ) -> DocumentSync:
        """Sync document to external service"""
        integration = await self.get_integration(integration_id)
        if not integration:
            raise ValueError(f"Integration {integration_id} not found")
        
        try:
            if integration.type == IntegrationType.GOOGLE_DOCS:
                google_integration = GoogleDocsIntegration(
                    IntegrationCredentials(**integration.credentials)
                )
                result = await google_integration.create_document(
                    title=document_title,
                    content=document_content
                )
                external_id = result["document_id"]
                
            elif integration.type == IntegrationType.OFFICE_365:
                office_integration = Office365Integration(
                    IntegrationCredentials(**integration.credentials)
                )
                result = await office_integration.create_document(
                    title=document_title,
                    content=document_content
                )
                external_id = result["document_id"]
                
            else:
                raise ValueError(f"Unsupported integration type: {integration.type}")
            
            # Create sync record
            sync = DocumentSync(
                document_id=document_id,
                external_id=external_id,
                integration_id=integration_id,
                format=format,
                metadata=result
            )
            
            self.document_syncs[f"{document_id}_{integration_id}"] = sync
            
            # Update integration last sync
            integration.last_sync = datetime.utcnow()
            
            logger.info(f"Synced document {document_id} to {integration_id}")
            return sync
            
        except Exception as e:
            logger.error(f"Error syncing document to {integration_id}: {e}")
            raise
    
    async def send_notification(
        self,
        integration_id: str,
        notification_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send notification through integration"""
        integration = await self.get_integration(integration_id)
        if not integration:
            raise ValueError(f"Integration {integration_id} not found")
        
        try:
            if integration.type == IntegrationType.SLACK:
                slack_integration = SlackIntegration(
                    IntegrationCredentials(**integration.credentials)
                )
                
                if notification_data.get("type") == "document_created":
                    return await slack_integration.create_document_notification(
                        document_title=notification_data["title"],
                        document_url=notification_data["url"],
                        channel=integration.settings.get("default_channel", "#general")
                    )
                else:
                    return await slack_integration.send_message(
                        channel=integration.settings.get("default_channel", "#general"),
                        message=notification_data.get("message", "Notification from BUL")
                    )
            
            else:
                raise ValueError(f"Unsupported notification integration type: {integration.type}")
                
        except Exception as e:
            logger.error(f"Error sending notification through {integration_id}: {e}")
            raise
    
    async def create_crm_record(
        self,
        integration_id: str,
        record_type: str,
        record_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create record in CRM system"""
        integration = await self.get_integration(integration_id)
        if not integration:
            raise ValueError(f"Integration {integration_id} not found")
        
        try:
            if integration.type == IntegrationType.SALESFORCE:
                salesforce_integration = SalesforceIntegration(
                    IntegrationCredentials(**integration.credentials)
                )
                
                if record_type == "lead":
                    return await salesforce_integration.create_lead(record_data)
                elif record_type == "opportunity":
                    return await salesforce_integration.create_opportunity(record_data)
                else:
                    raise ValueError(f"Unsupported Salesforce record type: {record_type}")
            
            elif integration.type == IntegrationType.HUBSPOT:
                hubspot_integration = HubSpotIntegration(
                    IntegrationCredentials(**integration.credentials)
                )
                
                if record_type == "contact":
                    return await hubspot_integration.create_contact(record_data)
                elif record_type == "deal":
                    return await hubspot_integration.create_deal(record_data)
                else:
                    raise ValueError(f"Unsupported HubSpot record type: {record_type}")
            
            else:
                raise ValueError(f"Unsupported CRM integration type: {integration.type}")
                
        except Exception as e:
            logger.error(f"Error creating CRM record in {integration_id}: {e}")
            raise
    
    async def get_integration_analytics(self, integration_id: str) -> Dict[str, Any]:
        """Get analytics for integration"""
        integration = await self.get_integration(integration_id)
        if not integration:
            raise ValueError(f"Integration {integration_id} not found")
        
        # Get sync statistics
        syncs = [sync for sync in self.document_syncs.values() if sync.integration_id == integration_id]
        
        return {
            "integration_id": integration_id,
            "integration_name": integration.name,
            "integration_type": integration.type.value,
            "status": integration.status.value,
            "total_syncs": len(syncs),
            "successful_syncs": len([s for s in syncs if s.status == "synced"]),
            "failed_syncs": len([s for s in syncs if s.status == "failed"]),
            "last_sync": integration.last_sync.isoformat() if integration.last_sync else None,
            "created_at": integration.created_at.isoformat(),
            "updated_at": integration.updated_at.isoformat()
        }
    
    async def get_system_integration_analytics(self) -> Dict[str, Any]:
        """Get system-wide integration analytics"""
        total_integrations = len(self.integrations)
        active_integrations = len([i for i in self.integrations.values() if i.status == IntegrationStatus.ACTIVE])
        total_syncs = len(self.document_syncs)
        
        # Group by type
        integrations_by_type = {}
        for integration in self.integrations.values():
            integration_type = integration.type.value
            if integration_type not in integrations_by_type:
                integrations_by_type[integration_type] = 0
            integrations_by_type[integration_type] += 1
        
        return {
            "total_integrations": total_integrations,
            "active_integrations": active_integrations,
            "total_syncs": total_syncs,
            "integrations_by_type": integrations_by_type,
            "recent_syncs": [
                {
                    "document_id": sync.document_id,
                    "integration_id": sync.integration_id,
                    "status": sync.status,
                    "last_sync": sync.last_sync.isoformat()
                }
                for sync in list(self.document_syncs.values())[-10:]  # Last 10 syncs
            ]
        }


# Global integration manager instance
integration_manager = ThirdPartyIntegrationManager()














