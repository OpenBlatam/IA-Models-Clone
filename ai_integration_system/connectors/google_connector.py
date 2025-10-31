"""
Google Workspace Connector for AI Integration System
Handles integration with Google Docs, Sheets, Drive, and Gmail
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import base64
import io

from ..integration_engine import PlatformConnector, IntegrationResult, IntegrationStatus

logger = logging.getLogger(__name__)

class GoogleConnector(PlatformConnector):
    """Google Workspace platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.credentials_file = config.get("credentials_file")
        self.scopes = config.get("scopes", [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/gmail.send"
        ])
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        
    async def authenticate(self) -> bool:
        """Authenticate with Google APIs"""
        try:
            if not self.credentials_file:
                logger.error("Missing Google credentials file")
                return False
            
            # Load credentials and get access token
            credentials = await self._load_credentials()
            if not credentials:
                return False
            
            # Test authentication by making a simple API call
            test_url = "https://www.googleapis.com/drive/v3/about"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(test_url, headers=headers) as response:
                    if response.status == 200:
                        logger.info("Google authentication successful")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Google authentication failed: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Google authentication error: {str(e)}")
            return False
    
    async def _load_credentials(self) -> bool:
        """Load Google credentials and get access token"""
        try:
            # This is a simplified version - in production, you'd use proper OAuth2 flow
            # For now, we'll assume the credentials file contains the necessary tokens
            
            import json
            with open(self.credentials_file, 'r') as f:
                creds_data = json.load(f)
            
            self.access_token = creds_data.get("access_token")
            self.refresh_token = creds_data.get("refresh_token")
            
            if not self.access_token:
                logger.error("No access token found in credentials file")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading credentials: {str(e)}")
            return False
    
    async def create_content(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create content in Google Workspace"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id=content_data.get("content_id", "unknown"),
                    platform="google",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Determine content type and create accordingly
            content_type = content_data.get("content_type", "").lower()
            
            if "document" in content_type or "doc" in content_type:
                return await self._create_google_doc(content_data)
            elif "sheet" in content_type or "spreadsheet" in content_type:
                return await self._create_google_sheet(content_data)
            elif "email" in content_type or "gmail" in content_type:
                return await self._send_gmail(content_data)
            elif "file" in content_type or "drive" in content_type:
                return await self._upload_to_drive(content_data)
            else:
                # Default to creating a Google Doc
                return await self._create_google_doc(content_data)
                
        except Exception as e:
            logger.error(f"Error creating Google content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="google",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_google_doc(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create a Google Doc"""
        try:
            # Create new document
            create_url = "https://docs.googleapis.com/v1/documents"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            doc_data = {
                "title": content_data.get("title", "AI Generated Document")
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=doc_data, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        document_id = result_data.get("documentId")
                        
                        if document_id:
                            # Add content to the document
                            await self._add_content_to_doc(document_id, content_data)
                            
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="google",
                                status=IntegrationStatus.COMPLETED,
                                external_id=document_id,
                                response_data=result_data
                            )
                        else:
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="google",
                                status=IntegrationStatus.FAILED,
                                error_message="No document ID returned"
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="google",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating Google Doc: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="google",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _add_content_to_doc(self, document_id: str, content_data: Dict[str, Any]):
        """Add content to a Google Doc"""
        try:
            # Prepare content for insertion
            content_text = content_data.get("content", "")
            title = content_data.get("title", "")
            author = content_data.get("author", "AI Assistant")
            
            # Format content with title and author
            formatted_content = f"{title}\n\n{content_text}\n\n---\nGenerated by {author} via AI Integration System"
            
            # Insert content
            update_url = f"https://docs.googleapis.com/v1/documents/{document_id}:batchUpdate"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Create requests for content insertion
            requests = [
                {
                    "insertText": {
                        "location": {
                            "index": 1
                        },
                        "text": formatted_content
                    }
                }
            ]
            
            update_data = {"requests": requests}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(update_url, json=update_data, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Content added to Google Doc {document_id}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Error adding content to Google Doc: {error_text}")
                        
        except Exception as e:
            logger.error(f"Error adding content to Google Doc: {str(e)}")
    
    async def _create_google_sheet(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create a Google Sheet"""
        try:
            # Create new spreadsheet
            create_url = "https://sheets.googleapis.com/v4/spreadsheets"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            sheet_data = {
                "properties": {
                    "title": content_data.get("title", "AI Generated Spreadsheet")
                },
                "sheets": [
                    {
                        "properties": {
                            "title": "Sheet1"
                        }
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=sheet_data, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        spreadsheet_id = result_data.get("spreadsheetId")
                        
                        if spreadsheet_id:
                            # Add content to the sheet
                            await self._add_content_to_sheet(spreadsheet_id, content_data)
                            
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="google",
                                status=IntegrationStatus.COMPLETED,
                                external_id=spreadsheet_id,
                                response_data=result_data
                            )
                        else:
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="google",
                                status=IntegrationStatus.FAILED,
                                error_message="No spreadsheet ID returned"
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="google",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating Google Sheet: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="google",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _add_content_to_sheet(self, spreadsheet_id: str, content_data: Dict[str, Any]):
        """Add content to a Google Sheet"""
        try:
            # Prepare data for the sheet
            title = content_data.get("title", "")
            content = content_data.get("content", "")
            author = content_data.get("author", "AI Assistant")
            
            # Create data rows
            data = [
                [title],
                [""],
                [content],
                [""],
                ["Generated by", author],
                ["Date", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")]
            ]
            
            # Update sheet
            update_url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/Sheet1!A1:B6"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            update_data = {
                "values": data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.put(update_url, json=update_data, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Content added to Google Sheet {spreadsheet_id}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Error adding content to Google Sheet: {error_text}")
                        
        except Exception as e:
            logger.error(f"Error adding content to Google Sheet: {str(e)}")
    
    async def _send_gmail(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Send email via Gmail"""
        try:
            # Prepare email data
            to_email = content_data.get("to_email", "recipient@example.com")
            subject = content_data.get("title", "AI Generated Email")
            content = content_data.get("content", "")
            author = content_data.get("author", "AI Assistant")
            
            # Create email message
            email_message = self._create_email_message(to_email, subject, content, author)
            
            # Send email
            send_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            send_data = {
                "raw": base64.urlsafe_b64encode(email_message.encode()).decode()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(send_url, json=send_data, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        message_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="google",
                            status=IntegrationStatus.COMPLETED,
                            external_id=message_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="google",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error sending Gmail: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="google",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    def _create_email_message(self, to_email: str, subject: str, content: str, author: str) -> str:
        """Create RFC 2822 email message"""
        from_email = "ai-integration@example.com"  # Configure this
        
        message = f"""From: {from_email}
To: {to_email}
Subject: {subject}
Content-Type: text/html; charset=UTF-8

<html>
<body>
<h2>{subject}</h2>
<p>{content.replace(chr(10), '<br>')}</p>
<hr>
<p><em>Generated by {author} via AI Integration System</em></p>
<p><small>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</small></p>
</body>
</html>
"""
        return message
    
    async def _upload_to_drive(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Upload file to Google Drive"""
        try:
            # Prepare file data
            filename = content_data.get("filename", "ai_generated_file.txt")
            content = content_data.get("content", "")
            title = content_data.get("title", "AI Generated File")
            
            # Create file metadata
            file_metadata = {
                "name": filename,
                "description": f"{title} - Generated by AI Integration System"
            }
            
            # Upload file
            upload_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }
            
            # Create multipart form data
            boundary = "----formdata-boundary"
            body = f"""--{boundary}
Content-Disposition: form-data; name="metadata"

{json.dumps(file_metadata)}
--{boundary}
Content-Disposition: form-data; name="file"; filename="{filename}"
Content-Type: text/plain

{content}
--{boundary}--
"""
            
            headers["Content-Type"] = f"multipart/related; boundary={boundary}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(upload_url, data=body, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        file_id = result_data.get("id")
                        
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="google",
                            status=IntegrationStatus.COMPLETED,
                            external_id=file_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="google",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error uploading to Google Drive: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="google",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def update_content(self, external_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update existing content in Google Workspace"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id=content_data.get("content_id", "unknown"),
                    platform="google",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Determine content type and update accordingly
            content_type = content_data.get("content_type", "").lower()
            
            if "document" in content_type or "doc" in content_type:
                return await self._update_google_doc(external_id, content_data)
            elif "sheet" in content_type or "spreadsheet" in content_type:
                return await self._update_google_sheet(external_id, content_data)
            else:
                return await self._update_google_doc(external_id, content_data)
                
        except Exception as e:
            logger.error(f"Error updating Google content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="google",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _update_google_doc(self, document_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update a Google Doc"""
        try:
            # Clear existing content and add new content
            content_text = content_data.get("content", "")
            title = content_data.get("title", "Updated AI Generated Document")
            
            formatted_content = f"{title}\n\n{content_text}\n\n---\nUpdated by AI Integration System"
            
            update_url = f"https://docs.googleapis.com/v1/documents/{document_id}:batchUpdate"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Create requests for content update
            requests = [
                {
                    "deleteContentRange": {
                        "range": {
                            "startIndex": 1,
                            "endIndex": -1
                        }
                    }
                },
                {
                    "insertText": {
                        "location": {
                            "index": 1
                        },
                        "text": formatted_content
                    }
                }
            ]
            
            update_data = {"requests": requests}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(update_url, json=update_data, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="google",
                            status=IntegrationStatus.COMPLETED,
                            external_id=document_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="google",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error updating Google Doc: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="google",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _update_google_sheet(self, spreadsheet_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update a Google Sheet"""
        try:
            # Clear existing content and add new content
            title = content_data.get("title", "Updated AI Generated Spreadsheet")
            content = content_data.get("content", "")
            
            data = [
                [title],
                [""],
                [content],
                [""],
                ["Updated by", "AI Integration System"],
                ["Date", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")]
            ]
            
            update_url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/Sheet1!A1:B6"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            update_data = {
                "values": data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.put(update_url, json=update_data, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="google",
                            status=IntegrationStatus.COMPLETED,
                            external_id=spreadsheet_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="google",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error updating Google Sheet: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="google",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def delete_content(self, external_id: str) -> IntegrationResult:
        """Delete content from Google Workspace"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="google",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Delete file from Google Drive
            delete_url = f"https://www.googleapis.com/drive/v3/files/{external_id}"
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(delete_url, headers=headers) as response:
                    if response.status == 200:
                        return IntegrationResult(
                            request_id="unknown",
                            platform="google",
                            status=IntegrationStatus.COMPLETED,
                            external_id=external_id,
                            response_data={"status": "deleted"}
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="google",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error deleting Google content: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="google",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def get_content_status(self, external_id: str) -> IntegrationResult:
        """Get content status from Google Workspace"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="google",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Get file info from Google Drive
            get_url = f"https://www.googleapis.com/drive/v3/files/{external_id}"
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(get_url, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="google",
                            status=IntegrationStatus.COMPLETED,
                            external_id=external_id,
                            response_data=result_data
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="google",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error getting Google content status: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="google",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )



























