"""
Slack Connector for AI Integration System
Handles integration with Slack workspace for messaging and file sharing
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

class SlackConnector(PlatformConnector):
    """Slack platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bot_token = config.get("bot_token")
        self.app_token = config.get("app_token")
        self.signing_secret = config.get("signing_secret")
        self.base_url = "https://slack.com/api"
        self.workspace_id = config.get("workspace_id")
        
    async def authenticate(self) -> bool:
        """Authenticate with Slack API"""
        try:
            if not self.bot_token:
                logger.error("Missing required Slack bot token")
                return False
            
            # Test authentication by getting bot info
            auth_url = f"{self.base_url}/auth.test"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(auth_url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            logger.info("Slack authentication successful")
                            return True
                        else:
                            logger.error(f"Slack authentication failed: {result.get('error')}")
                            return False
                    else:
                        error_text = await response.text()
                        logger.error(f"Slack authentication failed: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Slack authentication error: {str(e)}")
            return False
    
    async def create_content(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create content in Slack (messages, files, etc.)"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id=content_data.get("content_id", "unknown"),
                    platform="slack",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Determine content type and create accordingly
            content_type = content_data.get("content_type", "").lower()
            
            if "message" in content_type or "post" in content_type:
                return await self._send_message(content_data)
            elif "file" in content_type:
                return await self._upload_file(content_data)
            elif "channel" in content_type:
                return await self._create_channel(content_data)
            else:
                # Default to sending a message
                return await self._send_message(content_data)
                
        except Exception as e:
            logger.error(f"Error creating Slack content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="slack",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _send_message(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Send a message to Slack"""
        try:
            # Prepare message data
            message_data = {
                "channel": content_data.get("channel", "#general"),
                "text": content_data.get("title", "AI Generated Message"),
                "blocks": self._create_message_blocks(content_data)
            }
            
            # Add optional parameters
            if content_data.get("thread_ts"):
                message_data["thread_ts"] = content_data["thread_ts"]
            
            if content_data.get("reply_broadcast"):
                message_data["reply_broadcast"] = content_data["reply_broadcast"]
            
            # Send message
            send_url = f"{self.base_url}/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(send_url, json=message_data, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        if result_data.get("ok"):
                            message_ts = result_data.get("ts")
                            
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="slack",
                                status=IntegrationStatus.COMPLETED,
                                external_id=message_ts,
                                response_data=result_data
                            )
                        else:
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="slack",
                                status=IntegrationStatus.FAILED,
                                error_message=f"Slack API error: {result_data.get('error')}"
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="slack",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error sending Slack message: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="slack",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _upload_file(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Upload a file to Slack"""
        try:
            # Prepare file data
            file_data = {
                "channels": content_data.get("channel", "#general"),
                "title": content_data.get("title", "AI Generated File"),
                "initial_comment": content_data.get("content", ""),
                "filename": content_data.get("filename", "ai_generated_file.txt")
            }
            
            # Handle file content
            file_content = content_data.get("file_content", "")
            if isinstance(file_content, str):
                file_content = file_content.encode('utf-8')
            
            # Upload file
            upload_url = f"{self.base_url}/files.upload"
            headers = {
                "Authorization": f"Bearer {self.bot_token}"
            }
            
            # Create form data
            data = aiohttp.FormData()
            for key, value in file_data.items():
                data.add_field(key, value)
            data.add_field('file', file_content, filename=file_data["filename"])
            
            async with aiohttp.ClientSession() as session:
                async with session.post(upload_url, data=data, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        if result_data.get("ok"):
                            file_id = result_data.get("file", {}).get("id")
                            
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="slack",
                                status=IntegrationStatus.COMPLETED,
                                external_id=file_id,
                                response_data=result_data
                            )
                        else:
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="slack",
                                status=IntegrationStatus.FAILED,
                                error_message=f"Slack API error: {result_data.get('error')}"
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="slack",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error uploading Slack file: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="slack",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_channel(self, content_data: Dict[str, Any]) -> IntegrationResult:
        """Create a new Slack channel"""
        try:
            # Prepare channel data
            channel_data = {
                "name": content_data.get("channel_name", "ai-generated-channel"),
                "is_private": content_data.get("is_private", False)
            }
            
            # Create channel
            create_url = f"{self.base_url}/conversations.create"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_url, json=channel_data, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        if result_data.get("ok"):
                            channel_id = result_data.get("channel", {}).get("id")
                            
                            # Post initial message if provided
                            if content_data.get("content"):
                                await self._send_initial_channel_message(channel_id, content_data)
                            
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="slack",
                                status=IntegrationStatus.COMPLETED,
                                external_id=channel_id,
                                response_data=result_data
                            )
                        else:
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="slack",
                                status=IntegrationStatus.FAILED,
                                error_message=f"Slack API error: {result_data.get('error')}"
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="slack",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error creating Slack channel: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="slack",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def _send_initial_channel_message(self, channel_id: str, content_data: Dict[str, Any]):
        """Send initial message to newly created channel"""
        try:
            message_data = {
                "channel": channel_id,
                "text": content_data.get("content", "Welcome to the new channel!"),
                "blocks": self._create_message_blocks(content_data)
            }
            
            send_url = f"{self.base_url}/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(send_url, json=message_data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if not result.get("ok"):
                            logger.warning(f"Failed to send initial channel message: {result.get('error')}")
        except Exception as e:
            logger.error(f"Error sending initial channel message: {str(e)}")
    
    def _create_message_blocks(self, content_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create Slack message blocks for rich formatting"""
        blocks = []
        
        # Header block
        if content_data.get("title"):
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": content_data["title"]
                }
            })
        
        # Content block
        if content_data.get("content"):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": content_data["content"]
                }
            })
        
        # Author block
        if content_data.get("author"):
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Author:* {content_data['author']}"
                    }
                ]
            })
        
        # Tags block
        if content_data.get("tags"):
            tags_text = " ".join([f"`{tag}`" for tag in content_data["tags"]])
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Tags:* {tags_text}"
                    }
                ]
            })
        
        # Footer block
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"*Generated by AI Integration System* • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                }
            ]
        })
        
        return blocks
    
    async def update_content(self, external_id: str, content_data: Dict[str, Any]) -> IntegrationResult:
        """Update existing content in Slack"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id=content_data.get("content_id", "unknown"),
                    platform="slack",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Update message
            update_data = {
                "channel": content_data.get("channel", "#general"),
                "ts": external_id,
                "text": content_data.get("title", "Updated AI Generated Message"),
                "blocks": self._create_message_blocks(content_data)
            }
            
            update_url = f"{self.base_url}/chat.update"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(update_url, json=update_data, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        if result_data.get("ok"):
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="slack",
                                status=IntegrationStatus.COMPLETED,
                                external_id=external_id,
                                response_data=result_data
                            )
                        else:
                            return IntegrationResult(
                                request_id=content_data.get("content_id", "unknown"),
                                platform="slack",
                                status=IntegrationStatus.FAILED,
                                error_message=f"Slack API error: {result_data.get('error')}"
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id=content_data.get("content_id", "unknown"),
                            platform="slack",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error updating Slack content: {str(e)}")
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="slack",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def delete_content(self, external_id: str) -> IntegrationResult:
        """Delete content from Slack"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="slack",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Delete message
            delete_data = {
                "channel": "#general",  # You might want to store channel info
                "ts": external_id
            }
            
            delete_url = f"{self.base_url}/chat.delete"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(delete_url, json=delete_data, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        if result_data.get("ok"):
                            return IntegrationResult(
                                request_id="unknown",
                                platform="slack",
                                status=IntegrationStatus.COMPLETED,
                                external_id=external_id,
                                response_data=result_data
                            )
                        else:
                            return IntegrationResult(
                                request_id="unknown",
                                platform="slack",
                                status=IntegrationStatus.FAILED,
                                error_message=f"Slack API error: {result_data.get('error')}"
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="slack",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error deleting Slack content: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="slack",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def get_content_status(self, external_id: str) -> IntegrationResult:
        """Get content status from Slack"""
        try:
            if not await self.authenticate():
                return IntegrationResult(
                    request_id="unknown",
                    platform="slack",
                    status=IntegrationStatus.FAILED,
                    error_message="Authentication failed"
                )
            
            # Get message info
            get_url = f"{self.base_url}/conversations.history"
            params = {
                "channel": "#general",  # You might want to store channel info
                "latest": external_id,
                "limit": 1
            }
            headers = {
                "Authorization": f"Bearer {self.bot_token}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(get_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        if result_data.get("ok"):
                            messages = result_data.get("messages", [])
                            if messages:
                                return IntegrationResult(
                                    request_id="unknown",
                                    platform="slack",
                                    status=IntegrationStatus.COMPLETED,
                                    external_id=external_id,
                                    response_data=messages[0]
                                )
                            else:
                                return IntegrationResult(
                                    request_id="unknown",
                                    platform="slack",
                                    status=IntegrationStatus.FAILED,
                                    error_message="Message not found"
                                )
                        else:
                            return IntegrationResult(
                                request_id="unknown",
                                platform="slack",
                                status=IntegrationStatus.FAILED,
                                error_message=f"Slack API error: {result_data.get('error')}"
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            request_id="unknown",
                            platform="slack",
                            status=IntegrationStatus.FAILED,
                            error_message=f"HTTP error: {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"Error getting Slack content status: {str(e)}")
            return IntegrationResult(
                request_id="unknown",
                platform="slack",
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
    
    async def get_workspace_info(self) -> Dict[str, Any]:
        """Get Slack workspace information"""
        try:
            if not await self.authenticate():
                return {"error": "Authentication failed"}
            
            # Get team info
            team_url = f"{self.base_url}/team.info"
            headers = {
                "Authorization": f"Bearer {self.bot_token}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(team_url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            return result.get("team", {})
                        else:
                            return {"error": result.get("error")}
                    else:
                        return {"error": f"HTTP error: {response.status}"}
        except Exception as e:
            logger.error(f"Error getting workspace info: {str(e)}")
            return {"error": str(e)}
    
    async def list_channels(self) -> List[Dict[str, Any]]:
        """List available channels"""
        try:
            if not await self.authenticate():
                return []
            
            # Get channels list
            channels_url = f"{self.base_url}/conversations.list"
            params = {"types": "public_channel,private_channel"}
            headers = {
                "Authorization": f"Bearer {self.bot_token}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(channels_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            return result.get("channels", [])
                        else:
                            logger.error(f"Error listing channels: {result.get('error')}")
                            return []
                    else:
                        logger.error(f"HTTP error listing channels: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error listing channels: {str(e)}")
            return []



























