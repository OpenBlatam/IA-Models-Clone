"""
Gamma App - Real Improvement Integration
Integration system for real improvements that actually work
"""

import asyncio
import logging
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import aiohttp

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Integration types"""
    GITHUB = "github"
    JIRA = "jira"
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    API = "api"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"

class IntegrationStatus(Enum):
    """Integration status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    CONFIGURING = "configuring"

@dataclass
class IntegrationConfig:
    """Integration configuration"""
    integration_id: str
    name: str
    type: IntegrationType
    config: Dict[str, Any]
    status: IntegrationStatus = IntegrationStatus.CONFIGURING
    enabled: bool = True
    created_at: datetime = None
    last_sync: Optional[datetime] = None
    sync_interval: int = 300  # seconds
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class IntegrationData:
    """Integration data"""
    data_id: str
    integration_id: str
    data_type: str
    data: Dict[str, Any]
    created_at: datetime = None
    processed: bool = False

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class RealImprovementIntegration:
    """
    Integration system for real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize improvement integration"""
        self.project_root = Path(project_root)
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.integration_data: Dict[str, List[IntegrationData]] = {}
        self.integration_logs: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize with default integrations
        self._initialize_default_integrations()
        
        logger.info(f"Real Improvement Integration initialized for {self.project_root}")
    
    def _initialize_default_integrations(self):
        """Initialize default integrations"""
        # GitHub integration
        github_integration = IntegrationConfig(
            integration_id="github_main",
            name="GitHub Main Repository",
            type=IntegrationType.GITHUB,
            config={
                "repository": "your-org/your-repo",
                "token": "your-github-token",
                "base_url": "https://api.github.com",
                "webhook_secret": "your-webhook-secret"
            },
            sync_interval=600
        )
        self.integrations[github_integration.integration_id] = github_integration
        
        # JIRA integration
        jira_integration = IntegrationConfig(
            integration_id="jira_main",
            name="JIRA Main Project",
            type=IntegrationType.JIRA,
            config={
                "base_url": "https://your-domain.atlassian.net",
                "username": "your-email@domain.com",
                "api_token": "your-jira-token",
                "project_key": "PROJ"
            },
            sync_interval=900
        )
        self.integrations[jira_integration.integration_id] = jira_integration
        
        # Slack integration
        slack_integration = IntegrationConfig(
            integration_id="slack_main",
            name="Slack Main Workspace",
            type=IntegrationType.SLACK,
            config={
                "bot_token": "xoxb-your-slack-bot-token",
                "channel": "#improvements",
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
            },
            sync_interval=300
        )
        self.integrations[slack_integration.integration_id] = slack_integration
    
    def create_integration(self, name: str, type: IntegrationType, 
                          config: Dict[str, Any], sync_interval: int = 300) -> str:
        """Create integration"""
        try:
            integration_id = f"integration_{int(time.time() * 1000)}"
            
            integration = IntegrationConfig(
                integration_id=integration_id,
                name=name,
                type=type,
                config=config,
                sync_interval=sync_interval
            )
            
            self.integrations[integration_id] = integration
            self.integration_data[integration_id] = []
            self.integration_logs[integration_id] = []
            
            logger.info(f"Integration created: {name}")
            return integration_id
            
        except Exception as e:
            logger.error(f"Failed to create integration: {e}")
            raise
    
    async def test_integration(self, integration_id: str) -> Dict[str, Any]:
        """Test integration connection"""
        try:
            if integration_id not in self.integrations:
                return {"success": False, "error": "Integration not found"}
            
            integration = self.integrations[integration_id]
            
            # Test based on integration type
            if integration.type == IntegrationType.GITHUB:
                result = await self._test_github_integration(integration)
            elif integration.type == IntegrationType.JIRA:
                result = await self._test_jira_integration(integration)
            elif integration.type == IntegrationType.SLACK:
                result = await self._test_slack_integration(integration)
            elif integration.type == IntegrationType.WEBHOOK:
                result = await self._test_webhook_integration(integration)
            elif integration.type == IntegrationType.API:
                result = await self._test_api_integration(integration)
            else:
                result = {"success": False, "error": f"Unknown integration type: {integration.type}"}
            
            # Update integration status
            if result["success"]:
                integration.status = IntegrationStatus.ACTIVE
                self._log_integration(integration_id, "test_success", "Integration test successful")
            else:
                integration.status = IntegrationStatus.ERROR
                self._log_integration(integration_id, "test_failed", f"Integration test failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to test integration: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_github_integration(self, integration: IntegrationConfig) -> Dict[str, Any]:
        """Test GitHub integration"""
        try:
            config = integration.config
            headers = {
                "Authorization": f"token {config['token']}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Test API connection
            async with aiohttp.ClientSession() as session:
                url = f"{config['base_url']}/repos/{config['repository']}"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "message": f"Connected to repository: {data['full_name']}",
                            "data": {
                                "repository": data['full_name'],
                                "stars": data['stargazers_count'],
                                "forks": data['forks_count']
                            }
                        }
                    else:
                        return {"success": False, "error": f"GitHub API error: {response.status}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_jira_integration(self, integration: IntegrationConfig) -> Dict[str, Any]:
        """Test JIRA integration"""
        try:
            config = integration.config
            auth = (config['username'], config['api_token'])
            
            # Test API connection
            async with aiohttp.ClientSession() as session:
                url = f"{config['base_url']}/rest/api/3/myself"
                async with session.get(url, auth=auth) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "message": f"Connected to JIRA as: {data['displayName']}",
                            "data": {
                                "user": data['displayName'],
                                "email": data['emailAddress']
                            }
                        }
                    else:
                        return {"success": False, "error": f"JIRA API error: {response.status}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_slack_integration(self, integration: IntegrationConfig) -> Dict[str, Any]:
        """Test Slack integration"""
        try:
            config = integration.config
            
            # Test bot token
            headers = {"Authorization": f"Bearer {config['bot_token']}"}
            
            async with aiohttp.ClientSession() as session:
                url = "https://slack.com/api/auth.test"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('ok'):
                            return {
                                "success": True,
                                "message": f"Connected to Slack as: {data['user']}",
                                "data": {
                                    "user": data['user'],
                                    "team": data['team']
                                }
                            }
                        else:
                            return {"success": False, "error": f"Slack API error: {data.get('error', 'Unknown error')}"}
                    else:
                        return {"success": False, "error": f"Slack API error: {response.status}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_webhook_integration(self, integration: IntegrationConfig) -> Dict[str, Any]:
        """Test webhook integration"""
        try:
            config = integration.config
            
            # Test webhook endpoint
            test_payload = {
                "test": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['url'],
                    json=test_payload,
                    headers=config.get('headers', {}),
                    timeout=30
                ) as response:
                    if response.status in [200, 201, 202]:
                        return {
                            "success": True,
                            "message": "Webhook endpoint is accessible",
                            "data": {"status_code": response.status}
                        }
                    else:
                        return {"success": False, "error": f"Webhook error: {response.status}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_api_integration(self, integration: IntegrationConfig) -> Dict[str, Any]:
        """Test API integration"""
        try:
            config = integration.config
            
            # Test API endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    config['url'],
                    headers=config.get('headers', {}),
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "message": "API endpoint is accessible",
                            "data": data
                        }
                    else:
                        return {"success": False, "error": f"API error: {response.status}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def sync_integration(self, integration_id: str) -> Dict[str, Any]:
        """Sync integration data"""
        try:
            if integration_id not in self.integrations:
                return {"success": False, "error": "Integration not found"}
            
            integration = self.integrations[integration_id]
            
            if not integration.enabled:
                return {"success": False, "error": "Integration is disabled"}
            
            self._log_integration(integration_id, "sync_started", "Integration sync started")
            
            # Sync based on integration type
            if integration.type == IntegrationType.GITHUB:
                result = await self._sync_github_integration(integration)
            elif integration.type == IntegrationType.JIRA:
                result = await self._sync_jira_integration(integration)
            elif integration.type == IntegrationType.SLACK:
                result = await self._sync_slack_integration(integration)
            else:
                result = {"success": False, "error": f"Sync not implemented for type: {integration.type}"}
            
            # Update last sync time
            integration.last_sync = datetime.utcnow()
            
            if result["success"]:
                self._log_integration(integration_id, "sync_completed", "Integration sync completed successfully")
            else:
                integration.retry_count += 1
                self._log_integration(integration_id, "sync_failed", f"Integration sync failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to sync integration: {e}")
            return {"success": False, "error": str(e)}
    
    async def _sync_github_integration(self, integration: IntegrationConfig) -> Dict[str, Any]:
        """Sync GitHub integration"""
        try:
            config = integration.config
            headers = {
                "Authorization": f"token {config['token']}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Get recent commits
            async with aiohttp.ClientSession() as session:
                url = f"{config['base_url']}/repos/{config['repository']}/commits"
                async with session.get(url, headers=headers, params={"per_page": 10}) as response:
                    if response.status == 200:
                        commits = await response.json()
                        
                        # Process commits
                        for commit in commits:
                            data = IntegrationData(
                                data_id=f"commit_{commit['sha']}",
                                integration_id=integration.integration_id,
                                data_type="commit",
                                data={
                                    "sha": commit['sha'],
                                    "message": commit['commit']['message'],
                                    "author": commit['commit']['author']['name'],
                                    "date": commit['commit']['author']['date'],
                                    "url": commit['html_url']
                                }
                            )
                            self.integration_data[integration.integration_id].append(data)
                        
                        return {
                            "success": True,
                            "message": f"Synced {len(commits)} commits",
                            "data_count": len(commits)
                        }
                    else:
                        return {"success": False, "error": f"GitHub API error: {response.status}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _sync_jira_integration(self, integration: IntegrationConfig) -> Dict[str, Any]:
        """Sync JIRA integration"""
        try:
            config = integration.config
            auth = (config['username'], config['api_token'])
            
            # Get recent issues
            async with aiohttp.ClientSession() as session:
                url = f"{config['base_url']}/rest/api/3/search"
                params = {
                    "jql": f"project = {config['project_key']} ORDER BY updated DESC",
                    "maxResults": 10
                }
                async with session.get(url, auth=auth, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        issues = data.get('issues', [])
                        
                        # Process issues
                        for issue in issues:
                            data = IntegrationData(
                                data_id=f"issue_{issue['key']}",
                                integration_id=integration.integration_id,
                                data_type="issue",
                                data={
                                    "key": issue['key'],
                                    "summary": issue['fields']['summary'],
                                    "status": issue['fields']['status']['name'],
                                    "assignee": issue['fields'].get('assignee', {}).get('displayName'),
                                    "updated": issue['fields']['updated']
                                }
                            )
                            self.integration_data[integration.integration_id].append(data)
                        
                        return {
                            "success": True,
                            "message": f"Synced {len(issues)} issues",
                            "data_count": len(issues)
                        }
                    else:
                        return {"success": False, "error": f"JIRA API error: {response.status}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _sync_slack_integration(self, integration: IntegrationConfig) -> Dict[str, Any]:
        """Sync Slack integration"""
        try:
            config = integration.config
            headers = {"Authorization": f"Bearer {config['bot_token']}"}
            
            # Get channel history
            async with aiohttp.ClientSession() as session:
                url = "https://slack.com/api/conversations.history"
                params = {
                    "channel": config['channel'],
                    "limit": 10
                }
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        messages = data.get('messages', [])
                        
                        # Process messages
                        for message in messages:
                            data = IntegrationData(
                                data_id=f"message_{message['ts']}",
                                integration_id=integration.integration_id,
                                data_type="message",
                                data={
                                    "text": message.get('text', ''),
                                    "user": message.get('user', ''),
                                    "timestamp": message['ts'],
                                    "type": message.get('type', 'message')
                                }
                            )
                            self.integration_data[integration.integration_id].append(data)
                        
                        return {
                            "success": True,
                            "message": f"Synced {len(messages)} messages",
                            "data_count": len(messages)
                        }
                    else:
                        return {"success": False, "error": f"Slack API error: {response.status}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _log_integration(self, integration_id: str, event: str, message: str):
        """Log integration event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if integration_id not in self.integration_logs:
            self.integration_logs[integration_id] = []
        
        self.integration_logs[integration_id].append(log_entry)
        
        logger.info(f"Integration {integration_id}: {event} - {message}")
    
    def get_integration_data(self, integration_id: str, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get integration data"""
        try:
            if integration_id not in self.integration_data:
                return []
            
            data_list = self.integration_data[integration_id]
            
            if data_type:
                data_list = [d for d in data_list if d.data_type == data_type]
            
            return [
                {
                    "data_id": d.data_id,
                    "data_type": d.data_type,
                    "data": d.data,
                    "created_at": d.created_at.isoformat(),
                    "processed": d.processed
                }
                for d in data_list
            ]
            
        except Exception as e:
            logger.error(f"Failed to get integration data: {e}")
            return []
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get integration summary"""
        total_integrations = len(self.integrations)
        active_integrations = len([i for i in self.integrations.values() if i.status == IntegrationStatus.ACTIVE])
        error_integrations = len([i for i in self.integrations.values() if i.status == IntegrationStatus.ERROR])
        
        total_data = sum(len(data) for data in self.integration_data.values())
        
        return {
            "total_integrations": total_integrations,
            "active_integrations": active_integrations,
            "error_integrations": error_integrations,
            "total_data_points": total_data,
            "integration_types": list(set(i.type.value for i in self.integrations.values())),
            "last_sync_times": {
                i.integration_id: i.last_sync.isoformat() if i.last_sync else None
                for i in self.integrations.values()
            }
        }
    
    def get_integration_logs(self, integration_id: str) -> List[Dict[str, Any]]:
        """Get integration logs"""
        return self.integration_logs.get(integration_id, [])
    
    def enable_integration(self, integration_id: str) -> bool:
        """Enable integration"""
        try:
            if integration_id in self.integrations:
                self.integrations[integration_id].enabled = True
                self._log_integration(integration_id, "enabled", "Integration enabled")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to enable integration: {e}")
            return False
    
    def disable_integration(self, integration_id: str) -> bool:
        """Disable integration"""
        try:
            if integration_id in self.integrations:
                self.integrations[integration_id].enabled = False
                self._log_integration(integration_id, "disabled", "Integration disabled")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to disable integration: {e}")
            return False

# Global integration instance
improvement_integration = None

def get_improvement_integration() -> RealImprovementIntegration:
    """Get improvement integration instance"""
    global improvement_integration
    if not improvement_integration:
        improvement_integration = RealImprovementIntegration()
    return improvement_integration













