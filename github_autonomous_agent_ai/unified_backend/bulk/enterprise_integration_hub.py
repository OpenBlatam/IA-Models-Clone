"""
BUL Enterprise Integration Hub
=============================

Advanced enterprise integration hub for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import requests
import sqlite3
from dataclasses import dataclass
from enum import Enum
import yaml
import xml.etree.ElementTree as ET
from collections import defaultdict
import base64
import hashlib
import hmac

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Integration types."""
    REST_API = "rest_api"
    SOAP_API = "soap_api"
    WEBHOOK = "webhook"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    MESSAGE_QUEUE = "message_queue"
    EMAIL = "email"
    SMS = "sms"
    FTP = "ftp"
    SFTP = "sftp"

class AuthenticationType(Enum):
    """Authentication types."""
    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"

@dataclass
class IntegrationEndpoint:
    """Integration endpoint definition."""
    id: str
    name: str
    description: str
    integration_type: IntegrationType
    url: str
    authentication: AuthenticationType
    credentials: Dict[str, Any]
    headers: Dict[str, str]
    parameters: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = None

@dataclass
class IntegrationMapping:
    """Integration data mapping definition."""
    id: str
    name: str
    source_endpoint: str
    target_endpoint: str
    field_mappings: Dict[str, str]
    transformations: List[Dict[str, Any]]
    schedule: Optional[str] = None
    enabled: bool = True
    created_at: datetime = None

@dataclass
class IntegrationJob:
    """Integration job definition."""
    id: str
    name: str
    mapping_id: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    records_processed: int = 0
    records_successful: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None

class EnterpriseIntegrationHub:
    """Advanced enterprise integration hub for BUL system."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.endpoints = {}
        self.mappings = {}
        self.jobs = {}
        self.integration_history = []
        self.init_integration_environment()
        self.load_endpoints()
        self.load_mappings()
        self.load_jobs()
    
    def init_integration_environment(self):
        """Initialize integration environment."""
        print("🔗 Initializing enterprise integration environment...")
        
        # Create integration directories
        self.integration_dir = Path("enterprise_integration")
        self.integration_dir.mkdir(exist_ok=True)
        
        self.mappings_dir = Path("integration_mappings")
        self.mappings_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path("integration_logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.init_integration_database()
        
        print("✅ Enterprise integration environment initialized")
    
    def init_integration_database(self):
        """Initialize integration database."""
        conn = sqlite3.connect("enterprise_integration.db")
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integration_endpoints (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                integration_type TEXT,
                url TEXT,
                authentication TEXT,
                credentials TEXT,
                headers TEXT,
                parameters TEXT,
                enabled BOOLEAN,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integration_mappings (
                id TEXT PRIMARY KEY,
                name TEXT,
                source_endpoint TEXT,
                target_endpoint TEXT,
                field_mappings TEXT,
                transformations TEXT,
                schedule TEXT,
                enabled BOOLEAN,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integration_jobs (
                id TEXT PRIMARY KEY,
                name TEXT,
                mapping_id TEXT,
                status TEXT,
                started_at DATETIME,
                completed_at DATETIME,
                records_processed INTEGER,
                records_successful INTEGER,
                records_failed INTEGER,
                error_message TEXT,
                result_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_endpoints(self):
        """Load existing integration endpoints."""
        conn = sqlite3.connect("enterprise_integration.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM integration_endpoints")
        rows = cursor.fetchall()
        
        for row in rows:
            endpoint = IntegrationEndpoint(
                id=row[0],
                name=row[1],
                description=row[2],
                integration_type=IntegrationType(row[3]),
                url=row[4],
                authentication=AuthenticationType(row[5]),
                credentials=json.loads(row[6]) if row[6] else {},
                headers=json.loads(row[7]) if row[7] else {},
                parameters=json.loads(row[8]) if row[8] else {},
                enabled=bool(row[9]),
                created_at=datetime.fromisoformat(row[10])
            )
            self.endpoints[endpoint.id] = endpoint
        
        conn.close()
        
        # Create default endpoints if none exist
        if not self.endpoints:
            self.create_default_endpoints()
        
        print(f"✅ Loaded {len(self.endpoints)} integration endpoints")
    
    def load_mappings(self):
        """Load existing integration mappings."""
        conn = sqlite3.connect("enterprise_integration.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM integration_mappings")
        rows = cursor.fetchall()
        
        for row in rows:
            mapping = IntegrationMapping(
                id=row[0],
                name=row[1],
                source_endpoint=row[2],
                target_endpoint=row[3],
                field_mappings=json.loads(row[4]),
                transformations=json.loads(row[5]),
                schedule=row[6],
                enabled=bool(row[7]),
                created_at=datetime.fromisoformat(row[8])
            )
            self.mappings[mapping.id] = mapping
        
        conn.close()
        
        print(f"✅ Loaded {len(self.mappings)} integration mappings")
    
    def load_jobs(self):
        """Load existing integration jobs."""
        conn = sqlite3.connect("enterprise_integration.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM integration_jobs")
        rows = cursor.fetchall()
        
        for row in rows:
            job = IntegrationJob(
                id=row[0],
                name=row[1],
                mapping_id=row[2],
                status=row[3],
                started_at=datetime.fromisoformat(row[4]) if row[4] else None,
                completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
                records_processed=row[6] or 0,
                records_successful=row[7] or 0,
                records_failed=row[8] or 0,
                error_message=row[9],
                result_data=json.loads(row[10]) if row[10] else None
            )
            self.jobs[job.id] = job
        
        conn.close()
        
        print(f"✅ Loaded {len(self.jobs)} integration jobs")
    
    def create_default_endpoints(self):
        """Create default integration endpoints."""
        default_endpoints = [
            {
                'id': 'bul_api',
                'name': 'BUL API',
                'description': 'BUL system API endpoint',
                'integration_type': IntegrationType.REST_API,
                'url': 'http://localhost:8000',
                'authentication': AuthenticationType.NONE,
                'credentials': {},
                'headers': {'Content-Type': 'application/json'},
                'parameters': {}
            },
            {
                'id': 'salesforce_api',
                'name': 'Salesforce API',
                'description': 'Salesforce CRM integration',
                'integration_type': IntegrationType.REST_API,
                'url': 'https://api.salesforce.com',
                'authentication': AuthenticationType.OAUTH2,
                'credentials': {'client_id': '', 'client_secret': ''},
                'headers': {'Content-Type': 'application/json'},
                'parameters': {}
            },
            {
                'id': 'hubspot_api',
                'name': 'HubSpot API',
                'description': 'HubSpot marketing automation',
                'integration_type': IntegrationType.REST_API,
                'url': 'https://api.hubapi.com',
                'authentication': AuthenticationType.API_KEY,
                'credentials': {'api_key': ''},
                'headers': {'Content-Type': 'application/json'},
                'parameters': {}
            },
            {
                'id': 'slack_webhook',
                'name': 'Slack Webhook',
                'description': 'Slack notification webhook',
                'integration_type': IntegrationType.WEBHOOK,
                'url': 'https://hooks.slack.com/services/...',
                'authentication': AuthenticationType.NONE,
                'credentials': {},
                'headers': {'Content-Type': 'application/json'},
                'parameters': {}
            },
            {
                'id': 'email_smtp',
                'name': 'Email SMTP',
                'description': 'Email integration via SMTP',
                'integration_type': IntegrationType.EMAIL,
                'url': 'smtp.gmail.com:587',
                'authentication': AuthenticationType.BASIC,
                'credentials': {'username': '', 'password': ''},
                'headers': {},
                'parameters': {}
            }
        ]
        
        for endpoint_data in default_endpoints:
            self.create_endpoint(
                endpoint_id=endpoint_data['id'],
                name=endpoint_data['name'],
                description=endpoint_data['description'],
                integration_type=endpoint_data['integration_type'],
                url=endpoint_data['url'],
                authentication=endpoint_data['authentication'],
                credentials=endpoint_data['credentials'],
                headers=endpoint_data['headers'],
                parameters=endpoint_data['parameters']
            )
    
    def create_endpoint(self, endpoint_id: str, name: str, description: str,
                       integration_type: IntegrationType, url: str,
                       authentication: AuthenticationType, credentials: Dict[str, Any],
                       headers: Dict[str, str], parameters: Dict[str, Any]) -> IntegrationEndpoint:
        """Create a new integration endpoint."""
        endpoint = IntegrationEndpoint(
            id=endpoint_id,
            name=name,
            description=description,
            integration_type=integration_type,
            url=url,
            authentication=authentication,
            credentials=credentials,
            headers=headers,
            parameters=parameters,
            enabled=True,
            created_at=datetime.now()
        )
        
        self.endpoints[endpoint_id] = endpoint
        
        # Save to database
        conn = sqlite3.connect("enterprise_integration.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO integration_endpoints 
            (id, name, description, integration_type, url, authentication, 
             credentials, headers, parameters, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (endpoint_id, name, description, integration_type.value, url, 
              authentication.value, json.dumps(credentials), json.dumps(headers),
              json.dumps(parameters), True, endpoint.created_at.isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created integration endpoint: {name}")
        return endpoint
    
    def create_mapping(self, mapping_id: str, name: str, source_endpoint: str,
                      target_endpoint: str, field_mappings: Dict[str, str],
                      transformations: List[Dict[str, Any]] = None,
                      schedule: str = None) -> IntegrationMapping:
        """Create a new integration mapping."""
        if source_endpoint not in self.endpoints:
            raise ValueError(f"Source endpoint {source_endpoint} not found")
        
        if target_endpoint not in self.endpoints:
            raise ValueError(f"Target endpoint {target_endpoint} not found")
        
        mapping = IntegrationMapping(
            id=mapping_id,
            name=name,
            source_endpoint=source_endpoint,
            target_endpoint=target_endpoint,
            field_mappings=field_mappings,
            transformations=transformations or [],
            schedule=schedule,
            enabled=True,
            created_at=datetime.now()
        )
        
        self.mappings[mapping_id] = mapping
        
        # Save to database
        conn = sqlite3.connect("enterprise_integration.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO integration_mappings 
            (id, name, source_endpoint, target_endpoint, field_mappings, 
             transformations, schedule, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (mapping_id, name, source_endpoint, target_endpoint, 
              json.dumps(field_mappings), json.dumps(transformations or []),
              schedule, True, mapping.created_at.isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created integration mapping: {name}")
        return mapping
    
    async def execute_integration(self, mapping_id: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute an integration mapping."""
        if mapping_id not in self.mappings:
            raise ValueError(f"Mapping {mapping_id} not found")
        
        mapping = self.mappings[mapping_id]
        source_endpoint = self.endpoints[mapping.source_endpoint]
        target_endpoint = self.endpoints[mapping.target_endpoint]
        
        print(f"🔗 Executing integration: {mapping.name}")
        print(f"   Source: {source_endpoint.name}")
        print(f"   Target: {target_endpoint.name}")
        
        # Create job
        job_id = f"job_{mapping_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = IntegrationJob(
            id=job_id,
            name=f"Integration Job {mapping.name}",
            mapping_id=mapping_id,
            status="running",
            started_at=datetime.now()
        )
        
        self.jobs[job_id] = job
        self._save_job(job)
        
        try:
            # Get data from source
            source_data = await self._get_data_from_source(source_endpoint, data)
            
            # Transform data
            transformed_data = await self._transform_data(source_data, mapping)
            
            # Send data to target
            result = await self._send_data_to_target(target_endpoint, transformed_data)
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.now()
            job.records_processed = len(transformed_data) if isinstance(transformed_data, list) else 1
            job.records_successful = job.records_processed
            job.result_data = result
            
            self._save_job(job)
            self._log_integration(mapping_id, "completed", result)
            
            print(f"✅ Integration completed successfully")
            print(f"   Records processed: {job.records_processed}")
            
            return {
                'job_id': job_id,
                'status': 'completed',
                'records_processed': job.records_processed,
                'records_successful': job.records_successful,
                'result': result
            }
            
        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.now()
            job.error_message = str(e)
            
            self._save_job(job)
            self._log_integration(mapping_id, "failed", {'error': str(e)})
            
            logger.error(f"Integration {mapping_id} failed: {e}")
            raise
    
    async def _get_data_from_source(self, endpoint: IntegrationEndpoint, data: Optional[Dict[str, Any]] = None) -> Any:
        """Get data from source endpoint."""
        if endpoint.integration_type == IntegrationType.REST_API:
            return await self._get_data_from_rest_api(endpoint, data)
        elif endpoint.integration_type == IntegrationType.SOAP_API:
            return await self._get_data_from_soap_api(endpoint, data)
        elif endpoint.integration_type == IntegrationType.DATABASE:
            return await self._get_data_from_database(endpoint, data)
        elif endpoint.integration_type == IntegrationType.FILE_SYSTEM:
            return await self._get_data_from_file_system(endpoint, data)
        else:
            raise ValueError(f"Unsupported source integration type: {endpoint.integration_type}")
    
    async def _send_data_to_target(self, endpoint: IntegrationEndpoint, data: Any) -> Dict[str, Any]:
        """Send data to target endpoint."""
        if endpoint.integration_type == IntegrationType.REST_API:
            return await self._send_data_to_rest_api(endpoint, data)
        elif endpoint.integration_type == IntegrationType.SOAP_API:
            return await self._send_data_to_soap_api(endpoint, data)
        elif endpoint.integration_type == IntegrationType.WEBHOOK:
            return await self._send_data_to_webhook(endpoint, data)
        elif endpoint.integration_type == IntegrationType.EMAIL:
            return await self._send_data_to_email(endpoint, data)
        elif endpoint.integration_type == IntegrationType.SMS:
            return await self._send_data_to_sms(endpoint, data)
        else:
            raise ValueError(f"Unsupported target integration type: {endpoint.integration_type}")
    
    async def _get_data_from_rest_api(self, endpoint: IntegrationEndpoint, data: Optional[Dict[str, Any]] = None) -> Any:
        """Get data from REST API endpoint."""
        headers = endpoint.headers.copy()
        
        # Add authentication
        if endpoint.authentication == AuthenticationType.BASIC:
            username = endpoint.credentials.get('username', '')
            password = endpoint.credentials.get('password', '')
            auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers['Authorization'] = f"Basic {auth_string}"
        
        elif endpoint.authentication == AuthenticationType.BEARER:
            token = endpoint.credentials.get('token', '')
            headers['Authorization'] = f"Bearer {token}"
        
        elif endpoint.authentication == AuthenticationType.API_KEY:
            api_key = endpoint.credentials.get('api_key', '')
            key_name = endpoint.credentials.get('key_name', 'X-API-Key')
            headers[key_name] = api_key
        
        # Make request
        response = requests.get(endpoint.url, headers=headers, params=endpoint.parameters)
        response.raise_for_status()
        
        return response.json()
    
    async def _send_data_to_rest_api(self, endpoint: IntegrationEndpoint, data: Any) -> Dict[str, Any]:
        """Send data to REST API endpoint."""
        headers = endpoint.headers.copy()
        
        # Add authentication
        if endpoint.authentication == AuthenticationType.BASIC:
            username = endpoint.credentials.get('username', '')
            password = endpoint.credentials.get('password', '')
            auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers['Authorization'] = f"Basic {auth_string}"
        
        elif endpoint.authentication == AuthenticationType.BEARER:
            token = endpoint.credentials.get('token', '')
            headers['Authorization'] = f"Bearer {token}"
        
        elif endpoint.authentication == AuthenticationType.API_KEY:
            api_key = endpoint.credentials.get('api_key', '')
            key_name = endpoint.credentials.get('key_name', 'X-API-Key')
            headers[key_name] = api_key
        
        # Make request
        response = requests.post(endpoint.url, json=data, headers=headers, params=endpoint.parameters)
        response.raise_for_status()
        
        return {
            'status_code': response.status_code,
            'response': response.json() if response.content else None
        }
    
    async def _send_data_to_webhook(self, endpoint: IntegrationEndpoint, data: Any) -> Dict[str, Any]:
        """Send data to webhook endpoint."""
        headers = endpoint.headers.copy()
        
        # Add webhook signature if configured
        if 'secret' in endpoint.credentials:
            secret = endpoint.credentials['secret']
            payload = json.dumps(data)
            signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            headers['X-Hub-Signature-256'] = f"sha256={signature}"
        
        # Make request
        response = requests.post(endpoint.url, json=data, headers=headers)
        response.raise_for_status()
        
        return {
            'status_code': response.status_code,
            'response': response.text
        }
    
    async def _send_data_to_email(self, endpoint: IntegrationEndpoint, data: Any) -> Dict[str, Any]:
        """Send data to email endpoint."""
        # This would implement email sending
        # For now, return a placeholder
        return {
            'status': 'sent',
            'recipient': endpoint.credentials.get('recipient', ''),
            'subject': data.get('subject', 'Integration Data')
        }
    
    async def _send_data_to_sms(self, endpoint: IntegrationEndpoint, data: Any) -> Dict[str, Any]:
        """Send data to SMS endpoint."""
        # This would implement SMS sending
        # For now, return a placeholder
        return {
            'status': 'sent',
            'recipient': endpoint.credentials.get('recipient', ''),
            'message': data.get('message', 'Integration Data')
        }
    
    async def _get_data_from_soap_api(self, endpoint: IntegrationEndpoint, data: Optional[Dict[str, Any]] = None) -> Any:
        """Get data from SOAP API endpoint."""
        # This would implement SOAP API integration
        # For now, return a placeholder
        return {}
    
    async def _send_data_to_soap_api(self, endpoint: IntegrationEndpoint, data: Any) -> Dict[str, Any]:
        """Send data to SOAP API endpoint."""
        # This would implement SOAP API integration
        # For now, return a placeholder
        return {'status': 'sent'}
    
    async def _get_data_from_database(self, endpoint: IntegrationEndpoint, data: Optional[Dict[str, Any]] = None) -> Any:
        """Get data from database endpoint."""
        # This would implement database integration
        # For now, return a placeholder
        return []
    
    async def _get_data_from_file_system(self, endpoint: IntegrationEndpoint, data: Optional[Dict[str, Any]] = None) -> Any:
        """Get data from file system endpoint."""
        # This would implement file system integration
        # For now, return a placeholder
        return {}
    
    async def _transform_data(self, data: Any, mapping: IntegrationMapping) -> Any:
        """Transform data according to mapping rules."""
        if isinstance(data, list):
            transformed_data = []
            for item in data:
                transformed_item = {}
                
                # Apply field mappings
                for source_field, target_field in mapping.field_mappings.items():
                    if source_field in item:
                        transformed_item[target_field] = item[source_field]
                
                # Apply transformations
                for transformation in mapping.transformations:
                    transformed_item = self._apply_transformation(transformed_item, transformation)
                
                transformed_data.append(transformed_item)
            
            return transformed_data
        
        else:
            # Single item transformation
            transformed_item = {}
            
            # Apply field mappings
            for source_field, target_field in mapping.field_mappings.items():
                if source_field in data:
                    transformed_item[target_field] = data[source_field]
            
            # Apply transformations
            for transformation in mapping.transformations:
                transformed_item = self._apply_transformation(transformed_item, transformation)
            
            return transformed_item
    
    def _apply_transformation(self, data: Dict[str, Any], transformation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single transformation to data."""
        transformation_type = transformation.get('type')
        field = transformation.get('field')
        value = transformation.get('value')
        
        if transformation_type == 'set':
            data[field] = value
        
        elif transformation_type == 'format':
            if field in data:
                format_string = transformation.get('format', '{}')
                data[field] = format_string.format(data[field])
        
        elif transformation_type == 'uppercase':
            if field in data:
                data[field] = str(data[field]).upper()
        
        elif transformation_type == 'lowercase':
            if field in data:
                data[field] = str(data[field]).lower()
        
        elif transformation_type == 'concat':
            fields = transformation.get('fields', [])
            separator = transformation.get('separator', ' ')
            values = [str(data.get(f, '')) for f in fields]
            data[field] = separator.join(values)
        
        elif transformation_type == 'split':
            if field in data:
                separator = transformation.get('separator', ',')
                data[field] = str(data[field]).split(separator)
        
        elif transformation_type == 'replace':
            if field in data:
                old_value = transformation.get('old_value', '')
                new_value = transformation.get('new_value', '')
                data[field] = str(data[field]).replace(old_value, new_value)
        
        return data
    
    def _save_job(self, job: IntegrationJob):
        """Save integration job to database."""
        conn = sqlite3.connect("enterprise_integration.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO integration_jobs 
            (id, name, mapping_id, status, started_at, completed_at, 
             records_processed, records_successful, records_failed, 
             error_message, result_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (job.id, job.name, job.mapping_id, job.status,
              job.started_at.isoformat() if job.started_at else None,
              job.completed_at.isoformat() if job.completed_at else None,
              job.records_processed, job.records_successful, job.records_failed,
              job.error_message, json.dumps(job.result_data) if job.result_data else None))
        
        conn.commit()
        conn.close()
    
    def _log_integration(self, mapping_id: str, status: str, result: Dict[str, Any]):
        """Log integration event."""
        integration_event = {
            'mapping_id': mapping_id,
            'status': status,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        self.integration_history.append(integration_event)
        
        # Save integration history
        with open(self.logs_dir / "integration_history.json", 'w') as f:
            json.dump(self.integration_history, f, indent=2)
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        total_jobs = len(self.jobs)
        completed_jobs = len([j for j in self.jobs.values() if j.status == 'completed'])
        failed_jobs = len([j for j in self.jobs.values() if j.status == 'failed'])
        running_jobs = len([j for j in self.jobs.values() if j.status == 'running'])
        
        total_records = sum(j.records_processed for j in self.jobs.values())
        successful_records = sum(j.records_successful for j in self.jobs.values())
        
        return {
            'total_endpoints': len(self.endpoints),
            'total_mappings': len(self.mappings),
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'running_jobs': running_jobs,
            'success_rate': (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            'total_records_processed': total_records,
            'successful_records': successful_records,
            'record_success_rate': (successful_records / total_records * 100) if total_records > 0 else 0
        }
    
    def generate_integration_report(self) -> str:
        """Generate enterprise integration report."""
        stats = self.get_integration_stats()
        
        report = f"""
BUL Enterprise Integration Hub Report
====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ENDPOINTS
---------
Total Endpoints: {stats['total_endpoints']}
"""
        
        for endpoint_id, endpoint in self.endpoints.items():
            report += f"""
{endpoint.name} ({endpoint_id}):
  Type: {endpoint.integration_type.value}
  URL: {endpoint.url}
  Authentication: {endpoint.authentication.value}
  Enabled: {endpoint.enabled}
  Created: {endpoint.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report += f"""
MAPPINGS
--------
Total Mappings: {stats['total_mappings']}
"""
        
        for mapping_id, mapping in self.mappings.items():
            report += f"""
{mapping.name} ({mapping_id}):
  Source: {mapping.source_endpoint}
  Target: {mapping.target_endpoint}
  Fields: {len(mapping.field_mappings)}
  Transformations: {len(mapping.transformations)}
  Schedule: {mapping.schedule or 'Manual'}
  Enabled: {mapping.enabled}
  Created: {mapping.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report += f"""
STATISTICS
----------
Total Jobs: {stats['total_jobs']}
Completed: {stats['completed_jobs']}
Failed: {stats['failed_jobs']}
Running: {stats['running_jobs']}
Success Rate: {stats['success_rate']:.1f}%

Records Processed: {stats['total_records_processed']}
Successful Records: {stats['successful_records']}
Record Success Rate: {stats['record_success_rate']:.1f}%
"""
        
        # Show recent jobs
        recent_jobs = sorted(
            self.jobs.values(),
            key=lambda x: x.started_at or datetime.min,
            reverse=True
        )[:10]
        
        if recent_jobs:
            report += f"""
RECENT JOBS
-----------
"""
            for job in recent_jobs:
                report += f"""
{job.name} ({job.id}):
  Status: {job.status}
  Started: {job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else 'N/A'}
  Records: {job.records_processed}
  Success: {job.records_successful}
"""
        
        return report

def main():
    """Main enterprise integration hub function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Enterprise Integration Hub")
    parser.add_argument("--create-endpoint", help="Create a new integration endpoint")
    parser.add_argument("--create-mapping", help="Create a new integration mapping")
    parser.add_argument("--execute-integration", help="Execute an integration mapping")
    parser.add_argument("--list-endpoints", action="store_true", help="List all endpoints")
    parser.add_argument("--list-mappings", action="store_true", help="List all mappings")
    parser.add_argument("--list-jobs", action="store_true", help="List all jobs")
    parser.add_argument("--stats", action="store_true", help="Show integration statistics")
    parser.add_argument("--report", action="store_true", help="Generate integration report")
    parser.add_argument("--name", help="Name for endpoint/mapping")
    parser.add_argument("--description", help="Description for endpoint/mapping")
    parser.add_argument("--integration-type", choices=['rest_api', 'soap_api', 'webhook', 'database', 'file_system', 'message_queue', 'email', 'sms', 'ftp', 'sftp'],
                       help="Integration type")
    parser.add_argument("--url", help="Endpoint URL")
    parser.add_argument("--authentication", choices=['none', 'basic', 'bearer', 'api_key', 'oauth2', 'custom'],
                       help="Authentication type")
    parser.add_argument("--source-endpoint", help="Source endpoint ID for mapping")
    parser.add_argument("--target-endpoint", help="Target endpoint ID for mapping")
    parser.add_argument("--field-mappings", help="JSON string of field mappings")
    parser.add_argument("--transformations", help="JSON string of transformations")
    parser.add_argument("--schedule", help="Schedule for mapping")
    
    args = parser.parse_args()
    
    integration_hub = EnterpriseIntegrationHub()
    
    print("🔗 BUL Enterprise Integration Hub")
    print("=" * 40)
    
    if args.create_endpoint:
        if not all([args.name, args.description, args.integration_type, args.url, args.authentication]):
            print("❌ Error: --name, --description, --integration-type, --url, and --authentication are required")
            return 1
        
        endpoint = integration_hub.create_endpoint(
            endpoint_id=args.create_endpoint,
            name=args.name,
            description=args.description,
            integration_type=IntegrationType(args.integration_type),
            url=args.url,
            authentication=AuthenticationType(args.authentication),
            credentials={},
            headers={'Content-Type': 'application/json'},
            parameters={}
        )
        print(f"✅ Created endpoint: {endpoint.name}")
    
    elif args.create_mapping:
        if not all([args.name, args.description, args.source_endpoint, args.target_endpoint]):
            print("❌ Error: --name, --description, --source-endpoint, and --target-endpoint are required")
            return 1
        
        field_mappings = {}
        if args.field_mappings:
            try:
                field_mappings = json.loads(args.field_mappings)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --field-mappings")
                return 1
        
        transformations = []
        if args.transformations:
            try:
                transformations = json.loads(args.transformations)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --transformations")
                return 1
        
        mapping = integration_hub.create_mapping(
            mapping_id=args.create_mapping,
            name=args.name,
            description=args.description,
            source_endpoint=args.source_endpoint,
            target_endpoint=args.target_endpoint,
            field_mappings=field_mappings,
            transformations=transformations,
            schedule=args.schedule
        )
        print(f"✅ Created mapping: {mapping.name}")
    
    elif args.execute_integration:
        async def execute_integration():
            try:
                result = await integration_hub.execute_integration(args.execute_integration)
                print(f"✅ Integration executed successfully")
                print(f"   Job ID: {result['job_id']}")
                print(f"   Records processed: {result['records_processed']}")
                print(f"   Records successful: {result['records_successful']}")
            except Exception as e:
                print(f"❌ Integration execution failed: {e}")
        
        asyncio.run(execute_integration())
    
    elif args.list_endpoints:
        endpoints = integration_hub.endpoints
        if endpoints:
            print(f"\n🔗 Integration Endpoints ({len(endpoints)}):")
            print("-" * 60)
            for endpoint_id, endpoint in endpoints.items():
                print(f"{endpoint.name} ({endpoint_id}):")
                print(f"  Type: {endpoint.integration_type.value}")
                print(f"  URL: {endpoint.url}")
                print(f"  Authentication: {endpoint.authentication.value}")
                print(f"  Enabled: {endpoint.enabled}")
                print()
        else:
            print("No endpoints found.")
    
    elif args.list_mappings:
        mappings = integration_hub.mappings
        if mappings:
            print(f"\n🔗 Integration Mappings ({len(mappings)}):")
            print("-" * 60)
            for mapping_id, mapping in mappings.items():
                print(f"{mapping.name} ({mapping_id}):")
                print(f"  Source: {mapping.source_endpoint}")
                print(f"  Target: {mapping.target_endpoint}")
                print(f"  Fields: {len(mapping.field_mappings)}")
                print(f"  Schedule: {mapping.schedule or 'Manual'}")
                print()
        else:
            print("No mappings found.")
    
    elif args.list_jobs:
        jobs = integration_hub.jobs
        if jobs:
            print(f"\n🔗 Integration Jobs ({len(jobs)}):")
            print("-" * 60)
            for job_id, job in jobs.items():
                print(f"{job.name} ({job_id}):")
                print(f"  Status: {job.status}")
                print(f"  Started: {job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else 'N/A'}")
                print(f"  Records: {job.records_processed}")
                print()
        else:
            print("No jobs found.")
    
    elif args.stats:
        stats = integration_hub.get_integration_stats()
        print(f"\n📊 Integration Statistics:")
        print(f"   Total Endpoints: {stats['total_endpoints']}")
        print(f"   Total Mappings: {stats['total_mappings']}")
        print(f"   Total Jobs: {stats['total_jobs']}")
        print(f"   Completed: {stats['completed_jobs']}")
        print(f"   Failed: {stats['failed_jobs']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        print(f"   Records Processed: {stats['total_records_processed']}")
        print(f"   Record Success Rate: {stats['record_success_rate']:.1f}%")
    
    elif args.report:
        report = integration_hub.generate_integration_report()
        print(report)
        
        # Save report
        report_file = f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        stats = integration_hub.get_integration_stats()
        print(f"🔗 Endpoints: {stats['total_endpoints']}")
        print(f"🔗 Mappings: {stats['total_mappings']}")
        print(f"🔗 Jobs: {stats['total_jobs']}")
        print(f"✅ Success Rate: {stats['success_rate']:.1f}%")
        print(f"📊 Records Processed: {stats['total_records_processed']}")
        print(f"\n💡 Use --list-endpoints to see all endpoints")
        print(f"💡 Use --create-mapping to create a new mapping")
        print(f"💡 Use --execute-integration to run an integration")
        print(f"💡 Use --report to generate integration report")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
