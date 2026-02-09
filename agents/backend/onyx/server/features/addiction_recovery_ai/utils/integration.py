"""
Integration with External Systems
"""

import requests
from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class ExternalAPIClient:
    """Client for external API integration"""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0
    ):
        """
        Initialize external API client
        
        Args:
            base_url: Base URL
            api_key: Optional API key
            timeout: Request timeout
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        logger.info(f"ExternalAPIClient initialized: {base_url}")
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """GET request"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"GET request failed: {e}")
            raise
    
    def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """POST request"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.post(
                url,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"POST request failed: {e}")
            raise


class WebhookHandler:
    """Handle webhooks for external integrations"""
    
    def __init__(self):
        """Initialize webhook handler"""
        self.webhooks = {}
        logger.info("WebhookHandler initialized")
    
    def register_webhook(
        self,
        event_type: str,
        url: str,
        secret: Optional[str] = None
    ):
        """
        Register webhook
        
        Args:
            event_type: Event type
            url: Webhook URL
            secret: Optional secret for signing
        """
        self.webhooks[event_type] = {
            "url": url,
            "secret": secret
        }
        logger.info(f"Webhook registered: {event_type} -> {url}")
    
    def trigger_webhook(
        self,
        event_type: str,
        payload: Dict[str, Any]
    ):
        """
        Trigger webhook
        
        Args:
            event_type: Event type
            payload: Payload data
        """
        if event_type not in self.webhooks:
            return
        
        webhook = self.webhooks[event_type]
        
        try:
            response = requests.post(
                webhook["url"],
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()
            logger.info(f"Webhook triggered: {event_type}")
        except Exception as e:
            logger.error(f"Webhook failed: {e}")


class DataExporter:
    """Export data to external systems"""
    
    def __init__(self):
        """Initialize data exporter"""
        logger.info("DataExporter initialized")
    
    def export_to_json(
        self,
        data: Any,
        filepath: str,
        indent: int = 2
    ):
        """Export to JSON"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        logger.info(f"Data exported to JSON: {filepath}")
    
    def export_to_csv(
        self,
        data: List[Dict[str, Any]],
        filepath: str
    ):
        """Export to CSV"""
        import csv
        
        if not data:
            return
        
        fieldnames = data[0].keys()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Data exported to CSV: {filepath}")
    
    def export_to_database(
        self,
        data: List[Dict[str, Any]],
        connection_string: str,
        table_name: str
    ):
        """
        Export to database (placeholder)
        
        Args:
            data: Data to export
            connection_string: Database connection string
            table_name: Table name
        """
        logger.warning("Database export not implemented. Requires database driver.")
        # In practice, you'd use SQLAlchemy, psycopg2, etc.

