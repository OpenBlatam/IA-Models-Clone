"""
BUL - Business Universal Language (External Integrations)
========================================================

External API integrations and enterprise service connections.
"""

import asyncio
import aiohttp
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegrationType(str, Enum):
    """Integration type enumeration."""
    CRM = "crm"
    ERP = "erp"
    EMAIL = "email"
    CALENDAR = "calendar"
    DOCUMENT = "document"
    PAYMENT = "payment"
    ANALYTICS = "analytics"
    SOCIAL = "social"

@dataclass
class IntegrationConfig:
    """Integration configuration."""
    name: str
    type: IntegrationType
    base_url: str
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30
    retry_count: int = 3
    enabled: bool = True

class ExternalIntegrations:
    """External integrations manager."""
    
    def __init__(self):
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.setup_default_integrations()
    
    def setup_default_integrations(self):
        """Setup default integrations."""
        # Salesforce CRM
        self.integrations["salesforce"] = IntegrationConfig(
            name="Salesforce CRM",
            type=IntegrationType.CRM,
            base_url="https://api.salesforce.com",
            api_key="your_salesforce_api_key",
            headers={"Authorization": "Bearer your_token"}
        )
        
        # Microsoft Dynamics ERP
        self.integrations["dynamics"] = IntegrationConfig(
            name="Microsoft Dynamics",
            type=IntegrationType.ERP,
            base_url="https://api.dynamics.com",
            api_key="your_dynamics_api_key"
        )
        
        # Gmail Integration
        self.integrations["gmail"] = IntegrationConfig(
            name="Gmail",
            type=IntegrationType.EMAIL,
            base_url="https://gmail.googleapis.com",
            api_key="your_gmail_api_key"
        )
        
        # Google Calendar
        self.integrations["calendar"] = IntegrationConfig(
            name="Google Calendar",
            type=IntegrationType.CALENDAR,
            base_url="https://www.googleapis.com/calendar",
            api_key="your_calendar_api_key"
        )
        
        # Google Drive
        self.integrations["drive"] = IntegrationConfig(
            name="Google Drive",
            type=IntegrationType.DOCUMENT,
            base_url="https://www.googleapis.com/drive",
            api_key="your_drive_api_key"
        )
        
        # Stripe Payment
        self.integrations["stripe"] = IntegrationConfig(
            name="Stripe Payment",
            type=IntegrationType.PAYMENT,
            base_url="https://api.stripe.com",
            api_key="your_stripe_api_key"
        )
        
        # Google Analytics
        self.integrations["analytics"] = IntegrationConfig(
            name="Google Analytics",
            type=IntegrationType.ANALYTICS,
            base_url="https://analytics.googleapis.com",
            api_key="your_analytics_api_key"
        )
        
        # LinkedIn Social
        self.integrations["linkedin"] = IntegrationConfig(
            name="LinkedIn",
            type=IntegrationType.SOCIAL,
            base_url="https://api.linkedin.com",
            api_key="your_linkedin_api_key"
        )
    
    async def start_session(self):
        """Start aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def make_request(self, integration_name: str, endpoint: str, method: str = "GET", 
                          data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make request to external integration."""
        if integration_name not in self.integrations:
            raise ValueError(f"Integration {integration_name} not found")
        
        config = self.integrations[integration_name]
        
        if not config.enabled:
            raise ValueError(f"Integration {integration_name} is disabled")
        
        await self.start_session()
        
        url = f"{config.base_url}/{endpoint.lstrip('/')}"
        headers = config.headers or {}
        
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            ) as response:
                response_data = await response.json()
                
                if response.status >= 400:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"External API error: {response_data}"
                    )
                
                return {
                    "status_code": response.status,
                    "data": response_data,
                    "headers": dict(response.headers)
                }
        
        except aiohttp.ClientError as e:
            logger.error(f"Request error for {integration_name}: {e}")
            raise HTTPException(status_code=500, detail=f"External API request failed: {str(e)}")
    
    async def test_integration(self, integration_name: str) -> Dict[str, Any]:
        """Test integration connectivity."""
        try:
            config = self.integrations[integration_name]
            
            # Test endpoints for different integration types
            test_endpoints = {
                IntegrationType.CRM: "/v1/accounts",
                IntegrationType.ERP: "/v1/entities",
                IntegrationType.EMAIL: "/v1/messages",
                IntegrationType.CALENDAR: "/v3/calendars",
                IntegrationType.DOCUMENT: "/v3/files",
                IntegrationType.PAYMENT: "/v1/charges",
                IntegrationType.ANALYTICS: "/v4/reports",
                IntegrationType.SOCIAL: "/v2/people"
            }
            
            endpoint = test_endpoints.get(config.type, "/v1/test")
            
            result = await self.make_request(integration_name, endpoint)
            
            return {
                "integration": integration_name,
                "status": "connected",
                "response_time": result.get("response_time", 0),
                "last_tested": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                "integration": integration_name,
                "status": "error",
                "error": str(e),
                "last_tested": datetime.now().isoformat()
            }
    
    async def sync_data(self, integration_name: str, data_type: str) -> Dict[str, Any]:
        """Sync data from external integration."""
        try:
            config = self.integrations[integration_name]
            
            # Define sync endpoints for different data types
            sync_endpoints = {
                "contacts": "/v1/contacts",
                "accounts": "/v1/accounts",
                "opportunities": "/v1/opportunities",
                "products": "/v1/products",
                "orders": "/v1/orders",
                "invoices": "/v1/invoices",
                "events": "/v3/events",
                "documents": "/v3/files",
                "analytics": "/v4/reports"
            }
            
            endpoint = sync_endpoints.get(data_type, f"/v1/{data_type}")
            
            result = await self.make_request(integration_name, endpoint)
            
            return {
                "integration": integration_name,
                "data_type": data_type,
                "status": "synced",
                "records_count": len(result.get("data", [])),
                "synced_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                "integration": integration_name,
                "data_type": data_type,
                "status": "error",
                "error": str(e),
                "synced_at": datetime.now().isoformat()
            }

# Global integrations instance
external_integrations = ExternalIntegrations()

# FastAPI app for integrations
app = FastAPI(
    title="BUL External Integrations",
    description="External API integrations and enterprise service connections",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class IntegrationTestRequest(BaseModel):
    integration_name: str = Field(..., description="Name of the integration to test")

class DataSyncRequest(BaseModel):
    integration_name: str = Field(..., description="Name of the integration")
    data_type: str = Field(..., description="Type of data to sync")

class IntegrationConfigRequest(BaseModel):
    name: str = Field(..., description="Integration name")
    type: IntegrationType = Field(..., description="Integration type")
    base_url: str = Field(..., description="Base URL for the integration")
    api_key: Optional[str] = Field(None, description="API key")
    username: Optional[str] = Field(None, description="Username")
    password: Optional[str] = Field(None, description="Password")
    timeout: int = Field(30, description="Request timeout in seconds")
    retry_count: int = Field(3, description="Number of retries")
    enabled: bool = Field(True, description="Whether integration is enabled")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BUL External Integrations",
        "version": "1.0.0",
        "status": "operational",
        "integrations": list(external_integrations.integrations.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/integrations")
async def get_integrations():
    """Get all available integrations."""
    return {
        "integrations": [
            {
                "name": name,
                "type": config.type,
                "base_url": config.base_url,
                "enabled": config.enabled,
                "timeout": config.timeout,
                "retry_count": config.retry_count
            }
            for name, config in external_integrations.integrations.items()
        ],
        "total": len(external_integrations.integrations)
    }

@app.post("/integrations/test")
async def test_integration(request: IntegrationTestRequest):
    """Test integration connectivity."""
    try:
        result = await external_integrations.test_integration(request.integration_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrations/sync")
async def sync_data(request: DataSyncRequest):
    """Sync data from external integration."""
    try:
        result = await external_integrations.sync_data(request.integration_name, request.data_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrations/configure")
async def configure_integration(request: IntegrationConfigRequest):
    """Configure integration settings."""
    try:
        config = IntegrationConfig(
            name=request.name,
            type=request.type,
            base_url=request.base_url,
            api_key=request.api_key,
            username=request.username,
            password=request.password,
            timeout=request.timeout,
            retry_count=request.retry_count,
            enabled=request.enabled
        )
        
        external_integrations.integrations[request.name] = config
        
        return {
            "message": "Integration configured successfully",
            "integration": request.name,
            "status": "configured"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/integrations/{integration_name}/status")
async def get_integration_status(integration_name: str):
    """Get integration status."""
    if integration_name not in external_integrations.integrations:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    config = external_integrations.integrations[integration_name]
    
    return {
        "integration": integration_name,
        "type": config.type,
        "enabled": config.enabled,
        "base_url": config.base_url,
        "timeout": config.timeout,
        "retry_count": config.retry_count
    }

@app.post("/integrations/{integration_name}/enable")
async def enable_integration(integration_name: str):
    """Enable integration."""
    if integration_name not in external_integrations.integrations:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    external_integrations.integrations[integration_name].enabled = True
    
    return {
        "message": f"Integration {integration_name} enabled",
        "status": "enabled"
    }

@app.post("/integrations/{integration_name}/disable")
async def disable_integration(integration_name: str):
    """Disable integration."""
    if integration_name not in external_integrations.integrations:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    external_integrations.integrations[integration_name].enabled = False
    
    return {
        "message": f"Integration {integration_name} disabled",
        "status": "disabled"
    }

@app.get("/integrations/health")
async def get_integrations_health():
    """Get health status of all integrations."""
    health_status = {}
    
    for integration_name in external_integrations.integrations.keys():
        try:
            result = await external_integrations.test_integration(integration_name)
            health_status[integration_name] = result
        except Exception as e:
            health_status[integration_name] = {
                "integration": integration_name,
                "status": "error",
                "error": str(e)
            }
    
    return {
        "health_check": health_status,
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """Startup event."""
    await external_integrations.start_session()
    logger.info("External integrations service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event."""
    await external_integrations.close_session()
    logger.info("External integrations service stopped")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
