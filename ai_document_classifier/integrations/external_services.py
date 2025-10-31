"""
External Services Integration
=============================

Integration with external services for enhanced document classification,
template generation, and document processing.
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
from datetime import datetime, timedelta
import hashlib
import base64
from pathlib import Path

logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """Types of external services"""
    AI_CLASSIFICATION = "ai_classification"
    DOCUMENT_GENERATION = "document_generation"
    TRANSLATION = "translation"
    GRAMMAR_CHECK = "grammar_check"
    PLAGIARISM_CHECK = "plagiarism_check"
    CONTENT_ANALYSIS = "content_analysis"
    TEMPLATE_STORAGE = "template_storage"

@dataclass
class ServiceConfig:
    """Configuration for external service"""
    name: str
    service_type: ServiceType
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    rate_limit: Optional[int] = None
    enabled: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class ServiceResponse:
    """Response from external service"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    processing_time: float = 0.0
    service_name: str = ""
    timestamp: datetime = None

class ExternalServiceManager:
    """
    Manager for external service integrations
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize external service manager
        
        Args:
            config_file: Path to service configuration file
        """
        self.services: Dict[str, ServiceConfig] = {}
        self.service_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        # Load service configurations
        if config_file:
            self.load_config(config_file)
        else:
            self._load_default_services()
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Document-Classifier/1.0',
            'Content-Type': 'application/json'
        })
    
    def _load_default_services(self):
        """Load default service configurations"""
        default_services = [
            ServiceConfig(
                name="openai_gpt",
                service_type=ServiceType.AI_CLASSIFICATION,
                base_url="https://api.openai.com/v1",
                timeout=30,
                retry_attempts=3,
                enabled=False  # Requires API key
            ),
            ServiceConfig(
                name="huggingface",
                service_type=ServiceType.AI_CLASSIFICATION,
                base_url="https://api-inference.huggingface.co",
                timeout=30,
                retry_attempts=3,
                enabled=False  # Requires API key
            ),
            ServiceConfig(
                name="google_translate",
                service_type=ServiceType.TRANSLATION,
                base_url="https://translation.googleapis.com",
                timeout=30,
                retry_attempts=3,
                enabled=False  # Requires API key
            ),
            ServiceConfig(
                name="grammarly",
                service_type=ServiceType.GRAMMAR_CHECK,
                base_url="https://api.grammarly.com",
                timeout=30,
                retry_attempts=3,
                enabled=False  # Requires API key
            )
        ]
        
        for service in default_services:
            self.services[service.name] = service
    
    def load_config(self, config_file: str):
        """Load service configurations from file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            for service_data in config_data.get('services', []):
                service = ServiceConfig(**service_data)
                self.services[service.name] = service
            
            logger.info(f"Loaded {len(self.services)} service configurations")
        except Exception as e:
            logger.error(f"Failed to load service config: {e}")
    
    def add_service(self, service: ServiceConfig):
        """Add a new service configuration"""
        self.services[service.name] = service
        logger.info(f"Added service: {service.name}")
    
    def remove_service(self, service_name: str):
        """Remove a service configuration"""
        if service_name in self.services:
            del self.services[service_name]
            logger.info(f"Removed service: {service_name}")
    
    def enable_service(self, service_name: str, api_key: Optional[str] = None):
        """Enable a service with optional API key"""
        if service_name in self.services:
            self.services[service_name].enabled = True
            if api_key:
                self.services[service_name].api_key = api_key
            logger.info(f"Enabled service: {service_name}")
    
    def disable_service(self, service_name: str):
        """Disable a service"""
        if service_name in self.services:
            self.services[service_name].enabled = False
            logger.info(f"Disabled service: {service_name}")
    
    def classify_with_external_ai(
        self, 
        text: str, 
        service_name: str = "openai_gpt",
        model: str = "gpt-3.5-turbo"
    ) -> ServiceResponse:
        """
        Classify document using external AI service
        
        Args:
            text: Text to classify
            service_name: Name of the AI service to use
            model: Model to use for classification
            
        Returns:
            ServiceResponse with classification result
        """
        if service_name not in self.services:
            return ServiceResponse(
                success=False,
                error=f"Service {service_name} not found",
                service_name=service_name
            )
        
        service = self.services[service_name]
        if not service.enabled:
            return ServiceResponse(
                success=False,
                error=f"Service {service_name} is disabled",
                service_name=service_name
            )
        
        start_time = datetime.now()
        
        try:
            if service_name == "openai_gpt":
                return self._classify_with_openai(text, service, model)
            elif service_name == "huggingface":
                return self._classify_with_huggingface(text, service, model)
            else:
                return ServiceResponse(
                    success=False,
                    error=f"Unsupported AI service: {service_name}",
                    service_name=service_name
                )
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ServiceResponse(
                success=False,
                error=str(e),
                service_name=service_name,
                processing_time=processing_time
            )
    
    def _classify_with_openai(self, text: str, service: ServiceConfig, model: str) -> ServiceResponse:
        """Classify using OpenAI API"""
        if not service.api_key:
            return ServiceResponse(
                success=False,
                error="OpenAI API key not provided",
                service_name=service.name
            )
        
        start_time = datetime.now()
        
        headers = {
            'Authorization': f'Bearer {service.api_key}',
            'Content-Type': 'application/json'
        }
        
        prompt = f"""
        Analyze the following document description and classify it into one of these types:
        - novel (fiction, story, narrative)
        - contract (legal agreement, terms)
        - design (technical, architectural, engineering)
        - business_plan (business strategy, market analysis)
        - academic_paper (research, study, analysis)
        - technical_manual (instructions, procedures, guides)
        - marketing_material (campaigns, promotions, advertisements)
        - user_manual (user guides, tutorials, help)
        - report (analysis, findings, recommendations)
        - proposal (suggestions, recommendations, project plans)
        
        Document description: "{text}"
        
        Respond with a JSON object containing:
        - document_type: the classified type
        - confidence: confidence score (0.0 to 1.0)
        - keywords: list of relevant keywords found
        - reasoning: brief explanation of the classification
        """
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.3
        }
        
        response = self.session.post(
            f"{service.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=service.timeout
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            try:
                classification_data = json.loads(content)
                return ServiceResponse(
                    success=True,
                    data=classification_data,
                    status_code=response.status_code,
                    processing_time=processing_time,
                    service_name=service.name
                )
            except json.JSONDecodeError:
                return ServiceResponse(
                    success=False,
                    error="Failed to parse OpenAI response",
                    status_code=response.status_code,
                    processing_time=processing_time,
                    service_name=service.name
                )
        else:
            return ServiceResponse(
                success=False,
                error=f"OpenAI API error: {response.text}",
                status_code=response.status_code,
                processing_time=processing_time,
                service_name=service.name
            )
    
    def _classify_with_huggingface(self, text: str, service: ServiceConfig, model: str) -> ServiceResponse:
        """Classify using Hugging Face API"""
        if not service.api_key:
            return ServiceResponse(
                success=False,
                error="Hugging Face API key not provided",
                service_name=service.name
            )
        
        start_time = datetime.now()
        
        headers = {
            'Authorization': f'Bearer {service.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "inputs": text,
            "parameters": {
                "return_all_scores": True,
                "max_length": 512
            }
        }
        
        response = self.session.post(
            f"{service.base_url}/models/{model}",
            headers=headers,
            json=payload,
            timeout=service.timeout
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if response.status_code == 200:
            result = response.json()
            return ServiceResponse(
                success=True,
                data=result,
                status_code=response.status_code,
                processing_time=processing_time,
                service_name=service.name
            )
        else:
            return ServiceResponse(
                success=False,
                error=f"Hugging Face API error: {response.text}",
                status_code=response.status_code,
                processing_time=processing_time,
                service_name=service.name
            )
    
    def translate_text(
        self, 
        text: str, 
        target_language: str = "es",
        service_name: str = "google_translate"
    ) -> ServiceResponse:
        """
        Translate text using external translation service
        
        Args:
            text: Text to translate
            target_language: Target language code
            service_name: Translation service to use
            
        Returns:
            ServiceResponse with translation result
        """
        if service_name not in self.services:
            return ServiceResponse(
                success=False,
                error=f"Service {service_name} not found",
                service_name=service_name
            )
        
        service = self.services[service_name]
        if not service.enabled:
            return ServiceResponse(
                success=False,
                error=f"Service {service_name} is disabled",
                service_name=service_name
            )
        
        start_time = datetime.now()
        
        try:
            if service_name == "google_translate":
                return self._translate_with_google(text, target_language, service)
            else:
                return ServiceResponse(
                    success=False,
                    error=f"Unsupported translation service: {service_name}",
                    service_name=service_name
                )
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ServiceResponse(
                success=False,
                error=str(e),
                service_name=service_name,
                processing_time=processing_time
            )
    
    def _translate_with_google(self, text: str, target_language: str, service: ServiceConfig) -> ServiceResponse:
        """Translate using Google Translate API"""
        if not service.api_key:
            return ServiceResponse(
                success=False,
                error="Google Translate API key not provided",
                service_name=service.name
            )
        
        start_time = datetime.now()
        
        params = {
            'key': service.api_key,
            'q': text,
            'target': target_language,
            'format': 'text'
        }
        
        response = self.session.post(
            f"{service.base_url}/language/translate/v2",
            params=params,
            timeout=service.timeout
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if response.status_code == 200:
            result = response.json()
            translated_text = result['data']['translations'][0]['translatedText']
            
            return ServiceResponse(
                success=True,
                data={
                    'original_text': text,
                    'translated_text': translated_text,
                    'target_language': target_language
                },
                status_code=response.status_code,
                processing_time=processing_time,
                service_name=service.name
            )
        else:
            return ServiceResponse(
                success=False,
                error=f"Google Translate API error: {response.text}",
                status_code=response.status_code,
                processing_time=processing_time,
                service_name=service.name
            )
    
    def check_grammar(self, text: str, service_name: str = "grammarly") -> ServiceResponse:
        """
        Check grammar using external service
        
        Args:
            text: Text to check
            service_name: Grammar service to use
            
        Returns:
            ServiceResponse with grammar check result
        """
        # Placeholder implementation
        # In a real implementation, you would integrate with actual grammar services
        return ServiceResponse(
            success=False,
            error="Grammar check service not implemented",
            service_name=service_name
        )
    
    def analyze_content(self, text: str, analysis_type: str = "sentiment") -> ServiceResponse:
        """
        Analyze content using external services
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (sentiment, readability, etc.)
            
        Returns:
            ServiceResponse with analysis result
        """
        # Placeholder implementation
        return ServiceResponse(
            success=False,
            error="Content analysis service not implemented"
        )
    
    def generate_document(
        self, 
        template_data: Dict[str, Any], 
        format: str = "pdf",
        service_name: str = "document_generator"
    ) -> ServiceResponse:
        """
        Generate document using external service
        
        Args:
            template_data: Template and content data
            format: Output format (pdf, docx, html)
            service_name: Document generation service
            
        Returns:
            ServiceResponse with generated document
        """
        # Placeholder implementation
        return ServiceResponse(
            success=False,
            error="Document generation service not implemented"
        )
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of a specific service"""
        if service_name not in self.services:
            return {"error": f"Service {service_name} not found"}
        
        service = self.services[service_name]
        return {
            "name": service.name,
            "type": service.service_type.value,
            "enabled": service.enabled,
            "has_api_key": bool(service.api_key),
            "timeout": service.timeout,
            "retry_attempts": service.retry_attempts
        }
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services"""
        return {
            name: self.get_service_status(name) 
            for name in self.services.keys()
        }
    
    def test_service_connection(self, service_name: str) -> ServiceResponse:
        """Test connection to a service"""
        if service_name not in self.services:
            return ServiceResponse(
                success=False,
                error=f"Service {service_name} not found",
                service_name=service_name
            )
        
        service = self.services[service_name]
        if not service.enabled:
            return ServiceResponse(
                success=False,
                error=f"Service {service_name} is disabled",
                service_name=service_name
            )
        
        start_time = datetime.now()
        
        try:
            # Simple health check request
            response = self.session.get(
                f"{service.base_url}/health",
                timeout=5,
                headers={'Authorization': f'Bearer {service.api_key}'} if service.api_key else {}
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ServiceResponse(
                success=response.status_code == 200,
                data={"status_code": response.status_code},
                status_code=response.status_code,
                processing_time=processing_time,
                service_name=service_name
            )
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ServiceResponse(
                success=False,
                error=str(e),
                service_name=service_name,
                processing_time=processing_time
            )
    
    def save_config(self, config_file: str):
        """Save service configurations to file"""
        config_data = {
            "services": [
                {
                    "name": service.name,
                    "service_type": service.service_type.value,
                    "base_url": service.base_url,
                    "timeout": service.timeout,
                    "retry_attempts": service.retry_attempts,
                    "rate_limit": service.rate_limit,
                    "enabled": service.enabled,
                    "metadata": service.metadata
                }
                for service in self.services.values()
            ]
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved service configurations to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save service config: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize service manager
    manager = ExternalServiceManager()
    
    # Enable OpenAI service (requires API key)
    # manager.enable_service("openai_gpt", "your-api-key-here")
    
    # Test classification
    # result = manager.classify_with_external_ai("I want to write a novel")
    # print(f"Classification result: {result.success}")
    # if result.success:
    #     print(f"Document type: {result.data.get('document_type')}")
    
    # Get service status
    status = manager.get_all_services_status()
    print("Service Status:")
    for name, service_status in status.items():
        print(f"  {name}: {'Enabled' if service_status['enabled'] else 'Disabled'}")
    
    print("External service manager initialized successfully")



























