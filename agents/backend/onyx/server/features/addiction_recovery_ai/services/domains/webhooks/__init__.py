"""
Webhooks domain services
"""

from services.domains import register_service

try:
    from services.webhook_service import WebhookService
    
    def register_services():
        register_service("webhooks", "webhook", WebhookService)
except ImportError:
    pass



