"""
Chatbot domain services
"""

from services.domains import register_service

try:
    from services.chatbot_service import ChatbotService
    
    def register_services():
        register_service("chatbot", "chatbot", ChatbotService)
except ImportError:
    pass



