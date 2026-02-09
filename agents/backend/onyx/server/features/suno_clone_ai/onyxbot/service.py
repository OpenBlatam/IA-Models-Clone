"""Bot Service - Servicio de bot"""
from typing import Optional
from .base import BaseBot
from chat.service import ChatService
from agents.service import AgentService
from connectors.service import ConnectorService

class BotService:
    def __init__(self, chat_service: Optional[ChatService] = None, agent_service: Optional[AgentService] = None, connector_service: Optional[ConnectorService] = None):
        self.chat_service = chat_service
        self.agent_service = agent_service
        self.connector_service = connector_service

