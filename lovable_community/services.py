"""
Servicios de negocio para la comunidad Lovable (modularizado)

Este archivo mantiene compatibilidad hacia atrás importando desde los módulos modulares.
Los servicios están ahora organizados en:
- services/ranking.py: RankingService
- services/chat/service.py: ChatService (modular con Repository Pattern)
"""

from .services import *
