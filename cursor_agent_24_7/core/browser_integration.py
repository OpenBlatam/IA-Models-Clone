"""
Browser Integration
===================

Sistema de integración con navegador para inspeccionar páginas web,
siguiendo las mejores prácticas de Devin de no asumir contenido de links
sin visitarlos y usar capacidades de navegación cuando sea necesario.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BrowserSession:
    """Sesión de navegador"""
    session_id: str
    url: str
    visited_at: datetime = field(default_factory=datetime.now)
    content_retrieved: bool = False
    content: Optional[str] = None
    screenshot_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "session_id": self.session_id,
            "url": self.url,
            "visited_at": self.visited_at.isoformat(),
            "content_retrieved": self.content_retrieved,
            "has_content": self.content is not None,
            "has_screenshot": self.screenshot_path is not None
        }


@dataclass
class LinkToVisit:
    """Link a visitar"""
    url: str
    reason: str
    visited: bool = False
    content_retrieved: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "url": self.url,
            "reason": self.reason,
            "visited": self.visited,
            "content_retrieved": self.content_retrieved,
            "timestamp": self.timestamp.isoformat()
        }


class BrowserIntegration:
    """
    Integración con navegador.
    
    Permite inspeccionar páginas web cuando sea necesario,
    siguiendo las mejores prácticas de Devin de no asumir
    contenido de links sin visitarlos.
    """
    
    def __init__(self) -> None:
        """Inicializar integración con navegador"""
        self.sessions: Dict[str, BrowserSession] = {}
        self.links_to_visit: List[LinkToVisit] = []
        self.browser_available: bool = self._check_browser_available()
        logger.info("🌐 Browser integration initialized")
    
    def _check_browser_available(self) -> bool:
        """Verificar si hay capacidades de navegador disponibles"""
        try:
            import playwright
            return True
        except ImportError:
            try:
                import selenium
                return True
            except ImportError:
                logger.warning("No browser automation library available (playwright/selenium)")
                return False
    
    def should_visit_link(self, url: str, reason: str = "") -> bool:
        """
        Determinar si se debe visitar un link.
        
        Según las reglas de Devin:
        - No asumir contenido de links sin visitarlos
        - Usar capacidades de navegación cuando sea necesario
        
        Args:
            url: URL a visitar.
            reason: Razón para visitar (opcional).
        
        Returns:
            True si se debe visitar.
        """
        if not self.browser_available:
            return False
        
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        return True
    
    async def visit_link(
        self,
        url: str,
        reason: str = "",
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visitar un link y obtener su contenido.
        
        Args:
            url: URL a visitar.
            reason: Razón para visitar.
            task_id: ID de la tarea (opcional).
        
        Returns:
            Resultado de la visita.
        """
        if not self.should_visit_link(url, reason):
            return {
                "success": False,
                "error": "Browser not available or invalid URL"
            }
        
        link = LinkToVisit(url=url, reason=reason)
        self.links_to_visit.append(link)
        
        try:
            session_id = f"session_{len(self.sessions)}"
            
            if self._check_browser_available():
                content = await self._visit_with_playwright(url)
            else:
                content = await self._visit_with_requests(url)
            
            session = BrowserSession(
                session_id=session_id,
                url=url,
                content_retrieved=content is not None,
                content=content
            )
            
            self.sessions[session_id] = session
            link.visited = True
            link.content_retrieved = content is not None
            
            logger.info(f"✅ Visited link: {url}")
            
            return {
                "success": True,
                "session_id": session_id,
                "url": url,
                "content_retrieved": content is not None,
                "content_length": len(content) if content else 0
            }
        
        except Exception as e:
            logger.error(f"Error visiting link {url}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _visit_with_playwright(self, url: str) -> Optional[str]:
        """Visitar URL usando Playwright"""
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                content = await page.content()
                await browser.close()
                return content
        except Exception as e:
            logger.warning(f"Could not use Playwright: {e}")
            return None
    
    async def _visit_with_requests(self, url: str) -> Optional[str]:
        """Visitar URL usando requests (fallback)"""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.warning(f"Could not fetch URL: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[BrowserSession]:
        """Obtener sesión"""
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Obtener todas las sesiones"""
        return [s.to_dict() for s in self.sessions.values()]
    
    def get_links_to_visit(self) -> List[Dict[str, Any]]:
        """Obtener todos los links a visitar"""
        return [l.to_dict() for l in self.links_to_visit]

