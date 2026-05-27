"""
🚀 TruthGPT Python API - System 5.9 Gold Standard
Official entry point for the TruthGPT System.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TruthGPT.API")

class TruthGPT_API:
    """
    High-level API for TruthGPT interaction.
    Connects to the Swarm Orchestrator and Research Hub.
    """

    def __init__(self):
        self._orchestrator = None
        self._registry = None

    async def _ensure_initialized(self):
        """Lazy load components to avoid circular imports."""
        if self._orchestrator is None:
            try:
                from optimization_core.agents.client import AgentClient
                from optimization_core.agents.engines import engine_registry
                llm = engine_registry.get_engine("deepseek")
                self._orchestrator = AgentClient(use_swarm=True, llm_engine=llm)
            except ImportError:
                logger.error("Failed to load Swarm Orchestrator components.")

        if self._registry is None:
            try:
                from optimization_core.modules.base.core_system.core.papers.paper_registry import PaperRegistry
                self._registry = PaperRegistry()
            except ImportError:
                logger.error("Failed to load Paper Registry.")

    async def ask(self, prompt: str, user_id: str = "default_user") -> str:
        """Ask the TruthGPT Swarm a question."""
        await self._ensure_initialized()
        if not self._orchestrator:
            return "Error: Orchestrator offline."
        
        response = await self._orchestrator.run(user_id=user_id, prompt=prompt, return_response=True)
        return response.content

    def list_papers(self, limit: int = 10):
        """List discovered SOTA papers."""
        # Non-async version for basic listing
        from optimization_core.modules.base.core_system.core.papers.paper_registry import PaperRegistry
        reg = PaperRegistry()
        return reg.list_papers()[:limit]

    def get_paper_info(self, paper_id: str):
        """Get details for a specific paper."""
        from optimization_core.modules.base.core_system.core.papers.paper_registry import PaperRegistry
        reg = PaperRegistry()
        papers = reg.list_papers()
        return next((p for p in papers if p.paper_id == paper_id), None)

    def apply_paper(self, paper_id: str):
        """Apply a paper's optimization techniques."""
        from optimization_core.modules.base.core_system.core.papers.paper_registry import PaperRegistry
        reg = PaperRegistry()
        return reg.load_paper(paper_id)

# Singleton Instance
api = TruthGPT_API()

# Re-exporting common methods for direct access
async def ask(prompt: str, user_id: str = "default_user"):
    return await api.ask(prompt, user_id)

def list_papers(limit: int = 10):
    return api.list_papers(limit)
