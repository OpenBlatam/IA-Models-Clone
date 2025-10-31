"""
Content Service

This service orchestrates content-related functionality including
content lifecycle management, content analysis, and content operations.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.base import BaseService
from ..core.config import SystemConfig
from ..core.exceptions import ContentError, StorageError

logger = logging.getLogger(__name__)


class ContentService(BaseService[Dict[str, Any]]):
    """Service for managing content operations and lifecycle"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._content_engines = {}
        self._analyzers = {}
    
    async def _start(self) -> bool:
        """Start the content service"""
        try:
            # Initialize content engines
            await self._initialize_content_engines()
            
            # Initialize analyzers
            await self._initialize_analyzers()
            
            logger.info("Content service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start content service: {e}")
            return False
    
    async def _stop(self) -> bool:
        """Stop the content service"""
        try:
            # Cleanup analyzers
            await self._cleanup_analyzers()
            
            # Cleanup content engines
            await self._cleanup_content_engines()
            
            logger.info("Content service stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop content service: {e}")
            return False
    
    async def _initialize_content_engines(self):
        """Initialize content engines"""
        try:
            # Import here to avoid circular imports
            from ..engines.content_lifecycle_engine import ContentLifecycleEngine
            from ..engines.content_governance_engine import ContentGovernanceEngine
            
            # Initialize content lifecycle engine
            if self.config.features.get("content_lifecycle", False):
                self._content_engines["lifecycle"] = ContentLifecycleEngine(self.config)
                await self._content_engines["lifecycle"].initialize()
            
            # Initialize content governance engine
            if self.config.features.get("content_governance", False):
                self._content_engines["governance"] = ContentGovernanceEngine(self.config)
                await self._content_engines["governance"].initialize()
            
            logger.info("Content engines initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize content engines: {e}")
            raise ContentError(f"Failed to initialize content engines: {str(e)}")
    
    async def _initialize_analyzers(self):
        """Initialize content analyzers"""
        try:
            # Import here to avoid circular imports
            from ..analyzers.content_analyzer import ContentAnalyzer
            
            # Initialize content analyzer
            if self.config.features.get("content_analysis", False):
                self._analyzers["content"] = ContentAnalyzer(self.config)
                await self._analyzers["content"].initialize()
            
            logger.info("Content analyzers initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize content analyzers: {e}")
            raise ContentError(f"Failed to initialize content analyzers: {str(e)}")
    
    async def _cleanup_content_engines(self):
        """Cleanup content engines"""
        try:
            for engine_name, engine in self._content_engines.items():
                await engine.shutdown()
            
            self._content_engines.clear()
            logger.info("Content engines cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup content engines: {e}")
    
    async def _cleanup_analyzers(self):
        """Cleanup analyzers"""
        try:
            for analyzer_name, analyzer in self._analyzers.items():
                await analyzer.shutdown()
            
            self._analyzers.clear()
            logger.info("Content analyzers cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup analyzers: {e}")
    
    async def create_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new content with analysis and governance checks"""
        try:
            if not self._initialized:
                raise ContentError("Content service not initialized")
            
            content = content_data.get("content", "")
            content_type = content_data.get("content_type", "text")
            metadata = content_data.get("metadata", {})
            
            # Create content using lifecycle engine
            if "lifecycle" not in self._content_engines:
                raise ContentError("Content lifecycle engine not available")
            
            lifecycle_result = await self._content_engines["lifecycle"].create_content(
                content, content_type, metadata
            )
            
            # Perform content analysis
            analysis_result = {}
            if "content" in self._analyzers:
                analysis_result = await self._analyzers["content"].analyze(content)
            
            # Check governance compliance
            governance_result = {}
            if "governance" in self._content_engines:
                governance_result = await self._content_engines["governance"].check_content_compliance(
                    content
                )
            
            return {
                "content_id": lifecycle_result.get("content_id"),
                "lifecycle": lifecycle_result,
                "analysis": analysis_result,
                "governance": governance_result,
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Failed to create content: {e}")
            raise ContentError(f"Failed to create content: {str(e)}")
    
    async def analyze_content(self, content: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze content using available analyzers"""
        try:
            if not self._initialized:
                raise ContentError("Content service not initialized")
            
            if "content" not in self._analyzers:
                raise ContentError("Content analyzer not available")
            
            result = await self._analyzers["content"].analyze(content, analysis_type=analysis_type)
            
            return result
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            raise ContentError(f"Content analysis failed: {str(e)}")
    
    async def update_content(self, content_id: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing content"""
        try:
            if not self._initialized:
                raise ContentError("Content service not initialized")
            
            if "lifecycle" not in self._content_engines:
                raise ContentError("Content lifecycle engine not available")
            
            content = content_data.get("content", "")
            metadata = content_data.get("metadata", {})
            
            # Update content using lifecycle engine
            lifecycle_result = await self._content_engines["lifecycle"].update_content(
                content_id, content, metadata
            )
            
            # Re-analyze updated content
            analysis_result = {}
            if "content" in self._analyzers:
                analysis_result = await self._analyzers["content"].analyze(content)
            
            # Re-check governance compliance
            governance_result = {}
            if "governance" in self._content_engines:
                governance_result = await self._content_engines["governance"].check_content_compliance(
                    content
                )
            
            return {
                "content_id": content_id,
                "lifecycle": lifecycle_result,
                "analysis": analysis_result,
                "governance": governance_result,
                "status": "updated"
            }
            
        except Exception as e:
            logger.error(f"Failed to update content: {e}")
            raise ContentError(f"Failed to update content: {str(e)}")
    
    async def search_content(self, query: str, filters: Dict[str, Any] = None, 
                           limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Search content using lifecycle engine"""
        try:
            if not self._initialized:
                raise ContentError("Content service not initialized")
            
            if "lifecycle" not in self._content_engines:
                raise ContentError("Content lifecycle engine not available")
            
            results = await self._content_engines["lifecycle"].search_content(
                query, filters, limit, offset
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            raise ContentError(f"Content search failed: {str(e)}")
    
    async def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get content by ID"""
        try:
            if not self._initialized:
                raise ContentError("Content service not initialized")
            
            if "lifecycle" not in self._content_engines:
                raise ContentError("Content lifecycle engine not available")
            
            content = await self._content_engines["lifecycle"].get_content(content_id)
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to get content: {e}")
            raise ContentError(f"Failed to get content: {str(e)}")
    
    async def delete_content(self, content_id: str) -> Dict[str, Any]:
        """Delete content"""
        try:
            if not self._initialized:
                raise ContentError("Content service not initialized")
            
            if "lifecycle" not in self._content_engines:
                raise ContentError("Content lifecycle engine not available")
            
            result = await self._content_engines["lifecycle"].delete_content(content_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete content: {e}")
            raise ContentError(f"Failed to delete content: {str(e)}")
    
    async def batch_analyze_content(self, content_list: List[str], 
                                  analysis_type: str = "comprehensive") -> List[Dict[str, Any]]:
        """Analyze multiple content pieces in batch"""
        try:
            if not self._initialized:
                raise ContentError("Content service not initialized")
            
            if "content" not in self._analyzers:
                raise ContentError("Content analyzer not available")
            
            results = await self._analyzers["content"].batch_analyze(
                content_list, analysis_type=analysis_type
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Batch content analysis failed: {e}")
            raise ContentError(f"Batch content analysis failed: {str(e)}")
    
    def get_content_status(self) -> Dict[str, Any]:
        """Get content service status"""
        base_status = self.get_health_status()
        base_status.update({
            "content_engines": list(self._content_engines.keys()),
            "analyzers": list(self._analyzers.keys()),
            "features_enabled": {
                "content_lifecycle": "lifecycle" in self._content_engines,
                "content_governance": "governance" in self._content_engines,
                "content_analysis": "content" in self._analyzers
            }
        })
        return base_status





















