"""
LUT Manager for Color Grading AI
=================================

Manages LUT (Look-Up Table) files for color grading.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LUTInfo:
    """LUT information."""
    name: str
    path: str
    format: str  # cube, 3dl, etc.
    description: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class LUTManager:
    """
    Manages LUT files.
    
    Features:
    - Load and validate LUTs
    - List available LUTs
    - Apply LUTs to media
    - Convert between LUT formats
    """
    
    def __init__(self, luts_dir: str = "luts"):
        """
        Initialize LUT manager.
        
        Args:
            luts_dir: Directory containing LUT files
        """
        self.luts_dir = Path(luts_dir)
        self.luts_dir.mkdir(parents=True, exist_ok=True)
        self._luts: Dict[str, LUTInfo] = {}
        self._scan_luts()
    
    def _scan_luts(self):
        """Scan directory for LUT files."""
        lut_extensions = [".cube", ".3dl", ".csp", ".dat"]
        
        for lut_file in self.luts_dir.glob("*"):
            if lut_file.suffix.lower() in lut_extensions:
                lut_info = LUTInfo(
                    name=lut_file.stem,
                    path=str(lut_file),
                    format=lut_file.suffix.lower()[1:],  # Remove dot
                    category="custom"
                )
                self._luts[lut_info.name] = lut_info
        
        logger.info(f"Scanned {len(self._luts)} LUT files")
    
    def get_lut(self, name: str) -> Optional[LUTInfo]:
        """
        Get LUT by name.
        
        Args:
            name: LUT name
            
        Returns:
            LUTInfo or None
        """
        return self._luts.get(name)
    
    def list_luts(
        self,
        category: Optional[str] = None,
        format: Optional[str] = None
    ) -> List[LUTInfo]:
        """
        List available LUTs.
        
        Args:
            category: Filter by category
            format: Filter by format
            
        Returns:
            List of LUTInfo
        """
        luts = list(self._luts.values())
        
        if category:
            luts = [l for l in luts if l.category == category]
        
        if format:
            luts = [l for l in luts if l.format == format]
        
        return luts
    
    def add_lut(
        self,
        name: str,
        path: str,
        format: str,
        description: Optional[str] = None,
        category: Optional[str] = None
    ) -> LUTInfo:
        """
        Add LUT to manager.
        
        Args:
            name: LUT name
            path: Path to LUT file
            format: LUT format
            description: Optional description
            category: Optional category
            
        Returns:
            LUTInfo
        """
        lut_info = LUTInfo(
            name=name,
            path=path,
            format=format,
            description=description,
            category=category
        )
        
        self._luts[name] = lut_info
        return lut_info
    
    def validate_lut(self, lut_path: str) -> bool:
        """
        Validate LUT file.
        
        Args:
            lut_path: Path to LUT file
            
        Returns:
            True if valid
        """
        # Basic validation - check file exists and has correct extension
        path = Path(lut_path)
        if not path.exists():
            return False
        
        valid_extensions = [".cube", ".3dl", ".csp", ".dat"]
        return path.suffix.lower() in valid_extensions




