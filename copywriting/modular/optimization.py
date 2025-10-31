from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import sys
import importlib.util
from typing import Dict, Any, List, Tuple
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Modular Optimization Detection.

Clean optimization detection and performance assessment.
"""


@dataclass
class OptimizationInfo:
    """Information about an optimization library."""
    name: str
    available: bool
    speedup: str
    impact: str
    description: str
    install_command: str

class OptimizationDetector:
    """Clean optimization detection and assessment."""
    
    def __init__(self) -> Any:
        self.optimizations: Dict[str, OptimizationInfo] = {}
        self.performance_level = "BASIC"
        self.total_speedup = 1.0
        
    def detect_all(self) -> Dict[str, OptimizationInfo]:
        """Detect all optimizations."""
        optimizations_to_check = [
            # JSON Processing
            ("orjson", "5x", "HIGH", "Ultra-fast JSON processing", "pip install orjson"),
            ("ujson", "3x", "MEDIUM", "Fast JSON processing", "pip install ujson"),
            
            # Data Processing  
            ("polars", "10x", "HIGH", "Ultra-fast dataframes", "pip install polars"),
            ("numpy", "5x", "MEDIUM", "Numerical computing", "pip install numpy"),
            
            # Async & Event Loop
            ("uvloop", "4x", "HIGH", "Fast event loop (Unix only)", "pip install uvloop"),
            ("httpx", "2x", "MEDIUM", "Modern async HTTP client", "pip install httpx"),
            ("aiofiles", "2x", "MEDIUM", "Async file operations", "pip install aiofiles"),
            
            # Caching
            ("redis", "3x", "HIGH", "High-performance caching", "pip install redis"),
            ("hiredis", "2x", "MEDIUM", "Fast Redis protocol", "pip install hiredis"),
            
            # Compression & Hashing
            ("xxhash", "4x", "MEDIUM", "Fast hashing", "pip install xxhash"),
            ("lz4", "3x", "MEDIUM", "Fast compression", "pip install lz4"),
            
            # Monitoring
            ("prometheus_client", "1x", "LOW", "Production metrics", "pip install prometheus-client"),
            ("structlog", "1x", "LOW", "Structured logging", "pip install structlog"),
        ]
        
        for name, speedup, impact, description, install_cmd in optimizations_to_check:
            self.optimizations[name] = self._check_optimization(
                name, speedup, impact, description, install_cmd
            )
        
        self._calculate_performance_level()
        return self.optimizations
    
    def _check_optimization(self, name: str, speedup: str, impact: str, 
                          description: str, install_cmd: str) -> OptimizationInfo:
        """Check if an optimization is available."""
        
        # Platform-specific checks
        if name == "uvloop" and sys.platform == "win32":
            return OptimizationInfo(
                name=name,
                available=False,
                speedup="N/A",
                impact="PLATFORM",
                description=f"{description} (Windows not supported)",
                install_command=install_cmd
            )
        
        # Check if library is available
        try:
            spec = importlib.util.find_spec(name)
            available = spec is not None
        except (ImportError, ValueError):
            available = False
        
        return OptimizationInfo(
            name=name,
            available=available,
            speedup=speedup if available else "0x",
            impact=impact,
            description=description,
            install_command=install_cmd
        )
    
    def _calculate_performance_level(self) -> Any:
        """Calculate performance level and total speedup."""
        available_count = sum(1 for opt in self.optimizations.values() if opt.available)
        high_impact_count = sum(
            1 for opt in self.optimizations.values() 
            if opt.available and opt.impact == "HIGH"
        )
        
        # Calculate performance level
        if high_impact_count >= 3:
            self.performance_level = "ULTRA"
        elif high_impact_count >= 2:
            self.performance_level = "HIGH"  
        elif available_count >= 3:
            self.performance_level = "MEDIUM"
        else:
            self.performance_level = "BASIC"
        
        # Calculate realistic speedup
        speedup = 1.0
        
        if self.optimizations.get("orjson", OptimizationInfo("", False, "", "", "", "")).available:
            speedup *= 3.0  # Conservative estimate
        if self.optimizations.get("polars", OptimizationInfo("", False, "", "", "", "")).available:
            speedup *= 2.0  # For copywriting workload
        if self.optimizations.get("uvloop", OptimizationInfo("", False, "", "", "", "")).available:
            speedup *= 2.0
        if self.optimizations.get("redis", OptimizationInfo("", False, "", "", "", "")).available:
            speedup *= 1.5  # Cache benefits
        
        self.total_speedup = min(speedup, 15.0)  # Realistic cap
    
    def get_missing_high_impact(self) -> List[OptimizationInfo]:
        """Get missing high-impact optimizations."""
        return [
            opt for opt in self.optimizations.values()
            if not opt.available and opt.impact == "HIGH"
        ]
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        missing_high_impact = self.get_missing_high_impact()
        return [opt.install_command for opt in missing_high_impact[:5]]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        available = sum(1 for opt in self.optimizations.values() if opt.available)
        total = len(self.optimizations)
        
        return {
            "performance_level": self.performance_level,
            "total_speedup": f"{self.total_speedup:.1f}x",
            "optimizations_available": f"{available}/{total}",
            "high_impact_missing": len(self.get_missing_high_impact()),
            "recommendations": self.get_recommendations()
        }

@lru_cache(maxsize=1)
def get_optimization_detector() -> OptimizationDetector:
    """Get cached optimization detector."""
    detector = OptimizationDetector()
    detector.detect_all()
    return detector

def get_optimization_level() -> str:
    """Get current optimization level."""
    return get_optimization_detector().performance_level

def get_performance_speedup() -> float:
    """Get expected performance speedup."""
    return get_optimization_detector().total_speedup

# Export optimization utilities
__all__ = [
    "OptimizationInfo",
    "OptimizationDetector", 
    "get_optimization_detector",
    "get_optimization_level",
    "get_performance_speedup"
] 