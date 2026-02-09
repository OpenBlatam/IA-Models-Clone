"""
Export utilities for separation results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..logger import logger


def export_separation_metadata(
    results: Dict[str, str],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Export separation results metadata to JSON.
    
    Args:
        results: Dictionary of separated sources
        output_path: Path to save metadata
        metadata: Additional metadata to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "sources": results,
        "num_sources": len(results),
        "metadata": metadata or {}
    }
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Exported metadata to {output_path}")


def export_separation_report(
    results: Dict[str, str],
    analysis: Optional[Dict[str, Any]] = None,
    quality: Optional[Dict[str, Any]] = None,
    output_path: str = "separation_report.json"
):
    """
    Export comprehensive separation report.
    
    Args:
        results: Dictionary of separated sources
        analysis: Audio analysis results
        quality: Quality metrics
        output_path: Path to save report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "separation": {
            "sources": results,
            "num_sources": len(results)
        },
        "analysis": analysis or {},
        "quality": quality or {}
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Exported separation report to {output_path}")


def import_separation_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Import separation metadata from JSON.
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Dictionary with metadata
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def create_separation_summary(
    results: Dict[str, str],
    output_path: str = "separation_summary.txt"
):
    """
    Create a text summary of separation results.
    
    Args:
        results: Dictionary of separated sources
        output_path: Path to save summary
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Audio Separation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Number of Sources: {len(results)}\n\n")
        f.write("Separated Sources:\n")
        f.write("-" * 50 + "\n")
        
        for source_name, source_path in results.items():
            file_size = Path(source_path).stat().st_size if Path(source_path).exists() else 0
            f.write(f"  {source_name}:\n")
            f.write(f"    Path: {source_path}\n")
            f.write(f"    Size: {file_size / 1024 / 1024:.2f} MB\n\n")
    
    logger.info(f"Created separation summary at {output_path}")

