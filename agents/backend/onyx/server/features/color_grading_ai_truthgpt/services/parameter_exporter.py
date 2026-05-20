"""
Parameter Exporter for Color Grading AI
=======================================

Exports color grading parameters to various formats.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ColorParameters:
    """Color grading parameters."""
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    color_balance: Dict[str, float] = None
    curves: Optional[Dict[str, Any]] = None
    lut: Optional[str] = None
    
    def __post_init__(self):
        if self.color_balance is None:
            self.color_balance = {"r": 0.0, "g": 0.0, "b": 0.0}


class ParameterExporter:
    """
    Exports color parameters to various formats.
    
    Formats:
    - JSON
    - DaVinci Resolve (.drx)
    - Premiere Pro (.prfpset)
    - Final Cut Pro XML
    - LUT files
    """
    
    def __init__(self):
        """Initialize parameter exporter."""
        pass
    
    def export_json(
        self,
        params: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Export to JSON.
        
        Args:
            params: Color parameters
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        with open(output_path, "w") as f:
            json.dump(params, f, indent=2)
        
        return output_path
    
    def export_davinci_resolve(
        self,
        params: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Export to DaVinci Resolve format.
        
        Args:
            params: Color parameters
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        # Simplified DaVinci Resolve XML format
        drx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<DRT>
    <ColorCorrection>
        <Brightness>{params.get('brightness', 0.0)}</Brightness>
        <Contrast>{params.get('contrast', 1.0)}</Contrast>
        <Saturation>{params.get('saturation', 1.0)}</Saturation>
        <ColorBalance>
            <Red>{params.get('color_balance', {}).get('r', 0.0)}</Red>
            <Green>{params.get('color_balance', {}).get('g', 0.0)}</Green>
            <Blue>{params.get('color_balance', {}).get('b', 0.0)}</Blue>
        </ColorBalance>
    </ColorCorrection>
</DRT>"""
        
        with open(output_path, "w") as f:
            f.write(drx_content)
        
        return output_path
    
    def export_premiere_pro(
        self,
        params: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Export to Premiere Pro format.
        
        Args:
            params: Color parameters
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        # Premiere Pro preset format (simplified)
        preset = {
            "version": "1.0",
            "name": "Color Grading Preset",
            "parameters": {
                "brightness": params.get("brightness", 0.0),
                "contrast": params.get("contrast", 1.0),
                "saturation": params.get("saturation", 1.0),
                "colorBalance": params.get("color_balance", {}),
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(preset, f, indent=2)
        
        return output_path
    
    def export_lut_cube(
        self,
        params: Dict[str, Any],
        output_path: str,
        size: int = 33
    ) -> str:
        """
        Export to LUT cube format.
        
        Args:
            params: Color parameters
            output_path: Output file path
            size: LUT size (default 33)
            
        Returns:
            Path to exported file
        """
        # Generate LUT cube file
        lines = [
            f"TITLE Color Grading LUT",
            f"LUT_3D_SIZE {size}",
            ""
        ]
        
        # Generate LUT entries (simplified - would need proper color space transformation)
        import numpy as np
        
        for b in np.linspace(0, 1, size):
            for g in np.linspace(0, 1, size):
                for r in np.linspace(0, 1, size):
                    # Apply color grading
                    r_out = r * params.get("contrast", 1.0) + params.get("brightness", 0.0)
                    g_out = g * params.get("contrast", 1.0) + params.get("brightness", 0.0)
                    b_out = b * params.get("contrast", 1.0) + params.get("brightness", 0.0)
                    
                    # Apply color balance
                    color_balance = params.get("color_balance", {})
                    r_out += color_balance.get("r", 0.0)
                    g_out += color_balance.get("g", 0.0)
                    b_out += color_balance.get("b", 0.0)
                    
                    # Clamp
                    r_out = max(0, min(1, r_out))
                    g_out = max(0, min(1, g_out))
                    b_out = max(0, min(1, b_out))
                    
                    lines.append(f"{r_out:.6f} {g_out:.6f} {b_out:.6f}")
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        
        return output_path
    
    def export_all_formats(
        self,
        params: Dict[str, Any],
        base_path: str
    ) -> Dict[str, str]:
        """
        Export to all formats.
        
        Args:
            params: Color parameters
            base_path: Base path for exports
            
        Returns:
            Dictionary with format -> path mapping
        """
        base = Path(base_path).stem
        output_dir = Path(base_path).parent
        
        exports = {}
        
        # JSON
        exports["json"] = self.export_json(
            params,
            str(output_dir / f"{base}.json")
        )
        
        # DaVinci Resolve
        exports["drx"] = self.export_davinci_resolve(
            params,
            str(output_dir / f"{base}.drx")
        )
        
        # Premiere Pro
        exports["prfpset"] = self.export_premiere_pro(
            params,
            str(output_dir / f"{base}.prfpset")
        )
        
        # LUT
        exports["cube"] = self.export_lut_cube(
            params,
            str(output_dir / f"{base}.cube")
        )
        
        return exports




