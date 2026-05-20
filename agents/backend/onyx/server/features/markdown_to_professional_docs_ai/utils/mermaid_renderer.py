"""Mermaid Diagram Renderer - Render Mermaid diagrams to images"""
import re
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MermaidRenderer:
    """Render Mermaid diagrams to PNG/SVG images"""
    
    def __init__(self):
        self.mermaid_cli_available = self._check_mermaid_cli()
    
    def _check_mermaid_cli(self) -> bool:
        """Check if Mermaid CLI is available"""
        try:
            result = subprocess.run(
                ['mmdc', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def render_diagram(
        self,
        mermaid_code: str,
        output_format: str = "png",
        theme: str = "default"
    ) -> Optional[bytes]:
        """
        Render Mermaid diagram to image
        
        Args:
            mermaid_code: Mermaid diagram code
            output_format: Output format (png, svg)
            theme: Mermaid theme (default, dark, forest, neutral)
            
        Returns:
            Image bytes or None if rendering fails
        """
        if not self.mermaid_cli_available:
            logger.warning("Mermaid CLI not available. Install with: npm install -g @mermaid-js/mermaid-cli")
            return None
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write Mermaid code to file
                input_file = Path(temp_dir) / "diagram.mmd"
                input_file.write_text(mermaid_code, encoding='utf-8')
                
                # Output file
                output_file = Path(temp_dir) / f"diagram.{output_format}"
                
                # Build command
                cmd = [
                    'mmdc',
                    '-i', str(input_file),
                    '-o', str(output_file),
                    '-t', theme,
                    '-b', 'white'
                ]
                
                # Execute
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=30
                )
                
                if result.returncode == 0 and output_file.exists():
                    return output_file.read_bytes()
                else:
                    logger.error(f"Mermaid rendering failed: {result.stderr.decode()}")
                    return None
        except Exception as e:
            logger.error(f"Error rendering Mermaid diagram: {e}")
            return None
    
    def extract_mermaid_blocks(self, markdown_content: str) -> list:
        """
        Extract Mermaid code blocks from Markdown
        
        Args:
            markdown_content: Markdown content
            
        Returns:
            List of Mermaid code blocks
        """
        mermaid_blocks = []
        pattern = r'```mermaid\n(.*?)```'
        
        for match in re.finditer(pattern, markdown_content, re.DOTALL):
            mermaid_code = match.group(1).strip()
            mermaid_blocks.append({
                "code": mermaid_code,
                "type": self._detect_diagram_type(mermaid_code)
            })
        
        return mermaid_blocks
    
    def _detect_diagram_type(self, code: str) -> str:
        """Detect Mermaid diagram type"""
        code_lower = code.lower()
        
        if code_lower.startswith('graph') or code_lower.startswith('flowchart'):
            return "flowchart"
        elif code_lower.startswith('sequence'):
            return "sequence"
        elif code_lower.startswith('class'):
            return "class"
        elif code_lower.startswith('state'):
            return "state"
        elif code_lower.startswith('er'):
            return "er"
        elif code_lower.startswith('gantt'):
            return "gantt"
        elif code_lower.startswith('pie'):
            return "pie"
        else:
            return "unknown"

