"""
Model Documentation Generator
Generate model architecture documentation
"""

import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelDocGenerator:
    """
    Generate model documentation
    """
    
    @staticmethod
    def generate_model_doc(model: nn.Module, output_path: Path) -> None:
        """
        Generate documentation for model
        
        Args:
            model: Model to document
            output_path: Output file path
        """
        doc_lines = [f"# {model.__class__.__name__}", ""]
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        doc_lines.append("## Model Information")
        doc_lines.append("")
        doc_lines.append(f"- **Total Parameters**: {total_params:,}")
        doc_lines.append(f"- **Trainable Parameters**: {trainable_params:,}")
        doc_lines.append(f"- **Non-trainable Parameters**: {total_params - trainable_params:,}")
        doc_lines.append("")
        
        # Architecture
        doc_lines.append("## Architecture")
        doc_lines.append("")
        doc_lines.append("```")
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf node
                params = sum(p.numel() for p in module.parameters())
                doc_lines.append(f"{name}: {module.__class__.__name__} ({params:,} params)")
        
        doc_lines.append("```")
        doc_lines.append("")
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(doc_lines))
        logger.info(f"Generated model docs: {output_path}")



