"""
PDF Editor
==========

Editor for modifying PDF documents.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import fitz  # PyMuPDF
from pathlib import Path

logger = logging.getLogger(__name__)


class AnnotationType(str, Enum):
    """Types of annotations."""
    HIGHLIGHT = "highlight"
    TEXT = "text"
    NOTE = "note"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    LINE = "line"
    FREETEXT = "freetext"
    INK = "ink"


class Annotation:
    """PDF annotation."""
    
    def __init__(
        self,
        annotation_id: str,
        page_number: int,
        annotation_type: AnnotationType,
        content: str,
        position: Dict[str, float],
        properties: Optional[Dict[str, Any]] = None
    ):
        self.id = annotation_id
        self.page_number = page_number
        self.type = annotation_type
        self.content = content
        self.position = position
        self.properties = properties or {}
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "page_number": self.page_number,
            "type": self.type.value,
            "content": self.content,
            "position": self.position,
            "properties": self.properties,
            "created_at": self.created_at.isoformat()
        }


class PDFEditor:
    """PDF document editor."""
    
    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Path("./uploads/pdf_variantes")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.annotations: Dict[str, List[Annotation]] = {}
        logger.info("Initialized PDF editor")
    
    async def edit_page(
        self,
        file_id: str,
        page_number: int,
        new_content: str,
        preserve_formatting: bool = True
    ) -> Annotation:
        """
        Edit a page in the PDF.
        
        Args:
            file_id: File ID of the PDF
            page_number: Page number to edit
            new_content: New content for the page
            preserve_formatting: Preserve original formatting
            
        Returns:
            Annotation object
        """
        file_path = self.upload_dir / f"{file_id}.pdf"
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_id}")
        
        try:
            doc = fitz.open(str(file_path))
            page = doc[page_number - 1]
            
            # Create annotation for the edit
        annotation = Annotation(
                annotation_id=f"{file_id}_edit_{page_number}",
                page_number=page_number,
                annotation_type=AnnotationType.FREETEXT,
                content=new_content,
                position={"x": 50, "y": 50, "width": 500, "height": 400},
                properties={"font": "helv", "size": 12}
            )
            
            # Add annotation to PDF
            rect = fitz.Rect(
                annotation.position["x"],
                annotation.position["y"],
                annotation.position["x"] + annotation.position["width"],
                annotation.position["y"] + annotation.position["height"]
            )
            
            page.insert_freetext(
                rect,
                new_content,
                fontsize=12,
                color=(0, 0, 0),
                fill=(1, 1, 0)
            )
            
            # Save modifications
            doc.saveIncr()
            doc.close()
            
            # Store annotation
            if file_id not in self.annotations:
                self.annotations[file_id] = []
            self.annotations[file_id].append(annotation)
            
            logger.info(f"Edited page {page_number} of {file_id}")
            
            return annotation
            
        except Exception as e:
            logger.error(f"Error editing page: {e}")
            raise
    
    async def add_annotation(
        self,
        file_id: str,
        page_number: int,
        annotation_type: AnnotationType,
        content: str,
        position: Dict[str, float],
        properties: Optional[Dict[str, Any]] = None
    ) -> Annotation:
        """Add annotation to PDF."""
        file_path = self.upload_dir / f"{file_id}.pdf"
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_id}")
        
        try:
            doc = fitz.open(str(file_path))
            page = doc[page_number - 1]
            
            annotation = Annotation(
                annotation_id=f"{file_id}_annot_{len(self.annotations.get(file_id, []))}",
                page_number=page_number,
                annotation_type=annotation_type,
                content=content,
                position=position,
                properties=properties
            )
            
            rect = fitz.Rect(
                position.get("x", 0),
                position.get("y", 0),
                position.get("x", 0) + position.get("width", 100),
                position.get("y", 0) + position.get("height", 100)
            )
            
            # Add annotation based on type
            if annotation_type == AnnotationType.HIGHLIGHT:
                page.add_highlight_annot(rect)
            elif annotation_type == AnnotationType.TEXT:
                page.insert_text(
                    (position.get("x", 0), position.get("y", 0)),
                    content,
                    fontsize=properties.get("fontsize", 12)
                )
            elif annotation_type == AnnotationType.FREETEXT:
                page.insert_freetext(
                    rect,
                    content,
                    fontsize=properties.get("fontsize", 12),
                    color=properties.get("color", (0, 0, 0)),
                    fill=properties.get("fill", (1, 1, 0))
                )
            
            doc.saveIncr()
            doc.close()
            
            # Store annotation
            if file_id not in self.annotations:
                self.annotations[file_id] = []
            self.annotations[file_id].append(annotation)
            
            logger.info(f"Added {annotation_type.value} annotation to page {page_number}")
            
            return annotation
            
        except Exception as e:
            logger.error(f"Error adding annotation: {e}")
            raise
    
    async def get_annotations(self, file_id: str) -> List[Annotation]:
        """Get all annotations for a file."""
        return self.annotations.get(file_id, [])
    
    async def remove_annotation(self, file_id: str, annotation_id: str) -> bool:
        """Remove an annotation."""
        if file_id in self.annotations:
            self.annotations[file_id] = [
                annot for annot in self.annotations[file_id]
                if annot.id != annotation_id
            ]
            return True
        return False
    
    async def replace_text(
        self,
        file_id: str,
        page_number: int,
        old_text: str,
        new_text: str
    ) -> bool:
        """Replace text in PDF."""
        file_path = self.upload_dir / f"{file_id}.pdf"
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_id}")
        
        try:
            doc = fitz.open(str(file_path))
            page = doc[page_number - 1]
            
            # Search for text
            text_instances = page.search_for(old_text)
            
            if not text_instances:
                doc.close()
                return False
            
            # Replace text
            for inst in text_instances:
                page.add_redact_annot(inst)
            
            page.apply_redactions()
            
            # Add new text
            for inst in text_instances:
                page.insert_text(
                    inst.tl,
                    new_text,
                    fontsize=11
                )
            
            doc.saveIncr()
            doc.close()
            
            logger.info(f"Replaced '{old_text}' with '{new_text}' on page {page_number}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error replacing text: {e}")
            return False
    
    async def copy_page(
        self,
        file_id: str,
        source_page: int,
        target_page: int
    ) -> bool:
        """Copy a page within the PDF."""
        file_path = self.upload_dir / f"{file_id}.pdf"
        
        try:
            doc = fitz.open(str(file_path))
            
            # Get source page
            src_page = doc[source_page - 1]
            
            # Insert after target page
            doc.new_page(target_page)
            doc[target_page].show_pdf_page(
                doc[target_page].rect,
                doc,
                source_page - 1
            )
            
            doc.saveIncr()
            doc.close()
            
            logger.info(f"Copied page {source_page} to {target_page}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error copying page: {e}")
            return False
    
    async def reorder_pages(self, file_id: str, new_order: List[int]) -> bool:
        """Reorder pages in PDF."""
        file_path = self.upload_dir / f"{file_id}.pdf"
        
        try:
            doc = fitz.open(str(file_path))
            
            if len(new_order) != len(doc):
                doc.close()
                return False
            
            # Create new PDF with reordered pages
            doc.select(new_order)
            
            # Save
            output_path = self.upload_dir / f"{file_id}_reordered.pdf"
            doc.save(str(output_path))
            doc.close()
            
            # Replace original
            file_path.unlink()
            output_path.rename(file_path)
            
            logger.info(f"Reordered pages: {new_order}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error reordering pages: {e}")
            return False
