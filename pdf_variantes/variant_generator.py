"""
PDF Variant Generator
====================

Generator for creating variants of PDF documents.
"""

import logging
from typing import Dict, Any, Optional, BinaryIO, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import io
import PyPDF2
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class VariantType(str, Enum):
    """Types of PDF variants."""
    SUMMARY = "summary"
    OUTLINE = "outline"
    HIGHLIGHTS = "highlights"
    NOTES = "notes"
    QUIZ = "quiz"
    PRESENTATION = "presentation"
    TRANSLATED = "translated"
    ABRIDGED = "abridged"
    EXPANDED = "expanded"


@dataclass
class VariantOptions:
    """Options for variant generation."""
    max_length: Optional[int] = None
    style: str = "academic"
    include_images: bool = True
    include_tables: bool = True
    language: str = "en"
    tone: str = "neutral"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_length": self.max_length,
            "style": self.style,
            "include_images": self.include_images,
            "include_tables": self.include_tables,
            "language": self.language,
            "tone": self.tone
        }


class PDFVariantGenerator:
    """Generator for PDF variants."""
    
    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Path("./uploads/pdf_variantes")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized PDF variant generator")
    
    async def generate(
        self,
        file: BinaryIO,
        variant_type: VariantType,
        options: Optional[VariantOptions] = None
    ) -> Dict[str, Any]:
        """Generate a PDF variant."""
        if options is None:
            options = VariantOptions()
        
        # Map variant type to handler
        handlers = {
            VariantType.SUMMARY: self._generate_summary,
            VariantType.OUTLINE: self._generate_outline,
            VariantType.HIGHLIGHTS: self._generate_highlights,
            VariantType.NOTES: self._generate_notes,
            VariantType.QUIZ: self._generate_quiz,
            VariantType.PRESENTATION: self._generate_presentation
        }
        
        handler = handlers.get(variant_type)
        
        if not handler:
            raise ValueError(f"Unknown variant type: {variant_type}")
        
        try:
            variant = await handler(file, options)
            variant["variant_type"] = variant_type.value
            variant["generated_at"] = datetime.utcnow().isoformat()
            variant["options"] = options.to_dict()
            
            logger.info(f"Generated {variant_type.value} variant")
            
            return variant
            
        except Exception as e:
            logger.error(f"Error generating variant: {e}")
            raise
    
    async def _generate_summary(self, file: BinaryIO, options: VariantOptions) -> Dict[str, Any]:
        """Generate summary variant."""
        # Extract text
        text = await self._extract_text(file)
        
        # Generate summary (simplified)
        sentences = text.split('.')
        summary = '. '.join(sentences[:3]) if len(sentences) > 3 else text
        
        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text) if len(text) > 0 else 1.0
        }
    
    async def _generate_outline(self, file: BinaryIO, options: VariantOptions) -> Dict[str, Any]:
        """Generate outline variant."""
        # Use PyPDF2 to get structure
        pdf_reader = PyPDF2.PdfReader(file)
        
        outline = {
            "title": pdf_reader.metadata.get('/Title', 'Untitled') if pdf_reader.metadata else 'Untitled',
            "num_pages": len(pdf_reader.pages),
            "sections": []
        }
        
        # Try to extract headings (simplified)
        for i, page in enumerate(pdf_reader.pages[:5]):  # First 5 pages
            try:
                text = page.extract_text()
                lines = [line.strip() for line in text.split('\n') if line.strip()][:3]
                outline["sections"].append({
                    "page": i + 1,
                    "headings": lines
                })
            except:
                pass
        
        return outline
    
    async def _generate_highlights(self, file: BinaryIO, options: VariantOptions) -> Dict[str, Any]:
        """Generate highlights variant."""
        text = await self._extract_text(file)
        
        # Extract key sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
        highlights = sentences[:10]  # Top 10 highlights
        
        return {
            "highlights": highlights,
            "num_highlights": len(highlights)
        }
    
    async def _generate_notes(self, file: BinaryIO, options: VariantOptions) -> Dict[str, Any]:
        """Generate notes variant."""
        text = await self._extract_text(file)
        
        # Create notes from key concepts
        keywords = await self._extract_keywords(text)
        
        notes = []
        for keyword in keywords[:15]:
            # Find context for keyword
            keyword_occurrences = [i for i, word in enumerate(text.lower().split()) if keyword.lower() in word.lower()]
            
            if keyword_occurrences:
                # Get surrounding context
                idx = keyword_occurrences[0]
                words = text.split()
                context_start = max(0, idx - 10)
                context_end = min(len(words), idx + 10)
                context = ' '.join(words[context_start:context_end])
                
                notes.append({
                    "keyword": keyword,
                    "context": context
                })
        
        return {
            "notes": notes,
            "num_notes": len(notes)
        }
    
    async def _generate_quiz(self, file: BinaryIO, options: VariantOptions) -> Dict[str, Any]:
        """Generate quiz variant."""
        text = await self._extract_text(file)
        
        # Extract sentences for questions
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 40]
        questions = []
        
        for i, sentence in enumerate(sentences[:10]):  # First 10 sentences
            words = sentence.split()
            if len(words) > 5:
                # Create a question from the sentence
                question = f"What is {sentence[:100]}?"
                answer = sentence
                
                questions.append({
                    "question_id": i + 1,
                    "question": question,
                    "answer": answer
                })
        
        return {
            "questions": questions,
            "total_questions": len(questions)
        }
    
    async def _generate_presentation(self, file: BinaryIO, options: VariantOptions) -> Dict[str, Any]:
        """Generate presentation variant."""
        text = await self._extract_text(file)
        
        # Split text into slides
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        slides = []
        
        for i, para in enumerate(paragraphs[:20]):  # Max 20 slides
            # Break into bullet points
            sentences = [s.strip() for s in para.split('.') if len(s.strip()) > 20]
            
            slides.append({
                "slide_number": i + 1,
                "content": para[:500],  # Max 500 chars per slide
                "bullet_points": sentences[:3]  # Max 3 bullets per slide
            })
        
        return {
            "slides": slides,
            "total_slides": len(slides)
        }
    
    async def _extract_text(self, file: BinaryIO) -> str:
        """Extract text from PDF."""
        file.seek(0)
        
        try:
            # Try PyMuPDF first
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text_parts = []
            
            for page in doc:
                text_parts.append(page.get_text())
            
            doc.close()
            return '\n\n'.join(text_parts)
            
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            
            # Fallback to PyPDF2
            try:
                file.seek(0)
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for page in pdf_reader.pages:
                    text_parts.append(page.extract_text())
                
                return '\n\n'.join(text_parts)
                
            except Exception as e2:
                logger.error(f"Text extraction failed: {e2}")
                return ""
    
    async def _extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords from text."""
        import re
        from collections import Counter
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Split into words
        words = text.lower().split()
        
        # Filter out short words
        keywords = [w for w in words if len(w) > 3]
        
        # Count frequency
        keyword_counts = Counter(keywords)
        
        # Get top keywords
        top_keywords = [word for word, _ in keyword_counts.most_common(max_keywords)]
        
        return top_keywords
    
    def get_available_variants(self) -> List[VariantType]:
        """Get list of available variant types."""
        return list(VariantType)