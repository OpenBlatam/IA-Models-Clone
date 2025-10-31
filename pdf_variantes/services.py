"""
PDF Variantes Services
======================

Service layer for orchestrating all PDF processing features.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import asyncio

from .upload import PDFUploadHandler, PDFMetadata
from .editor import PDFEditor, Annotation
from .variant_generator import PDFVariantGenerator, VariantType, VariantOptions
from .topic_extractor import PDFTopicExtractor, Topic
from .brainstorming import PDFBrainstorming, BrainstormIdea
from .advanced_features import PDFVariantesAdvanced, ContentEnhancement
from .ai_enhanced import AIPDFProcessor

logger = logging.getLogger(__name__)


class PDFVariantesService:
    """Orchestrator service for all PDF processing features."""
    
    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Path("./uploads/pdf_variantes")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all components
        self.upload_handler = PDFUploadHandler(upload_dir)
        self.editor = PDFEditor(upload_dir)
        self.variant_generator = PDFVariantGenerator(upload_dir)
        self.topic_extractor = PDFTopicExtractor(upload_dir)
        self.brainstorming = PDFBrainstorming(upload_dir)
        self.advanced = PDFVariantesAdvanced(upload_dir)
        self.ai_processor = AIPDFProcessor(upload_dir)
        
        logger.info("Initialized PDF Variantes Service")
    
    async def process_pdf_completely(
        self,
        file_content: bytes,
        filename: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete PDF processing pipeline.
        
        Args:
            file_content: PDF file content
            filename: Original filename
            options: Processing options
            
        Returns:
            Complete processing results
        """
        options = options or {}
        
        logger.info(f"Starting complete PDF processing for {filename}")
        
        # 1. Upload and extract metadata
        metadata, text_content = await self.upload_handler.upload_pdf(
            file_content=file_content,
            filename=filename,
            auto_process=True,
            extract_text=True,
            detect_language=True
        )
        
        file_id = metadata.file_id
        
        # 2. Extract topics (async)
        topics_task = asyncio.create_task(
            self.topic_extractor.extract_topics(file_id)
        )
        
        # 3. Auto-tag document (async)
        tags_task = asyncio.create_task(
            self.ai_processor.auto_tag_document(file_id)
        )
        
        # 4. Generate intelligent summary (async)
        summary_task = asyncio.create_task(
            self.ai_processor.intelligent_summarization(file_id)
        )
        
        # 5. Extract key insights (async)
        insights_task = asyncio.create_task(
            self.ai_processor.extract_key_insights(file_id)
        )
        
        # Wait for async tasks
        topics, tags, summary, insights = await asyncio.gather(
            topics_task,
            tags_task,
            summary_task,
            insights_task
        )
        
        # 6. Generate brainstorm ideas from topics
        ideas = await self.brainstorming.generate_ideas(
            topics=[t.topic for t in topics[:10]],
            number_of_ideas=20,
            diversity_level=options.get("diversity_level", 0.7)
        )
        
        # Save ideas
        await self.brainstorming.save_ideas(file_id, ideas)
        
        # 7. Generate analytics report
        analytics = await self.advanced.generate_analytics_report(file_id)
        
        results = {
            "file_id": file_id,
            "metadata": metadata.to_dict() if hasattr(metadata, 'to_dict') else {
                "file_id": metadata.file_id,
                "original_filename": metadata.original_filename,
                "file_size": metadata.file_size,
                "page_count": metadata.page_count,
                "word_count": metadata.word_count,
                "language": metadata.language
            },
            "topics": [t.to_dict() if hasattr(t, 'to_dict') else {
                "topic": t.topic,
                "category": t.category,
                "relevance_score": t.relevance_score
            } for t in topics],
            "tags": tags,
            "summary": summary,
            "insights": insights,
            "brainstorm_ideas": [idea.to_dict() if hasattr(idea, 'to_dict') else {
                "idea": idea.idea,
                "category": idea.category
            } for idea in ideas[:10]],
            "analytics": analytics,
            "processing_time": datetime.utcnow().isoformat(),
            "available_variants": [vt.value for vt in VariantType]
        }
        
        logger.info(f"Completed processing for {file_id}")
        
        return results
    
    async def generate_multiple_variants(
        self,
        file_id: str,
        variant_types: List[VariantType],
        options: Optional[VariantOptions] = None
    ) -> Dict[str, Any]:
        """Generate multiple variants of a PDF."""
        logger.info(f"Generating {len(variant_types)} variants for {file_id}")
        
        file_path = self.upload_handler.get_file_path(file_id)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_id}")
        
        variants = {}
        
        with open(file_path, "rb") as f:
            for variant_type in variant_types:
                try:
                    variant = await self.variant_generator.generate(
                        file=f,
                        variant_type=variant_type,
                        options=options
                    )
                    variants[variant_type.value] = variant
                    
                    # Reset file pointer
                    f.seek(0)
                    
                except Exception as e:
                    logger.error(f"Error generating {variant_type.value} variant: {e}")
                    variants[variant_type.value] = {"error": str(e)}
        
        return {
            "file_id": file_id,
            "variants": variants,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def add_intelligent_annotations(
        self,
        file_id: str,
        auto_annotate: bool = True
    ) -> List[Annotation]:
        """Add intelligent annotations to PDF."""
        logger.info(f"Adding intelligent annotations to {file_id}")
        
        if not auto_annotate:
            return []
        
        # Extract topics for context
        topics = await self.topic_extractor.extract_topics(file_id)
        
        # Extract insights
        insights = await self.ai_processor.extract_key_insights(file_id)
        
        annotations = []
        
        # Create annotations based on insights
        for i, insight in enumerate(insights[:5]):
            annotation = await self.editor.add_annotation(
                file_id=file_id,
                page_number=(i % 10) + 1,  # Distribute across pages
                annotation_type="highlight",
                content=insight.get("insight", ""),
                position={
                    "x": 50,
                    "y": 50 + i * 100,
                    "width": 500,
                    "height": 80
                },
                properties={
                    "color": "#FFFF00",
                    "opacity": 0.3
                }
            )
            annotations.append(annotation)
        
        return annotations
    
    async def create_enhanced_summary(
        self,
        file_id: str,
        enhancement_type: ContentEnhancement = ContentEnhancement.SUMMARIZATION
    ) -> Dict[str, Any]:
        """Create enhanced summary with AI."""
        logger.info(f"Creating enhanced summary for {file_id}")
        
        # Get basic summary
        summary = await self.ai_processor.intelligent_summarization(file_id)
        
        # Enhance with AI
        enhanced = await self.advanced.enhance_content(
            text=summary.get("summary", ""),
            enhancement_type=enhancement_type
        )
        
        return {
            "original_summary": summary,
            "enhanced_summary": enhanced.enhanced_text,
            "improvements": enhanced.suggestions,
            "confidence": enhanced.confidence_score
        }
    
    async def batch_process_files(
        self,
        files: List[Dict[str, bytes]],
        process_all_features: bool = True
    ) -> Dict[str, Any]:
        """Batch process multiple PDF files."""
        logger.info(f"Batch processing {len(files)} files")
        
        results = []
        
        for file_data in files:
            filename = file_data.get("filename", "unknown.pdf")
            content = file_data.get("content")
            
            try:
                if process_all_features:
                    result = await self.process_pdf_completely(
                        file_content=content,
                        filename=filename
                    )
                else:
                    # Just upload
                    metadata, _ = await self.upload_handler.upload_pdf(
                        file_content=content,
                        filename=filename
                    )
                    result = {"file_id": metadata.file_id, "metadata": metadata.to_dict()}
                
                results.append({
                    "filename": filename,
                    "status": "success",
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                results.append({
                    "filename": filename,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "total_files": len(files),
            "processed": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results
        }
    
    async def export_complete_analysis(
        self,
        file_id: str,
        format: str = "json"
    ) -> bytes:
        """Export complete analysis of PDF."""
        logger.info(f"Exporting complete analysis for {file_id}")
        
        # Gather all data
        topics = await self.topic_extractor.extract_topics(file_id)
        ideas = await self.brainstorming.load_ideas(file_id)
        analytics = await self.advanced.generate_analytics_report(file_id)
        summary = await self.ai_processor.intelligent_summarization(file_id)
        insights = await self.ai_processor.extract_key_insights(file_id)
        tags = await self.ai_processor.auto_tag_document(file_id)
        
        analysis = {
            "file_id": file_id,
            "exported_at": datetime.utcnow().isoformat(),
            "topics": [t.to_dict() if hasattr(t, 'to_dict') else {} for t in topics],
            "brainstorm_ideas": [i.to_dict() if hasattr(i, 'to_dict') else {} for i in ideas],
            "analytics": analytics,
            "summary": summary,
            "insights": insights,
            "tags": tags
        }
        
        if format == "json":
            import json
            return json.dumps(analysis, indent=2).encode()
        
        return b""  # Other formats can be added