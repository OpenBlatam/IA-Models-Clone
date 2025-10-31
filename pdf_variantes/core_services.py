"""Core service functions with functional approach."""

from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
import logging

from .utils import extract_text_content, detect_language, calculate_readability
from .exceptions import PDFNotFoundError, ProcessingError

logger = logging.getLogger(__name__)


async def process_pdf_completely(
    file_content: bytes,
    filename: str,
    upload_handler,
    variant_generator,
    topic_extractor,
    brainstorming,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Complete PDF processing pipeline."""
    options = options or {}
    
    # Upload and extract metadata
    metadata, text_content = await upload_handler.upload_pdf(
        file_content=file_content,
        filename=filename,
        auto_process=True,
        extract_text=True
    )
    
    file_id = metadata.file_id
    
    # Parallel processing
    topics_task = asyncio.create_task(
        topic_extractor.extract_topics(file_id)
    )
    
    ideas_task = asyncio.create_task(
        brainstorming.generate_ideas(
            topics=["sample"],  # TODO: Extract from topics
            number_of_ideas=20
        )
    )
    
    topics, ideas = await asyncio.gather(topics_task, ideas_task)
    
    return {
        "file_id": file_id,
        "metadata": metadata.to_dict() if hasattr(metadata, 'to_dict') else {},
        "topics": [t.to_dict() if hasattr(t, 'to_dict') else t for t in topics],
        "ideas": [i.to_dict() if hasattr(i, 'to_dict') else i for i in ideas],
        "processing_time": "0.0"  # TODO: Implement timing
    }


async def generate_multiple_variants(
    file_id: str,
    variant_types: List[str],
    variant_generator,
    upload_handler
) -> Dict[str, Any]:
    """Generate multiple variants of a PDF."""
    file_path = upload_handler.get_file_path(file_id)
    
    if not file_path.exists():
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    variants = {}
    
    with open(file_path, "rb") as f:
        for variant_type in variant_types:
            try:
                variant = await variant_generator.generate(
                    file=f,
                    variant_type=variant_type
                )
                variants[variant_type] = variant
                f.seek(0)  # Reset file pointer
            except Exception as e:
                logger.error(f"Error generating {variant_type} variant: {e}")
                variants[variant_type] = {"error": str(e)}
    
    return {
        "file_id": file_id,
        "variants": variants
    }


async def extract_document_insights(
    file_id: str,
    topic_extractor,
    ai_processor
) -> Dict[str, Any]:
    """Extract comprehensive document insights."""
    try:
        # Extract topics
        topics = await topic_extractor.extract_topics(file_id)
        
        # Extract insights using AI
        insights = await ai_processor.extract_key_insights(file_id)
        
        return {
            "file_id": file_id,
            "topics": [t.to_dict() if hasattr(t, 'to_dict') else t for t in topics],
            "insights": insights,
            "extracted_at": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
    except Exception as e:
        logger.error(f"Error extracting insights: {e}")
        raise ProcessingError(f"Failed to extract insights: {e}")


def validate_processing_request(
    file_id: str,
    upload_handler,
    limits: Dict[str, int]
) -> bool:
    """Validate processing request against limits."""
    file_path = upload_handler.get_file_path(file_id)
    
    if not file_path.exists():
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    # TODO: Implement more sophisticated limit checking
    return True


async def batch_process_files(
    files: List[Dict[str, bytes]],
    process_all_features: bool = True,
    upload_handler=None,
    variant_generator=None,
    topic_extractor=None,
    brainstorming=None
) -> Dict[str, Any]:
    """Batch process multiple PDF files."""
    results = []
    
    for file_data in files:
        filename = file_data.get("filename", "unknown.pdf")
        content = file_data.get("content")
        
        try:
            if process_all_features:
                result = await process_pdf_completely(
                    content, filename, upload_handler,
                    variant_generator, topic_extractor, brainstorming
                )
            else:
                metadata, _ = await upload_handler.upload_pdf(
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
