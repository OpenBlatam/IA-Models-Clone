"""
Statistics calculation utilities for professional documents.
"""

from typing import Dict, Any, List
from collections import Counter
from .models import ProfessionalDocument


def calculate_document_stats(documents: List[ProfessionalDocument]) -> Dict[str, Any]:
    """Calculate statistics from a list of documents."""
    if not documents:
        return {
            "total_documents": 0,
            "documents_by_type": {},
            "total_word_count": 0,
            "average_document_length": 0.0
        }
    
    total_documents = len(documents)
    documents_by_type = Counter(doc.document_type.value for doc in documents)
    total_word_count = sum(doc.word_count for doc in documents)
    average_document_length = total_word_count / total_documents if total_documents > 0 else 0.0
    
    return {
        "total_documents": total_documents,
        "documents_by_type": dict(documents_by_type),
        "total_word_count": total_word_count,
        "average_document_length": average_document_length
    }






