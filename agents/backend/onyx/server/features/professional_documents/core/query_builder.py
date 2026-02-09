"""
Query builder utilities for filtering and searching documents.

Helper classes for building complex queries and filters.
"""

from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from .models import ProfessionalDocument, DocumentType


class DocumentQueryBuilder:
    """Builder for constructing document queries with filters."""
    
    def __init__(self):
        self._filters: List[Callable[[ProfessionalDocument], bool]] = []
        self._sort_key: Optional[Callable[[ProfessionalDocument], Any]] = None
        self._reverse: bool = False
        self._limit: Optional[int] = None
        self._offset: int = 0
    
    def filter_by_type(self, document_type: DocumentType) -> "DocumentQueryBuilder":
        """Filter documents by type."""
        self._filters.append(lambda doc: doc.document_type == document_type)
        return self
    
    def filter_by_author(self, author: str) -> "DocumentQueryBuilder":
        """Filter documents by author."""
        self._filters.append(lambda doc: doc.author == author)
        return self
    
    def filter_by_company(self, company: str) -> "DocumentQueryBuilder":
        """Filter documents by company."""
        self._filters.append(lambda doc: doc.company == company)
        return self
    
    def filter_by_status(self, status: str) -> "DocumentQueryBuilder":
        """Filter documents by status."""
        self._filters.append(lambda doc: doc.status == status)
        return self
    
    def filter_by_date_range(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> "DocumentQueryBuilder":
        """Filter documents by date range."""
        if start_date:
            self._filters.append(lambda doc: doc.date_created >= start_date)
        if end_date:
            self._filters.append(lambda doc: doc.date_created <= end_date)
        return self
    
    def filter_by_word_count(
        self,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None
    ) -> "DocumentQueryBuilder":
        """Filter documents by word count range."""
        if min_words is not None:
            self._filters.append(lambda doc: doc.word_count >= min_words)
        if max_words is not None:
            self._filters.append(lambda doc: doc.word_count <= max_words)
        return self
    
    def search_in_title(self, search_term: str) -> "DocumentQueryBuilder":
        """Search for term in document title."""
        term_lower = search_term.lower()
        self._filters.append(lambda doc: term_lower in doc.title.lower())
        return self
    
    def search_in_content(self, search_term: str) -> "DocumentQueryBuilder":
        """Search for term in document content."""
        term_lower = search_term.lower()
        self._filters.append(
            lambda doc: any(
                term_lower in section.content.lower()
                for section in doc.sections
            )
        )
        return self
    
    def sort_by_date(self, reverse: bool = True) -> "DocumentQueryBuilder":
        """Sort by creation date."""
        self._sort_key = lambda doc: doc.date_created
        self._reverse = reverse
        return self
    
    def sort_by_word_count(self, reverse: bool = False) -> "DocumentQueryBuilder":
        """Sort by word count."""
        self._sort_key = lambda doc: doc.word_count
        self._reverse = reverse
        return self
    
    def sort_by_title(self, reverse: bool = False) -> "DocumentQueryBuilder":
        """Sort by title."""
        self._sort_key = lambda doc: doc.title.lower()
        self._reverse = reverse
        return self
    
    def limit(self, limit: int) -> "DocumentQueryBuilder":
        """Limit number of results."""
        self._limit = limit
        return self
    
    def offset(self, offset: int) -> "DocumentQueryBuilder":
        """Set offset for pagination."""
        self._offset = offset
        return self
    
    def apply(self, documents: List[ProfessionalDocument]) -> List[ProfessionalDocument]:
        """
        Apply all filters and sorting to a list of documents.
        
        Args:
            documents: List of documents to filter
            
        Returns:
            Filtered and sorted list of documents
        """
        # Apply filters
        for filter_func in self._filters:
            documents = [doc for doc in documents if filter_func(doc)]
        
        # Apply sorting
        if self._sort_key:
            documents = sorted(documents, key=self._sort_key, reverse=self._reverse)
        
        # Apply pagination
        if self._offset > 0:
            documents = documents[self._offset:]
        if self._limit:
            documents = documents[:self._limit]
        
        return documents
    
    def count(self, documents: List[ProfessionalDocument]) -> int:
        """
        Count documents matching the filters.
        
        Args:
            documents: List of documents to count
            
        Returns:
            Number of matching documents
        """
        filtered = documents
        for filter_func in self._filters:
            filtered = [doc for doc in filtered if filter_func(doc)]
        return len(filtered)


def create_query_builder() -> DocumentQueryBuilder:
    """Create a new DocumentQueryBuilder instance."""
    return DocumentQueryBuilder()






