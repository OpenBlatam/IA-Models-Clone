"""Document review and quality assurance"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReviewComment:
    """Review comment"""
    reviewer: str
    comment: str
    line_number: Optional[int] = None
    severity: str = "info"  # info, warning, error
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ReviewResult:
    """Document review result"""
    document_path: str
    reviewer: str
    status: str  # approved, rejected, needs_revision
    comments: List[ReviewComment]
    score: float  # 0-100
    reviewed_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentReviewer:
    """Document review and quality assurance"""
    
    def __init__(self):
        self.reviews: Dict[str, List[ReviewResult]] = {}
        self.quality_checks = []
        self._load_default_checks()
    
    def _load_default_checks(self):
        """Load default quality checks"""
        self.quality_checks = [
            {
                "name": "spelling",
                "check": self._check_spelling,
                "weight": 0.2
            },
            {
                "name": "grammar",
                "check": self._check_grammar,
                "weight": 0.2
            },
            {
                "name": "structure",
                "check": self._check_structure,
                "weight": 0.2
            },
            {
                "name": "formatting",
                "check": self._check_formatting,
                "weight": 0.2
            },
            {
                "name": "completeness",
                "check": self._check_completeness,
                "weight": 0.2
            }
        ]
    
    def review_document(
        self,
        document_path: str,
        reviewer: str,
        parsed_content: Optional[Dict[str, Any]] = None
    ) -> ReviewResult:
        """
        Review a document
        
        Args:
            document_path: Path to document
            reviewer: Reviewer name
            parsed_content: Optional parsed content
            
        Returns:
            Review result
        """
        comments = []
        scores = []
        
        # Run quality checks
        for check in self.quality_checks:
            try:
                check_result = check["check"](document_path, parsed_content)
                if check_result.get("comments"):
                    comments.extend(check_result["comments"])
                if check_result.get("score") is not None:
                    scores.append(check_result["score"] * check["weight"])
            except Exception as e:
                logger.error(f"Error running quality check {check['name']}: {e}")
        
        # Calculate overall score
        overall_score = sum(scores) if scores else 70.0
        
        # Determine status
        if overall_score >= 90:
            status = "approved"
        elif overall_score >= 70:
            status = "needs_revision"
        else:
            status = "rejected"
        
        result = ReviewResult(
            document_path=document_path,
            reviewer=reviewer,
            status=status,
            comments=comments,
            score=overall_score,
            reviewed_at=datetime.now()
        )
        
        # Store review
        if document_path not in self.reviews:
            self.reviews[document_path] = []
        self.reviews[document_path].append(result)
        
        return result
    
    def _check_spelling(
        self,
        document_path: str,
        parsed_content: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check spelling"""
        # Placeholder - would use spell checker
        return {
            "score": 85.0,
            "comments": []
        }
    
    def _check_grammar(
        self,
        document_path: str,
        parsed_content: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check grammar"""
        # Placeholder - would use grammar checker
        return {
            "score": 80.0,
            "comments": []
        }
    
    def _check_structure(
        self,
        document_path: str,
        parsed_content: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check document structure"""
        comments = []
        score = 100.0
        
        if parsed_content:
            headings = parsed_content.get("headings", [])
            if len(headings) < 2:
                comments.append(ReviewComment(
                    reviewer="system",
                    comment="Document should have at least 2 headings for better structure",
                    severity="warning"
                ))
                score -= 10
            
            paragraphs = parsed_content.get("paragraphs", [])
            if len(paragraphs) < 3:
                comments.append(ReviewComment(
                    reviewer="system",
                    comment="Document should have more content paragraphs",
                    severity="info"
                ))
                score -= 5
        
        return {
            "score": max(0, score),
            "comments": comments
        }
    
    def _check_formatting(
        self,
        document_path: str,
        parsed_content: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check formatting"""
        # Placeholder
        return {
            "score": 90.0,
            "comments": []
        }
    
    def _check_completeness(
        self,
        document_path: str,
        parsed_content: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check completeness"""
        comments = []
        score = 100.0
        
        if parsed_content:
            # Check for required elements
            has_title = len(parsed_content.get("headings", [])) > 0
            has_content = len(parsed_content.get("paragraphs", [])) > 0
            
            if not has_title:
                comments.append(ReviewComment(
                    reviewer="system",
                    comment="Document should have a title",
                    severity="warning"
                ))
                score -= 15
            
            if not has_content:
                comments.append(ReviewComment(
                    reviewer="system",
                    comment="Document should have content",
                    severity="error"
                ))
                score -= 30
        
        return {
            "score": max(0, score),
            "comments": comments
        }
    
    def get_reviews(self, document_path: str) -> List[ReviewResult]:
        """Get all reviews for a document"""
        return self.reviews.get(document_path, [])
    
    def get_latest_review(self, document_path: str) -> Optional[ReviewResult]:
        """Get latest review for a document"""
        reviews = self.get_reviews(document_path)
        if reviews:
            return sorted(reviews, key=lambda x: x.reviewed_at, reverse=True)[0]
        return None


# Global reviewer
_reviewer: Optional[DocumentReviewer] = None


def get_document_reviewer() -> DocumentReviewer:
    """Get global document reviewer"""
    global _reviewer
    if _reviewer is None:
        _reviewer = DocumentReviewer()
    return _reviewer

