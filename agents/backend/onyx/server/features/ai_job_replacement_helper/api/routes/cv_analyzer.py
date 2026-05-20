"""
CV Analyzer endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.cv_analyzer import CVAnalyzerService

router = APIRouter()
cv_analyzer = CVAnalyzerService()


@router.post("/analyze/{user_id}")
async def analyze_cv(
    user_id: str,
    cv_content: Dict[str, Any],
    target_job: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Analizar un CV"""
    try:
        analysis = cv_analyzer.analyze_cv(user_id, cv_content, target_job)
        return {
            "cv_id": analysis.cv_id,
            "overall_score": analysis.overall_score,
            "ats_score": analysis.ats_score,
            "feedback_by_section": {
                section: {
                    "score": feedback.score,
                    "strengths": feedback.strengths,
                    "weaknesses": feedback.weaknesses,
                    "suggestions": feedback.suggestions,
                    "priority": feedback.priority,
                }
                for section, feedback in analysis.feedback_by_section.items()
            },
            "missing_elements": analysis.missing_elements,
            "keyword_analysis": analysis.keyword_analysis,
            "suggestions": analysis.suggestions,
            "analyzed_at": analysis.analyzed_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




