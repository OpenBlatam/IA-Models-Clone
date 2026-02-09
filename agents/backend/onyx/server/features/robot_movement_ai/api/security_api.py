"""
Security API Endpoints
======================

Endpoints para seguridad y cumplimiento.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from ..core.security_audit import get_security_auditor
from ..core.compliance_checker import get_compliance_checker, ComplianceStandard

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/security", tags=["security"])


@router.post("/audit/file")
async def audit_file(
    file_path: str
) -> Dict[str, Any]:
    """Auditar archivo en busca de problemas de seguridad."""
    try:
        auditor = get_security_auditor()
        issues = auditor.audit_file(file_path)
        return {
            "file_path": file_path,
            "issues_count": len(issues),
            "issues": [
                {
                    "issue_id": issue.issue_id,
                    "severity": issue.severity,
                    "category": issue.category,
                    "description": issue.description,
                    "recommendation": issue.recommendation,
                    "line_number": issue.line_number
                }
                for issue in issues
            ]
        }
    except Exception as e:
        logger.error(f"Error auditing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit/directory")
async def audit_directory(
    directory: str,
    pattern: str = Query("*.py", description="File pattern")
) -> Dict[str, Any]:
    """Auditar directorio en busca de problemas de seguridad."""
    try:
        auditor = get_security_auditor()
        issues = auditor.audit_directory(directory, pattern=pattern)
        report = auditor.get_audit_report()
        return report
    except Exception as e:
        logger.error(f"Error auditing directory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/report")
async def get_audit_report() -> Dict[str, Any]:
    """Obtener reporte de auditoría de seguridad."""
    try:
        auditor = get_security_auditor()
        report = auditor.get_audit_report()
        return report
    except Exception as e:
        logger.error(f"Error getting audit report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compliance/check")
async def check_compliance(
    file_path: Optional[str] = None,
    directory: Optional[str] = None,
    standard: Optional[str] = None
) -> Dict[str, Any]:
    """Verificar cumplimiento de estándares."""
    try:
        checker = get_compliance_checker()
        
        compliance_standard = None
        if standard:
            try:
                compliance_standard = ComplianceStandard(standard)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid standard: {standard}"
                )
        
        if file_path:
            results = checker.check_file(file_path, standard=compliance_standard)
            return {
                "file_path": file_path,
                "results": [
                    {
                        "rule_id": r.rule_id,
                        "passed": r.passed,
                        "message": r.message
                    }
                    for r in results
                ]
            }
        elif directory:
            summary = checker.check_directory(directory, standard=compliance_standard)
            return summary
        else:
            raise HTTPException(
                status_code=400,
                detail="Either file_path or directory must be provided"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking compliance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance/summary")
async def get_compliance_summary() -> Dict[str, Any]:
    """Obtener resumen de cumplimiento."""
    try:
        checker = get_compliance_checker()
        summary = checker.get_compliance_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting compliance summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






