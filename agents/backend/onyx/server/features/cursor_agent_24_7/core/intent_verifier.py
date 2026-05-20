"""
Intent Verifier
===============

Sistema que verifica que se cumplió la intención del usuario,
no solo los requisitos técnicos, siguiendo las mejores prácticas de Devin.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class IntentCheck:
    """Verificación de intención"""
    aspect: str
    description: str
    verified: bool = False
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "aspect": self.aspect,
            "description": self.description,
            "verified": self.verified,
            "confidence": self.confidence,
            "evidence": self.evidence
        }


@dataclass
class IntentVerification:
    """Verificación de intención completa"""
    task_id: str
    original_request: str
    checks: List[IntentCheck] = field(default_factory=list)
    overall_verified: bool = False
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_check(
        self,
        aspect: str,
        description: str,
        verified: bool,
        confidence: float = 0.0,
        evidence: Optional[List[str]] = None
    ) -> IntentCheck:
        """Agregar verificación de intención"""
        check = IntentCheck(
            aspect=aspect,
            description=description,
            verified=verified,
            confidence=confidence,
            evidence=evidence or []
        )
        self.checks.append(check)
        return check
    
    def evaluate(self) -> None:
        """Evaluar todas las verificaciones de intención"""
        if not self.checks:
            self.overall_verified = False
            self.confidence_score = 0.0
            return
        
        verified_count = sum(1 for c in self.checks if c.verified)
        total_confidence = sum(c.confidence for c in self.checks if c.verified)
        
        self.overall_verified = verified_count == len(self.checks)
        self.confidence_score = total_confidence / len(self.checks) if self.checks else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "task_id": self.task_id,
            "original_request": self.original_request,
            "checks": [c.to_dict() for c in self.checks],
            "overall_verified": self.overall_verified,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat()
        }


class IntentVerifier:
    """
    Verificador de intención del usuario.
    
    Verifica que se cumplió la intención del usuario, no solo
    los requisitos técnicos, siguiendo las mejores prácticas de Devin.
    """
    
    def __init__(self) -> None:
        """Inicializar verificador de intención"""
        self.verifications: Dict[str, IntentVerification] = {}
        logger.info("🎯 Intent verifier initialized")
    
    async def verify_user_intent(
        self,
        task_id: str,
        original_request: str,
        task_description: str,
        changes_made: Optional[List[Dict[str, Any]]] = None,
        agent: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Verificar que se cumplió la intención del usuario.
        
        Args:
            task_id: ID de la tarea.
            original_request: Solicitud original del usuario.
            task_description: Descripción de la tarea.
            changes_made: Lista de cambios realizados (opcional).
            agent: Instancia del agente (opcional).
        
        Returns:
            Resultado de la verificación de intención.
        """
        verification = IntentVerification(
            task_id=task_id,
            original_request=original_request
        )
        
        # Verificación 1: La descripción de la tarea coincide con la solicitud
        description_match = self._check_description_match(
            original_request,
            task_description
        )
        verification.add_check(
            "description_match",
            "Task description matches user request",
            description_match["match"],
            confidence=description_match["confidence"],
            evidence=description_match.get("evidence", [])
        )
        
        # Verificación 2: Se realizaron cambios relevantes
        if changes_made:
            relevant_changes = self._check_relevant_changes(
                original_request,
                changes_made
            )
            verification.add_check(
                "relevant_changes",
                "Relevant changes were made",
                relevant_changes["relevant"],
                confidence=relevant_changes["confidence"],
                evidence=relevant_changes.get("evidence", [])
            )
        else:
            verification.add_check(
                "relevant_changes",
                "Relevant changes were made",
                False,
                confidence=0.0,
                evidence=["No changes recorded"]
            )
        
        # Verificación 3: Todos los archivos mencionados fueron modificados
        mentioned_files = self._extract_mentioned_files(original_request)
        if mentioned_files:
            files_modified = self._check_files_modified(
                mentioned_files,
                changes_made or []
            )
            verification.add_check(
                "files_modified",
                "All mentioned files were modified",
                files_modified["all_modified"],
                confidence=files_modified["confidence"],
                evidence=files_modified.get("evidence", [])
            )
        
        # Verificación 4: La funcionalidad solicitada está implementada
        functionality_check = self._check_functionality_implemented(
            original_request,
            changes_made or []
        )
        verification.add_check(
            "functionality_implemented",
            "Requested functionality is implemented",
            functionality_check["implemented"],
            confidence=functionality_check["confidence"],
            evidence=functionality_check.get("evidence", [])
        )
        
        # Evaluar todas las verificaciones
        verification.evaluate()
        self.verifications[task_id] = verification
        
        return {
            "success": verification.overall_verified,
            "confidence": verification.confidence_score,
            "checks": [c.to_dict() for c in verification.checks],
            "can_report": verification.overall_verified and verification.confidence_score >= 0.7
        }
    
    def _check_description_match(
        self,
        original_request: str,
        task_description: str
    ) -> Dict[str, Any]:
        """Verificar que la descripción coincide con la solicitud"""
        original_lower = original_request.lower()
        description_lower = task_description.lower()
        
        keywords = original_lower.split()
        matches = sum(1 for kw in keywords if kw in description_lower)
        
        match_ratio = matches / len(keywords) if keywords else 0.0
        match = match_ratio >= 0.5
        
        return {
            "match": match,
            "confidence": match_ratio,
            "evidence": [
                f"Keyword match ratio: {match_ratio:.2f}",
                f"Matched {matches}/{len(keywords)} keywords"
            ]
        }
    
    def _check_relevant_changes(
        self,
        original_request: str,
        changes_made: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verificar que los cambios son relevantes"""
        if not changes_made:
            return {
                "relevant": False,
                "confidence": 0.0,
                "evidence": ["No changes made"]
            }
        
        original_lower = original_request.lower()
        relevant_count = 0
        
        for change in changes_made:
            change_desc = str(change.get("description", "")).lower()
            change_file = str(change.get("file_path", "")).lower()
            
            if any(kw in change_desc or kw in change_file 
                   for kw in original_lower.split() if len(kw) > 3):
                relevant_count += 1
        
        relevance_ratio = relevant_count / len(changes_made) if changes_made else 0.0
        relevant = relevance_ratio >= 0.6
        
        return {
            "relevant": relevant,
            "confidence": relevance_ratio,
            "evidence": [
                f"Relevant changes: {relevant_count}/{len(changes_made)}",
                f"Relevance ratio: {relevance_ratio:.2f}"
            ]
        }
    
    def _extract_mentioned_files(self, request: str) -> List[str]:
        """Extraer archivos mencionados en la solicitud"""
        import re
        file_patterns = [
            r'([a-zA-Z0-9_/\\]+\.(py|js|ts|tsx|jsx|java|go|rs|cpp|c|h))',
            r'([a-zA-Z0-9_/\\]+\.(json|yaml|yml|toml|ini|cfg))',
        ]
        
        files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, request)
            files.extend([m[0] if isinstance(m, tuple) else m for m in matches])
        
        return list(set(files))
    
    def _check_files_modified(
        self,
        mentioned_files: List[str],
        changes_made: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verificar que los archivos mencionados fueron modificados"""
        if not changes_made:
            return {
                "all_modified": False,
                "confidence": 0.0,
                "evidence": ["No changes made"]
            }
        
        modified_files = [
            str(c.get("file_path", "")).lower()
            for c in changes_made
        ]
        
        mentioned_lower = [f.lower() for f in mentioned_files]
        modified_count = sum(
            1 for f in mentioned_lower
            if any(mf in f or f in mf for mf in modified_files)
        )
        
        all_modified = modified_count == len(mentioned_files)
        confidence = modified_count / len(mentioned_files) if mentioned_files else 0.0
        
        return {
            "all_modified": all_modified,
            "confidence": confidence,
            "evidence": [
                f"Modified: {modified_count}/{len(mentioned_files)}",
                f"Files: {', '.join(mentioned_files[:5])}"
            ]
        }
    
    def _check_functionality_implemented(
        self,
        original_request: str,
        changes_made: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verificar que la funcionalidad solicitada está implementada"""
        if not changes_made:
            return {
                "implemented": False,
                "confidence": 0.0,
                "evidence": ["No changes made"]
            }
        
        request_lower = original_request.lower()
        functionality_keywords = [
            "add", "create", "implement", "fix", "update", "modify",
            "improve", "enhance", "refactor", "remove", "delete"
        ]
        
        has_functionality_keyword = any(
            kw in request_lower for kw in functionality_keywords
        )
        
        change_count = len(changes_made)
        implemented = has_functionality_keyword and change_count > 0
        confidence = min(1.0, change_count / 3.0) if implemented else 0.0
        
        return {
            "implemented": implemented,
            "confidence": confidence,
            "evidence": [
                f"Changes made: {change_count}",
                f"Has functionality keyword: {has_functionality_keyword}"
            ]
        }
    
    def get_verification(self, task_id: str) -> Optional[IntentVerification]:
        """Obtener verificación de intención"""
        return self.verifications.get(task_id)
    
    def get_all_verifications(self) -> List[Dict[str, Any]]:
        """Obtener todas las verificaciones"""
        return [v.to_dict() for v in self.verifications.values()]

