"""
Planning Verifier
=================

Sistema que verifica que se tiene toda la información necesaria antes de
sugerir un plan, siguiendo las mejores prácticas de Devin de conocer todas
las ubicaciones y referencias antes de sugerir un plan.
"""

import logging
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PlanningCheck:
    """Verificación de planificación"""
    check_type: str
    description: str
    passed: bool = False
    details: Optional[str] = None
    missing_items: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "check_type": self.check_type,
            "description": self.description,
            "passed": self.passed,
            "details": self.details,
            "missing_items": self.missing_items
        }


@dataclass
class PlanningVerification:
    """Verificación completa de planificación"""
    plan_id: str
    checks: List[PlanningCheck] = field(default_factory=list)
    all_checks_passed: bool = False
    can_suggest_plan: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_check(
        self,
        check_type: str,
        description: str,
        passed: bool,
        details: Optional[str] = None,
        missing_items: Optional[List[str]] = None
    ) -> PlanningCheck:
        """Agregar verificación"""
        check = PlanningCheck(
            check_type=check_type,
            description=description,
            passed=passed,
            details=details,
            missing_items=missing_items or []
        )
        self.checks.append(check)
        self._evaluate()
        return check
    
    def _evaluate(self) -> None:
        """Evaluar todas las verificaciones"""
        self.all_checks_passed = all(check.passed for check in self.checks)
        self.can_suggest_plan = self.all_checks_passed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "plan_id": self.plan_id,
            "checks": [c.to_dict() for c in self.checks],
            "all_checks_passed": self.all_checks_passed,
            "can_suggest_plan": self.can_suggest_plan,
            "timestamp": self.timestamp.isoformat()
        }


class PlanningVerifier:
    """
    Verificador de planificación.
    
    Verifica que se tiene toda la información necesaria antes de sugerir
    un plan, siguiendo las mejores prácticas de Devin:
    - Conocer todas las ubicaciones a editar
    - Conocer todas las referencias a actualizar
    - Tener toda la información necesaria
    """
    
    def __init__(self) -> None:
        """Inicializar verificador de planificación"""
        self.verifications: Dict[str, PlanningVerification] = {}
        logger.info("📋 Planning verifier initialized")
    
    async def verify_before_suggesting_plan(
        self,
        plan_id: str,
        locations_to_edit: Optional[List[Dict[str, Any]]] = None,
        references_to_update: Optional[List[Dict[str, Any]]] = None,
        information_gathered: Optional[Dict[str, Any]] = None,
        agent: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Verificar antes de sugerir plan.
        
        Según las reglas de Devin:
        - Debe conocer todas las ubicaciones a editar
        - Debe conocer todas las referencias a actualizar
        - Debe tener toda la información necesaria
        
        Args:
            plan_id: ID del plan.
            locations_to_edit: Lista de ubicaciones a editar (opcional).
            references_to_update: Lista de referencias a actualizar (opcional).
            information_gathered: Información recopilada (opcional).
            agent: Instancia del agente (opcional).
        
        Returns:
            Resultado de la verificación.
        """
        verification = PlanningVerification(plan_id=plan_id)
        
        # Verificación 1: Ubicaciones conocidas
        if locations_to_edit:
            locations_known = len(locations_to_edit) > 0
            missing_locations = []
            
            if not locations_known:
                missing_locations.append("No locations identified")
            
            verification.add_check(
                "locations_known",
                "All locations to edit are known",
                locations_known,
                details=f"{len(locations_to_edit)} locations identified" if locations_known else "No locations identified",
                missing_items=missing_locations
            )
        else:
            verification.add_check(
                "locations_known",
                "All locations to edit are known",
                False,
                details="No locations provided",
                missing_items=["Locations not identified"]
            )
        
        # Verificación 2: Referencias conocidas
        if references_to_update:
            references_known = len(references_to_update) > 0
            missing_references = []
            
            if not references_known:
                missing_references.append("No references identified")
            
            verification.add_check(
                "references_known",
                "All references to update are known",
                references_known,
                details=f"{len(references_to_update)} references identified" if references_known else "No references identified",
                missing_items=missing_references
            )
        else:
            # Verificar si hay referencias usando code_understanding
            if agent and hasattr(agent, 'code_understanding') and agent.code_understanding:
                try:
                    # Intentar encontrar referencias si hay ubicaciones
                    if locations_to_edit:
                        all_references = []
                        for location in locations_to_edit:
                            if location.get('symbol'):
                                refs = agent.code_understanding.find_references(
                                    location.get('file_path', ''),
                                    location.get('symbol', '')
                                )
                                all_references.extend(refs)
                        
                        if all_references:
                            verification.add_check(
                                "references_known",
                                "All references to update are known",
                                True,
                                details=f"{len(all_references)} references found automatically"
                            )
                        else:
                            verification.add_check(
                                "references_known",
                                "All references to update are known",
                                True,
                                details="No references found (may not be needed)"
                            )
                    else:
                        verification.add_check(
                            "references_known",
                            "All references to update are known",
                            True,
                            details="No locations to check references for"
                        )
                except Exception as e:
                    verification.add_check(
                        "references_known",
                        "All references to update are known",
                        False,
                        details=f"Error checking references: {e}",
                        missing_items=["Could not verify references"]
                    )
            else:
                verification.add_check(
                    "references_known",
                    "All references to update are known",
                    True,
                    details="Code understanding not available, assuming no references needed"
                )
        
        # Verificación 3: Información recopilada
        if information_gathered:
            info_complete = information_gathered.get('complete', False)
            missing_info = information_gathered.get('missing', [])
            
            verification.add_check(
                "information_gathered",
                "All necessary information gathered",
                info_complete,
                details=f"Information status: {'complete' if info_complete else 'incomplete'}",
                missing_items=missing_info
            )
        else:
            verification.add_check(
                "information_gathered",
                "All necessary information gathered",
                True,
                details="No specific information requirements"
            )
        
        # Verificación 4: Contexto del código entendido
        if agent and hasattr(agent, 'code_understanding') and agent.code_understanding:
            try:
                structure = agent.code_understanding.analyze_codebase_structure()
                if structure:
                    verification.add_check(
                        "codebase_understood",
                        "Codebase structure understood",
                        True,
                        details=f"Analyzed {structure.get('total_files', 0)} files"
                    )
                else:
                    verification.add_check(
                        "codebase_understood",
                        "Codebase structure understood",
                        False,
                        details="Could not analyze codebase structure",
                        missing_items=["Codebase analysis failed"]
                    )
            except Exception as e:
                verification.add_check(
                    "codebase_understood",
                    "Codebase structure understood",
                    False,
                    details=f"Error analyzing codebase: {e}",
                    missing_items=["Codebase analysis error"]
                )
        else:
            verification.add_check(
                "codebase_understood",
                "Codebase structure understood",
                True,
                details="Code understanding not available"
            )
        
        # Verificación 5: Convenciones conocidas
        if agent and hasattr(agent, 'code_conventions') and agent.code_conventions:
            try:
                conventions = agent.code_conventions.get_conventions()
                if conventions:
                    verification.add_check(
                        "conventions_known",
                        "Code conventions understood",
                        True,
                        details="Conventions analyzed"
                    )
                else:
                    verification.add_check(
                        "conventions_known",
                        "Code conventions understood",
                        False,
                        details="Could not determine conventions",
                        missing_items=["Conventions not analyzed"]
                    )
            except Exception as e:
                verification.add_check(
                    "conventions_known",
                    "Code conventions understood",
                    False,
                    details=f"Error analyzing conventions: {e}",
                    missing_items=["Conventions analysis error"]
                )
        else:
            verification.add_check(
                "conventions_known",
                "Code conventions understood",
                True,
                details="Code conventions analyzer not available"
            )
        
        self.verifications[plan_id] = verification
        
        return {
            "success": verification.can_suggest_plan,
            "all_checks_passed": verification.all_checks_passed,
            "can_suggest_plan": verification.can_suggest_plan,
            "checks": [c.to_dict() for c in verification.checks],
            "issues": [
                {
                    "check": c.check_type,
                    "description": c.description,
                    "missing": c.missing_items
                }
                for c in verification.checks
                if not c.passed
            ]
        }
    
    def get_verification(self, plan_id: str) -> Optional[PlanningVerification]:
        """Obtener verificación"""
        return self.verifications.get(plan_id)
    
    def get_all_verifications(self) -> List[Dict[str, Any]]:
        """Obtener todas las verificaciones"""
        return [v.to_dict() for v in self.verifications.values()]

