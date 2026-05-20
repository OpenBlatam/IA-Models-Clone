"""
Reference Tracker
=================

Sistema que rastrea y verifica referencias a código modificado,
asegurando que todas las referencias se actualicen cuando sea necesario.
"""

import logging
import ast
import re
from typing import Optional, Dict, Any, List, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CodeReference:
    """Referencia a código"""
    symbol_name: str
    file_path: str
    line: int
    reference_type: str
    context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "symbol_name": self.symbol_name,
            "file_path": self.file_path,
            "line": self.line,
            "reference_type": self.reference_type,
            "context": self.context
        }


@dataclass
class ReferenceChange:
    """Cambio en referencia"""
    reference: CodeReference
    status: str = "pending"
    updated: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "reference": self.reference.to_dict(),
            "status": self.status,
            "updated": self.updated,
            "error": self.error
        }


class ReferenceTracker:
    """
    Rastreador de referencias.
    
    Rastrea referencias a código modificado y verifica
    que todas se actualicen cuando sea necesario.
    """
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar rastreador.
        
        Args:
            workspace_root: Raíz del workspace.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.reference_cache: Dict[str, List[CodeReference]] = {}
        self.tracked_changes: Dict[str, List[ReferenceChange]] = {}
        logger.info("🔗 Reference tracker initialized")
    
    def find_references(
        self,
        symbol_name: str,
        file_path: str,
        line: int
    ) -> List[CodeReference]:
        """
        Encontrar todas las referencias a un símbolo.
        
        Args:
            symbol_name: Nombre del símbolo.
            file_path: Archivo donde está definido.
            line: Línea donde está definido.
        
        Returns:
            Lista de referencias encontradas.
        """
        cache_key = f"{symbol_name}:{file_path}:{line}"
        if cache_key in self.reference_cache:
            return self.reference_cache[cache_key]
        
        references: List[CodeReference] = []
        
        try:
            python_files = list(self.workspace_root.rglob("*.py"))
            
            for py_file in python_files:
                if not py_file.is_file():
                    continue
                
                try:
                    content = py_file.read_text(encoding='utf-8')
                    
                    if symbol_name in content:
                        lines = content.split('\n')
                        for idx, line_content in enumerate(lines, 1):
                            if symbol_name in line_content and idx != line:
                                try:
                                    tree = ast.parse(content)
                                    for node in ast.walk(tree):
                                        if isinstance(node, ast.Name) and node.id == symbol_name:
                                            references.append(CodeReference(
                                                symbol_name=symbol_name,
                                                file_path=str(py_file.relative_to(self.workspace_root)),
                                                line=idx,
                                                reference_type="usage",
                                                context=line_content.strip()[:100]
                                            ))
                                            break
                                except SyntaxError:
                                    references.append(CodeReference(
                                        symbol_name=symbol_name,
                                        file_path=str(py_file.relative_to(self.workspace_root)),
                                        line=idx,
                                        reference_type="usage",
                                        context=line_content.strip()[:100]
                                    ))
                except Exception as e:
                    logger.debug(f"Error reading {py_file}: {e}")
            
            self.reference_cache[cache_key] = references
            return references
            
        except Exception as e:
            logger.error(f"Error finding references: {e}", exc_info=True)
            return references
    
    def track_symbol_change(
        self,
        symbol_name: str,
        file_path: str,
        line: int,
        change_type: str
    ) -> List[ReferenceChange]:
        """
        Rastrear cambio en un símbolo y encontrar referencias.
        
        Args:
            symbol_name: Nombre del símbolo.
            file_path: Archivo donde está el símbolo.
            line: Línea donde está el símbolo.
            change_type: Tipo de cambio (rename, modify, delete).
        
        Returns:
            Lista de cambios de referencias a rastrear.
        """
        references = self.find_references(symbol_name, file_path, line)
        
        change_id = f"{symbol_name}:{file_path}:{line}:{change_type}"
        reference_changes: List[ReferenceChange] = []
        
        for ref in references:
            ref_change = ReferenceChange(
                reference=ref,
                status="pending"
            )
            reference_changes.append(ref_change)
        
        self.tracked_changes[change_id] = reference_changes
        
        logger.info(
            f"Tracking {len(references)} references for {symbol_name} "
            f"in {file_path}:{line}"
        )
        
        return reference_changes
    
    def verify_references_updated(self, change_id: str) -> Dict[str, Any]:
        """
        Verificar que todas las referencias fueron actualizadas.
        
        Args:
            change_id: ID del cambio a verificar.
        
        Returns:
            Resultado de la verificación.
        """
        if change_id not in self.tracked_changes:
            return {
                "success": False,
                "error": f"Change {change_id} not found"
            }
        
        reference_changes = self.tracked_changes[change_id]
        results = {
            "success": True,
            "total_references": len(reference_changes),
            "updated_references": 0,
            "pending_references": 0,
            "issues": []
        }
        
        for ref_change in reference_changes:
            ref = ref_change.reference
            file_path = self.workspace_root / ref.file_path
            
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    if ref.line <= len(lines):
                        line_content = lines[ref.line - 1]
                        if ref.symbol_name in line_content:
                            ref_change.updated = True
                            ref_change.status = "verified"
                            results["updated_references"] += 1
                        else:
                            ref_change.status = "missing"
                            results["pending_references"] += 1
                            results["issues"].append({
                                "type": "reference_not_updated",
                                "file": ref.file_path,
                                "line": ref.line,
                                "symbol": ref.symbol_name
                            })
                except Exception as e:
                    ref_change.status = "error"
                    ref_change.error = str(e)
                    results["issues"].append({
                        "type": "error",
                        "file": ref.file_path,
                        "error": str(e)
                    })
        
        if results["pending_references"] == 0 and len(results["issues"]) == 0:
            results["success"] = True
        else:
            results["success"] = False
        
        return results
    
    def get_tracked_changes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Obtener todos los cambios rastreados"""
        return {
            change_id: [rc.to_dict() for rc in changes]
            for change_id, changes in self.tracked_changes.items()
        }

