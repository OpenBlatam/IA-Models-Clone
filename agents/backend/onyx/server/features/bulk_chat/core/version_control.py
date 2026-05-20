"""
Version Control - Control de Versiones
======================================

Sistema de control de versiones con commits, branches, diffs y merge automático.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib
import json

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Tipo de cambio."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"
    MOVE = "move"


@dataclass
class Commit:
    """Commit de versión."""
    commit_id: str
    branch: str
    message: str
    author: str
    timestamp: datetime
    changes: List[Dict[str, Any]] = field(default_factory=list)
    parent_commit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Branch:
    """Branch de versión."""
    branch_name: str
    head_commit: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Diff:
    """Diff entre commits."""
    file_path: str
    change_type: ChangeType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    added_lines: int = 0
    removed_lines: int = 0
    modified_lines: int = 0


class VersionControl:
    """Sistema de control de versiones."""
    
    def __init__(self):
        self.commits: Dict[str, Commit] = {}
        self.branches: Dict[str, Branch] = {}
        self.current_branch: str = "main"
        self.file_history: Dict[str, List[str]] = defaultdict(list)  # file_path -> [commit_ids]
        self._lock = asyncio.Lock()
        
        # Crear branch principal
        self.branches["main"] = Branch(branch_name="main")
    
    async def create_branch(
        self,
        branch_name: str,
        from_branch: Optional[str] = None,
    ) -> str:
        """Crear nuevo branch."""
        if branch_name in self.branches:
            raise ValueError(f"Branch already exists: {branch_name}")
        
        source_branch = from_branch or self.current_branch
        source_head = self.branches[source_branch].head_commit
        
        branch = Branch(
            branch_name=branch_name,
            head_commit=source_head,
        )
        
        async with self._lock:
            self.branches[branch_name] = branch
        
        logger.info(f"Created branch: {branch_name} from {source_branch}")
        return branch_name
    
    async def commit(
        self,
        message: str,
        author: str,
        changes: List[Dict[str, Any]],
        branch: Optional[str] = None,
    ) -> str:
        """
        Crear commit.
        
        Args:
            message: Mensaje del commit
            author: Autor del commit
            changes: Lista de cambios
            branch: Branch (current si None)
        
        Returns:
            ID del commit
        """
        branch_name = branch or self.current_branch
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch not found: {branch_name}")
        
        # Generar commit ID
        commit_data = f"{message}{author}{datetime.now().isoformat()}{json.dumps(changes)}"
        commit_id = hashlib.sha256(commit_data.encode()).hexdigest()[:16]
        
        # Obtener parent commit
        parent_commit = self.branches[branch_name].head_commit
        
        commit = Commit(
            commit_id=commit_id,
            branch=branch_name,
            message=message,
            author=author,
            timestamp=datetime.now(),
            changes=changes,
            parent_commit=parent_commit,
        )
        
        async with self._lock:
            self.commits[commit_id] = commit
            self.branches[branch_name].head_commit = commit_id
            
            # Actualizar historial de archivos
            for change in changes:
                file_path = change.get("file_path")
                if file_path:
                    self.file_history[file_path].append(commit_id)
        
        logger.info(f"Created commit: {commit_id} on {branch_name}")
        return commit_id
    
    async def switch_branch(self, branch_name: str) -> bool:
        """Cambiar de branch."""
        if branch_name not in self.branches:
            return False
        
        self.current_branch = branch_name
        logger.info(f"Switched to branch: {branch_name}")
        return True
    
    async def merge_branch(
        self,
        source_branch: str,
        target_branch: str,
        merge_message: str = "Merge branch",
        author: str = "system",
    ) -> str:
        """Fusionar branch."""
        if source_branch not in self.branches or target_branch not in self.branches:
            raise ValueError("Both branches must exist")
        
        source_head = self.branches[source_branch].head_commit
        target_head = self.branches[target_branch].head_commit
        
        if not source_head:
            raise ValueError(f"Source branch {source_branch} has no commits")
        
        # Obtener cambios del source branch
        source_changes = []
        commit = self.commits.get(source_head)
        while commit:
            source_changes.extend(commit.changes)
            if commit.parent_commit:
                commit = self.commits.get(commit.parent_commit)
            else:
                break
        
        # Crear commit de merge
        merge_commit_id = await self.commit(
            message=merge_message,
            author=author,
            changes=source_changes,
            branch=target_branch,
        )
        
        logger.info(f"Merged {source_branch} into {target_branch}")
        return merge_commit_id
    
    def get_commit(self, commit_id: str) -> Optional[Dict[str, Any]]:
        """Obtener commit."""
        commit = self.commits.get(commit_id)
        if not commit:
            return None
        
        return {
            "commit_id": commit.commit_id,
            "branch": commit.branch,
            "message": commit.message,
            "author": commit.author,
            "timestamp": commit.timestamp.isoformat(),
            "changes_count": len(commit.changes),
            "parent_commit": commit.parent_commit,
            "metadata": commit.metadata,
        }
    
    def get_branch_history(self, branch_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtener historial de branch."""
        branch = self.branches.get(branch_name)
        if not branch:
            return []
        
        history = []
        commit_id = branch.head_commit
        
        while commit_id and len(history) < limit:
            commit = self.commits.get(commit_id)
            if not commit:
                break
            
            history.append({
                "commit_id": commit.commit_id,
                "message": commit.message,
                "author": commit.author,
                "timestamp": commit.timestamp.isoformat(),
                "changes_count": len(commit.changes),
            })
            
            commit_id = commit.parent_commit
        
        return history
    
    def get_file_history(self, file_path: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtener historial de archivo."""
        commit_ids = self.file_history.get(file_path, [])
        
        history = []
        for commit_id in commit_ids[-limit:]:
            commit = self.commits.get(commit_id)
            if commit:
                history.append({
                    "commit_id": commit.commit_id,
                    "message": commit.message,
                    "author": commit.author,
                    "timestamp": commit.timestamp.isoformat(),
                })
        
        return history
    
    def get_diff(self, commit_id1: str, commit_id2: str) -> List[Dict[str, Any]]:
        """Obtener diff entre dos commits."""
        commit1 = self.commits.get(commit_id1)
        commit2 = self.commits.get(commit_id2)
        
        if not commit1 or not commit2:
            return []
        
        # Obtener cambios únicos
        changes1 = {c.get("file_path"): c for c in commit1.changes}
        changes2 = {c.get("file_path"): c for c in commit2.changes}
        
        diffs = []
        all_files = set(changes1.keys()) | set(changes2.keys())
        
        for file_path in all_files:
            change1 = changes1.get(file_path)
            change2 = changes2.get(file_path)
            
            if change1 and not change2:
                diffs.append({
                    "file_path": file_path,
                    "change_type": "deleted",
                    "old_content": change1.get("content"),
                })
            elif change2 and not change1:
                diffs.append({
                    "file_path": file_path,
                    "change_type": "created",
                    "new_content": change2.get("content"),
                })
            elif change1 and change2:
                diffs.append({
                    "file_path": file_path,
                    "change_type": "modified",
                    "old_content": change1.get("content"),
                    "new_content": change2.get("content"),
                })
        
        return diffs
    
    def get_version_control_summary(self) -> Dict[str, Any]:
        """Obtener resumen de control de versiones."""
        return {
            "total_commits": len(self.commits),
            "total_branches": len(self.branches),
            "current_branch": self.current_branch,
            "tracked_files": len(self.file_history),
        }
















