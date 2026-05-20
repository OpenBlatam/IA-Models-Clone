"""
Database integration layer for persistent storage.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database operations for the tutor system.
    Supports both file-based and future SQL database integration.
    """
    
    def __init__(self, db_path: str = "data/tutor_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data directories
        (self.db_path / "students").mkdir(exist_ok=True)
        (self.db_path / "conversations").mkdir(exist_ok=True)
        (self.db_path / "evaluations").mkdir(exist_ok=True)
        (self.db_path / "reports").mkdir(exist_ok=True)
    
    def save_student_profile(self, student_id: str, profile: Dict[str, Any]):
        """Save student profile to database."""
        file_path = self.db_path / "students" / f"{student_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"Saved profile for student {student_id}")
    
    def load_student_profile(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Load student profile from database."""
        file_path = self.db_path / "students" / f"{student_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def save_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]):
        """Save conversation to database."""
        file_path = self.db_path / "conversations" / f"{conversation_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"Saved conversation {conversation_id}")
    
    def load_conversation(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load conversation from database."""
        file_path = self.db_path / "conversations" / f"{conversation_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def save_evaluation(self, student_id: str, evaluation_id: str, evaluation: Dict[str, Any]):
        """Save evaluation to database."""
        eval_dir = self.db_path / "evaluations" / student_id
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = eval_dir / f"{evaluation_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"Saved evaluation {evaluation_id} for student {student_id}")
    
    def load_evaluations(self, student_id: str) -> List[Dict[str, Any]]:
        """Load all evaluations for a student."""
        eval_dir = self.db_path / "evaluations" / student_id
        if not eval_dir.exists():
            return []
        
        evaluations = []
        for file_path in eval_dir.glob("*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                evaluations.append(json.load(f))
        
        return evaluations
    
    def save_report(self, report_id: str, report: Dict[str, Any]):
        """Save report to database."""
        file_path = self.db_path / "reports" / f"{report_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"Saved report {report_id}")
    
    def get_all_students(self) -> List[str]:
        """Get list of all student IDs."""
        students_dir = self.db_path / "students"
        if not students_dir.exists():
            return []
        
        return [
            f.stem for f in students_dir.glob("*.json")
        ]
    
    def get_student_stats(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive stats for a student."""
        profile = self.load_student_profile(student_id)
        evaluations = self.load_evaluations(student_id)
        conversations = [
            f.stem for f in (self.db_path / "conversations").glob("*.json")
            if student_id in f.stem
        ]
        
        return {
            "student_id": student_id,
            "profile_exists": profile is not None,
            "total_evaluations": len(evaluations),
            "total_conversations": len(conversations),
            "last_activity": self._get_last_activity(student_id, evaluations, conversations)
        }
    
    def _get_last_activity(
        self,
        student_id: str,
        evaluations: List[Dict],
        conversations: List[str]
    ) -> Optional[str]:
        """Get last activity timestamp."""
        timestamps = []
        
        for eval_data in evaluations:
            if "evaluated_at" in eval_data:
                timestamps.append(eval_data["evaluated_at"])
        
        # Check conversation files
        for conv_id in conversations:
            conv_file = self.db_path / "conversations" / f"{conv_id}.json"
            if conv_file.exists():
                with open(conv_file, "r", encoding="utf-8") as f:
                    conv_data = json.load(f)
                    if conv_data:
                        last_msg = conv_data[-1]
                        if "timestamp" in last_msg:
                            timestamps.append(last_msg["timestamp"])
        
        return max(timestamps) if timestamps else None
    
    def backup_database(self, backup_path: Optional[str] = None):
        """Create a backup of the database."""
        import shutil
        
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backups/tutor_db_backup_{timestamp}"
        
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        shutil.copytree(self.db_path, backup_path / "tutor_db", dirs_exist_ok=True)
        logger.info(f"Database backed up to {backup_path}")






