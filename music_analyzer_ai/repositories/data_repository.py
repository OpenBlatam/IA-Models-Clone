"""
Data Repository - Store and retrieve data
"""

from typing import Optional, Dict, Any, List
import json
import logging
from pathlib import Path

from .repository import BaseRepository

logger = logging.getLogger(__name__)


class DataRepository(BaseRepository):
    """
    Repository for data storage and retrieval
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        super().__init__("DataRepository")
        self.storage_path = Path(storage_path) if storage_path else Path("./data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save_data(self, data_id: str, data: Any) -> bool:
        """Save data to disk"""
        try:
            data_path = self.storage_path / f"{data_id}.json"
            
            # Convert to JSON-serializable format
            if isinstance(data, dict):
                json_data = data
            else:
                json_data = {"data": str(data)}
            
            with open(data_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Saved data {data_id} to {data_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving data {data_id}: {str(e)}")
            return False
    
    def load_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        """Load data from disk"""
        try:
            data_path = self.storage_path / f"{data_id}.json"
            
            if not data_path.exists():
                logger.warning(f"Data {data_id} not found at {data_path}")
                return None
            
            with open(data_path, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading data {data_id}: {str(e)}")
            return None








