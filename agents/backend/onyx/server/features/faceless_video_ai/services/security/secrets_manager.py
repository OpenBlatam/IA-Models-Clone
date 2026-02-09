"""
Secrets Manager
Secure secrets management
"""

from typing import Optional, Dict, Any
import os
import json
from pathlib import Path
import logging

from .encryption import get_encryption_service

logger = logging.getLogger(__name__)


class SecretsManager:
    """Manages secrets securely"""
    
    def __init__(self, secrets_file: Optional[str] = None):
        self.secrets_file = Path(secrets_file) if secrets_file else Path("/tmp/faceless_video/secrets.encrypted")
        self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
        self.encryption_service = get_encryption_service()
        self.secrets: Dict[str, str] = {}
        self.load_secrets()
    
    def load_secrets(self):
        """Load secrets from encrypted file"""
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    encrypted = f.read()
                    decrypted = self.encryption_service.decrypt(encrypted)
                    self.secrets = json.loads(decrypted)
                logger.info("Secrets loaded")
            except Exception as e:
                logger.warning(f"Failed to load secrets: {str(e)}")
                self.secrets = {}
        else:
            # Load from environment as fallback
            self._load_from_env()
    
    def _load_from_env(self):
        """Load secrets from environment variables"""
        env_secrets = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "STABILITY_AI_API_KEY": os.getenv("STABILITY_AI_API_KEY"),
            "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "GOOGLE_GEMINI_API_KEY": os.getenv("GOOGLE_GEMINI_API_KEY"),
        }
        
        self.secrets.update({k: v for k, v in env_secrets.items() if v})
    
    def save_secrets(self):
        """Save secrets to encrypted file"""
        try:
            secrets_json = json.dumps(self.secrets)
            encrypted = self.encryption_service.encrypt(secrets_json)
            
            with open(self.secrets_file, 'w') as f:
                f.write(encrypted)
            
            logger.info("Secrets saved")
        except Exception as e:
            logger.error(f"Failed to save secrets: {str(e)}")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret value"""
        return self.secrets.get(key, default)
    
    def set_secret(self, key: str, value: str):
        """Set secret value"""
        self.secrets[key] = value
        self.save_secrets()
        logger.info(f"Secret {key} updated")
    
    def delete_secret(self, key: str) -> bool:
        """Delete secret"""
        if key in self.secrets:
            del self.secrets[key]
            self.save_secrets()
            logger.info(f"Secret {key} deleted")
            return True
        return False


_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager(secrets_file: Optional[str] = None) -> SecretsManager:
    """Get secrets manager instance (singleton)"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(secrets_file=secrets_file)
    return _secrets_manager

