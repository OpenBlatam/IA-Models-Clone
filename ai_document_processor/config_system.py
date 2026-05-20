"""
Configuration System for AI Document Processor
Real, working configuration management features for document processing
"""

import asyncio
import logging
import json
import time
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class ConfigSystem:
    """Real working configuration system for AI document processing"""
    
    def __init__(self):
        self.config_file = "config.json"
        self.config_dir = "config"
        self.backup_dir = "config_backups"
        
        # Default configuration
        self.default_config = {
            "system": {
                "name": "AI Document Processor",
                "version": "6.0.0",
                "environment": "development",
                "debug": False,
                "log_level": "INFO"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "reload": True,
                "cors_origins": ["*"],
                "max_request_size": 50 * 1024 * 1024,  # 50MB
                "request_timeout": 300  # 5 minutes
            },
            "ai": {
                "models": {
                    "spacy_model": "en_core_web_sm",
                    "nltk_data": ["vader_lexicon", "punkt", "stopwords", "wordnet"],
                    "transformers": {
                        "classifier": "distilbert-base-uncased-finetuned-sst-2-english",
                        "summarizer": "facebook/bart-large-cnn",
                        "qa": "distilbert-base-cased-distilled-squad"
                    }
                },
                "processing": {
                    "max_text_length": 1000000,  # 1MB
                    "batch_size": 10,
                    "cache_ttl": 3600,  # 1 hour
                    "enable_caching": True
                }
            },
            "security": {
                "rate_limiting": {
                    "enabled": True,
                    "max_requests_per_minute": 100,
                    "max_requests_per_hour": 1000
                },
                "api_keys": {
                    "enabled": False,
                    "default_expiry_hours": 24
                },
                "file_validation": {
                    "enabled": True,
                    "max_file_size": 50 * 1024 * 1024,  # 50MB
                    "allowed_extensions": [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".jpg", ".jpeg", ".png"]
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 60,  # seconds
                "health_check_interval": 30,  # seconds
                "alert_thresholds": {
                    "cpu_usage": 80.0,
                    "memory_usage": 85.0,
                    "disk_usage": 90.0
                }
            },
            "notifications": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_email": ""
                },
                "webhooks": {
                    "enabled": False,
                    "urls": []
                }
            },
            "backup": {
                "enabled": True,
                "interval_hours": 24,
                "max_backups": 10,
                "include_data": True
            },
            "workflow": {
                "enabled": True,
                "max_concurrent_workflows": 5,
                "default_timeout": 300  # 5 minutes
            }
        }
        
        self.current_config = self.default_config.copy()
        self.config_history = []
        
        # Ensure directories exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.current_config = self._merge_configs(self.default_config, loaded_config)
                    logger.info("Configuration loaded from file")
            else:
                logger.info("No configuration file found, using defaults")
                self._save_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded configuration with defaults"""
        merged = default.copy()
        
        for key, value in loaded.items():
            if key in merged:
                if isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_configs(merged[key], value)
                else:
                    merged[key] = value
            else:
                merged[key] = value
        
        return merged
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            # Create backup before saving
            if os.path.exists(self.config_file):
                backup_name = f"config_backup_{int(time.time())}.json"
                backup_path = os.path.join(self.backup_dir, backup_name)
                os.rename(self.config_file, backup_path)
            
            # Save current configuration
            with open(self.config_file, 'w') as f:
                json.dump(self.current_config, f, indent=2)
            
            # Add to history
            self.config_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "save",
                "config_hash": hash(json.dumps(self.current_config, sort_keys=True))
            })
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    async def update_config(self, section: str, key: str, value: Any) -> Dict[str, Any]:
        """Update configuration value"""
        try:
            # Validate section and key
            if section not in self.current_config:
                return {"error": f"Section '{section}' not found"}
            
            # Create backup of current config
            backup_config = json.loads(json.dumps(self.current_config))
            
            # Update configuration
            if key in self.current_config[section]:
                old_value = self.current_config[section][key]
                self.current_config[section][key] = value
                
                # Save configuration
                self._save_config()
                
                # Add to history
                self.config_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "update",
                    "section": section,
                    "key": key,
                    "old_value": old_value,
                    "new_value": value
                })
                
                return {
                    "status": "updated",
                    "section": section,
                    "key": key,
                    "old_value": old_value,
                    "new_value": value
                }
            else:
                return {"error": f"Key '{key}' not found in section '{section}'"}
                
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return {"error": str(e)}
    
    async def get_config(self, section: str = None, key: str = None) -> Dict[str, Any]:
        """Get configuration value(s)"""
        try:
            if section is None:
                return self.current_config.copy()
            elif key is None:
                if section in self.current_config:
                    return {section: self.current_config[section]}
                else:
                    return {"error": f"Section '{section}' not found"}
            else:
                if section in self.current_config and key in self.current_config[section]:
                    return {
                        "section": section,
                        "key": key,
                        "value": self.current_config[section][key]
                    }
                else:
                    return {"error": f"Key '{key}' not found in section '{section}'"}
                    
        except Exception as e:
            logger.error(f"Error getting configuration: {e}")
            return {"error": str(e)}
    
    async def reset_config(self, section: str = None) -> Dict[str, Any]:
        """Reset configuration to defaults"""
        try:
            if section is None:
                # Reset entire configuration
                old_config = self.current_config.copy()
                self.current_config = self.default_config.copy()
                self._save_config()
                
                # Add to history
                self.config_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "reset_all"
                })
                
                return {
                    "status": "reset",
                    "message": "Configuration reset to defaults"
                }
            else:
                # Reset specific section
                if section in self.default_config:
                    old_section = self.current_config[section].copy()
                    self.current_config[section] = self.default_config[section].copy()
                    self._save_config()
                    
                    # Add to history
                    self.config_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "action": "reset_section",
                        "section": section
                    })
                    
                    return {
                        "status": "reset",
                        "section": section,
                        "message": f"Section '{section}' reset to defaults"
                    }
                else:
                    return {"error": f"Section '{section}' not found"}
                    
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return {"error": str(e)}
    
    async def export_config(self, format: str = "json") -> Dict[str, Any]:
        """Export configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == "json":
                filename = f"config_export_{timestamp}.json"
                filepath = os.path.join(self.config_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(self.current_config, f, indent=2)
                
            elif format == "yaml":
                filename = f"config_export_{timestamp}.yaml"
                filepath = os.path.join(self.config_dir, filename)
                
                with open(filepath, 'w') as f:
                    yaml.dump(self.current_config, f, default_flow_style=False)
                
            else:
                return {"error": f"Unsupported format: {format}"}
            
            return {
                "status": "exported",
                "format": format,
                "filename": filename,
                "filepath": filepath
            }
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return {"error": str(e)}
    
    async def import_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Import configuration"""
        try:
            # Validate configuration structure
            if not self._validate_config(config_data):
                return {"error": "Invalid configuration structure"}
            
            # Create backup of current config
            backup_config = self.current_config.copy()
            
            # Import new configuration
            self.current_config = self._merge_configs(self.default_config, config_data)
            self._save_config()
            
            # Add to history
            self.config_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "import"
            })
            
            return {
                "status": "imported",
                "message": "Configuration imported successfully"
            }
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return {"error": str(e)}
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        try:
            # Check required sections
            required_sections = ["system", "api", "ai", "security", "monitoring"]
            for section in required_sections:
                if section not in config:
                    return False
            
            # Validate system section
            if "name" not in config["system"] or "version" not in config["system"]:
                return False
            
            # Validate api section
            if "host" not in config["api"] or "port" not in config["api"]:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_config_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get configuration history"""
        return self.config_history[-limit:]
    
    def get_config_stats(self) -> Dict[str, Any]:
        """Get configuration statistics"""
        return {
            "config_file": self.config_file,
            "config_dir": self.config_dir,
            "backup_dir": self.backup_dir,
            "history_entries": len(self.config_history),
            "sections": list(self.current_config.keys()),
            "last_modified": self.config_history[-1]["timestamp"] if self.config_history else None
        }
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        env = self.current_config["system"]["environment"]
        
        env_configs = {
            "development": {
                "debug": True,
                "log_level": "DEBUG",
                "reload": True,
                "workers": 1
            },
            "staging": {
                "debug": False,
                "log_level": "INFO",
                "reload": False,
                "workers": 2
            },
            "production": {
                "debug": False,
                "log_level": "WARNING",
                "reload": False,
                "workers": 4
            }
        }
        
        return env_configs.get(env, env_configs["development"])

# Global instance
config_system = ConfigSystem()













