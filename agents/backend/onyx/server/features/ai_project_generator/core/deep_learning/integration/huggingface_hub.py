"""
Hugging Face Hub Integration
=============================

Integration with Hugging Face Hub for model sharing and versioning.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi, Repository, upload_file
    from huggingface_hub import login as hf_login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Install with: pip install huggingface_hub")


class HuggingFaceHubIntegration:
    """
    Integration with Hugging Face Hub.
    
    Provides functionality to:
    - Upload models
    - Download models
    - Version control
    - Model sharing
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize HF Hub integration.
        
        Args:
            token: Hugging Face token (optional, can use login())
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hub is required")
        
        self.api = HfApi()
        if token:
            hf_login(token=token)
    
    def login(self, token: Optional[str] = None) -> None:
        """
        Login to Hugging Face.
        
        Args:
            token: Hugging Face token
        """
        hf_login(token=token)
        logger.info("Logged in to Hugging Face Hub")
    
    def upload_model(
        self,
        model: torch.nn.Module,
        repo_id: str,
        model_name: str = "pytorch_model.bin",
        config: Optional[Dict[str, Any]] = None,
        private: bool = False
    ) -> None:
        """
        Upload model to Hugging Face Hub.
        
        Args:
            model: PyTorch model
            repo_id: Repository ID (username/repo-name)
            model_name: Name for model file
            config: Model configuration
            private: Whether repository is private
        """
        try:
            # Create repository
            repo = Repository(
                local_dir=f"./tmp_{repo_id.replace('/', '_')}",
                clone_from=repo_id,
                use_auth_token=True,
                private=private
            )
            
            # Save model
            model_path = Path(repo.local_dir) / model_name
            torch.save(model.state_dict(), model_path)
            
            # Save config if provided
            if config:
                import json
                config_path = Path(repo.local_dir) / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # Commit and push
            repo.git_add()
            repo.git_commit(f"Upload model: {model_name}")
            repo.git_push()
            
            logger.info(f"Model uploaded to {repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise
    
    def download_model(
        self,
        repo_id: str,
        model_name: str = "pytorch_model.bin",
        local_dir: Optional[Path] = None
    ) -> Path:
        """
        Download model from Hugging Face Hub.
        
        Args:
            repo_id: Repository ID
            model_name: Name of model file
            local_dir: Local directory to save (defaults to ./models/{repo_id})
            
        Returns:
            Path to downloaded model
        """
        if local_dir is None:
            local_dir = Path("./models") / repo_id.replace('/', '_')
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download file
            model_path = self.api.hf_hub_download(
                repo_id=repo_id,
                filename=model_name,
                local_dir=str(local_dir)
            )
            
            logger.info(f"Model downloaded to {model_path}")
            return Path(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise



