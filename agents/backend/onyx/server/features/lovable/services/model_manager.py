import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, Any

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton class to manage the loading of the LLM.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.model_name = "gpt2" # Using gpt2 as base, could be replaced with fine-tuned model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self._initialized = True
        logger.info(f"ModelManager initialized. Device: {self.device}")

    def load_model(self):
        """Loads the model if not already loaded."""
        if self.pipeline is not None:
            return

        logger.info(f"Loading model: {self.model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def get_pipeline(self) -> Any:
        """Returns the generation pipeline, loading it if necessary."""
        if self.pipeline is None:
            self.load_model()
        return self.pipeline
