import logging
from typing import Optional
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

class AIBaseService:
    """
    Base service for AI-powered features.
    Encapsulates common logic for model interaction.
    """

    def __init__(self):
        self.model_manager = ModelManager()

    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 200, 
        temperature: float = 0.7,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generates text using the shared model.
        
        Args:
            prompt: The input prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            num_return_sequences: Number of sequences to return.
            
        Returns:
            The generated text string.
        """
        pipeline = self.model_manager.get_pipeline()

        try:
            output = pipeline(
                prompt, 
                max_new_tokens=max_new_tokens, 
                num_return_sequences=num_return_sequences, 
                do_sample=do_sample, 
                temperature=temperature,
                truncation=True,
                pad_token_id=50256 # GPT-2 EOS token
            )
            
            generated_text = output[0]['generated_text']
            # Remove the prompt from the output if the model includes it
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise e
