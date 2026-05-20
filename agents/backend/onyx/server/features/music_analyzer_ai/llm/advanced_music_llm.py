"""
Advanced Music LLM
LLM-based music analysis and generation
"""

from typing import Dict, Any, Optional, List
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline
)

logger = logging.getLogger(__name__)


class MusicLLMAnalyzer:
    """
    Advanced LLM for music analysis and generation
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_8bit: bool = False
    ):
        self.model_name = model_name
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {}
        if use_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            **model_kwargs
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        logger.info(f"Initialized Music LLM with {model_name}")
    
    def analyze_music_features(
        self,
        track_name: str,
        features: Dict[str, Any]
    ) -> str:
        """Generate analysis using LLM"""
        prompt = f"""Analyze this music track:

Track: {track_name}
Tempo: {features.get('tempo', 'N/A')} BPM
Energy: {features.get('energy', 'N/A')}
Danceability: {features.get('danceability', 'N/A')}
Valence: {features.get('valence', 'N/A')}
Genre: {features.get('genre', 'N/A')}

Analysis:"""
        
        response = self.generator(
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return response[0]["generated_text"]
    
    def generate_recommendations(
        self,
        user_preferences: Dict[str, Any],
        num_recommendations: int = 5
    ) -> List[str]:
        """Generate music recommendations using LLM"""
        prompt = f"""Based on these preferences:
- Favorite genres: {user_preferences.get('genres', [])}
- Energy level: {user_preferences.get('energy', 'medium')}
- Mood: {user_preferences.get('mood', 'neutral')}

Recommend {num_recommendations} similar tracks:"""
        
        response = self.generator(
            prompt,
            max_length=150,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )
        
        # Parse recommendations from response
        text = response[0]["generated_text"]
        recommendations = [
            line.strip() for line in text.split("\n")
            if line.strip() and not line.strip().startswith("Based on")
        ]
        
        return recommendations[:num_recommendations]
    
    def generate_playlist_description(
        self,
        playlist_tracks: List[Dict[str, Any]]
    ) -> str:
        """Generate playlist description"""
        genres = [t.get("genre", "Unknown") for t in playlist_tracks]
        unique_genres = list(set(genres))
        
        prompt = f"""Create a playlist description for these tracks:
Genres: {', '.join(unique_genres)}
Number of tracks: {len(playlist_tracks)}

Description:"""
        
        response = self.generator(
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7
        )
        
        return response[0]["generated_text"]


class MusicTransformerEncoder:
    """
    Advanced transformer encoder for music features
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        from transformers import AutoModel
        
        self.model_name = model_name
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        logger.info(f"Initialized Music Transformer Encoder with {model_name}")
    
    def encode_music_text(
        self,
        text: str,
        max_length: int = 512
    ) -> torch.Tensor:
        """Encode music-related text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def encode_music_features(
        self,
        features: Dict[str, Any]
    ) -> torch.Tensor:
        """Encode music features as text and get embeddings"""
        feature_text = f"""Tempo: {features.get('tempo', 0)} BPM
Energy: {features.get('energy', 0)}
Danceability: {features.get('danceability', 0)}
Valence: {features.get('valence', 0)}
Genre: {features.get('genre', 'Unknown')}"""
        
        return self.encode_music_text(feature_text)

