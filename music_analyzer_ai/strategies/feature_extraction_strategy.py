"""
Feature Extraction Strategies
"""

from typing import Any
import numpy as np
import logging

from .strategy import IStrategy

logger = logging.getLogger(__name__)


class FeatureExtractionStrategy(IStrategy):
    """Base strategy for feature extraction"""
    
    @property
    def name(self) -> str:
        return "BaseFeatureExtraction"


class LibrosaStrategy(FeatureExtractionStrategy):
    """Extract features using librosa"""
    
    @property
    def name(self) -> str:
        return "LibrosaFeatureExtraction"
    
    def execute(self, data: Any) -> np.ndarray:
        """Extract features using librosa"""
        try:
            import librosa
            
            if isinstance(data, str):
                y, sr = librosa.load(data)
            else:
                y, sr = data, 22050
            
            # Extract common features
            features = []
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.append(mfcc.mean(axis=1))
            
            # Chroma
            chroma = librosa.feature.chroma(y=y, sr=sr)
            features.append(chroma.mean(axis=1))
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features.append(tonnetz.mean(axis=1))
            
            return np.concatenate(features)
        
        except Exception as e:
            logger.error(f"Librosa feature extraction failed: {str(e)}")
            raise


class TransformerStrategy(FeatureExtractionStrategy):
    """Extract features using transformers"""
    
    @property
    def name(self) -> str:
        return "TransformerFeatureExtraction"
    
    def execute(self, data: Any) -> np.ndarray:
        """Extract features using transformer model"""
        try:
            from ..core.transformer_analyzer import TransformerMusicAnalyzer
            
            analyzer = TransformerMusicAnalyzer(device="cuda")
            embeddings = analyzer.extract_embeddings(data)
            return embeddings
        
        except Exception as e:
            logger.error(f"Transformer feature extraction failed: {str(e)}")
            raise








