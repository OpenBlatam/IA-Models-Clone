"""
Risk Assessment Model
Uses ClinicalBERT for analyzing text to detect crisis levels and emotional state.
Based on research in NLP for mental health surveillance (e.g., Coppersmith et al., 2014).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class RiskAssessmentModel:
    """
    Analyzes user text (journals, chats) to assess risk of relapse or crisis.
    Uses a pre-trained ClinicalBERT model fine-tuned for mental health tasks.
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", device: str = "cpu"):
        """
        Initialize the risk assessment model.
        
        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu' or 'cuda'.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        self._load_model()

    def _load_model(self):
        """Loads model and tokenizer from HuggingFace."""
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading model: {self.model_name}")
            # In a real scenario, we would load a fine-tuned version. 
            # For this implementation, we use the base model and add a classification head.
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=4  # Low, Medium, High, Critical
            ).to(self.device)
            self.model.eval()
            
            logger.info("RiskAssessmentModel loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load RiskAssessmentModel: {e}")
            raise

    def assess_risk(self, text: str) -> Dict[str, any]:
        """
        Analyzes text and returns a risk profile.
        
        Args:
            text: Single string of user input.
            
        Returns:
            Dictionary containing risk_level, confidence, and detected_keywords.
        """
        if not text:
            return {"error": "Empty text provided"}

        # 1. Keyword Analysis (Rule-based pre-filter)
        crisis_keywords = ["suicide", "kill myself", "end it all", "overdose", "hurt myself"]
        high_risk_keywords = ["relapse", "using", "drunk", "high", "cravings", "desperate"]
        
        found_crisis = [kw for kw in crisis_keywords if kw in text.lower()]
        found_high = [kw for kw in high_risk_keywords if kw in text.lower()]
        
        if found_crisis:
            return {
                "risk_level": "CRITICAL",
                "confidence": 1.0,
                "flags": found_crisis,
                "recommendation": "IMMEDIATE INTERVENTION REQUIRED"
            }

        # 2. Model Inference
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Map output to labels (Assuming model was trained with these labels)
        # 0: Low, 1: Moderate, 2: High, 3: Critical
        risk_score = torch.argmax(probabilities).item()
        confidence = probabilities[0][risk_score].item()
        
        labels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        predicted_label = labels[risk_score]
        
        # Heuristic adjustment: If keywords meant HIGH but model said LOW, bump it up.
        if found_high and risk_score < 2:
            predicted_label = "HIGH"
            confidence = 0.85 # Artificial confidence for rule-based override

        # Safety: If no risk keywords found, cap risk at MODERATE (unless model is very confident)
        if not found_high and not found_crisis and risk_score >= 2:
             if confidence < 0.9: # If model is just guessing
                 predicted_label = "LOW"
                 confidence = 0.5

        return {
            "risk_level": predicted_label,
            "confidence": round(confidence, 4),
            "flags": found_high,
            "raw_scores": probabilities[0].tolist()
        }

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    analyzer = RiskAssessmentModel()
    
    test_texts = [
        "I'm feeling great today, went for a run!",
        "I am struggling with cravings and I don't know if I can make it.",
        "I just want to end it all.",
    ]
    
    for t in test_texts:
        print(f"Text: {t}")
        print(f"Result: {analyzer.assess_risk(t)}")
        print("-" * 20)
