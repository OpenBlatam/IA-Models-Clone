
import sys
import os
import torch
import logging

# Add the project root to the python path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.risk_assessment_model import RiskAssessmentModel
from core.models.relapse_prediction_model import RelapsePredictionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_risk_assessment():
    logger.info("Testing RiskAssessmentModel...")
    try:
        model = RiskAssessmentModel()
        
        # Test Case 1: Low Risk
        text_low = "I had a productive day at work and went to the gym."
        result_low = model.assess_risk(text_low)
        logger.info(f"Input: {text_low}")
        logger.info(f"Output: {result_low['risk_level']} (Confidence: {result_low['confidence']})")
        
        # Test Case 2: High Risk (Keywords)
        text_high = "I am having strong cravings and I feel like I might relapse."
        result_high = model.assess_risk(text_high)
        logger.info(f"Input: {text_high}")
        logger.info(f"Output: {result_high['risk_level']} (Confidence: {result_high['confidence']})")
        
        # Test Case 3: Critical Risk
        text_critical = "I want to kill myself."
        result_critical = model.assess_risk(text_critical)
        logger.info(f"Input: {text_critical}")
        logger.info(f"Output: {result_critical['risk_level']} (Confidence: {result_critical['confidence']})")
        
    except Exception as e:
        logger.error(f"RiskAssessmentModel test failed: {e}")
        raise

def test_relapse_prediction():
    logger.info("\nTesting RelapsePredictionModel...")
    try:
        model = RelapsePredictionModel()
        
        # Mock train to initialize weights decently
        # model.train_mock() 
        # Skipping full training for quick verification, just init is enough to run forward pass
        
        # Test Case 1: Deteriorating metrics
        # Craving(High), Mood(Low), Stress(High), Sleep(Bad), Triggers(Many)
        deteriorating_seq = [
            [2, 8, 2, 8, 0],
            [3, 7, 3, 7, 0],
            [5, 6, 4, 6, 1],
            [6, 5, 5, 5, 1],
            [7, 4, 6, 4, 2],
            [8, 3, 7, 3, 2],
            [9, 2, 8, 2, 3] 
        ]
        
        prob = model.predict_relapse_probability(deteriorating_seq)
        logger.info(f"Deteriorating Sequence Relapse Probability: {prob:.4f}")
        
    except Exception as e:
        logger.error(f"RelapsePredictionModel test failed: {e}")
        raise

if __name__ == "__main__":
    test_risk_assessment()
    test_relapse_prediction()
