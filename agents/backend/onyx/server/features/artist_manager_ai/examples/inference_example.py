#!/usr/bin/env python3
"""
Inference Example
=================

Example of using trained models for inference.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.factories import ModelFactory
from ml.prediction_service import PredictionService
from ml.data.preprocessing import FeatureExtractor


def main():
    """Main inference function."""
    print("🔮 Starting Inference Example")
    
    # 1. Create prediction service
    print("📦 Creating prediction service...")
    service = PredictionService(
        model_dir="models",
        device="auto"
    )
    print("✅ Service created")
    
    # 2. Predict event duration
    print("\n📅 Predicting event duration...")
    event_data = {
        "type": "concert",
        "start_time": "2024-01-01T20:00:00",
        "location": "Madison Square Garden",
        "description": "Major concert performance"
    }
    
    prediction = service.predict_event_duration(
        event_type="concert",
        historical_events=[],
        event_data=event_data
    )
    
    print(f"  Predicted Duration: {prediction.predicted_duration_hours} hours")
    print(f"  Confidence: {prediction.confidence:.1%}")
    print(f"  Recommendation: {prediction.recommendation}")
    
    # 3. Predict routine completion
    print("\n🔄 Predicting routine completion...")
    routine_data = {
        "type": "exercise",
        "scheduled_time": "09:00",
        "day_of_week": 0
    }
    
    routine_prediction = service.predict_routine_completion(
        routine_id="routine_1",
        completion_history=[],
        routine_data=routine_data
    )
    
    print(f"  Completion Rate: {routine_prediction.predicted_completion_rate:.1%}")
    print(f"  Confidence: {routine_prediction.confidence:.1%}")
    print(f"  Recommendation: {routine_prediction.recommendation}")
    
    # 4. Predict optimal time
    print("\n⏰ Predicting optimal event time...")
    time_prediction = service.predict_best_event_time(
        event_type="concert",
        historical_events=[],
        event_data=event_data
    )
    
    print(f"  Optimal Hour: {time_prediction['optimal_hour']}:00")
    print(f"  Confidence: {time_prediction['confidence']:.1%}")
    print(f"  Recommendation: {time_prediction['recommendation']}")
    
    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()




