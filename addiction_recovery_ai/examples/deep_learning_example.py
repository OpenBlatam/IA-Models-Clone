"""
Deep Learning Example - Addiction Recovery AI
"""

import torch
from addiction_recovery_ai import (
    create_enhanced_analyzer,
    create_sentiment_analyzer,
    create_progress_predictor,
    create_relapse_predictor,
    create_llm_coach,
    create_recovery_gradio_app
)


def main():
    """Example usage of deep learning features"""
    
    print("=== Addiction Recovery AI with Deep Learning ===\n")
    
    # Create enhanced analyzer
    print("Initializing Enhanced Analyzer...")
    analyzer = create_enhanced_analyzer(use_gpu=torch.cuda.is_available())
    
    # Example: Sentiment Analysis
    print("\n1. Sentiment Analysis:")
    print("-" * 50)
    journal_entry = "Today was challenging but I stayed strong. I'm proud of my progress."
    sentiment = analyzer.analyze_sentiment(journal_entry)
    print(f"Sentiment: {sentiment.get('label')} (confidence: {sentiment.get('score', 0):.2%})")
    
    # Example: Progress Prediction
    print("\n2. Progress Prediction:")
    print("-" * 50)
    features = {
        "days_sober": 30,
        "cravings_level": 3,
        "stress_level": 4,
        "support_level": 8,
        "mood_score": 7,
        "sleep_quality": 6,
        "exercise_frequency": 4,
        "therapy_sessions": 2,
        "medication_compliance": 1.0,
        "social_activity": 5
    }
    progress = analyzer.predict_progress(features)
    print(f"Recovery Progress: {progress:.2%}")
    
    # Example: Relapse Risk
    print("\n3. Relapse Risk Assessment:")
    print("-" * 50)
    sequence = [
        {"cravings_level": 3, "stress_level": 4, "mood_score": 7, "triggers_count": 1, "consumed": 0.0},
        {"cravings_level": 4, "stress_level": 5, "mood_score": 6, "triggers_count": 2, "consumed": 0.0},
        {"cravings_level": 5, "stress_level": 6, "mood_score": 5, "triggers_count": 3, "consumed": 0.0}
    ]
    risk = analyzer.predict_relapse_risk(sequence)
    print(f"Relapse Risk: {risk:.2%}")
    if risk > 0.6:
        print("⚠️ High risk detected!")
    elif risk > 0.3:
        print("⚠️ Moderate risk")
    else:
        print("✅ Low risk")
    
    # Example: AI Coaching
    print("\n4. AI Coaching:")
    print("-" * 50)
    coaching = analyzer.generate_coaching(
        user_situation="Feeling stressed about work deadlines",
        days_sober=30,
        current_challenge="Strong cravings in the evening"
    )
    print(f"Coaching: {coaching[:200]}...")
    
    # Example: Motivation
    print("\n5. Motivational Message:")
    print("-" * 50)
    motivation = analyzer.generate_motivation(
        milestone="30 Days Sober",
        achievement="You've completed your first month!"
    )
    print(f"Motivation: {motivation[:200]}...")
    
    print("\n=== Example Complete ===")


def launch_gradio():
    """Launch Gradio interface"""
    analyzer = create_enhanced_analyzer(use_gpu=torch.cuda.is_available())
    app = create_recovery_gradio_app(analyzer)
    app.launch(server_port=7860)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        launch_gradio()
    else:
        main()

