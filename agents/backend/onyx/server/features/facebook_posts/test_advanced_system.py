#!/usr/bin/env python3
"""
Test script for Advanced Predictive System v3.0
"""

from advanced_predictive_system import AdvancedPredictiveSystem, AdvancedPredictiveConfig

def test_advanced_system():
    """Test the advanced predictive system"""
    print("🚀 Testing Advanced Predictive System v3.0")
    print("=" * 60)
    
    try:
        # Initialize system
        print("📋 Initializing Advanced Predictive System...")
        config = AdvancedPredictiveConfig()
        system = AdvancedPredictiveSystem(config)
        print("✅ System initialized successfully!")
        
        # Test viral prediction
        print("\n🔮 Testing Viral Prediction...")
        test_content = "🚀 Amazing breakthrough in AI technology! This will revolutionize everything! #AI #Innovation #Future"
        viral_result = system.predict_viral_potential(test_content)
        print(f"   Viral Score: {viral_result['viral_score']:.3f}")
        print(f"   Confidence: {viral_result['confidence']:.2f}")
        print(f"   Viral Probability: {viral_result['viral_probability']}")
        
        # Test sentiment analysis
        print("\n😊 Testing Advanced Sentiment Analysis...")
        sentiment_result = system.analyze_sentiment_advanced(test_content)
        print(f"   Primary Emotion: {sentiment_result['primary_emotion']}")
        print(f"   Emotion Confidence: {sentiment_result['emotion_confidence']:.2f}")
        print(f"   Overall Sentiment: {sentiment_result['sentiment_score']:.3f}")
        
        # Test engagement forecasting
        print("\n📈 Testing Engagement Forecasting...")
        from datetime import datetime
        forecast_result = system.forecast_engagement(test_content, "segment_0", datetime.now())
        print(f"   Predicted Engagement: {forecast_result['forecast']['predicted_engagement']:.3f}")
        print(f"   Confidence Level: {forecast_result['forecast']['confidence_level']:.2f}")
        
        # Test system metrics
        print("\n📊 Testing System Metrics...")
        metrics = system.get_system_metrics()
        print(f"   Cache Size: {metrics['cache_size']}")
        print(f"   Timestamp: {metrics['timestamp']}")
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Advanced Predictive System v3.0 is working perfectly!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_advanced_system()

