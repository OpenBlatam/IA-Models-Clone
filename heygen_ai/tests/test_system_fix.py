"""
Test System Fix
==============

Simple test to verify the system works correctly after fixes.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported without errors"""
    print("🧪 Testing imports...")
    
    try:
        from sentient_ai_generator import SentientAIGenerator
        print("✅ Sentient AI Generator: OK")
    except ImportError as e:
        print(f"❌ Sentient AI Generator: {e}")
    
    try:
        from multiverse_testing_system import MultiverseTestingSystem
        print("✅ Multiverse Testing System: OK")
    except ImportError as e:
        print(f"❌ Multiverse Testing System: {e}")
    
    try:
        from quantum_ai_consciousness import QuantumAIConsciousnessSystem
        print("✅ Quantum AI Consciousness: OK")
    except ImportError as e:
        print(f"❌ Quantum AI Consciousness: {e}")
    
    try:
        from consciousness_integration_system import ConsciousnessIntegrationSystem
        print("✅ Consciousness Integration System: OK")
    except ImportError as e:
        print(f"❌ Consciousness Integration System: {e}")
    
    print("\n🎯 Import test completed!")

def test_basic_functionality():
    """Test basic functionality of the systems"""
    print("\n🧪 Testing basic functionality...")
    
    # Test Sentient AI Generator
    try:
        from sentient_ai_generator import SentientAIGenerator
        generator = SentientAIGenerator()
        print("✅ Sentient AI Generator: Initialized successfully")
    except Exception as e:
        print(f"❌ Sentient AI Generator: {e}")
    
    # Test Multiverse Testing System
    try:
        from multiverse_testing_system import MultiverseTestingSystem
        system = MultiverseTestingSystem()
        print("✅ Multiverse Testing System: Initialized successfully")
    except Exception as e:
        print(f"❌ Multiverse Testing System: {e}")
    
    # Test Quantum AI Consciousness
    try:
        from quantum_ai_consciousness import QuantumAIConsciousnessSystem
        consciousness = QuantumAIConsciousnessSystem()
        print("✅ Quantum AI Consciousness: Initialized successfully")
    except Exception as e:
        print(f"❌ Quantum AI Consciousness: {e}")
    
    print("\n🎯 Basic functionality test completed!")

def test_demo_system():
    """Test the demo system"""
    print("\n🧪 Testing demo system...")
    
    try:
        from demo_ultimate_breakthrough_system import demonstrate_ultimate_breakthrough_system
        print("✅ Demo system: Imported successfully")
        
        # Run a quick test
        print("Running demo system...")
        demonstrate_ultimate_breakthrough_system()
        print("✅ Demo system: Executed successfully")
        
    except Exception as e:
        print(f"❌ Demo system: {e}")
    
    print("\n🎯 Demo system test completed!")

if __name__ == "__main__":
    print("🚀 SYSTEM FIX TEST")
    print("=" * 50)
    
    test_imports()
    test_basic_functionality()
    test_demo_system()
    
    print("\n🎉 ALL TESTS COMPLETED!")
    print("=" * 50)
