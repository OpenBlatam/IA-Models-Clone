/**
 * TruthGPT Integrator - Direct integration with TruthGPT core modules
 */

import { exec } from 'child_process'
import { promisify } from 'util'
import * as fs from 'fs/promises'
import * as path from 'path'
import { toPascalCase } from './utils/code-generators'

const execAsync = promisify(exec)

interface IntegrationConfig {
  optimizationLevel: 'basic' | 'enhanced' | 'advanced' | 'ultra'
  useTruthGPTCore: boolean
  applyOptimizations: boolean
}

export async function integrateWithTruthGPT(
  modelPath: string,
  modelName: string,
  config: IntegrationConfig
): Promise<void> {
  try {
    const truthgptPath = path.resolve(process.cwd(), '../TruthGPT-main')
    
    // Create integration file
    const integrationCode = generateIntegrationCode(modelName, config)
    await fs.writeFile(
      path.join(modelPath, 'truthgpt_integration.py'),
      integrationCode
    )

    // Add optimization imports
    if (config.applyOptimizations) {
      const optimizedCode = addOptimizations(modelPath, config)
      await fs.appendFile(
        path.join(modelPath, `${modelName}.py`),
        optimizedCode
      )
    }

    // Create test file
    const testCode = generateTestCode(modelName)
    await fs.writeFile(
      path.join(modelPath, 'test_model.py'),
      testCode
    )

    console.log(`TruthGPT integration completed for ${modelName}`)
  } catch (error) {
    console.error('Error integrating with TruthGPT:', error)
    throw error
  }
}

function generateIntegrationCode(modelName: string, config: IntegrationConfig): string {
  return `"""
TruthGPT Integration for ${modelName}
Automatically generated integration with TruthGPT optimization core
"""

import sys
import os

# Add TruthGPT core to path
truthgpt_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, truthgpt_root)

try:
    from core.optimization import OptimizationEngine, OptimizationLevel
    from core.models import ModelManager
    from core.monitoring import PerformanceMonitor
    
    TRUTHGPT_AVAILABLE = True
except ImportError:
    TRUTHGPT_AVAILABLE = False
    print("Warning: TruthGPT core modules not found. Using basic optimization.")

class TruthGPTIntegration:
    """
    Integration layer for TruthGPT optimization features
    """
    
    def __init__(self, model, optimization_level='${config.optimizationLevel}'):
        self.model = model
        self.optimization_level = optimization_level
        self.optimizer = None
        self.monitor = None
        
        if TRUTHGPT_AVAILABLE:
            self._initialize_truthgpt()
    
    def _initialize_truthgpt(self):
        """Initialize TruthGPT components"""
        try:
            from core.optimization import OptimizationLevel
            
            level_map = {
                'basic': OptimizationLevel.BASIC,
                'enhanced': OptimizationLevel.ENHANCED,
                'advanced': OptimizationLevel.ADVANCED,
                'ultra': OptimizationLevel.ULTRA,
            }
            
            opt_level = level_map.get(self.optimization_level, OptimizationLevel.ENHANCED)
            
            # Initialize optimization engine
            self.optimizer = OptimizationEngine(level=opt_level)
            
            # Initialize performance monitor
            self.monitor = PerformanceMonitor()
            
            print(f"✅ TruthGPT integration initialized (Level: {self.optimization_level})")
        except Exception as e:
            print(f"⚠️ Could not initialize TruthGPT: {e}")
    
    def optimize_model(self):
        """Apply TruthGPT optimizations"""
        if not self.optimizer or not TRUTHGPT_AVAILABLE:
            return self.model
        
        try:
            print("🔧 Applying TruthGPT optimizations...")
            optimized_model = self.optimizer.optimize(self.model)
            print("✅ Model optimized successfully")
            return optimized_model
        except Exception as e:
            print(f"⚠️ Optimization failed: {e}")
            return self.model
    
    def monitor_training(self, x_train, y_train):
        """Monitor training with TruthGPT"""
        if not self.monitor or not TRUTHGPT_AVAILABLE:
            return
        
        try:
            self.monitor.start_monitoring()
            # Training happens here
            metrics = self.monitor.get_metrics()
            print(f"📊 Training metrics: {metrics}")
        except Exception as e:
            print(f"⚠️ Monitoring failed: {e}")
    
    def get_performance_report(self):
        """Get performance report from TruthGPT"""
        if not self.monitor or not TRUTHGPT_AVAILABLE:
            return None
        
        try:
            return self.monitor.generate_report()
        except Exception as e:
            print(f"⚠️ Could not generate report: {e}")
            return None

# Usage example
if __name__ == "__main__":
    print("TruthGPT Integration Module")
    print("=" * 50)
    print("This module provides integration with TruthGPT optimization core.")
    print("Import this in your model to enable advanced optimizations.")
`
}

function addOptimizations(modelPath: string, config: IntegrationConfig): string {
  return `

# TruthGPT Optimizations
from truthgpt_integration import TruthGPTIntegration

# Initialize TruthGPT integration
truthgpt = TruthGPTIntegration(
    model=self,
    optimization_level='${config.optimizationLevel}'
)

# Apply optimizations
self = truthgpt.optimize_model()
`
}

function generateTestCode(modelName: string): string {
  return `"""
Test script for ${modelName}
Automatically generated test suite
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from ${modelName} import *

def test_model_creation():
    """Test model creation"""
    print("Testing model creation...")
    try:
        model = ${toPascalCase(modelName)}()
        print("✅ Model created successfully")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_model_compilation():
    """Test model compilation"""
    print("Testing model compilation...")
    try:
        import truthgpt as tg
        model = ${toPascalCase(modelName)}()
        model.compile(
            optimizer=tg.optimizers.Adam(learning_rate=0.001),
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        print("✅ Model compiled successfully")
        return True
    except Exception as e:
        print(f"❌ Model compilation failed: {e}")
        return False

def test_model_forward():
    """Test model forward pass"""
    print("Testing model forward pass...")
    try:
        model = ${toPascalCase(modelName)}()
        # Generate dummy input
        dummy_input = np.random.randn(1, 128).astype(np.float32)
        output = model.predict(dummy_input)
        print(f"✅ Forward pass successful. Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Running ${modelName} Test Suite")
    print("=" * 50)
    
    tests = [
        test_model_creation,
        test_model_compilation,
        test_model_forward,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Tests: {passed}/{total} passed")
    print("=" * 50)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
`

// toPascalCase imported from utils/code-generators

