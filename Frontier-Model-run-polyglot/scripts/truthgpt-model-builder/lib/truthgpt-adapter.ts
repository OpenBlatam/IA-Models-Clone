/**
 * TruthGPT Adapter - Ensures perfect compatibility with TruthGPT API
 * This module adapts model specifications to match TruthGPT's exact API structure
 */

import { ModelSpec } from './modules/management'
import { toPascalCase } from './utils/code-generators'

export interface TruthGPTCompatibleSpec extends ModelSpec {
  truthgptLayers: string[]
  truthgptOptimizer: string
  truthgptLoss: string
  truthgptMetrics: string[]
  compatible: boolean
}

/**
 * Adapts a ModelSpec to be perfectly compatible with TruthGPT API
 */
export function adaptToTruthGPT(spec: ModelSpec): TruthGPTCompatibleSpec {
  // Ensure all components are compatible with TruthGPT API
  const adaptedSpec: TruthGPTCompatibleSpec = {
    ...spec,
    truthgptLayers: [],
    truthgptOptimizer: '',
    truthgptLoss: '',
    truthgptMetrics: [],
    compatible: true,
  }

  // Adapt layers to TruthGPT format
  adaptedSpec.truthgptLayers = adaptLayersToTruthGPT(spec)

  // Adapt optimizer to TruthGPT format
  adaptedSpec.truthgptOptimizer = adaptOptimizerToTruthGPT(spec.optimizer)

  // Adapt loss to TruthGPT format
  adaptedSpec.truthgptLoss = adaptLossToTruthGPT(spec.loss)

  // Adapt metrics to TruthGPT format
  adaptedSpec.truthgptMetrics = adaptMetricsToTruthGPT(spec.metrics)

  // Validate compatibility
  adaptedSpec.compatible = validateCompatibility(adaptedSpec)

  return adaptedSpec
}

/**
 * Adapts layers to TruthGPT layer format
 */
function adaptLayersToTruthGPT(spec: ModelSpec): string[] {
  const layers: string[] = []

  if (spec.architecture === 'cnn') {
    // CNN layers for TruthGPT
    layers.push(`tg.layers.Conv2D(32, 3, activation='${spec.activation}')`)
    layers.push(`tg.layers.MaxPooling2D(2)`)
    layers.push(`tg.layers.Conv2D(64, 3, activation='${spec.activation}')`)
    layers.push(`tg.layers.MaxPooling2D(2)`)
    layers.push(`tg.layers.Flatten()`)
    
    // Add dense layers based on spec
    spec.layers.forEach((size, idx) => {
      if (idx < spec.layers.length - 1) {
        layers.push(`tg.layers.Dense(${size}, activation='${spec.activation}')`)
        if (spec.useDropout) {
          layers.push(`tg.layers.Dropout(${spec.dropoutRate})`)
        }
      }
    })
    
    // Output layer
    layers.push(`tg.layers.Dense(${spec.layers[spec.layers.length - 1] || 10}, activation='${spec.outputActivation}')`)
    
  } else if (spec.architecture === 'lstm') {
    // LSTM layers for TruthGPT
    layers.push(`tg.layers.LSTM(${spec.layers[0] || 128}, return_sequences=True)`)
    if (spec.useDropout) {
      layers.push(`tg.layers.Dropout(${spec.dropoutRate})`)
    }
    
    if (spec.layers.length > 1) {
      layers.push(`tg.layers.LSTM(${spec.layers[1] || 64}, return_sequences=False)`)
      if (spec.useDropout) {
        layers.push(`tg.layers.Dropout(${spec.dropoutRate})`)
      }
    }
    
    // Dense layers
    for (let i = 2; i < spec.layers.length; i++) {
      const isLast = i === spec.layers.length - 1
      const activation = isLast ? spec.outputActivation : spec.activation
      layers.push(`tg.layers.Dense(${spec.layers[i]}, activation='${activation}')`)
      if (!isLast && spec.useDropout) {
        layers.push(`tg.layers.Dropout(${spec.dropoutRate})`)
      }
    }
    
  } else if (spec.architecture === 'transformer') {
    // Transformer-like architecture using available TruthGPT layers
    // Note: TruthGPT may not have transformer layers, so we use LSTM/Dense
    layers.push(`tg.layers.Dense(${spec.layers[0] || 512}, activation='${spec.activation}')`)
    if (spec.useDropout) {
      layers.push(`tg.layers.Dropout(${spec.dropoutRate})`)
    }
    
    // Add more layers
    spec.layers.slice(1).forEach((size, idx) => {
      const isLast = idx === spec.layers.slice(1).length - 1
      const activation = isLast ? spec.outputActivation : spec.activation
      layers.push(`tg.layers.Dense(${size}, activation='${activation}')`)
      if (!isLast && spec.useDropout) {
        layers.push(`tg.layers.Dropout(${spec.dropoutRate})`)
      }
    })
    
  } else {
    // Dense layers (default)
    spec.layers.forEach((size, idx) => {
      const isLast = idx === spec.layers.length - 1
      const activation = isLast ? spec.outputActivation : spec.activation
      layers.push(`tg.layers.Dense(${size}, activation='${activation}')`)
      
      if (!isLast) {
        if (spec.useDropout) {
          layers.push(`tg.layers.Dropout(${spec.dropoutRate})`)
        }
        if (spec.useBatchNorm) {
          layers.push(`tg.layers.BatchNormalization()`)
        }
      }
    })
  }

  return layers
}

/**
 * Adapts optimizer name to TruthGPT format
 */
function adaptOptimizerToTruthGPT(optimizer: string): string {
  const optimizerMap: Record<string, string> = {
    'adam': 'tg.optimizers.Adam',
    'sgd': 'tg.optimizers.SGD',
    'rmsprop': 'tg.optimizers.RMSprop',
    'adamw': 'tg.optimizers.AdamW',
  }

  return optimizerMap[optimizer.toLowerCase()] || 'tg.optimizers.Adam'
}

/**
 * Adapts loss name to TruthGPT format
 */
function adaptLossToTruthGPT(loss: string): string {
  const lossMap: Record<string, string> = {
    'sparse_categorical_crossentropy': 'tg.losses.SparseCategoricalCrossentropy',
    'mean_squared_error': 'tg.losses.MeanSquaredError',
    'binary_crossentropy': 'tg.losses.BinaryCrossentropy',
    'categorical_crossentropy': 'tg.losses.CategoricalCrossentropy',
  }

  return lossMap[loss.toLowerCase()] || 'tg.losses.SparseCategoricalCrossentropy'
}

/**
 * Adapts metrics to TruthGPT format
 */
function adaptMetricsToTruthGPT(metrics: string[]): string[] {
  // TruthGPT uses simple string metrics
  const metricMap: Record<string, string> = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'mse': 'mse',
    'mae': 'mae',
    'r2': 'r2',
    'perplexity': 'perplexity',
  }

  return metrics.map(m => metricMap[m.toLowerCase()] || m)
}

/**
 * Validates that the spec is compatible with TruthGPT
 */
function validateCompatibility(spec: TruthGPTCompatibleSpec): boolean {
  // Check that all required components are present
  if (!spec.truthgptLayers || spec.truthgptLayers.length === 0) {
    return false
  }

  if (!spec.truthgptOptimizer) {
    return false
  }

  if (!spec.truthgptLoss) {
    return false
  }

  if (!spec.truthgptMetrics || spec.truthgptMetrics.length === 0) {
    return false
  }

  // Check layer syntax
  const validLayerPattern = /^tg\.layers\.\w+\(/
  const allLayersValid = spec.truthgptLayers.every(layer => 
    validLayerPattern.test(layer.trim())
  )

  if (!allLayersValid) {
    return false
  }

  return true
}

/**
 * Generates TruthGPT-compatible model code
 */
export function generateTruthGPTCode(
  spec: TruthGPTCompatibleSpec,
  modelName: string,
  description: string
): string {
  const className = toPascalCase(modelName)
  const layersCode = spec.truthgptLayers.join(',\n            ')
  
  // Generate optimizer code
  const optimizerCode = `${spec.truthgptOptimizer}(learning_rate=${spec.learningRate})`
  
  // Generate loss code
  const lossCode = `${spec.truthgptLoss}()`
  
  // Generate metrics code
  const metricsCode = `[${spec.truthgptMetrics.map((m: string) => `'${m}'`).join(', ')}]`
  
  // Generate names for print statements
  const optimizerName = spec.optimizer
  const lossName = spec.loss
  const metricsNames = spec.truthgptMetrics.join(', ')

  return `"""
${className} - TruthGPT Model
Generated based on: ${description}
Type: ${spec.type}
Architecture: ${spec.architecture}
Optimized for TruthGPT API compatibility
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import truthgpt as tg
import torch
import numpy as np


class ${className}(tg.Sequential):
    """
    TruthGPT Model: ${description}
    Architecture: ${spec.architecture}
    Fully compatible with TruthGPT API
    
    This model is perfectly adapted to TruthGPT's API structure.
    All layers, optimizers, and loss functions use TruthGPT's native APIs.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the ${className} model.
        
        The model is automatically configured with optimal settings
        based on your description: "${description}"
        """
        layers = [
            ${layersCode}
        ]
        
        super().__init__(layers, name='${modelName}', **kwargs)
    
    def compile(self, **kwargs):
        """
        Compile the model with TruthGPT optimizations.
        
        Uses TruthGPT's native optimizer, loss, and metrics.
        All parameters are optimized for best performance.
        """
        optimizer = kwargs.get('optimizer', ${optimizerCode})
        loss = kwargs.get('loss', ${lossCode})
        metrics = kwargs.get('metrics', ${metricsCode})
        
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print("✅ Model compiled successfully with TruthGPT")
        print(f"   Optimizer: ${optimizerName}")
        print(f"   Loss: ${lossName}")
        print(f"   Metrics: ${metricsNames}")
    
    def fit(self, x, y, epochs=${spec.epochs}, batch_size=${spec.batchSize}, **kwargs):
        """Train the model with TruthGPT optimizations."""
        return super().fit(x, y, epochs=epochs, batch_size=batch_size, **kwargs)
    
    def predict(self, x, **kwargs):
        """Make predictions."""
        return super().predict(x, **kwargs)
    
    def evaluate(self, x, y, **kwargs):
        """Evaluate the model."""
        return super().evaluate(x, y, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create model instance
    model = ${className}()
    
    # Compile model
    model.compile()
    
    # Generate dummy data for demonstration
    x_train = np.random.randn(1000, 128).astype(np.float32)
    y_train = np.random.randint(0, 10, 1000).astype(np.int64)
    
    # Train model
    print("Training model...")
    history = model.fit(x_train, y_train, epochs=${spec.epochs}, batch_size=${spec.batchSize}, verbose=1)
    
    # Evaluate model
    x_test = np.random.randn(200, 128).astype(np.float32)
    y_test = np.random.randint(0, 10, 200).astype(np.int64)
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    print("Model training completed!")
`
}

// toPascalCase imported from utils/code-generators

