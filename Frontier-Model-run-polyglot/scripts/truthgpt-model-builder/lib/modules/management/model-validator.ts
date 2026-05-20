/**
 * Enhanced Model Validator - Validates model specifications and descriptions
 */

import { ModelSpec } from './model-analyzer'

export interface ValidationResult {
  isValid: boolean
  errors: string[]
  warnings: string[]
  suggestions: string[]
}

/**
 * Validates a model specification
 */
export function validateModelSpec(spec: ModelSpec): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []
  const suggestions: string[] = []

  // Validate layers
  if (!spec.layers || spec.layers.length === 0) {
    errors.push('Model must have at least one layer')
  } else {
    if (spec.layers.length < 2) {
      warnings.push('Model has only one layer, consider adding more layers for better performance')
    }
    if (spec.layers.length > 10) {
      warnings.push('Model has many layers, this may cause overfitting or slow training')
      suggestions.push('Consider using dropout or reducing the number of layers')
    }
    
    // Check layer sizes
    spec.layers.forEach((size, idx) => {
      if (size < 1) {
        errors.push(`Layer ${idx + 1} has invalid size: ${size}`)
      }
      if (size > 10000) {
        warnings.push(`Layer ${idx + 1} is very large (${size}), this may cause memory issues`)
      }
    })
    
    // Check for decreasing sizes (common pattern)
    for (let i = 1; i < spec.layers.length; i++) {
      if (spec.layers[i] > spec.layers[i - 1] * 2) {
        warnings.push(`Layer ${i + 1} is much larger than previous layer, this may cause issues`)
      }
    }
  }

  // Validate learning rate
  if (spec.learningRate <= 0) {
    errors.push('Learning rate must be positive')
  } else if (spec.learningRate > 1) {
    errors.push('Learning rate is too high (should be < 1)')
  } else if (spec.learningRate < 0.00001) {
    warnings.push('Learning rate is very small, training may be very slow')
  } else if (spec.learningRate > 0.1) {
    warnings.push('Learning rate is high, may cause training instability')
    suggestions.push('Consider using a lower learning rate or learning rate scheduling')
  }

  // Validate batch size
  if (spec.batchSize < 1) {
    errors.push('Batch size must be at least 1')
  } else if (spec.batchSize > 1024) {
    warnings.push('Batch size is very large, may cause memory issues')
  } else if (spec.batchSize === 1) {
    warnings.push('Batch size of 1 may cause unstable gradients')
    suggestions.push('Consider using a larger batch size if possible')
  }

  // Validate epochs
  if (spec.epochs < 1) {
    errors.push('Epochs must be at least 1')
  } else if (spec.epochs > 1000) {
    warnings.push('Very high number of epochs, training may take a very long time')
    suggestions.push('Consider using early stopping')
  } else if (spec.epochs < 5) {
    warnings.push('Few epochs may not be enough for the model to learn')
  }

  // Validate dropout
  if (spec.useDropout) {
    if (spec.dropoutRate < 0 || spec.dropoutRate >= 1) {
      errors.push('Dropout rate must be between 0 and 1')
    } else if (spec.dropoutRate > 0.5) {
      warnings.push('Very high dropout rate may prevent the model from learning')
    }
  }

  // Validate architecture-specific settings
  if (spec.architecture === 'cnn' && spec.type !== 'vision') {
    warnings.push('CNN architecture is typically used for vision tasks')
  }
  
  if (spec.architecture === 'lstm' && spec.type !== 'nlp' && spec.type !== 'time-series') {
    warnings.push('LSTM architecture is typically used for sequence tasks')
  }

  // Validate optimizer
  const validOptimizers = ['adam', 'sgd', 'rmsprop', 'adamw']
  if (!validOptimizers.includes(spec.optimizer)) {
    errors.push(`Invalid optimizer: ${spec.optimizer}`)
  }

  // Validate loss function
  const validLosses = [
    'sparse_categorical_crossentropy',
    'categorical_crossentropy',
    'binary_crossentropy',
    'mean_squared_error',
    'mean_absolute_error',
  ]
  if (!validLosses.includes(spec.loss)) {
    warnings.push(`Loss function ${spec.loss} may not be standard`)
  }

  // Type-specific validations
  if (spec.type === 'classification') {
    if (spec.outputActivation !== 'softmax' && spec.outputActivation !== 'sigmoid') {
      warnings.push('Classification models typically use softmax or sigmoid output activation')
    }
  }

  if (spec.type === 'regression') {
    if (spec.outputActivation !== 'linear') {
      warnings.push('Regression models typically use linear output activation')
    }
    if (spec.loss !== 'mean_squared_error' && spec.loss !== 'mean_absolute_error') {
      warnings.push('Regression models typically use MSE or MAE loss')
    }
  }

  // Generate suggestions based on type
  if (spec.type === 'nlp' && spec.architecture === 'dense') {
    suggestions.push('Consider using LSTM or Transformer architecture for NLP tasks')
  }

  if (spec.type === 'vision' && spec.architecture !== 'cnn') {
    suggestions.push('Consider using CNN architecture for vision tasks')
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    suggestions,
  }
}

/**
 * Validates a model description
 */
export function validateDescription(desc: string): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []
  const suggestions: string[] = []

  if (!desc || desc.trim().length === 0) {
    errors.push('Description cannot be empty')
    return { isValid: false, errors, warnings, suggestions }
  }

  if (desc.length < 10) {
    errors.push('Description is too short (minimum 10 characters)')
    suggestions.push('Provide more details about what the model should do')
  }

  if (desc.length > 1000) {
    warnings.push('Description is very long, may contain unnecessary information')
  }

  // Check for common issues
  const lowerDesc = desc.toLowerCase()
  
  if (!lowerDesc.includes('model') && !lowerDesc.includes('modelo')) {
    if (!lowerDesc.includes('predict') && !lowerDesc.includes('predecir') &&
        !lowerDesc.includes('classify') && !lowerDesc.includes('clasificar') &&
        !lowerDesc.includes('detect') && !lowerDesc.includes('detectar')) {
      warnings.push('Description may not clearly indicate what the model should do')
      suggestions.push('Try to be more specific about the model\'s purpose')
    }
  }

  // Check for vague terms
  const vagueTerms = ['something', 'algo', 'thing', 'cosa', 'good', 'bueno']
  if (vagueTerms.some(term => lowerDesc.includes(term))) {
    warnings.push('Description contains vague terms')
    suggestions.push('Be more specific about what the model should do')
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    suggestions,
  }
}


