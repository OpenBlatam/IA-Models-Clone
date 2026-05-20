/**
 * Model Optimizer - Advanced optimizations for model specifications
 */

import { ModelSpec } from './model-analyzer'

export interface OptimizationResult {
  optimized: ModelSpec
  improvements: string[]
  performance: {
    estimatedTrainingTime: number // in minutes
    estimatedMemoryUsage: number // in MB
    estimatedAccuracy: number // percentage
  }
}

/**
 * Optimizes a model specification for best performance
 */
export function optimizeModelSpec(spec: ModelSpec): OptimizationResult {
  const optimized = { ...spec }
  const improvements: string[] = []

  // Optimize layer sizes
  const originalLayers = [...optimized.layers]
  optimized.layers = optimizeLayers(optimized.layers, optimized.architecture)
  if (JSON.stringify(originalLayers) !== JSON.stringify(optimized.layers)) {
    improvements.push('Optimized layer sizes for better performance')
  }

  // Optimize learning rate
  const originalLR = optimized.learningRate
  optimized.learningRate = optimizeLearningRate(optimized.learningRate, optimized.architecture, optimized.type)
  if (originalLR !== optimized.learningRate) {
    improvements.push(`Adjusted learning rate from ${originalLR} to ${optimized.learningRate}`)
  }

  // Optimize batch size
  const originalBatch = optimized.batchSize
  optimized.batchSize = optimizeBatchSize(optimized.batchSize, optimized.architecture, optimized.layers)
  if (originalBatch !== optimized.batchSize) {
    improvements.push(`Optimized batch size from ${originalBatch} to ${optimized.batchSize}`)
  }

  // Optimize epochs
  const originalEpochs = optimized.epochs
  optimized.epochs = optimizeEpochs(optimized.epochs, optimized.type, optimized.complexity)
  if (originalEpochs !== optimized.epochs) {
    improvements.push(`Adjusted epochs from ${originalEpochs} to ${optimized.epochs}`)
  }

  // Optimize dropout
  if (optimized.useDropout) {
    const originalDropout = optimized.dropoutRate
    optimized.dropoutRate = optimizeDropout(optimized.dropoutRate, optimized.architecture, optimized.layers.length)
    if (originalDropout !== optimized.dropoutRate) {
      improvements.push(`Optimized dropout rate from ${originalDropout} to ${optimized.dropoutRate}`)
    }
  }

  // Calculate performance metrics
  const performance = estimatePerformance(optimized)

  return {
    optimized,
    improvements,
    performance,
  }
}

function optimizeLayers(layers: number[], architecture: string): number[] {
  const optimized = [...layers]

  // Ensure layers decrease in size (common pattern)
  for (let i = 1; i < optimized.length; i++) {
    if (optimized[i] > optimized[i - 1]) {
      optimized[i] = Math.floor(optimized[i - 1] * 0.8)
      if (optimized[i] < 32) optimized[i] = 32
    }
  }

  // Architecture-specific optimizations
  if (architecture === 'cnn') {
    // CNN layers typically decrease faster
    for (let i = 1; i < optimized.length; i++) {
      optimized[i] = Math.floor(optimized[i - 1] * 0.6)
      if (optimized[i] < 64) optimized[i] = 64
    }
  }

  return optimized
}

function optimizeLearningRate(lr: number, architecture: string, type: string): number {
  // Architecture-specific optimal ranges
  const optimalRanges: Record<string, [number, number]> = {
    transformer: [0.0001, 0.001],
    lstm: [0.001, 0.01],
    cnn: [0.001, 0.01],
    dense: [0.001, 0.01],
  }

  const [min, max] = optimalRanges[architecture] || [0.001, 0.01]

  if (lr < min) return min
  if (lr > max) return max

  // Type-specific adjustments
  if (type === 'generative' || type === 'nlp') {
    return Math.min(lr, 0.001)
  }

  return lr
}

function optimizeBatchSize(batchSize: number, architecture: string, layers: number[]): number {
  const maxLayerSize = Math.max(...layers)

  // Memory considerations
  if (maxLayerSize > 1000) {
    return Math.min(batchSize, 16)
  }
  if (maxLayerSize > 500) {
    return Math.min(batchSize, 32)
  }

  // Architecture-specific
  if (architecture === 'transformer') {
    return Math.min(batchSize, 16)
  }
  if (architecture === 'cnn') {
    return Math.min(batchSize, 32)
  }

  // Power of 2 for efficiency
  const powersOf2 = [1, 2, 4, 8, 16, 32, 64, 128]
  const optimal = powersOf2.find(p => p >= batchSize) || 32
  return Math.min(optimal, 128)
}

function optimizeEpochs(epochs: number, type: string, complexity?: any): number {
  // Type-specific optimal epochs
  const typeEpochs: Record<string, number> = {
    generative: 50,
    nlp: 30,
    vision: 20,
    classification: 15,
    regression: 20,
    'time-series': 25,
  }

  const optimal = typeEpochs[type] || 15

  if (complexity === 'simple') {
    return Math.min(epochs, optimal)
  } else if (complexity === 'complex' || complexity === 'very-complex') {
    return Math.max(epochs, optimal * 1.5)
  }

  return Math.max(epochs, optimal)
}

function optimizeDropout(dropout: number, architecture: string, numLayers: number): number {
  // More layers = higher dropout
  if (numLayers > 5) {
    return Math.max(dropout, 0.3)
  }

  // Architecture-specific
  if (architecture === 'transformer') {
    return 0.1
  }
  if (architecture === 'cnn') {
    return 0.3
  }

  return dropout
}

function estimatePerformance(spec: any): {
  estimatedTrainingTime: number
  estimatedMemoryUsage: number
  estimatedAccuracy: number
} {
  // Estimate training time (minutes)
  const baseTime = 5
  const layerFactor = spec.layers.length * 0.5
  const epochFactor = spec.epochs * 0.1
  const batchFactor = 64 / spec.batchSize
  const archFactor = spec.architecture === 'transformer' ? 2 : spec.architecture === 'cnn' ? 1.5 : 1
  
  const estimatedTrainingTime = baseTime * layerFactor * epochFactor * batchFactor * archFactor

  // Estimate memory usage (MB)
  const params = spec.layers.reduce((sum, size, idx) => {
    if (idx === 0) return sum
    return sum + spec.layers[idx - 1] * size
  }, 0)
  
  const estimatedMemoryUsage = params * 4 / (1024 * 1024) * spec.batchSize // 4 bytes per param

  // Estimate accuracy (percentage) - rough heuristic
  let estimatedAccuracy = 70
  if (spec.layers.length >= 3) estimatedAccuracy += 5
  if (spec.useDropout) estimatedAccuracy += 3
  if (spec.useBatchNorm) estimatedAccuracy += 2
  if (spec.epochs >= 20) estimatedAccuracy += 5
  if (spec.architecture === 'transformer') estimatedAccuracy += 5
  estimatedAccuracy = Math.min(estimatedAccuracy, 95)

  return {
    estimatedTrainingTime: Math.round(estimatedTrainingTime * 10) / 10,
    estimatedMemoryUsage: Math.round(estimatedMemoryUsage * 10) / 10,
    estimatedAccuracy: Math.round(estimatedAccuracy),
  }
}

