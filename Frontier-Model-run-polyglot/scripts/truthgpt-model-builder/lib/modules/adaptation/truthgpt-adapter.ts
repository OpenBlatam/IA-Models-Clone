/**
 * TruthGPT Adapter - Ensures perfect compatibility with TruthGPT API
 * This module adapts model specifications to match TruthGPT's exact API structure
 */

import { ModelSpec } from '../management/model-analyzer'
import { toPascalCase } from '../../utils/code-generators'

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
  const adaptedSpec: TruthGPTCompatibleSpec = {
    ...spec,
    truthgptLayers: [],
    truthgptOptimizer: '',
    truthgptLoss: '',
    truthgptMetrics: [],
    compatible: true,
  }

  adaptedSpec.truthgptLayers = adaptLayersToTruthGPT(spec)
  adaptedSpec.truthgptOptimizer = adaptOptimizerToTruthGPT(spec.optimizer)
  adaptedSpec.truthgptLoss = adaptLossToTruthGPT(spec.loss)
  adaptedSpec.truthgptMetrics = adaptMetricsToTruthGPT(spec.metrics)
  adaptedSpec.compatible = validateCompatibility(adaptedSpec)

  return adaptedSpec
}

function adaptLayersToTruthGPT(spec: ModelSpec): string[] {
  return spec.layers.map((size, index) => {
    const layerType = index === 0 ? 'Input' : index === spec.layers.length - 1 ? 'Output' : 'Dense'
    return `${layerType}(${size})`
  })
}

function adaptOptimizerToTruthGPT(optimizer: string): string {
  const optimizerMap: Record<string, string> = {
    adam: 'Adam',
    sgd: 'SGD',
    rmsprop: 'RMSprop',
    adamw: 'AdamW',
  }
  return optimizerMap[optimizer.toLowerCase()] || 'Adam'
}

function adaptLossToTruthGPT(loss: string): string {
  return toPascalCase(loss)
}

function adaptMetricsToTruthGPT(metrics: string[]): string[] {
  return metrics.map(metric => toPascalCase(metric))
}

function validateCompatibility(spec: TruthGPTCompatibleSpec): boolean {
  return (
    spec.truthgptLayers.length > 0 &&
    spec.truthgptOptimizer !== '' &&
    spec.truthgptLoss !== '' &&
    spec.truthgptMetrics.length > 0
  )
}

