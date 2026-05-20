/**
 * Hook para optimización automática de modelos
 * =============================================
 */

import { useCallback, useMemo } from 'react'

export interface ModelOptimizationSuggestions {
  layers?: {
    suggestion: string
    reason: string
    impact: 'low' | 'medium' | 'high'
  }[]
  optimizer?: {
    suggestion: string
    reason: string
    impact: 'low' | 'medium' | 'high'
  }
  loss?: {
    suggestion: string
    reason: string
    impact: 'low' | 'medium' | 'high'
  }
  hyperparameters?: {
    parameter: string
    current?: any
    suggested: any
    reason: string
  }[]
  warnings?: string[]
  recommendations?: string[]
}

export interface UseModelOptimizerResult {
  optimizeModel: (description: string, currentSpec?: any) => ModelOptimizationSuggestions
  getComplexityEstimate: (description: string) => {
    complexity: 'low' | 'medium' | 'high'
    estimatedParams: number
    estimatedTime: number // minutos
    estimatedMemory: number // MB
  }
}

/**
 * Hook para optimización automática de modelos
 */
export function useModelOptimizer(): UseModelOptimizerResult {
  const optimizeModel = useCallback((
    description: string,
    currentSpec?: any
  ): ModelOptimizationSuggestions => {
    const lowerDesc = description.toLowerCase()
    const suggestions: ModelOptimizationSuggestions = {
      warnings: [],
      recommendations: []
    }

    // Análisis de complejidad
    const isComplex = lowerDesc.includes('deep') || 
                     lowerDesc.includes('many layers') ||
                     lowerDesc.includes('residual') ||
                     lowerDesc.includes('attention')

    // Sugerencias de capas
    if (lowerDesc.includes('cnn') || lowerDesc.includes('convolutional') || lowerDesc.includes('image')) {
      if (!currentSpec?.layers?.some((l: any) => l.type === 'dropout')) {
        suggestions.layers = [
          {
            suggestion: 'Agregar Dropout después de capas convolucionales',
            reason: 'Previene overfitting en CNNs',
            impact: 'high'
          }
        ]
      }

      if (!currentSpec?.layers?.some((l: any) => l.type === 'batchnormalization')) {
        suggestions.layers?.push({
          suggestion: 'Considerar BatchNormalization',
          reason: 'Mejora la estabilidad del entrenamiento',
          impact: 'medium'
        })
      }
    }

    // Sugerencias de optimizador
    if (isComplex) {
      suggestions.optimizer = {
        suggestion: 'Adam con learning rate decay',
        reason: 'Mejor rendimiento en modelos complejos',
        impact: 'high'
      }
    } else {
      suggestions.optimizer = {
        suggestion: 'Adam',
        reason: 'Optimizador robusto para la mayoría de casos',
        impact: 'medium'
      }
    }

    // Sugerencias de función de pérdida
    if (lowerDesc.includes('classification') || lowerDesc.includes('classify')) {
      const numClasses = lowerDesc.match(/\b(\d+)\s*(?:class|category)/i)?.[1]
      if (numClasses && parseInt(numClasses) > 2) {
        suggestions.loss = {
          suggestion: 'SparseCategoricalCrossentropy',
          reason: 'Apropiado para clasificación multi-clase',
          impact: 'high'
        }
      } else {
        suggestions.loss = {
          suggestion: 'BinaryCrossentropy',
          reason: 'Apropiado para clasificación binaria',
          impact: 'high'
        }
      }
    } else if (lowerDesc.includes('regression') || lowerDesc.includes('predict')) {
      suggestions.loss = {
        suggestion: 'MeanSquaredError',
        reason: 'Apropiado para problemas de regresión',
        impact: 'high'
      }
    }

    // Sugerencias de hiperparámetros
    suggestions.hyperparameters = []
    
    if (isComplex && (!currentSpec?.batch_size || currentSpec.batch_size > 32)) {
      suggestions.hyperparameters.push({
        parameter: 'batch_size',
        current: currentSpec?.batch_size,
        suggested: 32,
        reason: 'Batch size más pequeño ayuda con modelos complejos'
      })
    }

    if (!currentSpec?.learning_rate) {
      suggestions.hyperparameters.push({
        parameter: 'learning_rate',
        suggested: isComplex ? 0.0001 : 0.001,
        reason: 'Learning rate inicial recomendado'
      })
    }

    // Warnings
    if (lowerDesc.length < 20) {
      suggestions.warnings?.push('La descripción es muy corta. Proporciona más detalles para mejores sugerencias.')
    }

    if (isComplex && !suggestions.layers?.some(l => l.suggestion.includes('Dropout'))) {
      suggestions.warnings?.push('Modelos complejos se benefician de regularización (Dropout)')
    }

    // Recomendaciones generales
    suggestions.recommendations = [
      'Usa early stopping para evitar overfitting',
      'Considera data augmentation para mejorar generalización',
      'Monitorea las métricas de validación durante el entrenamiento'
    ]

    return suggestions
  }, [])

  const getComplexityEstimate = useCallback((description: string) => {
    const lowerDesc = description.toLowerCase()
    let complexity: 'low' | 'medium' | 'high' = 'low'
    let estimatedParams = 10000
    let estimatedTime = 5 // minutos
    let estimatedMemory = 100 // MB

    // Determinar complejidad
    if (lowerDesc.includes('deep') || 
        lowerDesc.includes('many layers') ||
        lowerDesc.includes('residual') ||
        lowerDesc.includes('attention') ||
        lowerDesc.includes('transformer')) {
      complexity = 'high'
      estimatedParams = 1000000
      estimatedTime = 30
      estimatedMemory = 500
    } else if (lowerDesc.includes('cnn') || 
               lowerDesc.includes('lstm') ||
               lowerDesc.includes('gru')) {
      complexity = 'medium'
      estimatedParams = 100000
      estimatedTime = 15
      estimatedMemory = 250
    }

    // Ajustar según tamaño de datos
    if (lowerDesc.includes('large') || lowerDesc.includes('big')) {
      estimatedTime *= 2
      estimatedMemory *= 2
    }

    return {
      complexity,
      estimatedParams,
      estimatedTime,
      estimatedMemory
    }
  }, [])

  return {
    optimizeModel,
    getComplexityEstimate
  }
}

