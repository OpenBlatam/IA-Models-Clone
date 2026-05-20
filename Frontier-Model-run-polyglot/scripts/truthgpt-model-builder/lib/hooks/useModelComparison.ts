/**
 * Hook para comparación de modelos
 * =================================
 */

import { useState, useCallback, useMemo } from 'react'

export interface ModelComparison {
  id: string
  modelIds: string[]
  metrics: {
    [key: string]: {
      [modelId: string]: number
    }
  }
  createdAt: number
}

export interface ModelMetrics {
  accuracy?: number
  loss?: number
  precision?: number
  recall?: number
  f1?: number
  trainingTime?: number
  inferenceTime?: number
  [key: string]: number | undefined
}

export interface UseModelComparisonResult {
  comparisons: ModelComparison[]
  createComparison: (modelIds: string[], metrics: Record<string, ModelMetrics>) => string
  getComparison: (id: string) => ModelComparison | undefined
  getBestModel: (comparisonId: string, metric: string) => string | null
  getComparisonSummary: (id: string) => {
    totalModels: number
    metrics: string[]
    bestModel: string | null
    averageScores: Record<string, number>
  }
  removeComparison: (id: string) => void
  clear: () => void
}

/**
 * Hook para comparar modelos y sus métricas
 */
export function useModelComparison(): UseModelComparisonResult {
  const [comparisons, setComparisons] = useState<ModelComparison[]>([])

  const createComparison = useCallback((
    modelIds: string[],
    metrics: Record<string, ModelMetrics>
  ): string => {
    const id = `comparison-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    
    // Convertir métricas al formato de comparación
    const comparisonMetrics: ModelComparison['metrics'] = {}
    
    Object.keys(metrics).forEach(metricName => {
      comparisonMetrics[metricName] = {}
      modelIds.forEach(modelId => {
        const value = metrics[metricName][modelId]
        if (value !== undefined) {
          comparisonMetrics[metricName][modelId] = value
        }
      })
    })

    const comparison: ModelComparison = {
      id,
      modelIds,
      metrics: comparisonMetrics,
      createdAt: Date.now()
    }

    setComparisons(prev => [...prev, comparison])
    return id
  }, [])

  const getComparison = useCallback((id: string) => {
    return comparisons.find(c => c.id === id)
  }, [comparisons])

  const getBestModel = useCallback((
    comparisonId: string,
    metric: string
  ): string | null => {
    const comparison = getComparison(comparisonId)
    if (!comparison) return null

    const metricValues = comparison.metrics[metric]
    if (!metricValues) return null

    // Encontrar el modelo con el mejor valor
    // Para métricas como accuracy, más alto es mejor
    // Para métricas como loss, más bajo es mejor
    const isHigherBetter = !metric.toLowerCase().includes('loss') && 
                          !metric.toLowerCase().includes('error')

    let bestModel: string | null = null
    let bestValue: number | null = null

    Object.entries(metricValues).forEach(([modelId, value]) => {
      if (bestValue === null) {
        bestModel = modelId
        bestValue = value
      } else if (isHigherBetter && value > bestValue) {
        bestModel = modelId
        bestValue = value
      } else if (!isHigherBetter && value < bestValue) {
        bestModel = modelId
        bestValue = value
      }
    })

    return bestModel
  }, [getComparison])

  const getComparisonSummary = useCallback((id: string) => {
    const comparison = getComparison(id)
    if (!comparison) {
      return {
        totalModels: 0,
        metrics: [],
        bestModel: null,
        averageScores: {}
      }
    }

    const metrics = Object.keys(comparison.metrics)
    const averageScores: Record<string, number> = {}

    metrics.forEach(metric => {
      const values = Object.values(comparison.metrics[metric])
      if (values.length > 0) {
        averageScores[metric] = values.reduce((a, b) => a + b, 0) / values.length
      }
    })

    // Determinar el mejor modelo general (basado en accuracy si existe)
    const bestModel = getBestModel(id, 'accuracy') || 
                     getBestModel(id, metrics[0]) ||
                     comparison.modelIds[0]

    return {
      totalModels: comparison.modelIds.length,
      metrics,
      bestModel,
      averageScores
    }
  }, [getComparison, getBestModel])

  const removeComparison = useCallback((id: string) => {
    setComparisons(prev => prev.filter(c => c.id !== id))
  }, [])

  const clear = useCallback(() => {
    setComparisons([])
  }, [])

  return {
    comparisons,
    createComparison,
    getComparison,
    getBestModel,
    getComparisonSummary,
    removeComparison,
    clear
  }
}

