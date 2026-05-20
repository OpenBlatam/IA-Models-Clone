/**
 * Hook para analytics y métricas avanzadas de modelos
 * ====================================================
 */

import { useState, useCallback, useRef, useEffect } from 'react'

export interface ModelAnalytics {
  totalCreated: number
  totalCompleted: number
  totalFailed: number
  averageCreationTime: number
  averagePollingTime: number
  successRate: number
  errorRate: number
  mostCommonErrors: Map<string, number>
  timeSeries: Array<{
    timestamp: number
    event: 'created' | 'completed' | 'failed'
    duration?: number
  }>
}

export interface UseModelAnalyticsResult {
  analytics: ModelAnalytics
  recordCreation: (modelId: string) => void
  recordCompletion: (modelId: string, duration: number) => void
  recordFailure: (modelId: string, error: Error, duration: number) => void
  recordError: (error: Error) => void
  getStats: () => {
    successRate: number
    averageTime: number
    totalOperations: number
    errorBreakdown: Array<{ error: string; count: number }>
  }
  reset: () => void
  exportData: () => string
}

/**
 * Hook para tracking y analytics de modelos
 */
export function useModelAnalytics(): UseModelAnalyticsResult {
  const [analytics, setAnalytics] = useState<ModelAnalytics>({
    totalCreated: 0,
    totalCompleted: 0,
    totalFailed: 0,
    averageCreationTime: 0,
    averagePollingTime: 0,
    successRate: 0,
    errorRate: 0,
    mostCommonErrors: new Map(),
    timeSeries: []
  })

  const creationTimes = useRef<number[]>([])
  const pollingTimes = useRef<number[]>([])
  const creationStartTimes = useRef<Map<string, number>>(new Map())

  const recordCreation = useCallback((modelId: string) => {
    const startTime = Date.now()
    creationStartTimes.current.set(modelId, startTime)

    setAnalytics(prev => ({
      ...prev,
      totalCreated: prev.totalCreated + 1,
      timeSeries: [
        ...prev.timeSeries,
        { timestamp: startTime, event: 'created' }
      ].slice(-1000) // Mantener últimas 1000 entradas
    }))
  }, [])

  const recordCompletion = useCallback((modelId: string, duration: number) => {
    const startTime = creationStartTimes.current.get(modelId)
    const totalDuration = startTime ? Date.now() - startTime : duration

    creationTimes.current.push(totalDuration)
    creationStartTimes.current.delete(modelId)

    setAnalytics(prev => {
      const newCreationTimes = [...creationTimes.current]
      const avgCreationTime = newCreationTimes.length > 0
        ? newCreationTimes.reduce((a, b) => a + b, 0) / newCreationTimes.length
        : 0

      const total = prev.totalCreated
      const completed = prev.totalCompleted + 1
      const successRate = total > 0 ? (completed / total) * 100 : 0

      return {
        ...prev,
        totalCompleted: completed,
        averageCreationTime: avgCreationTime,
        successRate,
        errorRate: total > 0 ? ((prev.totalFailed / total) * 100) : 0,
        timeSeries: [
          ...prev.timeSeries,
          { timestamp: Date.now(), event: 'completed', duration: totalDuration }
        ].slice(-1000)
      }
    })
  }, [])

  const recordFailure = useCallback((modelId: string, error: Error, duration: number) => {
    const startTime = creationStartTimes.current.get(modelId)
    const totalDuration = startTime ? Date.now() - startTime : duration

    creationTimes.current.push(totalDuration)
    creationStartTimes.current.delete(modelId)

    setAnalytics(prev => {
      const newErrors = new Map(prev.mostCommonErrors)
      const errorKey = error.message || 'Unknown error'
      newErrors.set(errorKey, (newErrors.get(errorKey) || 0) + 1)

      const total = prev.totalCreated
      const failed = prev.totalFailed + 1

      return {
        ...prev,
        totalFailed: failed,
        mostCommonErrors: newErrors,
        errorRate: total > 0 ? ((failed / total) * 100) : 0,
        successRate: total > 0 ? ((prev.totalCompleted / total) * 100) : 0,
        timeSeries: [
          ...prev.timeSeries,
          { timestamp: Date.now(), event: 'failed', duration: totalDuration }
        ].slice(-1000)
      }
    })
  }, [])

  const recordError = useCallback((error: Error) => {
    setAnalytics(prev => {
      const newErrors = new Map(prev.mostCommonErrors)
      const errorKey = error.message || 'Unknown error'
      newErrors.set(errorKey, (newErrors.get(errorKey) || 0) + 1)

      return {
        ...prev,
        mostCommonErrors: newErrors
      }
    })
  }, [])

  const getStats = useCallback(() => {
    const total = analytics.totalCreated
    const successRate = total > 0 ? (analytics.totalCompleted / total) * 100 : 0
    const avgTime = creationTimes.current.length > 0
      ? creationTimes.current.reduce((a, b) => a + b, 0) / creationTimes.current.length
      : 0

    const errorBreakdown = Array.from(analytics.mostCommonErrors.entries())
      .map(([error, count]) => ({ error, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10)

    return {
      successRate,
      averageTime: avgTime,
      totalOperations: total,
      errorBreakdown
    }
  }, [analytics])

  const reset = useCallback(() => {
    setAnalytics({
      totalCreated: 0,
      totalCompleted: 0,
      totalFailed: 0,
      averageCreationTime: 0,
      averagePollingTime: 0,
      successRate: 0,
      errorRate: 0,
      mostCommonErrors: new Map(),
      timeSeries: []
    })
    creationTimes.current = []
    pollingTimes.current = []
    creationStartTimes.current.clear()
  }, [])

  const exportData = useCallback(() => {
    return JSON.stringify({
      analytics,
      stats: getStats(),
      timestamp: Date.now()
    }, null, 2)
  }, [analytics, getStats])

  return {
    analytics,
    recordCreation,
    recordCompletion,
    recordFailure,
    recordError,
    getStats,
    reset,
    exportData
  }
}

