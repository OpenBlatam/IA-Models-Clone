/**
 * Hook para monitoreo y optimización de rendimiento de modelos
 * ==============================================================
 */

import { useState, useCallback, useRef, useEffect } from 'react'

export interface ModelPerformanceMetrics {
  creationTime?: number
  pollingTime?: number
  totalTime?: number
  apiCalls: number
  apiErrors: number
  cacheHits: number
  cacheMisses: number
}

export interface UseModelPerformanceResult {
  metrics: ModelPerformanceMetrics
  startTimer: (operation: 'creation' | 'polling') => () => void
  recordApiCall: (success: boolean) => void
  recordCacheHit: (hit: boolean) => void
  resetMetrics: () => void
  getAverageTime: () => number
}

/**
 * Hook para monitorear y optimizar rendimiento de operaciones de modelos
 */
export function useModelPerformance(): UseModelPerformanceResult {
  const [metrics, setMetrics] = useState<ModelPerformanceMetrics>({
    apiCalls: 0,
    apiErrors: 0,
    cacheHits: 0,
    cacheMisses: 0,
  })

  const timersRef = useRef<{
    creation?: { start: number; end?: number }
    polling?: { start: number; end?: number }
  }>({})

  const startTimer = useCallback((operation: 'creation' | 'polling') => {
    const startTime = Date.now()
    timersRef.current[operation] = { start: startTime }

    return () => {
      const endTime = Date.now()
      const duration = endTime - startTime

      setMetrics(prev => {
        const updated = { ...prev }
        
        if (operation === 'creation') {
          updated.creationTime = duration
          if (updated.pollingTime) {
            updated.totalTime = duration + updated.pollingTime
          }
        } else if (operation === 'polling') {
          updated.pollingTime = duration
          if (updated.creationTime) {
            updated.totalTime = updated.creationTime + duration
          }
        }

        return updated
      })

      timersRef.current[operation] = {
        ...timersRef.current[operation]!,
        end: endTime,
      }
    }
  }, [])

  const recordApiCall = useCallback((success: boolean) => {
    setMetrics(prev => ({
      ...prev,
      apiCalls: prev.apiCalls + 1,
      apiErrors: success ? prev.apiErrors : prev.apiErrors + 1,
    }))
  }, [])

  const recordCacheHit = useCallback((hit: boolean) => {
    setMetrics(prev => ({
      ...prev,
      cacheHits: hit ? prev.cacheHits + 1 : prev.cacheHits,
      cacheMisses: hit ? prev.cacheMisses : prev.cacheMisses + 1,
    }))
  }, [])

  const resetMetrics = useCallback(() => {
    setMetrics({
      apiCalls: 0,
      apiErrors: 0,
      cacheHits: 0,
      cacheMisses: 0,
    })
    timersRef.current = {}
  }, [])

  const getAverageTime = useCallback(() => {
    if (!metrics.creationTime) return 0
    return metrics.totalTime || metrics.creationTime
  }, [metrics])

  return {
    metrics,
    startTimer,
    recordApiCall,
    recordCacheHit,
    resetMetrics,
    getAverageTime,
  }
}

