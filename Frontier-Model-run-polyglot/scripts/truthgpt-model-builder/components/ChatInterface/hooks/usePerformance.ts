/**
 * Custom hook for performance optimization
 * Handles caching, memoization, and performance metrics
 */

import { useState, useCallback, useMemo, useRef, useEffect } from 'react'

export interface PerformanceState {
  showPerformance: boolean
  performanceMetrics: Map<string, number>
  messageCache: Map<string, { content: string, timestamp: number }>
  cacheEnabled: boolean
  realTimeStats: boolean
  renderCount: number
  averageRenderTime: number
}

export interface PerformanceActions {
  setShowPerformance: (show: boolean) => void
  setCacheEnabled: (enabled: boolean) => void
  setRealTimeStats: (enabled: boolean) => void
  cacheMessage: (id: string, content: string) => void
  getCachedMessage: (id: string) => string | null
  clearCache: () => void
  recordMetric: (name: string, value: number) => void
  getMetric: (name: string) => number | undefined
  resetMetrics: () => void
  measureRender: (componentName: string) => () => void
}

const CACHE_TTL = 3600000 // 1 hour

export function usePerformance(): PerformanceState & PerformanceActions {
  const [showPerformance, setShowPerformance] = useState(false)
  const [performanceMetrics, setPerformanceMetrics] = useState<Map<string, number>>(new Map())
  const [messageCache, setMessageCache] = useState<Map<string, { content: string, timestamp: number }>>(new Map())
  const [cacheEnabled, setCacheEnabled] = useState(true)
  const [realTimeStats, setRealTimeStats] = useState(false)
  const [renderCount, setRenderCount] = useState(0)
  const [averageRenderTime, setAverageRenderTime] = useState(0)

  const renderTimesRef = useRef<number[]>([])
  const renderStartTimeRef = useRef<number>(0)

  // Cleanup expired cache entries
  useEffect(() => {
    if (!cacheEnabled) return

    const interval = setInterval(() => {
      setMessageCache(prev => {
        const now = Date.now()
        const next = new Map()
        for (const [key, value] of prev.entries()) {
          if (now - value.timestamp < CACHE_TTL) {
            next.set(key, value)
          }
        }
        return next
      })
    }, 60000) // Check every minute

    return () => clearInterval(interval)
  }, [cacheEnabled])

  const cacheMessage = useCallback((id: string, content: string) => {
    if (!cacheEnabled) return

    setMessageCache(prev => {
      const next = new Map(prev)
      next.set(id, {
        content,
        timestamp: Date.now(),
      })
      return next
    })
  }, [cacheEnabled])

  const getCachedMessage = useCallback((id: string): string | null => {
    if (!cacheEnabled) return null

    const cached = messageCache.get(id)
    if (!cached) return null

    const now = Date.now()
    if (now - cached.timestamp > CACHE_TTL) {
      setMessageCache(prev => {
        const next = new Map(prev)
        next.delete(id)
        return next
      })
      return null
    }

    return cached.content
  }, [cacheEnabled, messageCache])

  const clearCache = useCallback(() => {
    setMessageCache(new Map())
  }, [])

  const recordMetric = useCallback((name: string, value: number) => {
    setPerformanceMetrics(prev => {
      const next = new Map(prev)
      next.set(name, value)
      return next
    })
  }, [])

  const getMetric = useCallback((name: string): number | undefined => {
    return performanceMetrics.get(name)
  }, [performanceMetrics])

  const resetMetrics = useCallback(() => {
    setPerformanceMetrics(new Map())
    setRenderCount(0)
    setAverageRenderTime(0)
    renderTimesRef.current = []
  }, [])

  const measureRender = useCallback((componentName: string) => {
    const startTime = performance.now()
    renderStartTimeRef.current = startTime
    setRenderCount(prev => prev + 1)

    return () => {
      const endTime = performance.now()
      const renderTime = endTime - startTime
      
      renderTimesRef.current.push(renderTime)
      if (renderTimesRef.current.length > 100) {
        renderTimesRef.current.shift()
      }

      const avg = renderTimesRef.current.reduce((a, b) => a + b, 0) / renderTimesRef.current.length
      setAverageRenderTime(avg)

      if (realTimeStats) {
        recordMetric(`${componentName}_render_time`, renderTime)
        recordMetric('average_render_time', avg)
      }
    }
  }, [realTimeStats, recordMetric])

  // Memoized performance summary
  const performanceSummary = useMemo(() => {
    return {
      cacheSize: messageCache.size,
      cacheHitRate: performanceMetrics.get('cache_hits') || 0,
      totalRenders: renderCount,
      averageRenderTime,
      metricsCount: performanceMetrics.size,
    }
  }, [messageCache.size, performanceMetrics, renderCount, averageRenderTime])

  return {
    // State
    showPerformance,
    performanceMetrics,
    messageCache,
    cacheEnabled,
    realTimeStats,
    renderCount,
    averageRenderTime,
    // Actions
    setShowPerformance,
    setCacheEnabled,
    setRealTimeStats,
    cacheMessage,
    getCachedMessage,
    clearCache,
    recordMetric,
    getMetric,
    resetMetrics,
    measureRender,
    // Computed
    performanceSummary,
  } as PerformanceState & PerformanceActions & { performanceSummary: typeof performanceSummary }
}




