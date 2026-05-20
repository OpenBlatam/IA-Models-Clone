/**
 * Hook para optimización de rendimiento avanzada
 * ================================================
 */

import { useCallback, useMemo, useRef } from 'react'

export interface PerformanceConfig {
  debounceDelay?: number
  throttleDelay?: number
  maxCacheSize?: number
  enableMemoization?: boolean
  batchSize?: number
}

export interface UseModelPerformanceOptimizerResult {
  debounce: <T extends (...args: any[]) => any>(fn: T, delay?: number) => T
  throttle: <T extends (...args: any[]) => any>(fn: T, delay?: number) => T
  memoize: <T extends (...args: any[]) => any>(fn: T, keyFn?: (...args: Parameters<T>) => string) => T
  batch: <T>(items: T[], processor: (items: T[]) => Promise<void>, size?: number) => Promise<void>
  optimizeRenders: <T>(value: T, deps: any[]) => T
}

/**
 * Hook para optimización de rendimiento
 */
export function useModelPerformanceOptimizer(
  config: PerformanceConfig = {}
): UseModelPerformanceOptimizerResult {
  const {
    debounceDelay = 300,
    throttleDelay = 100,
    maxCacheSize = 100,
    enableMemoization = true,
    batchSize = 10
  } = config

  const cacheRef = useRef<Map<string, any>>(new Map())
  const throttleRefs = useRef<Map<string, { lastCall: number; timeout?: NodeJS.Timeout }>>(new Map())

  const debounce = useCallback(<T extends (...args: any[]) => any>(
    fn: T,
    delay: number = debounceDelay
  ): T => {
    let timeoutId: NodeJS.Timeout | null = null

    return ((...args: Parameters<T>) => {
      if (timeoutId) {
        clearTimeout(timeoutId)
      }

      timeoutId = setTimeout(() => {
        fn(...args)
        timeoutId = null
      }, delay)
    }) as T
  }, [debounceDelay])

  const throttle = useCallback(<T extends (...args: any[]) => any>(
    fn: T,
    delay: number = throttleDelay
  ): T => {
    const key = fn.toString()
    const throttleState = throttleRefs.current.get(key) || { lastCall: 0 }

    return ((...args: Parameters<T>) => {
      const now = Date.now()

      if (now - throttleState.lastCall >= delay) {
        throttleState.lastCall = now
        fn(...args)
      } else {
        if (!throttleState.timeout) {
          throttleState.timeout = setTimeout(() => {
            throttleState.lastCall = Date.now()
            throttleState.timeout = undefined
            fn(...args)
          }, delay - (now - throttleState.lastCall))
        }
      }

      throttleRefs.current.set(key, throttleState)
    }) as T
  }, [throttleDelay])

  const memoize = useCallback(<T extends (...args: any[]) => any>(
    fn: T,
    keyFn?: (...args: Parameters<T>) => string
  ): T => {
    if (!enableMemoization) return fn

    return ((...args: Parameters<T>) => {
      const key = keyFn ? keyFn(...args) : JSON.stringify(args)
      
      if (cacheRef.current.has(key)) {
        return cacheRef.current.get(key)
      }

      const result = fn(...args)

      // Limitar tamaño del caché
      if (cacheRef.current.size >= maxCacheSize) {
        const firstKey = cacheRef.current.keys().next().value
        cacheRef.current.delete(firstKey)
      }

      cacheRef.current.set(key, result)
      return result
    }) as T
  }, [enableMemoization, maxCacheSize])

  const batch = useCallback(async <T,>(
    items: T[],
    processor: (items: T[]) => Promise<void>,
    size: number = batchSize
  ): Promise<void> => {
    for (let i = 0; i < items.length; i += size) {
      const batchItems = items.slice(i, i + size)
      await processor(batchItems)
    }
  }, [batchSize])

  const optimizeRenders = useCallback(<T,>(
    value: T,
    deps: any[]
  ): T => {
    // Usar useMemo internamente sería ideal, pero como esto es un hook,
    // retornamos el valor directamente y el componente puede usar useMemo
    return value
  }, [])

  return {
    debounce,
    throttle,
    memoize,
    batch,
    optimizeRenders
  }
}

