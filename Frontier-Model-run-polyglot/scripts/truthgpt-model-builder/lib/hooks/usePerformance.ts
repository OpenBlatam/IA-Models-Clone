/**
 * Hook usePerformance
 * ===================
 * 
 * Hook para medir performance en componentes
 */

import { useCallback, useRef } from 'react'
import { measurePerformance, measurePerformanceSync, PerformanceMonitor } from '../utils/performanceMonitor'

/**
 * Hook para medir performance
 */
export function usePerformance() {
  const measure = useCallback(async <T,>(
    name: string,
    fn: () => T | Promise<T>,
    metadata?: Record<string, any>
  ): Promise<T> => {
    return measurePerformance(name, fn, metadata)
  }, [])

  const measureSync = useCallback(<T,>(
    name: string,
    fn: () => T,
    metadata?: Record<string, any>
  ): T => {
    return measurePerformanceSync(name, fn, metadata)
  }, [])

  return {
    measure,
    measureSync
  }
}






