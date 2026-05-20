/**
 * Performance Optimizer
 * Optimizaciones de rendimiento para ChatInterface
 */

export class PerformanceOptimizer {
  private debounceTimers: Map<string, NodeJS.Timeout> = new Map()
  private memoizedResults: Map<string, any> = new Map()
  private maxMemoSize = 100

  /**
   * Debounce con key
   */
  debounce<T extends (...args: any[]) => any>(
    key: string,
    func: T,
    wait: number
  ): (...args: Parameters<T>) => void {
    return (...args: Parameters<T>) => {
      const existing = this.debounceTimers.get(key)
      if (existing) {
        clearTimeout(existing)
      }

      const timer = setTimeout(() => {
        func(...args)
        this.debounceTimers.delete(key)
      }, wait)

      this.debounceTimers.set(key, timer)
    }
  }

  /**
   * Memoizar resultado (función sin argumentos)
   */
  memoizeValue<T>(key: string, fn: () => T, maxAge: number = 60000): T {
    const cached = this.memoizedResults.get(key)
    if (cached && Date.now() - cached.timestamp < maxAge) {
      return cached.value
    }

    const result = fn()
    this.memoizedResults.set(key, {
      value: result,
      timestamp: Date.now(),
    })

    // Limpiar cache si es muy grande
    if (this.memoizedResults.size > this.maxMemoSize) {
      const firstKey = this.memoizedResults.keys().next().value
      this.memoizedResults.delete(firstKey)
    }

    return result
  }

  /**
   * Memoizar función (con argumentos)
   */
  memoize<T extends (...args: any[]) => any>(
    key: string,
    fn: T,
    maxAge: number = 60000
  ): T {
    const cacheKey = (args: any[]) => `${key}:${JSON.stringify(args)}`
    
    return ((...args: Parameters<T>) => {
      const argsKey = cacheKey(args)
      const cached = this.memoizedResults.get(argsKey)
      if (cached && Date.now() - cached.timestamp < maxAge) {
        return cached.value
      }

      const result = fn(...args)
      this.memoizedResults.set(argsKey, {
        value: result,
        timestamp: Date.now(),
      })

      // Limpiar cache si es muy grande
      if (this.memoizedResults.size > this.maxMemoSize) {
        const firstKey = this.memoizedResults.keys().next().value
        this.memoizedResults.delete(firstKey)
      }

      return result
    }) as T
  }

  /**
   * Limpiar cache
   */
  clearCache(key?: string): void {
    if (key) {
      this.memoizedResults.delete(key)
    } else {
      this.memoizedResults.clear()
    }
  }

  /**
   * Limpiar todos los debounces
   */
  clearDebounces(): void {
    for (const timer of this.debounceTimers.values()) {
      clearTimeout(timer)
    }
    this.debounceTimers.clear()
  }

  /**
   * Limpiar todo
   */
  cleanup(): void {
    this.clearDebounces()
    this.clearCache()
  }
}

// Singleton instance
let performanceOptimizerInstance: PerformanceOptimizer | null = null

export function getPerformanceOptimizer(): PerformanceOptimizer {
  if (!performanceOptimizerInstance) {
    performanceOptimizerInstance = new PerformanceOptimizer()
  }
  return performanceOptimizerInstance
}

export default PerformanceOptimizer


