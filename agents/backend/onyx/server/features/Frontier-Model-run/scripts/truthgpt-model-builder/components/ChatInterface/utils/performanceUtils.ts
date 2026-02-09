/**
 * Utility functions for performance optimization
 */

/**
 * Debounce function
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

/**
 * Throttle function
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

/**
 * Memoize function results
 */
export function memoize<Args extends any[], Return>(
  fn: (...args: Args) => Return,
  keyFn?: (...args: Args) => string
): (...args: Args) => Return {
  const cache = new Map<string, Return>()
  
  return (...args: Args): Return => {
    const key = keyFn ? keyFn(...args) : JSON.stringify(args)
    
    if (cache.has(key)) {
      return cache.get(key)!
    }
    
    const result = fn(...args)
    cache.set(key, result)
    return result
  }
}

/**
 * Batch function calls
 */
export function batchCalls<T extends (...args: any[]) => any>(
  fn: T,
  delay: number = 100
): (...args: Parameters<T>) => void {
  let batch: Parameters<T>[] = []
  let timeout: NodeJS.Timeout | null = null

  return (...args: Parameters<T>) => {
    batch.push(args)

    if (timeout) clearTimeout(timeout)

    timeout = setTimeout(() => {
      if (batch.length > 0) {
        fn(...batch[0]) // Call with first set of args
        batch = []
      }
    }, delay)
  }
}

/**
 * Measure function execution time
 */
export function measureTime<T extends (...args: any[]) => any>(
  fn: T,
  label?: string
): T {
  return ((...args: Parameters<T>) => {
    const start = performance.now()
    const result = fn(...args)
    const end = performance.now()
    
    if (label) {
      console.log(`${label}: ${(end - start).toFixed(2)}ms`)
    }
    
    return result
  }) as T
}

/**
 * Lazy load function
 */
export function lazyLoad<T>(
  loader: () => Promise<T>
): () => Promise<T> {
  let promise: Promise<T> | null = null
  
  return () => {
    if (!promise) {
      promise = loader()
    }
    return promise
  }
}

/**
 * Request animation frame throttle
 */
export function rafThrottle<T extends (...args: any[]) => any>(
  fn: T
): (...args: Parameters<T>) => void {
  let rafId: number | null = null
  let lastArgs: Parameters<T> | null = null

  return (...args: Parameters<T>) => {
    lastArgs = args

    if (rafId === null) {
      rafId = requestAnimationFrame(() => {
        if (lastArgs) {
          fn(...lastArgs)
        }
        rafId = null
        lastArgs = null
      })
    }
  }
}

/**
 * Intersection observer for lazy loading
 */
export function createIntersectionObserver(
  callback: (entries: IntersectionObserverEntry[]) => void,
  options?: IntersectionObserverInit
): IntersectionObserver {
  return new IntersectionObserver(callback, {
    root: null,
    rootMargin: '50px',
    threshold: 0.1,
    ...options,
  })
}

/**
 * Virtual scrolling helper
 */
export interface VirtualScrollOptions {
  itemHeight: number
  containerHeight: number
  overscan?: number
}

export function calculateVirtualScrollRange(
  scrollTop: number,
  options: VirtualScrollOptions
): { start: number; end: number; total: number } {
  const { itemHeight, containerHeight, overscan = 5 } = options
  
  const start = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan)
  const visibleCount = Math.ceil(containerHeight / itemHeight)
  const end = start + visibleCount + overscan * 2
  
  return { start, end, total: end - start }
}

/**
 * Batch DOM updates
 */
export function batchDOMUpdates(updates: (() => void)[]): void {
  // Use requestAnimationFrame for batching
  requestAnimationFrame(() => {
    updates.forEach(update => update())
  })
}

/**
 * Performance monitor
 */
export class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map()

  start(label: string): () => void {
    const startTime = performance.now()
    
    return () => {
      const duration = performance.now() - startTime
      const existing = this.metrics.get(label) || []
      existing.push(duration)
      this.metrics.set(label, existing)
    }
  }

  getAverage(label: string): number {
    const metrics = this.metrics.get(label) || []
    if (metrics.length === 0) return 0
    return metrics.reduce((a, b) => a + b, 0) / metrics.length
  }

  getMetrics(label?: string): Map<string, number[]> | number[] {
    if (label) {
      return this.metrics.get(label) || []
    }
    return this.metrics
  }

  clear(label?: string): void {
    if (label) {
      this.metrics.delete(label)
    } else {
      this.metrics.clear()
    }
  }
}

export const performanceMonitor = new PerformanceMonitor()




