/**
 * Utilidades de Performance
 * =========================
 * 
 * Funciones para optimizar el rendimiento
 */

// ============================================================================
// MEMOIZACIÓN
// ============================================================================

/**
 * Memoiza una función con cache LRU
 */
export function memoize<T extends (...args: any[]) => any>(
  fn: T,
  maxSize: number = 100
): T {
  const cache = new Map<string, ReturnType<T>>()
  const accessOrder: string[] = []

  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = JSON.stringify(args)
    
    if (cache.has(key)) {
      // Mover al final (LRU)
      const index = accessOrder.indexOf(key)
      accessOrder.splice(index, 1)
      accessOrder.push(key)
      return cache.get(key)!
    }

    const result = fn(...args)
    
    // Si el cache está lleno, eliminar el menos usado
    if (cache.size >= maxSize) {
      const oldest = accessOrder.shift()!
      cache.delete(oldest)
    }
    
    cache.set(key, result)
    accessOrder.push(key)
    
    return result
  }) as T
}

/**
 * Memoiza una función con TTL (Time To Live)
 */
export function memoizeWithTTL<T extends (...args: any[]) => any>(
  fn: T,
  ttl: number = 60000 // 1 minuto por defecto
): T {
  const cache = new Map<string, { value: ReturnType<T>; expires: number }>()

  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = JSON.stringify(args)
    const cached = cache.get(key)
    
    if (cached && cached.expires > Date.now()) {
      return cached.value
    }

    const result = fn(...args)
    cache.set(key, {
      value: result,
      expires: Date.now() + ttl
    })
    
    return result
  }) as T
}

// ============================================================================
// DEBOUNCING Y THROTTLING
// ============================================================================

/**
 * Debounce mejorado con cancelación
 */
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): T & { cancel: () => void } {
  let timeoutId: NodeJS.Timeout | null = null

  const debounced = ((...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    timeoutId = setTimeout(() => {
      fn(...args)
      timeoutId = null
    }, delay)
  }) as T & { cancel: () => void }

  debounced.cancel = () => {
    if (timeoutId) {
      clearTimeout(timeoutId)
      timeoutId = null
    }
  }

  return debounced
}

/**
 * Throttle mejorado
 */
export function throttle<T extends (...args: any[]) => any>(
  fn: T,
  limit: number
): T {
  let inThrottle: boolean = false
  let lastResult: ReturnType<T>

  return ((...args: Parameters<T>): ReturnType<T> => {
    if (!inThrottle) {
      lastResult = fn(...args)
      inThrottle = true
      setTimeout(() => {
        inThrottle = false
      }, limit)
    }
    return lastResult
  }) as T
}

// ============================================================================
// LAZY LOADING
// ============================================================================

/**
 * Carga perezosa de un módulo
 */
export function lazyLoad<T>(
  loader: () => Promise<{ default: T }>,
  fallback?: T
): () => Promise<T> {
  let cached: T | null = null
  let loading: Promise<T> | null = null

  return async (): Promise<T> => {
    if (cached) return cached
    if (loading) return loading

    loading = loader().then(module => {
      cached = module.default
      loading = null
      return cached
    }).catch(error => {
      loading = null
      if (fallback) {
        cached = fallback
        return fallback
      }
      throw error
    })

    return loading
  }
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

/**
 * Procesa items en lotes
 */
export async function batchProcess<T, R>(
  items: T[],
  processor: (batch: T[]) => Promise<R[]>,
  batchSize: number = 10
): Promise<R[]> {
  const results: R[] = []

  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize)
    const batchResults = await processor(batch)
    results.push(...batchResults)
  }

  return results
}

/**
 * Procesa items en lotes con límite de concurrencia
 */
export async function batchProcessConcurrent<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  concurrency: number = 5
): Promise<R[]> {
  const results: R[] = []
  const executing: Promise<void>[] = []

  for (const item of items) {
    const promise = processor(item).then(result => {
      results.push(result)
      executing.splice(executing.indexOf(promise), 1)
    })

    executing.push(promise)

    if (executing.length >= concurrency) {
      await Promise.race(executing)
    }
  }

  await Promise.all(executing)
  return results
}

// ============================================================================
// PERFORMANCE MONITORING
// ============================================================================

/**
 * Mide el tiempo de ejecución de una función
 */
export async function measureTime<T>(
  fn: () => Promise<T> | T,
  label?: string
): Promise<{ result: T; time: number }> {
  const start = performance.now()
  const result = await fn()
  const time = performance.now() - start

  if (label) {
    console.log(`[${label}] Tiempo de ejecución: ${time.toFixed(2)}ms`)
  }

  return { result, time }
}

/**
 * Crea un profiler para múltiples operaciones
 */
export function createProfiler() {
  const timings: Map<string, number[]> = new Map()

  return {
    start(label: string): () => void {
      const start = performance.now()
      return () => {
        const time = performance.now() - start
        const existing = timings.get(label) || []
        existing.push(time)
        timings.set(label, existing)
      }
    },

    getStats(): Record<string, { count: number; avg: number; min: number; max: number }> {
      const stats: Record<string, { count: number; avg: number; min: number; max: number }> = {}

      for (const [label, times] of timings.entries()) {
        const sum = times.reduce((a, b) => a + b, 0)
        stats[label] = {
          count: times.length,
          avg: sum / times.length,
          min: Math.min(...times),
          max: Math.max(...times)
        }
      }

      return stats
    },

    reset(): void {
      timings.clear()
    }
  }
}

// ============================================================================
// OPTIMIZACIONES DE RENDER
// ============================================================================

/**
 * Compara objetos profundamente para evitar re-renders innecesarios
 */
export function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (a == null || b == null) return false
  if (typeof a !== 'object' || typeof b !== 'object') return false

  const keysA = Object.keys(a as Record<string, unknown>)
  const keysB = Object.keys(b as Record<string, unknown>)

  if (keysA.length !== keysB.length) return false

  for (const key of keysA) {
    if (!keysB.includes(key)) return false
    if (!deepEqual(
      (a as Record<string, unknown>)[key],
      (b as Record<string, unknown>)[key]
    )) return false
  }

  return true
}

/**
 * Crea una versión estable de un objeto para usar como dependencia
 */
export function stableObject<T extends Record<string, unknown>>(obj: T): T {
  const keys = Object.keys(obj).sort()
  const stable: Record<string, unknown> = {}
  
  for (const key of keys) {
    stable[key] = obj[key]
  }
  
  return stable as T
}







