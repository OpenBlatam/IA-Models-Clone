/**
 * Service Improvements for TruthGPT Service
 * Adds caching, metrics, rate limiting, and retry logic
 */

import { messageCache, modelCache } from './advanced-cache'
import { performanceMetrics } from './performance-metrics'
import { rateLimiter } from './rate-limiter'

interface RetryConfig {
  maxRetries?: number
  initialDelay?: number
  maxDelay?: number
  backoffMultiplier?: number
}

/**
 * Execute function with retry logic
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  config: RetryConfig = {}
): Promise<T> {
  const {
    maxRetries = 3,
    initialDelay = 1000,
    maxDelay = 30000,
    backoffMultiplier = 2
  } = config

  let lastError: Error | null = null
  let delay = initialDelay

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const startTime = Date.now()
      const result = await fn()
      const duration = Date.now() - startTime

      performanceMetrics.record('retry_operation', duration, true, {
        attempt,
        maxRetries
      })

      return result
    } catch (error) {
      lastError = error as Error
      const duration = Date.now() - Date.now()

      performanceMetrics.record('retry_operation', duration, false, {
        attempt,
        maxRetries,
        error: lastError.message
      })

      if (attempt < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, delay))
        delay = Math.min(delay * backoffMultiplier, maxDelay)
      }
    }
  }

  throw lastError || new Error('Retry failed')
}

/**
 * Execute function with rate limiting
 */
export async function withRateLimit<T>(
  fn: () => Promise<T>,
  operation: string = 'operation'
): Promise<T> {
  // Check rate limit
  if (!rateLimiter.canProceed()) {
    const status = rateLimiter.getStatus()
    throw new Error(
      `Rate limit exceeded. Wait ${status.waitTimeMs}ms before retrying.`
    )
  }

  rateLimiter.consume()

  const startTime = Date.now()
  let success = false
  let result: T

  try {
    result = await fn()
    success = true
    return result
  } catch (error) {
    throw error
  } finally {
    const duration = Date.now() - startTime
    rateLimiter.recordRequest(success, duration)
  }
}

/**
 * Execute function with caching
 */
export async function withCache<T>(
  key: string,
  fn: () => Promise<T>,
  cache: typeof messageCache | typeof modelCache = messageCache,
  ttl?: number
): Promise<T> {
  // Try cache first
  const cached = cache.get(key)
  if (cached !== null) {
    performanceMetrics.record('cache_hit', 0, true, { key })
    return cached as T
  }

  // Cache miss, execute function
  const startTime = Date.now()
  try {
    const result = await fn()
    const duration = Date.now() - startTime

    // Store in cache
    cache.set(key, result, ttl)

    performanceMetrics.record('cache_miss', duration, true, { key })
    return result
  } catch (error) {
    const duration = Date.now() - startTime
    performanceMetrics.record('cache_miss', duration, false, { key })
    throw error
  }
}

/**
 * Execute function with metrics tracking
 */
export async function withMetrics<T>(
  operation: string,
  fn: () => Promise<T>,
  metadata?: Record<string, any>
): Promise<T> {
  const startTime = Date.now()
  let success = false

  try {
    const result = await fn()
    success = true
    return result
  } catch (error) {
    throw error
  } finally {
    const duration = Date.now() - startTime
    performanceMetrics.record(operation, duration, success, metadata)
  }
}

/**
 * Combined wrapper: cache + metrics + rate limit + retry
 */
export async function enhancedExecute<T>(
  operation: string,
  fn: () => Promise<T>,
  options: {
    cacheKey?: string
    cache?: typeof messageCache | typeof modelCache
    cacheTTL?: number
    retryConfig?: RetryConfig
    metadata?: Record<string, any>
  } = {}
): Promise<T> {
  const {
    cacheKey,
    cache,
    cacheTTL,
    retryConfig,
    metadata = {}
  } = options

  // Wrapper function
  const executeFn = async (): Promise<T> => {
    return withMetrics(operation, async () => {
      if (cacheKey && cache) {
        return withCache(cacheKey, fn, cache, cacheTTL)
      }
      return fn()
    }, metadata)
  }

  // Add rate limiting
  const rateLimitedFn = () => withRateLimit(executeFn, operation)

  // Add retry if configured
  if (retryConfig) {
    return withRetry(rateLimitedFn, retryConfig)
  }

  return rateLimitedFn()
}

/**
 * Generate cache key from parameters
 */
export function generateCacheKey(prefix: string, ...params: any[]): string {
  const paramsStr = params
    .map(p => typeof p === 'object' ? JSON.stringify(p) : String(p))
    .join(':')
  
  return `${prefix}:${paramsStr}`
}

/**
 * Get performance dashboard data
 */
export function getPerformanceDashboard() {
  const cacheStats = {
    messages: messageCache.getStats(),
    models: modelCache.getStats()
  }

  const metrics = performanceMetrics.getSummary()
  const rateLimitStats = rateLimiter.getStats()

  return {
    timestamp: new Date().toISOString(),
    cache: cacheStats,
    metrics,
    rateLimiter: rateLimitStats,
    health: {
      cacheHitRate: cacheStats.messages.hitRate,
      successRate: metrics.overallSuccessRate,
      avgDuration: metrics.avgDuration,
      status: metrics.overallSuccessRate >= 95 && metrics.avgDuration < 5000
        ? 'healthy'
        : metrics.overallSuccessRate >= 80
        ? 'degraded'
        : 'critical'
    }
  }
}


