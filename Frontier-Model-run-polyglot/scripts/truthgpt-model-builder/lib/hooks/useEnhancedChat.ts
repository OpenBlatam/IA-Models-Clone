/**
 * Enhanced Chat Hook for TruthGPT ChatInterface
 * Integrates caching, metrics, and rate limiting
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { messageCache } from '../advanced-cache'
import { generateCacheKey } from '../service-improvements'
import { performanceMetrics } from '../performance-metrics'
import { rateLimiter } from '../rate-limiter'
import { enhancedExecute } from '../service-improvements'

interface UseEnhancedChatOptions {
  enableCache?: boolean
  enableMetrics?: boolean
  enableRateLimit?: boolean
  cacheTTL?: number
}

export function useEnhancedChat(options: UseEnhancedChatOptions = {}) {
  const {
    enableCache = true,
    enableMetrics = true,
    enableRateLimit = true,
    cacheTTL = 1800000 // 30 minutes
  } = options

  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<any>(null)
  const rateLimitStatusRef = useRef<any>(null)

  // Update metrics periodically
  useEffect(() => {
    if (!enableMetrics) return

    const interval = setInterval(() => {
      const summary = performanceMetrics.getSummary()
      setMetrics(summary)
    }, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [enableMetrics])

  // Update rate limit status
  useEffect(() => {
    if (!enableRateLimit) return

    const interval = setInterval(() => {
      rateLimitStatusRef.current = rateLimiter.getStatus()
    }, 1000) // Update every second

    return () => clearInterval(interval)
  }, [enableRateLimit])

  /**
   * Enhanced send message function
   */
  const sendMessage = useCallback(async (
    message: string,
    handler: (message: string) => Promise<any>
  ): Promise<any> => {
    setIsLoading(true)
    setError(null)

    try {
      // Check rate limit
      if (enableRateLimit && !rateLimiter.canProceed()) {
        const status = rateLimiter.getStatus()
        throw new Error(
          `Rate limit exceeded. Please wait ${Math.ceil((status.waitTimeMs || 0) / 1000)} seconds.`
        )
      }

      // Generate cache key
      const cacheKey = enableCache 
        ? generateCacheKey('message', message)
        : undefined

      // Check cache
      if (enableCache && cacheKey) {
        const cached = messageCache.get(cacheKey)
        if (cached !== null) {
          setIsLoading(false)
          return cached
        }
      }

      // Execute with enhancements
      const result = await enhancedExecute(
        'send_message',
        () => handler(message),
        {
          cacheKey,
          cache: enableCache ? messageCache : undefined,
          cacheTTL: enableCache ? cacheTTL : undefined,
          retryConfig: {
            maxRetries: 2,
            initialDelay: 1000,
            maxDelay: 5000
          },
          metadata: { messageLength: message.length }
        }
      )

      setIsLoading(false)
      return result
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      setIsLoading(false)
      throw err
    }
  }, [enableCache, enableRateLimit, cacheTTL])

  /**
   * Clear cache
   */
  const clearCache = useCallback(() => {
    messageCache.clear()
  }, [])

  /**
   * Get cache statistics
   */
  const getCacheStats = useCallback(() => {
    return messageCache.getStats()
  }, [])

  /**
   * Get performance metrics
   */
  const getMetrics = useCallback(() => {
    return performanceMetrics.getSummary()
  }, [])

  /**
   * Get rate limit status
   */
  const getRateLimitStatus = useCallback(() => {
    return rateLimiter.getStatus()
  }, [])

  return {
    sendMessage,
    isLoading,
    error,
    metrics,
    rateLimitStatus: rateLimitStatusRef.current,
    clearCache,
    getCacheStats,
    getMetrics,
    getRateLimitStatus
  }
}

