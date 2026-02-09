/**
 * Performance Tests
 */

import { getSmartCache } from '@/lib/smart-cache'
import { getSmartHistory } from '@/lib/smart-history'
import { getPerformanceOptimizer } from '@/lib/performance-optimizer'

describe('Performance Tests', () => {
  describe('Cache Performance', () => {
    it('should handle large cache efficiently', () => {
      const cache = getSmartCache('perf-test', { maxSize: 1000 })
      const startTime = Date.now()

      // Add 1000 items
      for (let i = 0; i < 1000; i++) {
        cache.set(`key-${i}`, { value: i, data: Array(100).fill(i) })
      }

      // Access all items
      for (let i = 0; i < 1000; i++) {
        cache.get(`key-${i}`)
      }

      const endTime = Date.now()
      const duration = endTime - startTime

      expect(duration).toBeLessThan(1000) // Should complete in < 1 second
    })

    it('should maintain performance with LRU eviction', () => {
      const cache = getSmartCache('lru-perf', { maxSize: 100, strategy: 'lru' })
      const startTime = Date.now()

      // Add 200 items (should evict 100)
      for (let i = 0; i < 200; i++) {
        cache.set(`key-${i}`, { value: i })
        cache.get(`key-${i}`) // Access to maintain order
      }

      const endTime = Date.now()
      const duration = endTime - startTime

      expect(duration).toBeLessThan(500) // Should be fast even with eviction
    })
  })

  describe('History Performance', () => {
    it('should handle large history efficiently', () => {
      const history = getSmartHistory()
      const startTime = Date.now()

      // Add 1000 models
      for (let i = 0; i < 1000; i++) {
        history.addModel({
          modelId: `model-${i}`,
          modelName: `test-${i}`,
          description: `classification model ${i}`,
          status: 'completed' as const,
          duration: 5000 + i,
          startTime: Date.now() - i * 1000,
          endTime: Date.now() - i * 1000,
        })
      }

      const endTime = Date.now()
      const duration = endTime - startTime

      expect(duration).toBeLessThan(2000) // Should complete in < 2 seconds
    })

    it('should search large history quickly', () => {
      const history = getSmartHistory()

      // Add 500 models
      for (let i = 0; i < 500; i++) {
        history.addModel({
          modelId: `model-${i}`,
          modelName: `test-${i}`,
          description: i % 2 === 0 ? 'classification' : 'regression',
          status: 'completed' as const,
          startTime: Date.now() - i * 1000,
          endTime: Date.now() - i * 1000,
        })
      }

      const startTime = Date.now()
      const results = history.search({ query: 'classification', limit: 10 })
      const endTime = Date.now()
      const duration = endTime - startTime

      expect(results.length).toBeLessThanOrEqual(10)
      expect(duration).toBeLessThan(100) // Should be very fast
    })

    it('should handle complex filters efficiently', () => {
      const history = getSmartHistory()

      // Add 1000 models with various statuses
      for (let i = 0; i < 1000; i++) {
        history.addModel({
          modelId: `model-${i}`,
          modelName: `test-${i}`,
          description: 'test',
          status: i % 3 === 0 ? 'completed' : i % 3 === 1 ? 'failed' : 'creating' as any,
          duration: i % 3 === 0 ? 5000 : undefined,
          startTime: Date.now() - i * 1000,
          endTime: i % 3 === 0 ? Date.now() - i * 1000 : undefined,
        })
      }

      const startTime = Date.now()
      const results = history.search({
        status: 'completed',
        maxDuration: 10000,
        sortBy: 'duration',
        sortOrder: 'asc',
        limit: 20,
      })
      const endTime = Date.now()
      const duration = endTime - startTime

      expect(results.length).toBeLessThanOrEqual(20)
      expect(duration).toBeLessThan(200) // Should be fast even with complex filters
    })
  })

  describe('Optimizer Performance', () => {
    it('should improve performance with memoization', () => {
      const optimizer = getPerformanceOptimizer()

      let callCount = 0
      const expensiveFn = jest.fn((n: number) => {
        callCount++
        // Simulate expensive operation
        let sum = 0
        for (let i = 0; i < n; i++) {
          sum += i
        }
        return sum
      })

      const memoized = optimizer.memoize('expensive', expensiveFn, 60000)

      // First call
      const start1 = Date.now()
      const result1 = memoized(10000)
      const time1 = Date.now() - start1

      // Second call (should be cached)
      const start2 = Date.now()
      const result2 = memoized(10000)
      const time2 = Date.now() - start2

      expect(result1).toBe(result2)
      expect(time2).toBeLessThan(time1)
      expect(expensiveFn).toHaveBeenCalledTimes(1) // Only called once
    })

    it('should debounce efficiently', (done) => {
      const optimizer = getPerformanceOptimizer()
      const fn = jest.fn()

      const debounced = optimizer.debounce('test', fn, 100)

      // Call 100 times rapidly
      for (let i = 0; i < 100; i++) {
        debounced()
      }

      setTimeout(() => {
        expect(fn).toHaveBeenCalledTimes(1) // Should only execute once
        done()
      }, 200)
    })
  })

  describe('Memory Usage', () => {
    it('should limit memory usage with cache size limits', () => {
      const cache = getSmartCache('memory-test', { maxSize: 100 })
      
      // Add more items than limit
      for (let i = 0; i < 200; i++) {
        cache.set(`key-${i}`, { data: Array(1000).fill(i) })
      }

      // Cache should not exceed max size
      expect(cache.keys().length).toBeLessThanOrEqual(100)
    })

    it('should clean expired entries to free memory', () => {
      const cache = getSmartCache('memory-clean', { maxSize: 1000 })
      
      // Add items with short TTL
      for (let i = 0; i < 500; i++) {
        cache.set(`key-${i}`, { value: i }, 100) // 100ms TTL
      }

      // Wait for expiration
      setTimeout(() => {
        const cleaned = cache.cleanExpired()
        expect(cleaned).toBeGreaterThan(0)
        expect(cache.keys().length).toBeLessThan(500)
      }, 200)
    })
  })
})










