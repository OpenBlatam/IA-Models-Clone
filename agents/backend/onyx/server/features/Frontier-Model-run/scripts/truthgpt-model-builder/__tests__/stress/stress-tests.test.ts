/**
 * Stress Tests
 */

import { getSmartCache } from '@/lib/smart-cache'
import { getSmartHistory } from '@/lib/smart-history'
import { getRealTimeMetrics } from '@/lib/realtime-metrics'
import { getPerformanceOptimizer } from '@/lib/performance-optimizer'

describe('Stress Tests', () => {
  describe('High Volume Operations', () => {
    it('should handle 1000 cache operations', () => {
      const cache = getSmartCache('stress-test', { maxSize: 100 })
      const startTime = Date.now()

      for (let i = 0; i < 1000; i++) {
        cache.set(`key-${i}`, `value-${i}`)
      }

      const duration = Date.now() - startTime
      expect(duration).toBeLessThan(2000)
      expect(cache.keys().length).toBeLessThanOrEqual(100)
    })

    it('should handle 1000 history additions', () => {
      const history = getSmartHistory()
      const startTime = Date.now()

      for (let i = 0; i < 1000; i++) {
        history.addModel({
          modelId: `model-${i}`,
          modelName: `test-${i}`,
          description: 'test',
          status: 'completed' as const,
          startTime: Date.now() - i * 1000,
          endTime: Date.now() - i * 1000,
        })
      }

      const duration = Date.now() - startTime
      expect(duration).toBeLessThan(5000)
      expect(history.getAllModels().length).toBe(1000)
    })

    it('should handle 1000 metric updates', () => {
      const metrics = getRealTimeMetrics()
      metrics.registerMetric('stress-metric', 'Stress Test', '')
      const startTime = Date.now()

      for (let i = 0; i < 1000; i++) {
        metrics.updateMetric('stress-metric', i)
      }

      const duration = Date.now() - startTime
      expect(duration).toBeLessThan(2000)
      
      const metric = metrics.getMetric('stress-metric')
      expect(metric?.currentValue).toBe(999)
    })
  })

  describe('Concurrent Operations', () => {
    it('should handle concurrent cache operations', async () => {
      const cache = getSmartCache('concurrent', { maxSize: 100 })
      
      const operations = Array(100).fill(null).map((_, i) => {
        return Promise.all([
          Promise.resolve(cache.set(`key-${i}`, `value-${i}`)),
          Promise.resolve(cache.get(`key-${i}`)),
        ])
      })

      await Promise.all(operations)
      expect(cache.keys().length).toBeLessThanOrEqual(100)
    })

    it('should handle concurrent history operations', async () => {
      const history = getSmartHistory()

      const operations = Array(100).fill(null).map((_, i) => {
        return Promise.resolve(history.addModel({
          modelId: `model-${i}`,
          modelName: `test-${i}`,
          description: 'test',
          status: 'completed' as const,
          startTime: Date.now(),
          endTime: Date.now(),
        }))
      })

      await Promise.all(operations)
      expect(history.getAllModels().length).toBe(100)
    })
  })

  describe('Memory Stress', () => {
    it('should handle large data structures', () => {
      const cache = getSmartCache('memory-stress', { maxSize: 10 })
      const largeData = {
        data: Array(10000).fill('x').join(''),
        nested: {
          level1: {
            level2: {
              level3: Array(1000).fill('y').join(''),
            },
          },
        },
      }

      cache.set('large-key', largeData)
      const retrieved = cache.get('large-key')
      
      expect(retrieved).toBeDefined()
      expect(retrieved.data.length).toBe(10000)
    })
  })

  describe('Performance Under Stress', () => {
    it('should maintain performance with optimizer under stress', () => {
      const optimizer = getPerformanceOptimizer()
      
      let callCount = 0
      const expensiveFn = jest.fn((n: number) => {
        callCount++
        let sum = 0
        for (let i = 0; i < n; i++) {
          sum += i
        }
        return sum
      })

      const memoized = optimizer.memoize('stress-fn', expensiveFn, 60000)

      const startTime = Date.now()
      
      // Call with same argument many times
      for (let i = 0; i < 1000; i++) {
        memoized(1000)
      }

      const duration = Date.now() - startTime

      // Should be fast due to memoization
      expect(duration).toBeLessThan(1000)
      expect(expensiveFn).toHaveBeenCalledTimes(1)
    })
  })

  describe('Long Running Operations', () => {
    it('should handle long running operations', async () => {
      const history = getSmartHistory()
      const startTime = Date.now()

      // Add models over time
      for (let i = 0; i < 500; i++) {
        history.addModel({
          modelId: `model-${i}`,
          modelName: `test-${i}`,
          description: 'test',
          status: 'completed' as const,
          startTime: Date.now() - i * 1000,
          endTime: Date.now() - i * 1000,
        })

        // Small delay to simulate real usage
        if (i % 100 === 0) {
          await new Promise(resolve => setTimeout(resolve, 10))
        }
      }

      const duration = Date.now() - startTime
      expect(duration).toBeLessThan(10000)
      expect(history.getAllModels().length).toBe(500)
    })
  })
})










