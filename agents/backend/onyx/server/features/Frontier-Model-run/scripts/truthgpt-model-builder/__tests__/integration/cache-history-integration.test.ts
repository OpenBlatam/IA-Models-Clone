/**
 * Integration Tests - Cache + History
 */

import { getSmartCache } from '@/lib/smart-cache'
import { getSmartHistory } from '@/lib/smart-history'
import { getPerformanceOptimizer } from '@/lib/performance-optimizer'

describe('Cache + History Integration', () => {
  let cache: ReturnType<typeof getSmartCache>
  let history: ReturnType<typeof getSmartHistory>
  let optimizer: ReturnType<typeof getPerformanceOptimizer>

  beforeEach(() => {
    cache = getSmartCache('history-cache', { maxSize: 50, strategy: 'lru' })
    history = getSmartHistory()
    optimizer = getPerformanceOptimizer()

    if (typeof window !== 'undefined') {
      localStorage.clear()
    }
  })

  afterEach(() => {
    cache.clear()
    history.clear()
    optimizer.cleanup()
  })

  describe('Cached Search', () => {
    it('should cache search results', () => {
      // Add models
      for (let i = 0; i < 10; i++) {
        history.addModel({
          modelId: `model-${i}`,
          modelName: `classification-${i}`,
          description: `classification model ${i}`,
          status: 'completed' as const,
          startTime: Date.now() - i * 1000,
          endTime: Date.now() - i * 1000,
        })
      }

      // First search (should query)
      const searchFn = jest.fn((query: string) => {
        return history.search({ query })
      })

      const memoizedSearch = optimizer.memoize('search', searchFn, 60000)

      const results1 = memoizedSearch('classification')
      expect(searchFn).toHaveBeenCalledTimes(1)

      // Second search (should use cache)
      const results2 = memoizedSearch('classification')
      expect(searchFn).toHaveBeenCalledTimes(1) // Still 1, from cache

      expect(results1).toEqual(results2)
    })

    it('should invalidate cache on new models', () => {
      const searchFn = jest.fn((query: string) => {
        return history.search({ query })
      })

      const memoizedSearch = optimizer.memoize('search', searchFn, 60000)

      // Initial search
      memoizedSearch('classification')
      expect(searchFn).toHaveBeenCalledTimes(1)

      // Add new model
      history.addModel({
        modelId: 'new-model',
        modelName: 'classification-new',
        description: 'classification model',
        status: 'completed' as const,
        startTime: Date.now(),
        endTime: Date.now(),
      })

      // Clear cache to simulate invalidation
      optimizer.clearCache('search')

      // Search again
      memoizedSearch('classification')
      expect(searchFn).toHaveBeenCalledTimes(2) // Called again after clear
    })
  })

  describe('Performance Optimization', () => {
    it('should optimize frequent searches', () => {
      // Add many models
      for (let i = 0; i < 100; i++) {
        history.addModel({
          modelId: `model-${i}`,
          modelName: `test-${i}`,
          description: i % 2 === 0 ? 'classification' : 'regression',
          status: 'completed' as const,
          startTime: Date.now() - i * 1000,
          endTime: Date.now() - i * 1000,
        })
      }

      const searchFn = jest.fn((query: string) => {
        return history.search({ query, limit: 10 })
      })

      const memoizedSearch = optimizer.memoize('search', searchFn, 60000)

      // Multiple searches for same query
      for (let i = 0; i < 10; i++) {
        memoizedSearch('classification')
      }

      // Should only call once due to memoization
      expect(searchFn).toHaveBeenCalledTimes(1)
    })
  })

  describe('Cache Strategy with History', () => {
    it('should use LRU strategy for history cache', () => {
      const lruCache = getSmartCache('lru-history', { maxSize: 3, strategy: 'lru' })

      // Add 3 searches
      lruCache.set('search-1', history.search({ query: 'test1' }))
      lruCache.set('search-2', history.search({ query: 'test2' }))
      lruCache.set('search-3', history.search({ query: 'test3' }))

      // Access search-1 to make it recently used
      lruCache.get('search-1')

      // Add new search (should evict search-2)
      lruCache.set('search-4', history.search({ query: 'test4' }))

      expect(lruCache.get('search-1')).toBeDefined() // Recently used
      expect(lruCache.get('search-2')).toBeNull() // Evicted
      expect(lruCache.get('search-3')).toBeDefined()
      expect(lruCache.get('search-4')).toBeDefined()
    })
  })

  describe('Debounced History Updates', () => {
    it('should debounce history additions', (done) => {
      let addCount = 0
      const addModel = jest.fn(() => {
        addCount++
      })

      const debouncedAdd = optimizer.debounce('add-model', addModel, 100)

      // Rapid additions
      for (let i = 0; i < 5; i++) {
        debouncedAdd()
      }

      setTimeout(() => {
        expect(addModel).toHaveBeenCalledTimes(1)
        done()
      }, 200)
    })
  })

  describe('Cache Statistics', () => {
    it('should track cache performance with history', () => {
      // Add models
      for (let i = 0; i < 20; i++) {
        history.addModel({
          modelId: `model-${i}`,
          modelName: `test-${i}`,
          description: 'test',
          status: 'completed' as const,
          startTime: Date.now(),
          endTime: Date.now(),
        })
      }

      // Cache multiple searches
      const queries = ['classification', 'regression', 'nlp']
      queries.forEach(query => {
        const results = history.search({ query })
        cache.set(`search-${query}`, results)
      })

      // Access cached results
      queries.forEach(query => {
        cache.get(`search-${query}`)
      })

      const stats = cache.getStats()
      expect(stats.totalHits).toBeGreaterThan(0)
      expect(stats.hitRate).toBeGreaterThan(0)
    })
  })
})










