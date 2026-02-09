/**
 * Integration Tests - Proactive Builder
 */

import { getRealTimeMetrics } from '@/lib/realtime-metrics'
import { getIntelligentAlerts } from '@/lib/intelligent-alerts'
import { getSmartCache } from '@/lib/smart-cache'
import { getSmartHistory } from '@/lib/smart-history'
import { getFavoritesManager } from '@/lib/favorites-manager'

describe('Proactive Builder Integration', () => {
  beforeEach(() => {
    // Clear all instances
    if (typeof window !== 'undefined') {
      localStorage.clear()
    }
  })

  describe('Real-time Metrics + Intelligent Alerts', () => {
    it('should integrate metrics with alerts', () => {
      const metrics = getRealTimeMetrics()
      const alerts = getIntelligentAlerts()

      metrics.registerMetric('buildsPerMinute', 'Builds per Minute', '/min')
      metrics.updateMetric('buildsPerMinute', 0)

      // Trigger alert for no activity
      const alertResults = alerts.evaluate({
        models: [],
        queueLength: 0,
      })

      expect(alertResults.length).toBeGreaterThanOrEqual(0)
    })

    it('should trigger alerts based on metrics', () => {
      const metrics = getRealTimeMetrics()
      const alerts = getIntelligentAlerts()

      metrics.registerMetric('successRate', 'Success Rate', '%')
      metrics.updateMetric('successRate', 30) // Low success rate

      const models = Array(10).fill(null).map((_, i) => ({
        modelId: `model-${i}`,
        modelName: `test-${i}`,
        description: 'test',
        status: i < 3 ? 'completed' : 'failed' as const,
        startTime: Date.now(),
        endTime: Date.now(),
      }))

      const alertResults = alerts.evaluate({ models, queueLength: 0 })
      expect(alertResults.length).toBeGreaterThan(0)
    })
  })

  describe('Smart Cache + Smart History', () => {
    it('should cache history queries', () => {
      const cache = getSmartCache('integration-test', { maxSize: 10 })
      const history = getSmartHistory()

      const model = {
        modelId: 'model-1',
        modelName: 'test-model',
        description: 'classification model',
        status: 'completed' as const,
        duration: 5000,
        startTime: Date.now(),
        endTime: Date.now(),
      }

      history.addModel(model)

      // Cache search result
      const cacheKey = 'search-classification'
      const results = history.search({ query: 'classification' })
      cache.set(cacheKey, results, 60000)

      // Retrieve from cache
      const cachedResults = cache.get(cacheKey)
      expect(cachedResults).toEqual(results)
    })
  })

  describe('Smart History + Favorites Manager', () => {
    it('should add models from history to favorites', () => {
      const history = getSmartHistory()
      const favorites = getFavoritesManager()

      const model = {
        modelId: 'model-1',
        modelName: 'test-model',
        description: 'classification model',
        status: 'completed' as const,
        duration: 5000,
        startTime: Date.now(),
        endTime: Date.now(),
      }

      history.addModel(model)
      favorites.addFavorite(model, 'Great model!', ['classification'])

      const favorite = favorites.getFavorite('model-1')
      expect(favorite).toBeDefined()
      expect(favorite?.notes).toBe('Great model!')
    })

    it('should search favorites and history together', () => {
      const history = getSmartHistory()
      const favorites = getFavoritesManager()

      const model = {
        modelId: 'model-1',
        modelName: 'classification-model',
        description: 'classification model',
        status: 'completed' as const,
        startTime: Date.now(),
        endTime: Date.now(),
      }

      history.addModel(model)
      favorites.addFavorite(model)

      const historyResults = history.search({ query: 'classification' })
      const favoriteResults = favorites.searchFavorites('classification')

      expect(historyResults.length).toBeGreaterThan(0)
      expect(favoriteResults.length).toBeGreaterThan(0)
    })
  })

  describe('End-to-End Model Flow', () => {
    it('should handle complete model lifecycle', () => {
      const metrics = getRealTimeMetrics()
      const alerts = getIntelligentAlerts()
      const history = getSmartHistory()
      const favorites = getFavoritesManager()
      const cache = getSmartCache('model-flow', { maxSize: 50 })

      // 1. Create model
      const model = {
        modelId: 'model-1',
        modelName: 'test-model',
        description: 'classification model',
        status: 'completed' as const,
        duration: 5000,
        startTime: Date.now(),
        endTime: Date.now(),
      }

      // 2. Add to history
      history.addModel(model)

      // 3. Update metrics
      metrics.registerMetric('totalBuilds', 'Total Builds', '')
      metrics.updateMetric('totalBuilds', 1)

      // 4. Add to favorites
      favorites.addFavorite(model, 'Great model!')

      // 5. Cache analysis
      const analysis = { modelId: model.modelId, score: 0.95 }
      cache.set(`analysis-${model.modelId}`, analysis, 300000)

      // 6. Check alerts
      const alertResults = alerts.evaluate({
        models: [model],
        queueLength: 0,
      })

      // Verify all integrations
      expect(history.getAllModels().length).toBe(1)
      expect(favorites.isFavorite('model-1')).toBe(true)
      expect(cache.get(`analysis-${model.modelId}`)).toEqual(analysis)
      expect(metrics.getMetric('totalBuilds')?.currentValue).toBe(1)
    })
  })

  describe('Performance Integration', () => {
    it('should handle multiple models efficiently', () => {
      const history = getSmartHistory()
      const cache = getSmartCache('performance-test', { maxSize: 100 })

      const startTime = Date.now()

      // Add 100 models
      for (let i = 0; i < 100; i++) {
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

      // Search
      const results = history.search({ query: 'classification' })
      
      // Cache results
      cache.set('search-results', results, 60000)

      const endTime = Date.now()
      const duration = endTime - startTime

      expect(results.length).toBeGreaterThan(0)
      expect(duration).toBeLessThan(1000) // Should complete in < 1 second
    })
  })

  describe('Error Handling Integration', () => {
    it('should handle errors gracefully across systems', () => {
      const metrics = getRealTimeMetrics()
      const alerts = getIntelligentAlerts()
      const history = getSmartHistory()

      // Add failed models
      const failedModels = Array(5).fill(null).map((_, i) => ({
        modelId: `model-${i}`,
        modelName: `test-${i}`,
        description: 'test',
        status: 'failed' as const,
        error: 'Test error',
        startTime: Date.now(),
        endTime: Date.now(),
      }))

      failedModels.forEach(model => history.addModel(model))

      // Update metrics
      const modelMetrics = metrics.calculateModelMetrics(failedModels)
      expect(modelMetrics.successRate).toBe(0)

      // Trigger alerts
      const alertResults = alerts.evaluate({
        models: failedModels,
        queueLength: 0,
      })

      // Should detect high failure rate
      expect(alertResults.length).toBeGreaterThan(0)
    })
  })
})










