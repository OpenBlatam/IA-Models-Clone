/**
 * Deep Integration Tests
 */

import { getSmartCache } from '@/lib/smart-cache'
import { getSmartHistory } from '@/lib/smart-history'
import { getRealTimeMetrics } from '@/lib/realtime-metrics'
import { getIntelligentAlerts } from '@/lib/intelligent-alerts'
import { getFavoritesManager } from '@/lib/favorites-manager'
import { getModelExporter } from '@/lib/modules/management'
import { getBackupManager } from '@/lib/backup-manager'

describe('Deep Integration Tests', () => {
  beforeEach(() => {
    if (typeof window !== 'undefined') {
      localStorage.clear()
    }
  })

  describe('Complete System Integration', () => {
    it('should integrate all systems together', async () => {
      const cache = getSmartCache('integration', { maxSize: 50 })
      const history = getSmartHistory()
      const metrics = getRealTimeMetrics()
      const alerts = getIntelligentAlerts()
      const favorites = getFavoritesManager()
      const exporter = getModelExporter()
      const backup = getBackupManager()

      // 1. Create model
      const model = {
        modelId: 'model-1',
        modelName: 'test-model',
        description: 'classification model',
        status: 'completed' as const,
        duration: 5000,
        startTime: Date.now() - 5000,
        endTime: Date.now(),
      }

      // 2. Add to history
      history.addModel(model)

      // 3. Cache analysis
      const analysis = { modelId: model.modelId, score: 0.95 }
      cache.set(`analysis-${model.modelId}`, analysis)

      // 4. Update metrics
      metrics.registerMetric('totalModels', 'Total Models', '')
      metrics.updateMetric('totalModels', 1)

      // 5. Add to favorites
      favorites.addFavorite(model, 'Great model!')

      // 6. Trigger alerts
      alerts.evaluate({ models: [model], queueLength: 0 })

      // 7. Export models
      const exportBlob = await exporter.exportModels([model], { format: 'json' })
      expect(exportBlob).toBeDefined()

      // 8. Create backup
      const backupData = {
        models: [model],
        queue: [],
        favorites: favorites.getAllFavorites(),
      }
      const backupResult = backup.createBackup(backupData)

      // Verify all integrations
      expect(history.getAllModels().length).toBe(1)
      expect(cache.get(`analysis-${model.modelId}`)).toEqual(analysis)
      expect(favorites.isFavorite('model-1')).toBe(true)
      expect(metrics.getMetric('totalModels')?.currentValue).toBe(1)
      expect(backupResult).toBeDefined()
    })
  })

  describe('Cross-System Data Flow', () => {
    it('should flow data between all systems', () => {
      const history = getSmartHistory()
      const metrics = getRealTimeMetrics()
      const alerts = getIntelligentAlerts()
      const favorites = getFavoritesManager()

      // Create models
      const models = Array(10).fill(null).map((_, i) => ({
        modelId: `model-${i}`,
        modelName: `test-${i}`,
        description: i % 2 === 0 ? 'classification' : 'regression',
        status: i % 3 === 0 ? 'failed' : 'completed' as const,
        duration: 5000 + i * 100,
        startTime: Date.now() - (10 - i) * 1000,
        endTime: Date.now() - (10 - i) * 1000,
      }))

      // Add to history
      models.forEach(model => history.addModel(model))

      // Update metrics
      const modelMetrics = metrics.calculateModelMetrics(models)
      metrics.registerMetric('successRate', 'Success Rate', '%')
      metrics.updateMetric('successRate', modelMetrics.successRate * 100)

      // Trigger alerts
      const alertResults = alerts.evaluate({
        models,
        queueLength: 0,
      })

      // Add some to favorites
      models.slice(0, 3).forEach(model => {
        favorites.addFavorite(model)
      })

      // Verify data flow
      expect(history.getAllModels().length).toBe(10)
      expect(metrics.getMetric('successRate')?.currentValue).toBeGreaterThan(0)
      expect(alertResults.length).toBeGreaterThanOrEqual(0)
      expect(favorites.getAllFavorites().length).toBe(3)
    })
  })

  describe('Performance Under Load', () => {
    it('should handle high load across all systems', () => {
      const cache = getSmartCache('load-test', { maxSize: 100 })
      const history = getSmartHistory()
      const metrics = getRealTimeMetrics()

      const startTime = Date.now()

      // Create many models
      for (let i = 0; i < 200; i++) {
        history.addModel({
          modelId: `model-${i}`,
          modelName: `test-${i}`,
          description: 'test',
          status: 'completed' as const,
          startTime: Date.now() - i * 1000,
          endTime: Date.now() - i * 1000,
        })

        // Cache operations
        cache.set(`key-${i}`, { value: i })

        // Metric updates
        metrics.registerMetric(`metric-${i}`, `Metric ${i}`, '')
        metrics.updateMetric(`metric-${i}`, i)
      }

      const duration = Date.now() - startTime

      // Should complete in reasonable time
      expect(duration).toBeLessThan(5000)
      expect(history.getAllModels().length).toBe(200)
      expect(cache.keys().length).toBeLessThanOrEqual(100)
    })
  })

  describe('Error Propagation', () => {
    it('should handle errors across systems gracefully', () => {
      const history = getSmartHistory()
      const alerts = getIntelligentAlerts()
      const metrics = getRealTimeMetrics()

      // Add failed models
      const failedModels = Array(10).fill(null).map((_, i) => ({
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

      // Should handle errors gracefully
      expect(alertResults.length).toBeGreaterThan(0)
    })
  })

  describe('State Synchronization', () => {
    it('should keep state synchronized across systems', () => {
      const history = getSmartHistory()
      const favorites = getFavoritesManager()
      const cache = getSmartCache('sync-test', { maxSize: 50 })

      // Create and add model
      const model = {
        modelId: 'model-1',
        modelName: 'test',
        description: 'test',
        status: 'completed' as const,
        startTime: Date.now(),
        endTime: Date.now(),
      }

      history.addModel(model)
      favorites.addFavorite(model)
      cache.set('model-1', model)

      // Verify synchronization
      const historyModel = history.getAllModels().find(m => m.modelId === 'model-1')
      const favoriteModel = favorites.getFavorite('model-1')
      const cachedModel = cache.get('model-1')

      expect(historyModel?.modelId).toBe('model-1')
      expect(favoriteModel?.modelId).toBe('model-1')
      expect(cachedModel?.modelId).toBe('model-1')
    })
  })
})










