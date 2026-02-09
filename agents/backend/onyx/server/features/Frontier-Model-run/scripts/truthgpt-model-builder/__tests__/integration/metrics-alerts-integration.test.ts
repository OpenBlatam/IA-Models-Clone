/**
 * Integration Tests - Metrics + Alerts
 */

import { getRealTimeMetrics } from '@/lib/realtime-metrics'
import { getIntelligentAlerts } from '@/lib/intelligent-alerts'

describe('Metrics + Alerts Integration', () => {
  let metrics: ReturnType<typeof getRealTimeMetrics>
  let alerts: ReturnType<typeof getIntelligentAlerts>

  beforeEach(() => {
    metrics = getRealTimeMetrics()
    alerts = getIntelligentAlerts()
  })

  afterEach(() => {
    metrics.clear()
    alerts.clear()
  })

  describe('Real-time Alert Triggering', () => {
    it('should trigger alerts based on metric values', () => {
      metrics.registerMetric('successRate', 'Success Rate', '%')
      metrics.registerMetric('buildsPerMinute', 'Builds per Minute', '/min')
      metrics.registerMetric('queueLength', 'Queue Length', '')

      // Update metrics to trigger alerts
      metrics.updateMetric('successRate', 30) // Low success rate
      metrics.updateMetric('buildsPerMinute', 0) // No builds
      metrics.updateMetric('queueLength', 25) // Long queue

      const models = Array(10).fill(null).map((_, i) => ({
        modelId: `model-${i}`,
        modelName: `test-${i}`,
        description: 'test',
        status: i < 3 ? 'completed' : 'failed' as const,
        startTime: Date.now(),
        endTime: Date.now(),
      }))

      const alertResults = alerts.evaluate({
        models,
        queueLength: 25,
      })

      expect(alertResults.length).toBeGreaterThan(0)
    })

    it('should track metric trends and trigger alerts', () => {
      metrics.registerMetric('successRate', 'Success Rate', '%')

      // Simulate declining success rate
      for (let i = 0; i < 10; i++) {
        metrics.updateMetric('successRate', 100 - i * 10)
      }

      const metric = metrics.getMetric('successRate')
      expect(metric?.trend).toBe('down')

      // Trigger alert evaluation
      const models = Array(10).fill(null).map((_, i) => ({
        modelId: `model-${i}`,
        modelName: `test-${i}`,
        description: 'test',
        status: i < 5 ? 'completed' : 'failed' as const,
        startTime: Date.now(),
        endTime: Date.now(),
      }))

      const alertResults = alerts.evaluate({ models, queueLength: 0 })
      expect(alertResults.length).toBeGreaterThan(0)
    })
  })

  describe('Metric Subscriptions with Alerts', () => {
    it('should notify alerts when metrics change', (done) => {
      metrics.registerMetric('successRate', 'Success Rate', '%')

      const alertCallback = jest.fn()
      alerts.subscribe(alertCallback)

      // Subscribe to metrics
      metrics.subscribe('successRate', (metric) => {
        if (metric.currentValue < 50) {
          alerts.evaluate({
            models: [],
            queueLength: 0,
          })
        }
      })

      // Update metric to low value
      metrics.updateMetric('successRate', 30)

      setTimeout(() => {
        expect(alertCallback).toHaveBeenCalled()
        done()
      }, 100)
    })
  })

  describe('Alert Cooldown with Metrics', () => {
    it('should respect cooldown even with metric updates', async () => {
      metrics.registerMetric('queueLength', 'Queue Length', '')

      // Register alert with cooldown
      alerts.registerRule({
        id: 'long-queue-alert',
        name: 'Long Queue',
        condition: (data: { queueLength: number }) => data.queueLength > 20,
        severity: 'warning',
        message: 'Queue is too long',
        cooldown: 1000,
      })

      // Trigger alert first time
      metrics.updateMetric('queueLength', 25)
      const alerts1 = alerts.evaluate({ models: [], queueLength: 25 })
      expect(alerts1.length).toBeGreaterThan(0)

      // Update metric again (should respect cooldown)
      metrics.updateMetric('queueLength', 30)
      const alerts2 = alerts.evaluate({ models: [], queueLength: 30 })
      expect(alerts2.length).toBe(0) // Cooldown active

      // Wait for cooldown
      await new Promise(resolve => setTimeout(resolve, 1100))

      // Update metric again (cooldown expired)
      metrics.updateMetric('queueLength', 35)
      const alerts3 = alerts.evaluate({ models: [], queueLength: 35 })
      expect(alerts3.length).toBeGreaterThan(0)
    })
  })

  describe('Metric Statistics for Alerts', () => {
    it('should use metric statistics in alert evaluation', () => {
      metrics.registerMetric('duration', 'Duration', 'ms')

      // Add duration data points
      const durations = [5000, 10000, 15000, 20000, 25000]
      durations.forEach(duration => {
        metrics.updateMetric('duration', duration)
      })

      const metric = metrics.getMetric('duration')
      expect(metric?.avgValue).toBeGreaterThan(0)
      expect(metric?.minValue).toBe(5000)
      expect(metric?.maxValue).toBe(25000)

      // Use in alert evaluation
      alerts.registerRule({
        id: 'long-duration-alert',
        name: 'Long Duration',
        condition: (data: { avgDuration: number }) => data.avgDuration > 12000,
        severity: 'warning',
        message: 'Average duration is too high',
      })

      const alertResults = alerts.evaluate({
        models: [],
        queueLength: 0,
        avgDuration: metric?.avgValue || 0,
      })

      expect(alertResults.length).toBeGreaterThan(0)
    })
  })

  describe('Multiple Metrics Multiple Alerts', () => {
    it('should handle multiple metrics and alerts simultaneously', () => {
      metrics.registerMetric('successRate', 'Success Rate', '%')
      metrics.registerMetric('buildsPerMinute', 'Builds per Minute', '/min')
      metrics.registerMetric('queueLength', 'Queue Length', '')

      // Update all metrics
      metrics.updateMetric('successRate', 40)
      metrics.updateMetric('buildsPerMinute', 0)
      metrics.updateMetric('queueLength', 22)

      const models = Array(10).fill(null).map((_, i) => ({
        modelId: `model-${i}`,
        modelName: `test-${i}`,
        description: 'test',
        status: i < 4 ? 'completed' : 'failed' as const,
        startTime: Date.now(),
        endTime: Date.now(),
      }))

      const alertResults = alerts.evaluate({
        models,
        queueLength: 22,
      })

      // Should trigger multiple alerts
      expect(alertResults.length).toBeGreaterThan(0)
    })
  })
})










