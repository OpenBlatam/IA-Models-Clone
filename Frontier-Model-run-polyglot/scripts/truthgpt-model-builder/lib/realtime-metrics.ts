/**
 * Real-time Metrics
 * Sistema de métricas en tiempo real
 */

import { ProactiveBuildResult } from '../components/ProactiveModelBuilder'

export interface MetricData {
  timestamp: number
  value: number
  label?: string
}

export interface RealTimeMetric {
  id: string
  name: string
  unit: string
  data: MetricData[]
  currentValue: number
  minValue: number
  maxValue: number
  avgValue: number
  trend: 'up' | 'down' | 'stable'
}

export type MetricCallback = (metric: RealTimeMetric) => void

export class RealTimeMetrics {
  private metrics: Map<string, RealTimeMetric> = new Map()
  private callbacks: Map<string, Set<MetricCallback>> = new Map()
  private maxDataPoints: number = 100
  private updateInterval: number = 1000 // 1 segundo
  private intervalId: NodeJS.Timeout | null = null

  /**
   * Registrar métrica
   */
  registerMetric(
    id: string,
    name: string,
    unit: string = ''
  ): RealTimeMetric {
    const metric: RealTimeMetric = {
      id,
      name,
      unit,
      data: [],
      currentValue: 0,
      minValue: Infinity,
      maxValue: -Infinity,
      avgValue: 0,
      trend: 'stable',
    }

    this.metrics.set(id, metric)
    return metric
  }

  /**
   * Actualizar métrica
   */
  updateMetric(id: string, value: number, label?: string): void {
    const metric = this.metrics.get(id)
    if (!metric) return

    const now = Date.now()
    metric.data.push({
      timestamp: now,
      value,
      label,
    })

    // Limitar datos
    if (metric.data.length > this.maxDataPoints) {
      metric.data.shift()
    }

    // Actualizar estadísticas
    metric.currentValue = value
    metric.minValue = Math.min(metric.minValue, value)
    metric.maxValue = Math.max(metric.maxValue, value)

    // Calcular promedio
    const sum = metric.data.reduce((acc, d) => acc + d.value, 0)
    metric.avgValue = sum / metric.data.length

    // Calcular tendencia
    if (metric.data.length >= 2) {
      const recent = metric.data.slice(-5)
      const older = metric.data.slice(-10, -5)
      if (older.length > 0) {
        const recentAvg = recent.reduce((acc, d) => acc + d.value, 0) / recent.length
        const olderAvg = older.reduce((acc, d) => acc + d.value, 0) / older.length
        if (recentAvg > olderAvg * 1.1) {
          metric.trend = 'up'
        } else if (recentAvg < olderAvg * 0.9) {
          metric.trend = 'down'
        } else {
          metric.trend = 'stable'
        }
      }
    }

    // Notificar callbacks
    const callbacks = this.callbacks.get(id)
    if (callbacks) {
      callbacks.forEach(callback => callback(metric))
    }
  }

  /**
   * Suscribirse a métrica
   */
  subscribe(id: string, callback: MetricCallback): () => void {
    if (!this.callbacks.has(id)) {
      this.callbacks.set(id, new Set())
    }
    this.callbacks.get(id)!.add(callback)

    // Retornar función de unsubscribe
    return () => {
      const callbacks = this.callbacks.get(id)
      if (callbacks) {
        callbacks.delete(callback)
      }
    }
  }

  /**
   * Obtener métrica
   */
  getMetric(id: string): RealTimeMetric | undefined {
    return this.metrics.get(id)
  }

  /**
   * Obtener todas las métricas
   */
  getAllMetrics(): RealTimeMetric[] {
    return Array.from(this.metrics.values())
  }

  /**
   * Iniciar actualización automática
   */
  startAutoUpdate(updateFn: () => void): void {
    if (this.intervalId) {
      clearInterval(this.intervalId)
    }

    this.intervalId = setInterval(updateFn, this.updateInterval)
  }

  /**
   * Detener actualización automática
   */
  stopAutoUpdate(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
  }

  /**
   * Calcular métricas de modelos
   */
  calculateModelMetrics(models: ProactiveBuildResult[]): {
    buildsPerMinute: number
    successRate: number
    avgDuration: number
    queueLength: number
  } {
    const now = Date.now()
    const oneMinuteAgo = now - 60000

    const recentModels = models.filter(
      m => (m.endTime || m.startTime || 0) > oneMinuteAgo
    )

    const buildsPerMinute = recentModels.length
    const successRate =
      recentModels.length > 0
        ? recentModels.filter(m => m.status === 'completed').length / recentModels.length
        : 0

    const durations = recentModels
      .filter(m => m.duration)
      .map(m => m.duration!)
    const avgDuration =
      durations.length > 0
        ? durations.reduce((sum, d) => sum + d, 0) / durations.length
        : 0

    return {
      buildsPerMinute,
      successRate,
      avgDuration,
      queueLength: 0, // Se actualiza desde fuera
    }
  }

  /**
   * Limpiar métricas
   */
  clear(): void {
    this.metrics.clear()
    this.callbacks.clear()
    this.stopAutoUpdate()
  }

  /**
   * Exportar métricas
   */
  exportMetrics(): string {
    return JSON.stringify(
      Array.from(this.metrics.entries()).map(([id, metric]) => ({
        id,
        ...metric,
      })),
      null,
      2
    )
  }
}

// Singleton instance
let realtimeMetricsInstance: RealTimeMetrics | null = null

export function getRealTimeMetrics(): RealTimeMetrics {
  if (!realtimeMetricsInstance) {
    realtimeMetricsInstance = new RealTimeMetrics()
  }
  return realtimeMetricsInstance
}

export default RealTimeMetrics










